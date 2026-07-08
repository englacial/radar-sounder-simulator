"""Output Dataset assembly and I/O per docs/output.md.

Documented conventions:

- The ``twtt`` coordinate is the *start* of each fast-time bin, ``t0 + k*dt``
  (bin k spans [t0 + k*dt, t0 + (k+1)*dt), the simc/fixture convention).
- ``nadir_twtt`` = 2·(platform z − z of the horizontally nearest facet
  center)/c in the local frame, i.e. vertical distance to the DEM at nadir via
  the nearest facet centroid.
- ``first_return_twtt`` is the minimum *exact* per-facet twtt = 2r/c over
  facets inside the twtt window (simc's fret definition, computed in float64,
  not read off the binned power); that facet's centroid gives
  ``first_return_lat/lon``.
"""

import datetime
import json
import warnings

import numpy as np
import xarray as xr

import soundersim


def build_dataset(power, dropped_power, *, scene, frame, facets, track,
                  sim_config, field=None):
    """Assemble the output Dataset for one simulation run.

    Coherent runs pass ``field`` (complex64, same dims as power); the Dataset
    then also carries ``field`` plus ``frequency``/``wavelength`` attrs per
    docs/output.md.
    """
    rc = sim_config.radar
    power = np.asarray(power, dtype=np.float32)
    n_traces, n_samples = power.shape[0], power.shape[1]
    pos = track.positions

    # Nadir: horizontally nearest facet center (see module docstring).
    d2 = ((facets.centers[None, :, :2] - pos[:, None, :2]) ** 2).sum(-1)
    nadir_z = facets.centers[d2.argmin(axis=1), 2]
    nadir_twtt = 2.0 * (pos[:, 2] - nadir_z) / rc.c

    # First return: min exact facet twtt within the window (float64).
    r = np.linalg.norm(facets.centers[None, :, :] - pos[:, None, :], axis=-1)
    ft = 2.0 * r / rc.c
    ft = np.where((ft >= rc.t0) & (ft < rc.t0 + n_samples * rc.dt), ft, np.inf)
    fi = ft.argmin(axis=1)
    first_twtt = ft[np.arange(n_traces), fi]
    first_llh = frame.local_to_llh(facets.centers[fi])

    dims = ("slow_time", "twtt") + (("side",) if power.ndim == 3 else ())
    trace = np.arange(n_traces)
    ds = xr.Dataset(
        {
            "power": (dims, power,
                      {"units": "1", "long_name": "relative received power",
                       "comment": "relative linear power"}),
            "nadir_twtt": ("slow_time", nadir_twtt,
                           {"units": "s", "long_name": "two-way time to surface at nadir"}),
            "first_return_twtt": ("slow_time", first_twtt,
                                  {"units": "s", "long_name": "earliest in-window arrival"}),
            "first_return_lat": ("slow_time", first_llh[:, 0], {"units": "degrees_north"}),
            "first_return_lon": ("slow_time", first_llh[:, 1], {"units": "degrees_east"}),
            "dropped_power": ("slow_time", np.asarray(dropped_power, np.float32),
                              {"units": "1", "long_name": "facet power outside the twtt window"}),
        },
        coords={
            "slow_time": trace,  # integer trace numbers: synthetic nav has no timestamps
            "twtt": ("twtt", rc.t0 + np.arange(n_samples) * rc.dt,
                     {"units": "s", "long_name": "two-way travel time (bin start)"}),
            "trace": ("slow_time", trace),
            "lat": ("slow_time", scene.nav_llh[:, 0], {"units": "degrees_north"}),
            "lon": ("slow_time", scene.nav_llh[:, 1], {"units": "degrees_east"}),
            "elevation": ("slow_time", scene.nav_llh[:, 2],
                          {"units": "m", "comment": "above WGS84 ellipsoid"}),
            "x": ("slow_time", pos[:, 0], {"units": "m", "comment": "local scene frame"}),
            "y": ("slow_time", pos[:, 1], {"units": "m", "comment": "local scene frame"}),
            "z": ("slow_time", pos[:, 2], {"units": "m", "comment": "local scene frame"}),
        },
        attrs={
            "mode": sim_config.mode,
            "config": sim_config.model_dump_json(),
            "soundersim_version": soundersim.__version__,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dem_source": f"synthetic:{scene.name}",
            "dem_crs": str(scene.crs),
            "scene_params": json.dumps(scene.params),
            "scene_frame": json.dumps(
                {"origin_lat": frame.lat0, "origin_lon": frame.lon0,
                 "origin_h": frame.h0, "orientation": "ENU"}),
        },
    )
    if field is not None:
        ds["field"] = (dims, np.asarray(field, dtype=np.complex64),
                       {"units": "1", "long_name": "relative received field",
                        "comment": "complex field at the carrier; power = |field|^2"})
        ds.attrs["frequency"] = rc.f0
        ds.attrs["wavelength"] = rc.wavelength
    if power.ndim == 3:
        ds = ds.assign_coords(side=("side", np.array(["left", "right"])))
    return ds


def combine(ds, dim):
    """Combine an optional split dimension with the mode-correct rule."""
    if ds.attrs["mode"] == "incoherent":
        return ds.power.sum(dim)
    return abs(ds.field.sum(dim)) ** 2


def save(ds, path, strict=False):
    """Write to NetCDF4 via h5netcdf.

    Incoherent Datasets are plain NetCDF4. Coherent Datasets contain complex
    ``field`` data: by default they are written with ``invalid_netcdf=True``
    (valid HDF5, read back transparently by ``load``/xarray). ``strict=True``
    instead splits ``field`` into float32 ``field_real``/``field_imag`` for
    strict-NetCDF interchange; ``load`` reassembles them.
    """
    if "field" in ds and strict:
        f = ds.field
        ds = ds.drop_vars("field")
        ds["field_real"] = f.dims, np.ascontiguousarray(f.values.real), f.attrs
        ds["field_imag"] = f.dims, np.ascontiguousarray(f.values.imag), f.attrs
        ds.to_netcdf(path, engine="h5netcdf")
    elif "field" in ds:
        with warnings.catch_warnings():
            # the deliberate, documented invalid_netcdf choice; don't nag
            warnings.filterwarnings("ignore",
                                    message=".*invalid netcdf features.*")
            ds.to_netcdf(path, engine="h5netcdf", invalid_netcdf=True)
    else:
        ds.to_netcdf(path, engine="h5netcdf")
    return path


def load(path):
    """Read a Dataset written by ``save``, reassembling a strict-mode split
    ``field`` from ``field_real``/``field_imag``."""
    ds = xr.load_dataset(path, engine="h5netcdf")
    if "field_real" in ds:
        field = (ds.field_real.values + 1j * ds.field_imag.values).astype(
            np.complex64)
        ds["field"] = ds.field_real.dims, field, dict(ds.field_real.attrs)
        ds = ds.drop_vars(["field_real", "field_imag"])
    return ds
