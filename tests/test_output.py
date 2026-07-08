"""Output Dataset structure per docs/output.md, save round-trip, combine."""

import json

import numpy as np
import pytest
import xarray as xr

import soundersim
from soundersim.config import FacetConfig, RadarConfig, SimConfig
from soundersim.synthetic import flat_scene

RC = RadarConfig(dt=1e-8, n_samples=1250, t0=0.0)


@pytest.fixture(scope="module")
def ds():
    scene = flat_scene(extent=2000.0, n_traces=4, altitude=1000.0)
    cfg = SimConfig(mode="incoherent", split_sides=True, radar=RC,
                    facets=FacetConfig())
    return soundersim.simulate(scene, cfg)


def test_dataset_structure(ds):
    assert ds.power.dims == ("slow_time", "twtt", "side")
    assert ds.power.dtype == np.float32
    assert list(ds.side.values) == ["left", "right"]
    assert ds.sizes == {"slow_time": 4, "twtt": RC.n_samples, "side": 2}
    np.testing.assert_allclose(np.diff(ds.twtt), RC.dt)
    assert ds.twtt.values[0] == RC.t0
    for coord in ("trace", "lat", "lon", "elevation", "x", "y", "z"):
        assert ds[coord].dims == ("slow_time",)
    for var in ("nadir_twtt", "first_return_twtt", "first_return_lat",
                "first_return_lon", "dropped_power"):
        assert ds[var].dims == ("slow_time",)
    assert ds.attrs["mode"] == "incoherent"
    cfg = json.loads(ds.attrs["config"])  # full pydantic config round-trips
    assert cfg["radar"]["n_samples"] == RC.n_samples
    assert ds.attrs["soundersim_version"] == soundersim.__version__
    assert "created" in ds.attrs and "scene_frame" in ds.attrs


def test_physics_sanity(ds):
    # flat scene at 1000 m altitude: nadir and first return at 2h/c
    expected = 2 * 1000.0 / RC.c
    np.testing.assert_allclose(ds.nadir_twtt, expected, rtol=1e-3)
    np.testing.assert_allclose(ds.first_return_twtt, expected, rtol=1e-3)
    assert (ds.dropped_power >= 0).all()


def test_combine_rule(ds):
    combined = soundersim.combine(ds, "side")
    np.testing.assert_allclose(combined, ds.power.sum("side"))


def test_save_roundtrip(ds, tmp_path):
    path = soundersim.save(ds, tmp_path / "out.nc")
    back = xr.open_dataset(path)
    np.testing.assert_array_equal(back.power, ds.power)
    assert back.attrs["mode"] == "incoherent"
    back.close()


RC_COH = RadarConfig(dt=2e-8, n_samples=160, t0=6.5e-6, f0=195e6)


@pytest.fixture(scope="module")
def ds_coh():
    scene = flat_scene(extent=1200.0, n_traces=3, altitude=1000.0, posting=50.0)
    cfg = SimConfig(mode="coherent", split_sides=True, radar=RC_COH,
                    facets=FacetConfig(spacing=15.0))  # sub-Fresnel: no warning
    return soundersim.simulate(scene, cfg)


def test_coherent_dataset_structure(ds_coh):
    ds = ds_coh
    assert ds.attrs["mode"] == "coherent"
    assert ds.field.dims == ("slow_time", "twtt", "side")
    assert ds.field.dtype == np.complex64
    assert ds.power.dims == ds.field.dims and ds.power.dtype == np.float32
    # power is precomputed |field|^2 (docs/output.md contract)
    np.testing.assert_array_equal(ds.power.values,
                                  np.abs(ds.field.values) ** 2)
    assert ds.attrs["frequency"] == RC_COH.f0
    assert ds.attrs["wavelength"] == pytest.approx(RC_COH.c / RC_COH.f0)
    assert (ds.dropped_power >= 0).all()
    # flat scene at 1000 m AGL: nadir return where it belongs
    np.testing.assert_allclose(ds.nadir_twtt, 2 * 1000.0 / RC_COH.c, rtol=1e-3)


def test_coherent_combine_sums_fields(ds_coh):
    combined = soundersim.combine(ds_coh, "side")
    np.testing.assert_allclose(
        combined, np.abs(ds_coh.field.sum("side")) ** 2)
    # field-level combination differs from the incoherent power sum (flat
    # scene: left/right fields are nearly in phase, so ~2x, i.e. +3 dB)
    diff = np.abs(combined - ds_coh.power.sum("side")).max()
    assert diff > 0.1 * float(combined.max())


def test_coherent_save_roundtrip(ds_coh, tmp_path):
    path = soundersim.save(ds_coh, tmp_path / "coh.nc")
    back = soundersim.load(path)
    assert back.field.dtype == np.complex64
    np.testing.assert_array_equal(back.field.values, ds_coh.field.values)
    np.testing.assert_array_equal(back.power.values, ds_coh.power.values)
    assert back.attrs["mode"] == "coherent"


def test_coherent_save_strict_roundtrip(ds_coh, tmp_path):
    path = soundersim.save(ds_coh, tmp_path / "coh_strict.nc", strict=True)
    raw = xr.open_dataset(path, engine="h5netcdf")  # strict NetCDF-4: no complex
    assert "field" not in raw and "field_real" in raw and "field_imag" in raw
    assert raw.field_real.dtype == np.float32
    raw.close()
    back = soundersim.load(path)
    assert back.field.dtype == np.complex64
    np.testing.assert_array_equal(back.field.values, ds_coh.field.values)
    assert "field" in ds_coh  # the in-memory Dataset is untouched


def test_coherent_requires_f0():
    scene = flat_scene(extent=1200.0, n_traces=2, altitude=1000.0, posting=50.0)
    cfg = SimConfig(mode="coherent",
                    radar=RadarConfig(dt=2e-8, n_samples=160, t0=6.5e-6),
                    facets=FacetConfig(spacing=15.0))
    with pytest.raises(ValueError, match="f0"):
        soundersim.simulate(scene, cfg)


def test_coherent_fresnel_warning_surfaces():
    """Coarse facets at low altitude trip scene.check_facet_size's warning."""
    scene = flat_scene(extent=1200.0, n_traces=2, altitude=1000.0, posting=50.0)
    cfg = SimConfig(mode="coherent", radar=RC_COH, facets=FacetConfig())
    with pytest.warns(UserWarning, match="Fresnel"):
        soundersim.simulate(scene, cfg)
