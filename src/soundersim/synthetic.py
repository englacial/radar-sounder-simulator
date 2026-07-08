"""Synthetic test scenes: projected DEMs + straight-line nav for verification.

All scenes live in EPSG:3413, centered near lat 75 N, lon -40 E. DEM values and
nav heights are metres above the WGS84 ellipsoid (no geoid anywhere). Nav is a
straight line through the scene center along the easting (along-track) axis, so
the northing axis is cross-track.
"""

from dataclasses import dataclass, field

import numpy as np
import rasterio
from affine import Affine
from pyproj import Proj, Transformer

CRS = "EPSG:3413"
CENTER_LATLON = (75.0, -40.0)  # deg

_to_proj = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
_to_geo = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


@dataclass
class SyntheticScene:
    """A synthetic DEM plus nav track and the parameters that generated it."""

    name: str
    dem: np.ndarray  # (ny, nx) float32, height above WGS84 ellipsoid (m)
    transform: Affine  # DEM pixel -> projected (E, N)
    crs: str
    nav_llh: np.ndarray  # (n_traces, 3) float64: lat deg, lon deg, ellip. height m
    params: dict = field(default_factory=dict)


def _build(name, z_func, params, *, n_traces=20, altitude=1000.0,
           extent=8000.0, posting=50.0, spacing=100.0):
    """Assemble a scene from a surface function z_func(E, N) -> height (m)."""
    cx, cy = _to_proj.transform(CENTER_LATLON[1], CENTER_LATLON[0])
    nx = ny = int(round(extent / posting))
    x_min, y_max = cx - extent / 2, cy + extent / 2
    transform = Affine.translation(x_min, y_max) * Affine.scale(posting, -posting)

    cols, rows = np.arange(nx), np.arange(ny)
    xs = x_min + (cols + 0.5) * posting
    ys = y_max - (rows + 0.5) * posting
    X, Y = np.meshgrid(xs, ys)
    dem = z_func(X, Y).astype(np.float32)

    # Correct the projected trace spacing by the local point scale factor so the
    # true (ECEF) spacing equals the requested value near the pole.
    k = Proj(CRS).get_factors(*CENTER_LATLON[::-1]).meridional_scale
    offs = (np.arange(n_traces) - (n_traces - 1) / 2) * spacing * k
    nav_x, nav_y = cx + offs, np.full(n_traces, cy)
    lon, lat = _to_geo.transform(nav_x, nav_y)
    h = float(dem.mean()) + altitude
    nav_llh = np.column_stack([lat, lon, np.full(n_traces, h)]).astype(np.float64)

    p = {"n_traces": n_traces, "altitude": altitude, "extent": extent,
         "posting": posting, "spacing": spacing, **params}
    return SyntheticScene(name, dem, transform, crs=CRS, nav_llh=nav_llh, params=p)


def flat_scene(elevation=500.0, **kw):
    """Flat plane at constant elevation."""
    return _build("flat", lambda X, Y: np.full_like(X, elevation),
                  {"elevation": elevation}, **kw)


def tilted_scene(slope_deg=5.0, elevation=500.0, **kw):
    """Plane tilted across-track (northing) by slope_deg degrees."""
    m = np.tan(np.deg2rad(slope_deg))
    cy = _to_proj.transform(CENTER_LATLON[1], CENTER_LATLON[0])[1]
    return _build("tilted", lambda X, Y: elevation + m * (Y - cy),
                  {"slope_deg": slope_deg, "elevation": elevation}, **kw)


def hill_scene(height=200.0, sigma=800.0, offset=1500.0, elevation=500.0, **kw):
    """Gaussian hill offset across-track (northing) from the nav line."""
    cx, cy = _to_proj.transform(CENTER_LATLON[1], CENTER_LATLON[0])
    yc = cy + offset

    def z(X, Y):
        return elevation + height * np.exp(
            -((X - cx) ** 2 + (Y - yc) ** 2) / (2 * sigma ** 2))

    return _build("hill", z, {"height": height, "sigma": sigma,
                              "offset": offset, "elevation": elevation}, **kw)


def sinusoid_scene(amplitude=100.0, wavelength=2000.0, elevation=500.0, **kw):
    """Sinusoidal surface varying across-track (northing)."""
    cy = _to_proj.transform(CENTER_LATLON[1], CENTER_LATLON[0])[1]
    return _build("sinusoid",
                  lambda X, Y: elevation
                  + amplitude * np.sin(2 * np.pi * (Y - cy) / wavelength),
                  {"amplitude": amplitude, "wavelength": wavelength,
                   "elevation": elevation}, **kw)


def crater_scene(depth=200.0, sigma=800.0, elevation=500.0, **kw):
    """Crater/valley: inverted Gaussian (negative relief) at scene center."""
    cx, cy = _to_proj.transform(CENTER_LATLON[1], CENTER_LATLON[0])

    def z(X, Y):
        return elevation - depth * np.exp(
            -((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))

    return _build("crater", z, {"depth": depth, "sigma": sigma,
                                "elevation": elevation}, **kw)


ALL_SCENES = (flat_scene, tilted_scene, hill_scene, sinusoid_scene, crater_scene)


@dataclass
class MultilayerScene:
    """A stack of interface DEMs (top-down) sharing one grid and nav track.

    ``dems[k]`` is interface k's height grid (surface first). ``media`` (Medium
    list) has one more entry than ``dems``. Feed the DEMs to ``build_facets`` or
    ``layered.build_layered_scene`` (the surface DEM being the reference).
    """

    name: str
    dems: list  # list[np.ndarray] (ny, nx) float32, top-down
    transform: Affine
    crs: str
    nav_llh: np.ndarray
    media: list  # list[Medium], len == len(dems) + 1
    params: dict = field(default_factory=dict)

    @property
    def dem(self):
        """The top (surface) DEM -- the reference footprint."""
        return self.dems[0]


def _default_slab_media():
    from .config import Medium
    return [Medium(name="air", eps_r=1.0), Medium(name="ice", eps_r=3.17),
            Medium(name="bed", eps_r=6.0)]


def _build_multi(name, z_funcs, media, params, *, n_traces=20, altitude=1000.0,
                 extent=8000.0, posting=50.0, spacing=100.0):
    """Assemble a multilayer scene from top-down surface functions z(E, N)."""
    cx, cy = _to_proj.transform(CENTER_LATLON[1], CENTER_LATLON[0])
    nx = ny = int(round(extent / posting))
    x_min, y_max = cx - extent / 2, cy + extent / 2
    transform = Affine.translation(x_min, y_max) * Affine.scale(posting, -posting)

    cols, rows = np.arange(nx), np.arange(ny)
    xs = x_min + (cols + 0.5) * posting
    ys = y_max - (rows + 0.5) * posting
    X, Y = np.meshgrid(xs, ys)
    dems = [zf(X, Y).astype(np.float32) for zf in z_funcs]

    k = Proj(CRS).get_factors(*CENTER_LATLON[::-1]).meridional_scale
    offs = (np.arange(n_traces) - (n_traces - 1) / 2) * spacing * k
    nav_x, nav_y = cx + offs, np.full(n_traces, cy)
    lon, lat = _to_geo.transform(nav_x, nav_y)
    h = float(dems[0].mean()) + altitude
    nav_llh = np.column_stack([lat, lon, np.full(n_traces, h)]).astype(np.float64)

    p = {"n_traces": n_traces, "altitude": altitude, "extent": extent,
         "posting": posting, "spacing": spacing, **params}
    return MultilayerScene(name, dems, transform, crs=CRS, nav_llh=nav_llh,
                           media=media, params=p)


def _flat(elev):
    return lambda X, Y: np.full_like(X, elev)


def slab_scene(surface=500.0, depth=300.0, media=None, **kw):
    """Flat surface at ``surface`` over a flat bed ``depth`` metres below it."""
    return _build_multi(
        "slab", [_flat(surface), _flat(surface - depth)],
        media or _default_slab_media(),
        {"surface": surface, "depth": depth}, **kw)


def tilted_bed_scene(surface=500.0, depth=300.0, slope_deg=5.0, media=None, **kw):
    """Flat surface over a bed tilted across-track (northing) by slope_deg."""
    m = np.tan(np.deg2rad(slope_deg))
    cy = _to_proj.transform(CENTER_LATLON[1], CENTER_LATLON[0])[1]
    return _build_multi(
        "tilted_bed",
        [_flat(surface), lambda X, Y: (surface - depth) + m * (Y - cy)],
        media or _default_slab_media(),
        {"surface": surface, "depth": depth, "slope_deg": slope_deg}, **kw)


def rough_bed_scene(surface=500.0, depth=300.0, amplitude=50.0,
                    wavelength=2000.0, media=None, **kw):
    """Flat surface over a sinusoidal (across-track) bed."""
    cy = _to_proj.transform(CENTER_LATLON[1], CENTER_LATLON[0])[1]
    return _build_multi(
        "rough_bed",
        [_flat(surface),
         lambda X, Y: (surface - depth)
         + amplitude * np.sin(2 * np.pi * (Y - cy) / wavelength)],
        media or _default_slab_media(),
        {"surface": surface, "depth": depth, "amplitude": amplitude,
         "wavelength": wavelength}, **kw)


def rough_surface_scene(surface=500.0, depth=300.0, amplitude=50.0,
                        wavelength=2000.0, media=None, **kw):
    """Sinusoidal (across-track) surface over a flat bed."""
    cy = _to_proj.transform(CENTER_LATLON[1], CENTER_LATLON[0])[1]
    return _build_multi(
        "rough_surface",
        [lambda X, Y: surface
         + amplitude * np.sin(2 * np.pi * (Y - cy) / wavelength),
         _flat(surface - depth)],
        media or _default_slab_media(),
        {"surface": surface, "depth": depth, "amplitude": amplitude,
         "wavelength": wavelength}, **kw)


def offset_stack_scene(surface=500.0, spacings=(2.0, 2.0, 2.0), eps_r=3.17,
                       **kw):
    """Flat surface plus N firn layers, each ``spacings[i]`` below the previous.

    Layer DEMs are explicit constant-offset copies of the surface (the physical
    content of the config-level offset-interface stack). Media between layers
    all share ``eps_r``; the substrate is the last.
    """
    from .config import Medium
    z_funcs = [_flat(surface)]
    depth = 0.0
    for s in spacings:
        depth += s
        z_funcs.append(_flat(surface - depth))
    media = [Medium(name="air", eps_r=1.0)]
    media += [Medium(name=f"firn_{i}", eps_r=eps_r)
              for i in range(len(spacings) + 1)]
    return _build_multi("offset_stack", z_funcs, media,
                        {"surface": surface, "spacings": list(spacings),
                         "eps_r": eps_r}, **kw)


MULTILAYER_SCENES = (slab_scene, tilted_bed_scene, rough_bed_scene,
                     rough_surface_scene, offset_stack_scene)


def write_dem_geotiff(scene: SyntheticScene, path):
    """Write the scene DEM as a single-band float32 GeoTIFF (nodata=-9999)."""
    ny, nx = scene.dem.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=ny, width=nx, count=1,
        dtype="float32", crs=scene.crs, transform=scene.transform, nodata=-9999.0,
    ) as dst:
        dst.write(scene.dem, 1)


def nav_ecef(scene: SyntheticScene) -> np.ndarray:
    """Nav positions as (n, 3) float64 ECEF (WGS84) from nav_llh."""
    lat, lon, h = scene.nav_llh.T
    x, y, z = _to_ecef.transform(lon, lat, h)
    return np.column_stack([x, y, z]).astype(np.float64)
