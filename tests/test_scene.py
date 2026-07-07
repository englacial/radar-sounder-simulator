"""Geometry-layer tests for LocalFrame and build_facets (CPU, float64)."""

import numpy as np
from pyproj import Proj, Transformer

from soundersim import synthetic as syn
from soundersim.scene import LocalFrame, build_facets


def _angle_deg(a, b):
    """Angle (deg) between rows of unit-ish vectors a and b."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0)))


def test_llh_round_trip_1mm():
    """llh -> local -> llh recovers position < 1 mm within 25 km of anchor."""
    frame = LocalFrame(75.0, -40.0)
    rng = np.random.default_rng(0)
    enu = rng.uniform(-25_000, 25_000, (3000, 3))
    enu[:, 2] = rng.uniform(-2000, 2000, 3000)  # realistic vertical extent
    back = frame.llh_to_local(frame.local_to_llh(enu))
    assert np.abs(back - enu).max() < 1e-3  # metres


def test_ecef_round_trip_1mm():
    """ecef -> local -> ecef recovers position < 1 mm within 25 km of anchor."""
    frame = LocalFrame(75.0, -40.0)
    rng = np.random.default_rng(1)
    enu = rng.uniform(-25_000, 25_000, (3000, 3))
    xyz = frame.local_to_ecef(enu)
    back = frame.local_to_ecef(frame.ecef_to_local(xyz))
    assert np.abs(back - xyz).max() < 1e-3  # metres


def test_flat_total_area_matches_independent_per_cell():
    """On a flat (ellipsoid-height=0) scene the facet total equals the sum of
    per-cell true ground areas computed independently as projected_area /
    areal_scale. The DEM is placed at ellipsoidal height 0 so facets lie on the
    ellipsoid; a nonzero height would inflate true area by ~(1+h/R)^2, and using
    the scene-center scale alone would err ~1e-4 because areal_scale varies over
    the 8 km window -- hence the per-cell reference."""
    scene = syn.flat_scene(elevation=0.0)
    frame = LocalFrame.centered_on(scene)
    facets = build_facets(scene.dem, scene.transform, scene.crs, frame)

    posting = scene.params["posting"]
    ny, nx = scene.dem.shape
    cj, ci = np.meshgrid(np.arange(nx - 1) + 1.0, np.arange(ny - 1) + 1.0)
    E, N = scene.transform * (cj, ci)  # cell centers (projected)
    lon, lat = Transformer.from_crs(
        scene.crs, "EPSG:4326", always_xy=True).transform(E.ravel(), N.ravel())
    areal = Proj(scene.crs).get_factors(lon, lat).areal_scale
    expected = np.sum(posting ** 2 / areal)
    assert abs(facets.areas.sum() - expected) / expected < 1e-6


def test_flat_normals_follow_local_up():
    """Each facet normal is within 0.1 deg of the ellipsoidal UP at its own
    location (facets track the curved ellipsoid, not the anchor's global +z)."""
    scene = syn.flat_scene(elevation=500.0)
    frame = LocalFrame.centered_on(scene)
    facets = build_facets(scene.dem, scene.transform, scene.crs, frame)
    ang = _angle_deg(facets.normals, frame.up_at(facets.centers))
    assert ang.max() < 0.1
    assert np.all(facets.normals[:, 2] > 0.0)  # oriented upward


def test_tilted_normals_match_true_slope():
    """A plane tilted by slope_deg in projected northing has a true ground slope
    atan(tan(slope)*k) (k = along-slope point scale of EPSG:3413), because
    projected distances are scaled by k while heights are true metres. Facet
    normals tilt from local-up by that true angle within 0.05 deg."""
    slope = 5.0
    scene = syn.tilted_scene(slope_deg=slope)
    frame = LocalFrame.centered_on(scene)
    facets = build_facets(scene.dem, scene.transform, scene.crs, frame)
    ang = _angle_deg(facets.normals, frame.up_at(facets.centers))
    k = Proj(scene.crs).get_factors(-40.0, 75.0).meridional_scale
    true_slope = np.degrees(np.arctan(np.tan(np.deg2rad(slope)) * k))
    assert abs(ang.mean() - true_slope) < 0.05


def test_gaussian_hill_area_converges():
    """Surface-excess area (hill minus flat) converges as O(posting^2): the error
    at 100 m posting is ~4x that at 50 m, measured against the finest (25 m)
    facet reference."""
    def excess(posting):
        hill = syn.hill_scene(posting=posting)
        frame = LocalFrame.centered_on(hill)
        flat = syn.flat_scene(posting=posting,
                              elevation=float(hill.params["elevation"]))
        fh = build_facets(hill.dem, hill.transform, hill.crs, frame)
        ff = build_facets(flat.dem, flat.transform, flat.crs, frame)
        return fh.areas.sum() - ff.areas.sum()

    e100, e50, e25 = excess(100.0), excess(50.0), excess(25.0)
    assert abs(e100 - e25) / abs(e50 - e25) > 3.5  # ~4x, O(h^2)


def test_facet_index_map_and_shapes():
    """Facet arrays, grid mapping, and triangle/diagonal convention are consistent."""
    scene = syn.flat_scene()
    frame = LocalFrame.centered_on(scene)
    facets = build_facets(scene.dem, scene.transform, scene.crs, frame)
    ny, nx = scene.dem.shape
    n = (ny - 1) * (nx - 1) * 2
    assert facets.centers.shape == (n, 3)
    assert facets.normals.shape == (n, 3)
    assert facets.areas.shape == (n,)
    assert facets.grid_shape == (ny - 1, nx - 1)
    imap = facets.index_map()
    assert imap.shape == (ny - 1, nx - 1, 2)
    # index_map -> flat index round-trips to the stored cell/tri for a sample.
    for (i, j, t) in [(0, 0, 0), (3, 5, 1), (ny - 2, nx - 2, 1)]:
        k = imap[i, j, t]
        assert tuple(facets.cell[k]) == (i, j) and facets.tri[k] == t
    assert facets.centers.dtype == np.float64


def test_facet_spacing_subsample():
    """Passing spacing coarser than the posting reduces the facet count."""
    scene = syn.flat_scene(posting=50.0)
    frame = LocalFrame.centered_on(scene)
    fine = build_facets(scene.dem, scene.transform, scene.crs, frame)
    coarse = build_facets(scene.dem, scene.transform, scene.crs, frame, spacing=100.0)
    assert coarse.grid_shape[0] < fine.grid_shape[0]
    # ~2x coarser posting -> ~4x fewer facets.
    assert 3.0 < len(fine.areas) / len(coarse.areas) < 5.0
