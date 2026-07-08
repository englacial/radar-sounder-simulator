"""Geometry-layer tests for LocalFrame and build_facets (CPU, float64)."""

import warnings

import numpy as np
import pytest
from pyproj import Proj, Transformer

from soundersim import synthetic as syn
from soundersim.scene import LocalFrame, build_facets, check_facet_size

_ECEF = "EPSG:4978"


def _angle_deg(a, b):
    """Angle (deg) between rows of unit-ish vectors a and b."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0)))


def _vertices(scene, frame):
    """DEM cell corners as local-ENU vertices (ny, nx, 3), native posting."""
    dem = np.asarray(scene.dem, np.float64)
    ny, nx = dem.shape
    J, I = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
    E, N = scene.transform * (J, I)
    ecef = np.column_stack(Transformer.from_crs(scene.crs, _ECEF, always_xy=True)
                           .transform(E.ravel(), N.ravel(), dem.ravel()))
    return frame.ecef_to_local(ecef).reshape(ny, nx, 3)


def _triangle_agg(scene, frame):
    """Reference two-triangles-per-cell tessellation (the removed stage-1 build),
    aggregated per cell to (area sum, area-weighted unit normal)."""
    V = _vertices(scene, frame)
    v00, v01, v10, v11 = V[:-1, :-1], V[:-1, 1:], V[1:, :-1], V[1:, 1:]
    n0 = np.cross(v01 - v00, v11 - v00)  # each = 2 * area * unit normal
    n1 = np.cross(v11 - v00, v10 - v00)
    area = (0.5 * np.linalg.norm(n0, axis=-1)
            + 0.5 * np.linalg.norm(n1, axis=-1)).reshape(-1)
    nrm = (n0 + n1).reshape(-1, 3)
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    nrm *= np.where(nrm[:, 2] < 0.0, -1.0, 1.0)[:, None]
    return area, nrm


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
    """Facet arrays, grid mapping, and one-facet-per-cell convention are consistent."""
    scene = syn.flat_scene()
    frame = LocalFrame.centered_on(scene)
    facets = build_facets(scene.dem, scene.transform, scene.crs, frame)
    ny, nx = scene.dem.shape
    n = (ny - 1) * (nx - 1)
    for arr in (facets.centers, facets.normals, facets.e1, facets.e2):
        assert arr.shape == (n, 3)
    assert facets.areas.shape == (n,)
    assert facets.grid_shape == (ny - 1, nx - 1)
    imap = facets.index_map()
    assert imap.shape == (ny - 1, nx - 1)
    # index_map -> flat index round-trips to the stored source cell for a sample.
    for (i, j) in [(0, 0), (3, 5), (ny - 2, nx - 2)]:
        assert tuple(facets.cell[imap[i, j]]) == (i, j)
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


def test_rect_facet_vs_triangle_pair():
    """One-time cross-check of the rectangular mean-plane facet against the removed
    two-triangle-per-cell build. On planar cells |e1 x e2| equals the triangle-pair
    area exactly (both give the planar-quad area); the mean-plane normal equals the
    area-weighted triangle normal identically. Divergence appears only on cells with
    2D curvature (hill/crater), and is tiny: max per-cell area rel diff ~8e-6 on the
    hill (mean ~5e-7), normals still matching to ~0 deg."""
    for gen, area_tol in [(lambda: syn.flat_scene(elevation=0.0), 1e-12),
                          (syn.hill_scene, 1e-5)]:
        scene = gen()
        frame = LocalFrame.centered_on(scene)
        f = build_facets(scene.dem, scene.transform, scene.crs, frame)
        tri_area, tri_n = _triangle_agg(scene, frame)
        rel = np.abs(f.areas - tri_area) / tri_area
        assert rel.max() < area_tol, (scene.name, rel.max())
        assert _angle_deg(f.normals, tri_n).max() < 1e-6


def test_plane_fit_residual_bounded():
    """The 4 corners of every cell lie close to the stored mean plane: max
    corner-to-plane distance is a small fraction of the facet edge on all synthetic
    scenes (exact for planar cells; ~1.4e-3 of an edge where curvature is largest)."""
    for gen in syn.ALL_SCENES:
        scene = gen()
        frame = LocalFrame.centered_on(scene)
        f = build_facets(scene.dem, scene.transform, scene.crs, frame)
        V = _vertices(scene, frame)
        corners = [c.reshape(-1, 3) for c in
                   (V[:-1, :-1], V[:-1, 1:], V[1:, :-1], V[1:, 1:])]
        resid = np.zeros(len(f.centers))
        for cv in corners:
            resid = np.maximum(resid, np.abs(((cv - f.centers) * f.normals).sum(1)))
        edge = np.maximum(np.linalg.norm(f.e1, axis=1), np.linalg.norm(f.e2, axis=1))
        assert (resid / edge).max() < 5e-3, (scene.name, (resid / edge).max())


def test_gaussian_hill_normals_converge():
    """Curved-surface normals converge with posting. The integrated normal
    cosine-deficit sum(area * (1 - n . up)) -- a smooth surface functional of the
    normal field -- has O(h^2) discretization error: its change from 100 m is ~4x
    that from 50 m, against the 25 m facet reference."""
    def deficit(posting):
        s = syn.hill_scene(posting=posting)
        frame = LocalFrame.centered_on(s)
        f = build_facets(s.dem, s.transform, s.crs, frame)
        up = frame.up_at(f.centers)
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        return float((f.areas * (1.0 - (f.normals * up).sum(1))).sum())

    d100, d50, d25 = deficit(100.0), deficit(50.0), deficit(25.0)
    assert abs(d100 - d25) / abs(d50 - d25) > 3.5  # ~4x, O(h^2)


def test_subdivision_converges():
    """Sub-posting spacing bilinearly subdivides the DEM toward the target facet
    size. On a planar scene the refined normals match a native fine build exactly
    (intensive quantity, footprint-independent); on a curved scene, refining the
    same DEM converges Cauchy (successive total-area changes shrink ~4x) to the
    bilinear interpolant it manufactures."""
    # Planar: subdividing 50 m -> 25 m reproduces the native-25 m mean normal.
    sub = syn.tilted_scene(posting=50.0)
    fr_s = LocalFrame.centered_on(sub)
    f_sub = build_facets(sub.dem, sub.transform, sub.crs, fr_s, spacing=25.0)
    nat = syn.tilted_scene(posting=25.0)
    fr_n = LocalFrame.centered_on(nat)
    f_nat = build_facets(nat.dem, nat.transform, nat.crs, fr_n)
    tilt = lambda f, fr: _angle_deg(f.normals, fr.up_at(f.centers)).mean()
    assert abs(tilt(f_sub, fr_s) - tilt(f_nat, fr_n)) < 1e-4
    edge = np.maximum(np.linalg.norm(f_sub.e1, axis=1), np.linalg.norm(f_sub.e2, axis=1))
    assert 20.0 < np.median(edge) < 30.0        # ~target facet size
    f_native = build_facets(sub.dem, sub.transform, sub.crs, fr_s)  # 50 m, same DEM
    assert 3.0 < len(f_sub.areas) / len(f_native.areas) < 5.0  # ~4x facets

    # Curved: same 50 m DEM, finer targets -> convergent total area.
    hill = syn.hill_scene(posting=50.0)
    fr_h = LocalFrame.centered_on(hill)
    A = {s: build_facets(hill.dem, hill.transform, hill.crs, fr_h,
                         spacing=s).areas.sum() for s in (None, 25.0, 12.5)}
    d1, d2 = abs(A[25.0] - A[None]), abs(A[12.5] - A[25.0])
    assert d2 < 0.35 * d1  # successive refinement change shrinks ~4x


def test_check_facet_size_warns():
    """check_facet_size warns when facets exceed half a Fresnel-zone radius and is
    silent when they fit; the returned ratio scales with edge / sqrt(lambda*r)."""
    scene = syn.flat_scene(posting=50.0)
    frame = LocalFrame.centered_on(scene)
    f = build_facets(scene.dem, scene.transform, scene.crs, frame)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r_ok = check_facet_size(f, wavelength=1.54, min_range=14000.0)
    assert r_ok < 1.0
    with pytest.warns(UserWarning):
        r_bad = check_facet_size(f, wavelength=1.54, min_range=500.0)
    assert r_bad > 1.0
