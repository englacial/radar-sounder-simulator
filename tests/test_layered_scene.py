"""Layered scene: offset fast-path correctness, flat synthesis, variants."""

import numpy as np
import pytest

from soundersim import synthetic as syn
from soundersim.config import (DemInterface, FacetConfig, FlatInterface,
                               Medium, OffsetInterface, RadarConfig, SimConfig)
from soundersim.layered import build_layered_scene, offset_facets
from soundersim.scene import LocalFrame, build_facets


def _angle_deg(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0)))


@pytest.mark.parametrize("gen", [syn.flat_scene, syn.tilted_scene])
def test_offset_fastpath_matches_full_rebuild(gen):
    """Constant-elevation offset fast-path (shift centers along local up, reuse
    normals/areas) matches a full DEM->ECEF rebuild of the shifted DEM: centers
    < 1 mm over the 8 km scene at 50 m offset; normals essentially identical.

    The only measurable difference is a ~offset/R_earth (~8e-6) inflation of the
    true area for facets lifted 50 m higher on the ellipsoid, which the fast path
    deliberately ignores -- negligible (~7e-5 dB) for radiometry."""
    scene = gen(extent=8000.0)
    frame = LocalFrame.centered_on(scene)
    ref = build_facets(scene.dem, scene.transform, scene.crs, frame)

    offset = 50.0
    fast = offset_facets(ref, frame, offset)
    full = build_facets(np.asarray(scene.dem, np.float64) + offset,
                        scene.transform, scene.crs, frame)

    assert np.abs(fast.centers - full.centers).max() < 1e-3  # < 1 mm
    assert _angle_deg(fast.normals, full.normals).max() < 1e-4  # deg
    # Area ~ radius^2, so agrees to the ~2*offset/R ellipsoidal inflation only.
    rel_area = (np.abs(fast.areas - full.areas) / full.areas).max()
    assert rel_area < 3e-5
    assert rel_area == pytest.approx(2.0 * offset / 6.371e6, rel=0.2)


def test_offset_facets_geometry():
    """offset_facets shifts along local up by exactly the offset and preserves
    normals/areas/edges identically (shared arrays)."""
    scene = syn.flat_scene()
    frame = LocalFrame.centered_on(scene)
    ref = build_facets(scene.dem, scene.transform, scene.crs, frame)
    shifted = offset_facets(ref, frame, -2.0)
    disp = shifted.centers - ref.centers
    assert np.allclose(np.linalg.norm(disp, axis=1), 2.0)
    assert np.array_equal(shifted.normals, ref.normals)
    assert np.array_equal(shifted.areas, ref.areas)


def _cfg(interfaces, media):
    return SimConfig(mode="incoherent",
                     radar=RadarConfig(dt=1e-9, n_samples=64, t0=0.0),
                     facets=FacetConfig(), media=media, interfaces=interfaces)


def test_flat_interface_synthesis_over_reference_footprint():
    """A flat interface is synthesized on the reference DEM's grid; its facet
    count matches a same-grid build."""
    scene = syn.flat_scene(elevation=500.0)
    frame = LocalFrame.centered_on(scene)
    ref = build_facets(scene.dem, scene.transform, scene.crs, frame)
    cfg = _cfg([DemInterface(name="surface"),
                FlatInterface(name="bed", elevation=200.0)],
               [Medium(name="air", eps_r=1.0), Medium(name="ice", eps_r=3.17),
                Medium(name="bed", eps_r=6.0)])
    layered = build_layered_scene(cfg, frame, scene.dem, scene.transform,
                                  scene.crs)
    assert len(layered) == 2
    assert layered.interfaces[1].centers.shape == ref.centers.shape
    # Flat bed sits ~300 m below the surface facets.
    drop = ref.centers[:, 2].mean() - layered.interfaces[1].centers[:, 2].mean()
    assert drop == pytest.approx(300.0, abs=1.0)


def test_layered_scene_offset_interface_uses_fastpath():
    scene = syn.flat_scene(elevation=500.0)
    frame = LocalFrame.centered_on(scene)
    cfg = _cfg([DemInterface(name="surface"),
                OffsetInterface(name="firn", reference="surface", offset=-3.0)],
               [Medium(name="air", eps_r=1.0), Medium(name="firn", eps_r=2.5),
                Medium(name="ice", eps_r=3.17)])
    layered = build_layered_scene(cfg, frame, scene.dem, scene.transform,
                                  scene.crs)
    d = layered.interfaces[0].centers - layered.interfaces[1].centers
    assert np.allclose(np.linalg.norm(d, axis=1), 3.0)
    assert np.array_equal(layered.interfaces[1].normals,
                          layered.interfaces[0].normals)


@pytest.mark.parametrize("factory", syn.MULTILAYER_SCENES)
def test_multilayer_variants_build(factory):
    scene = factory()
    frame = LocalFrame.centered_on(
        type("S", (), {"dem": scene.dem, "transform": scene.transform,
                       "crs": scene.crs})())
    assert len(scene.media) == len(scene.dems) + 1
    ny, nx = scene.dem.shape
    n = (ny - 1) * (nx - 1)
    for dem in scene.dems:
        assert dem.shape == (ny, nx)
        f = build_facets(dem, scene.transform, scene.crs, frame)
        assert f.centers.shape == (n, 3)
        assert f.areas.shape == (n,)


def test_slab_scene_depth():
    scene = syn.slab_scene(surface=500.0, depth=300.0)
    assert len(scene.dems) == 2
    assert np.allclose(scene.dems[0], 500.0)
    assert np.allclose(scene.dems[1], 200.0)


def test_offset_stack_scene_layers():
    scene = syn.offset_stack_scene(surface=500.0, spacings=(2.0, 3.0, 5.0))
    assert len(scene.dems) == 4  # surface + 3
    assert len(scene.media) == 5
    assert np.allclose(scene.dems[-1], 500.0 - 10.0)
