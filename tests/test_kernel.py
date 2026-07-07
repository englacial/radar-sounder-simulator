"""Incoherent kernel tests: hand-computed physics, energy conservation,
left/right split, block padding, and f32 kernel vs f64 NumPy reference."""

import numpy as np
import pytest

from soundersim.config import FacetConfig, RadarConfig, SimConfig
from soundersim.kernels.incoherent import incoherent_cluttergram
from soundersim.nav import nav_to_frame
from soundersim.scene import LocalFrame, build_facets
from soundersim.synthetic import sinusoid_scene

C = 299792458.0
RC = dict(t0=0.0, dt=1e-8, n_samples=1250, c=C)


def reference_incoherent(positions, centers, normals, areas, *, t0, dt, n_samples, c):
    """Pure-NumPy float64 reference implementation of the kernel."""
    T = len(positions)
    out = np.zeros((T, n_samples))
    dropped = np.zeros(T)
    for i, p in enumerate(positions):
        d = p - centers
        r = np.linalg.norm(d, axis=1)
        cos = (d * normals).sum(axis=1) / r
        pwr = (areas * cos) ** 2 / r ** 4
        b = np.floor((2 * r / c - t0) / dt).astype(int)
        ok = (b >= 0) & (b < n_samples)
        np.add.at(out[i], b[ok], pwr[ok])
        dropped[i] = pwr[~ok].sum()
    return out, dropped


@pytest.fixture(scope="module")
def small_scene():
    """Sinusoid scene geometry in its local frame (3042 facets, 5 traces)."""
    scene = sinusoid_scene(extent=2000.0, n_traces=5, spacing=200.0)
    frame = LocalFrame.centered_on(scene)
    facets = build_facets(scene.dem, scene.transform, scene.crs, frame)
    track = nav_to_frame(scene.nav_llh, frame)
    return facets, track


def test_single_facet_power_and_twtt():
    """One triangle, one trace: power and bin exact vs hand computation."""
    v1, v2, v3 = np.array([[0.0, 0, 0], [30.0, 0, 0], [0.0, 40, 10]])
    center = (v1 + v2 + v3) / 3
    raw_n = np.cross(v2 - v1, v3 - v1)
    area = np.linalg.norm(raw_n) / 2
    normal = raw_n / np.linalg.norm(raw_n)
    p = np.array([[50.0, -30.0, 1500.0]])

    d = p[0] - center
    r = np.linalg.norm(d)
    cos = d @ normal / r
    expected_power = (area * cos) ** 2 / r ** 4
    expected_bin = int(np.floor(2 * r / C / RC["dt"]))

    power, dropped = incoherent_cluttergram(
        p, np.array([[0.0, -1.0, 0.0]]), center[None], normal[None],
        np.array([area]), **RC)
    assert power.shape == (1, RC["n_samples"])
    assert dropped[0] == 0.0
    nz = np.nonzero(power[0])[0]
    assert list(nz) == [expected_bin]
    assert power[0, expected_bin] == pytest.approx(expected_power, rel=1e-5)


def test_out_of_window_power_is_dropped_not_wrapped():
    """A facet beyond the window lands entirely in dropped, nowhere in bins."""
    center = np.array([[0.0, 0.0, 0.0]])
    normal = np.array([[0.0, 0.0, 1.0]])
    area = np.array([1250.0])
    p = np.array([[0.0, 0.0, 1000.0]])  # twtt 6.67e-6 > 32-sample window
    power, dropped = incoherent_cluttergram(
        p, np.array([[0.0, -1.0, 0.0]]), center, normal, area,
        t0=0.0, dt=1e-8, n_samples=32, c=C)
    assert power.sum() == 0.0
    assert dropped[0] == pytest.approx((area[0] * 1.0) ** 2 / 1000.0 ** 4, rel=1e-5)


def test_energy_conservation(small_scene):
    """Sum of binned power + dropped = sum of per-facet powers (f64)."""
    facets, track = small_scene
    power, dropped = incoherent_cluttergram(
        track.positions, track.u_ct, facets.centers, facets.normals,
        facets.areas, **RC)
    d = track.positions[:, None, :] - facets.centers[None]
    r = np.linalg.norm(d, axis=-1)
    cos = (d * facets.normals[None]).sum(-1) / r
    total = ((facets.areas[None] * cos) ** 2 / r ** 4).sum(axis=1)
    np.testing.assert_allclose(power.sum(axis=1) + dropped, total, rtol=1e-4)


def test_left_plus_right_equals_combined(small_scene):
    facets, track = small_scene
    combined, d0 = incoherent_cluttergram(
        track.positions, track.u_ct, facets.centers, facets.normals,
        facets.areas, **RC)
    split, d1 = incoherent_cluttergram(
        track.positions, track.u_ct, facets.centers, facets.normals,
        facets.areas, split_sides=True, **RC)
    assert split.shape == combined.shape + (2,)
    assert split[..., 0].sum() > 0 and split[..., 1].sum() > 0
    np.testing.assert_allclose(split.sum(axis=-1), combined,
                               rtol=1e-5, atol=combined.max() * 1e-6)
    np.testing.assert_allclose(d0, d1, rtol=1e-5)


def test_block_processing_matches_single_block(small_scene):
    """Fixed-size facet blocks (with zero-area padding) change nothing."""
    facets, track = small_scene
    a, da = incoherent_cluttergram(
        track.positions, track.u_ct, facets.centers, facets.normals,
        facets.areas, **RC)
    b, db = incoherent_cluttergram(
        track.positions, track.u_ct, facets.centers, facets.normals,
        facets.areas, block_size=1000, **RC)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=a.max() * 1e-7)
    np.testing.assert_allclose(da, db, rtol=1e-5)


def test_f32_kernel_vs_f64_reference(small_scene):
    """f32 JAX kernel vs f64 NumPy reference on the sinusoid scene.

    Total power agrees tightly. Per bin (above -60 dB of peak) the relative
    difference is <= 1e-3 except for rare bin-edge migrations: a facet whose
    twtt/dt lands within f32 rounding of a bin edge may bin one sample apart.
    Those are identified exactly: few (<0.5% of bins) and energy-local (3-bin
    neighborhood sums still agree to 1e-3).
    """
    facets, track = small_scene
    power, dropped = incoherent_cluttergram(
        track.positions, track.u_ct, facets.centers, facets.normals,
        facets.areas, **RC)
    ref, ref_dropped = reference_incoherent(
        track.positions, facets.centers, facets.normals, facets.areas, **RC)
    np.testing.assert_allclose(power.sum(), ref.sum(), rtol=1e-4)
    np.testing.assert_allclose(dropped, ref_dropped, rtol=1e-4, atol=1e-30)
    mask = ref > ref.max() * 1e-6  # bins above -60 dB rel peak
    rel = np.abs(power - ref) / np.where(mask, ref, np.inf)
    bad = np.argwhere(rel > 1e-3)
    assert len(bad) <= 0.005 * mask.sum(), f"{len(bad)} outlier bins"
    for t, k in bad:  # each outlier is a local bin-edge migration
        lo, hi = max(k - 1, 0), k + 2
        assert power[t, lo:hi].sum() == pytest.approx(
            ref[t, lo:hi].sum(), rel=1e-3)
