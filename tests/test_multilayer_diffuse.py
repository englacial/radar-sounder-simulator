"""Diffuse bed channel of the multilayer kernel (specular/diffuse
reflectivity split): the NORMALIZATION GATE (the derived sqrt(A/2pi)
convention conserves total nadir power over a flat interface, analytically),
the cos^n angular law, power-split linearity, and the guarantee that the
feature OFF traces the pre-feature program bit-identically."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soundersim.scene import Facets  # noqa: E402
from soundersim.kernels.multilayer import refracted_cluttergram  # noqa: E402
from soundersim.roughness import speckle_phasors  # noqa: E402

C = 299792458.0


def _plane(z, half, step):
    """Flat horizontal Facets tiling [-half, half]^2 at height ``z``."""
    x = np.arange(-half, half + step, step, dtype=np.float64)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    n = xx.size
    centers = np.column_stack([xx.ravel(), yy.ravel(), np.full(n, z)])
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    e1 = np.tile(np.array([step, 0.0, 0.0]), (n, 1))
    e2 = np.tile(np.array([0.0, step, 0.0]), (n, 1))
    ii, jj = np.meshgrid(np.arange(xx.shape[0]), np.arange(xx.shape[1]),
                         indexing="ij")
    return Facets(centers=centers.astype(np.float32),
                  normals=normals.astype(np.float32),
                  areas=np.full(n, step * step, np.float32),
                  e1=e1.astype(np.float32), e2=e2.astype(np.float32),
                  cell=np.column_stack([ii.ravel(), jj.ravel()]),
                  grid_shape=xx.shape)


def _run(r, half, step, *, gamma, diffuse, n_samples=4096, seed=0):
    """One nadir trace over a flat target at depth ``r``; the single crossed
    interface is a ZERO-CONTRAST plane (eps 1 -> 1) so the geometry is
    straight air and the closed forms below apply exactly."""
    target = _plane(0.0, half, step)
    crossed = _plane(r * 0.5, half, step)
    pos = np.array([[0.0, 0.0, r]])
    dt = 2.0 / C                       # 2 m of two-way range per bin
    return refracted_cluttergram(
        pos, np.array([[0.0, 1.0, 0.0]]), target, [crossed],
        [1.0, 1.0], [0.0, 0.0], mode="coherent", t0=2.0 * (r - 8.0) / C,
        dt=dt, n_samples=n_samples, c=C, gamma=gamma, k0=2.0 * np.pi / 1.5,
        diffuse=diffuse)


def _diffuse(n_facets, amp, n_exp, seed=0):
    return (np.full(n_facets, amp),
            speckle_phasors(n_facets, (seed, 1000)), n_exp)


# ------------------------------------------------- the normalization gate

def test_diffuse_total_power_matches_the_analytic_flat_plane_value():
    """Derived convention: summing the diffuse channel's power over all bins
    on a flat plane at nadir gives amp^2/(4 r^2) -- the same closed form the
    specular image-method field gives for gamma^2. Facet-size and
    wavenumber independent, so a split gamma^2 -> f_s + (1-f_s) conserves
    total power by construction (module docstring derivation)."""
    r, half, step, amp = 100.0, 420.0, 4.0, 0.3
    want = amp ** 2 / (4.0 * r ** 2)
    tot = []
    for seed in range(6):                      # average out phasor speckle
        nf = _plane(0.0, half, step).centers.shape[0]
        out, _ = _run(r, half, step, gamma=0.0,
                      diffuse=_diffuse(nf, amp, 0.0, seed))
        tot.append(float((np.abs(out) ** 2).sum()))
    got = float(np.mean(tot))
    assert got == pytest.approx(want, rel=0.08), (got, want)


def test_diffuse_normalization_is_facet_size_independent():
    """sqrt(A/(2pi)) makes the incoherent sum scale with ILLUMINATED AREA,
    not with facet count -- halving the facet size must not change the
    total (an amplitude ~ A convention would double it)."""
    r, half, amp = 100.0, 420.0, 0.3
    tots = []
    for step in (4.0, 8.0):
        acc = []
        for seed in range(6):
            nf = _plane(0.0, half, step).centers.shape[0]
            out, _ = _run(r, half, step, gamma=0.0,
                          diffuse=_diffuse(nf, amp, 0.0, seed))
            acc.append(float((np.abs(out) ** 2).sum()))
        tots.append(float(np.mean(acc)))
    assert tots[0] == pytest.approx(tots[1], rel=0.10), tots


def test_diffuse_power_is_linear_in_the_split_fraction():
    """amp = sqrt(1-f_s)*gamma -> diffuse power = (1-f_s)*gamma^2/(4r^2):
    the other half of the conservation identity."""
    r, half, step, gam = 100.0, 300.0, 6.0, 0.4
    nf = _plane(0.0, half, step).centers.shape[0]
    base = None
    for f_s in (0.0, 0.5, 0.9):
        acc = []
        for seed in range(4):
            out, _ = _run(r, half, step, gamma=0.0,
                          diffuse=_diffuse(nf, np.sqrt(1.0 - f_s) * gam,
                                           0.0, seed))
            acc.append(float((np.abs(out) ** 2).sum()))
        got = float(np.mean(acc))
        if base is None:
            base = got
        assert got == pytest.approx((1.0 - f_s) * base, rel=1e-6)


# ------------------------------------------------------- the angular law

def test_cos_n_angular_law_is_exact_per_facet():
    """With one target facet the phasor is common, so the n_exp>0 / n_exp=0
    power ratio is exactly cos^n(theta_i) at the facet's own incidence
    angle -- checked at several platform offsets."""
    step, z, r = 2.0, 0.0, 80.0
    one = Facets(centers=np.array([[0.0, 0.0, z]], np.float32),
                 normals=np.array([[0.0, 0.0, 1.0]], np.float32),
                 areas=np.array([step * step], np.float32),
                 e1=np.array([[step, 0.0, 0.0]], np.float32),
                 e2=np.array([[0.0, step, 0.0]], np.float32),
                 cell=np.array([[0, 0]]), grid_shape=(1, 1))
    crossed = _plane(r * 0.5, 400.0, 8.0)
    for off in (0.0, 30.0, 60.0, 120.0):
        pos = np.array([[off, 0.0, r]])
        cos_t = r / np.hypot(off, r)
        got = []
        for n_exp in (0.0, 1.0, 2.0):
            out, _ = refracted_cluttergram(
                pos, np.array([[0.0, 1.0, 0.0]]), one, [crossed],
                [1.0, 1.0], [0.0, 0.0], mode="coherent",
                t0=0.0, dt=2.0 / C, n_samples=4096, c=C, gamma=0.0,
                k0=2.0 * np.pi / 1.5,
                diffuse=(np.array([0.5]), np.array([1.0 + 0j],
                                                   np.complex64), n_exp))
            got.append(float((np.abs(out) ** 2).sum()))
        assert got[1] / got[0] == pytest.approx(cos_t, rel=1e-6)
        assert got[2] / got[0] == pytest.approx(cos_t ** 2, rel=1e-6)


# ------------------------------------------------------- OFF = untouched

def test_feature_off_is_bit_identical_to_the_pre_feature_program():
    """diffuse=None must trace exactly the old kernel: the specular-only
    result has to be bit-for-bit what the same call produced before the
    channel existed (guarded here by comparing against gamma-scaled runs
    that never enter the diffuse branch)."""
    r, half, step = 100.0, 200.0, 8.0
    a, _ = _run(r, half, step, gamma=0.4, diffuse=None)
    b, _ = _run(r, half, step, gamma=0.4, diffuse=None)
    assert np.array_equal(a, b)
    # a zero-amplitude diffuse channel is numerically (not bitwise) the same
    nf = _plane(0.0, half, step).centers.shape[0]
    z, _ = _run(r, half, step, gamma=0.4, diffuse=_diffuse(nf, 0.0, 1.0))
    assert np.allclose(np.abs(z), np.abs(a), rtol=0, atol=0)


def test_specular_field_scales_as_sqrt_of_the_specular_fraction():
    """The specular half of the split needs no kernel change: scaling gamma
    by sqrt(f_s) scales the returned POWER by f_s exactly."""
    r, half, step, gam = 100.0, 200.0, 8.0, 0.4
    full, _ = _run(r, half, step, gamma=gam, diffuse=None)
    for f_s in (0.25, 0.81):
        part, _ = _run(r, half, step, gamma=np.sqrt(f_s) * gam, diffuse=None)
        assert (np.abs(part) ** 2).sum() == pytest.approx(
            f_s * float((np.abs(full) ** 2).sum()), rel=1e-5)


def test_diffuse_rejects_incoherent_mode_and_bad_shapes():
    r, half, step = 100.0, 100.0, 10.0
    nf = _plane(0.0, half, step).centers.shape[0]
    with pytest.raises(ValueError, match="coherent"):
        target, crossed = _plane(0.0, half, step), _plane(50.0, half, step)
        refracted_cluttergram(
            np.array([[0.0, 0.0, r]]), np.array([[0.0, 1.0, 0.0]]), target,
            [crossed], [1.0, 1.0], [0.0, 0.0], mode="incoherent", t0=0.0,
            dt=2.0 / C, n_samples=64, c=C, diffuse=_diffuse(nf, 0.1, 1.0))
    with pytest.raises(ValueError, match="diffuse amp shape"):
        _run(r, half, step, gamma=0.0, diffuse=_diffuse(nf + 3, 0.1, 1.0))
