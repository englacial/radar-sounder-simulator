"""Coherent LPA facet kernel vs the M9 float64 brute-force referee (M10).

Measured LPA validity envelope (test_single_facet_lpa_vs_brute_force, errors
normalized by the facet's sinc=1 amplitude envelope (k/2pi)|G|cos(theta)A/r^2):

    r=300lam   th=0    15     30     45     60
    L= 0.5lam  0.001  0.003  0.007  0.008  0.005
    L= 1.0lam  0.004  0.002  0.002  0.005  0.005
    L= 2.0lam  0.014  0.005  0.001  0.001  0.003
    L= 5.0lam  0.087  0.015  0.001  0.000  0.001
    L=10.0lam  0.343  0.010  0.000  0.001  0.001
    (r=100lam: L=2 @ 0deg -> 0.042; L=5 @ 0deg -> 0.26; L=10 @ 0deg -> 0.90)

i.e. the 5% breakdown is at LARGE SIZE x NEAR-NADIR incidence -- the neglected
quadratic (Fresnel) phase k*L^2/(4r), crossing ~5% at L ~ 0.23*sqrt(lam*r) at
nadir; off-nadir (>= 15 deg) the sinc suppression makes even 10lam facets
agree to <1.5% of the envelope. (Errors relative to the LOCAL amplitude blow
up near sinc nulls where the field -> 0, hence the envelope normalization.)

Phase precision (test_phase_precision, 20 km range at 195 MHz): the shipped
strategy -- f32 hot loop on 2k*(r - r_ref) with a per-trace f64 reference range
folded back in complex128 -- measures ~2.3 mm equivalent range error (lam/660),
vs the lam/50 = 30.7 mm requirement. A naive f32 exp(-2jkr) measures ~2.1 mm
here (both are floored by the f32 rounding of r itself at 20 km); the
reference-range subtraction is kept because the naive path's phase-argument
storage error grows linearly with absolute range (marginal by ~500 km) while
the subtracted argument tracks only the in-scene range spread.
"""

import numpy as np
import pytest

from soundersim.compare.brute_force import _contributions
from soundersim.kernels.coherent import coherent_cluttergram, lpa_contributions

LAM = 1.0
K = 2.0 * np.pi / LAM
GAMMA = -0.281
C = 299792458.0
UCT = np.array([[0.0, -1.0, 0.0]])


def facet_samples(centers, e1, e2, spacing):
    """Cell-centered sub-samples of parallelogram facets sharing edge vectors.

    centers (N, 3); e1/e2 (3,) common edge vectors. Returns points (N, S, 3),
    normals (S*N-compatible (3,)), per-sample area (scalar).
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    n1 = max(1, int(np.ceil(np.linalg.norm(e1) / spacing)))
    n2 = max(1, int(np.ceil(np.linalg.norm(e2) / spacing)))
    u = (np.arange(n1) + 0.5) / n1 - 0.5
    v = (np.arange(n2) + 0.5) / n2 - 0.5
    U, V = np.meshgrid(u, v, indexing="ij")
    offsets = U.ravel()[:, None] * e1 + V.ravel()[:, None] * e2  # (S, 3)
    pts = np.atleast_2d(centers)[:, None, :] + offsets[None, :, :]
    raw = np.cross(e1, e2)
    area = np.linalg.norm(raw)
    return pts, raw / area, area / (n1 * n2)


def bf_facet_fields(p, centers, e1, e2, spacing=LAM / 12):
    """Float64 brute-force field of each facet (sub-wavelength integration)."""
    pts, nrm, dA = facet_samples(centers, e1, e2, spacing)
    n, s = pts.shape[:2]
    contrib, _ = _contributions(p, pts.reshape(-1, 3), np.tile(nrm, (n * s, 1)),
                                np.full(n * s, dA), K, GAMMA)
    return contrib.reshape(n, s).sum(axis=1)


def lpa_facet_fields(p, centers, e1, e2, k=K):
    """Float64 evaluation of the kernel's LPA closed form (xp=np path)."""
    centers = np.atleast_2d(centers)
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    raw = np.cross(e1, e2)
    area = np.linalg.norm(raw)
    n = len(centers)
    contrib, _ = lpa_contributions(
        np.asarray(p, float), centers, np.tile(raw / area, (n, 1)),
        np.full(n, area), np.tile(e1, (n, 1)), np.tile(e2, (n, 1)),
        k, GAMMA, xp=np)
    return contrib


def test_single_facet_lpa_vs_brute_force():
    """LPA closed form vs sub-lambda brute-force integration of one facet.

    Complex error normalized by the sinc=1 envelope; see module docstring for
    the full measured table and the nadir breakdown at L ~ 0.23*sqrt(lam*r).
    """
    r = 300.0 * LAM
    center = np.zeros(3)
    worst_small, worst_large_offnadir = 0.0, 0.0
    for L in (0.5, 1.0, 2.0, 5.0, 10.0):
        e1, e2 = np.array([L, 0.0, 0.0]), np.array([0.0, L, 0.0])
        for th in (0.0, 15.0, 30.0, 45.0, 60.0):
            t = np.deg2rad(th)
            p = r * np.array([np.sin(t), 0.12 * np.sin(t), 0.0])
            p[2] = np.sqrt(r ** 2 - p[0] ** 2 - p[1] ** 2)
            bf = bf_facet_fields(p, center, e1, e2)[0]
            lp = lpa_facet_fields(p, center, e1, e2)[0]
            env = (K / (2 * np.pi)) * abs(GAMMA) * (p[2] / r) * L * L / r ** 2
            err = abs(lp - bf) / env
            if L <= 2.0:
                worst_small = max(worst_small, err)
            elif th >= 15.0:
                worst_large_offnadir = max(worst_large_offnadir, err)
            # amplitude AND phase where the field is not near a sinc null,
            # inside the validity region (large-L nadir is the documented
            # LPA breakdown, asserted separately below)
            if L <= 2.0 and abs(bf) > 0.2 * env:
                assert abs(abs(lp) / abs(bf) - 1.0) < 0.03  # measured <= 0.023
                assert abs(np.degrees(np.angle(lp / bf))) < 2.0  # measured <= 0.8
    assert worst_small < 0.02          # measured 0.014 (L=2lam at nadir)
    assert worst_large_offnadir < 0.02  # measured 0.015 (L=5lam, 15 deg)
    # documented breakdown: nadir error exceeds 5% for L >~ 0.23 sqrt(lam r)
    nadir = np.array([0.0, 0.0, r])
    for L, lo, hi in ((5.0, 0.05, 0.15), (10.0, 0.25, 0.45)):
        e1, e2 = np.array([L, 0.0, 0.0]), np.array([0.0, L, 0.0])
        bf = bf_facet_fields(nadir, center, e1, e2)[0]
        lp = lpa_facet_fields(nadir, center, e1, e2)[0]
        env = (K / (2 * np.pi)) * abs(GAMMA) * L * L / r ** 2
        assert lo < abs(lp - bf) / env < hi  # measured 0.087 / 0.343


def test_small_facet_limit():
    """As facet size -> 0 the LPA field equals the brute-force single-sample
    expression (sinc -> 1) to 1e-10 relative (float64, sinc residual x^2/6)."""
    rng = np.random.default_rng(3)
    for _ in range(4):
        c0 = rng.uniform(-5, 5, 3)
        p = rng.uniform(-20, 20, 3) + np.array([0.0, 0.0, 80.0])
        q1, q2 = rng.normal(size=3), rng.normal(size=3)
        e1 = 1e-6 * q1 / np.linalg.norm(q1)
        q2 -= (q2 @ e1) * e1 / (e1 @ e1)
        e2 = 1e-6 * q2 / np.linalg.norm(q2)
        raw = np.cross(e1, e2)
        area = np.linalg.norm(raw)
        lp = lpa_facet_fields(p, c0, e1, e2)[0]
        bf, _ = _contributions(p, c0[None], (raw / area)[None],
                               np.array([area]), K, GAMMA)
        assert abs(lp - bf[0]) / abs(bf[0]) < 1e-10


def test_multi_facet_flat_scene_vs_brute_force():
    """15x15 facets (L=2lam) tiling a 30lam plate at h=200lam, one trace.

    Per fast-time bin the kernel field matches the facet-wise brute-force
    integration binned identically (LPA + f32 error only); the total field
    also matches the CONTINUOUS brute-force plate, confirming the facet
    decomposition. (Per-bin comparison against continuously-binned brute force
    is dominated by facets straddling bin-edge range rings -- a binning
    granularity effect, not a field error -- hence the facet-binned reference.)
    """
    d, ncell, h = 2.0 * LAM, 15, 200.0 * LAM
    ax = (np.arange(ncell) - (ncell - 1) / 2) * d
    X, Y = np.meshgrid(ax, ax)
    centers = np.column_stack([X.ravel(), Y.ravel(), np.zeros(ncell ** 2)])
    e1v, e2v = np.array([d, 0.0, 0.0]), np.array([0.0, d, 0.0])
    n = len(centers)
    normals = np.tile([0.0, 0.0, 1.0], (n, 1))
    areas = np.full(n, d * d)
    e1, e2 = np.tile(e1v, (n, 1)), np.tile(e2v, (n, 1))

    p = np.array([[3.1 * LAM, -1.7 * LAM, h]])
    r64 = np.linalg.norm(p[0] - centers, axis=1)
    t0 = 2 * r64.min() / C - 2e-10
    dt = (2 * (r64.max() - r64.min()) / C + 4e-10) / 8  # ~8 occupied bins
    n_samples = 12
    field, dropped = coherent_cluttergram(
        p, UCT, centers, normals, areas, e1, e2, k=K, gamma=GAMMA,
        t0=t0, dt=dt, n_samples=n_samples, c=C)
    assert field.dtype == np.complex64 and dropped[0] == 0.0

    # facet-wise brute force, binned by facet-center range
    bf = bf_facet_fields(p[0], centers, e1v, e2v)
    b = np.floor((2 * r64 / C - t0) / dt).astype(int)
    ref = np.zeros(n_samples, complex)
    np.add.at(ref, b, bf)
    occ = np.abs(ref) > 1e-3 * np.abs(ref).max()
    assert occ.sum() >= 6
    amp = np.abs(np.abs(field[0, occ]) / np.abs(ref[occ]) - 1.0)
    ph = np.abs(np.degrees(np.angle(field[0, occ] / ref[occ])))
    assert amp.max() < 0.01   # measured 6e-4
    assert ph.max() < 3.0     # measured 1.2 deg (aggregate LPA phase)

    # total field vs the continuous plate (envelope < 1%, phase < 3 deg)
    pts, nrm, dA = facet_samples(np.zeros(3), ncell * e1v, ncell * e2v,
                                 LAM / 12)
    cont, _ = _contributions(p[0], pts[0], np.tile(nrm, (pts.shape[1], 1)),
                             np.full(pts.shape[1], dA), K, GAMMA)
    tot_bf, tot_k = cont.sum(), field[0].sum()
    assert abs(abs(tot_k) / abs(tot_bf) - 1.0) < 0.01        # measured 5e-4
    assert abs(np.degrees(np.angle(tot_k / tot_bf))) < 3.0   # measured 1.2

    # f32 kernel vs the same LPA formula in f64: pure float error
    lpa64 = lpa_facet_fields(p[0], centers, e1v, e2v)
    ref64 = np.zeros(n_samples, complex)
    np.add.at(ref64, b, lpa64)
    assert (np.abs(field[0] - ref64).max() / np.abs(ref64).max()) < 5e-4
    # measured 3e-5


def test_phase_precision_20km_195mhz():
    """Plan constraint 2 gate: kernel (f32 + f64 reference-range foldback)
    phase vs the f64 brute force at 20 km / 195 MHz < lam/50 equivalent range
    error. Measured ~2.3 mm = lam/660 (requirement 30.7 mm); see module
    docstring for the strategy decision."""
    lam = C / 195e6
    k = 2 * np.pi / lam
    L = 5.0
    rng = np.random.default_rng(4)
    nf = 12
    centers = np.column_stack([rng.uniform(-2000, 2000, nf),
                               rng.uniform(-2000, 2000, nf),
                               rng.uniform(-30, 30, nf)])
    e1 = np.tile([L, 0.0, 0.0], (nf, 1))
    e2 = np.tile([0.0, L, 0.0], (nf, 1))
    normals = np.tile([0.0, 0.0, 1.0], (nf, 1))
    areas = np.full(nf, L * L)
    p = np.array([[0.0, 0.0, 20000.0]])
    r64 = np.linalg.norm(p[0] - centers, axis=1)
    t0, dt, n_samples = 2 * r64.min() / C - 1e-8, 5e-9, 512
    b = np.floor((2 * r64 / C - t0) / dt).astype(int)
    assert len(set(b)) == nf  # one facet per bin -> bin value = facet field

    field, _ = coherent_cluttergram(p, UCT, centers, normals, areas, e1, e2,
                                    k=k, gamma=GAMMA, t0=t0, dt=dt,
                                    n_samples=n_samples, c=C)
    for i in range(nf):
        pts, nrm, dA = facet_samples(centers[i], e1[i], e2[i], lam / 10)
        contrib, _ = _contributions(p[0], pts[0],
                                    np.tile(nrm, (pts.shape[1], 1)),
                                    np.full(pts.shape[1], dA), k, GAMMA)
        dphi = np.angle(field[0, b[i]] / contrib.sum())
        assert abs(dphi) / (2 * k) < lam / 50  # measured max ~lam/660


def test_energy_bookkeeping():
    """Sum of |binned field|^2 + dropped equals the sum of per-contribution
    |field|^2 when facets occupy distinct bins; dropped collects exactly the
    out-of-window contributions' power."""
    nf = 8
    rng = np.random.default_rng(9)
    # ranges spaced ~3 bins apart; last two facets pushed out of the window
    centers = np.column_stack([rng.uniform(-40, 40, nf),
                               rng.uniform(-40, 40, nf),
                               60.0 * np.arange(nf) - 100.0])
    L = 2.0
    e1 = np.tile([L, 0.0, 0.0], (nf, 1))
    e2 = np.tile([0.0, L, 0.0], (nf, 1))
    normals = np.tile([0.0, 0.0, 1.0], (nf, 1))
    areas = np.full(nf, L * L)
    p = np.array([[0.0, 0.0, 800.0]])
    r64 = np.linalg.norm(p[0] - centers, axis=1)
    t0, dt = 2 * r64.min() / C - 1e-8, 4e-8
    b_all = np.floor((2 * r64 / C - t0) / dt).astype(int)
    n_samples = int(np.sort(b_all)[-3])  # window cuts off the two latest bins
    assert len(set(b_all)) == nf

    field, dropped = coherent_cluttergram(p, UCT, centers, normals, areas,
                                          e1, e2, k=K, gamma=GAMMA, t0=t0,
                                          dt=dt, n_samples=n_samples, c=C)
    contrib, _ = lpa_contributions(p[0], centers, normals, areas, e1, e2,
                                   K, GAMMA, xp=np)
    out = b_all >= n_samples
    assert out.sum() >= 2
    total = (np.abs(contrib) ** 2).sum()
    binned = (np.abs(field[0]) ** 2).sum()
    np.testing.assert_allclose(binned + dropped[0], total, rtol=1e-4)
    np.testing.assert_allclose(dropped[0], (np.abs(contrib[out]) ** 2).sum(),
                               rtol=1e-4)


def _sinusoid_facets(ncell=20, d=3.0, amp=1.5):
    """Small rough-ish facet grid built from vertex heights (float64)."""
    ax = np.arange(ncell + 1) * d - ncell * d / 2
    X, Y = np.meshgrid(ax, ax)
    Z = amp * np.sin(2 * np.pi * X / 17.0) * np.cos(2 * np.pi * Y / 23.0)
    V = np.stack([X, Y, Z], axis=-1)
    v00, v01, v10, v11 = V[:-1, :-1], V[:-1, 1:], V[1:, :-1], V[1:, 1:]
    e1 = ((v01 + v11) - (v00 + v10)) / 2.0
    e2 = ((v10 + v11) - (v00 + v01)) / 2.0
    centers = ((v00 + v01 + v10 + v11) / 4.0).reshape(-1, 3)
    raw = np.cross(e1, e2).reshape(-1, 3)
    areas = np.linalg.norm(raw, axis=1)
    normals = raw / areas[:, None]
    return centers, normals, areas, e1.reshape(-1, 3), e2.reshape(-1, 3)


def test_split_sides_fields_sum_to_combined():
    """Left + right FIELDS equal the combined field; dropped power matches."""
    centers, normals, areas, e1, e2 = _sinusoid_facets()
    p = np.array([[5.0, 2.0, 150.0], [-8.0, -3.0, 160.0]])
    uct = np.tile(UCT, (2, 1))
    rc = dict(k=K, gamma=GAMMA, t0=2 * 140.0 / C, dt=2e-9, n_samples=64, c=C)
    comb, d0 = coherent_cluttergram(p, uct, centers, normals, areas, e1, e2,
                                    **rc)
    split, d1 = coherent_cluttergram(p, uct, centers, normals, areas, e1, e2,
                                     split_sides=True, **rc)
    assert split.shape == comb.shape + (2,)
    assert np.abs(split[..., 0]).sum() > 0 and np.abs(split[..., 1]).sum() > 0
    np.testing.assert_allclose(split.sum(axis=-1), comb, rtol=1e-5,
                               atol=np.abs(comb).max() * 1e-5)
    np.testing.assert_allclose(d0, d1, rtol=1e-5)


def test_block_processing_matches_single_block():
    centers, normals, areas, e1, e2 = _sinusoid_facets()
    p = np.array([[5.0, 2.0, 150.0]])
    rc = dict(k=K, gamma=GAMMA, t0=2 * 140.0 / C, dt=2e-9, n_samples=64, c=C)
    a, da = coherent_cluttergram(p, UCT, centers, normals, areas, e1, e2, **rc)
    b, db = coherent_cluttergram(p, UCT, centers, normals, areas, e1, e2,
                                 block_size=97, **rc)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=np.abs(a).max() * 1e-6)
    np.testing.assert_allclose(da, db, rtol=1e-5)
