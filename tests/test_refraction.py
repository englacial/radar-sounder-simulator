"""M15: two-point refraction solve vs analytics and the Fermat referee.

Verification-first, like M9: soundersim.refraction.snell_crossing (JAX,
local-plane Snell solve) is checked against closed forms, against
soundersim.compare.fermat.fermat_crossing (brute-force float64 travel-time
minimization on the true surface), and against itself across dtypes and
iteration budgets. Measured numbers quoted in docstrings are from the seeded
sweeps below on x86-64 CPU.

D3-1 evidence (rough surfaces, test_rough_surface_local_plane_error): the
local-plane answer is exact when the plane is tangent AT the crossing; the
error is purely an ANCHORING error, linear in roughness amplitude A and
quadratic in the anchor-to-crossing offset delta (plane extrapolation
delta_z ~ A*k^2*delta^2/2 entering the delay as |n2*cos(theta2) -
n1*cos(theta1)|*delta_z). Midpoint anchoring over a ~200 m offset gives
~6 m/A of optical-path error (20 ns/m of A at Lambda = 300 m); facet-scale
anchoring (delta <= 25 m, M16's situation: each facet is its own local plane)
gives <= 0.08 m for A <= 1 m. Local-plane per facet is justified as long as
facets are small against the roughness wavelength.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from soundersim.compare.fermat import fermat_crossing
from soundersim.refraction import snell_crossing

N_ICE = float(np.sqrt(3.17))
C = 299792458.0
K0_195 = 2.0 * np.pi * 195e6 / C          # vacuum wavenumber at 195 MHz
ORIGIN = np.zeros(3)
UP = np.array([0.0, 0.0, 1.0])


def _opl(res, n1, n2):
    return n1 * np.asarray(res.s1, np.float64) + n2 * np.asarray(res.s2,
                                                                 np.float64)


def _sweep(n, seed=0):
    """Seeded geometry sweep: airborne + stratospheric platforms over tilted
    planes; returns (p, q, plane_point, plane_normal, opposite_side_mask)."""
    rng = np.random.default_rng(seed)
    h = np.where(rng.random(n) < 0.5, rng.uniform(300, 3000, n),
                 rng.uniform(14e3, 20e3, n))
    depth = rng.uniform(10, 4000, n)
    off = rng.uniform(-5000, 5000, (n, 2))
    p = np.column_stack([np.zeros(n), np.zeros(n), h])
    q = np.column_stack([off[:, 0], off[:, 1], -depth])
    tilt = rng.uniform(0, np.deg2rad(20), n)
    az = rng.uniform(0, 2 * np.pi, n)
    nrm = np.column_stack([np.sin(tilt) * np.cos(az),
                           np.sin(tilt) * np.sin(az), np.cos(tilt)])
    pt = rng.uniform(-50, 50, (n, 3))
    opp = (np.sum((p - pt) * nrm, -1) * np.sum((q - pt) * nrm, -1)) < 0
    return p, q, pt, nrm, opp


def test_vertical_incidence_exact():
    p, q = np.array([100.0, -50.0, 1200.0]), np.array([100.0, -50.0, -300.0])
    r = snell_crossing(p, q, ORIGIN, UP, 1.0, N_ICE, xp=np)
    assert bool(r.valid)
    np.testing.assert_array_equal(r.x, [100.0, -50.0, 0.0])
    assert float(r.theta1) == 0.0 and float(r.theta2) == 0.0
    assert float(r.s1) == 1200.0 and float(r.s2) == 300.0
    assert float(r.residual) == 0.0


def test_equal_indices_straight_line():
    p, q = np.array([0.0, 0.0, 1000.0]), np.array([600.0, 300.0, -500.0])
    r = snell_crossing(p, q, ORIGIN, UP, 1.3, 1.3, xp=np)
    assert bool(r.valid)
    np.testing.assert_array_equal(r.x, p + (1000.0 / 1500.0) * (q - p))
    np.testing.assert_allclose(float(r.theta1), float(r.theta2), rtol=1e-15)


def test_mirror_symmetry_and_reciprocity():
    p = np.array([0.0, 0.0, 900.0])
    qr = np.array([1200.0, -300.0, -800.0])
    ql = np.array([-1200.0, 300.0, -800.0])
    rr = snell_crossing(p, qr, ORIGIN, UP, 1.0, N_ICE, xp=np)
    rl = snell_crossing(p, ql, ORIGIN, UP, 1.0, N_ICE, xp=np)
    np.testing.assert_allclose(rr.x, -rl.x, atol=1e-9)   # mirrored crossing
    np.testing.assert_allclose(rr.theta1, rl.theta1, atol=1e-15)
    # Reciprocity: the upward solve (swap endpoints and indices) is the same
    # path -- this exercises the denser-medium-first branch (n1 > n2).
    ru = snell_crossing(qr, p, ORIGIN, UP, N_ICE, 1.0, xp=np)
    assert bool(ru.valid)
    np.testing.assert_allclose(ru.x, rr.x, atol=1e-9)
    np.testing.assert_allclose(ru.theta1, rr.theta2, atol=1e-12)
    np.testing.assert_allclose(ru.theta2, rr.theta1, atol=1e-12)


def test_snell_residual_sweep_f64():
    """|n1 sin(theta1) - n2 sin(theta2)| < 1e-6 (measured: <= 1.4e-12) over a
    broad randomized geometry sweep, both propagation directions; same-side
    pairs are masked invalid with finite values."""
    p, q, pt, nrm, opp = _sweep(4096)
    down = snell_crossing(p, q, pt, nrm, 1.0, N_ICE, xp=np)
    up = snell_crossing(q, p, pt, nrm, N_ICE, 1.0, xp=np)
    for r in (down, up):
        assert r.valid[opp].all()
        assert np.abs(r.residual[opp]).max() < 1e-6
        assert all(np.isfinite(np.asarray(v)).all() for v in r[:-1])
    assert not down.valid[~opp].any()
    # Up/down solves agree on the crossing (reciprocity across the sweep).
    assert np.linalg.norm(down.x - up.x, axis=-1)[opp].max() < 1e-9


def test_vs_fermat_flat():
    """Flat interface: agreement with the brute-force referee to its own
    precision floor (~1e-4 m crossing; opl is stationary, ~1e-11 m)."""
    flat = lambda x, y: 0.0 * x
    cases = [(np.array([0.0, 0.0, 500.0]), np.array([300.0, -100.0, -100.0])),
             (np.array([0.0, 0.0, 3000.0]), np.array([4000.0, 1000.0, -2000.0])),
             (np.array([0.0, 0.0, 20000.0]), np.array([2000.0, 0.0, -1000.0])),
             (np.array([0.0, 0.0, 300.0]), np.array([50.0, 20.0, -10.0]))]
    for p, q in cases:
        r = snell_crossing(p, q, ORIGIN, UP, 1.0, N_ICE, xp=np)
        f = fermat_crossing(p, q, flat, 1.0, N_ICE)
        assert bool(r.valid)
        assert np.linalg.norm(r.x - f.x) < 2e-4
        assert abs(_opl(r, 1.0, N_ICE) - f.opl) < 1e-9


@pytest.mark.parametrize("deg", [5.0, 15.0])
def test_vs_fermat_tilted(deg):
    """Tilted plane (the plane IS the surface): exact up to referee floor."""
    m = np.tan(np.deg2rad(deg))
    srf = lambda x, y: m * x + 0.3 * m * y
    nrm = np.array([-m, -0.3 * m, 1.0])
    nrm /= np.linalg.norm(nrm)
    p, q = np.array([0.0, 0.0, 2000.0]), np.array([1500.0, 400.0, -900.0])
    r = snell_crossing(p, q, ORIGIN, nrm, 1.0, N_ICE, xp=np)
    f = fermat_crossing(p, q, srf, 1.0, N_ICE)
    assert bool(r.valid)
    assert np.linalg.norm(r.x - f.x) < 2e-4
    assert abs(_opl(r, 1.0, N_ICE) - f.opl) < 1e-9


def test_rough_surface_local_plane_error():
    """D3-1 evidence: local-plane error on a gently rough (sinusoid) surface.

    Sinusoid amplitude A in {0.25, 1, 4} m, wavelength 300 m; p at 800 m
    altitude, q at 700 m depth, 1.6 km offset. Measured (recorded per repo
    convention):

    - Plane tangent at the HORIZONTAL MIDPOINT (~200 m from the true
      crossing): crossing error ~ 20*A m and optical-path error ~ 6*A m
      (5.0 / 19.9 / 77.9 ns of delay error at A = 0.25 / 1 / 4 m) -- linear in
      A (log-log slope ~1.0), coefficient set by the tangent-plane
      extrapolation slope error k*delta over the anchor-to-crossing offset
      delta.
    - Plane tangent AT the true crossing: error collapses to the referee
      floor (crossing <= 2e-5 m, opl <= 1e-12 m) for every A -- the
      local-plane answer is exact when anchored correctly; the rough-surface
      error is purely an anchoring error.
    - Anchor offset sweep (A = 1 m): opl error 0.015 / 0.078 / 0.40 m at
      delta = 12.5 / 25 / 50 m -- quadratic in delta (extrapolation
      delta_z ~ A*k^2*delta^2/2). At facet scale (delta = 25 m) the opl error
      measures 0.025 / 0.078 / 0.033 m for A = 0.25 / 1 / 4 m (not strictly
      proportional to A: the curvature term is phase-dependent), i.e.
      <= ~0.3 ns of delay error for meter-scale roughness.
    """
    lam_s = 300.0
    kx = 2.0 * np.pi / lam_s
    p, q = np.array([0.0, 0.0, 800.0]), np.array([1600.0, 0.0, -700.0])
    xm = 0.5 * (p[0] + q[0])
    amps = np.array([0.25, 1.0, 4.0])

    def tangent_plane(A, xa):
        slope = A * kx * np.cos(kx * xa)
        nrm = np.array([-slope, 0.0, 1.0])
        nrm /= np.linalg.norm(nrm)
        return np.array([xa, 0.0, A * np.sin(kx * xa)]), nrm

    err_mid_x, err_mid_opl, err_anc_x, err_anc_opl, err_d25 = ([] for _ in
                                                               range(5))
    for A in amps:
        srf = lambda x, y, A=A: A * np.sin(kx * x) + 0.0 * y
        f = fermat_crossing(p, q, srf, 1.0, N_ICE)
        # (a) anchored at the horizontal midpoint (per plan)
        pt, nrm = tangent_plane(A, xm)
        r = snell_crossing(p, q, pt, nrm, 1.0, N_ICE, xp=np)
        err_mid_x.append(np.linalg.norm(r.x - f.x))
        err_mid_opl.append(abs(_opl(r, 1.0, N_ICE) - f.opl))
        # (b) anchored at the true crossing (facet-local analog)
        pt, nrm = tangent_plane(A, f.x[0])
        r = snell_crossing(p, q, pt, nrm, 1.0, N_ICE, xp=np)
        err_anc_x.append(np.linalg.norm(r.x - f.x))
        err_anc_opl.append(abs(_opl(r, 1.0, N_ICE) - f.opl))
        # (c) anchor offset 25 m (facet-scale anchoring error)
        pt, nrm = tangent_plane(A, f.x[0] + 25.0)
        r = snell_crossing(p, q, pt, nrm, 1.0, N_ICE, xp=np)
        err_d25.append(abs(_opl(r, 1.0, N_ICE) - f.opl))

    # (a) midpoint anchoring: error linear in A (slope ~1), meters-scale.
    slope = np.polyfit(np.log(amps), np.log(err_mid_opl), 1)[0]
    assert 0.8 < slope < 1.2
    assert err_mid_opl[-1] < 10.0 * amps[-1]
    # (b) crossing-anchored: exact to the referee floor.
    assert max(err_anc_x) < 1e-3
    assert max(err_anc_opl) < 1e-9
    # (c) facet-scale (25 m) anchoring: sub-decimeter opl error (<= ~0.5 ns)
    # across the amplitude range (measured max 0.078 m at A = 1 m).
    assert max(err_d25) < 0.15


def test_degenerate_and_upward_masked_not_nan():
    """Same-side / on-plane endpoints mask invalid with finite values; upward
    (n2 -> n1, n2 > n1) paths refract validly -- at a converged two-point
    solution sin(theta_ice) <= 1/n_ice automatically, so genuine TIR cannot
    occur (see module docstring); the pathological 89.99-deg-exit case masks
    rather than NaNs."""
    cases = [
        (np.array([0.0, 0.0, 1000.0]), np.array([500.0, 0.0, 200.0]),
         1.0, N_ICE, False),                     # both above: no crossing
        (np.array([0.0, 0.0, 0.0]), np.array([500.0, 0.0, -200.0]),
         1.0, N_ICE, False),                     # p on the plane
        (np.array([0.0, 0.0, 1000.0]), np.array([500.0, 0.0, 0.0]),
         1.0, N_ICE, False),                     # q on the plane
        (np.array([0.0, 0.0, -50.0]), np.array([8000.0, 0.0, 60.0]),
         N_ICE, 1.0, True),                      # upward, 89.6 deg exit
        (np.array([0.0, 0.0, -5.0]), np.array([50000.0, 0.0, 5.0]),
         N_ICE, 1.0, False),                     # upward, 89.99 deg exit
        (np.array([0.0, 0.0, 300.0]), np.array([7000.0, 7000.0, -10.0]),
         1.0, N_ICE, True),                      # grazing airborne, 88.3 deg
    ]
    for p, q, n1, n2, expect_valid in cases:
        r = snell_crossing(p, q, ORIGIN, UP, n1, n2, xp=np)
        assert bool(np.all(r.valid)) == expect_valid, (p, q, n1, n2)
        assert all(np.isfinite(np.asarray(v)).all() for v in r[:-1])
        if expect_valid and n1 > n2:  # no-TIR bound at the converged solution
            assert np.sin(float(r.theta1)) <= n2 / n1 + 1e-12


def test_iteration_budget():
    """The fixed default budget (25) matches an 80-iteration solve to
    < 1 mm crossing error across the sweep (measured: 1.5e-11 m)."""
    p, q, pt, nrm, opp = _sweep(4096)
    r25 = snell_crossing(p, q, pt, nrm, 1.0, N_ICE, n_iter=25, xp=np)
    r80 = snell_crossing(p, q, pt, nrm, 1.0, N_ICE, n_iter=80, xp=np)
    dx = np.linalg.norm(r25.x - r80.x, axis=-1)[opp]
    assert dx.max() < 1e-3


def test_f32_vs_f64():
    """f32 JAX solve vs f64 NumPy solve (the M15 dtype decision, measured):
    crossing error <= 4.6 cm max / 0.3 mm median (facet sizes are tens of
    meters); direct f32 optical-path error <= 5.7 mm = 0.023 rad two-way at
    195 MHz; recomputing s1/s2 in f64 from the f32 crossing point leaves
    <= 6.2e-4 m = 0.0025 rad (Fermat stationarity), the recommended
    coherent-phase route for M16."""
    p, q, pt, nrm, opp = _sweep(4096)
    r64 = snell_crossing(p, q, pt, nrm, 1.0, N_ICE, xp=np)
    f32 = lambda a: jnp.asarray(np.asarray(a, np.float32))
    r32 = snell_crossing(f32(p), f32(q), f32(pt), f32(nrm), 1.0, N_ICE)
    assert np.asarray(r32.valid)[opp].all()
    x32 = np.asarray(r32.x, np.float64)
    assert np.linalg.norm(x32 - r64.x, axis=-1)[opp].max() < 0.1
    opl64 = _opl(r64, 1.0, N_ICE)[opp]
    opl32 = _opl(r32, 1.0, N_ICE)[opp]
    assert np.abs(opl32 - opl64).max() < 0.02          # ~0.08 rad at 195 MHz
    opl_re = (np.linalg.norm(p - x32, axis=-1)
              + N_ICE * np.linalg.norm(x32 - q, axis=-1))[opp]
    assert np.abs(opl_re - opl64).max() < 2e-3         # ~0.008 rad at 195 MHz
    assert np.abs(opl_re - opl64).max() * K0_195 < 0.05


def test_broadcasting_one_p_many_q_many_planes():
    p = np.array([0.0, 0.0, 1000.0])
    q = np.stack([np.array([300.0, -200.0, -50.0]),
                  np.array([-1500.0, 0.0, -2000.0])])
    nrm = np.stack([UP, UP])
    pt = np.stack([ORIGIN, np.array([0.0, 0.0, -10.0])])
    r = snell_crossing(p, q, pt, nrm, 1.0, N_ICE, xp=np)
    assert r.x.shape == (2, 3) and r.valid.shape == (2,)
    for i in range(2):
        ri = snell_crossing(p, q[i], pt[i], nrm[i], 1.0, N_ICE, xp=np)
        np.testing.assert_array_equal(r.x[i], ri.x)


def test_runtime_vectorized():
    """Per-pair cost of the jitted f32 solve (feeds M16 sizing). Measured
    ~58 ns/pair at 1e6 pairs on a 24-core CPU (~510 ns/pair for the NumPy
    f64 path); the 2 us/pair assertion is a generous regression guard."""
    n = 1_000_000
    rng = np.random.default_rng(1)
    p = np.column_stack([rng.uniform(-100, 100, n), rng.uniform(-100, 100, n),
                         rng.uniform(300, 20000, n)]).astype(np.float32)
    q = np.column_stack([rng.uniform(-5000, 5000, n),
                         rng.uniform(-5000, 5000, n),
                         -rng.uniform(10, 4000, n)]).astype(np.float32)
    pt, nrm = ORIGIN.astype(np.float32), UP.astype(np.float32)
    fn = jax.jit(lambda p, q: snell_crossing(p, q, pt, nrm, 1.0, N_ICE))
    jax.block_until_ready(fn(jnp.asarray(p), jnp.asarray(q)))  # compile
    best = min(_timed(fn, p, q) for _ in range(3))
    per_pair = best / n
    print(f"\nsnell_crossing: {per_pair * 1e9:.0f} ns/pair "
          f"({n} pairs, f32, jitted, CPU)")
    assert per_pair < 2e-6


def _timed(fn, p, q):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(jnp.asarray(p), jnp.asarray(q)))
    return time.perf_counter() - t0
