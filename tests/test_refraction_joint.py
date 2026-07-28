"""D+ validation: joint multi-interface refraction solve vs analytics, the
two-point solve, the multi-interface Fermat referee, and itself.

Verification-first, like M15 (tests/test_refraction.py): the joint solver
(soundersim.refraction_joint.joint_crossings, block-tridiagonal Newton) is
checked against the existing two-point solve (N=1), against the analytic
common-ray-parameter solution on flat parallel stacks (where the sequential
chain's error is also recorded -- the approximation D+ removes), against
soundersim.compare.fermat.fermat_path on tilted (exact) and gently rough
stacks, for reciprocity, for masking, for its fixed iteration budgets across
airborne/stratospheric sweeps at N in {2, 5, 20, 80}, and for O(1)-in-N
compile cost. Measured numbers quoted in docstrings are from the seeded runs
below on x86-64 CPU (float64 under jax.enable_x64()).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from soundersim.compare.fermat import fermat_path
from soundersim.refraction import snell_crossing
from soundersim.refraction_joint import joint_crossings, sequential_chain

N_ICE = float(np.sqrt(3.17))
UP = np.array([0.0, 0.0, 1.0])


def _opl(res, n_media):
    """Total optical path sum_k n_k s_k from a JointCrossings result."""
    s = np.asarray(res.s, np.float64)
    n = np.asarray(n_media, np.float64).reshape((s.shape[0],)
                                                + (1,) * (s.ndim - 1))
    return np.sum(n * s, axis=0)


def _flat_planes(zs):
    """(N, 3) plane points at depths zs + matching up normals."""
    o = np.zeros((len(zs), 3))
    o[:, 2] = zs
    return o, np.broadcast_to(UP, (len(zs), 3)).copy()


def _flat_stack_analytic(p, q, zs, n):
    """Exact flat-parallel-stack solution via the common ray parameter:
    solve sum_k d_k * tan(theta_k(pr)) = R for pr = n_k sin(theta_k) (scipy
    brentq, f64), then crossings and optical path in closed form."""
    p, q, n = (np.asarray(a, np.float64) for a in (p, q, n))
    z = np.concatenate([[p[2]], np.asarray(zs, np.float64), [q[2]]])
    d = -np.diff(z)
    assert (d > 0).all()
    dr = q[:2] - p[:2]
    r = np.hypot(*dr)
    f = lambda pr: np.sum(d * pr / np.sqrt(n * n - pr * pr)) - r
    pr = brentq(f, 0.0, n.min() * (1.0 - 1e-14), xtol=1e-15, rtol=1e-15)
    tan = pr / np.sqrt(n * n - pr * pr)
    off = np.cumsum(d * tan)[:-1]
    uh = dr / r
    x = np.column_stack([p[0] + off * uh[0], p[1] + off * uh[1], zs])
    opl = float(np.sum(n * n * d / np.sqrt(n * n - pr * pr)))
    return x, opl


def _sweep_two_point(nn, seed=0):
    """Seeded single-interface sweep (test_refraction.py style)."""
    rng = np.random.default_rng(seed)
    h = np.where(rng.random(nn) < 0.5, rng.uniform(300, 3000, nn),
                 rng.uniform(14e3, 20e3, nn))
    depth = rng.uniform(10, 4000, nn)
    off = rng.uniform(-5000, 5000, (nn, 2))
    p = np.column_stack([np.zeros(nn), np.zeros(nn), h])
    q = np.column_stack([off[:, 0], off[:, 1], -depth])
    tilt = rng.uniform(0, np.deg2rad(20), nn)
    az = rng.uniform(0, 2 * np.pi, nn)
    nrm = np.column_stack([np.sin(tilt) * np.cos(az),
                           np.sin(tilt) * np.sin(az), np.cos(tilt)])
    pt = rng.uniform(-50, 50, (nn, 3))
    opp = (np.sum((p - pt) * nrm, -1) * np.sum((q - pt) * nrm, -1)) < 0
    return p, q, pt, nrm, opp


def _firn_stack(nn, batch, rng):
    """Airborne/stratospheric platforms over nn flat firn-like layers (tiny
    monotone index steps 1.0 -> 1.9, random depths 1-120 m) with q below."""
    h = np.where(rng.random(batch) < 0.5, rng.uniform(300, 3000, batch),
                 rng.uniform(14e3, 20e3, batch))
    p = np.column_stack([np.zeros(batch), np.zeros(batch), h])
    # Random depths over 1-119 m with a guaranteed minimum gap (>= ~0.4 m at
    # N=80): micron-thin gaps under a 20 km platform hit the documented
    # absolute-coordinate conditioning limit, and mean firn-layer spacing in
    # the motivating sweep is ~1.5 m.
    gaps = rng.uniform(0.3, 1.0, (batch, nn))
    depths = 1.0 + 118.0 * np.cumsum(gaps, axis=1) / gaps.sum(1)[:, None]
    o = np.zeros((nn, batch, 3))
    o[..., 2] = -depths.T
    nrm = np.broadcast_to(UP, (nn, batch, 3)).copy()
    steps = rng.uniform(0.2, 1.0, (batch, nn))
    n = 1.0 + 0.9 * np.concatenate(
        [np.zeros((batch, 1)), np.cumsum(steps, 1) / steps.sum(1)[:, None]],
        axis=1)
    q = np.column_stack([rng.uniform(-2000, 2000, (batch, 2)),
                         -(depths[:, -1] + rng.uniform(5, 400, batch))])
    return p, q, o, nrm, n.T


def _finite(res):
    return all(np.isfinite(np.asarray(v)).all() for v in res[:-2])


def test_n1_matches_two_point():
    """N=1 is the same problem as snell_crossing: crossing/angles/lengths
    agree to 1e-9 over a 512-case sweep (measured: 2.1e-11 m crossing,
    1.1e-13 rad angles); same-side lanes are masked by both."""
    with jax.enable_x64():
        p, q, pt, nrm, opp = _sweep_two_point(512)
        r2 = snell_crossing(p, q, pt, nrm, 1.0, N_ICE, xp=np)
        rj = joint_crossings(p, q, pt[None], nrm[None],
                             np.array([1.0, N_ICE]))
        assert _finite(rj)
        v = np.asarray(rj.valid)
        assert v[opp].all() and not v[~opp].any()
        dx = np.linalg.norm(np.asarray(rj.x)[0] - r2.x, axis=-1)
        assert dx[opp].max() < 1e-9
        for a, b in [(np.asarray(rj.theta1)[0], r2.theta1),
                     (np.asarray(rj.theta2)[0], r2.theta2)]:
            assert np.abs(a - b)[opp].max() < 1e-12
        assert np.abs(np.asarray(rj.s)[0] - r2.s1)[opp].max() < 1e-8
        assert np.abs(np.asarray(rj.s)[1] - r2.s2)[opp].max() < 1e-8


@pytest.mark.parametrize("zs,n,q", [
    ([0.0, -40.0], [1.0, 1.3, 1.78], [800.0, 300.0, -200.0]),
    ([0.0, -40.0, -120.0], [1.0, 1.25, 1.5, 1.9], [1200.0, -400.0, -350.0]),
])
def test_flat_stack_analytic_vs_sequential(zs, n, q):
    """Flat parallel stacks vs the exact ray-parameter solution: joint
    matches crossings/opl to tight tolerance while the sequential chain --
    the approximation D+ removes -- errs by meters in the crossings and
    meters in opl (measured: joint <= 1.3e-13 m x / 2.3e-13 m opl; chain
    43/120 m x, 4.5/18.8 m opl for N=2/N=3), with the chain's recomputed
    stationarity residual (0.17-0.20) exposing the approximation."""
    with jax.enable_x64():
        p = np.array([0.0, 0.0, 500.0])
        q = np.array(q)
        o, nrm = _flat_planes(zs)
        x_ref, opl_ref = _flat_stack_analytic(p, q, zs, n)

        rj = joint_crossings(p, q, o, nrm, np.array(n))
        rs = sequential_chain(p, q, o, nrm, np.array(n))
        assert bool(np.all(rj.valid)) and _finite(rj)
        ex_j = np.linalg.norm(np.asarray(rj.x) - x_ref, axis=-1).max()
        eo_j = abs(float(_opl(rj, n)) - opl_ref)
        ex_s = np.linalg.norm(np.asarray(rs.x) - x_ref, axis=-1).max()
        eo_s = abs(float(_opl(rs, n)) - opl_ref)
        print(f"\nflat N={len(zs)}: joint x/opl err {ex_j:.2e}/{eo_j:.2e} m; "
              f"sequential {ex_s:.2e}/{eo_s:.2e} m; "
              f"chain residual {float(rs.residual):.2e}, "
              f"joint residual {float(rj.residual):.2e}")
        assert ex_j < 1e-6 and eo_j < 1e-8
        # The chain's error is real (this is what D+ fixes) ...
        assert ex_s > 0.1 and eo_s > 1e-4
        # ... and the joint solve beats it by orders of magnitude.
        assert ex_j < 1e-3 * ex_s and eo_j < 1e-3 * eo_s
        # The recomputed chain residual exposes the chaining approximation.
        assert float(rs.residual) > 1e-4 > 1e5 * float(rj.residual)


@pytest.mark.parametrize("nsurf", [2, 3])
def test_vs_fermat_tilted_stack(nsurf):
    """Tilted planar stacks (the planes ARE the surfaces, so the local-plane
    model is exact): joint matches the referee to its floor (measured:
    <= 1.1e-6 m crossing, <= 9.1e-13 m opl) and the sequential chain's
    deviation is orders of magnitude larger (0.94/1.96 m opl at N=2/3)."""
    slopes = [(0.08, 0.03), (0.05, 0.01), (0.02, -0.04)][:nsurf]
    zoff = [0.0, -300.0, -520.0][:nsurf]
    n = np.array([1.0, 1.35, 1.6, 1.78][:nsurf + 1])
    fns = [lambda x, y, a=a, b=b, z0=z0: z0 + a * x + b * y
           for (a, b), z0 in zip(slopes, zoff)]
    o = np.array([[0.0, 0.0, z0] for z0 in zoff])
    nrm = np.array([[-a, -b, 1.0] for a, b in slopes])
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    p, q = np.array([0.0, 0.0, 2000.0]), np.array([1500.0, 400.0, -900.0])
    with jax.enable_x64():
        f = fermat_path(p, q, fns, n)
        rj = joint_crossings(p, q, o, nrm, n)
        rs = sequential_chain(p, q, o, nrm, n)
        assert bool(np.all(rj.valid))
        ex_j = np.linalg.norm(np.asarray(rj.x) - f.x, axis=-1).max()
        eo_j = abs(float(_opl(rj, n)) - f.opl)
        eo_s = abs(float(_opl(rs, n)) - f.opl)
        print(f"\ntilted N={nsurf}: joint-vs-referee x/opl "
              f"{ex_j:.2e}/{eo_j:.2e} m; sequential opl dev {eo_s:.2e} m")
        assert ex_j < 1e-4 and eo_j < 1e-7
        assert eo_s > 1e-4 and eo_j < 1e-2 * eo_s


def test_vs_fermat_rough_stack():
    """Gently rough stacks (sinusoids, A = 0.5/0.8 m, Lambda = 300/240 m):
    the joint solve on local TANGENT planes anchored at the referee crossings
    matches the referee to the anchoring floor (measured < 1e-12 m opl of
    2856 m -- the local-plane model is exact when anchored correctly, M15's
    result), while the sequential chain on the SAME planes keeps its
    chaining error (measured 15.9 m opl at these contrasts). With
    facet-scale anchors (25 m offset) both are dominated by the shared
    local-plane model error (measured: joint 7.3e-2 m opl, matching M15's
    ~0.078 m facet-scale figure) and the joint deviation stays at or below
    the chain's (14.9 m)."""
    k1, k2 = 2 * np.pi / 300.0, 2 * np.pi / 240.0
    a1, a2 = 0.5, 0.8
    f1 = lambda x, y: a1 * np.sin(k1 * x) + 0.0 * y
    f2 = lambda x, y: -60.0 + a2 * np.cos(k2 * x + 1.0) + 0.0 * y
    fns = [f1, f2]
    n = np.array([1.0, 1.35, 1.78])
    p, q = np.array([0.0, 0.0, 800.0]), np.array([1600.0, 200.0, -700.0])

    def tangent_planes(anchors_xy):
        o, ns = [], []
        for (ax, ay), fn, a, k, ph in zip(anchors_xy, fns, (a1, a2),
                                          (k1, k2), (0.0, 1.0)):
            o.append([ax, ay, float(fn(ax, ay))])
            slope = (a * k * np.cos(k * ax) if fn is f1
                     else -a * k * np.sin(k * ax + ph))
            v = np.array([-slope, 0.0, 1.0])
            ns.append(v / np.linalg.norm(v))
        return np.array(o), np.array(ns)

    with jax.enable_x64():
        f = fermat_path(p, q, fns, n)
        # (a) tangent planes anchored AT the referee crossings: local-plane
        # model error vanishes; only the chaining error remains for the chain.
        o, nrm = tangent_planes(f.x[:, :2])
        rj = joint_crossings(p, q, o, nrm, n)
        rs = sequential_chain(p, q, o, nrm, n)
        eo_j = abs(float(_opl(rj, n)) - f.opl)
        eo_s = abs(float(_opl(rs, n)) - f.opl)
        assert bool(np.all(rj.valid))
        assert eo_j < 1e-8
        assert eo_s > 1e-4 and eo_j < 1e-2 * eo_s
        # (b) facet-scale anchoring (tangent planes 25 m from the true
        # crossings, M16's situation): the shared local-plane model error
        # dominates; joint stays <= the chain's deviation (+ slack).
        o2, nrm2 = tangent_planes(f.x[:, :2] + np.array([25.0, 0.0]))
        rj2 = joint_crossings(p, q, o2, nrm2, n)
        rs2 = sequential_chain(p, q, o2, nrm2, n)
        eo_j2 = abs(float(_opl(rj2, n)) - f.opl)
        eo_s2 = abs(float(_opl(rs2, n)) - f.opl)
        print(f"\nrough stack: anchored-at-crossing joint/seq opl dev "
              f"{eo_j:.2e}/{eo_s:.2e} m; 25 m-offset-anchor {eo_j2:.2e}/"
              f"{eo_s2:.2e} m (referee opl {f.opl:.6f} m)")
        assert bool(np.all(rj2.valid))
        assert eo_j2 < 0.3                      # local-plane model scale
        assert eo_j2 < eo_s2 + 0.01


def test_reciprocity():
    """Swapping p <-> q with the plane stack and media reversed gives the
    same path: crossings (reversed) to < 1e-8 m, swapped angles, same opl."""
    slopes = [(0.08, 0.03), (0.05, 0.01), (0.02, -0.04)]
    zoff = [0.0, -300.0, -520.0]
    n = np.array([1.0, 1.35, 1.6, 1.78])
    o = np.array([[0.0, 0.0, z] for z in zoff])
    nrm = np.array([[-a, -b, 1.0] for a, b in slopes])
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    p, q = np.array([0.0, 0.0, 2000.0]), np.array([1500.0, 400.0, -900.0])
    with jax.enable_x64():
        dn = joint_crossings(p, q, o, nrm, n)
        up = joint_crossings(q, p, o[::-1].copy(), nrm[::-1].copy(),
                             n[::-1].copy())
        assert bool(np.all(dn.valid)) and bool(np.all(up.valid))
        assert np.linalg.norm(np.asarray(up.x)[::-1] - np.asarray(dn.x),
                              axis=-1).max() < 1e-8
        np.testing.assert_allclose(np.asarray(up.theta1)[::-1],
                                   np.asarray(dn.theta2), atol=1e-12)
        np.testing.assert_allclose(np.asarray(up.theta2)[::-1],
                                   np.asarray(dn.theta1), atol=1e-12)
        np.testing.assert_allclose(np.asarray(up.s)[::-1], np.asarray(dn.s),
                                   atol=1e-8)
        assert abs(float(_opl(up, n[::-1])) - float(_opl(dn, n))) < 1e-8


def test_masking_same_side_finite():
    """Shadow/same-side geometry masks invalid with finite values (never
    NaN), matching refraction.py's style: q above the deeper plane, q above
    both planes, and p on a plane."""
    o, nrm = _flat_planes([0.0, -100.0])
    n = np.array([1.0, 1.3, 1.78])
    cases = [
        (np.array([0.0, 0.0, 500.0]), np.array([400.0, 0.0, -50.0])),  # in-between q
        (np.array([0.0, 0.0, 500.0]), np.array([400.0, 0.0, 50.0])),   # q above all
        (np.array([0.0, 0.0, 0.0]), np.array([400.0, 0.0, -200.0])),   # p on plane
    ]
    with jax.enable_x64():
        for p, q in cases:
            r = joint_crossings(p, q, o, nrm, n)
            assert not bool(np.any(r.valid)), (p, q)
            assert _finite(r)
        # Control: the same stack with a proper q is valid.
        r = joint_crossings(cases[0][0], np.array([400.0, 0.0, -200.0]),
                            o, nrm, n)
        assert bool(np.all(r.valid))


def test_convergence_budget_sweep():
    """The fixed default budget (20 damped Newton steps) converges across
    seeded airborne/stratospheric sweeps: firn-like stacks N in {2, 5, 20,
    80} (tiny index steps), a high-contrast air -> ice -> bedrock-ish case,
    and a tilted low-N case. Doubling the budget moves no crossing by more
    than 1 mm (measured worst: exactly 0 -- converged lanes reject further
    steps) and every lane is valid (worst residual 1.3e-12)."""
    with jax.enable_x64():
        worst_dx, worst_res, worst_case = 0.0, 0.0, None
        rng = np.random.default_rng(7)
        cases = []
        for nn in (2, 5, 20, 80):
            cases.append((f"firn N={nn}", _firn_stack(nn, 256, rng)))
        # High-contrast: air over ice over a dense (n=3) half-space.
        b = 256
        p = np.column_stack([np.zeros(b), np.zeros(b),
                             rng.uniform(500, 3000, b)])
        o = np.zeros((2, b, 3))
        o[1, :, 2] = -1000.0
        nrm = np.broadcast_to(UP, (2, b, 3)).copy()
        q = np.column_stack([rng.uniform(-3000, 3000, (b, 2)),
                             -1000.0 - rng.uniform(50, 1000, b)])
        cases.append(("high contrast", (p, q, o, nrm,
                                        np.array([1.0, 1.78, 3.0]))))
        # Tilted two-layer case (guaranteed plane separation).
        tilt = rng.uniform(0, np.deg2rad(3), (2, b))
        az = rng.uniform(0, 2 * np.pi, (2, b))
        nrm_t = np.stack([np.sin(tilt) * np.cos(az),
                          np.sin(tilt) * np.sin(az), np.cos(tilt)], axis=-1)
        o_t = np.zeros((2, b, 3))
        o_t[1, :, 2] = -500.0
        q_t = np.column_stack([rng.uniform(-1500, 1500, (b, 2)),
                               -800.0 - rng.uniform(0, 700, b)])
        cases.append(("tilted N=2", (p, q_t, o_t, nrm_t,
                                     np.array([1.0, 1.35, 1.78]))))

        for name, (p, q, o, nrm, n) in cases:
            r1 = joint_crossings(p, q, o, nrm, n)
            r2 = joint_crossings(p, q, o, nrm, n, n_newton=40)
            assert bool(np.all(r1.valid)), name
            dx = np.linalg.norm(np.asarray(r1.x) - np.asarray(r2.x),
                                axis=-1).max()
            res = float(np.max(np.asarray(r1.residual)))
            if dx >= worst_dx:
                worst_dx, worst_case = float(dx), name
            worst_res = max(worst_res, res)
            assert dx < 1e-3, name
        print(f"\nbudget sweep: worst 20-vs-40-step crossing diff "
              f"{worst_dx:.2e} m ({worst_case}); worst residual "
              f"{worst_res:.2e}")


def test_compile_flat_in_n_and_runtime():
    """The lax.scan structure keeps the compiled graph O(1) in N: jit
    trace+compile at N=80 within a few x of N=10 (measured 0.29 s vs
    0.29 s), vs the sequential kernel chain's O(N^2) growth. Runtime
    measured ~0.41 ms/target at N=80, batch 4096 (f64, CPU) -- ~5 us per
    target-interface with the full 20 x 10 damped-Newton budget."""
    with jax.enable_x64():
        rng = np.random.default_rng(11)
        tc = {}
        compiled = {}
        args80 = None
        for nn in (10, 80):
            p, q, o, nrm, n = _firn_stack(nn, 64, rng)
            args = tuple(jnp.asarray(a) for a in (p, q, o, nrm, n))
            t0 = time.perf_counter()
            compiled[nn] = jax.jit(joint_crossings).lower(*args).compile()
            tc[nn] = time.perf_counter() - t0
        assert tc[80] < 4.0 * tc[10] + 2.0, tc
        # Runtime per target at N=80.
        bsz = 4096
        p, q, o, nrm, n = _firn_stack(80, bsz, np.random.default_rng(12))
        args = tuple(jnp.asarray(a) for a in (p, q, o, nrm, n))
        fn = jax.jit(joint_crossings)
        r = jax.block_until_ready(fn(*args))            # compile + warm
        assert bool(np.asarray(r.valid).all())
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            best = min(best, time.perf_counter() - t0)
        per = best / bsz
        print(f"\ncompile: N=10 {tc[10]:.2f} s, N=80 {tc[80]:.2f} s "
              f"(ratio {tc[80] / tc[10]:.2f}); runtime {per * 1e6:.1f} "
              f"us/target (N=80, batch {bsz}, f64, jitted CPU)")
        assert per < 1e-3
