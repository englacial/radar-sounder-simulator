"""Sequential-vs-joint refraction regression for the multilayer kernel (D+).

The joint path (kernels/multilayer.py ``refraction="joint"``, solver
``refraction_joint.joint_crossings``) must (1) agree tightly with the
sequential chain on firn-like low-contrast stacks -- where the chaining error
is second-order in the index steps and the chain is the validated reference,
(2) differ in a characterized way on high-contrast rough interfaces -- where
its delays must sit CLOSER to the multi-interface Fermat referee than the
chain's (the error D+ removes), (3) be exactly invariant to the no-op
interface padding that lets target layers share compiled executables, and
(4) be converged at the kernel's fixed pass-2 Newton budget (a quadrupled
budget must not move the fields).

Thresholds were set from the first run (repo convention); measured then
(x86-64 CPU, f64 kernel path):

- firn-like (eps 1.0/1.5/1.55/1.6/1.65, 3 crossings, rough target):
  max per-bin |dE|/peak 1.0e-2 (bin-crossing spikes: mm-scale opl changes
  move whole contributions across fast-time bins), window-integrated field
  delta 1.3e-3, total power delta 1.3e-3; halving the sub-surface steps
  shrinks the per-bin delta to 1.9e-3 (0.19x; the power/total deltas are
  cancellation-dominated aggregates and are not cleanly monotone).
- high contrast (eps 1.0/2.2/4.5, rough mid interface + rough bed):
  max per-bin |dE|/peak 8.8e-2, total power delta 8.7e-2 (recorded, loose
  gates 0.3/0.25). Referee spot-check on TILTED PLANAR interfaces (facet
  planes are exact there, isolating the chaining error) at 30-40 deg
  incidence: kernel-anchored joint |opl err| <= 6.4e-7 m (the pass-2
  residual tolerance at this scale) vs sequential 0.40-0.46 m on every
  facet. On the ROUGH near-nadir stack both paths are instead dominated by
  the SHARED facet-plane anchoring error (measured max 8.4e-3 m joint vs
  1.6e-2 m sequential, not per-facet ordered) -- the M15 local-plane error,
  which the joint solve keeps by design.
- padding: bitwise identical (0.0) for pad_to in {4, 8} vs unpadded.
- budget: doubled pass-2 budget moves the field by 0.0 (bitwise) on the
  high-contrast case.
"""

import numpy as np

import soundersim
from soundersim import synthetic as syn
from soundersim.compare.brute_force_layered import surface_facets
from soundersim.compare.fermat import fermat_path
from soundersim.config import (DemInterface, FacetConfig, Medium,
                               OffsetInterface, RadarConfig, SimConfig)
from soundersim.kernels.multilayer import _grid_consts, refracted_cluttergram
from soundersim.physics import C, fresnel_normal
from soundersim.refraction import snell_crossing
from soundersim.refraction_joint import (joint_crossings,
                                         sequential_chain)

F0 = 195e6
K0 = 2.0 * np.pi * F0 / C
P = np.array([[0.0, 0.0, 500.0], [20.0, 0.0, 500.0]])
UCT = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])
FLAT = lambda x, y: 0.0 * x  # noqa: E731


def _firn_case(scale=1.0):
    """Flat surface + 2 flat firn layers over a gently rough target at
    -40 m; index steps below the surface scale with ``scale``."""
    eps = [1.0, 1.5, 1.5 + 0.05 * scale, 1.5 + 0.10 * scale,
           1.5 + 0.15 * scale]
    crossed = [surface_facets(80.0, 4.0, FLAT),
               surface_facets(80.0, 4.0, FLAT, z0=-8.0),
               surface_facets(80.0, 4.0, FLAT, z0=-17.0)]
    tgt = surface_facets(80.0, 4.0,
                         lambda x, y: 0.1 * np.sin(2 * np.pi * y / 30.0),
                         z0=-40.0)
    kw = dict(mode="coherent", t0=3.3e-6, dt=1e-8, n_samples=80, c=C,
              gamma=fresnel_normal(eps[3], eps[4]), k0=K0)
    return tgt, crossed, eps[:4], kw


MID_FN = lambda x, y: -20.0 + 0.3 * np.cos(2 * np.pi * x / 40.0)  # noqa: E731
BED_FN = lambda x, y: -60.0 + 0.5 * np.sin(2 * np.pi * y / 30.0)  # noqa: E731
EPS_HC = [1.0, 2.2, 4.5]


def _highcontrast_case():
    """Rough mid interface + rough bed under a flat surface, strong
    contrasts (air / dense firn-like / bedrock-like)."""
    crossed = [surface_facets(80.0, 4.0, FLAT),
               surface_facets(80.0, 4.0, lambda x, y: MID_FN(x, y) + 20.0,
                              z0=-20.0)]
    tgt = surface_facets(80.0, 4.0, lambda x, y: BED_FN(x, y) + 60.0,
                         z0=-60.0)
    kw = dict(mode="coherent", t0=3.3e-6, dt=1e-8, n_samples=100, c=C,
              gamma=fresnel_normal(EPS_HC[1], EPS_HC[2]), k0=K0)
    return tgt, crossed, EPS_HC, kw


def _both(tgt, crossed, eps, kw, **joint_kw):
    att = [0.0] * len(eps)
    seq, ds = refracted_cluttergram(P, UCT, tgt, crossed, eps, att, **kw)
    jnt, dj = refracted_cluttergram(P, UCT, tgt, crossed, eps, att,
                                    refraction="joint", **joint_kw, **kw)
    return seq, jnt, ds, dj


def _deltas(seq, jnt):
    peak = np.abs(seq).max()
    dfield = float(np.abs(jnt - seq).max() / peak)
    ps, pj = (np.abs(seq) ** 2).sum(), (np.abs(jnt) ** 2).sum()
    dtot = float(abs(jnt.sum() - seq.sum()) / abs(seq.sum()))
    return dfield, float(abs(pj - ps) / ps), dtot


def test_firn_like_agreement_second_order():
    """Low-contrast stacks: joint and sequential agree tightly (per-bin,
    window-integrated field, and total power -- module docstring for the
    measured values), and the per-bin disagreement shrinks at least linearly
    with the sub-surface index steps (the chaining error the joint solve
    removes is second-order small; the per-bin delta also carries
    bin-crossing quantization, so quadratic is not asserted)."""
    d1 = _deltas(*_both(*_firn_case(1.0))[:2])
    d0 = _deltas(*_both(*_firn_case(0.5))[:2])
    print(f"\nfirn-like dE/peak, dP/P, dTot: x1 {d1[0]:.2e}/{d1[1]:.2e}/"
          f"{d1[2]:.2e}; x0.5 {d0[0]:.2e}/{d0[1]:.2e}/{d0[2]:.2e}")
    # measured 1.0e-2 / 1.3e-3 / 6.5e-4
    assert d1[0] < 2.5e-2 and d1[1] < 5e-3 and d1[2] < 3e-3
    # halved steps -> at most ~half the per-bin delta (measured 0.19x)
    assert d0[0] < 0.6 * d1[0]


def test_high_contrast_characterized_and_droppless():
    """Strong contrasts + rough interfaces: the paths genuinely differ
    (that's the physics fix, recorded loosely) but the joint run stays
    fully valid (no dropped power) and energy-comparable."""
    seq, jnt, ds, dj = _both(*_highcontrast_case())
    dfield, dpow, _ = _deltas(seq, jnt)
    print(f"\nhigh contrast dE/peak {dfield:.2e}, dP/P {dpow:.2e}")
    assert float(np.abs(dj).max()) == 0.0 and float(np.abs(ds).max()) == 0.0
    assert dfield < 0.3 and dpow < 0.25           # measured 8.8e-2 / 8.7e-2
    assert dfield > 1e-3                           # the paths DO differ here


def _kernel_anchored_opl(p, q, crossed, n, joint):
    """f64 replica of the kernel's two-pass geometry for one target: pass 1
    against the mean planes, facet lookup via the affine fit, pass 2 against
    the anchored local facet planes; returns the optical path length."""
    consts = [_grid_consts(f) for f in crossed]

    def lookup(cst, x):
        gc, gn, coef, _, _ = cst
        rc = x[:2] @ coef[:2] + coef[2]
        r = int(np.clip(round(rc[0]), 0, gc.shape[0] - 1))
        c_ = int(np.clip(round(rc[1]), 0, gc.shape[1] - 1))
        nr = gn[r, c_].astype(np.float64)
        return gc[r, c_].astype(np.float64), nr / np.linalg.norm(nr)

    if joint:
        mp = np.array([c[3] for c in consts])
        mn = np.array([c[4] for c in consts])
        r1 = sequential_chain(p, q, mp, mn, n, n_iter=10)
        pts, nrs = zip(*(lookup(c, np.asarray(r1.x)[i])
                         for i, c in enumerate(consts)))
        r2 = joint_crossings(p, q, np.array(pts), np.array(nrs), n,
                             n_newton=6, n_backtrack=4)
        assert bool(np.all(r2.valid))
        return float(np.sum(n * np.asarray(r2.s)))
    cur, opl = p, 0.0
    for i, cst in enumerate(consts):
        r1 = snell_crossing(cur, q, cst[3], cst[4], n[i], n[i + 1],
                            n_iter=10, xp=np)
        pt, nr = lookup(cst, np.asarray(r1.x))
        r2 = snell_crossing(cur, q, pt, nr, n[i], n[i + 1], xp=np)
        assert bool(r2.valid)
        opl += n[i] * float(r2.s1)
        cur = np.asarray(r2.x)
    return opl + n[-1] * float(np.linalg.norm(cur - q))


def test_joint_delays_closer_to_fermat_referee():
    """Spot-check kernel-anchored path delays against the multi-interface
    Fermat referee on the true surfaces. TILTED PLANAR interfaces at
    30-40 deg incidence: the facet planes lie exactly on the interfaces, so
    the shared local-plane anchoring error vanishes and the comparison
    isolates the chaining error -- the joint geometry must beat the
    sequential chain on every checked facet by >> 10x (measured <= 6.6e-10 m
    vs 0.056-0.35 m). On rough interfaces near nadir both paths are instead
    dominated by the shared anchoring error (module docstring), which the
    joint solve keeps by design -- recorded, not gated."""
    import jax

    surf_fn = lambda x, y: 0.04 * x - 0.02 * y                  # noqa: E731
    mid_fn = lambda x, y: -25.0 - 0.06 * x + 0.03 * y           # noqa: E731
    crossed = [surface_facets(80.0, 4.0, surf_fn),
               surface_facets(80.0, 4.0, lambda x, y: mid_fn(x, y) + 25.0,
                              z0=-25.0)]
    tgt = surface_facets(80.0, 4.0, lambda x, y: BED_FN(x, y) + 60.0,
                         z0=-60.0)
    p = np.array([-350.0, 120.0, 500.0])
    n = np.sqrt(np.asarray(EPS_HC, np.float64))
    rng = np.random.default_rng(3)
    qs = tgt.centers[rng.choice(len(tgt.centers), 6, replace=False)]
    errs = {"seq": [], "joint": []}
    with jax.enable_x64():
        for q in qs.astype(np.float64):
            ref = fermat_path(p, q, [surf_fn, mid_fn], n).opl
            errs["seq"].append(abs(_kernel_anchored_opl(
                p, q, crossed, n, joint=False) - ref))
            errs["joint"].append(abs(_kernel_anchored_opl(
                p, q, crossed, n, joint=True) - ref))
    es, ej = np.array(errs["seq"]), np.array(errs["joint"])
    print(f"\nopl err vs Fermat referee (tilted planes): joint max "
          f"{ej.max():.2e} m, sequential {es.min():.2e}-{es.max():.2e} m")
    assert (ej < 0.1 * es).all()
    assert ej.max() < 1e-6                        # measured 6.6e-10 m


def test_padding_is_exact_noop():
    """pad_to > n_crossed (the compile-sharing construction) must not change
    the output: the index-matched horizontal pads are exact pass-throughs
    and their air sub-segments book with the surface incidence cosine."""
    tgt, crossed, eps, kw = _firn_case()
    att = [0.0] * len(eps)
    base, _ = refracted_cluttergram(P, UCT, tgt, crossed, eps, att,
                                    refraction="joint", **kw)
    for pad in (4, 8):
        padded, _ = refracted_cluttergram(P, UCT, tgt, crossed, eps, att,
                                          refraction="joint", pad_to=pad, **kw)
        d = float(np.abs(padded - base).max() / np.abs(base).max())
        assert d < 1e-6, (pad, d)                 # measured exactly 0.0


def test_joint_budget_converged():
    """Quadrupling the pass-2 Newton budget must not move the fields: the
    fixed kernel budget (6 steps, 4 halvings) is converged even on the
    high-contrast rough case (the chain init is close and converged lanes
    reject further steps; measured bitwise-equal)."""
    tgt, crossed, eps, kw = _highcontrast_case()
    att = [0.0] * len(eps)
    base, _ = refracted_cluttergram(P, UCT, tgt, crossed, eps, att,
                                    refraction="joint", **kw)
    dbl, _ = refracted_cluttergram(P, UCT, tgt, crossed, eps, att,
                                   refraction="joint", joint_newton=24,
                                   joint_backtrack=10, **kw)
    d = float(np.abs(dbl - base).max() / np.abs(base).max())
    print(f"\ndoubled-budget field delta/peak: {d:.2e}")
    assert d < 1e-9


def test_simulate_both_paths_end_to_end():
    """simulate() honors SimConfig.refraction: an offset firn stack runs on
    both paths, stays finite, keeps the exact nadir delay bins, and the two
    paths agree to the firn-like tolerance."""
    scene = syn.offset_stack_scene(surface=500.0, spacings=(3.0, 4.0),
                                   extent=800.0, n_traces=2, altitude=500.0)
    eps = [1.0, 1.5, 1.6, 1.7]
    media = [Medium(name=f"m{i}", eps_r=e) for i, e in enumerate(eps)]
    ifaces = [DemInterface(name="surface"),
              OffsetInterface(name="l1", reference="surface", offset=-3.0),
              OffsetInterface(name="l2", reference="surface", offset=-7.0)]
    out = {}
    for refr in ("joint", "sequential"):
        cfg = SimConfig(mode="coherent", refraction=refr,
                        radar=RadarConfig(dt=5e-9, n_samples=800, t0=2.5e-6,
                                          f0=F0),
                        facets=FacetConfig(spacing=10.0), media=media,
                        interfaces=ifaces)
        ds = soundersim.simulate(scene, cfg)
        assert np.isfinite(ds.field.values).all()
        # deepest layer's nadir delay lands in the exact chained bin
        expected = float(ds.nadir_twtt.sel(layer="l2")[0])
        trace = np.abs(ds.field.sel(layer="l2")[0].values) ** 2
        assert np.nonzero(trace)[0][0] == int(np.floor((expected - 2.5e-6)
                                                       / 5e-9))
        out[refr] = ds.field.values
    peak = np.abs(out["sequential"]).max()
    d = float(np.abs(out["joint"] - out["sequential"]).max() / peak)
    print(f"\nsimulate() joint-vs-sequential dE/peak: {d:.2e}")
    assert d < 0.02
