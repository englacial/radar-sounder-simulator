"""Refracted-path multilayer kernel: both modes through one geometry path.

For a target interface j (facets in medium j, reflecting off medium j+1) the
ray from the platform refracts through interfaces 0..j-1. Two refraction
solvers share all the radiometry below: the SEQUENTIAL chain (stage 3, the
original path, described next) and the JOINT solve (D+,
``refraction_joint.joint_crossings``; see "Joint path" below). Per (trace,
target facet) the sequential chain is solved interface by interface
(top-down):

1. Two-point Snell solve (``refraction.snell_crossing``) from the current
   point to the TARGET facet center against the crossed interface's mean
   plane, with the index pair (n_i, n_i+1).
2. The crossing estimate selects the nearest facet of that interface (affine
   grid-index fit), and the solve is repeated against that facet's local
   tangent plane -- facet-scale anchoring, the M15-measured regime where the
   local-plane optical-path error is <= ~0.1 m for meter-scale roughness.
3. The crossing becomes the current point for the next interface.

Sequential approximation (documented per plan D3-1): each two-point solve
treats the remaining stack below interface i as a single medium n_i+1, so for
MORE than one crossed interface the polyline is not the exact multi-interface
Fermat path (exact chaining is out of scope). It is exact for one crossing
(the surface+bed case), exact for any number of zero-contrast crossings, and
its error vanishes with layer contrast -- the firn-stack case (many closely
spaced, low-contrast layers) is exactly where it is accurate. Because every
step is a two-point boundary solve, genuine TIR cannot arise (see
refraction.py); the dropped channel accounts for same-side/shadowed geometry
(e.g. a bed facet locally above the surface plane) and non-converged solves.

Per contribution, with per-leg geometric lengths s_i, leg cosines c_i (leg
i < j: incidence angle at interface i; final leg: refraction angle at
interface j-1 -- identical for parallel planes), n_i = sqrt(eps_i), and the
``physics.refraction_spreading`` lengths L_par/L_perp:

- delay: twtt = 2 * sum_i n_i s_i / c  (optical path, float64)
- coherent field: (j k_j / 2pi) * gamma_j * cos(theta_t) * A * sinc * sinc
    * prod_i (1 - gamma_i(theta)^2)              two-way FIELD transmission
    * (n_0 c_0)/(n_j c_j) / (L_par * L_perp)     two-way refracted spreading
    * 10**(-loss_db_oneway/10)                   two-way field attenuation
    * exp(-2j k_0 * opl)
  with k_j = k_0 n_j in the sinc arguments and theta_t the incidence at the
  target facet from the REFRACTED arrival direction. gamma_j is the target
  interface's normal-incidence coefficient (the stage-2 convention -- see
  simulate.py for the rationale); crossings use angle-dependent TE Fresnel.
- incoherent power: (A cos(theta_t) * transmission * spreading)**2 *
  10**(-2*loss_db_oneway/10)  -- the same factors in the power domain, no
  reflectivity at the target (stage-1/simc convention).

Precision: the solve, path lengths and phase run in float64 INSIDE the jitted
per-trace path via a scoped ``jax.enable_x64()`` (the M15 recommendation --
f64 legs from the crossing point; facet coordinates are stored float32 like
the other kernels). The fixed-size block/scan/vmap structure matches
incoherent.py/coherent.py; dropped power per trace accumulates out-of-window
AND invalid-path (shadow/non-converged) contributions.

Antenna pattern (M22): a non-isotropic pattern weights each contribution by
the two-way antenna gain evaluated at the DEPARTURE direction of the AIR leg
(platform -> first crossing point): field *= g**2 (coherent) / power *= g**4
(incoherent), g the one-way FIELD gain (antenna.py). Pattern kind is static
in the factory key; pattern vector/params are traced.

Joint path (``refraction="joint"``): the whole crossed stack is solved at
once by the block-tridiagonal damped Newton of ``refraction_joint`` -- the
true stationary path of the anchored local planes, removing the sequential
chaining approximation. Two passes mirror the sequential design: pass 1
anchors the facet lookup from the SAME mean-plane chain the sequential
kernel uses (``refraction_joint.sequential_chain``, 10 iterations -- the
lookup only needs sub-facet accuracy, and identical anchors mean the two
paths differ only by the chaining approximation itself), the per-interface
facet lookup (same affine fit, vmapped over the interface axis) selects the
local tangent planes, and pass 2 runs ``joint_crossings`` against those.
Because both passes and the lookup run over a leading interface axis
(``lax.scan``/``vmap`` inside ``refraction_joint``), the compiled graph is
O(1) in the number of crossed interfaces -- vs the sequential chain's
unrolled O(N^2) graph and its per-layer-count recompile. To share one
executable across target layers, the per-call stack is PADDED to ``pad_to``
interfaces with index-matched no-op planes: horizontal planes in the
platform-to-surface air gap with the air index on both sides (gamma = 0
exactly, straight pass-through), whose air sub-segments are booked with the
SURFACE incidence cosine so the split air leg accumulates spreading exactly
like the unsplit one (simulate() pads to power-of-two buckets: ~log2(N)
executables serve all N target layers, measured bitwise-identical to the
unpadded call). Validity comes from the joint solve (non-converged /
same-side / evanescent lanes -> the dropped channel, like the sequential
path's invalid solves). Newton budgets are FIXED statics:
``joint_newton=6`` damped steps with ``joint_backtrack=4`` halvings --
measured bitwise-equal to a (24, 10) budget on firn-like AND 45-deg
high-contrast rough cases (the chain init is that close; quadratic Newton
needs <= 4 accepted steps), regression-gated by the doubled-budget test in
tests/test_multilayer_joint.py. Runtime: the joint path costs ~5-10x the
sequential chain per crossing (fixed damped-Newton budgets vs the cheap 1-D
sigma iteration); its win is COMPILE time -- seconds, flat in N, vs the
chain's O(N^2) minutes -- so deep stacks win on first call and lose on
cached repeats (measured numbers: the ``refraction_joint`` report case).

Sub-facet roughness (Gerekos et al. 2023, roughness.py, docs/roughness.md;
coherent mode only): ``roughness=(sigma_m, corr_length_m, phasors,
n_terms)`` applies to the TARGET reflection exactly as in coherent.py, but
with the LOCAL-medium wavenumber and refracted arrival direction: K = 2 k_j
cos(theta_t), in-plane coefficients A0 = 2 k_j (rhat.e1)/|e1| etc., and the
incoherent term sqrt(D_Phi)*phi_r carries the full non-phase factor
(kj/2pi)*gamma*cos_t*spread*att_f -- transmission, refracted spreading,
attenuation and antenna weighting identical to the coherent term.
``crossed_sigma`` (per crossed interface) adds the TRANSMISSION coherent
attenuation: each crossing multiplies the two-way field by
exp(-2 sigma_i^2 K_t^2), K_t = k0 (n_i cos(theta1) - n_i+1 cos(theta2)) --
the down- and up-going rays cross the SAME interface point in the monostatic
geometry, so their phase perturbations add coherently ((2 K_t)^2 sigma^2 / 2
in the exponent, i.e. the one-way exp(-sigma^2 K_t^2 / 2) to the 4th power).
The incoherent (diffusely re-radiated) transmission term is NOT modeled --
for the low-contrast firn case K_t is ~5 percent of the reflection K, making
the whole crossing effect near-negligible there. Smooth defaults trace
exactly the pre-roughness program; sigma = 0 is bit-identical to it.

Diffuse bed channel (``diffuse=(amp, phasors, n_exp)``, coherent mode only):
an INCOHERENT per-facet return that runs alongside the specular one, for
reflectivity models that split the target into a mirror-like part and a
diffusely scattering part (a facet's specular lobe only points back when the
facet is nearly normal to the ray; the rest of the reflected power leaves
the specular direction and is modelled as a cos^n law). Per facet

    a_diff = sqrt(A/(2 pi)) * amp * cos_t^(1 + n_exp/2) * spread * att_f

with a frozen unit random phasor, added to the same complex accumulator as
the specular amplitude. ``amp`` is a per-facet FIELD amplitude on the same
scale as ``gamma``.

NORMALIZATION (derived, unit-tested in tests/test_multilayer_diffuse.py).
The prefactor sqrt(A/(2 pi)) is fixed by requiring that splitting a target's
power reflectivity as Gamma^2 -> f_s Gamma^2 (specular, field * sqrt(f_s))
plus (1-f_s) Gamma^2 (diffuse, amp = sqrt(1-f_s) Gamma) conserves the TOTAL
returned power at nadir over a flat interface. The specular coherent sum is
the image-method mirror field: integrating (k/2pi) Gamma cos_t spread over a
flat plane at range r (spread = 1/r^2, stationary phase) gives |E|^2 =
Gamma^2/(4 r^2). The incoherent sum of the diffuse channel over the same
plane is sum_i a_i^2 = (1/(2 pi)) amp^2 Int (cos_t spread)^2 dA, and
Int (cos_t/r'^2)^2 dA = pi/(2 r^2) exactly, so sum_i a_i^2 = amp^2/(4 r^2).
With amp^2 = (1-f_s) Gamma^2 the two channels sum to Gamma^2/(4 r^2) for any
f_s, independently of facet size, range and wavenumber (the k/2pi prefactor
cancels -- the diffuse channel is frequency-flat by construction, as a
sigma^0 law should be). The extra cos^n_exp factor is a SHAPE on top,
normalized to 1 at nadir, so n_exp > 0 books slightly less than (1-f_s) of
the power away from nadir (recorded, not compensated).

Compilation caching: the jitted callable is built once per static
configuration -- ``(mode, split_sides, n_samples, n_crossed, pattern,
refraction, joint budgets, roughness statics, diffuse, grazing-fix
statics)``, memoized via
``functools.lru_cache`` -- with
every run-varying number (t0/dt/c/gamma/k0, per-leg eps/index/attenuation,
interface lookup constants, facet blocks, positions) passed as traced
arguments, so repeat calls (and calls that change only numeric values, e.g.
layer depths) hit jax's shape-based jit cache instead of recompiling. The
joint path drops ``n_crossed`` from the key (the padded interface-axis LENGTH
drives jax's shape cache instead; the pad count itself is traced). Tracing
happens on first call, inside the caller's ``jax.enable_x64()`` scope, which
keeps the f64 path f64.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ..antenna import gain_fn
from ..physics import fresnel_te
from ..refraction_joint import joint_crossings, sequential_chain
from ..roughness import d_phi, mean_attenuation
from .geometry import (along_track_order, auto_block_size, block_windows,
                       twtt_bin, window_reach_m)

TWO_PI = 2.0 * np.pi
_C_MIN = 1e-9  # grazing-cosine clamp: keeps 1/c^2 finite; amplitudes -> 0


# ---------------------------------------------------------------------------
# Component-form vector math (2026-08-24 runtime work, "1b"). The sequential
# path keeps every 3-vector as a tuple of (..., ) arrays instead of a
# (..., 3) array: XLA:CPU then fuses the whole per-facet chain into a few
# loops instead of breaking it at every ``sum(axis=-1)`` reduction, which
# materialised hundreds of (traces, block) float64 temporaries per scan step
# (the kernel was DRAM-bandwidth-bound). Dot products sum in the same
# ((x + y) + z) order as the reduce they replace.
# ---------------------------------------------------------------------------
def _dot3(a, b):
    return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]


def _c3(v):
    """(..., 3) array -> component tuple."""
    return (v[..., 0], v[..., 1], v[..., 2])


def _snell_c(p, q, o, nrm, n1, n2, n_iter):
    """``refraction.snell_crossing`` in component form (same operations, same
    order; see that docstring). p/q/o/nrm are 3-tuples of broadcastable
    arrays, ``nrm`` unit length. Returns (x (3-tuple), theta1, theta2, s1,
    s2, residual, valid)."""
    dt = jnp.result_type(*q)
    eps = jnp.finfo(dt).eps
    tiny = jnp.asarray(1e-30, dt)
    tol = 1e-9 if eps < 1e-10 else 1e-3
    h1 = _dot3((p[0] - o[0], p[1] - o[1], p[2] - o[2]), nrm)
    h2 = _dot3((q[0] - o[0], q[1] - o[1], q[2] - o[2]), nrm)
    a, b = jnp.abs(h1), jnp.abs(h2)
    fp = (p[0] - h1 * nrm[0], p[1] - h1 * nrm[1], p[2] - h1 * nrm[2])
    fq = (q[0] - h2 * nrm[0], q[1] - h2 * nrm[1], q[2] - h2 * nrm[2])
    w = (fq[0] - fp[0], fq[1] - fp[1], fq[2] - fp[2])
    L = jnp.sqrt(_dot3(w, w))
    Lm = jnp.maximum(L, tiny)
    u = (w[0] / Lm, w[1] / Lm, w[2] / Lm)
    s = jnp.maximum(a + b + L, tiny)
    inv_s = 1.0 / s                     # one divide; cheap multiplies fuse
    ah, bh, Lh = a * inv_s, b * inv_s, L * inv_s
    swap = n1 > n2
    ratio = jnp.minimum(n1, n2) / jnp.maximum(n1, n2)
    ar = jnp.where(swap, bh, ah)
    bd = jnp.where(swap, ah, bh)
    one = jnp.asarray(1.0, dt)
    hi0 = one - 8.0 * eps
    sig = jnp.minimum(
        Lh / jnp.maximum(jnp.sqrt(Lh * Lh + (ah + bh) ** 2), tiny), hi0)
    lo = jnp.zeros_like(Lh)
    hi = hi0 * jnp.ones_like(Lh)
    # XLA:CPU fuses an expensive op (sqrt/rsqrt/divide/exp/...) into its
    # consumer only when it has exactly ONE consumer; every shared sqrt or
    # divide otherwise materialises a (traces, block) float64 temporary. The
    # iteration below is the same Newton step written so that each expensive
    # op is used once (rsqrt for F, sqrt + divide for F'; F' = ar c1^-3/2 +
    # bd ratio c2^-3/2) -- one loop fusion per iteration instead of ~7.
    # The iterations run as a real loop (fori_loop, carry = sig/lo/hi): an
    # unrolled chain lets XLA duplicate the cheap ops across iterations
    # (measured 8x slower) -- the loop body is one fusion, the carry the only
    # per-iteration materialisation.
    def newton(_, carry):
        sig, lo, hi = carry
        v = ratio * sig
        c1 = jnp.maximum((one - sig) * (one + sig), tiny)
        c2 = jnp.maximum((one - v) * (one + v), tiny)
        F = (ar * sig * jax.lax.rsqrt(c1) + bd * v * jax.lax.rsqrt(c2)
             - Lh)
        lo = jnp.where(F <= 0, sig, lo)
        hi = jnp.where(F <= 0, hi, sig)
        Fp = (ar / (c1 * jnp.sqrt(c1))
              + (bd * ratio) / (c2 * jnp.sqrt(c2)))
        sn = sig - F / jnp.maximum(Fp, tiny)
        ok = (sn >= lo) & (sn <= hi) & jnp.isfinite(sn)
        return jnp.where(ok, sn, 0.5 * (lo + hi)), lo, hi

    sig, lo, hi = jax.lax.fori_loop(0, n_iter, newton, (sig, lo, hi))
    c1 = jnp.maximum((one - sig) * (one + sig), tiny)
    tr = ar * sig * jax.lax.rsqrt(c1)
    t = jnp.clip(jnp.where(swap, Lh - tr, tr), 0.0, Lh)
    d2 = Lh - t
    r1 = jnp.maximum(jnp.sqrt(ah * ah + t * t), tiny)
    r2 = jnp.maximum(jnp.sqrt(bh * bh + d2 * d2), tiny)
    residual = n1 * t / r1 - n2 * d2 / r2
    s1, s2 = s * r1, s * r2
    st = s * t
    x = (fp[0] + st * u[0], fp[1] + st * u[1], fp[2] + st * u[2])
    theta1 = jnp.arctan2(t, ah)
    theta2 = jnp.arctan2(d2, bh)
    valid = (h1 * h2 < 0) & (jnp.abs(residual) <= tol)
    return x, theta1, theta2, s1, s2, residual, valid


def _grid_consts(facets):
    """Per-interface lookup constants: gridded centers/normals (float32), an
    affine (x, y, 1) -> (row, col) cell-index fit, and the area-weighted mean
    plane (float64 NumPy -- traced as f64 constants under the x64 scope)."""
    ny, nx = facets.grid_shape
    grid_c = facets.centers.reshape(ny, nx, 3).astype(np.float32)
    grid_n = facets.normals.reshape(ny, nx, 3).astype(np.float32)
    a = np.column_stack([facets.centers[:, 0], facets.centers[:, 1],
                         np.ones(len(facets.centers))])
    coef, *_ = np.linalg.lstsq(a, facets.cell.astype(np.float64), rcond=None)
    w = (facets.areas / facets.areas.sum())[:, None]
    mp = (facets.centers * w).sum(0)
    mn = (facets.normals * w).sum(0)
    mn = mn / np.linalg.norm(mn)
    return grid_c, grid_n, coef, mp, mn


def _joint_consts(crossed, pad_to, z_platform):
    """Stacked per-interface lookup constants for the joint path.

    Real interfaces (top-down) are padded at the TOP to ``pad_to`` entries
    with index-matched no-op planes: horizontal planes spread through the
    platform-to-surface air gap (below ``z_platform``, above the highest
    surface facet). Grids of differing shapes are stacked into the max shape;
    ``shp`` carries each interface's true (ny, nx) so lookups clip inside the
    valid region (padded cells are never gathered). No-op entries get a 1x1
    "grid" holding the plane point + up normal and a zero affine fit (every
    lookup lands on it). Returns (gc, gn, coef, shp, mp, mn, k_pad).
    """
    per = [_grid_consts(f) for f in crossed]
    k = pad_to - len(crossed)
    if k < 0:
        raise ValueError(f"pad_to={pad_to} < {len(crossed)} crossed interfaces")
    ny = max(g[0].shape[0] for g in per)
    nx = max(g[0].shape[1] for g in per)
    gc = np.zeros((pad_to, ny, nx, 3), np.float32)
    gn = np.zeros((pad_to, ny, nx, 3), np.float32)
    gn[..., 2] = 1.0
    coef = np.zeros((pad_to, 3, 2))
    shp = np.ones((pad_to, 2), np.int32)
    mp = np.zeros((pad_to, 3))
    mn = np.zeros((pad_to, 3))
    mn[:, 2] = 1.0
    if k:
        z_surf = float(crossed[0].centers[:, 2].max())
        gap = float(z_platform) - z_surf
        if gap <= 0:
            raise ValueError(
                "joint refraction padding needs the platform above the "
                f"surface (gap {gap:.1f} m)")
        zs = z_surf + gap * (np.arange(k, 0, -1) / (k + 1.0))  # top-down
        gc[:k, :, :, 2] = zs[:, None, None]
        mp[:k, 2] = zs
    for i, (c, n, cf, p0, n0) in enumerate(per):
        j = k + i
        gc[j, :c.shape[0], :c.shape[1]] = c
        gn[j, :n.shape[0], :n.shape[1]] = n
        coef[j] = cf
        shp[j] = c.shape[:2]
        mp[j] = p0
        mn[j] = n0
    return gc, gn, coef, shp, mp, mn, np.int32(k)


@functools.lru_cache(maxsize=None)
def _refracted_fn(coherent, split_sides, n_samples, n_crossed,
                  pattern="isotropic", refraction="sequential",
                  joint_newton=6, joint_backtrack=4, rough_terms=0,
                  rough_cross=False, gamma_facet=False, diffuse=False,
                  taper=False, rough_area=False):
    """Build (once per static configuration) the jitted vmapped kernel.

    Everything numeric that can vary between runs is a traced argument:
    ``blocks`` the target facet blocks, ``consts`` the per-crossed-interface
    lookup constants (``_grid_consts`` tuples), ``n_leg``/``eps_leg``/``att``
    per-leg index, permittivity, attenuation (dB/km) arrays, plus the window
    and radar scalars. Array-shape changes retrace via jit's own cache; only
    the tuple above forces a new build. Must be CALLED under
    ``jax.enable_x64()`` so tracing happens in the f64 scope.
    """
    n_seg = (2 if split_sides else 1) * n_samples  # +1 overflow slot for drops
    gfn = None if pattern == "isotropic" else gain_fn(pattern)

    def path(p, q, consts, n_leg, eps_leg, att, k0, sig_c):
        """Chained refracted path platform -> facet centers (float64),
        component form (``_snell_c``; p/q/returned points are 3-tuples).

        Leg i < j takes the incidence cosine at interface i (its lower end);
        the final leg takes the refraction cosine at interface j-1. For
        parallel planes these equal the per-medium ray angles exactly.
        ``rough_cross`` folds the two-way transmission roughness attenuation
        exp(-2 sig_c[i]^2 K_t^2) into tau2 (module docstring). ``consts``
        per crossed interface: (tab (ny, nx, 6) float32 [center, normal],
        coef, mp, mn) -- one gather per lookup.
        """
        shape = q[0].shape
        cur = tuple(p[i] + jnp.zeros(shape, q[0].dtype) for i in range(3))
        valid = jnp.ones(shape, bool)
        opl = jnp.zeros(shape, q[0].dtype)
        loss_db, sum_par, sum_perp = opl, opl, opl
        tau2 = jnp.ones_like(opl)
        c_first = c_last = x_first = None
        for i, (tab, coef, mp, mn) in enumerate(consts):
            # Pass 1 (mean plane) only anchors the facet lookup: sub-facet
            # accuracy is plenty, and Newton is quadratic -- 10 iterations is
            # orders of magnitude better than a facet width here.
            x1 = _snell_c(cur, q, (mp[0], mp[1], mp[2]), (mn[0], mn[1], mn[2]),
                          n_leg[i], n_leg[i + 1], 10)[0]
            rc0 = x1[0] * coef[0, 0] + x1[1] * coef[1, 0] + coef[2, 0]
            rc1 = x1[0] * coef[0, 1] + x1[1] * coef[1, 1] + coef[2, 1]
            row = jnp.clip(jnp.round(rc0), 0, tab.shape[0] - 1)
            col = jnp.clip(jnp.round(rc1), 0, tab.shape[1] - 1)
            row, col = row.astype(jnp.int32), col.astype(jnp.int32)
            g6 = tab[row, col].astype(q[0].dtype)
            pt = (g6[..., 0], g6[..., 1], g6[..., 2])
            nr = (g6[..., 3], g6[..., 4], g6[..., 5])
            inn = jax.lax.rsqrt(_dot3(nr, nr))
            nr = (nr[0] * inn, nr[1] * inn, nr[2] * inn)
            x2, th1, th2, s1, _, _, ok = _snell_c(cur, q, pt, nr, n_leg[i],
                                                 n_leg[i + 1], 25)
            valid &= ok
            c_inc = jnp.maximum(jnp.cos(th1), _C_MIN)
            if i == 0:
                c_first = c_inc
                x_first = x2  # first crossing: departure leg is p -> here
            g = fresnel_te(eps_leg[i], eps_leg[i + 1], c_inc, xp=jnp).gamma
            tau2 = tau2 * (1.0 - g * g)
            if rough_cross:
                kt = k0 * (n_leg[i] * c_inc - n_leg[i + 1] * jnp.cos(th2))
                tau2 = tau2 * jnp.exp(-2.0 * (sig_c[i] * kt) ** 2)
            opl = opl + n_leg[i] * s1
            loss_db = loss_db + s1 * (att[i] / 1000.0)
            sum_perp = sum_perp + s1 / n_leg[i]
            sum_par = sum_par + s1 / (n_leg[i] * c_inc * c_inc)
            c_last = jnp.maximum(jnp.cos(th2), _C_MIN)
            cur = x2
        return (cur, valid, opl, loss_db, sum_par, sum_perp, tau2, c_first,
                c_last, x_first)

    def path_joint(p, q, consts, n_leg, eps_leg, att, k0, sig_c):
        """Joint refracted path (float64): one block-tridiagonal Newton over
        ALL crossed interfaces (+ no-op padding, module docstring). Same
        returns as ``path`` (final leg excluded; added by ``step``): the
        padded air sub-segments book with the surface incidence cosine, so
        the split air leg accumulates exactly like the unsplit one. Pass 1
        anchors facets from the SAME mean-plane chain the sequential kernel
        uses (sub-facet accuracy is all the lookup needs; the joint pass-2
        then solves on identical local planes, so the two paths differ only
        by the chaining approximation itself)."""
        gc, gn, coef, shp, mp, mn, k_pad = consts
        n_pad = mp.shape[0]
        r1 = sequential_chain(p, q, mp, mn, n_leg, n_iter=10)

        def lookup(gc_i, gn_i, coef_i, shp_i, x_i):
            rc = (x_i[..., 0, None] * coef_i[0] + x_i[..., 1, None] * coef_i[1]
                  + coef_i[2])
            row = jnp.clip(jnp.round(rc[..., 0]), 0,
                           shp_i[0] - 1).astype(jnp.int32)
            col = jnp.clip(jnp.round(rc[..., 1]), 0,
                           shp_i[1] - 1).astype(jnp.int32)
            pt = gc_i[row, col].astype(q.dtype)
            nr = gn_i[row, col].astype(q.dtype)
            return pt, nr / jnp.linalg.norm(nr, axis=-1, keepdims=True)

        pt, nr = jax.vmap(lookup)(gc, gn, coef, shp, r1.x)
        r2 = joint_crossings(p, q, pt, nr, n_leg, n_newton=joint_newton,
                             n_backtrack=joint_backtrack)
        s_up = r2.s[:-1]                              # segments above interfaces
        c_inc = jnp.maximum(jnp.cos(r2.theta1), _C_MIN)
        c_surf = jnp.take(c_inc, k_pad, axis=0)       # first REAL interface
        is_pad = (jnp.arange(n_pad) < k_pad)[:, None]
        g = fresnel_te(eps_leg[:-1, None], eps_leg[1:, None], c_inc,
                       xp=jnp).gamma                  # pads: eps-matched, g = 0
        tau2 = jnp.prod(1.0 - g * g, axis=0)
        if rough_cross:
            # pads: index-matched and sigma 0 -> K_t = 0, factor exactly 1
            kt = k0 * (n_leg[:-1, None] * jnp.cos(r2.theta1)
                       - n_leg[1:, None] * jnp.cos(r2.theta2))
            tau2 = tau2 * jnp.prod(
                jnp.exp(-2.0 * (sig_c[:, None] * kt) ** 2), axis=0)
        c_par = jnp.where(is_pad, c_surf[None], c_inc)
        nl = n_leg[:-1, None]
        opl = jnp.sum(nl * s_up, axis=0)
        loss_db = jnp.sum(s_up * (att[:-1, None] / 1000.0), axis=0)
        sum_perp = jnp.sum(s_up / nl, axis=0)
        sum_par = jnp.sum(s_up / (nl * c_par * c_par), axis=0)
        c_last = jnp.maximum(jnp.cos(r2.theta2[-1]), _C_MIN)
        return (r2.x[-1], r2.valid, opl, loss_db, sum_par, sum_perp, tau2,
                c_surf, c_last, r2.x[0])

    def one_trace(p, u, pv, off, n_win, blocks, consts, n_leg, eps_leg, att,
                  t0, dt, c, gamma, k0, pa, pb, sig_t, l_t, sig_c, n_exp, tps):
        if refraction == "sequential":
            assert len(consts) == n_crossed

        def step(carry, i):
            hist, dropped = carry
            # this trace's i-th window block (geometry.block_windows): the
            # per-trace offset makes the block fetch a batched gather
            j = off + i
            blk = [jax.lax.dynamic_index_in_dim(x, j, 0, False)
                   for x in blocks]
            # per-facet gamma rides the blocked scan like the phasors do; the
            # scalar path (gamma_facet=False) traces exactly the old program
            fg = None
            blk_ = list(blk)
            fdif = fdph = None
            if diffuse:
                fdif, fdph = blk_[-2], blk_[-1]
                blk_ = blk_[:-2]
            if rough_terms and gamma_facet:
                fc, fn, fa, f1, f2, fph, fg = blk_
            elif rough_terms:
                fc, fn, fa, f1, f2, fph = blk_
            elif gamma_facet:
                fc, fn, fa, f1, f2, fg = blk_
            else:
                fc, fn, fa, f1, f2 = blk_
            gam = fg if gamma_facet else gamma
            q64 = fc.astype(jnp.float64)
            pc = (p[0], p[1], p[2])
            if refraction == "sequential":
                q = _c3(q64)
                (cur, valid, opl, loss_db, sum_par, sum_perp, tau2, c0,
                 c_last, x_first) = path(pc, q, consts, n_leg, eps_leg, att,
                                         k0, sig_c)
            else:
                (cur, valid, opl, loss_db, sum_par, sum_perp, tau2, c0,
                 c_last, x_first) = path_joint(p, q64, consts, n_leg,
                                               eps_leg, att, k0, sig_c)
                q, cur, x_first = _c3(q64), _c3(cur), _c3(x_first)
            # Final leg (medium j): crossing -> facet (component form).
            d = (cur[0] - q[0], cur[1] - q[1], cur[2] - q[2])  # facet -> crossing
            s_j = jnp.sqrt(_dot3(d, d))
            inv = 1.0 / jnp.maximum(s_j, 1e-30)
            rhat = (d[0] * inv, d[1] * inv, d[2] * inv)
            fn64 = _c3(fn.astype(jnp.float64))
            nj = n_leg[-1]
            opl = opl + nj * s_j
            loss_db = loss_db + s_j * (att[-1] / 1000.0)
            sum_perp = sum_perp + s_j / nj
            sum_par = sum_par + s_j / (nj * c_last * c_last)
            l_perp = n_leg[0] * sum_perp
            l_par = n_leg[0] * c0 * c0 * sum_par
            flux = (n_leg[0] * c0) / (nj * c_last)
            att_f = 10.0 ** (-loss_db / 10.0)  # two-way FIELD = one-way power
            cos_t = _dot3(rhat, fn64)
            spread = tau2 * flux / (l_par * l_perp)
            if gfn is not None:
                # Antenna gain at the air-leg departure direction; g**2 on the
                # (squared-in-incoherent) amplitude = field convention.
                d0 = (x_first[0] - pc[0], x_first[1] - pc[1],
                      x_first[2] - pc[2])
                dm = jnp.maximum(jnp.sqrt(_dot3(d0, d0)), 1e-30)
                dhat0 = jnp.stack([d0[0] / dm, d0[1] / dm, d0[2] / dm], -1)
                g = gfn(dhat0, pv, pa, pb)
                spread = spread * (g * g)
            if coherent:
                kj = k0 * nj
                f1d, f2d = _c3(f1.astype(jnp.float64)), _c3(f2.astype(jnp.float64))
                s1 = jnp.sinc(_dot3(rhat, f1d) * (kj / np.pi))
                s2 = jnp.sinc(_dot3(rhat, f2d) * (kj / np.pi))
                amp = ((kj / TWO_PI) * gam * cos_t * fa * s1 * s2 * spread
                       * att_f)
                if taper:
                    # off-specular taper on the SPECULAR term only
                    # (coherent.py _off_specular_taper, same form; cos_t is
                    # the incidence cosine at the REFRACTED arrival)
                    ct2 = jnp.clip(cos_t, _C_MIN, 1.0) ** 2
                    amp = amp * jnp.exp(-(1.0 - ct2)
                                        / (ct2 * (2.0 * tps * tps)))
                if rough_terms:
                    d1 = _dot3(rhat, f1d)
                    d2 = _dot3(rhat, f2d)
                    l1 = jnp.sqrt(_dot3(f1d, f1d))
                    l2 = jnp.sqrt(_dot3(f2d, f2d))
                    kk = 2.0 * kj * cos_t
                    dp = d_phi(sig_t, l_t, kk, 2.0 * kj * d1 / l1,
                               2.0 * kj * d2 / l2, l1, l2,
                               n_terms=rough_terms, area_only=rough_area)
                    # area-mask: zero-padded block slots (f1 = f2 = 0) make
                    # the d_phi args 0/0 -> NaN; the smooth term is killed by
                    # fa = 0 but the incoherent term has no area factor
                    amp = (amp * mean_attenuation(sig_t, kk)
                           + jnp.where(fa > 0,
                                       (kj / TWO_PI) * gam * cos_t * spread
                                       * att_f * jnp.sqrt(dp) * fph, 0.0))
                if diffuse:
                    # module docstring: sqrt(A/2pi) * amp * cos_t^(1+n/2) *
                    # spread * att_f, incoherent (frozen unit phasor). The
                    # area mask kills zero-padded block slots.
                    ct = jnp.maximum(cos_t, 0.0)
                    a_d = (jnp.sqrt(jnp.maximum(fa, 0.0) / TWO_PI)
                           * fdif.astype(jnp.float64) * ct
                           * ct ** (0.5 * n_exp) * spread * att_f)
                    amp = amp + jnp.where(fa > 0, a_d * fdph, 0.0)
                contrib = (1j * amp * jnp.exp(-2j * k0 * opl)).astype(
                    jnp.complex64)
                pwr = (jnp.real(contrib) ** 2
                       + jnp.imag(contrib) ** 2).astype(jnp.float32)
            else:
                a = fa * cos_t * spread
                contrib = (a * a * att_f * att_f).astype(jnp.float32)
                pwr = contrib
            b = twtt_bin(2.0 * opl / c, t0, dt)
            ok = valid & (b >= 0) & (b < n_samples)
            if split_sides:
                right = (_dot3((q[0] - pc[0], q[1] - pc[1], q[2] - pc[2]),
                               (u[0], u[1], u[2])) > 0).astype(jnp.int32)
                b = b + right * n_samples
            seg = jnp.where(ok, b, n_seg)
            h = jax.ops.segment_sum(contrib, seg, num_segments=n_seg + 1)
            drop = jnp.sum(jnp.where(ok, jnp.float32(0.0), pwr))
            return (hist + h[:n_seg], dropped + drop), None

        init = (jnp.zeros(n_seg, jnp.complex64 if coherent else jnp.float32),
                jnp.float32(0.0))
        (hist, dropped), _ = jax.lax.scan(step, init, jnp.arange(n_win))
        return hist, dropped

    return jax.jit(jax.vmap(one_trace, in_axes=(0, 0, 0, 0) + (None,) * 18),
                   static_argnums=(4,))


def refracted_cluttergram(positions, u_ct, target, crossed, eps_leg, att_leg,
                          *, mode, t0, dt, n_samples, c, gamma=0.0, k0=None,
                          split_sides=False, pattern=None, roughness=None,
                          crossed_sigma=None, diffuse=None, taper_s=None,
                          d_phi_area=False, block_size=None,
                          refraction="sequential", pad_to=None,
                          joint_newton=6, joint_backtrack=4,
                          window_cull=True):
    """Binned refracted-path contributions from one target interface.

    positions/u_ct: (T, 3) platform positions / cross-track unit vectors
    (local frame). ``target``/``crossed``: Facets of the target interface and
    the interfaces above it (top-down). ``eps_leg``/``att_leg``: per-leg medium
    permittivity and one-way attenuation (dB/km), len == len(crossed) + 1.
    ``gamma``/``k0`` are the target reflection coefficient and vacuum
    wavenumber (coherent mode only). ``gamma`` may be a scalar or an
    (n_facets,) per-facet FIELD coefficient array (e.g. a spatially varying
    bed reflectivity); the array rides the blocked scan like the roughness
    phasors (float64 -- an array of a constant is bit-identical to the
    scalar), and requires coherent mode (the incoherent path books no target
    reflectivity by convention). ``pattern``: None (isotropic) or an
    ``antenna.pattern_args`` tuple -- contributions then carry the two-way
    antenna gain at the air-leg departure direction (g**2 field / g**4 power).

    ``roughness`` (coherent mode only): None (smooth) or ``(sigma_m,
    corr_length_m, phasors, n_terms)`` for the TARGET reflection --
    ``phasors`` the (n_facets,) complex per-facet speckle phasors
    (``roughness.speckle_phasors``), ``n_terms`` the static series length
    (``roughness.n_terms_for``). ``crossed_sigma`` (coherent mode only):
    None or per-crossed-interface RMS heights (m, len == len(crossed),
    zeros where smooth) for the two-way transmission attenuation
    exp(-2 sigma^2 K_t^2) per crossing (module docstring).
    ``taper_s``/``d_phi_area`` (coherent mode only): the grazing-fix pair
    (coherent.py, config.py ``GrazingFixConfig``) -- the off-specular taper
    s_eff on the target's SPECULAR term (None = off) and the area-term-only
    D_Phi; the defaults trace exactly the pre-fix program.

    ``refraction`` selects the crossing solver (module docstring): the
    kernel-level default stays ``"sequential"``; simulate() passes the
    ``SimConfig.refraction`` choice. For ``"joint"``, ``pad_to`` (default
    ``len(crossed)``) pads the stack with no-op interfaces so calls with
    different layer counts share one executable, and
    ``joint_newton``/``joint_backtrack`` are the pass-2 Newton budgets
    (compile-time statics). ``block_size`` defaults to
    ``geometry.auto_block_size`` (~256k f64 lanes per step; 4096 for the
    joint path, whose damped-Newton candidate buffers scale with block *
    pad_to * backtrack) and is also the per-trace window granularity. ``window_cull``: per-trace
    along-track facet windowing (geometry.py ``block_windows``) -- each trace
    scans only the facet blocks that can bin inside the fast-time window;
    exact for ``out`` (skipped facets are silent), ``dropped`` excludes them.
    False scans every block (regression use).

    Returns ``(out, dropped)`` NumPy arrays: out is float32 power (incoherent)
    or complex64 field (coherent), (T, n_samples) or (T, n_samples, 2) with
    ``split_sides`` ([left, right]); dropped (T,) float32 accumulates the
    power of out-of-window AND invalid-path (shadow/non-converged)
    contributions.
    """
    coherent = mode == "coherent"
    joint = refraction == "joint"
    if (roughness is not None or crossed_sigma is not None) and not coherent:
        raise ValueError("roughness requires coherent mode")
    if (taper_s is not None or d_phi_area) and not coherent:
        raise ValueError("the grazing fix requires coherent mode")
    gamma_arr = np.asarray(gamma, np.float64)
    gamma_facet = gamma_arr.ndim > 0
    if gamma_facet and not coherent:
        raise ValueError("per-facet gamma requires coherent mode")
    eps = np.asarray(eps_leg, np.float64)
    n_leg = np.sqrt(eps)
    att = np.asarray(att_leg, np.float64)
    pos = np.asarray(positions, np.float64)
    sig_c = (np.zeros(1) if crossed_sigma is None
             else np.asarray(crossed_sigma, np.float64))
    if joint:
        consts = _joint_consts(crossed, pad_to or len(crossed),
                               pos[:, 2].min())
        kp = int(consts[-1])
        n_leg = np.concatenate([np.full(kp, n_leg[0]), n_leg])
        eps = np.concatenate([np.full(kp, eps[0]), eps])
        att = np.concatenate([np.full(kp, att[0]), att])
        if crossed_sigma is not None:
            sig_c = np.concatenate([np.zeros(kp), sig_c])
    else:
        # sequential path: one interleaved (ny, nx, 6) [center, normal]
        # lookup table per crossed interface (a single gather in ``path``)
        consts = tuple(
            (np.concatenate([gc, gn], axis=-1), coef, mp, mn)
            for gc, gn, coef, mp, mn in map(_grid_consts, crossed))

    n = target.centers.shape[0]
    block_size = min(block_size or (4096 if joint else auto_block_size(
        len(pos), 1 << 18)), n)
    n_blocks = -(-n // block_size)
    pad = n_blocks * block_size - n

    # Along-track facet order + per-trace block windows (geometry.py): the
    # per-facet sum is order-dependent only at rounding level; skipped
    # blocks are provably silent (twtt >= window end), so ``out`` equals the
    # all-facet sum and only ``dropped`` shrinks.
    order, s_t, s_sorted = along_track_order(pos, target.centers)
    reach = (window_reach_m(t0, dt, n_samples, c) if window_cull
             else np.inf)
    off, n_win = block_windows(s_sorted, s_t, reach, block_size, n_blocks)

    def blocks(a):
        a = np.asarray(a, dtype=np.float32)[order]
        a = np.pad(a, ((0, pad),) + ((0, 0),) * (a.ndim - 1))
        return jnp.asarray(a.reshape(n_blocks, block_size, *a.shape[1:]))

    blk = (blocks(target.centers), blocks(target.normals),
           blocks(target.areas), blocks(target.e1), blocks(target.e2))
    if roughness is not None:
        sig_t, l_t, phasors, n_terms = roughness
        ph = np.pad(np.asarray(phasors, np.complex64)[order], (0, pad))
        blk = blk + (jnp.asarray(ph.reshape(n_blocks, block_size)),)
    else:
        sig_t = l_t = 0.0
        n_terms = 0
    if diffuse is not None:
        if not coherent:
            raise ValueError("diffuse channel requires coherent mode")
        d_amp, d_ph, n_exp = diffuse
        d_amp = np.asarray(d_amp, np.float64)
        if d_amp.shape != (n,):
            raise ValueError(
                f"diffuse amp shape {d_amp.shape} != ({n},)")
    else:
        n_exp = 0.0
    if gamma_facet:
        if gamma_arr.shape != (n,):
            raise ValueError(
                f"per-facet gamma shape {gamma_arr.shape} != ({n},)")
        # float64 NumPy, converted under the x64 scope below (like positions)
        # so an array of a constant matches the scalar path bit-exactly
        blk = blk + (np.pad(gamma_arr[order], (0, pad)).reshape(
            n_blocks, block_size),)
    if diffuse is not None:
        # float64 amplitude + complex64 frozen phasors, appended LAST so the
        # unpacking above stays positional (see one_trace)
        blk = blk + (
            np.pad(d_amp[order], (0, pad)).reshape(n_blocks, block_size),
            jnp.asarray(np.pad(np.asarray(d_ph, np.complex64)[order],
                               (0, pad)).reshape(n_blocks, block_size)))
    # Positions stay float64 NumPy: converted under the x64 scope below, so
    # the platform coordinates (the largest magnitudes) are not truncated.
    uct = np.asarray(u_ct, np.float64)

    kind, pv, pa, pb = pattern or ("isotropic", np.zeros((len(pos), 3)),
                                   0.0, 0.0)
    pv = np.asarray(pv, np.float64)
    pa, pb = np.asarray(pa, np.float64), np.asarray(pb, np.float64)

    fn = _refracted_fn(coherent, split_sides, int(n_samples),
                       None if joint else len(crossed), kind, refraction,
                       int(joint_newton), int(joint_backtrack), int(n_terms),
                       crossed_sigma is not None, gamma_facet,
                       diffuse is not None, taper_s is not None,
                       bool(d_phi_area))
    with jax.enable_x64():
        hist, dropped = fn(pos, uct, pv, jnp.asarray(off), n_win, blk, consts,
                           n_leg, eps, att,
                           np.float64(t0), np.float64(dt), np.float64(c),
                           np.float64(0.0 if gamma_facet else gamma),
                           np.float64(0.0 if k0 is None else k0), pa, pb,
                           np.float64(sig_t), np.float64(l_t), sig_c,
                           np.float64(n_exp),
                           np.float64(0.0 if taper_s is None else taper_s))
        hist, dropped = np.asarray(hist), np.asarray(dropped)
    if split_sides:
        hist = hist.reshape(-1, 2, n_samples).transpose(0, 2, 1)
    return hist, dropped
