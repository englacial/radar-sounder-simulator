"""Refracted-path multilayer kernel: both modes through one geometry path.

For a target interface j (facets in medium j, reflecting off medium j+1) the
ray from the platform refracts through interfaces 0..j-1. Per (trace, target
facet) the chain is solved sequentially, interface by interface (top-down):

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

Compilation caching: the jitted callable is built once per static
configuration -- ``(mode, split_sides, n_samples, n_crossed, pattern)``, memoized via
``functools.lru_cache`` -- with every run-varying number (t0/dt/c/gamma/k0,
per-leg eps/index/attenuation, interface lookup constants, facet blocks,
positions) passed as traced arguments, so repeat calls (and calls that change
only numeric values, e.g. layer depths) hit jax's shape-based jit cache
instead of recompiling. Tracing happens on first call, inside the caller's
``jax.enable_x64()`` scope, which keeps the f64 path f64.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ..antenna import gain_fn
from ..physics import fresnel_te
from ..refraction import snell_crossing
from .geometry import twtt_bin

TWO_PI = 2.0 * np.pi
_C_MIN = 1e-9  # grazing-cosine clamp: keeps 1/c^2 finite; amplitudes -> 0


def _grid_consts(facets):
    """Per-interface lookup constants: gridded centers/normals (float32), an
    affine (x, y, 1) -> (row, col) cell-index fit, and the area-weighted mean
    plane (float64 NumPy -- traced as f64 constants under the x64 scope)."""
    ny, nx = facets.grid_shape
    grid_c = jnp.asarray(facets.centers.reshape(ny, nx, 3).astype(np.float32))
    grid_n = jnp.asarray(facets.normals.reshape(ny, nx, 3).astype(np.float32))
    a = np.column_stack([facets.centers[:, 0], facets.centers[:, 1],
                         np.ones(len(facets.centers))])
    coef, *_ = np.linalg.lstsq(a, facets.cell.astype(np.float64), rcond=None)
    w = (facets.areas / facets.areas.sum())[:, None]
    mp = (facets.centers * w).sum(0)
    mn = (facets.normals * w).sum(0)
    mn = mn / np.linalg.norm(mn)
    return grid_c, grid_n, coef, mp, mn


@functools.lru_cache(maxsize=None)
def _refracted_fn(coherent, split_sides, n_samples, n_crossed,
                  pattern="isotropic"):
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

    def path(p, q, consts, n_leg, eps_leg, att):
        """Chained refracted path platform -> facet centers (float64).

        Leg i < j takes the incidence cosine at interface i (its lower end);
        the final leg takes the refraction cosine at interface j-1. For
        parallel planes these equal the per-medium ray angles exactly.
        """
        cur = p + jnp.zeros_like(q)
        valid = jnp.ones(q.shape[:-1], bool)
        opl = jnp.zeros(q.shape[:-1], q.dtype)
        loss_db, sum_par, sum_perp = opl, opl, opl
        tau2 = jnp.ones_like(opl)
        c_first = c_last = x_first = None
        for i, (gc, gn, coef, mp, mn) in enumerate(consts):
            # Pass 1 (mean plane) only anchors the facet lookup: sub-facet
            # accuracy is plenty, and Newton is quadratic -- 10 iterations is
            # orders of magnitude better than a facet width here.
            r1 = snell_crossing(cur, q, mp, mn, n_leg[i], n_leg[i + 1],
                                n_iter=10)
            rc = (r1.x[..., 0, None] * coef[0] + r1.x[..., 1, None] * coef[1]
                  + coef[2])
            row = jnp.clip(jnp.round(rc[..., 0]), 0, gc.shape[0] - 1)
            col = jnp.clip(jnp.round(rc[..., 1]), 0, gc.shape[1] - 1)
            row, col = row.astype(jnp.int32), col.astype(jnp.int32)
            pt = gc[row, col].astype(q.dtype)
            nr = gn[row, col].astype(q.dtype)
            nr = nr / jnp.linalg.norm(nr, axis=-1, keepdims=True)
            r2 = snell_crossing(cur, q, pt, nr, n_leg[i], n_leg[i + 1])
            valid &= r2.valid
            c_inc = jnp.maximum(jnp.cos(r2.theta1), _C_MIN)
            if i == 0:
                c_first = c_inc
                x_first = r2.x  # first crossing: departure leg is p -> here
            g = fresnel_te(eps_leg[i], eps_leg[i + 1], c_inc, xp=jnp).gamma
            tau2 = tau2 * (1.0 - g * g)
            opl = opl + n_leg[i] * r2.s1
            loss_db = loss_db + r2.s1 * (att[i] / 1000.0)
            sum_perp = sum_perp + r2.s1 / n_leg[i]
            sum_par = sum_par + r2.s1 / (n_leg[i] * c_inc * c_inc)
            c_last = jnp.maximum(jnp.cos(r2.theta2), _C_MIN)
            cur = r2.x
        return (cur, valid, opl, loss_db, sum_par, sum_perp, tau2, c_first,
                c_last, x_first)

    def one_trace(p, u, pv, blocks, consts, n_leg, eps_leg, att, t0, dt, c,
                  gamma, k0, pa, pb):
        assert len(consts) == n_crossed
        cb, nb, ab, e1b, e2b = blocks

        def step(carry, blk):
            hist, dropped = carry
            fc, fn, fa, f1, f2 = blk
            q = fc.astype(jnp.float64)
            (cur, valid, opl, loss_db, sum_par, sum_perp, tau2, c0,
             c_last, x_first) = path(p, q, consts, n_leg, eps_leg, att)
            # Final leg (medium j): crossing -> facet.
            d = cur - q                       # facet -> last crossing (up-path)
            s_j = jnp.sqrt(jnp.sum(d * d, axis=-1))
            rhat = d / jnp.maximum(s_j, 1e-30)[..., None]
            nj = n_leg[-1]
            opl = opl + nj * s_j
            loss_db = loss_db + s_j * (att[-1] / 1000.0)
            sum_perp = sum_perp + s_j / nj
            sum_par = sum_par + s_j / (nj * c_last * c_last)
            l_perp = n_leg[0] * sum_perp
            l_par = n_leg[0] * c0 * c0 * sum_par
            flux = (n_leg[0] * c0) / (nj * c_last)
            att_f = 10.0 ** (-loss_db / 10.0)  # two-way FIELD = one-way power
            cos_t = jnp.sum(rhat * fn.astype(jnp.float64), axis=-1)
            spread = tau2 * flux / (l_par * l_perp)
            if gfn is not None:
                # Antenna gain at the air-leg departure direction; g**2 on the
                # (squared-in-incoherent) amplitude = field convention.
                d0 = x_first - p
                dhat0 = d0 / jnp.maximum(
                    jnp.sqrt(jnp.sum(d0 * d0, axis=-1)), 1e-30)[..., None]
                g = gfn(dhat0, pv, pa, pb)
                spread = spread * (g * g)
            if coherent:
                kj = k0 * nj
                s1 = jnp.sinc(jnp.sum(rhat * f1.astype(jnp.float64), -1)
                              * (kj / np.pi))
                s2 = jnp.sinc(jnp.sum(rhat * f2.astype(jnp.float64), -1)
                              * (kj / np.pi))
                amp = ((kj / TWO_PI) * gamma * cos_t * fa * s1 * s2 * spread
                       * att_f)
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
                right = (jnp.sum((q - p) * u, axis=-1) > 0).astype(jnp.int32)
                b = b + right * n_samples
            seg = jnp.where(ok, b, n_seg)
            h = jax.ops.segment_sum(contrib, seg, num_segments=n_seg + 1)
            drop = jnp.sum(jnp.where(ok, jnp.float32(0.0), pwr))
            return (hist + h[:n_seg], dropped + drop), None

        init = (jnp.zeros(n_seg, jnp.complex64 if coherent else jnp.float32),
                jnp.float32(0.0))
        (hist, dropped), _ = jax.lax.scan(step, init, (cb, nb, ab, e1b, e2b))
        return hist, dropped

    return jax.jit(jax.vmap(one_trace, in_axes=(0, 0, 0) + (None,) * 12))


def refracted_cluttergram(positions, u_ct, target, crossed, eps_leg, att_leg,
                          *, mode, t0, dt, n_samples, c, gamma=0.0, k0=None,
                          split_sides=False, pattern=None, block_size=65536):
    """Binned refracted-path contributions from one target interface.

    positions/u_ct: (T, 3) platform positions / cross-track unit vectors
    (local frame). ``target``/``crossed``: Facets of the target interface and
    the interfaces above it (top-down). ``eps_leg``/``att_leg``: per-leg medium
    permittivity and one-way attenuation (dB/km), len == len(crossed) + 1.
    ``gamma``/``k0`` are the target reflection coefficient and vacuum
    wavenumber (coherent mode only). ``pattern``: None (isotropic) or an
    ``antenna.pattern_args`` tuple -- contributions then carry the two-way
    antenna gain at the air-leg departure direction (g**2 field / g**4 power).

    Returns ``(out, dropped)`` NumPy arrays: out is float32 power (incoherent)
    or complex64 field (coherent), (T, n_samples) or (T, n_samples, 2) with
    ``split_sides`` ([left, right]); dropped (T,) float32 accumulates the
    power of out-of-window AND invalid-path (shadow/non-converged)
    contributions.
    """
    coherent = mode == "coherent"
    eps = np.asarray(eps_leg, np.float64)
    n_leg = np.sqrt(eps)
    att = np.asarray(att_leg, np.float64)
    consts = tuple(_grid_consts(f) for f in crossed)

    n = target.centers.shape[0]
    block_size = min(block_size, n)
    n_blocks = -(-n // block_size)
    pad = n_blocks * block_size - n

    def blocks(a):
        a = np.asarray(a, dtype=np.float32)
        a = np.pad(a, ((0, pad),) + ((0, 0),) * (a.ndim - 1))
        return jnp.asarray(a.reshape(n_blocks, block_size, *a.shape[1:]))

    blk = (blocks(target.centers), blocks(target.normals),
           blocks(target.areas), blocks(target.e1), blocks(target.e2))
    # Positions stay float64 NumPy: converted under the x64 scope below, so
    # the platform coordinates (the largest magnitudes) are not truncated.
    pos = np.asarray(positions, np.float64)
    uct = np.asarray(u_ct, np.float64)

    kind, pv, pa, pb = pattern or ("isotropic", np.zeros((len(pos), 3)),
                                   0.0, 0.0)
    pv = np.asarray(pv, np.float64)
    pa, pb = np.asarray(pa, np.float64), np.asarray(pb, np.float64)

    fn = _refracted_fn(coherent, split_sides, int(n_samples), len(crossed),
                       kind)
    with jax.enable_x64():
        hist, dropped = fn(pos, uct, pv, blk, consts, n_leg, eps, att,
                           np.float64(t0), np.float64(dt), np.float64(c),
                           np.float64(gamma),
                           np.float64(0.0 if k0 is None else k0), pa, pb)
        hist, dropped = np.asarray(hist), np.asarray(dropped)
    if split_sides:
        hist = hist.reshape(-1, 2, n_samples).transpose(0, 2, 1)
    return hist, dropped
