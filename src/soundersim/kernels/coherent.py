"""Coherent (complex-field) clutter kernel: JAX float32/complex64, vmapped.

Per rectangular facet (center m, un-normalized edge vectors e1/e2 spanning the
mean plane, area A = |e1 x e2|, unit normal n), the linear-phase-approximation
(LPA) physical-optics field in the M9 normalization convention
(claude_notes/coherent_normalization.md):

    field = (j*k/2pi) * gamma * cos(theta) * A
            * sinc(k * rhat.e1) * sinc(k * rhat.e2) * exp(-2j*k*r) / r**2

with sinc(x) = sin(x)/x (unnormalized) and rhat the unit vector facet -> platform.
Derivation: offsets rho = u*e1 + v*e2, u,v in [-1/2, 1/2]; LPA two-way phase
across the facet is exp(-2j*k*rhat.rho), so the facet integral of the
brute-force integrand factorizes into A * sinc(k*rhat.e1) * sinc(k*rhat.e2).
As the facet size -> 0 both sincs -> 1 and this reduces exactly to the
brute-force single-sample expression.

Contributions are complex-summed into fast-time bins floor((2r/c - t0)/dt);
out-of-window contributions are dropped (never wrapped) with their power
|contribution|**2 accumulated per trace. Same fixed-size facet blocking as the
incoherent kernel.

``interp_bins`` (stage 4, M20): when enabled, each facet's complex
contribution is split LINEARLY between the two adjacent bins b/b+1 with
weights (1-w)/w, w = (twtt - t0)/dt - b, so the envelope-delay quantization
error drops from O(dt) to the linear-interp residual (the carrier phase was
always exact). The two split parts are windowed and dropped independently
(dropped power accumulates |part|^2); the complex SUM of a facet's binned
contributions is preserved exactly, but sum-of-|field|^2 is not (fields, not
powers, are split) -- the M20 energy bookkeeping test documents both. The
flag is a static in the jit-factory cache key (it changes the graph
structure); the default False path traces exactly the pre-stage-4 program
(regression-gated bit-compatible).

Antenna pattern (M22): a non-isotropic pattern weights each facet's FIELD by
g**2 (two-way, monostatic; g the one-way FIELD gain, see antenna.py),
evaluated in-kernel from the departure direction. The pattern KIND is static
in the jit-factory key (the isotropic path traces exactly the pre-M22
program); the per-trace pattern vector and parameters are traced, so value
changes never recompile.

Grazing fix (opt-in, config.py ``GrazingFixConfig``): at grazing incidence
the LPA phase ramp across a facet is 2kL sin(theta) >> pi, the sinc tails
stop converging with facet size (facet-lattice aliasing: grid lobes), and
the resulting floor is non-physical. ``taper_s`` multiplies each facet's
SMOOTH (specular) field by exp(-tan^2(alpha)/(2 s_eff^2)) with alpha the
off-normal arrival angle (``_off_specular_taper``); the removed power is
grid aliasing, not physical power, so it is dropped, not re-booked -- the
physical off-specular return is the D_Phi incoherent channel, which
``d_phi_area`` reduces to its facet-area-scaling (infinite-surface PO) term.
Both default off and then trace exactly the legacy program.

Phase precision (stage-2 plan constraint 2, decided by measurement): the f32
hot loop computes the carrier phase from 2k*(r - r_ref) with r_ref a per-trace
float64 reference range (platform -> facet-centroid distance), keeping the
argument small; the constant exp(-2j*k*r_ref) is folded back per trace in
complex128 outside the loop (2k*r_ref reduced mod 2pi in float64). Measured in
tests/test_coherent_kernel.py: ~1e-4 wavelengths equivalent range error at
20 km / 195 MHz, vs the lambda/50 requirement (a naive f32 exp(-2jkr) path
measures ~1e-3 wavelengths there -- also passing, but with 10x less margin).

Sub-facet roughness (Gerekos et al. 2023, roughness.py, docs/roughness.md):
``roughness=(sigma_m, corr_length_m, phasors, n_terms)`` replaces each
facet's response by

    F * <Phi> + F * sqrt(D_Phi) * phi_r

with F = j*(k/2pi)*gamma*cos(theta)*exp(-2jkr)/r^2 the non-phase-integral
factor (identical for both terms, antenna weighting included), <Phi> the
smooth sinc*sinc response times exp(-sigma^2 K^2/2) (K = 2 k cos(theta)),
D_Phi the Eq 21 incoherent variance from the facet's in-plane coefficients
A0 = 2k(rhat.e1)/|e1| etc., and phi_r the per-facet frozen speckle phasor
(one realization per run: along-track speckle decorrelates through F's
phase/geometry, as in the paper). ``n_terms`` is static (series length);
``roughness=None`` (default) traces exactly the smooth program, and
sigma = 0 is bit-identical to it (attenuation 1.0 exact, D_Phi 0.0 exact;
regression-gated).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ..antenna import gain_fn
from ..roughness import d_phi, mean_attenuation
from .geometry import (along_track_order, auto_block_size, block_windows,
                       twtt_bin, window_reach_m)

TWO_PI = 2.0 * np.pi


def _off_specular_taper(cos, taper_s, xp=jnp):
    """Coherent off-specular taper T(alpha) = exp(-tan^2(alpha)/(2 s_eff^2))
    on the FIELD, with alpha the arrival angle off the facet normal
    (cos = cos(alpha)) and ``taper_s`` the effective rms slope s_eff
    (config.py ``GrazingFixConfig``): a facet only mirrors power back within
    its sub-facet slope spread, so the non-converging sinc/grid-lobe tails at
    grazing are suppressed while near-specular (glinting) facets keep T ~ 1.
    Back-facing facets (cos <= 0) taper to 0 via the clamp."""
    c = xp.clip(cos, 1e-9, 1.0)
    c2 = c * c
    return xp.exp(-(1.0 - c2) / (c2 * (2.0 * taper_s * taper_s)))


def lpa_contributions(position, centers, normals, areas, e1, e2, k, gamma,
                      r_ref=0.0, xp=jnp, taper_s=None):
    """Per-facet complex LPA field contributions and ranges.

    Phase is computed from (r - r_ref); the caller owes a constant factor
    exp(-2j*k*r_ref) (r_ref=0 gives the absolute field). ``xp`` selects the
    array module (jnp inside the kernel; np for float64 reference use --
    dtype follows the inputs). ``taper_s``: None (exact legacy program) or
    the s_eff of ``_off_specular_taper``.
    """
    d = position - centers
    r = xp.sqrt(xp.sum(d * d, axis=-1))
    cos = xp.sum(d * normals, axis=-1) / r
    rhat = d / r[..., None]
    # xp.sinc is the normalized sinc sin(pi x)/(pi x); divide args by pi.
    s1 = xp.sinc(xp.sum(rhat * e1, axis=-1) * (k / np.pi))
    s2 = xp.sinc(xp.sum(rhat * e2, axis=-1) * (k / np.pi))
    amp = (k / TWO_PI) * gamma * cos * areas * s1 * s2 / (r * r)
    if taper_s is not None:
        amp = amp * _off_specular_taper(cos, taper_s, xp=xp)
    phase = (-2.0 * k) * (r - r_ref)  # dtype (and precision) follows the inputs
    return 1j * amp * xp.exp(1j * phase), r


def rough_lpa_contributions(position, centers, normals, areas, e1, e2, k,
                            gamma, sigma, l, phasors, n_terms, r_ref=0.0,
                            taper_s=None, area_only=False):
    """Rough-facet LPA contributions (module docstring): the smooth response
    times exp(-sigma^2 K^2 / 2) plus the incoherent sqrt(D_Phi)*phi_r term
    with the same non-phase factor. Ops shared with ``lpa_contributions`` are
    computed identically (sigma = 0 is bit-identical to the smooth path).
    ``taper_s`` tapers the SMOOTH (specular) term only
    (``_off_specular_taper``; the D_Phi term is the physical off-specular
    channel and is never tapered); ``area_only`` selects the area-term-only
    D_Phi (roughness.d_phi). Both default to the exact legacy program."""
    d = position - centers
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    cos = jnp.sum(d * normals, axis=-1) / r
    rhat = d / r[..., None]
    d1 = jnp.sum(rhat * e1, axis=-1)
    d2 = jnp.sum(rhat * e2, axis=-1)
    s1 = jnp.sinc(d1 * (k / np.pi))
    s2 = jnp.sinc(d2 * (k / np.pi))
    amp = (k / TWO_PI) * gamma * cos * areas * s1 * s2 / (r * r)
    if taper_s is not None:
        amp = amp * _off_specular_taper(cos, taper_s)
    phase = (-2.0 * k) * (r - r_ref)
    # facet-local in-plane coefficients: sinc arg k*d1 == Lx*A0/2
    l1 = jnp.sqrt(jnp.sum(e1 * e1, axis=-1))
    l2 = jnp.sqrt(jnp.sum(e2 * e2, axis=-1))
    kk = 2.0 * k * cos
    dp = d_phi(sigma, l, kk, 2.0 * k * d1 / l1, 2.0 * k * d2 / l2, l1, l2,
               n_terms=n_terms, area_only=area_only)
    # area-mask: zero-padded block slots have e1 = e2 = 0, so l1 = l2 = 0 and
    # the d_phi args are 0/0 -> NaN; the smooth term is killed by areas = 0
    # but the incoherent term has no area factor, so mask it explicitly
    amp_i = jnp.where(areas > 0,
                      (k / TWO_PI) * gamma * cos / (r * r) * jnp.sqrt(dp), 0.0)
    contrib = amp * mean_attenuation(sigma, kk) + amp_i * phasors
    return 1j * contrib * jnp.exp(1j * phase), r


@functools.lru_cache(maxsize=None)
def _coherent_fn(split_sides, n_samples, interp, pattern="isotropic",
                 rough_terms=0, gamma_facet=False, taper=False,
                 rough_area=False):
    """Jitted vmapped kernel for one static configuration; run-varying
    numbers (facet blocks, positions, reference ranges, k/gamma/t0/dt/c,
    pattern vector/params pv/pa/pb) are traced arguments, so value changes
    reuse the compiled kernel. ``interp`` (static: it changes the graph)
    enables sub-bin linear splitting; False traces exactly the pre-stage-4
    program. ``pattern`` (static) selects the antenna gain graph; "isotropic"
    traces exactly the pre-M22 program. ``rough_terms`` (static: series
    length) switches the per-facet response to the rough-facet form (module
    docstring); 0 (smooth) traces exactly the pre-roughness program (the
    phasor blocks and sigma/l scalars are then unused). ``taper`` (static:
    it changes the graph) enables the off-specular taper -- the traced
    ``tps`` scalar carries s_eff -- and ``rough_area`` the area-term-only
    D_Phi; both False trace exactly the pre-grazing-fix program."""
    n_seg = (2 if split_sides else 1) * n_samples  # +1 overflow slot for drops
    gfn = None if pattern == "isotropic" else gain_fn(pattern)

    def one_trace(p, u, pv, rr, off, n_win, cb, nb, ab, e1b, e2b, phb, gfb,
                  sig, lc, kf, gf, t0, dt, c, ga, gb, tps):
        def step(carry, i):
            hist, dropped = carry
            # this trace's i-th window block (geometry.block_windows): the
            # per-trace offset makes the block fetch a batched gather
            j = off + i
            take = lambda x: jax.lax.dynamic_index_in_dim(x, j, 0, False)
            fc, fn, fa, f1, f2 = (take(cb), take(nb), take(ab), take(e1b),
                                  take(e2b))
            # per-facet gamma / phasors ride the blocked scan; the scalar
            # path (gamma_facet=False) traces exactly the old program
            fph = take(phb) if rough_terms else None
            fg = take(gfb) if gamma_facet else None
            gam = fg if gamma_facet else gf
            ts = tps if taper else None
            if rough_terms:
                contrib, r = rough_lpa_contributions(
                    p, fc, fn, fa, f1, f2, kf, gam, sig, lc, fph, rough_terms,
                    r_ref=rr, taper_s=ts, area_only=rough_area)
            else:
                contrib, r = lpa_contributions(p, fc, fn, fa, f1, f2, kf, gam,
                                               r_ref=rr, taper_s=ts)
            if gfn is not None:
                g = gfn((fc - p) / r[..., None], pv, ga, gb)
                contrib = contrib * (g * g)  # two-way FIELD gain
            twtt = 2.0 * r / c
            b = twtt_bin(twtt, t0, dt)
            if interp:
                w = (twtt - t0) / dt - b.astype(jnp.float32)
                parts = ((contrib * (1.0 - w), b), (contrib * w, b + 1))
            else:
                parts = ((contrib, b),)
            if split_sides:
                right = (jnp.sum((fc - p) * u, axis=-1) > 0).astype(jnp.int32)
            for pc, pb in parts:
                valid = (pb >= 0) & (pb < n_samples)
                pwr = jnp.real(pc) ** 2 + jnp.imag(pc) ** 2
                if split_sides:
                    pb = pb + right * n_samples
                seg = jnp.where(valid, pb, n_seg)
                h = jax.ops.segment_sum(pc, seg, num_segments=n_seg + 1)
                drop = jnp.sum(jnp.where(valid, jnp.float32(0.0), pwr))
                hist, dropped = hist + h[:n_seg], dropped + drop
            return (hist, dropped), None

        init = (jnp.zeros(n_seg, jnp.complex64), jnp.float32(0.0))
        (hist, dropped), _ = jax.lax.scan(step, init, jnp.arange(n_win))
        return hist, dropped

    return jax.jit(jax.vmap(one_trace, in_axes=(0, 0, 0, 0, 0) + (None,) * 18),
                   static_argnums=(5,))


def coherent_cluttergram(positions, u_ct, centers, normals, areas, e1, e2, *,
                         k, gamma, t0, dt, n_samples, c, split_sides=False,
                         interp_bins=False, pattern=None, roughness=None,
                         taper_s=None, d_phi_area=False, block_size=None,
                         window_cull=True):
    """Binned coherent field for every trace.

    Same conventions as ``incoherent_cluttergram`` (local-frame float inputs,
    drop-not-wrap binning, [left, right] side order); returns ``(field,
    dropped)`` with field complex64 (T, n_samples) or (T, n_samples, 2) and
    dropped float32 (T,) accumulating |contribution|**2 outside the window.
    ``interp_bins`` splits each contribution linearly between adjacent bins
    (module docstring); default False is bit-compatible with stage 2.
    ``gamma``: scalar reflection coefficient, or an (n_facets,) per-facet
    FIELD coefficient array (spatially varying reflectivity) broadcast like
    the other facet arrays -- an array of a constant matches the scalar
    bit-exactly.
    ``pattern``: None (isotropic) or an ``antenna.pattern_args`` tuple --
    per-facet fields then carry the g**2 two-way antenna weighting.
    ``roughness``: None (smooth, the default -- traces the pre-roughness
    program) or ``(sigma_m, corr_length_m, phasors, n_terms)`` with
    ``phasors`` the (n_facets,) complex per-facet speckle phasors
    (``roughness.speckle_phasors``) and ``n_terms`` the static series length
    (``roughness.n_terms_for``); see the module docstring.
    ``taper_s``/``d_phi_area``: the grazing-fix pair (config.py
    ``GrazingFixConfig``) -- the coherent off-specular taper s_eff (None =
    off) and the area-term-only D_Phi; the defaults trace exactly the
    pre-fix program.
    ``window_cull``: per-trace along-track facet windowing (geometry.py
    ``block_windows``): each trace scans only the facet blocks that can bin
    inside the fast-time window. Exact (skipped facets are silent); False
    scans every block (regression use). ``block_size`` (default
    ``geometry.auto_block_size``, ~512k f32 lanes per step) sets the window
    granularity and is otherwise a performance knob.
    """
    n = centers.shape[0]
    block_size = min(block_size or auto_block_size(len(positions), 1 << 19),
                     n)
    n_blocks = -(-n // block_size)
    pad = n_blocks * block_size - n

    # Along-track facet order + per-trace block windows (geometry.py): the
    # facet SUM is order-dependent only at complex64 ulp level; skipped
    # blocks are provably silent, so the result equals the all-facet sum.
    order, s_t, s_sorted = along_track_order(positions, centers)
    reach = (window_reach_m(t0, dt, n_samples, c) if window_cull
             else np.inf)
    off, n_win = block_windows(s_sorted, s_t, reach, block_size, n_blocks)

    def blocks(a):
        a = np.asarray(a, dtype=np.float32)[order]
        a = np.pad(a, ((0, pad),) + ((0, 0),) * (a.ndim - 1))
        return jnp.asarray(a.reshape(n_blocks, block_size, *a.shape[1:]))

    cb, nb, ab = blocks(centers), blocks(normals), blocks(areas)
    e1b, e2b = blocks(e1), blocks(e2)
    pos64 = np.asarray(positions, dtype=np.float64)
    pos = jnp.asarray(pos64.astype(np.float32))
    uct = jnp.asarray(np.asarray(u_ct, dtype=np.float32))

    # Per-trace f64 reference range (platform -> facet centroid) and its
    # constant phase, reduced mod 2pi in f64 and folded back after the scan.
    r_ref64 = np.linalg.norm(
        pos64 - np.asarray(centers, np.float64)[order].mean(0), axis=1)
    phase_ref = np.exp(-1j * ((2.0 * k * r_ref64) % TWO_PI))  # complex128 (T,)
    r_ref = jnp.asarray(r_ref64.astype(np.float32))

    kind, pv, pa, pb = pattern or ("isotropic", np.zeros((len(pos), 3)),
                                   0.0, 0.0)
    pv = jnp.asarray(np.asarray(pv, np.float32))
    pa, pb = np.asarray(pa, np.float32), np.asarray(pb, np.float32)

    if roughness is not None:
        sigma, lcorr, phasors, n_terms = roughness
        ph = np.pad(np.asarray(phasors, np.complex64)[order], (0, pad))
        phb = jnp.asarray(ph.reshape(n_blocks, block_size))
    else:
        sigma = lcorr = 0.0
        n_terms = 0
        phb = jnp.zeros((), jnp.complex64)  # unused (rough_terms == 0)

    # gamma: scalar, or an (n_facets,) per-facet FIELD coefficient array that
    # rides the blocked scan like the phasors (float32, the kernel precision:
    # an array of a constant is bit-identical to the scalar path)
    gamma_arr = np.asarray(gamma, np.float32)
    gamma_facet = gamma_arr.ndim > 0
    if gamma_facet:
        if gamma_arr.shape != (n,):
            raise ValueError(
                f"per-facet gamma shape {gamma_arr.shape} != ({n},)")
        gfb = jnp.asarray(np.pad(gamma_arr[order], (0, pad)).reshape(
            n_blocks, block_size))
        gamma = 0.0
    else:
        gfb = jnp.zeros((), jnp.float32)  # unused (gamma_facet == False)

    fn = _coherent_fn(split_sides, int(n_samples), bool(interp_bins), kind,
                      int(n_terms), gamma_facet, taper_s is not None,
                      bool(d_phi_area))
    hist, dropped = fn(pos, uct, pv, r_ref, jnp.asarray(off), n_win,
                       cb, nb, ab, e1b, e2b, phb, gfb,
                       np.float32(sigma), np.float32(lcorr),
                       np.float32(k), np.float32(gamma), np.float32(t0),
                       np.float32(dt), np.float32(c), pa, pb,
                       np.float32(0.0 if taper_s is None else taper_s))
    field = (np.asarray(hist) * phase_ref[:, None]).astype(np.complex64)
    if split_sides:
        field = field.reshape(-1, 2, n_samples).transpose(0, 2, 1)
    return field, np.asarray(dropped)
