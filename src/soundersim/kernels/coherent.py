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

Phase precision (stage-2 plan constraint 2, decided by measurement): the f32
hot loop computes the carrier phase from 2k*(r - r_ref) with r_ref a per-trace
float64 reference range (platform -> facet-centroid distance), keeping the
argument small; the constant exp(-2j*k*r_ref) is folded back per trace in
complex128 outside the loop (2k*r_ref reduced mod 2pi in float64). Measured in
tests/test_coherent_kernel.py: ~1e-4 wavelengths equivalent range error at
20 km / 195 MHz, vs the lambda/50 requirement (a naive f32 exp(-2jkr) path
measures ~1e-3 wavelengths there -- also passing, but with 10x less margin).
"""

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import ranges_and_cos, twtt_bin

TWO_PI = 2.0 * np.pi


def lpa_contributions(position, centers, normals, areas, e1, e2, k, gamma,
                      r_ref=0.0, xp=jnp):
    """Per-facet complex LPA field contributions and ranges.

    Phase is computed from (r - r_ref); the caller owes a constant factor
    exp(-2j*k*r_ref) (r_ref=0 gives the absolute field). ``xp`` selects the
    array module (jnp inside the kernel; np for float64 reference use --
    dtype follows the inputs).
    """
    d = position - centers
    r = xp.sqrt(xp.sum(d * d, axis=-1))
    cos = xp.sum(d * normals, axis=-1) / r
    rhat = d / r[..., None]
    # xp.sinc is the normalized sinc sin(pi x)/(pi x); divide args by pi.
    s1 = xp.sinc(xp.sum(rhat * e1, axis=-1) * (k / np.pi))
    s2 = xp.sinc(xp.sum(rhat * e2, axis=-1) * (k / np.pi))
    amp = (k / TWO_PI) * gamma * cos * areas * s1 * s2 / (r * r)
    phase = (-2.0 * k) * (r - r_ref)  # dtype (and precision) follows the inputs
    return 1j * amp * xp.exp(1j * phase), r


def coherent_cluttergram(positions, u_ct, centers, normals, areas, e1, e2, *,
                         k, gamma, t0, dt, n_samples, c, split_sides=False,
                         block_size=65536):
    """Binned coherent field for every trace.

    Same conventions as ``incoherent_cluttergram`` (local-frame float inputs,
    drop-not-wrap binning, [left, right] side order); returns ``(field,
    dropped)`` with field complex64 (T, n_samples) or (T, n_samples, 2) and
    dropped float32 (T,) accumulating |contribution|**2 outside the window.
    """
    n = centers.shape[0]
    block_size = min(block_size, n)
    n_blocks = -(-n // block_size)
    pad = n_blocks * block_size - n

    def blocks(a):
        a = np.asarray(a, dtype=np.float32)
        a = np.pad(a, ((0, pad),) + ((0, 0),) * (a.ndim - 1))
        return jnp.asarray(a.reshape(n_blocks, block_size, *a.shape[1:]))

    cb, nb, ab = blocks(centers), blocks(normals), blocks(areas)
    e1b, e2b = blocks(e1), blocks(e2)
    pos64 = np.asarray(positions, dtype=np.float64)
    pos = jnp.asarray(pos64.astype(np.float32))
    uct = jnp.asarray(np.asarray(u_ct, dtype=np.float32))

    # Per-trace f64 reference range (platform -> facet centroid) and its
    # constant phase, reduced mod 2pi in f64 and folded back after the scan.
    r_ref64 = np.linalg.norm(pos64 - np.asarray(centers, np.float64).mean(0),
                             axis=1)
    phase_ref = np.exp(-1j * ((2.0 * k * r_ref64) % TWO_PI))  # complex128 (T,)
    r_ref = jnp.asarray(r_ref64.astype(np.float32))

    kf, gf = np.float32(k), np.float32(gamma)
    n_seg = (2 if split_sides else 1) * n_samples  # +1 overflow slot for drops

    def one_trace(p, u, rr):
        def step(carry, blk):
            hist, dropped = carry
            fc, fn, fa, f1, f2 = blk
            contrib, r = lpa_contributions(p, fc, fn, fa, f1, f2, kf, gf,
                                           r_ref=rr)
            b = twtt_bin(2.0 * r / c, t0, dt)
            valid = (b >= 0) & (b < n_samples)
            pwr = jnp.real(contrib) ** 2 + jnp.imag(contrib) ** 2
            if split_sides:
                right = (jnp.sum((fc - p) * u, axis=-1) > 0).astype(jnp.int32)
                b = b + right * n_samples
            seg = jnp.where(valid, b, n_seg)
            h = jax.ops.segment_sum(contrib, seg, num_segments=n_seg + 1)
            drop = jnp.sum(jnp.where(valid, jnp.float32(0.0), pwr))
            return (hist + h[:n_seg], dropped + drop), None

        init = (jnp.zeros(n_seg, jnp.complex64), jnp.float32(0.0))
        (hist, dropped), _ = jax.lax.scan(step, init, (cb, nb, ab, e1b, e2b))
        return hist, dropped

    hist, dropped = jax.jit(jax.vmap(one_trace))(pos, uct, r_ref)
    field = (np.asarray(hist) * phase_ref[:, None]).astype(np.complex64)
    if split_sides:
        field = field.reshape(-1, 2, n_samples).transpose(0, 2, 1)
    return field, np.asarray(dropped)
