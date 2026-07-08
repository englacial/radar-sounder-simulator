"""Incoherent (power-summing) clutter kernel: JAX float32, vmapped over traces.

Per facet: power = (A·cosθ)² / r⁴, twtt = 2r/c, scatter-added into fast-time
bins. Out-of-window power is dropped (never wrapped) and accumulated into a
per-trace scalar. Facets are processed in fixed-size blocks (last block padded
with zero-area facets) so memory stays bounded for large scenes.

The jitted callable is built once per ``(split_sides, n_samples)`` (lru_cache)
with the window scalars t0/dt/c traced, so repeat calls -- and calls that only
change numeric values -- reuse the compiled kernel via jit's shape cache.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import ranges_and_cos, twtt_bin


@functools.lru_cache(maxsize=None)
def _incoherent_fn(split_sides, n_samples):
    """Jitted vmapped kernel for one static configuration; run-varying
    numbers (facet blocks, positions, t0/dt/c) are traced arguments."""
    n_seg = (2 if split_sides else 1) * n_samples  # +1 overflow slot for drops

    def one_trace(p, u, cb, nb, ab, t0, dt, c):
        def step(carry, blk):
            hist, dropped = carry
            fc, fn, fa = blk
            r, cos = ranges_and_cos(p, fc, fn)
            pwr = (fa * cos) ** 2 / r ** 4
            b = twtt_bin(2.0 * r / c, t0, dt)
            valid = (b >= 0) & (b < n_samples)
            if split_sides:
                right = (jnp.sum((fc - p) * u, axis=-1) > 0).astype(jnp.int32)
                b = b + right * n_samples
            seg = jnp.where(valid, b, n_seg)
            h = jax.ops.segment_sum(pwr, seg, num_segments=n_seg + 1)
            return (hist + h[:n_seg], dropped + h[n_seg]), None

        init = (jnp.zeros(n_seg, jnp.float32), jnp.float32(0.0))
        (hist, dropped), _ = jax.lax.scan(step, init, (cb, nb, ab))
        return hist, dropped

    return jax.jit(jax.vmap(one_trace, in_axes=(0, 0) + (None,) * 6))


def incoherent_cluttergram(positions, u_ct, centers, normals, areas, *,
                           t0, dt, n_samples, c, split_sides=False,
                           block_size=65536):
    """Binned incoherent power for every trace.

    positions/u_ct: (T, 3) platform positions / cross-track (right) unit
    vectors; centers/normals/areas: (N, 3)/(N, 3)/(N,) facet arrays. All in the
    same local frame (pre-shifted near the origin, so float32 is safe).

    Returns ``(power, dropped)`` as float32 NumPy arrays: power is
    (T, n_samples), or (T, n_samples, 2) ordered [left, right] when
    ``split_sides`` (side of a facet = sign of (center − position)·u_ct).
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
    pos = jnp.asarray(np.asarray(positions, dtype=np.float32))
    uct = jnp.asarray(np.asarray(u_ct, dtype=np.float32))

    fn = _incoherent_fn(split_sides, int(n_samples))
    hist, dropped = fn(pos, uct, cb, nb, ab, np.float32(t0), np.float32(dt),
                       np.float32(c))
    hist = np.asarray(hist)
    if split_sides:
        hist = hist.reshape(-1, 2, n_samples).transpose(0, 2, 1)
    return hist, np.asarray(dropped)
