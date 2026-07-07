"""Incoherent (power-summing) clutter kernel: JAX float32, vmapped over traces.

Per facet: power = (A·cosθ)² / r⁴, twtt = 2r/c, scatter-added into fast-time
bins. Out-of-window power is dropped (never wrapped) and accumulated into a
per-trace scalar. Facets are processed in fixed-size blocks (last block padded
with zero-area facets) so memory stays bounded for large scenes.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import ranges_and_cos, twtt_bin


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
    n_seg = (2 if split_sides else 1) * n_samples  # +1 overflow slot for drops

    def one_trace(p, u):
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

    hist, dropped = jax.jit(jax.vmap(one_trace))(pos, uct)
    hist = np.asarray(hist)
    if split_sides:
        hist = hist.reshape(-1, 2, n_samples).transpose(0, 2, 1)
    return hist, np.asarray(dropped)
