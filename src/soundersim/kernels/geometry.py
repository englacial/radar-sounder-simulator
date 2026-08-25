"""Geometry helpers shared by simulation kernels (JAX; dtype follows inputs)."""

import jax.numpy as jnp


def ranges_and_cos(position, centers, normals):
    """Range and incidence cosine from one platform position to facet centers.

    cosθ = r̂·n̂ with r̂ the unit vector from facet center to platform.
    Returns (r, cosθ), each shaped like ``centers[..., 0]``.
    """
    d = position - centers
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    cos = jnp.sum(d * normals, axis=-1) / r
    return r, cos


def twtt_bin(twtt, t0, dt):
    """Fast-time bin index: floor((twtt - t0) / dt), int32.

    Bin k spans twtt in [t0 + k*dt, t0 + (k+1)*dt). floor (not int truncation)
    so pre-window times bin negative instead of wrapping toward zero.
    """
    return jnp.floor((twtt - t0) / dt).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Per-trace along-track facet windowing (2026-08-24 runtime work, "1a").
#
# A facet at horizontal distance d from the platform has twtt >= 2 d / c on
# ANY path (air leg straight-line >= d; refracted optical paths n_i s_i >=
# straight-line >= d since every n_i >= 1), so facets with d >= c * t_end / 2
# can only bin past the window end and never reach ``hist``. Projecting on a
# horizontal axis a, |a . (c - p)| <= d, so ordering facets along the track
# and giving each trace only the block range within +-R of its own
# projection excludes provably-silent facets only: the kept sum is
# bit-identical to the all-facet sum (skipped blocks add exact zeros); only
# the ``dropped`` diagnostic (power of out-of-window contributions) shrinks.
# ---------------------------------------------------------------------------
def window_reach_m(t0, dt, n_samples, c):
    """Conservative horizontal radius beyond which nothing can bin inside
    [t0, t0 + n_samples dt): c t_end / 2 padded by two bins, a 1e-5
    relative float32 margin and 1 m."""
    t_end = float(t0) + (float(n_samples) + 2.0) * float(dt)
    return 0.5 * float(c) * t_end * (1.0 + 1e-5) + 1.0


def along_track_order(positions, centers):
    """Horizontal along-track unit axis (first -> last position; +x if the
    track is a point), the stable facet argsort along it, and the per-trace
    / per-facet projections (``s_t``, ``s_f`` in the sorted order)."""
    import numpy as np

    pos = np.asarray(positions, np.float64)
    d = pos[-1, :2] - pos[0, :2]
    nrm = np.linalg.norm(d)
    a = d / nrm if nrm > 0 else np.array([1.0, 0.0])
    s_f = np.asarray(centers, np.float64)[:, :2] @ a
    order = np.argsort(s_f, kind="stable")
    return order, pos[:, :2] @ a, s_f[order]


def block_windows(s_sorted, s_t, reach, block_size, n_blocks):
    """Per-trace first block index and the common window length so that every
    facet with |s_f - s_t| <= reach lies in blocks [off, off + n_win).
    ``s_sorted`` is the (unpadded) sorted facet projection; padding blocks
    at the end are silent (zero area) and may be included freely."""
    import numpy as np

    if not np.isfinite(reach):
        return np.zeros(len(s_t), np.int32), int(n_blocks)
    lo = np.searchsorted(s_sorted, s_t - reach, "left")
    hi = np.searchsorted(s_sorted, s_t + reach, "right")
    b_lo = lo // block_size
    b_hi = np.minimum(-(-hi // block_size), n_blocks)
    n_win = int(max(1, int((b_hi - b_lo).max())))
    off = np.minimum(b_lo, n_blocks - n_win)
    return off.astype(np.int32), n_win


def auto_block_size(n_traces, target_pairs, lo=512, hi=16384):
    """Facet block size so that one scan step holds ~``target_pairs``
    (traces x facets) lanes: the per-step temporaries then stay cache-
    resident (measured 2026-08-24: the f64 bed kernel at 200 traces runs
    1.65x faster at 1024 than at 4096; 256 is dispatch-bound). Power of two
    in [lo, hi]."""
    import numpy as np

    b = 1 << int(round(np.log2(max(target_pairs / max(int(n_traces), 1), 1.0))))
    return int(np.clip(b, lo, hi))
