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
