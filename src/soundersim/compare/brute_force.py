"""Brute-force coherent reference simulator (pure NumPy, float64).

Direct physical-optics scalar summation over sub-wavelength surface samples.
This is a *referee*, not a tool: everything is materialized as dense float64 /
complex128 arrays, so keep scenes to O(1e5-1e6) samples. Its purpose is to be
trustworthy ground truth for the coherent facet kernel and for the Haynes 2018
closed-form benchmarks.

Normalization convention (adopted by the coherent kernel and benchmarks)
------------------------------------------------------------------------

    field = sum_i  (j*k / (2*pi)) * gamma * cos(theta_i) * dA_i
                   * exp(-2j*k*r_i) / r_i**2

with ``r_i`` the platform-to-sample distance and ``cos(theta_i) = r_hat . n_hat``
(``r_hat`` pointing from the sample to the platform). With this ``j*k/(2*pi)``
prefactor, the total backscattered field from an infinite flat interface at
nadir range ``h`` is the image-method result

    field_plate = gamma * exp(-2j*k*h) / (2*h)

i.e. a source reflected in the interface and seen at range ``2h``:
``|field|^2 = |gamma|^2 / (2h)^2`` matches Haynes et al. 2018, Eq. (21)
(infinite-mirror row of Table I). See claude_notes/coherent_normalization.md
for the derivation sketch.

Phase convention: ``exp(-2j*k*r)`` (phase delay grows with range), plus the
constant ``+pi/2`` from the ``j`` prefactor.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def _contributions(platform_xyz, surface_points, surface_normals, dA, k, gamma):
    """Per-sample complex field contributions and ranges (complex128/float64)."""
    p = np.asarray(platform_xyz, dtype=np.float64)
    pts = np.asarray(surface_points, dtype=np.float64)
    nrm = np.asarray(surface_normals, dtype=np.float64)
    dA = np.asarray(dA, dtype=np.float64)
    d = p - pts                                    # sample -> platform
    r = np.sqrt(np.sum(d * d, axis=-1))
    cos = np.sum(d * nrm, axis=-1) / r
    amp = (k / TWO_PI) * gamma * cos * dA / (r * r)
    contrib = 1j * amp * np.exp(-2j * k * r)
    return contrib, r


def brute_force_field(platform_xyz, surface_points, surface_normals, dA, k, gamma):
    """Total coherent field at the platform (complex scalar, complex128)."""
    contrib, _ = _contributions(platform_xyz, surface_points, surface_normals,
                                dA, k, gamma)
    return complex(contrib.sum())


def brute_force_trace(platform_xyz, surface_points, surface_normals, dA, k, gamma,
                      t0, dt, n_samples, c):
    """Complex fast-time trace: contributions binned at floor((2r/c - t0)/dt).

    Out-of-window contributions are dropped (never wrapped), matching the
    incoherent kernel's binning convention. Returns complex128 (n_samples,).
    """
    contrib, r = _contributions(platform_xyz, surface_points, surface_normals,
                                dA, k, gamma)
    bins = np.floor((2.0 * r / c - t0) / dt).astype(np.int64)
    ok = (bins >= 0) & (bins < n_samples)
    bins, contrib = bins[ok], contrib[ok]
    trace = np.bincount(bins, weights=contrib.real, minlength=n_samples).astype(
        np.float64) + 1j * np.bincount(bins, weights=contrib.imag,
                                       minlength=n_samples)
    return trace


def flat_plate_field(k, h, gamma):
    """Image-method field of an infinite flat plate at nadir range h (anchor)."""
    return gamma * np.exp(-2j * k * h) / (2.0 * h)


def flat_disk_samples(radius, spacing, taper_start=None):
    """Sample grid for a flat disk at z=0 (normals +z), cell-centered.

    Returns (points (N,3), normals (N,3), dA (N,)). ``spacing`` should be
    <= lambda/10 for reference use. If ``taper_start`` is given, dA carries a
    raised-cosine edge taper from taper_start to radius (weight 1 inside,
    smoothly to 0 at the rim) -- this suppresses the non-convergent Fresnel
    edge ringing of a hard rim so the plate result converges pointwise to the
    image-method value. taper_start=None gives a hard edge (for the
    Fresnel-oscillation checks).
    """
    n = int(np.ceil(2.0 * radius / spacing))
    ax = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * spacing
    X, Y = np.meshgrid(ax, ax)
    rho = np.hypot(X, Y).ravel()
    keep = rho <= radius
    x, y, rho = X.ravel()[keep], Y.ravel()[keep], rho[keep]
    points = np.column_stack([x, y, np.zeros_like(x)])
    normals = np.zeros_like(points)
    normals[:, 2] = 1.0
    dA = np.full(rho.shape, spacing * spacing, dtype=np.float64)
    if taper_start is not None:
        w = np.ones_like(rho)
        edge = rho > taper_start
        w[edge] = 0.5 * (1.0 + np.cos(np.pi * (rho[edge] - taper_start)
                                      / (radius - taper_start)))
        dA *= w
    return points, normals, dA


def flat_rectangle_samples(lx, ly, spacing):
    """Cell-centered sample grid for an lx-by-ly flat rectangle at z=0.

    Returns (points (N,3), normals (N,3), dA (N,)); dA is exactly
    (lx/nx)*(ly/ny) per sample so the total area is exact.
    """
    nx = max(1, int(np.ceil(lx / spacing)))
    ny = max(1, int(np.ceil(ly / spacing)))
    sx, sy = lx / nx, ly / ny
    ax = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) * sx
    ay = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * sy
    X, Y = np.meshgrid(ax, ay)
    x, y = X.ravel(), Y.ravel()
    points = np.column_stack([x, y, np.zeros_like(x)])
    normals = np.zeros_like(points)
    normals[:, 2] = 1.0
    dA = np.full(x.shape, sx * sy, dtype=np.float64)
    return points, normals, dA
