"""Two-media brute-force coherent referee (stage-3 M17, pure NumPy float64).

Ground truth for the refracted-path multilayer kernel on tiny scenes, extending
``brute_force.py`` below the surface: the bed is sampled sub-wavelength, each
sample's air->ice crossing is solved EXACTLY on the true surface (per-sample
Fermat), and the contributions are summed directly in the M9 normalization with
in-medium phase on each leg:

    contrib = j*(k0*n2/2pi) * gamma_bed * cos(theta_t) * dA
              * tau_down*tau_up * flux / (L_par * L_perp)
              * exp(-2j*k0*(n1*s1 + n2*s2))

with tau_down*tau_up = 1 - gamma_TE(theta1)^2 (Stokes), flux = (n1 c1)/(n2 c2),
and (L_par, L_perp) the ``physics.refraction_spreading`` effective lengths --
all evaluated at the EXACT crossing with the TRUE local surface normal there
(vs the kernel's facet-plane chaining). The referee is exact in geometry and
phase; its amplitude uses the flat-interface ray-tube factors of the local
tangent plane at the crossing (surface-curvature focusing is outside both the
referee and the kernel -- the same first-order amplitude physics, so the
comparison isolates the kernel's local-plane/facet-anchoring approximations).

Crossing solvers:

- Flat surface (a plane): ``refraction.snell_crossing`` with float64 NumPy
  inputs IS the exact Fermat solve -- use it directly.
- Rough (analytic) surface: ``fermat_crossing_batch`` below, a vectorized
  version of ``fermat.fermat_crossing``'s grid-zoom minimization (same method:
  dense evaluation is immune to local-minimum trapping while the cell resolves
  the surface; recenter-and-shrink converges geometrically; an edge hit
  recenters without shrinking). The optical path is stationary at the
  minimizer, so opl error is second-order in the remaining crossing error
  (mm-scale crossings give sub-1e-4 m opl, phase noise << 1 mrad).

Everything is dense float64/complex128 -- keep scenes to <= ~1e6 samples.
"""

import numpy as np

from ..physics import fresnel_te

TWO_PI = 2.0 * np.pi


def surface_facets(extent, spacing, z_fn, z0=0.0):
    """Planar-frame rectangular facets of z = z0 + z_fn(x, y) (float64).

    Mirrors ``scene.build_facets`` corner math on a local cell-centered grid
    spanning ``extent`` x ``extent`` about the origin (no map projection --
    for kernel-level referee scenes). Returns a ``scene.Facets``; sub-wavelength
    ``spacing`` makes it double as the referee's sample set (centers/normals/
    areas). ``z_fn(x, y)`` must broadcast; use ``lambda x, y: 0.0*x`` for flat.
    """
    from ..scene import Facets

    n = int(round(extent / spacing))
    ax = np.linspace(-extent / 2.0, extent / 2.0, n + 1)  # cell corners
    X, Y = np.meshgrid(ax, ax)
    V = np.stack([X, Y, z0 + np.broadcast_to(np.asarray(z_fn(X, Y), np.float64),
                                             X.shape)], axis=-1)
    v00, v01, v10, v11 = V[:-1, :-1], V[:-1, 1:], V[1:, :-1], V[1:, 1:]
    e1 = ((v01 + v11) - (v00 + v10)) / 2.0
    e2 = ((v10 + v11) - (v00 + v01)) / 2.0
    centers = ((v00 + v01 + v10 + v11) / 4.0).reshape(-1, 3)
    raw_n = np.cross(e1, e2).reshape(-1, 3)
    mag = np.linalg.norm(raw_n, axis=1)
    normals = raw_n / mag[:, None]
    normals *= np.where(normals[:, 2] < 0.0, -1.0, 1.0)[:, None]
    ci, cj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cell = np.stack([ci.ravel(), cj.ravel()], axis=1)
    return Facets(centers, normals, mag, e1.reshape(-1, 3), e2.reshape(-1, 3),
                  cell, (n, n))


def fermat_crossing_batch(p, q, surface_fn, n1, n2, *, x0, half0, n_grid=13,
                          n_zoom=8, zoom_cells=3, chunk=65536):
    """Vectorized two-media Fermat solve on a true (analytic) surface.

    Minimizes n1*|p-x| + n2*|x-q| over x = (xc, yc, surface_fn(xc, yc)) for one
    platform ``p`` (3,) against many targets ``q`` (N, 3), by per-target
    grid-zoom (``fermat.fermat_crossing`` method, batched). ``x0`` (N, 2) seeds
    the horizontal search center (e.g. the mean-plane Snell solution) and
    ``half0`` (m) the initial half-width -- it must cover the true crossing's
    offset from the seed, and the initial cell 2*half0/(n_grid-1) must resolve
    the surface roughness. Returns (x (N, 3), s1, s2, opl) float64.
    """
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    x0 = np.asarray(x0, np.float64)
    offs = np.linspace(-1.0, 1.0, n_grid)
    out_x = np.empty_like(q)
    for lo in range(0, len(q), chunk):
        sl = slice(lo, min(lo + chunk, len(q)))
        qc = q[sl]
        cx, cy = x0[sl, 0].copy(), x0[sl, 1].copy()
        half = np.full(len(qc), float(half0))
        for _ in range(n_zoom):
            X = cx[:, None, None] + half[:, None, None] * offs[None, :, None]
            Y = cy[:, None, None] + half[:, None, None] * offs[None, None, :]
            X, Y = np.broadcast_arrays(X, Y)
            Z = np.broadcast_to(np.asarray(surface_fn(X, Y), np.float64),
                                X.shape)
            T = (n1 * np.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2
                              + (Z - p[2]) ** 2)
                 + n2 * np.sqrt((X - qc[:, None, None, 0]) ** 2
                                + (Y - qc[:, None, None, 1]) ** 2
                                + (Z - qc[:, None, None, 2]) ** 2))
            flat = T.reshape(len(qc), -1).argmin(axis=1)
            row, col = np.divmod(flat, n_grid)
            m = np.arange(len(qc))
            cx, cy = X[m, row, col], Y[m, row, col]
            edge = ((row == 0) | (row == n_grid - 1)
                    | (col == 0) | (col == n_grid - 1))
            cell = 2.0 * half / (n_grid - 1)
            half = np.where(edge, half, zoom_cells * cell)
        out_x[sl, 0], out_x[sl, 1] = cx, cy
        out_x[sl, 2] = np.asarray(surface_fn(cx, cy), np.float64)
    s1 = np.linalg.norm(p - out_x, axis=1)
    s2 = np.linalg.norm(out_x - q, axis=1)
    return out_x, s1, s2, n1 * s1 + n2 * s2


def local_plane_opl(p, q, surf, n1, n2):
    """Float64 replica of the kernel's two-pass local-plane crossing chain.

    Reproduces ``kernels.multilayer`` for one crossed interface, in NumPy
    float64: (1) Snell solve against the interface's area-weighted mean plane,
    (2) nearest-facet lookup at that crossing, (3) re-solve against the facet's
    local tangent plane. Returns (x (N, 3), opl (N,), facet_index (N,)) for
    platform ``p`` (3,) and targets ``q`` (N, 3). Differencing against the
    exact Fermat opl on the true surface isolates the local-plane/anchoring
    error (the M15 error channel) with no tessellation or float32 confound.
    """
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    w = (surf.areas / surf.areas.sum())[:, None]
    mp = (surf.centers * w).sum(0)
    mn = (surf.normals * w).sum(0)
    mn = mn / np.linalg.norm(mn)
    from ..refraction import snell_crossing

    r1 = snell_crossing(p, q, mp, mn, n1, n2, xp=np)
    d2 = ((r1.x[:, None, :2] - surf.centers[None, :, :2]) ** 2).sum(-1)
    idx = d2.argmin(axis=1)
    r2 = snell_crossing(p, q, surf.centers[idx], surf.normals[idx], n1, n2,
                        xp=np)
    return r2.x, n1 * r2.s1 + n2 * r2.s2, idx


def two_media_trace(p, bed, x, surf_normal, eps1, eps2, gamma_bed, k0,
                    t0, dt, n_samples, c):
    """Referee complex fast-time trace from exact crossings (complex128).

    ``bed`` is a sub-wavelength Facets set (centers/normals/areas used);
    ``x`` (N, 3) the exact crossing per bed sample and ``surf_normal`` the true
    unit surface normal there ((3,) or (N, 3)). Out-of-window contributions are
    dropped, matching the kernel binning convention. Also returns the total
    (window-independent) complex field sum.
    """
    p = np.asarray(p, np.float64)
    n1, n2 = np.sqrt(eps1), np.sqrt(eps2)
    nrm = np.broadcast_to(np.asarray(surf_normal, np.float64), x.shape)
    d1 = p - x
    s1 = np.linalg.norm(d1, axis=1)
    d2 = x - bed.centers
    s2 = np.linalg.norm(d2, axis=1)
    c1 = np.abs(np.sum(d1 * nrm, axis=1)) / s1
    c2 = np.abs(np.sum(d2 * nrm, axis=1)) / s2
    tau2 = 1.0 - fresnel_te(eps1, eps2, c1).gamma ** 2   # tau_down * tau_up
    l_par = s1 + s2 * (n1 / n2) * c1 ** 2 / c2 ** 2      # n1-normalized
    l_perp = s1 + s2 * n1 / n2
    flux = (n1 * c1) / (n2 * c2)
    cos_t = np.sum((d2 / s2[:, None]) * bed.normals, axis=1)
    opl = n1 * s1 + n2 * s2
    contrib = (1j * (k0 * n2 / TWO_PI) * gamma_bed * cos_t * bed.areas
               * tau2 * flux / (l_par * l_perp) * np.exp(-2j * k0 * opl))
    bins = np.floor((2.0 * opl / c - t0) / dt).astype(np.int64)
    ok = (bins >= 0) & (bins < n_samples)
    trace = (np.bincount(bins[ok], weights=contrib.real[ok],
                         minlength=n_samples)
             + 1j * np.bincount(bins[ok], weights=contrib.imag[ok],
                                minlength=n_samples))
    return trace, complex(contrib.sum())
