"""Brute-force Fermat referee for the two-point refraction solve (M15).

Ground truth, not a tool. For a platform p in a medium of refractive index n1
and a target q in a medium of index n2 separated by the surface
z = surface_fn(x, y), minimize the two-media optical path

    T(xc, yc) = n1*|p - x| + n2*|x - q|,    x = (xc, yc, surface_fn(xc, yc)),

by dense float64 grid evaluation with iterative grid-zoom refinement (pure
NumPy). Method choice (documented per plan): a global coarse grid is immune to
local-minimum trapping as long as its cell resolves the surface roughness
(cell << roughness wavelength), and repeated recenter-and-shrink reaches
sub-micrometer crossing precision in a handful of passes with no derivatives;
if the running minimum lands on the box edge the box recenters WITHOUT
shrinking, so the search can walk beyond the initial guess region. Tiny scenes
only: cost is n_grid**2 surface evaluations per zoom pass.

The optical path is stationary at the minimizer, so ``opl`` is far more
accurate than the crossing point itself (second-order in the crossing error):
for km-scale geometry the crossing floors at ~1e-4 m -- below that cell size
the float64 rounding of T (~eps*T) exceeds the true T variation between
neighboring cells (~T''*cell^2) and the argmin is noise -- while opl is good
to ~1e-12 m (verified against scipy Nelder-Mead in tests/test_refraction.py).
"""

from typing import NamedTuple

import numpy as np


class FermatCrossing(NamedTuple):
    x: np.ndarray   # (3,) crossing point on the true surface
    s1: float       # |p - x|
    s2: float       # |x - q|
    opl: float      # optical path length n1*s1 + n2*s2


def fermat_crossing(p, q, surface_fn, n1, n2, *, extent=None, n_grid=241,
                    n_zoom=14, zoom_cells=3, tol=1e-7):
    """Minimize two-media travel time over the true surface (float64).

    ``surface_fn(x, y) -> z`` must accept broadcast NumPy arrays (an analytic
    or interpolated surface; sampled at grid resolution down to ``tol``).
    ``extent`` is the half-width [m] of the initial horizontal search box
    centered on the p/q horizontal midpoint; the default covers the horizontal
    separation plus a height-scaled margin (tilted/refracted crossings can
    fall outside the endpoint segment). Refinement stops once the grid cell is
    below ``tol`` m.
    """
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    cx, cy = 0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])
    if extent is None:
        zm = float(np.asarray(surface_fn(cx, cy)))
        extent = (0.75 * np.hypot(q[0] - p[0], q[1] - p[1])
                  + 0.5 * (abs(p[2] - zm) + abs(q[2] - zm)) + 25.0)
    half = float(extent)
    for _ in range(n_zoom):
        xs = np.linspace(cx - half, cx + half, n_grid)
        ys = np.linspace(cy - half, cy + half, n_grid)
        X, Y = np.meshgrid(xs, ys)
        Z = np.broadcast_to(np.asarray(surface_fn(X, Y), np.float64), X.shape)
        T = (n1 * np.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2)
             + n2 * np.sqrt((X - q[0]) ** 2 + (Y - q[1]) ** 2
                            + (Z - q[2]) ** 2))
        row, col = divmod(int(np.argmin(T)), n_grid)
        cx, cy = xs[col], ys[row]
        cell = xs[1] - xs[0]
        if row in (0, n_grid - 1) or col in (0, n_grid - 1):
            continue                        # edge hit: recenter, don't shrink
        if cell < tol:
            break
        half = zoom_cells * cell
    x = np.array([cx, cy, float(np.asarray(surface_fn(cx, cy)))])
    s1 = float(np.linalg.norm(p - x))
    s2 = float(np.linalg.norm(x - q))
    return FermatCrossing(x, s1, s2, float(n1 * s1 + n2 * s2))
