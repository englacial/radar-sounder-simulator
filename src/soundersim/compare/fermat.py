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

``fermat_path`` extends the referee to N interfaces (the D+ campaign,
tests/test_refraction_joint.py): scipy BFGS on the 2N horizontal crossing
coordinates with each point constrained to its own true surface, seeded
multistart, initialized from a sequential chain of two-point solves.
"""

from typing import NamedTuple

import numpy as np

from ..refraction import snell_crossing


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


class FermatPath(NamedTuple):
    """Multi-interface referee result (top-down)."""

    x: np.ndarray   # (N, 3) crossing points, each on its true surface
    s: np.ndarray   # (N+1,) segment lengths p -> x_0, ..., x_N-1 -> q
    opl: float      # total optical path sum_k n_k * s_k


def _surface_slopes(fn, x, y, h):
    """Central-difference surface slopes (df/dx, df/dy) at (x, y)."""
    fx = (float(fn(x + h, y)) - float(fn(x - h, y))) / (2.0 * h)
    fy = (float(fn(x, y + h)) - float(fn(x, y - h))) / (2.0 * h)
    return fx, fy


def _chain_init(p, q, fns, n, h):
    """Sequential-chain initial path: for each surface (top-down), anchor a
    tangent plane at the straight-line crossing of the current point -> q
    segment (bisection on the height mismatch), run the two-point Snell solve
    against it, and re-project the crossing onto the true surface."""
    cur = np.asarray(p, np.float64)
    out = []
    for i, fn in enumerate(fns):
        def g(t):
            r = cur + t * (q - cur)
            return r[2] - float(fn(r[0], r[1]))

        a, b, ga, gb = 0.0, 1.0, g(0.0), g(1.0)
        if ga * gb < 0:
            for _ in range(80):
                m = 0.5 * (a + b)
                if ga * g(m) <= 0:
                    b = m
                else:
                    a, ga = m, g(m)
            t = 0.5 * (a + b)
        else:                                   # degenerate: midpoint anchor
            t = 0.5
        anc = cur + t * (q - cur)
        anc[2] = float(fn(anc[0], anc[1]))
        fx, fy = _surface_slopes(fn, anc[0], anc[1], h)
        nrm = np.array([-fx, -fy, 1.0])
        nrm /= np.linalg.norm(nrm)
        r = snell_crossing(cur, q, anc, nrm, n[i], n[i + 1], xp=np)
        x = np.asarray(r.x, np.float64).copy()
        x[2] = float(fn(x[0], x[1]))
        out.append(x)
        cur = x
    return np.array(out)


def fermat_path(p, q, surface_fns, n_media, *, x0=None, n_multistart=4,
                jitter=10.0, seed=0, fd_step=1e-4, gtol=1e-12):
    """Multi-interface Fermat referee: ground truth, not a tool (float64,
    tiny scenes only).

    Minimizes the total optical path ``sum_k n_k |P_k+1 - P_k|`` over the N
    crossing points, each constrained to its own true surface
    ``z = surface_fns[i](x, y)`` (top-down; analytic/broadcastable, e.g.
    flat, tilted, gently rough), with ``n_media`` the N+1 per-medium indices.
    The 2N horizontal coordinates are the free variables (the constraint is
    exact by construction); scipy BFGS refines from the sequential-chain
    initial path (``x0`` (N, 3) overrides), with ``n_multistart`` seeded
    horizontal jitters (std ``jitter`` m) for robustness on rough surfaces.
    The gradient is analytic up to central-difference surface slopes
    (``fd_step``). Like ``fermat_crossing``, the returned ``opl`` is far more
    accurate than the crossing points (stationarity: second-order in the
    crossing error). Returns ``FermatPath``.
    """
    from scipy.optimize import minimize

    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    n = np.asarray(n_media, np.float64)
    fns = list(surface_fns)
    nn = len(fns)
    if len(n) != nn + 1:
        raise ValueError("n_media must have len(surface_fns) + 1 entries")

    def points(xy):
        xy = xy.reshape(nn, 2)
        z = np.array([float(fns[i](xy[i, 0], xy[i, 1])) for i in range(nn)])
        return np.vstack([p, np.column_stack([xy, z]), q])

    def cost(xy):
        s = np.linalg.norm(np.diff(points(xy), axis=0), axis=1)
        return float(np.dot(n, s))

    def grad(xy):
        pts = points(xy)
        d = np.diff(pts, axis=0)
        s = np.maximum(np.linalg.norm(d, axis=1), 1e-300)
        dhat = d / s[:, None]
        dg = n[:-1, None] * dhat[:-1] - n[1:, None] * dhat[1:]   # dT/dX_i
        xy2 = xy.reshape(nn, 2)
        out = np.empty((nn, 2))
        for i in range(nn):
            fx, fy = _surface_slopes(fns[i], xy2[i, 0], xy2[i, 1], fd_step)
            out[i] = (dg[i, 0] + dg[i, 2] * fx, dg[i, 1] + dg[i, 2] * fy)
        return out.ravel()

    if x0 is None:
        x0 = _chain_init(p, q, fns, n, fd_step)
    x0 = np.asarray(x0, np.float64)[:, :2].ravel()
    rng = np.random.default_rng(seed)
    best = None
    for k in range(max(1, int(n_multistart))):
        start = x0 if k == 0 else x0 + rng.normal(0.0, jitter, x0.shape)
        res = minimize(cost, start, jac=grad, method="BFGS",
                       options={"gtol": gtol, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    pts = points(best.x)
    s = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return FermatPath(pts[1:-1].copy(), s, float(np.dot(n, s)))
