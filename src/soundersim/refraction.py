"""Two-point refraction solve against a flat local plane (stage-3 M15).

For a platform ``p`` in a medium of refractive index ``n1`` and a target ``q``
in a medium of index ``n2``, separated by the plane through ``plane_point``
with unit normal ``plane_normal``, find the crossing point x on the plane where
Snell's law holds for the p -> x -> q path -- equivalently the minimum of the
optical path n1*|p-x| + n2*|x-q| over the plane.

Reduction to 1-D: with h1, h2 the signed heights of p, q above the plane
(a = |h1|, b = |h2|) and w the in-plane vector between their feet (L = |w|),
the crossing lies in the plane spanned by p, q and the normal (the objective
is strictly convex and mirror-symmetric about that plane), at horizontal
offset t = a*tan(theta1) from the foot of p. Iterating on t directly stalls at
grazing incidence (the Snell mismatch plateaus), so we solve in sigma =
sin(theta_rare), the sine of the angle in the RARER (smaller-index) medium;
Snell's law then fixes sin(theta_dense) = (n_rare/n_dense)*sigma < 1 with no
critical-angle singularity in the iteration variable (iterating the dense-side
sine instead puts a singularity at the critical angle, and upward near-critical
solves crawl), and the horizontal-closure mismatch

    F(sigma) = a_r*tan(theta_rare) + a_d*tan(theta_dense) - L,   sigma in [0, 1)

is strictly increasing AND strictly convex with F(0) = -L <= 0 and F ->
+infinity at 1. Convexity makes safeguarded Newton (bracket + midpoint
fallback) monotone and quadratic once an iterate lands at or above the root --
which the first Newton step from anywhere below guarantees -- so a FIXED
iteration count (JAX-friendly, no data-dependent trip counts) converges even
for grazing geometry. Lengths are non-dimensionalized by a+b+L for float32
conditioning, and the returned Snell residual is recomputed from the actual
crossing geometry (not the by-construction-zero sigma-space identity), so it
is an honest convergence check.

Total internal reflection: for endpoints strictly on opposite sides of the
plane the two-point Fermat problem is strictly convex, so a Snell-stationary
crossing ALWAYS exists, for either propagation direction (downward n1->n2 or
upward n2->n1); at the solution sin(theta) in the denser medium is
automatically <= n_rare/n_dense, so genuine TIR cannot occur in the two-point
boundary problem (it re-enters in M16 when a *given* incidence direction is
chained through further interfaces). The validity mask therefore flags the
failure modes this solver can actually see: endpoints on the same side of (or
on) the plane, and a Snell residual above tolerance after the fixed budget
(degenerate geometry / non-convergence). Invalid lanes carry finite
safe-guarded values, never NaN.

Dtype follows the inputs (codebase convention, kernels/geometry.py); pass
float64 NumPy arrays with ``xp=np`` for a full-precision solve. Measured over
the M15 sweep (tests/test_refraction.py; 4096 cases, airborne 300 m-3 km +
stratospheric 14-20 km, depths 10 m-4 km, offsets to 7 km, tilts to 20 deg):
f64 residual <= 1.4e-12 by 15 iterations (default budget 25; 25-vs-80
crossing diff 1.5e-11 m); the f32 JAX solve has crossing-point error <= 4.6 cm
max / 0.3 mm median (facets are tens of meters) and optical-path error <= 5.7
mm (0.023 rad two-way at 195 MHz -- inside lambda/50 but with little margin),
while recomputing s1/s2 in f64 from the f32 crossing point (one NumPy pass;
Fermat stationarity makes the path second-order in crossing error, first-order
only in the f32 rounding of the returned coordinates) leaves <= 6.2e-4 m
(0.0025 rad) -- the recommended coherent-phase route for M16. Cost: ~58
ns/pair jitted f32 on CPU at 1e6 pairs; ~510 ns/pair for the NumPy f64 path.
"""

from typing import Any, NamedTuple

import jax.numpy as jnp


class SnellCrossing(NamedTuple):
    """Result arrays, broadcast over the input batch shape."""

    x: Any          # (..., 3) crossing point on the plane
    theta1: Any     # (...) angle from the plane normal on p's side [rad]
    theta2: Any     # (...) angle from the plane normal on q's side [rad]
    s1: Any         # (...) |p - x|
    s2: Any         # (...) |x - q|
    residual: Any   # (...) n1*sin(theta1) - n2*sin(theta2)
    valid: Any      # (...) bool; False -> masked (same-side/on-plane endpoints
                    #       or residual above tolerance), values finite not NaN


def snell_crossing(p, q, plane_point, plane_normal, n1, n2, *, n_iter=25,
                   tol=None, xp=jnp):
    """Vectorized flat-interface two-point refraction solve.

    All array arguments broadcast: p/q/plane_point/plane_normal are (..., 3)
    (e.g. one p against many q and many local planes), n1/n2 scalars or (...).
    ``plane_normal`` must be unit length. ``tol`` is the |Snell residual|
    accepted as converged (default: 1e-9 for float64, 1e-3 for float32).
    ``xp`` selects the array module (jnp inside kernels; np for the float64
    reference path) -- dtype and precision follow the inputs. The fixed
    ``n_iter`` Newton/bisection loop is unrolled (JAX-jit safe).
    """
    p = xp.asarray(p)
    q = xp.asarray(q)
    o = xp.asarray(plane_point)
    nrm = xp.asarray(plane_normal)
    n1 = xp.asarray(n1)
    n2 = xp.asarray(n2)

    h1 = xp.sum((p - o) * nrm, axis=-1)     # signed height of p above plane
    h2 = xp.sum((q - o) * nrm, axis=-1)
    dt = h1.dtype
    eps = xp.finfo(dt).eps
    tiny = xp.asarray(1e-30, dt)
    if tol is None:
        # f64 residual converges to ~1e-12; the f32 residual is a round-off
        # floor (iteration-independent), measured <= ~5e-4 at 87 deg grazing
        # over the M15 sweep -- 1e-3 sits above it and far below genuine
        # non-convergence (O(0.1)).
        tol = 1e-9 if eps < 1e-10 else 1e-3

    a, b = xp.abs(h1), xp.abs(h2)
    fp = p - h1[..., None] * nrm            # foot of p on the plane
    fq = q - h2[..., None] * nrm
    w = fq - fp
    L = xp.sqrt(xp.sum(w * w, axis=-1))
    u = w / xp.maximum(L, tiny)[..., None]  # in-plane unit vector (0 if L=0)

    # Non-dimensionalize by the geometry scale (float32 conditioning).
    s = xp.maximum(a + b + L, tiny)
    ah, bh, Lh = a / s, b / s, L / s

    # Newton on sigma = sin(theta_rare): rare-side height ar, dense-side bd,
    # sin(theta_dense) = ratio*sigma with ratio = n_rare/n_dense <= 1.
    swap = n1 > n2                          # p sits in the denser medium
    ratio = xp.minimum(n1, n2) / xp.maximum(n1, n2)
    ar = xp.where(swap, bh, ah)
    bd = xp.where(swap, ah, bh)
    one = xp.asarray(1.0, dt)
    hi0 = one - 8.0 * eps
    # Straight-line initial guess (exact for n1 == n2 and for L == 0).
    sig = xp.minimum(Lh / xp.maximum(xp.sqrt(Lh * Lh + (ah + bh) ** 2), tiny),
                     hi0)
    lo = xp.zeros_like(Lh)
    hi = hi0 * xp.ones_like(Lh)
    for _ in range(n_iter):
        v = ratio * sig
        c1 = xp.maximum((one - sig) * (one + sig), tiny)  # cos^2(theta_rare)
        c2 = xp.maximum((one - v) * (one + v), tiny)      # cos^2(theta_dense)
        rc1, rc2 = xp.sqrt(c1), xp.sqrt(c2)
        F = ar * sig / rc1 + bd * v / rc2 - Lh
        lo = xp.where(F <= 0, sig, lo)      # keep the bracket around the root
        hi = xp.where(F <= 0, hi, sig)
        Fp = ar / (c1 * rc1) + bd * ratio / (c2 * rc2)
        sn = sig - F / xp.maximum(Fp, tiny)
        ok = (sn >= lo) & (sn <= hi) & xp.isfinite(sn)
        sig = xp.where(ok, sn, 0.5 * (lo + hi))

    # Crossing geometry from the converged sigma; the residual is recomputed
    # from actual angles so non-convergence shows up (sigma-space Snell is an
    # identity and would hide it).
    c1 = xp.maximum((one - sig) * (one + sig), tiny)
    tr = ar * sig / xp.sqrt(c1)             # offset from the rare-side foot
    t = xp.clip(xp.where(swap, Lh - tr, tr), 0.0, Lh)  # ... from the foot of p
    d2 = Lh - t
    r1 = xp.maximum(xp.sqrt(ah * ah + t * t), tiny)
    r2 = xp.maximum(xp.sqrt(bh * bh + d2 * d2), tiny)
    residual = n1 * t / r1 - n2 * d2 / r2
    s1, s2 = s * r1, s * r2
    x = fp + (s * t)[..., None] * u
    theta1 = xp.arctan2(t, ah)
    theta2 = xp.arctan2(d2, bh)
    valid = (h1 * h2 < 0) & (xp.abs(residual) <= tol)
    return SnellCrossing(x, theta1, theta2, s1, s2, residual, valid)
