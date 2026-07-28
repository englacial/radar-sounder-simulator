"""Joint multi-interface refraction solve ("D+", per
claude_notes/joint_refraction_solve_note.md).

For a platform ``p`` above N local interface planes and a target ``q`` below
them, solve the full refracted path in one shot: the unknowns are the N
crossing points, the objective is the total optical path ``sum_k n_k * s_k``
over the N+1 segments, and stationarity gives Snell's law at every interface
simultaneously. This removes the sequential-chaining approximation of the
kernel path (each two-point solve treating the whole stack below as one
medium); the solution is the true stationary path of the stack, still within
the local-plane-per-interface model.

Parameterization (documented decision): TWO in-plane offsets per interface,
i.e. 2x2 Jacobian blocks. The two-point solver's 1-D reduction rests on
mirror symmetry about the plane spanned by p, q and the (single) normal; with
N arbitrarily tilted planes there is no common symmetry plane -- the plane of
incidence rotates at each crossing -- so planar confinement fails in general
and the 2-D parameterization is the general answer. (For coplanar normals the
2-D solve simply converges onto the common plane; nothing is lost.)

System and solve: with polyline points P_0 = p, P_1..P_N the crossings,
P_N+1 = q, unit segment directions d_k and B_i the (3, 2) orthonormal in-plane
basis of interface i, the stationarity residual is the tangential wavevector
mismatch

    F_i = B_i^T (n_i d_i - n_i+1 d_i+1)          (Snell + plane of incidence)

whose Jacobian w.r.t. the in-plane offsets is BLOCK-TRIDIAGONAL (crossing i
couples only to its neighbors through the shared segments):

    A_i  =  B_i^T (W_i + W_i+1) B_i,   C_i = -B_i^T W_i+1 B_i+1,
    W_k  =  n_k (I - d_k d_k^T) / s_k            (lower blocks = C_i^T).

Each Newton step solves the block system by the Thomas algorithm (forward
elimination + back substitution) as two fixed-size ``lax.scan`` s over the
interface axis, with analytic 2x2 block inverses, so the compiled graph is
O(1) in N (and in the fixed iteration budgets); runtime is O(N) per target
and everything is batched over targets by plain broadcasting -- the plane
stack is passed as arrays with a leading interface axis, never a Python loop.

Initialization and safeguarding: the initial path is the existing sequential
chain (``refraction.snell_crossing`` scanned interface by interface -- the
kernel's current approximation, imported not reimplemented). Newton is damped
by step halving on the residual norm ``|F|^2`` (the Newton direction is a
descent direction for it whenever the Jacobian is nonsingular; a candidate
that fails to decrease it -- including any NaN excursion, since NaN
comparisons are False -- is rejected and the iterate stands), plus a tiny
Levenberg shift eps**0.75 on the diagonal blocks. Budgets are FIXED
(JAX-friendly, no data-dependent trip counts). Coordinates are
non-dimensionalized by |p - q| per target for float32 conditioning.

TIR / validity: for endpoints strictly on opposite sides of every plane the
joint boundary-value problem, like the two-point one, always has a minimizer
with real angles (the optical path is coercive and smooth), so genuine
evanescence cannot occur AT a converged solution; it re-enters as a symptom
of non-convergence or degenerate stacking. A path is masked invalid when the
recomputed residual exceeds ``tol`` after the budget, when the polyline fails
to cross any plane properly (neighboring points on the same side -- shadowed
/ same-side geometry), or when ANY crossing is evanescent
(n_in sin(theta_in) > n_out, either propagation direction). Invalid lanes
carry finite safeguarded values, never NaN, matching refraction.py.

Precision: dtype follows the inputs (codebase convention); run under
``jax.enable_x64()`` with float64 inputs for the validation-grade solve, as
the multilayer kernel does for its geometry path. Measured on the M15-style
sweeps in tests/test_refraction_joint.py (x86-64 CPU, f64): N=1 agrees with
``snell_crossing`` to 2.1e-11 m crossing / 1.1e-13 rad over 512 geometries;
on flat parallel stacks the joint path matches the analytic ray-parameter
solution to <= 1.3e-13 m crossing / 2.3e-13 m optical path where the
sequential chain errs by 43/120 m crossing and 4.5/18.8 m optical path
(N=2/N=3, air-firn-ice contrasts); on tilted planar stacks it matches the
multi-interface Fermat referee to <= 1.1e-6 m crossing / 9.1e-13 m optical
path (chain: 0.94/1.96 m). The default budget (20 Newton steps, 10 halvings)
vs a doubled budget moves no crossing (0 m) across N in {2, 5, 20, 80} firn
sweeps plus high-contrast (air/ice/n=3 bedrock) and tilted cases, worst
residual 1.3e-12; jit trace+compile is flat in N (0.29 s at N=10 AND N=80,
vs the sequential kernel chain's O(N^2) growth) with runtime ~0.41
ms/target at N=80 (batch 4096, f64 CPU) -- ~5 us per target-interface.

Conditioning limit: crossings are ABSOLUTE coordinates, so when adjacent
interfaces sit closer than ~1e-9 of the scene scale (sub-mm gaps under a
20 km platform) the connecting segment's direction is lost to cancellation
and the achievable residual floors near ~1e-8; such lanes mask as
non-converged. Physically meaningful stacks (>= cm spacing) are unaffected.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .refraction import snell_crossing


class JointCrossings(NamedTuple):
    """Result arrays: leading axis = interface (top-down), then batch."""

    x: Any          # (N, ..., 3) crossing points on the local planes
    theta1: Any     # (N, ...) incoming angle from the plane normal [rad]
    theta2: Any     # (N, ...) outgoing angle from the plane normal [rad]
    s: Any          # (N+1, ...) segment lengths p->x_0, ..., x_{N-1}->q
    residual: Any   # (...) max over interfaces |tangential Snell mismatch|
    converged: Any  # (...) bool; joint: residual <= tol; chain: all two-point
                    #       solves valid
    valid: Any      # (...) bool; False -> masked (non-converged, same-side,
                    #       or evanescent), values finite not NaN


def _prep(p, q, plane_points, plane_normals, n_media):
    """Broadcast everything to (interface axis,) + batch (+ 3)."""
    p, q = jnp.asarray(p), jnp.asarray(q)
    o, nrm = jnp.asarray(plane_points), jnp.asarray(plane_normals)
    nm = jnp.asarray(n_media)
    dt = jnp.result_type(p, q, o, nrm, nm)
    batch = jnp.broadcast_shapes(p.shape[:-1], q.shape[:-1], o.shape[1:-1],
                                 nrm.shape[1:-1], nm.shape[1:])

    def mid(a, tail):
        """Broadcast (K, *b[, 3]) to (K, *batch[, 3]) (align batch left)."""
        core = a.shape[1:-1] if tail else a.shape[1:]
        tl = a.shape[-1:] if tail else ()
        a = a.reshape(a.shape[:1] + (1,) * (len(batch) - len(core)) + core
                      + tl)
        return jnp.broadcast_to(a.astype(dt), a.shape[:1] + batch + tl)

    pb = jnp.broadcast_to(p.astype(dt), batch + (3,))
    qb = jnp.broadcast_to(q.astype(dt), batch + (3,))
    return pb, qb, mid(o, True), mid(nrm, True), mid(nm, False), dt, batch


def _chain_scan(pb, qb, ob, nrmb, n_seg, n_iter):
    """Sequential two-point chain (the kernel approximation) as a lax.scan
    over the interface axis: one traced snell_crossing regardless of N."""
    def step(cur, xs):
        oi, ni, na, nb = xs
        r = snell_crossing(cur, qb, oi, ni, na, nb, n_iter=n_iter)
        return r.x, (r.x, r.valid)

    _, (x, ok) = jax.lax.scan(step, pb, (ob, nrmb, n_seg[:-1], n_seg[1:]))
    return x, jnp.all(ok, axis=0)


def _polyline(x, pb, qb, nrmb, n_seg, tol, converged=None):
    """Honest geometry of the polyline p -> x_0..x_N-1 -> q: angles, segment
    lengths, recomputed tangential-Snell residual, validity masks."""
    dt = x.dtype
    eps = jnp.finfo(dt).eps
    tiny = jnp.asarray(1e-30, dt)
    P = jnp.concatenate([pb[None], x, qb[None]], axis=0)
    d = P[1:] - P[:-1]
    s = jnp.sqrt(jnp.sum(d * d, axis=-1))
    dhat = d / jnp.maximum(s, tiny)[..., None]
    din, dout = dhat[:-1], dhat[1:]
    ct1 = jnp.sum(nrmb * din, axis=-1)
    ct2 = jnp.sum(nrmb * dout, axis=-1)
    t1 = din - ct1[..., None] * nrmb
    t2 = dout - ct2[..., None] * nrmb
    st1 = jnp.sqrt(jnp.sum(t1 * t1, axis=-1))
    st2 = jnp.sqrt(jnp.sum(t2 * t2, axis=-1))
    theta1 = jnp.arctan2(st1, jnp.abs(ct1))
    theta2 = jnp.arctan2(st2, jnp.abs(ct2))
    # Recomputed stationarity residual (not the by-construction Newton view).
    dG = n_seg[:-1, ..., None] * din - n_seg[1:, ..., None] * dout
    rt = dG - jnp.sum(nrmb * dG, axis=-1, keepdims=True) * nrmb
    residual = jnp.max(jnp.sqrt(jnp.sum(rt * rt, axis=-1)), axis=0)
    # Neighboring path points must straddle each plane (else shadow/same-side)
    hp = jnp.sum(nrmb * (P[:-2] - x), axis=-1)
    hn = jnp.sum(nrmb * (P[2:] - x), axis=-1)
    sides = jnp.all(hp * hn < 0, axis=0)
    evan = jnp.any(n_seg[:-1] * st1 > n_seg[1:] * (1.0 + 64.0 * eps), axis=0)
    fin = (jnp.all(jnp.isfinite(x), axis=(0, -1)) & jnp.isfinite(residual)
           & jnp.all(jnp.isfinite(s), axis=0))
    if converged is None:
        converged = (residual <= tol) & fin
    valid = converged & sides & ~evan & fin
    return JointCrossings(x, theta1, theta2, s, residual, converged, valid)


def sequential_chain(p, q, plane_points, plane_normals, n_media, *, n_iter=25):
    """The existing kernel approximation on plain arrays: chained two-point
    solves top-down (each treating the stack below as one medium), then the
    ACTUAL polyline geometry. Same shapes/returns as ``joint_crossings`` --
    ``residual`` here measures the chaining approximation (nonzero for N > 1
    with contrast); ``converged`` = all two-point solves valid.

    ``plane_points``/``plane_normals`` are (N, ..., 3) with a leading
    interface axis (top-down), ``n_media`` (N+1,) or (N+1, ...) per-medium
    indices; p/q/batch dims broadcast. Normals must be unit length.
    """
    pb, qb, ob, nrmb, n_seg, dt, _ = _prep(p, q, plane_points, plane_normals,
                                           n_media)
    x, ok = _chain_scan(pb, qb, ob, nrmb, n_seg, n_iter)
    return _polyline(x, pb, qb, nrmb, n_seg, None, converged=ok)


def _plane_basis(nrm, tiny):
    """Orthonormal in-plane basis: (..., 3) unit normals -> (..., 3, 2)."""
    ez = jnp.zeros_like(nrm).at[..., 2].set(1.0)
    ex = jnp.zeros_like(nrm).at[..., 0].set(1.0)
    a = jnp.where((jnp.abs(nrm[..., 2]) < 0.9)[..., None], ez, ex)
    u = jnp.cross(a, nrm)
    u = u / jnp.maximum(jnp.sqrt(jnp.sum(u * u, axis=-1, keepdims=True)),
                        tiny)
    return jnp.stack([u, jnp.cross(nrm, u)], axis=-1)


def _inv2(m, tiny):
    """Batched analytic 2x2 inverse with determinant clamped away from 0."""
    a, b = m[..., 0, 0], m[..., 0, 1]
    c, d = m[..., 1, 0], m[..., 1, 1]
    det = a * d - b * c
    det = jnp.where(jnp.abs(det) < tiny, tiny, det)
    row0 = jnp.stack([d, -b], axis=-1)
    row1 = jnp.stack([-c, a], axis=-1)
    return jnp.stack([row0, row1], axis=-2) / det[..., None, None]


def _block_thomas(A, C, b, tiny):
    """Solve the symmetric block-tridiagonal system (diag blocks A_i (2x2),
    upper blocks C_i, lower C_i^T) for b, via forward elimination + back
    substitution as two lax.scans over the interface axis. All arrays carry
    arbitrary batch dims between the interface axis and the block dims."""
    zero = jnp.zeros_like(A[:1])
    c_prev = jnp.concatenate([zero, C], axis=0)   # C_{i-1}, zero at i = 0
    c_next = jnp.concatenate([C, zero], axis=0)   # C_i, zero at i = N-1

    def fwd(carry, xs):
        dinv_p, y_p = carry
        a_i, cp_i, b_i = xs
        cpt = jnp.swapaxes(cp_i, -1, -2) @ dinv_p
        dinv_i = _inv2(a_i - cpt @ cp_i, tiny)
        y_i = b_i - jnp.einsum("...ij,...j->...i", cpt, y_p)
        return (dinv_i, y_i), (dinv_i, y_i)

    init = (jnp.zeros_like(A[0]), jnp.zeros_like(b[0]))
    _, (dinv, y) = jax.lax.scan(fwd, init, (A, c_prev, b))

    def bwd(d_next, xs):
        dinv_i, y_i, cn_i = xs
        rhs = y_i - jnp.einsum("...ij,...j->...i", cn_i, d_next)
        d_i = jnp.einsum("...ij,...j->...i", dinv_i, rhs)
        return d_i, d_i

    _, delta = jax.lax.scan(bwd, jnp.zeros_like(b[0]), (dinv, y, c_next),
                            reverse=True)
    return delta


def joint_crossings(p, q, plane_points, plane_normals, n_media, *,
                    n_newton=20, n_backtrack=10, n_iter_init=25, tol=None):
    """Joint N-interface refraction solve (module docstring for the method).

    ``plane_points``/``plane_normals``: (N, ..., 3) local-plane stacks with a
    leading interface axis (top-down); ``n_media``: (N+1,) or (N+1, ...)
    per-medium refractive indices; ``p``/``q``: (..., 3). All batch dims
    broadcast (e.g. one p, many q, per-target plane stacks). Normals must be
    unit length. ``tol`` is the accepted |tangential Snell residual|
    (dimensionless; default 1e-9 float64 / 1e-3 float32). Budgets are fixed:
    ``n_newton`` damped Newton steps (each one block-Thomas solve +
    ``n_backtrack`` step-halving candidates), ``n_iter_init`` iterations for
    the sequential-chain initializer. Returns ``JointCrossings``.
    """
    pb, qb, ob, nrmb, n_seg, dt, batch = _prep(p, q, plane_points,
                                               plane_normals, n_media)
    eps = jnp.finfo(dt).eps
    tiny = jnp.asarray(1e-30, dt)
    if tol is None:
        tol = 1e-9 if eps < 1e-10 else 1e-3

    # Initializer: the sequential chain (physical coordinates; safeguarded
    # finite even where its own steps are masked).
    x0, _ = _chain_scan(pb, qb, ob, nrmb, n_seg, n_iter_init)

    # Non-dimensionalize by |p - q| per target (float32 conditioning).
    scale = jnp.maximum(jnp.sqrt(jnp.sum((pb - qb) ** 2, axis=-1)), tiny)
    inv = (1.0 / scale)[..., None]
    ps, qs, os_ = pb * inv, qb * inv, ob * inv
    basis = _plane_basis(nrmb, tiny)                       # (N, ..., 3, 2)
    ab = jnp.einsum("...kj,...k->...j", basis, x0 * inv - os_)

    lam = 2.0 ** (-jnp.arange(n_backtrack, dtype=dt))
    mu = eps ** 0.75                                       # Levenberg shift
    i2 = jnp.eye(2, dtype=dt)
    i3 = jnp.eye(3, dtype=dt)

    def geom(ab):
        x = os_ + jnp.einsum("...kj,...j->...k", basis, ab)
        pl = jnp.concatenate([ps[None], x, qs[None]], axis=0)
        d = pl[1:] - pl[:-1]
        s = jnp.maximum(jnp.sqrt(jnp.sum(d * d, axis=-1)), tiny)
        dhat = d / s[..., None]
        g = n_seg[..., None] * dhat
        f = jnp.einsum("...kj,...k->...j", basis, g[:-1] - g[1:])
        return x, s, dhat, f

    def merit(ab):
        f = geom(ab)[3]
        return jnp.sum(f * f, axis=(0, -1))

    def newton(ab, _):
        _, s, dhat, f = geom(ab)
        w = ((n_seg / s)[..., None, None]
             * (i3 - dhat[..., :, None] * dhat[..., None, :]))
        a = (jnp.einsum("...ki,...kl,...lj->...ij", basis, w[:-1] + w[1:],
                        basis) + mu * i2)
        c = -jnp.einsum("...ki,...kl,...lj->...ij", basis[:-1], w[1:-1],
                        basis[1:])
        delta = _block_thomas(a, c, -f, tiny)
        # Step halving on |F|^2: first (largest) decreasing candidate wins;
        # none decreasing (incl. NaN candidates) -> step rejected.
        cand = ab[None] + lam.reshape((-1,) + (1,) * ab.ndim) * delta[None]
        mc = jax.vmap(merit)(cand)
        m0 = jnp.sum(f * f, axis=(0, -1))
        dec = mc < m0[None]
        first = jnp.argmax(dec, axis=0)
        lam_sel = jnp.where(jnp.any(dec, axis=0), lam[first],
                            jnp.asarray(0.0, dt))
        return ab + lam_sel[None, ..., None] * delta, None

    ab, _ = jax.lax.scan(newton, ab, None, length=n_newton)
    x = geom(ab)[0] * scale[..., None]
    return _polyline(x, pb, qb, nrmb, n_seg, tol)
