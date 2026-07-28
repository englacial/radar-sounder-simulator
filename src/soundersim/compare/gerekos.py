"""Gerekos et al. 2023 rough-facet referees (float64 NumPy/SciPy).

Independent references for ``soundersim.roughness``:

- ``d_phi_ref``: the Eq 21 series in float64 with scipy.special.wofz (exact
  Faddeeva) and an adaptive, tail-checked term count -- the accuracy referee
  for the fixed-length JAX series.
- ``d_phi_quad``: brute-force 2-D quadrature of the Eq A8 center-difference
  integral -- validates the series algebra itself (no shared code path).
- ``mc_facet_moments``: Monte Carlo of the discretized rough-facet phase
  integral (Eq 29, the paper's Section 4.1 validation): correlated Gaussian
  surfaces via the Haynes generator, returning the ensemble total power
  <|Phi|^2> and mean field <Phi> to compare against
  |<Phi>|^2 = |Phi_smooth * e^{-sigma^2 K^2/2}|^2 and + D_Phi.
- ``rough_disk_power``: deterministic ensemble-mean power of a facet disk at
  nadir under the rough-facet model (|sum F <Phi>|^2 + sum |F|^2 D_Phi) for
  the Haynes 2018 rough-Fresnel-zone comparison (compare/haynes.py).

Conventions: monostatic, facet-local in-plane coefficients A0 = 2k(rhat.e1)/L1
etc. (the kernels' sinc arguments are L*A0/2), K = 2k cos(theta).
"""

import numpy as np
from scipy.special import gammaln, wofz

from .haynes import gaussian_surface

TWO_PI = 2.0 * np.pi
_SQRT_PI = np.sqrt(np.pi)


def _f_factor_ref(m, a0, edge, l):
    """F_A / F_B of Eqs 22-24 via the scaled combination
    e^{-x^2} erfi(x+iy) = i[e^{-x^2} - e^{-y^2} e^{2ixy} w(x+iy)] (wofz)."""
    x = a0 * l / (2.0 * np.sqrt(m))
    y = edge * np.sqrt(m) / l
    z = x + 1j * y
    t1 = -y * np.exp(-x * x) - np.exp(-y * y) * np.real(
        1j * z * np.exp(2j * x * y) * wofz(z))
    t2 = x * np.imag(wofz(x + 0j))
    return 1.0 - np.exp(-y * y) * np.cos(edge * a0) + _SQRT_PI * (t1 - t2)


def d_phi_ref(sigma, l, K, A0, B0, Lx, Ly, n_terms=None):
    """Float64 D_Phi (Eq 21) with an adaptive tail-checked term count.

    Scalar or broadcastable array parameters. ``n_terms=None`` sizes the sum
    from x = sigma^2 K^2 like haynes._series (peak + wide Poisson tail) and
    asserts the last term is negligible.
    """
    x = np.asarray(sigma, np.float64) ** 2 * np.asarray(K, np.float64) ** 2
    xm = float(np.max(x))
    if xm == 0.0:
        return np.zeros(np.broadcast(x, A0, B0, Lx, Ly).shape)
    adaptive = n_terms is None
    if adaptive:
        n_terms = int(np.ceil(xm + 20.0 * np.sqrt(xm + 1.0) + 60.0))
    m = np.arange(1, n_terms + 1, dtype=np.float64)
    sh = (1,) * np.ndim(x + A0 + B0 + Lx + Ly)
    m = m.reshape(m.shape + sh)  # leading series axis, broadcast the rest
    with np.errstate(divide="ignore"):
        logp = m * np.log(x) - gammaln(m + 1.0) - x
    terms = (np.exp(logp) * (np.float64(l) ** 4 / (m * m))
             * _f_factor_ref(m, A0, Lx, l) * _f_factor_ref(m, B0, Ly, l))
    if adaptive:
        assert np.all(np.abs(terms[-1]) <= 1e-14 * np.abs(terms).sum(axis=0)
                      + 1e-300)
    return terms.sum(axis=0)


def d_phi_quad(sigma, l, K, A0, B0, Lx, Ly):
    """Brute-force quadrature of the Eq A8 center-difference integral
    (scalar parameters; slow, test use only)."""
    from scipy.integrate import dblquad
    s2k2 = sigma * sigma * K * K

    def integrand(u2, u1):
        c = np.exp(-(u1 * u1 + u2 * u2) / (l * l))
        return ((Lx - abs(u1)) * (Ly - abs(u2)) * np.cos(A0 * u1 + B0 * u2)
                * (np.exp(-s2k2 * (1.0 - c)) - np.exp(-s2k2)))

    val, _ = dblquad(integrand, -Lx, Lx, -Ly, Ly, epsabs=1e-13, epsrel=1e-11)
    return val  # odd (sin) part vanishes by symmetry


def smooth_phase(A0, B0, Lx, Ly):
    """Smooth-facet LPA phase integral magnitude Lx*Ly*sinc(Lx A0/2)*
    sinc(Ly B0/2) (Eq 7; e^{-iD0} carried by the caller)."""
    return (Lx * Ly * np.sinc(Lx * A0 / (2.0 * np.pi))
            * np.sinc(Ly * B0 / (2.0 * np.pi)))


def mc_facet_moments(sigma, l, K, A0, B0, Lx, Ly, *, dx, n_real, rng):
    """Monte Carlo moments of the discretized rough-facet phase integral.

    Midpoint grid at spacing ``dx`` over the facet; correlated Gaussian
    surfaces from ``haynes.gaussian_surface`` on a periodic square grid at
    least 10 correlation lengths (and the facet) wide, cropped to the facet.
    Returns (mean |Phi|^2, mean Phi) over ``n_real`` realizations of
    Phi = sum dx^2 e^{i(A0 x + B0 y)} e^{-i K delta(x, y)}.
    """
    nx = max(int(round(Lx / dx)), 1)
    ny = max(int(round(Ly / dx)), 1)
    xs = (np.arange(nx) - (nx - 1) / 2.0) * dx
    ys = (np.arange(ny) - (ny - 1) / 2.0) * dx
    base = np.exp(1j * (A0 * xs[:, None] + B0 * ys[None, :]))
    n_surf = max(nx, ny, int(np.ceil(10.0 * l / dx)))
    p_sum = 0.0
    f_sum = 0.0 + 0.0j
    for _ in range(n_real):
        delta = sigma * gaussian_surface(n_surf, dx, l, rng)[:nx, :ny]
        phi = (base * np.exp(-1j * K * delta)).sum() * dx * dx
        p_sum += abs(phi) ** 2
        f_sum += phi
    return p_sum / n_real, f_sum / n_real


def facet_coeffs(theta, phi, k):
    """Monostatic in-plane LPA coefficients for a horizontal facet viewed
    from polar angle ``theta`` / azimuth ``phi``: (A0, B0, K) with
    A0 = 2k sin(theta) cos(phi), B0 = 2k sin(theta) sin(phi),
    K = 2k cos(theta)."""
    st, ct = np.sin(theta), np.cos(theta)
    return (2.0 * k * st * np.cos(phi), 2.0 * k * st * np.sin(phi),
            2.0 * k * ct)


def rough_disk_power(h, radius, d, k, gamma, sigma, l):
    """Deterministic ensemble-mean |field|^2 of a rough facet disk at nadir.

    Flat disk of square facets (spacing ``d``, centers within ``radius``)
    seen from (0, 0, h) in the soundersim normalization: returns
    (|sum F <Phi>|^2, sum |F|^2 D_Phi) with F = (k/2pi) gamma cos(theta)
    e^{-2jkr}/r^2 and per-facet A0/B0/K from the exact facet-to-platform
    geometry. Float64 throughout; the Haynes 2018 rough-Fresnel-zone referee.
    """
    n = int(np.ceil(2.0 * radius / d))
    ax = (np.arange(n) - (n - 1) / 2.0) * d
    X, Y = np.meshgrid(ax, ax)
    keep = np.hypot(X, Y).ravel() <= radius
    cx, cy = X.ravel()[keep], Y.ravel()[keep]
    r = np.sqrt(cx * cx + cy * cy + h * h)
    cos = h / r
    rx, ry = -cx / r, -cy / r  # rhat = (platform - center)/r, e1 = x, e2 = y
    A0 = 2.0 * k * rx
    B0 = 2.0 * k * ry
    K = 2.0 * k * cos
    f = 1j * (k / TWO_PI) * gamma * cos * np.exp(-2j * k * r) / (r * r)
    phi_c = (smooth_phase(A0, B0, d, d) * np.exp(-0.5 * (sigma * K) ** 2))
    dphi = d_phi_ref(sigma, l, K, A0, B0, d, d)
    return (abs((f * phi_c).sum()) ** 2, float((np.abs(f) ** 2 * dphi).sum()))
