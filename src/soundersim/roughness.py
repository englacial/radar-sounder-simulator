"""Gerekos et al. 2023 rough rectangular facet response (sub-facet roughness).

Facet with Gaussian roughness (RMS height ``sigma`` along the normal,
isotropic Gaussian correlation length ``l``) and LPA phase coefficients
A0/B0 over in-plane edge lengths Lx/Ly (paper Eqs 4-8; the kernels' sinc
arguments are Lx*A0/2 and Ly*B0/2). The ensemble-average phase response
splits into (Radio Science 58, doi:10.1029/2022RS007594):

- coherent mean (Eq 20): the smooth LPA response times ``mean_attenuation``
  exp(-sigma^2 K^2 / 2), with K = 2 k cos(theta) for monostatic reflection in
  the facet's local medium (Eq 15);
- incoherent variance ``d_phi`` (Eq 21): D_Phi = e^{-sigma^2 K^2} *
  sum_m (sigma^2 K^2)^m/m! * (l^4/m^2) * F_A(m) * F_B(m), the series of
  Eqs 22-24. A facet's total field is then F*<Phi> + F*sqrt(D_Phi)*phi_r
  with a per-facet unit random phasor phi_r (Eqs 25-28, ``speckle_phasors``)
  and F the facet's non-phase Stratton-Chu factor.

Numerical stability: Eq 22's erfi(A_m) grows like e^{Re(A_m)^2} and is NEVER
evaluated unscaled here. With A_m = x + iy (x = A0*l/(2*sqrt(m)),
y = Lx*sqrt(m)/l > 0) the prefactor e^{-x^2} is folded in exactly through the
Faddeeva function w(z) = e^{-z^2} erfc(-iz):

    e^{-x^2} * erfi(x + iy) = i * [e^{-x^2} - e^{-y^2} e^{2ixy} w(x + iy)]
    e^{-x^2} * x * erfi(x)  = x * Im{w(x)}

where w is evaluated only in the closed upper half-plane (|w| <= 1) via the
Weideman (1994) N-term rational approximation (single polyval, no branches;
N=32 measured at 3.2e-13 max relative error vs scipy.special.wofz over
x in [-60, 60], y in [0, 100]). Every term of F_A is then O(y) or smaller.
The Poisson-like series weight is computed in log space
(exp(m log(x) - lgamma(m+1) - x), folding in the e^{-sigma^2 K^2} prefactor)
so large sigma^2 K^2 never overflows. Validated against mpmath (50 digits)
and float64 brute-force quadrature of the Eq A8 integral in
tests/test_roughness.py.

Convergence (Appendix B): absolutely convergent; ``n_terms_for`` sizes the
fixed term count from sigma^2 K^2 (10 terms cover sigma <= lam/20; ~250 at
sigma ~ lam -- overshooting is cheap, terms decay factorially).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)
_SQRT_PI = np.sqrt(np.pi)


@functools.lru_cache(maxsize=None)
def _weideman_coeffs(n):
    """Weideman (1994) rational-approximation constants (L, a) for w(z)."""
    m = 2 * n
    k = np.arange(-m + 1, m)
    big_l = np.sqrt(n / np.sqrt(2.0))
    t = big_l * np.tan(np.pi * k / (2 * m))
    f = np.concatenate([[0.0], np.exp(-t * t) * (big_l * big_l + t * t)])
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / (2 * m)
    return big_l, a[1:n + 1][::-1].copy()


def faddeeva(z, n=32):
    """Faddeeva function w(z) = e^{-z^2} erfc(-iz) for Im(z) >= 0.

    Weideman (1994) n-term rational approximation: one complex polyval, no
    branches (vmap/jit friendly). Accuracy vs scipy.special.wofz: 4e-7 (n=16),
    3e-13 (n=32, default) max relative error in the closed upper half-plane.
    dtype follows the input (complex64 in the f32 kernels).
    """
    big_l, a = _weideman_coeffs(n)
    iz = 1j * z
    d = big_l - iz
    zz = (big_l + iz) / d
    p = jnp.polyval(jnp.asarray(a), zz)
    return 2.0 * p / (d * d) + _INV_SQRT_PI / d


def mean_attenuation(sigma, K):
    """Coherent (mean-field) roughness attenuation exp(-sigma^2 K^2 / 2)
    (Eq 20). K is the phase-perturbation wavenumber (Eq 15): 2 k cos(theta)
    for monostatic reflection, k1 cos(theta1) - k2 cos(theta2) per
    transmission crossing. Exactly 1.0 at sigma = 0."""
    sk = sigma * K
    return jnp.exp(-0.5 * (sk * sk))


def _f_factor(m, a0, edge, l):
    """One F factor of Eqs 22-24 (F_A with a0=A0, edge=Lx; F_B analogous),
    via the scaled-Faddeeva combination (module docstring). ``m`` scalar."""
    sq = jnp.sqrt(m)
    x = a0 * l / (2.0 * sq)
    y = edge * sq / l
    ex2 = jnp.exp(-x * x)
    ey2 = jnp.exp(-y * y)
    w_z = faddeeva(x + 1j * y)
    w_x = faddeeva(x + 0j)
    # 2*x*y = a0*edge exactly (m-independent)
    rot = jnp.exp(1j * (a0 * edge))
    # e^{-x^2} Re{A_m erfi(A_m)} and e^{-x^2} Re{A_m} erfi(Re{A_m})
    t1 = -y * ex2 - ey2 * jnp.real(1j * (x + 1j * y) * rot * w_z)
    t2 = x * jnp.imag(w_x)
    return 1.0 - ey2 * jnp.cos(edge * a0) + _SQRT_PI * (t1 - t2)


def d_phi(sigma, l, K, A0, B0, Lx, Ly, *, n_terms, area_only=False):
    """Incoherent phase-response variance D_Phi (Eq 21), units of area^2.

    Broadcasts over facet arrays (K/A0/B0/Lx/Ly); ``sigma``/``l`` are
    typically per-interface scalars. ``n_terms`` is the STATIC series length
    (see ``n_terms_for``); the sum runs as a fixed-length ``lax.scan`` so
    memory stays O(facets). Exactly 0.0 at sigma = 0 (the log-space Poisson
    weight underflows to 0 before any multiply). Clamped at >= 0 so
    sqrt(d_phi) is always finite.

    ``area_only`` (static) keeps only the facet-AREA-scaling part of each
    term's F_A * F_B -- the (-sqrt(pi) y e^{-x^2}) pieces of the
    sqrt(pi) e^{-x^2} Re{A_m erfi(A_m)} terms of Eqs 22-24, whose product is
    F_A * F_B -> pi * (Lx Ly m / l^2) * e^{-(A0^2 + B0^2) l^2 / (4m)} --
    which turns each series term into pi l^2 Lx Ly / m * exp(-(A0^2 + B0^2)
    l^2 / (4m)): the paper's Appendix-C infinite-surface law applied per
    facet. This drops the O(1) facet-EDGE remainder (the Dawson-tail part),
    whose effective sigma0 scales as 1/(Lx*Ly) -- facet-size dependent and
    unphysically dominant at grazing incidence. With area_only,
    sigma0 = (k^2/pi) gamma^2 cos^2 D_Phi/(Lx Ly) is exactly facet-size
    invariant (the grazing-fix option; config.py ``GrazingFixConfig``).
    """
    x = (sigma * K) ** 2
    shape = jnp.broadcast_shapes(jnp.shape(x), jnp.shape(A0), jnp.shape(B0),
                                 jnp.shape(Lx), jnp.shape(Ly), jnp.shape(l))
    dt = jnp.result_type(x, A0, B0, Lx, Ly, l)
    l4 = (l * l) * (l * l)
    logx = jnp.log(jnp.where(x > 0, x, 1.0)) + jnp.where(
        x > 0, 0.0, -jnp.inf)  # log(x), -inf at x = 0 (no nan, no 0*inf)

    def body(acc, mf):
        logp = mf * logx - gammaln(mf + 1.0) - x
        if area_only:
            term = jnp.exp(logp - (A0 * A0 + B0 * B0) * (l * l) / (4.0 * mf)) \
                * (np.pi * (l * l) / mf) * Lx * Ly
        else:
            term = jnp.exp(logp) * (l4 / (mf * mf)) \
                * _f_factor(mf, A0, Lx, l) * _f_factor(mf, B0, Ly, l)
        return acc + term, None

    ms = jnp.arange(1, n_terms + 1, dtype=dt)
    acc, _ = jax.lax.scan(body, jnp.zeros(shape, dt), ms)
    return jnp.maximum(acc, 0.0)


def n_terms_for(x_max):
    """Static series length for ``d_phi`` given the run's largest
    sigma^2 K^2 (e.g. (2 k_local sigma)^2 at nadir). Terms peak near
    m ~ x and decay factorially; the margin covers a wide Poisson tail.
    Floor of 10 (the measured sigma <= lam/20 requirement), cap 300
    (sigma ~ lam, the paper's ~250-term regime)."""
    x = float(x_max)
    return int(np.clip(np.ceil(x + 6.0 * np.sqrt(x) + 5.0), 10, 300))


def speckle_phasors(n, seed):
    """Deterministic per-facet unit-variance speckle phasors (Eq 25):
    (eps1 + i eps2)/sqrt(2), eps ~ N(0,1), complex64 (n,)."""
    rng = np.random.default_rng(seed)
    ph = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return (ph / np.sqrt(2.0)).astype(np.complex64)
