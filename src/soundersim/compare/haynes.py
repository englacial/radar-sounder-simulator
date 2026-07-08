"""Haynes et al. 2018 closed-form analytics for the coherent benchmark suite.

Implements the rough-surface coherence-loss function of Haynes et al. 2018,
"Geometric Power Fall-Off in Radar Sounding" (TGRS 56(11)), Eqs. (34)-(36) and
the dimensionless form Eqs. (41)-(44), plus the Gaussian correlated-surface
generator used to reproduce the paper's numerical ensembles (Sec. IV-A,
Eqs. (29)-(31)) and the discretization noise floor (Appendix C, Eq. 115).

Conventions (from the paper's Appendix B):

- Surface heights are Gaussian, z ~ N(0, sigma_h^2), with isotropic Gaussian
  correlation C(rho) = exp(-rho^2 / l^2)  (Eq. 101).
- The ensemble mean power of the Fresnel-zone phase integral is
  <|I|^2> = I_o * (1 - L(h, sigma_h, l, lam))            (Eq. 34/107)
  with I_o = 4 pi^2 h^2 / k^2 the flat-disk value (Eq. 16) and
  L = exp(-4 sigma_h^2 k^2) * sum_{m>=1} (2 sigma_h k)^{2m} / m!
      * exp(-4 l^2 / (m lam h))                          (Eq. 35/108).
  (The 4 in the last exponential is the paper's simulation-calibrated
  constant; their analytic derivation gave 2 pi^2, revised to 4 -- see the
  discussion around Eq. 107.)
- In soundersim's field normalization (claude_notes/coherent_normalization.md)
  the same quantity is |field|^2 = (Gamma^2 / h^2) * (1 - L): the sigma_h = 0
  limit is the first-Fresnel-zone value, 4x the infinite-plate power.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def fresnel_radius(lam, h):
    """First Fresnel zone radius sqrt(lam*h/2) for a flat surface (Eq. 6)."""
    return np.sqrt(lam * h / 2.0)


def _series(x, c):
    """sum_{m>=1} x^m/m! * exp(-c/m), evaluated in log space (Eq. 109-111).

    Terms peak near m ~ x; the summation range covers the peak plus a wide
    Poisson tail, then verifies the truncated tail is negligible.
    """
    if x == 0.0:
        return 0.0
    n_terms = int(np.ceil(x + 20.0 * np.sqrt(x + 1.0) + 60.0))
    m = np.arange(1, n_terms + 1, dtype=np.float64)
    log_term = m * np.log(x) - np.cumsum(np.log(m)) - c / m
    total = float(np.exp(log_term).sum())
    # last term must be far below the total (or below absolute tiny)
    assert np.exp(log_term[-1]) <= max(1e-14 * total, 1e-300)
    return total


def coherence_loss(h, sigma_h, l, lam):
    """Coherence-loss L(h, sigma_h, l, lam) of Haynes Eq. (35)/(108), scalar.

    L = exp(-(2 k sigma_h)^2) * sum_{m>=1} (2 k sigma_h)^{2m}/m!
        * exp(-4 l^2/(m lam h)).
    Limits: sigma_h -> 0 gives 0 (fully coherent); l -> 0 gives
    1 - exp(-(2 k sigma_h)^2) (coherent term only, Eq. 85); l -> inf gives 0.
    """
    k = TWO_PI / lam
    x = (2.0 * k * sigma_h) ** 2
    return float(np.exp(-x) * _series(x, 4.0 * l * l / (lam * h)))


def coherence_loss_dimensionless(sigma_lam, h_prime):
    """Dimensionless L(sigma_lam, h') of Haynes Eq. (43).

    sigma_lam = sigma_h/lam, h' = (h/lam)/(l/lam)^2; equals
    ``coherence_loss`` under that substitution.
    """
    x = (4.0 * np.pi * sigma_lam) ** 2
    return float(np.exp(-x) * _series(x, 4.0 / h_prime))


def mean_power(h, sigma_h, l, lam, gamma):
    """Analytic ensemble-mean |field|^2 of a Fresnel-zone disk (Eq. 34/36).

    In the soundersim normalization: (gamma^2/h^2) * (1 - L). sigma_h = 0
    recovers the flat-disk value gamma^2/h^2 (4x the infinite plate).
    """
    return gamma ** 2 / h ** 2 * (1.0 - coherence_loss(h, sigma_h, l, lam))


def noise_floor_power(dx, lam, h, gamma):
    """Discretization noise floor of the ensemble mean power (Eq. 115).

    Haynes: I_n = (dx)^2 * pi * lam * h / 2 in |I|^2 units; converted to the
    soundersim field normalization via |field|^2 = (k/2pi)^2 gamma^2/h^4 |I|^2.
    """
    return gamma ** 2 / h ** 2 * np.pi * dx * dx / (2.0 * lam * h)


def gaussian_surface(n, dx, corr_len, rng):
    """(n, n) zero-mean unit-variance Gaussian surface with isotropic Gaussian
    correlation C(rho) = exp(-rho^2/corr_len^2) on a grid of spacing dx.

    Circular FFT convolution of white noise with g(rho) = exp(-rho^2/a^2),
    a = corr_len/sqrt(2) (so g*g gives the target correlation), normalized by
    sqrt(sum g^2) -- the variance of the periodic filtered field is then
    exactly 1 (no per-realization empirical normalization, which would bias
    the sharp exp(-(2 k sigma)^2) coherent term). Requires corr_len >= ~2*dx
    for the discrete kernel to resolve the Gaussian.
    """
    c = np.arange(n, dtype=np.float64)
    c = np.minimum(c, n - c) * dx  # minimal-image (circular) distance
    r2 = c[:, None] ** 2 + c[None, :] ** 2
    a = corr_len / np.sqrt(2.0)
    g = np.exp(-r2 / (a * a))
    w = rng.standard_normal((n, n))
    z = np.fft.irfft2(np.fft.rfft2(w) * np.fft.rfft2(g), s=(n, n))
    return z / np.sqrt((g * g).sum())
