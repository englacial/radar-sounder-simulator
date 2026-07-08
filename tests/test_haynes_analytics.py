"""Fast CI checks of the Haynes 2018 coherence-loss analytics (compare/haynes.py).

Validates the Eq. (35) series implementation against its closed-form limits
(the only independent anchors available before the M12 ensemble benchmark
compares it against simulation):

- sigma_h -> 0:  L = 0 (fully coherent).
- l -> 0:        L = 1 - exp(-(2 k sigma_h)^2)  (Haynes Eq. 85: only the
                 coherent term survives; the series sums to e^x - 1 exactly).
- l -> inf:      L -> 0 (perfectly correlated surface is a tilted mirror).
- Dimensional Eq. (35) == dimensionless Eq. (43) under sigma_lam = sigma_h/lam,
  h' = (h/lam)/(l/lam)^2.
- Series convergence: analytic total via the l=0 identity is reproduced to
  1e-12 at the largest x used in the benchmarks (x ~ 40 at sigma_h = 0.5 lam).
- Monotonicity: L increases with sigma_h, decreases with l.

Also checks the FFT Gaussian-surface generator's variance and correlation
against the C(rho) = exp(-rho^2/l^2) convention (Haynes Eq. 101).
"""

import numpy as np

from soundersim.compare.haynes import (
    coherence_loss,
    coherence_loss_dimensionless,
    fresnel_radius,
    gaussian_surface,
    mean_power,
)


def test_l_zero_correlation_limit():
    """l = 0: L = 1 - exp(-(2k sigma)^2) exactly (series = e^x - 1)."""
    lam, h = 1.0, 2000.0
    for sigma in (0.01, 0.1, 0.25, 0.5):
        x = (2.0 * (2 * np.pi / lam) * sigma) ** 2
        expected = 1.0 - np.exp(-x)
        got = coherence_loss(h, sigma, 0.0, lam)
        np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_smooth_and_correlated_limits():
    lam, h = 1.0, 2000.0
    assert coherence_loss(h, 0.0, 2.0, lam) == 0.0
    # l -> inf: exp(-4 l^2/(m lam h)) kills every term
    assert coherence_loss(h, 0.5, 1e6, lam) < 1e-200
    # mean_power endpoints: sigma=0 -> gamma^2/h^2 (4x plate)
    g = -0.281
    np.testing.assert_allclose(mean_power(h, 0.0, 2.0, lam, g),
                               g * g / h ** 2, rtol=1e-15)


def test_dimensionless_equivalence():
    """Eq. (35) == Eq. (43) under the Eq. (41)-(42) normalizations."""
    rng = np.random.default_rng(11)
    for _ in range(8):
        lam = rng.uniform(0.5, 60.0)
        h = rng.uniform(500.0, 50000.0) * lam
        l = rng.uniform(0.5, 40.0) * lam
        sigma = rng.uniform(0.01, 0.5) * lam
        a = coherence_loss(h, sigma, l, lam)
        b = coherence_loss_dimensionless(sigma / lam, (h / lam) / (l / lam) ** 2)
        np.testing.assert_allclose(a, b, rtol=1e-10)


def test_monotonicity():
    lam, h = 1.0, 8000.0
    sig = np.array([coherence_loss(h, s, 4.0, lam)
                    for s in np.linspace(0.01, 0.5, 12)])
    assert np.all(np.diff(sig) > 0)          # rougher -> more loss
    ell = np.array([coherence_loss(h, 0.25, l, lam)
                    for l in (0.5, 2.0, 8.0, 32.0)])
    assert np.all(np.diff(ell) < 0)          # more correlated -> less loss
    assert np.all((sig >= 0) & (sig <= 1)) and np.all((ell >= 0) & (ell <= 1))


def test_fresnel_radius():
    np.testing.assert_allclose(fresnel_radius(1.0, 2000.0), np.sqrt(1000.0))


def test_gaussian_surface_statistics():
    """Variance ~ 1 and correlation ~ exp(-rho^2/l^2) (FFT autocorrelation)."""
    n, dx, l = 256, 0.25, 2.0
    z = gaussian_surface(n, dx, l, np.random.default_rng(42))
    assert abs(z.mean()) < 0.05
    assert abs(z.var() - 1.0) < 0.1  # ~ (n*dx/l)^2 independent patches

    # circular autocorrelation via FFT, normalized by the measured variance
    F = np.fft.rfft2(z)
    ac = np.fft.irfft2(F * np.conj(F), s=(n, n)) / (n * n * z.var())
    for lag_px in (4, 8, 12, 16):  # 1, 2, 3, 4 lambda at dx = 0.25
        rho = lag_px * dx
        expected = np.exp(-rho ** 2 / l ** 2)
        got = 0.5 * (ac[0, lag_px] + ac[lag_px, 0])
        assert abs(got - expected) < 0.05, (rho, got, expected)
