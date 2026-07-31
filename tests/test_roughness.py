"""Rough-facet response math (Gerekos et al. 2023): roughness.py vs the
float64 referees in compare/gerekos.py, brute-force quadrature, mpmath, and
Monte Carlo (the verification-plan items a/d cores; the full sweeps live in
tools/run_rough_facet.py -> the rough_facet report case).

Thresholds set from first-run measurements (recorded inline).
"""

import jax
import numpy as np
import pytest
from scipy.special import wofz

from soundersim import roughness as rg
from soundersim.compare import gerekos as gk

LAM = 1.0
K_W = 2.0 * np.pi / LAM
LX, LY = 4.0, 7.0  # paper Section 4.1 facet (in lam units)


@pytest.fixture(scope="module", autouse=True)
def _fresh_jax_caches():
    """The f64 precision referees below are order-sensitive: a rough-branch
    kernel compiled earlier in the session (default f32) leaves a jitted
    internal trace that faddeeva/d_phi then hit even under enable_x64,
    degrading them to f32 accuracy (measured 4e-8 vs the 1e-11 gates).
    Clearing jax caches makes this module order-independent."""
    jax.clear_caches()


def _geometries():
    """(A0, B0, K) for nadir, oblique in-plane, and off-principal-axis."""
    return [gk.facet_coeffs(np.radians(th), np.radians(ph), K_W)
            for th, ph in [(0.0, 0.0), (20.0, 0.0), (35.0, 55.0)]]


# ------------------------------------------------------------------ faddeeva

def test_faddeeva_vs_scipy_wofz():
    """Weideman N=32 w(z) vs scipy.special.wofz in the closed upper
    half-plane. Measured max rel err 3.2e-13 (f64); gate 1e-11."""
    rng = np.random.default_rng(3)
    z = rng.uniform(-60, 60, 3000) + 1j * 10 ** rng.uniform(-8, 2, 3000)
    z[:300] = np.real(z[:300])  # real axis included
    with jax.enable_x64():
        got = np.asarray(rg.faddeeva(z))
    rel = np.abs(got - wofz(z)) / np.abs(wofz(z))
    assert rel.max() < 1e-11, rel.max()


# --------------------------------------------------------------- d_phi f64

def test_d_phi_f64_vs_reference():
    """JAX series (f64, n_terms_for) vs the scipy-wofz adaptive referee.
    Measured max rel err 4.5e-10 (Weideman-limited); gate 1e-8."""
    worst = 0.0
    with jax.enable_x64():
        for l in (0.5, 1.0, 2.0):
            for sig in (0.02, 0.1, 0.25, 0.5):
                for a0, b0, kk in _geometries():
                    nt = rg.n_terms_for((sig * kk) ** 2)
                    got = float(rg.d_phi(sig, l, kk, a0, b0, LX, LY,
                                         n_terms=nt))
                    ref = float(gk.d_phi_ref(sig, l, kk, a0, b0, LX, LY))
                    worst = max(worst, abs(got - ref) / abs(ref))
    assert worst < 1e-8, worst


def test_d_phi_f32_absolute():
    """f32 kernel-dtype series vs the f64 referee: absolute error scaled by
    the smooth response (Lx*Ly)^2 -- the incoherent-amplitude error is then
    sqrt of this. Measured 4.1e-8; gate 1e-6 (~ -60 dB in amplitude)."""
    worst = 0.0
    for l in (0.5, 2.0):
        for sig in (0.05, 0.25):
            for a0, b0, kk in _geometries():
                nt = rg.n_terms_for((sig * kk) ** 2)
                got = float(rg.d_phi(
                    np.float32(sig), np.float32(l), np.float32(kk),
                    np.float32(a0), np.float32(b0), np.float32(LX),
                    np.float32(LY), n_terms=nt))
                ref = float(gk.d_phi_ref(sig, l, kk, a0, b0, LX, LY))
                worst = max(worst, abs(got - ref) / (LX * LY) ** 2)
    assert worst < 1e-6, worst


def test_reference_vs_quadrature():
    """Referee series vs brute-force 2-D quadrature of the Eq A8 integral
    (independent of all erfi/Faddeeva algebra). Measured ~2e-15; gate 1e-9."""
    for sig, l in [(0.1, 1.0), (0.3, 2.0), (0.5, 0.5)]:
        a0, b0, kk = _geometries()[2]
        q = gk.d_phi_quad(sig, l, kk, a0, b0, LX, LY)
        r = float(gk.d_phi_ref(sig, l, kk, a0, b0, LX, LY))
        assert abs(q - r) / abs(r) < 1e-9, (sig, l)


def test_d_phi_vs_mpmath_and_10_terms():
    """Verification item (d): fixed-count series vs mpmath (50 digits).

    Confirms the Appendix B claim that ~10 terms suffice for sigma <=
    lam/20 (the n_terms_for floor): measured n=10 rel err <= 3e-11 at
    sigma = lam/20 across l and geometry; the f64 full-length series is
    Weideman-limited at ~1e-10.
    """
    import mpmath as mp
    mp.mp.dps = 50

    def f_mp(m, a0, edge, l):
        am = (mp.mpf(a0) * l ** 2 + 1j * 2 * edge * m) / (2 * l * mp.sqrt(m))
        x = mp.re(am)
        return (1 - mp.e ** (-(mp.mpf(edge) ** 2 * m) / l ** 2)
                * mp.cos(edge * a0)
                + mp.sqrt(mp.pi) * mp.e ** (-x ** 2)
                * (mp.re(am * mp.erfi(am)) - x * mp.erfi(x)))

    def d_phi_mp(sig, l, kk, a0, b0, n=200):
        x = mp.mpf(sig) ** 2 * mp.mpf(kk) ** 2
        tot = mp.mpf(0)
        for m in range(1, n + 1):
            tot += (x ** m / mp.factorial(m) * (mp.mpf(l) ** 4 / m ** 2)
                    * f_mp(m, a0, LX, mp.mpf(l)) * f_mp(m, b0, LY, mp.mpf(l)))
        return float(mp.e ** (-x) * tot)

    worst10 = worst_full = 0.0
    with jax.enable_x64():
        for l in (0.5, 1.0, 2.0):
            for sig in (LAM / 40.0, LAM / 20.0):
                for a0, b0, kk in _geometries():
                    ref = d_phi_mp(sig, l, kk, a0, b0, n=60)
                    ten = float(rg.d_phi(sig, l, kk, a0, b0, LX, LY,
                                         n_terms=10))
                    nt = rg.n_terms_for((sig * kk) ** 2)
                    full = float(rg.d_phi(sig, l, kk, a0, b0, LX, LY,
                                          n_terms=nt))
                    worst10 = max(worst10, abs(ten - ref) / ref)
                    worst_full = max(worst_full, abs(full - ref) / ref)
    assert worst10 < 1e-9, worst10
    assert worst_full < 1e-8, worst_full


# ------------------------------------------------------------ limits, sizing

def test_smooth_limits_exact():
    """sigma = 0 must be EXACT (the kernels' bit-identity relies on it):
    attenuation 1.0, D_Phi 0.0, in both dtypes."""
    for dt in (np.float32, np.float64):
        with jax.enable_x64(dt is np.float64):
            assert float(rg.mean_attenuation(dt(0.0), dt(12.7))) == 1.0
            assert float(rg.d_phi(dt(0.0), dt(1.0), dt(12.7), dt(0.3),
                                  dt(0.2), dt(4.0), dt(7.0), n_terms=10)) == 0.0


def test_n_terms_for():
    assert rg.n_terms_for(0.0) == 10
    assert rg.n_terms_for((4.0 * np.pi / 20.0) ** 2) == 10  # sigma = lam/20
    x_lam = (4.0 * np.pi) ** 2  # sigma = lam at nadir
    assert 200 <= rg.n_terms_for(x_lam) <= 300
    assert rg.n_terms_for(1e6) == 300


def test_speckle_phasors():
    ph = rg.speckle_phasors(20000, seed=(3, 1))
    assert ph.dtype == np.complex64
    assert np.array_equal(ph, rg.speckle_phasors(20000, seed=(3, 1)))
    assert not np.array_equal(ph, rg.speckle_phasors(20000, seed=(3, 2)))
    assert np.mean(np.abs(ph) ** 2) == pytest.approx(1.0, abs=0.03)


# ------------------------------------------------------ Haynes 2018 (c) core

def test_haynes_disk_core():
    """Verification item (c) core: rough-facet ensemble power of a nadir
    Fresnel-zone facet disk vs the Haynes 2018 closed forms (full scan in the
    rough_facet report case). Measured total residual <= 0.09 dB, coherent
    <= 0.03 dB; gates 1 / 0.5 dB."""
    from soundersim.compare import haynes
    h, d, l, gamma = 8000.0 * LAM, 4.0 * LAM, 2.0 * LAM, -0.281
    rf = haynes.fresnel_radius(LAM, h)
    for sig in (0.05, 0.15, 0.25):
        coh, inc = gk.rough_disk_power(h, rf, d, K_W, gamma, sig, l)
        tot_db = 10.0 * np.log10((coh + inc)
                                 / haynes.mean_power(h, sig, l, LAM, gamma))
        assert abs(tot_db) <= 1.0, (sig, tot_db)
        if sig <= 0.15:  # coherent part still above numerical relevance
            coh_ref = gamma ** 2 / h ** 2 * np.exp(-(2.0 * K_W * sig) ** 2)
            assert abs(10.0 * np.log10(coh / coh_ref)) <= 0.5, sig


# ------------------------------------------------------- Monte Carlo (a) core

def test_facet_monte_carlo():
    """Analytic <|Phi|^2> = |<Phi>|^2 + D_Phi vs Monte Carlo of the
    discretized rough-facet phase integral (verification item a, core; the
    full sweep is the rough_facet report case). Measured residuals <= 0.25 dB
    at N=120; gate 1 dB (paper Fig 4 agreement level)."""
    rng = np.random.default_rng(42)
    dx = LAM / 30.0
    for l, sig, gi in [(0.5, 0.05, 0), (0.5, 0.2, 1), (1.0, 0.1, 2)]:
        a0, b0, kk = _geometries()[gi]
        mc_p, _ = gk.mc_facet_moments(sig, l, kk, a0, b0, LX, LY,
                                      dx=dx, n_real=120, rng=rng)
        coh = (gk.smooth_phase(a0, b0, LX, LY)
               * np.exp(-0.5 * (sig * kk) ** 2)) ** 2
        ana = coh + float(gk.d_phi_ref(sig, l, kk, a0, b0, LX, LY))
        db = 10.0 * np.log10(mc_p / ana)
        assert abs(db) <= 1.0, (l, sig, gi, db)
