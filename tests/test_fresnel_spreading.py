"""M16 physics: angle-dependent TE Fresnel coefficients and the refracted
divergence factor, validated against closed forms and independent numerics.

The divergence factor is checked two independent ways:

- finite-difference ray-tube tracing (exact forward Snell trace of
  neighboring rays) vs the L_par/L_perp formulas, and
- the image-in-dielectric closed form: a direct radial summation of the
  kernel's amplitude convention over a tapered bed annulus must reproduce
  tau_down*tau_up*Gamma_bed*exp(-2jk0(h+nd)) / (2*(h + d/n)) (Peters et al.
  2005 nadir form) -- this anchors the two-way spreading INCLUDING the
  (n0 c0)/(nj cj) flux factor and the in-medium k_j prefactor.
"""

import numpy as np
import pytest

from soundersim.physics import (fresnel_normal, fresnel_te,
                                refraction_spreading)
from soundersim.refraction import snell_crossing

N_ICE = float(np.sqrt(3.17))
C = 299792458.0


def test_fresnel_te_normal_incidence_matches_fresnel_normal():
    for e1, e2 in [(1.0, 3.17), (3.17, 6.0), (2.0, 1.5), (1.0, 80.0)]:
        r = fresnel_te(e1, e2, 1.0)
        assert r.gamma == pytest.approx(fresnel_normal(e1, e2), abs=1e-15)
        assert not r.tir


def test_fresnel_te_monotone_no_brewster():
    """TE |gamma| grows monotonically with incidence angle (no Brewster null),
    reaching 1 at grazing."""
    th = np.linspace(0.0, np.pi / 2 - 1e-6, 500)
    for e1, e2 in [(1.0, 3.17), (1.0, 6.0)]:
        g = np.abs(fresnel_te(e1, e2, np.cos(th)).gamma)
        assert (np.diff(g) > 0).all()
        assert g[-1] == pytest.approx(1.0, abs=1e-4)
        assert not np.any(np.sign(fresnel_te(e1, e2, np.cos(th)).gamma[1:])
                          != np.sign(fresnel_te(e1, e2, 1.0).gamma))


def test_fresnel_te_energy_consistency():
    """R + T = 1 in POWER with T = (n2 c2)/(n1 c1) * tau**2."""
    th = np.linspace(0.0, np.deg2rad(89.0), 200)
    for e1, e2 in [(1.0, 3.17), (3.17, 6.0), (3.17, 1.0)]:
        c1 = np.cos(th)
        r = fresnel_te(e1, e2, c1)
        ok = ~r.tir
        t_pow = (np.sqrt(e2) * r.cos_theta2) / (np.sqrt(e1) * c1) * r.tau ** 2
        np.testing.assert_allclose((r.gamma ** 2 + t_pow)[ok], 1.0, atol=1e-12)


def test_fresnel_te_two_way_reciprocity():
    """Stokes relation: tau_down(theta1) * tau_up(theta2) = 1 - gamma**2, and
    the up/down gammas are opposite."""
    th1 = np.linspace(0.0, np.deg2rad(80.0), 100)
    e1, e2 = 1.0, 3.17
    down = fresnel_te(e1, e2, np.cos(th1))
    up = fresnel_te(e2, e1, down.cos_theta2)
    assert not down.tir.any() and not up.tir.any()
    np.testing.assert_allclose(down.tau * up.tau, 1.0 - down.gamma ** 2,
                               atol=1e-12)
    np.testing.assert_allclose(up.gamma, -down.gamma, atol=1e-12)
    np.testing.assert_allclose(up.cos_theta2, np.cos(th1), atol=1e-12)


def test_fresnel_te_tir_flag_finite():
    crit = np.arcsin(1.0 / N_ICE)
    r = fresnel_te(3.17, 1.0, np.cos([crit - 0.01, crit + 0.01, 1.4]))
    assert list(r.tir) == [False, True, True]
    for v in (r.gamma, r.tau, r.cos_theta2):
        assert np.isfinite(v).all()


def _forward_trace(p, direction, z_target, n1, n2):
    """Exact forward Snell trace through the plane z=0 to depth z_target."""
    d = direction / np.linalg.norm(direction)
    t = -p[2] / d[2]
    x = p + t * d
    cos1 = -d[2]
    sin1_vec = d + cos1 * np.array([0.0, 0.0, 1.0])
    sin1 = np.linalg.norm(sin1_vec)
    sin2 = n1 * sin1 / n2
    cos2 = np.sqrt(1.0 - sin2 ** 2)
    hor = sin1_vec / sin1
    d2 = sin2 * hor - cos2 * np.array([0.0, 0.0, 1.0])
    q = x + (z_target - x[2]) / d2[2] * d2
    return x, q, t, np.linalg.norm(q - x), cos1, cos2, sin1


@pytest.mark.parametrize("theta1_deg", [10.0, 30.0, 50.0, 65.0])
def test_refraction_spreading_vs_ray_tube(theta1_deg):
    """L_par/L_perp vs finite-difference ray-tube widths from exact forward
    traces (in-plane width w_par = (c2/c1) L_par dtheta perpendicular to the
    refracted ray; out-of-plane w_perp = sin(theta1) L_perp dphi)."""
    n1, n2 = 1.0, N_ICE
    p = np.array([0.0, 0.0, 700.0])
    z_t = -400.0
    th = np.deg2rad(theta1_deg)
    d0 = np.array([np.sin(th), 0.0, -np.cos(th)])
    eps = 1e-6
    d_th = np.array([np.sin(th + eps), 0.0, -np.cos(th + eps)])
    d_ph = np.array([d0[0] * np.cos(eps), d0[0] * np.sin(eps), d0[2]])
    x0, q0, s1, s2, c1, c2, sin1 = _forward_trace(p, d0, z_t, n1, n2)
    _, qa, *_ = _forward_trace(p, d_th, z_t, n1, n2)
    _, qb, *_ = _forward_trace(p, d_ph, z_t, n1, n2)
    ray = (q0 - x0) / np.linalg.norm(q0 - x0)
    w_par = np.linalg.norm((qa - q0) - np.dot(qa - q0, ray) * ray)
    w_perp = np.linalg.norm((qb - q0) - np.dot(qb - q0, ray) * ray)
    l_par, l_perp = refraction_spreading([s1, s2], [c1, c2], [n1, n2])
    assert w_par / eps * (c1 / c2) == pytest.approx(l_par, rel=1e-4)
    assert w_perp / (sin1 * eps) == pytest.approx(l_perp, rel=1e-9)


def test_refraction_spreading_single_leg_reduces_to_range():
    l_par, l_perp = refraction_spreading([1234.5], [0.7], [1.0])
    assert l_par == pytest.approx(1234.5) and l_perp == pytest.approx(1234.5)


def test_divergence_image_method_anchor():
    """Radial direct summation of the kernel amplitude convention over a
    tapered flat bed reproduces the image-in-dielectric closed form
    tau_d*tau_u*Gamma_b*exp(-2jk0(h+nd))/(2(h+d/n)) to <1% / <0.5 deg."""
    f0 = 195e6
    k0 = 2.0 * np.pi * f0 / C
    h, d = 500.0, 60.0
    eps_ice, eps_bed = 3.17, 8.0
    n = np.sqrt(eps_ice)
    gam_b = fresnel_normal(eps_ice, eps_bed)

    drho = 0.005
    rho = np.arange(0.5 * drho, 160.0, drho)
    p = np.array([0.0, 0.0, h])
    q = np.column_stack([rho, np.zeros_like(rho), np.full_like(rho, -d)])
    r = snell_crossing(p, q, np.zeros(3), np.array([0.0, 0.0, 1.0]),
                       1.0, n, xp=np)
    assert r.valid.all()
    c1, c2 = np.cos(r.theta1), np.cos(r.theta2)
    fr = fresnel_te(1.0, eps_ice, c1)
    tau2 = 1.0 - fr.gamma ** 2                        # tau_down * tau_up
    l_par, l_perp = refraction_spreading(
        np.stack([r.s1, r.s2], -1), np.stack([c1, c2], -1), [1.0, n])
    flux = c1 / (n * c2)                              # (n0 c0)/(nj cj)
    w = np.ones_like(rho)                             # raised-cosine edge taper
    e = rho > 60.0
    w[e] = 0.5 * (1.0 + np.cos(np.pi * (rho[e] - 60.0) / 100.0))
    dA = 2.0 * np.pi * rho * drho * w
    opl = r.s1 + n * r.s2
    field = (1j * (k0 * n / (2.0 * np.pi)) * gam_b * c2 * dA * tau2 * flux
             / (l_par * l_perp) * np.exp(-2j * k0 * opl)).sum()

    tau_d, tau_u = 2.0 / (1.0 + n), 2.0 * n / (1.0 + n)
    expected = (tau_d * tau_u * gam_b * np.exp(-2j * k0 * (h + n * d))
                / (2.0 * (h + d / n)))
    ratio = field / expected
    assert abs(abs(ratio) - 1.0) < 0.01
    assert abs(np.angle(ratio, deg=True)) < 0.5
