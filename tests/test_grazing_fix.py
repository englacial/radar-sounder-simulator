"""Grazing-angle facet-lattice fix acceptance (config.py GrazingFixConfig):

- facet-size INVARIANCE: with both fixes ON, the effective sigma0 of a
  statistical rough surface in the 60-85 deg band is facet-size invariant
  (measured 0.05 dB spread over a x4 spacing sweep; gate 0.3 dB), while the
  OFF path depends on facet size two ways -- the D_Phi facet-edge remainder
  makes the band sigma0 fall ~1/L (measured 4.3 dB over x4; gate >= 3 dB),
  and the coherent sinc tails on the facet-grid axis are NON-MONOTONIC in
  facet size (measured 8.8 dB swing with 7 sign flips over L = 4..6 m at
  80-81 deg; the s7 aliasing artifact, documented here);
- analytic limits: nadir GO limit of the area-only D_Phi (rough-surface
  Gaussian-slope law, correct per-axis msq slope 2 sigma^2/l^2; measured
  <= 0.08 dB at nadir, 0.7 dB at 15 deg) and the moderate-angle
  infinite-surface PO law (closed-form series; measured 3e-9 dB);
- bit-exactness: fixes OFF is the legacy program (explicit default-kwarg
  guard here; the whole pre-existing suite is the real gate), exactly-nadir
  arrivals are bit-identical with the taper ON, and sigma = 0 still zeroes
  the incoherent term with area_only;
- near-specular preservation (the S2 constraint): a wall-like facet tilted
  40 deg returns its glint unchanged (T = 1 at alpha = 0), and off-glint
  arrivals are attenuated by exactly T^2 = exp(-tan^2(alpha)/s_eff^2),
  controlled by s_eff -- through both the coherent and the refracted
  (multilayer) kernels.

Thresholds set from first-run measurements (recorded inline).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from soundersim import roughness as rg
from soundersim.config import GrazingFixConfig, SimConfig
from soundersim.kernels.coherent import (coherent_cluttergram,
                                         lpa_contributions,
                                         rough_lpa_contributions)
from soundersim.kernels.multilayer import refracted_cluttergram
from soundersim.scene import Facets

C = 299792458.0
LAM = 1.0
K = 2.0 * np.pi / LAM
H = 200.0
S_EFF = 0.05
UCT = np.array([[0.0, -1.0, 0.0]])


def _annulus(d, th0=60.0, th1=85.0, quarter=True):
    """Flat facet annulus at z = 0 seen from (0,0,H) at incidence th0..th1."""
    r0, r1 = H * np.tan(np.deg2rad(th0)), H * np.tan(np.deg2rad(th1))
    n = int(np.ceil(2 * r1 / d))
    ax = (np.arange(n) - (n - 1) / 2) * d
    X, Y = np.meshgrid(ax, ax)
    rho = np.hypot(X, Y).ravel()
    keep = (rho >= r0) & (rho <= r1)
    if quarter:
        keep &= (X.ravel() >= 0) & (Y.ravel() >= 0)
    c = np.column_stack([X.ravel()[keep], Y.ravel()[keep],
                         np.zeros(keep.sum())])
    m = len(c)
    return (c, np.tile([0.0, 0.0, 1.0], (m, 1)), np.full(m, d * d),
            np.tile([d, 0.0, 0.0], (m, 1)), np.tile([0.0, d, 0.0], (m, 1)))


def _band_sigma0_db(facets, sigma, l, n_terms, taper_s, area_only):
    """Decorrelated-sum effective sigma0 (dB, gamma = 1) of the band:
    sum_i 4 pi r^4 (|smooth_i|^2 + |incoh_i|^2) / sum_i A_i, channels
    separated via zero/unit phasors (the unit-magnitude phasor makes the
    incoherent power its exact ensemble mean)."""
    pos = np.array([0.0, 0.0, H])
    with jax.enable_x64():
        a = [jnp.asarray(x, jnp.float64) for x in facets]
        z = jnp.zeros(len(facets[0]), jnp.complex128)
        cs, r = rough_lpa_contributions(pos, *a, K, 1.0, sigma, l, z, n_terms,
                                        taper_s=taper_s, area_only=area_only)
        ct, _ = rough_lpa_contributions(pos, *a, K, 1.0, sigma, l, z + 1.0,
                                        n_terms, taper_s=taper_s,
                                        area_only=area_only)
        p = np.asarray(jnp.abs(cs) ** 2 + jnp.abs(ct - cs) ** 2)
        r = np.asarray(r)
    return 10 * np.log10((4 * np.pi * r ** 4 * p).sum() / facets[2].sum())


def test_band_sigma0_facet_size_invariance():
    """Both fixes ON: 60-85 deg band sigma0 invariant over a x4 facet-size
    sweep (measured spread 0.05 dB; gate 0.3). Fixes OFF: the same metric
    falls ~1/L (D_Phi facet-edge remainder; measured -34.3 -> -38.7 dB over
    L = 4 -> 16 m, spread 4.3 dB; gates: spread >= 3 dB, steps >= 0.5)."""
    sigma, l = 0.3 * LAM, 1.0 * LAM
    nt = rg.n_terms_for((2 * K * sigma) ** 2)
    on, off = [], []
    for d in (4.0, 8.0, 16.0):
        f = _annulus(d)
        on.append(_band_sigma0_db(f, sigma, l, nt, S_EFF, True))
        off.append(_band_sigma0_db(f, sigma, l, nt, None, False))
    on, off = np.array(on), np.array(off)
    assert on.max() - on.min() <= 0.3
    assert off.max() - off.min() >= 3.0          # facet-size dependent
    assert np.all(np.diff(off) <= -0.5)          # ~1/L, monotonic in L
    # and the OFF artifact sits far above the physical ON level
    assert off.min() - on.max() >= 5.0           # measured 7.2 dB


def test_grid_axis_sinc_tail_nonmonotonic_off_dead_on():
    """The s7 artifact channel in isolation: smooth facets ON the facet-grid
    axis at 80-81 deg. OFF, the coherent sinc-tail sigma0 is non-monotonic
    in facet size (measured 8.8 dB swing, 7 sign flips over L = 4..6 m);
    ON, the taper removes it entirely (underflows to exactly 0 here)."""
    vals, p_on = [], []
    for d in np.arange(4.0, 6.01, 0.25):
        r0, r1 = H * np.tan(np.deg2rad(80)), H * np.tan(np.deg2rad(81))
        xs = np.arange(r0, r1, d)
        m = len(xs)
        f = (np.column_stack([xs, np.zeros(m), np.zeros(m)]),
             np.tile([0.0, 0.0, 1.0], (m, 1)), np.full(m, d * d),
             np.tile([d, 0.0, 0.0], (m, 1)), np.tile([0.0, d, 0.0], (m, 1)))
        pos = np.array([0.0, 0.0, H])
        c_off, r = lpa_contributions(pos, *f, K, 1.0, xp=np)
        c_on, _ = lpa_contributions(pos, *f, K, 1.0, xp=np, taper_s=S_EFF)
        vals.append(10 * np.log10(
            (4 * np.pi * r ** 4 * np.abs(c_off) ** 2).sum() / (m * d * d)))
        p_on.append(np.abs(c_on).sum())
    vals = np.array(vals)
    assert vals.max() - vals.min() >= 6.0
    assert (np.diff(np.sign(np.diff(vals))) != 0).sum() >= 3  # non-monotonic
    assert max(p_on) == 0.0  # taper kills the tails outright at 80 deg


def test_area_only_dphi_nadir_go_limit():
    """s4_gocheck geometry (sigma 3 m, l 30 m, 60 MHz, L 500 m): the
    area-only D_Phi sigma0 matches the geometric-optics Gaussian-slope law
    (per-axis msq slope sx^2 = 2 sigma^2/l^2: sigma0 =
    exp(-tan^2/(2 sx^2)) / (2 sx^2 cos^4)) near nadir. Measured 0.08 dB at
    0 deg, 0.36 at 10, 0.64 at 15; gates 0.3 / 1.0 dB."""
    fc, sigma, l, L = 60e6, 3.0, 30.0, 500.0
    k = 2 * np.pi * fc / C
    nt = rg.n_terms_for((2 * k * sigma) ** 2)
    sx2 = 2 * sigma * sigma / (l * l)
    for deg, gate in ((0.0, 0.3), (5.0, 0.5), (10.0, 1.0), (15.0, 1.0)):
        th = math.radians(deg)
        with jax.enable_x64():
            dp = float(rg.d_phi(sigma, l, 2 * k * math.cos(th),
                                2 * k * math.sin(th), 0.0, L, L,
                                n_terms=nt, area_only=True))
        s0 = (k ** 2 / np.pi) * math.cos(th) ** 2 * dp / (L * L)
        go = math.exp(-math.tan(th) ** 2 / (2 * sx2)) \
            / (2 * sx2 * math.cos(th) ** 4)
        assert abs(10 * math.log10(s0 / go)) <= gate, deg


def test_area_only_dphi_matches_infinite_surface_po_law():
    """Area-only D_Phi / (Lx Ly) IS the closed-form infinite-surface scalar
    PO law (Appendix C) at every angle -- facet size enters only through the
    trivial area factor. Measured <= 3e-9 dB (195 MHz); gate 1e-6."""
    sigma, l = 0.049474, 2.982179  # the campaign surface roughness
    th = np.deg2rad(np.array([0.0, 20.0, 40.0, 60.0, 75.0, 84.0]))
    for fc in (60e6, 195e6):
        k = 2 * np.pi * fc / C
        nt = rg.n_terms_for((2 * k * sigma) ** 2)
        with jax.enable_x64():
            dp = np.array(rg.d_phi(sigma, l, 2 * k * np.cos(th),
                                   2 * k * np.sin(th), np.zeros_like(th),
                                   40.0, 40.0, n_terms=nt, area_only=True))
        s0 = (k ** 2 / np.pi) * np.cos(th) ** 2 * dp / 1600.0
        g = (2 * k * sigma * np.cos(th)) ** 2
        acc = np.zeros_like(th)
        for m in range(1, 400):
            logw = m * np.log(g) - math.lgamma(m + 1) - g
            acc += np.exp(logw) * (l ** 2 / m) \
                * np.exp(-(k * l * np.sin(th)) ** 2 / m)
        po = k ** 2 * np.cos(th) ** 2 * acc
        assert np.max(np.abs(10 * np.log10(s0 / po))) <= 1e-6


# ------------------------------------------------- bit-exactness / OFF path

def _disk(d, radius):
    n = int(np.ceil(2.0 * radius / d))
    ax = (np.arange(n) - (n - 1) / 2.0) * d
    X, Y = np.meshgrid(ax, ax)
    keep = np.hypot(X, Y).ravel() <= radius
    c = np.column_stack([X.ravel()[keep], Y.ravel()[keep],
                         np.zeros(keep.sum())])
    m = len(c)
    return (c, np.tile([0.0, 0.0, 1.0], (m, 1)), np.full(m, d * d),
            np.tile([d, 0.0, 0.0], (m, 1)), np.tile([0.0, d, 0.0], (m, 1)))


def _disk_kw(h, radius):
    t0 = 2.0 * (h - 2.0) / C
    ns = int(np.ceil((np.sqrt(h * h + radius * radius) - h + 4.0) / 2.0)) + 3
    return dict(k=K, gamma=-0.281, t0=t0, dt=4.0 / C, n_samples=ns, c=C)


def test_fixes_off_is_the_default_legacy_call():
    """Explicit taper_s=None + d_phi_area=False is bit-identical to the bare
    legacy call (guards the defaults; the pre-existing suite is the real
    OFF-path regression gate)."""
    h = 2000.0
    disk = _disk(4.0 * LAM, 40.0 * LAM)
    kw = _disk_kw(h, 40.0 * LAM)
    legacy, d0 = coherent_cluttergram(np.array([[0.0, 0.0, h]]), UCT, *disk,
                                      **kw)
    off, d1 = coherent_cluttergram(np.array([[0.0, 0.0, h]]), UCT, *disk,
                                   taper_s=None, d_phi_area=False, **kw)
    assert np.array_equal(legacy, off) and np.array_equal(d0, d1)


def test_taper_exact_one_at_exact_normal_and_sigma0_still_zero():
    """alpha = 0 exactly -> T = exp(-0) = 1.0 exactly (bit-identical single
    facet), and sigma = 0 through the rough path with BOTH fixes on is still
    bit-identical to the smooth tapered program (area-only D_Phi is exactly
    0 at sigma = 0)."""
    f = [np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]),
         np.array([4.0]), np.array([[2.0, 0.0, 0.0]]),
         np.array([[0.0, 2.0, 0.0]])]
    pos = np.array([0.0, 0.0, 1500.0])
    c_off, _ = lpa_contributions(pos, *f, K, 1.0, xp=np)
    c_on, _ = lpa_contributions(pos, *f, K, 1.0, xp=np, taper_s=S_EFF)
    assert np.array_equal(c_off, c_on)
    h = 2000.0
    disk = _disk(4.0 * LAM, 40.0 * LAM)
    kw = _disk_kw(h, 40.0 * LAM)
    ph = rg.speckle_phasors(len(disk[0]), seed=(1, 0))
    a, _ = coherent_cluttergram(np.array([[0.0, 0.0, h]]), UCT, *disk,
                                taper_s=S_EFF, d_phi_area=True, **kw)
    b, _ = coherent_cluttergram(np.array([[0.0, 0.0, h]]), UCT, *disk,
                                roughness=(0.0, 2.0, ph, 10), taper_s=S_EFF,
                                d_phi_area=True, **kw)
    assert np.array_equal(a, b)


def test_config_validation():
    with pytest.raises(ValueError, match="s_eff"):
        GrazingFixConfig(s_eff=0.0)
    with pytest.raises(ValueError, match="coherent"):
        SimConfig(mode="incoherent", grazing_fix=GrazingFixConfig(),
                  radar=dict(dt=1e-8, n_samples=16, t0=0.0), facets={})
    cfg = SimConfig(mode="coherent", grazing_fix=GrazingFixConfig(s_eff=0.1),
                    radar=dict(dt=1e-8, n_samples=16, t0=0.0, f0=60e6),
                    facets={})
    assert cfg.grazing_fix.s_eff == 0.1


# ------------------------------------- near-specular preservation (S2 gate)

def test_wall_glint_survives_and_off_glint_level_is_set_by_s_eff():
    """A wall-like facet tilted 40 deg (a 30-55 deg off-nadir valley-wall
    return): the glint (arrival along the facet normal) is bit-identical
    with the taper ON, and arrivals alpha off the normal are attenuated by
    exactly T^2 = exp(-tan^2(alpha)/s_eff^2) -- the survival level is
    CONTROLLED by s_eff (measured = predicted to 3 decimals; gate 0.02 dB):
    s_eff 0.05 -> -2.1 dB at 2 deg, -13.3 at 5; s_eff 0.10 -> -0.5 / -3.3."""
    psi = np.deg2rad(40.0)
    d = 8.0
    f = [np.zeros((1, 3)), np.array([[np.sin(psi), 0.0, np.cos(psi)]]),
         np.array([d * d]),
         d * np.array([[np.cos(psi), 0.0, -np.sin(psi)]]),
         d * np.array([[0.0, 1.0, 0.0]])]
    for s_eff in (0.05, 0.10):
        for a_deg in (0.0, 2.0, 5.0):
            al = np.deg2rad(a_deg)
            pos = 3000.0 * np.array([np.sin(psi + al), 0.0, np.cos(psi + al)])
            c_off, _ = lpa_contributions(pos, *f, K, 1.0, xp=np)
            c_on, _ = lpa_contributions(pos, *f, K, 1.0, xp=np,
                                        taper_s=s_eff)
            if a_deg == 0.0:
                assert np.array_equal(c_off, c_on)
                continue
            got = np.abs(c_on[0]) ** 2 / np.abs(c_off[0]) ** 2
            pred = np.exp(-np.tan(al) ** 2 / s_eff ** 2)
            assert abs(10 * np.log10(got / pred)) <= 0.02


def _flat_grid(z, extent, d):
    n = int(round(extent / d))
    ax = (np.arange(n) - (n - 1) / 2.0) * d
    X, Y = np.meshgrid(ax, ax)
    c = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    m = len(c)
    rows, cols = np.divmod(np.arange(m), n)
    return Facets(c, np.tile([0.0, 0.0, 1.0], (m, 1)), np.full(m, d * d),
                  np.tile([d, 0.0, 0.0], (m, 1)),
                  np.tile([0.0, d, 0.0], (m, 1)),
                  np.column_stack([rows, cols]), (n, n))


def test_multilayer_taper_preserves_nadir_bed_and_rejects_incoherent():
    """Refracted-path wiring: a flat buried bed's total coherent (mirror)
    field is preserved with the taper ON -- the specular return comes from
    the first Fresnel zone (alpha << s_eff), so the total-field change is
    small and SHRINKS as s_eff grows (measured +0.41 / +0.19 / +0.06 dB at
    s_eff 0.05 / 0.1 / 0.2 -- slightly positive because the taper removes
    the truncated-aperture edge ringing; gate 0.6 dB, monotone). The
    per-BIN distribution legitimately changes (a 4 m range bin spans
    arrivals to 3.6 deg off-normal). The fix refuses incoherent mode."""
    h, zs, zb, d = 1000.0, 0.0, -300.0, 5.0
    surf, bed = _flat_grid(zs, 400.0, d), _flat_grid(zb, 400.0, d)
    eps = [1.0, 3.17]
    n_ice = math.sqrt(3.17)
    t0 = 2.0 * (h + n_ice * 300.0 - 10.0) / C
    kw = dict(mode="coherent", t0=t0, dt=4.0 / C, n_samples=40, c=C,
              gamma=-0.281, k0=K)
    pos, u = np.array([[0.0, 0.0, h]]), UCT
    off, _ = refracted_cluttergram(pos, u, bed, [surf], eps, [0.0, 0.0], **kw)
    prev = np.inf
    for s_eff in (S_EFF, 0.1, 0.2):
        on, _ = refracted_cluttergram(pos, u, bed, [surf], eps, [0.0, 0.0],
                                      taper_s=s_eff, d_phi_area=True, **kw)
        r = abs(20 * np.log10(np.abs(on[0].sum()) / np.abs(off[0].sum())))
        assert r <= 0.6 and r <= prev + 1e-6
        prev = r
    with pytest.raises(ValueError, match="coherent"):
        refracted_cluttergram(pos, u, bed, [surf], eps, [0.0, 0.0],
                              mode="incoherent", t0=t0, dt=4.0 / C,
                              n_samples=40, c=C, taper_s=S_EFF)
