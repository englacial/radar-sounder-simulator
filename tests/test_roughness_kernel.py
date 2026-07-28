"""Rough-facet kernel integration (verification-plan item b + kernel wiring):

- smooth-limit regressions: roughness=None AND sigma=0 bit-identical to the
  pre-roughness kernels (coherent + multilayer);
- coherent-kernel ensemble mean power vs the float64 rough-disk referee
  (|sum F <Phi>|^2 + sum |F|^2 D_Phi), and the coherent part in isolation
  via zero phasors;
- multilayer wiring: eps -> 1 reduction of the rough bed to the
  single-interface rough kernel (validates the incoherent term's F factor
  through the refracted path), buried-facet coherent attenuation with the
  LOCAL-medium wavenumber, and the two-way transmission attenuation
  exp(-2 sigma^2 K_t^2) through simulate();
- config validation and simulate() plumbing/determinism.

Thresholds set from first-run measurements (recorded inline).
"""

import numpy as np
import pytest

import soundersim
from soundersim import roughness as rg
from soundersim import synthetic as syn
from soundersim.compare import gerekos as gk
from soundersim.config import (DemInterface, FacetConfig, Medium, RadarConfig,
                               RoughnessConfig, SimConfig)
from soundersim.kernels.coherent import coherent_cluttergram
from soundersim.kernels.multilayer import refracted_cluttergram
from soundersim.scene import Facets

C = 299792458.0
UCT = np.array([[0.0, -1.0, 0.0]])
GAMMA = -0.281


def _disk_facets(d, radius):
    """Flat facet disk at z = 0 (spacing d, centers within radius)."""
    n = int(np.ceil(2.0 * radius / d))
    ax = (np.arange(n) - (n - 1) / 2.0) * d
    X, Y = np.meshgrid(ax, ax)
    keep = np.hypot(X, Y).ravel() <= radius
    centers = np.column_stack([X.ravel()[keep], Y.ravel()[keep],
                               np.zeros(keep.sum())])
    m = len(centers)
    return (centers, np.tile([0.0, 0.0, 1.0], (m, 1)), np.full(m, d * d),
            np.tile([d, 0.0, 0.0], (m, 1)), np.tile([0.0, d, 0.0], (m, 1)))


def _flat_grid_facets(z, extent, d):
    """Full flat Facets grid at height z (for the multilayer kernel)."""
    n = int(round(extent / d))
    ax = (np.arange(n) - (n - 1) / 2.0) * d
    X, Y = np.meshgrid(ax, ax)
    centers = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    m = len(centers)
    rows, cols = np.divmod(np.arange(m), n)
    return Facets(centers, np.tile([0.0, 0.0, 1.0], (m, 1)),
                  np.full(m, d * d), np.tile([d, 0.0, 0.0], (m, 1)),
                  np.tile([0.0, d, 0.0], (m, 1)),
                  np.column_stack([rows, cols]), (n, n))


# ------------------------------------------------------------- coherent kernel

LAM = 1.0
K_W = 2.0 * np.pi / LAM
H = 2000.0


def _disk_kw(h, radius):
    t0 = 2.0 * (h - 2.0) / C
    ns = int(np.ceil((np.sqrt(h * h + radius * radius) - h + 4.0) / 2.0)) + 3
    return dict(k=K_W, gamma=GAMMA, t0=t0, dt=4.0 / C, n_samples=ns, c=C)


@pytest.fixture(scope="module")
def disk():
    return _disk_facets(4.0 * LAM, 40.0 * LAM)


def test_coherent_smooth_and_sigma0_bitwise(disk):
    """Verification item (b): roughness=None traces the pre-roughness program
    (same code path, by construction) and sigma=0 through the ROUGH branch is
    bit-identical to it (attenuation exactly 1, D_Phi exactly 0)."""
    kw = _disk_kw(H, 40.0 * LAM)
    smooth, drop0 = coherent_cluttergram(np.array([[0.0, 0.0, H]]), UCT,
                                         *disk, **kw)
    ph = rg.speckle_phasors(len(disk[0]), seed=(1, 0))
    s0, drop1 = coherent_cluttergram(np.array([[0.0, 0.0, H]]), UCT, *disk,
                                     roughness=(0.0, 2.0, ph, 10), **kw)
    assert np.array_equal(smooth, s0)
    assert np.array_equal(drop0, drop1)


def test_coherent_ensemble_vs_reference(disk):
    """Kernel ensemble (over speckle seeds) mean total power vs the float64
    rough-disk referee; coherent part alone via zero phasors. Measured
    -0.15 dB (64 seeds) / -0.0016 dB; gates 0.5 / 0.05 dB."""
    sigma, l = 0.10 * LAM, 2.0 * LAM
    nt = rg.n_terms_for((2.0 * K_W * sigma) ** 2)
    kw = _disk_kw(H, 40.0 * LAM)
    pos = np.array([[0.0, 0.0, H]])
    m = len(disk[0])
    tots = []
    for s in range(64):
        ph = rg.speckle_phasors(m, seed=(2, s))
        f, _ = coherent_cluttergram(pos, UCT, *disk,
                                    roughness=(sigma, l, ph, nt), **kw)
        tots.append(abs(f[0].sum()) ** 2)
    coh, inc = gk.rough_disk_power(H, 40.0 * LAM, 4.0 * LAM, K_W, GAMMA,
                                   sigma, l)
    db = 10.0 * np.log10(np.mean(tots) / (coh + inc))
    assert abs(db) <= 0.5, db

    f_c, _ = coherent_cluttergram(pos, UCT, *disk,
                                  roughness=(sigma, l,
                                             np.zeros(m, np.complex64), nt),
                                  **kw)
    db_c = 10.0 * np.log10(abs(f_c[0].sum()) ** 2 / coh)
    assert abs(db_c) <= 0.05, db_c


# ------------------------------------------------------------ multilayer kernel

LAM_M = 2.0
K0 = 2.0 * np.pi / LAM_M
EPS_ICE = 3.17
N_ICE = float(np.sqrt(EPS_ICE))
H_ML, D_ML = 1000.0, 200.0


@pytest.fixture(scope="module")
def slab_facets():
    surf = _flat_grid_facets(0.0, 400.0, 10.0)
    bed = _flat_grid_facets(-D_ML, 400.0, 6.0)
    return surf, bed


def _ml_kw(**over):
    opl = H_ML + N_ICE * D_ML
    kw = dict(mode="coherent", t0=2.0 * (opl - 5.0) / C, dt=1e-8,
              n_samples=400, c=C, gamma=GAMMA, k0=K0)
    kw.update(over)
    return kw


def test_multilayer_sigma0_bitwise(slab_facets):
    """sigma=0 through the rough multilayer branch (target roughness AND
    crossing sigmas) is bit-identical to the smooth kernel."""
    surf, bed = slab_facets
    pos = np.array([[0.0, 0.0, H_ML]])
    eps = [1.0, EPS_ICE]  # per-LEG (air, ice), len == len(crossed) + 1
    att = [0.0, 0.0]
    smooth, drop0 = refracted_cluttergram(pos, UCT, bed, [surf], eps, att,
                                          **_ml_kw())
    ph = rg.speckle_phasors(len(bed.centers), seed=(4, 0))
    s0, drop1 = refracted_cluttergram(pos, UCT, bed, [surf], eps, att,
                                      roughness=(0.0, 1.0, ph, 10),
                                      crossed_sigma=np.zeros(1), **_ml_kw())
    assert np.array_equal(smooth, s0)
    assert np.array_equal(drop0, drop1)


def test_multilayer_eps1_rough_reduction(slab_facets):
    """eps -> 1: the rough bed through the refracted path equals the
    single-interface rough coherent kernel on the bed facets (same phasors)
    -- validates the incoherent term's F factor (transmission, spreading,
    delay) through the multilayer path. Measured max |diff| 4.3e-4 of peak;
    gate 5e-3 (the smooth eps->1 reduction's gate)."""
    _, bed = slab_facets
    pos = np.array([[0.0, 0.0, H_ML]])
    surf = _flat_grid_facets(0.0, 400.0, 10.0)
    sigma, l = 0.05 * LAM_M, 3.0
    nt = rg.n_terms_for((2.0 * K0 * sigma) ** 2)
    ph = rg.speckle_phasors(len(bed.centers), seed=(5, 0))
    kw = _ml_kw(t0=2.0 * (H_ML + D_ML - 5.0) / C)  # eps=1 optical path
    ml, _ = refracted_cluttergram(pos, UCT, bed, [surf], [1.0, 1.0],
                                  [0.0] * 2, roughness=(sigma, l, ph, nt),
                                  **kw)
    ref, _ = coherent_cluttergram(pos, UCT, bed.centers, bed.normals,
                                  bed.areas, bed.e1, bed.e2, k=K0,
                                  gamma=GAMMA, t0=kw["t0"], dt=kw["dt"],
                                  n_samples=kw["n_samples"], c=C,
                                  roughness=(sigma, l, ph, nt))
    peak = np.abs(ref).max()
    assert peak > 0.0  # window actually contains the return
    assert np.abs(ml - ref).max() <= 5e-3 * peak


def test_multilayer_bed_attenuation_local_k(slab_facets):
    """Buried-facet coherent attenuation uses the LOCAL-medium wavenumber:
    with zero phasors, the total bed field ratio rough/smooth equals
    exp(-sigma^2 (2 k0 n_ice)^2 / 2) -- NOT the free-space value. Measured
    |ratio/pred - 1| = 8.9e-5; gate 5e-3 (pred with the free-space k would
    miss by ~12 percent)."""
    surf, bed = slab_facets
    pos = np.array([[0.0, 0.0, H_ML]])
    eps = [1.0, EPS_ICE]
    att = [0.0] * 2
    sigma = 0.05 * LAM_M / N_ICE  # 0.05 wavelengths in ICE
    nt = rg.n_terms_for((2.0 * K0 * N_ICE * sigma) ** 2)
    smooth, _ = refracted_cluttergram(pos, UCT, bed, [surf], eps, att,
                                      **_ml_kw())
    zeros = np.zeros(len(bed.centers), np.complex64)
    rough, _ = refracted_cluttergram(pos, UCT, bed, [surf], eps, att,
                                     roughness=(sigma, 3.0, zeros, nt),
                                     **_ml_kw())
    ratio = abs(rough[0].sum()) / abs(smooth[0].sum())
    pred = np.exp(-0.5 * (sigma * 2.0 * K0 * N_ICE) ** 2)
    assert abs(ratio / pred - 1.0) <= 5e-3, (ratio, pred)


def test_transmission_attenuation_through_simulate():
    """Rough SURFACE, smooth bed, through simulate(): the bed layer's total
    field carries the two-way transmission attenuation exp(-2 sigma^2 K_t^2),
    K_t = k0 (cos(t1) - n_ice cos(t2)) ~ k0 (1 - n_ice) at nadir. Measured
    |ratio/pred - 1| = 7.9e-4; gate 0.02 -- the predicted attenuation is
    0.763x in field, and the one-way exp(-sigma^2 K_t^2 / 2) would miss the
    measurement by ~23 percent."""
    scene = syn.slab_scene(surface=500.0, depth=300.0, extent=2000.0,
                           n_traces=3, altitude=1000.0)
    media = [Medium(name="air", eps_r=1.0), Medium(name="ice", eps_r=EPS_ICE),
             Medium(name="rock", eps_r=6.0)]
    sigma = 0.15

    def cfg(rough):
        ifs = [DemInterface(name="surface",
                            roughness=(RoughnessConfig(sigma_m=sigma,
                                                       corr_length_m=5.0)
                                       if rough else None)),
               DemInterface(name="bed")]
        return SimConfig(mode="coherent",
                         radar=RadarConfig(dt=1e-8, n_samples=1250, t0=0.0,
                                           f0=C / LAM_M),
                         facets=FacetConfig(spacing=15.0), media=media,
                         interfaces=ifs)

    smooth = soundersim.simulate(scene, cfg(False))
    rough = soundersim.simulate(scene, cfg(True))
    f_s = smooth.field.sel(layer="bed").values[1].sum()
    f_r = rough.field.sel(layer="bed").values[1].sum()
    kt = K0 * (1.0 - N_ICE)
    pred = np.exp(-2.0 * (sigma * kt) ** 2)
    ratio = abs(f_r) / abs(f_s)
    assert abs(ratio / pred - 1.0) <= 0.02, (ratio, pred)
    # the surface layer is speckled but reproducible for a fixed seed
    again = soundersim.simulate(scene, cfg(True))
    assert np.array_equal(rough.field.values, again.field.values)


def test_simulate_surface_roughness_plumbing():
    """Single-interface coherent runs with surface roughness: deterministic
    per seed, different seeds draw different speckle, coherent power drops by
    the attenuation."""
    scene = syn.flat_scene(altitude=1000.0, n_traces=2, extent=1000.0,
                           posting=12.5)
    sigma = 0.10 * LAM_M

    def cfg(seed):
        return SimConfig(
            mode="coherent", roughness_seed=seed,
            radar=RadarConfig(dt=1e-8, n_samples=800, t0=0.0, f0=C / LAM_M),
            facets=FacetConfig(spacing=12.5),
            interfaces=[DemInterface(
                name="surface",
                roughness=RoughnessConfig(sigma_m=sigma, corr_length_m=4.0))])

    a = soundersim.simulate(scene, cfg(0))
    b = soundersim.simulate(scene, cfg(0))
    c = soundersim.simulate(scene, cfg(1))
    assert np.array_equal(a.field.values, b.field.values)
    assert not np.array_equal(a.field.values, c.field.values)


def test_config_roughness_validation():
    rc = RadarConfig(dt=1e-8, n_samples=10, t0=0.0, f0=195e6)
    with pytest.raises(ValueError, match="coherent"):
        SimConfig(mode="incoherent", radar=rc, facets=FacetConfig(),
                  interfaces=[DemInterface(
                      name="surface",
                      roughness=RoughnessConfig(sigma_m=0.1,
                                                corr_length_m=1.0))])
    with pytest.raises(ValueError):
        RoughnessConfig(sigma_m=-0.1, corr_length_m=1.0)
    with pytest.raises(ValueError):
        RoughnessConfig(sigma_m=0.1, corr_length_m=0.0)
    # JSON round trip
    cfg = SimConfig(mode="coherent", radar=rc, facets=FacetConfig(),
                    interfaces=[DemInterface(
                        name="surface",
                        roughness=RoughnessConfig(sigma_m=0.03,
                                                  corr_length_m=2.0))])
    back = SimConfig.model_validate_json(cfg.model_dump_json())
    assert back.interfaces[0].roughness.sigma_m == 0.03
    # incoherent-mode kernel guard
    with pytest.raises(ValueError, match="coherent"):
        refracted_cluttergram(
            np.zeros((1, 3)), UCT, _flat_grid_facets(-10.0, 40.0, 10.0),
            [_flat_grid_facets(0.0, 40.0, 10.0)], [1.0, 3.17], [0.0, 0.0],
            mode="incoherent", t0=0.0, dt=1e-8, n_samples=10, c=C,
            crossed_sigma=np.array([0.1]))


def test_coherent_rough_padding_finite(disk):
    """Zero-padded block slots (e1 = e2 = 0) must not inject NaN through the
    incoherent term's 0/0 d_phi arguments (area-mask regression): a padded
    rough run is finite and matches the unpadded one."""
    kw = _disk_kw(H, 40.0 * LAM)
    pos = np.array([[0.0, 0.0, H]])
    m = len(disk[0])
    ph = rg.speckle_phasors(m, seed=(7, 0))
    rough = (0.10 * LAM, 2.0 * LAM, ph, 10)
    full, dr0 = coherent_cluttergram(pos, UCT, *disk, roughness=rough, **kw)
    padded, dr1 = coherent_cluttergram(pos, UCT, *disk, roughness=rough,
                                       block_size=m // 3 + 1, **kw)
    assert np.all(np.isfinite(padded)) and np.all(np.isfinite(dr1))
    np.testing.assert_allclose(padded, full, rtol=2e-4, atol=1e-9)
    np.testing.assert_allclose(dr1, dr0, rtol=1e-6)


def test_multilayer_rough_padding_finite(slab_facets):
    """Same regression for the multilayer kernel, where joint-path blocking
    (block_size 4096) padded real firn scenes and produced all-NaN layers."""
    surf, bed = slab_facets
    pos = np.array([[0.0, 0.0, H_ML]])
    eps, att = [1.0, EPS_ICE], [0.0, 0.0]
    ph = rg.speckle_phasors(len(bed.centers), seed=(7, 1))
    # window opens at the PAD slots' fake arrival (opl ~ H_ML: zeroed facets
    # sit at the origin) so their NaN would land in-band, as in the real
    # firn scenes that exposed the bug
    kw = _ml_kw(t0=2.0 * (H_ML - 5.0) / C, n_samples=500)
    rough = (0.05 * LAM_M, 1.5 * LAM_M, ph, 10)
    full, dr0 = refracted_cluttergram(pos, UCT, bed, [surf], eps, att,
                                      roughness=rough, **kw)
    padded, dr1 = refracted_cluttergram(pos, UCT, bed, [surf], eps, att,
                                        roughness=rough,
                                        block_size=len(bed.centers) // 3 + 1,
                                        **kw)
    # pad-slot NaN can surface in the FIELD (in-window fake opl) or in the
    # DROP counter (out-of-window) depending on geometry -- gate both
    assert np.all(np.isfinite(padded)) and np.all(np.isfinite(dr1))
    np.testing.assert_allclose(padded, full, rtol=2e-4, atol=1e-9)
    np.testing.assert_allclose(dr1, dr0, rtol=1e-6)
