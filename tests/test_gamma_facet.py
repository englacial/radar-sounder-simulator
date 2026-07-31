"""Per-facet target reflectivity (gamma arrays) through the kernels.

- scalar-vs-constant-array bitwise regressions: an (n_facets,) array holding
  the scalar value produces bit-identical output on both the coherent and the
  multilayer kernel (smooth AND rough branches, sequential AND joint
  refraction) -- the scalar path itself is untouched (gamma_facet=False
  traces the old program);
- a two-zone gamma array produces the analytic relative power step: with a
  left/right-symmetric flat grid and split_sides, the left and right fields
  differ exactly by the zone gammas, so the binned power ratio is
  (g_left/g_right)^2;
- per-facet gamma in incoherent mode raises (the incoherent path books no
  target reflectivity by convention).
"""

import numpy as np
import pytest

from soundersim import roughness as rg
from soundersim.kernels.coherent import coherent_cluttergram
from soundersim.kernels.multilayer import refracted_cluttergram
from soundersim.scene import Facets

C = 299792458.0
UCT = np.array([[0.0, -1.0, 0.0]])
GAMMA = -0.281
LAM = 2.0
K0 = 2.0 * np.pi / LAM
EPS_ICE = 3.17
N_ICE = float(np.sqrt(EPS_ICE))
H, D = 1000.0, 200.0


def _flat_grid_facets(z, extent, d):
    """Full flat Facets grid at height z (even n: no facets on y = 0, so the
    grid is exactly mirror-symmetric about the split_sides plane)."""
    n = round(extent / d)
    assert n % 2 == 0
    ax = (np.arange(n) - (n - 1) / 2.0) * d
    X, Y = np.meshgrid(ax, ax)
    centers = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    m = len(centers)
    rows, cols = np.divmod(np.arange(m), n)
    return Facets(centers, np.tile([0.0, 0.0, 1.0], (m, 1)),
                  np.full(m, d * d), np.tile([d, 0.0, 0.0], (m, 1)),
                  np.tile([0.0, d, 0.0], (m, 1)),
                  np.column_stack([rows, cols]), (n, n))


@pytest.fixture(scope="module")
def slab():
    surf = _flat_grid_facets(0.0, 400.0, 10.0)
    bed = _flat_grid_facets(-D, 396.0, 6.0)
    return surf, bed


def _ml_kw(**over):
    opl = H + N_ICE * D
    kw = dict(mode="coherent", t0=2.0 * (opl - 5.0) / C, dt=1e-8,
              n_samples=400, c=C, gamma=GAMMA, k0=K0)
    kw.update(over)
    return kw


def _co_kw(**over):
    kw = dict(k=K0, gamma=GAMMA, t0=2.0 * (H - 5.0) / C, dt=1e-8,
              n_samples=400, c=C)
    kw.update(over)
    return kw


# ------------------------------------------- scalar vs constant array bitwise

def test_coherent_constant_array_bitwise(slab):
    _, bed = slab
    pos = np.array([[0.0, 0.0, H]])
    m = len(bed.centers)
    f_s, d_s = coherent_cluttergram(pos, UCT, bed.centers, bed.normals,
                                    bed.areas, bed.e1, bed.e2, **_co_kw())
    f_a, d_a = coherent_cluttergram(pos, UCT, bed.centers, bed.normals,
                                    bed.areas, bed.e1, bed.e2,
                                    **_co_kw(gamma=np.full(m, GAMMA)))
    assert np.array_equal(f_s, f_a)
    assert np.array_equal(d_s, d_a)
    # rough branch, same phasors on both sides
    ph = rg.speckle_phasors(m, seed=(11, 0))
    rkw = dict(roughness=(0.05 * LAM, 3.0, ph, 10))
    f_s, d_s = coherent_cluttergram(pos, UCT, bed.centers, bed.normals,
                                    bed.areas, bed.e1, bed.e2, **_co_kw(),
                                    **rkw)
    f_a, d_a = coherent_cluttergram(pos, UCT, bed.centers, bed.normals,
                                    bed.areas, bed.e1, bed.e2,
                                    **_co_kw(gamma=np.full(m, GAMMA)), **rkw)
    assert np.array_equal(f_s, f_a)
    assert np.array_equal(d_s, d_a)


@pytest.mark.parametrize("refraction", ["sequential", "joint"])
def test_multilayer_constant_array_bitwise(slab, refraction):
    surf, bed = slab
    pos = np.array([[0.0, 0.0, H]])
    eps, att = [1.0, EPS_ICE], [0.0, 10.0]
    m = len(bed.centers)
    kw = _ml_kw(refraction=refraction)
    f_s, d_s = refracted_cluttergram(pos, UCT, bed, [surf], eps, att, **kw)
    f_a, d_a = refracted_cluttergram(pos, UCT, bed, [surf], eps, att,
                                     **{**kw, "gamma": np.full(m, GAMMA)})
    assert np.array_equal(f_s, f_a)
    assert np.array_equal(d_s, d_a)
    # rough target + crossing sigma, same phasors on both sides
    ph = rg.speckle_phasors(m, seed=(12, 0))
    rkw = dict(roughness=(0.05 * LAM, 3.0, ph, 10),
               crossed_sigma=np.array([0.1]))
    f_s, d_s = refracted_cluttergram(pos, UCT, bed, [surf], eps, att, **kw,
                                     **rkw)
    f_a, d_a = refracted_cluttergram(pos, UCT, bed, [surf], eps, att,
                                     **{**kw, "gamma": np.full(m, GAMMA)},
                                     **rkw)
    assert np.array_equal(f_s, f_a)
    assert np.array_equal(d_s, d_a)


# ------------------------------------------------- two-zone analytic step

def test_coherent_two_zone_power_step(slab):
    """Left zone gamma g1, right zone g2 on a mirror-symmetric grid: the
    split_sides left/right binned powers differ by exactly (g1/g2)^2."""
    _, bed = slab
    pos = np.array([[0.0, 0.0, H]])
    g1, g2 = -0.4, -0.1
    right = np.sum((bed.centers - pos[0]) * UCT[0], axis=-1) > 0
    gam = np.where(right, g2, g1)
    f, _ = coherent_cluttergram(pos, UCT, bed.centers, bed.normals, bed.areas,
                                bed.e1, bed.e2, split_sides=True,
                                **_co_kw(gamma=gam))
    p_left = np.abs(f[0, :, 0]) ** 2
    p_right = np.abs(f[0, :, 1]) ** 2
    tot = p_right.sum()
    assert tot > 0.0
    np.testing.assert_allclose(p_left.sum() / tot, (g1 / g2) ** 2, rtol=1e-4)
    # per-bin too (the geometry factor is bin-by-bin symmetric)
    keep = p_right > 1e-6 * p_right.max()
    np.testing.assert_allclose(p_left[keep] / p_right[keep],
                               (g1 / g2) ** 2, rtol=1e-3)


def test_multilayer_two_zone_power_step(slab):
    """Same analytic step through the refracted path (bed under a surface)."""
    surf, bed = slab
    pos = np.array([[0.0, 0.0, H]])
    g1, g2 = -0.4, -0.1
    right = np.sum((bed.centers - pos[0]) * UCT[0], axis=-1) > 0
    gam = np.where(right, g2, g1)
    f, _ = refracted_cluttergram(pos, UCT, bed, [surf], [1.0, EPS_ICE],
                                 [0.0, 10.0], split_sides=True,
                                 **_ml_kw(gamma=gam))
    p_left = np.abs(f[0, :, 0]) ** 2
    p_right = np.abs(f[0, :, 1]) ** 2
    tot = p_right.sum()
    assert tot > 0.0
    np.testing.assert_allclose(p_left.sum() / tot, (g1 / g2) ** 2, rtol=1e-4)


# ----------------------------------------------------------------- validation

def test_incoherent_per_facet_gamma_raises(slab):
    surf, bed = slab
    pos = np.array([[0.0, 0.0, H]])
    with pytest.raises(ValueError, match="coherent"):
        refracted_cluttergram(pos, UCT, bed, [surf], [1.0, EPS_ICE],
                              [0.0, 0.0], **_ml_kw(
                                  mode="incoherent",
                                  gamma=np.full(len(bed.centers), 0.3)))


def test_bad_shape_raises(slab):
    surf, bed = slab
    pos = np.array([[0.0, 0.0, H]])
    with pytest.raises(ValueError, match="shape"):
        refracted_cluttergram(pos, UCT, bed, [surf], [1.0, EPS_ICE],
                              [0.0, 0.0], **_ml_kw(gamma=np.full(7, 0.3)))
    with pytest.raises(ValueError, match="shape"):
        coherent_cluttergram(pos, UCT, bed.centers, bed.normals, bed.areas,
                             bed.e1, bed.e2, **_co_kw(gamma=np.full(7, 0.3)))
