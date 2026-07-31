"""Unit tests for soundersim.firn (effective-contrast firn pipeline promoted
from tools/run_b26_comparison.py; see claude_notes/b26_comparison_findings.md).
"""

from pathlib import Path

import numpy as np
import pytest

from soundersim.config import OffsetInterface
from soundersim.firn import (FirnCore, eps_kovacs, firn_stack,
                             load_density_tab, tmm_reflection)
from soundersim.physics import fresnel_normal

FIXDIR = Path(__file__).parent / "fixtures" / "firn"
LAM = 299792458.0 / 195e6


@pytest.fixture(scope="module")
def b26():
    return FirnCore(FIXDIR / "ngt37C95.2_density.tab")


@pytest.fixture(scope="module")
def b25():
    return FirnCore(FIXDIR / "BER11C95_25_density.tab")


def test_eps_kovacs():
    # rho = 917 kg/m^3 (ice) -> eps ~ 3.15 (C&S 2020 Eq. 4)
    assert abs(eps_kovacs(917.0) - (1.0 + 0.845 * 0.917) ** 2) < 1e-12
    assert eps_kovacs(0.0) == 1.0


def test_load_density_tab_both_cores():
    for name, zmax, npts in (("ngt37C95.2_density.tab", 119.66, None),
                             ("BER11C95_25_density.tab", 178.213, 51175)):
        z, rho = load_density_tab(FIXDIR / name)
        assert z.shape == rho.shape and (np.diff(z) > 0).all()
        assert abs(float(z.max()) - zmax) < 0.01
        assert 200.0 < rho.min() and rho.max() < 950.0
        if npts:
            assert len(z) == npts


def test_tmm_single_interface_is_fresnel():
    # No slabs: half-space n1 | n2 reduces to the plain Fresnel coefficient.
    r = tmm_reflection([1.0, 1.5], dz=0.1, lam=LAM)
    assert abs(abs(r) - abs(fresnel_normal(1.0, 1.5 ** 2))) < 1e-12
    # Uniform stack: no contrast, no reflection.
    assert abs(tmm_reflection([1.3, 1.3, 1.3, 1.3], 0.05, LAM)) < 1e-14


def test_core_smoothing_and_point_eps(b26):
    # Edge-normalized boxcar: deepest sample stays physical (not halved).
    assert b26.eps[-1] > 2.9
    assert 1.5 < b26.point_eps(1.0) < 1.9          # near-surface firn
    assert 2.9 < b26.point_eps(119.0) < 3.2        # near-ice at depth
    d = b26.equal_depths(10)
    assert d[0] == 1.0 and d[-1] == b26.zmax and len(d) == 10


def test_effective_contrast_invariants(b26):
    depths = b26.equal_depths(10)
    eps, r = b26.effective_contrast_eps(depths, LAM)
    assert eps.shape == (11,) and r.shape == (10,)
    assert np.isfinite(eps).all() and (eps > 1.0).all() and (eps < 3.3).all()
    assert eps[0] == b26.point_eps(depths[0])       # firn0 point-sampled
    # plain Fresnel contrasts reproduce the segment |r| exactly
    n = np.sqrt(eps)
    gam = np.abs((n[:-1] - n[1:]) / (n[:-1] + n[1:]))
    assert np.abs(gam - r).max() < 1e-15
    # segments tile the profile: reflectivities are small and physical
    rdb = 20.0 * np.log10(r)
    assert (rdb < -20.0).all()


def test_effective_contrast_b25(b25):
    depths = b25.equal_depths(10)
    eps, r = b25.effective_contrast_eps(depths, LAM)
    assert np.isfinite(eps).all() and (eps > 1.0).all() and (eps < 3.4).all()
    n = np.sqrt(eps)
    gam = np.abs((n[:-1] - n[1:]) / (n[:-1] + n[1:]))
    assert np.abs(gam - r).max() < 1e-15
    # B25 reaches ice density by ~100 m: deep segment contrasts are weak
    assert (20.0 * np.log10(r[depths > 120.0]) < -40.0).all()


def test_contrast_density_is_phase_aware(b26):
    """The envelope is the COHERENT aggregate, so a smooth density ramp (large
    |d eps/dz|, no Bragg-scale structure) must not register as a horizon."""
    z, dens = b26.contrast_density(LAM, 4.418)
    assert z.shape == dens.shape and np.isfinite(dens).all()
    assert (dens >= 0).all() and dens.max() > 0
    # a synthetic pure ramp: strong gradient everywhere, no coherent return
    class _Ramp(FirnCore):
        def __init__(self):
            self.z_top, self.zmax = 1.0, 100.0
            self.z_raw = np.arange(0.0, 100.0, 0.001)
            self.rho_raw = 300.0 + 5.0 * self.z_raw
    _, ramp = _Ramp().contrast_density(LAM, 4.418)
    assert ramp.max() < 0.02 * dens.max()


def test_peak_depths(b26):
    for n in (5, 10, 20):
        d, prom, sep = b26.peak_depths(n, LAM, 4.418)
        assert d.shape == prom.shape == (n,)
        assert (np.diff(d) >= sep - 1e-9).all()      # min separation honored
        assert b26.z_top <= d[0] and d[-1] <= b26.zmax
        edges = np.concatenate([[b26.z_top], d, [b26.zmax]])
        assert np.diff(edges).max() <= 1.5 * (b26.zmax - b26.z_top) / n * 1.05
    # Peak placement centers each segment on a horizon instead of cutting
    # through it, so the in-phase strata stay together and the TOTAL captured
    # reflectivity rises (the median segment is unchanged -- it is the strong
    # segments that get stronger).
    d, _, _ = b26.peak_depths(10, LAM, 4.418)
    r_p = b26.segment_reflectivity(d, LAM, top=b26.z_top)
    r_u = b26.segment_reflectivity(b26.equal_depths(10), LAM)
    assert (r_p ** 2).sum() > 1.3 * (r_u ** 2).sum()


def test_segment_top_anchor(b26):
    """top= moves only the first segment edge; it is a no-op when depths[0]
    already is the anchor (so uniform stacks are bit-identical)."""
    du = b26.equal_depths(10)
    assert np.array_equal(b26.segment_reflectivity(du, LAM),
                          b26.segment_reflectivity(du, LAM, top=b26.z_top))
    d = np.array([6.0, 20.0, 50.0, 90.0])
    r0 = b26.segment_reflectivity(d, LAM)
    r1 = b26.segment_reflectivity(d, LAM, top=b26.z_top)
    assert r0[0] != r1[0] and np.array_equal(r0[1:], r1[1:])
    assert r1[0] > r0[0]                     # the wider top segment gathers more


def test_firn_stack(b26):
    depths = b26.equal_depths(5)
    eps, _ = b26.effective_contrast_eps(depths, LAM)
    media, ifaces = firn_stack(depths, eps, 15.0)
    assert [m.name for m in media] == ["air"] + [f"firn{i}" for i in range(5)] \
        + ["substrate"]
    assert media[0].eps_r == 1.0 and not media[0].attenuation_db_per_km
    assert all(m.attenuation_db_per_km == 15.0 for m in media[1:])
    assert [m.eps_r for m in media[1:]] == [float(x) for x in eps]
    assert ifaces[0].name == "surface"
    for i, iface in enumerate(ifaces[1:]):
        assert isinstance(iface, OffsetInterface)
        assert iface.reference == "surface"
        assert iface.offset == -float(depths[i])
        assert iface.roughness is None
    with pytest.raises(ValueError):
        firn_stack(depths, eps[:-1], 15.0)
    # roughness lands on internal interfaces only
    sig, cl = np.full(5, 0.03), np.full(5, 2.5)
    _, ifr = firn_stack(depths, eps, 15.0, roughness=(sig, cl))
    assert ifr[0].roughness is None
    assert all(i.roughness.sigma_m == 0.03 and i.roughness.corr_length_m == 2.5
               for i in ifr[1:])
