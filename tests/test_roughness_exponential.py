"""Exponential-ACF option of the sub-facet roughness model
(docs/roughness.md): the Gaussian path stays bit-identical, the exponential
area-only series matches an independent float64 sum of the C&S 2020 Eq 6 /
Gerekos Appendix C law and is converged at ``n_terms_for``, the option is
refused without the area-only D_Phi, and the runner/spec plumbing forks the
chunk cache keys only for the new ACF."""
import math
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from soundersim import roughness as rg  # noqa: E402
from soundersim.config import (DemInterface, FacetConfig, GrazingFixConfig,  # noqa: E402
                               RadarConfig, RoughnessConfig, SimConfig)
from soundersim.firn import firn_stack  # noqa: E402
from soundersim.kernels.coherent import coherent_cluttergram  # noqa: E402

import run_altitude_comparison as rac  # noqa: E402
import run_basal_clutter as rbc  # noqa: E402
import surface_roughness_b1 as b1  # noqa: E402
from clutter_spec import RunSpec, load_spec  # noqa: E402
from test_basal_hypotheses import _p  # noqa: E402

C = 299792458.0


def _series_ref(sigma, l, k, th, L, acf, n=600):
    """Independent numpy sum of the area-only series for a square facet."""
    x = (2 * k * sigma * np.cos(th)) ** 2
    kb = 2 * k * np.sin(th)
    acc = np.zeros_like(th)
    for m in range(1, n):
        logw = m * np.log(x) - math.lgamma(m + 1) - x
        if acf == "gaussian":
            w = np.pi * l ** 2 / m * np.exp(-(kb * l) ** 2 / (4 * m))
        else:
            w = 2 * np.pi * (l / m) ** 2 * (1 + (kb * l / m) ** 2) ** -1.5
        acc += np.exp(logw) * w
    return acc * L * L


def test_gaussian_path_bit_identical():
    sigma, l, L = 0.049474, 2.982179, 40.0
    th = np.deg2rad(np.array([0.0, 20.0, 45.0, 70.0]))
    for fc in (60e6, 400e6):
        k = 2 * np.pi * fc / C
        nt = rg.n_terms_for((2 * k * sigma) ** 2)
        args = (sigma, l, 2 * k * np.cos(th), 2 * k * np.sin(th),
                np.zeros_like(th), L, L)
        for area in (False, True):
            a = np.array(rg.d_phi(*args, n_terms=nt, area_only=area))
            b = np.array(rg.d_phi(*args, n_terms=nt, area_only=area,
                                  acf="gaussian"))
            assert np.array_equal(a, b)


def test_exponential_series_matches_reference_and_is_converged():
    """400 MHz, l = 5 m (k_B l up to 2 k l = 84): the n_terms_for count vs a
    600-term float64 sum, and vs the Gaussian tail difference."""
    sigma, l, L = 0.0515, 5.0, 40.0
    th = np.deg2rad(np.array([0.0, 5.0, 20.0, 45.0, 70.0, 89.0]))
    for fc in (60e6, 195e6, 400e6):
        k = 2 * np.pi * fc / C
        nt = rg.n_terms_for((2 * k * sigma) ** 2)
        with jax.enable_x64():
            e = np.array(rg.d_phi(sigma, l, 2 * k * np.cos(th),
                                  2 * k * np.sin(th), np.zeros_like(th), L, L,
                                  n_terms=nt, area_only=True,
                                  acf="exponential"))
        ref = _series_ref(sigma, l, k, th, L, "exponential")
        assert np.max(np.abs(10 * np.log10(e / ref))) < 1e-6
    # off-nadir the exponential tail is far above the Gaussian one at the
    # same (sigma, l): the reason for the option
    k = 2 * np.pi * 195e6 / C
    nt = rg.n_terms_for((2 * k * sigma) ** 2)
    with jax.enable_x64():
        e, g = (np.array(rg.d_phi(sigma, l, 2 * k * np.cos(th),
                                  2 * k * np.sin(th), np.zeros_like(th), L, L,
                                  n_terms=nt, area_only=True, acf=acf))
                for acf in ("exponential", "gaussian"))
    assert 1.9 < e[0] / g[0] < 2.0  # nadir: W_1 ratio 2, W_m -> 2/m
    assert 10 * np.log10(e[3] / g[3]) > 100


def test_exponential_requires_area_only():
    with pytest.raises(ValueError, match="area-only"):
        rg.d_phi(0.05, 2.0, 4.0, 0.0, 0.0, 10.0, 10.0, n_terms=10,
                 acf="exponential")
    with pytest.raises(ValueError, match="unknown"):
        rg.d_phi(0.05, 2.0, 4.0, 0.0, 0.0, 10.0, 10.0, n_terms=10,
                 area_only=True, acf="powerlaw")
    with pytest.raises(ValueError, match="grazing_fix"):
        SimConfig(mode="coherent", radar=RadarConfig(
            t0=0.0, dt=1e-8, n_samples=64, f0=195e6),
            facets=FacetConfig(spacing=10.0),
            interfaces=[DemInterface(roughness=RoughnessConfig(
                sigma_m=0.05, corr_length_m=5.0, acf="exponential"))])
    # kernel-level guard
    n = 4
    z = np.zeros(n)
    c = np.stack([np.arange(n) * 10.0, z, z], 1)
    nrm = np.tile([0.0, 0.0, 1.0], (n, 1))
    e1 = np.tile([10.0, 0.0, 0.0], (n, 1))
    e2 = np.tile([0.0, 10.0, 0.0], (n, 1))
    ph = rg.speckle_phasors(n, 0)
    kw = dict(k=4.0, gamma=-0.3, t0=0.0, dt=1e-8, n_samples=64, c=C)
    with pytest.raises(ValueError, match="area-only"):
        coherent_cluttergram(np.array([[0.0, 0.0, 500.0]]),
                             np.array([[0.0, 1.0, 0.0]]), c, nrm,
                             np.full(n, 100.0), e1, e2,
                             roughness=(0.05, 5.0, ph, 10, "exponential"),
                             **kw)
    # 5-tuple 'gaussian' == 4-tuple: identical program
    for extra in ((), ("gaussian",)):
        f, _ = coherent_cluttergram(np.array([[0.0, 0.0, 500.0]]),
                                    np.array([[0.0, 1.0, 0.0]]), c, nrm,
                                    np.full(n, 100.0), e1, e2,
                                    roughness=(0.05, 5.0, ph, 10) + extra,
                                    **kw)
        if not extra:
            f0 = f
    assert np.array_equal(f0, f)


def test_firn_stack_and_config_carry_acf():
    depths = np.array([5.0, 15.0])
    sig, cl = np.array([0.03, 0.02]), np.array([2.5, 1.5])
    _, ifr = firn_stack(depths, [1.5, 1.8, 2.0], 15.0,
                        roughness=(sig, cl, "exponential"))
    assert ifr[0].roughness is None
    assert all(i.roughness.acf == "exponential" for i in ifr[1:])
    _, ifg = firn_stack(depths, [1.5, 1.8, 2.0], 15.0, roughness=(sig, cl))
    assert all(i.roughness.acf == "gaussian" for i in ifg[1:])
    with pytest.raises(ValueError):
        RoughnessConfig(sigma_m=0.05, corr_length_m=1.0, acf="powerlaw")


def test_runner_keys_fork_only_for_exponential():
    p, rows = _p(), np.arange(198)
    fix = (rac.SURF_ROUGH_SIGMA_M, rac.SURF_ROUGH_CL_M)
    rid_t = rbc.chunk_rid(p, 0, rac.ATT_DB_PER_KM, True)
    m_t = rbc.chunk_meta(p, 0, rows, 17, 3365, rac.ATT_DB_PER_KM, True)
    # legacy forms unchanged (byte-identical rid + meta)
    assert rbc.chunk_rid(p, 0, rac.ATT_DB_PER_KM, fix + ("gaussian",)) == rid_t
    assert rbc.chunk_meta(p, 0, rows, 17, 3365, rac.ATT_DB_PER_KM,
                          fix + ("gaussian",)) == m_t
    assert "surf_rough_acf" not in m_t
    pair = (0.0515, 5.276)
    rid_g = rbc.chunk_rid(p, 0, rac.ATT_DB_PER_KM, pair)
    assert rid_g == "low_full_dgn_rssnr_proc_c00_srough_sr0.0515_5.276"
    assert rbc.chunk_rid(p, 0, rac.ATT_DB_PER_KM, pair + ("gaussian",)) == rid_g
    rid_e = rbc.chunk_rid(p, 0, rac.ATT_DB_PER_KM, pair + ("exponential",))
    assert rid_e == rid_g + "_exp"
    m_g = rbc.chunk_meta(p, 0, rows, 17, 3365, rac.ATT_DB_PER_KM, pair)
    m_e = rbc.chunk_meta(p, 0, rows, 17, 3365, rac.ATT_DB_PER_KM,
                         pair + ("exponential",))
    assert "surf_rough_acf" not in m_g
    assert m_e["surf_rough_acf"] == "exponential"
    assert m_e["surf_rough_sigma_l"] == m_g["surf_rough_sigma_l"] == [0.0515, 5.276]
    assert {k: v for k, v in m_e.items() if k != "surf_rough_acf"} == m_g
    cfg = rbc.sim_cfg(p["rc_sim"], 10.0, 15.0, pair + ("exponential",),
                      gfix=0.05)
    assert cfg.interfaces[0].roughness.acf == "exponential"
    with pytest.raises(ValueError, match="grazing_fix"):
        rbc.sim_cfg(p["rc_sim"], 10.0, 15.0, pair + ("exponential",))


def test_atm_exponential_resolution():
    tab = b1.load_table()
    s, l, inf = b1.resolve_exponential("greenland_geikie01_transit", "low",
                                       tab)
    assert (s, l) == (0.0515, 5.276) and inf["acf"] == "exponential"
    _, _, inf17 = b1.resolve_exponential("greenland_westcoast", "p3_2017",
                                         tab)
    assert inf17["spectrum"] == "westcoast_2017_exp"
    with pytest.raises(ValueError, match="powerlaw"):
        b1.resolve_exponential("antarctica_getz", "real_low", tab)
    p, line0 = _p(), rbc.LINE
    rbc.activate_line("greenland_geikie01_transit")
    try:
        pair, inf = rbc.resolve_surf_rough({"source": "atm_exponential"},
                                           {**p, "key": "high"}, info=True)
    finally:
        rbc.activate_line(line0)
    assert pair == (0.0515, 5.276, "exponential")
    assert rbc.surf_rough_pair([0.01, 0.5, "exponential"]) == (0.01, 0.5,
                                                               "exponential")
    assert rbc.surf_rough_pair([0.01, 0.5, "gaussian"]) == (0.01, 0.5)
    spec = load_spec(ROOT / "config/experiments/pilot_smoke_exp.yaml")
    assert spec.to_run_kwargs()["surf_rough"] == {"source": "atm_exponential"}
    doc = spec.model_dump()
    doc["run"]["physics"]["surface_roughness"] = {
        "sigma_m": 0.01, "corr_length_m": 0.5, "acf": "exponential"}
    kw = RunSpec.model_validate(doc).to_run_kwargs()
    assert kw["surf_rough"] == [0.01, 0.5, "exponential"]


def test_b26_rough_run_keys():
    import run_b26_comparison as b26
    assert b26._rough_key(*b26._rough_run((40, "mcords"))) == \
        "firn_N40_rough_mcords"
    assert b26._rough_run((40, "mcords")) == (40, "mcords", "gaussian", False)
    assert b26._rough_key(*b26._rough_run((20, "mcords", "exponential"))) == \
        "firn_N20_rough_mcords_exp"
    assert b26._rough_key(*b26._rough_run((20, "mcords", "gaussian", 1))) == \
        "firn_N20_rough_mcords_gfx"
    with pytest.raises(ValueError):
        b26._rough_run((20, "mcords", "exponential", 0))
    cfg = b26.firn_cfg(_p()["rc_sim"], 10.0, np.array([5.0, 15.0]),
                       rough=(np.array([0.03, 0.02]), np.array([2.5, 1.5]),
                              "exponential"), gfix=True)
    assert isinstance(cfg.grazing_fix, GrazingFixConfig)
    assert cfg.interfaces[1].roughness.acf == "exponential"
