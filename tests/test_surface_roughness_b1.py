"""Per-run surface roughness (path B1): spec parsing, per-pass resolution
from the ATM table, and chunk-cache key backward compatibility -- the
fixture (and the boolean) must keep every pre-existing ``_srough`` rid and
meta byte-identical; any other pair forks both."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_altitude_comparison as rac  # noqa: E402
import run_basal_clutter as rbc  # noqa: E402
import surface_roughness_b1 as b1  # noqa: E402
from clutter_spec import RunSpec, load_spec  # noqa: E402
from test_basal_hypotheses import _p  # noqa: E402

FIX = (rac.SURF_ROUGH_SIGMA_M, rac.SURF_ROUGH_CL_M)


def test_fixture_and_bool_keep_legacy_keys():
    p, rows = _p(), np.arange(198)
    rid_t = rbc.chunk_rid(p, 0, rac.ATT_DB_PER_KM, True)
    assert rid_t == "low_full_dgn_rssnr_proc_c00_srough"
    assert rbc.chunk_rid(p, 0, rac.ATT_DB_PER_KM, FIX) == rid_t
    assert rbc.chunk_rid(p, 0, rac.ATT_DB_PER_KM, list(FIX)) == rid_t
    m_t = rbc.chunk_meta(p, 0, rows, 17, 3365, rac.ATT_DB_PER_KM, True)
    assert rbc.chunk_meta(p, 0, rows, 17, 3365, rac.ATT_DB_PER_KM, FIX) == m_t
    assert "surf_rough_sigma_l" not in m_t and m_t["surf_rough"] is True
    cfg = rbc.sim_cfg(p["rc_sim"], 10.0, 15.0, FIX)
    assert cfg.interfaces[0].roughness.sigma_m == rac.SURF_ROUGH_SIGMA_M


def test_other_pair_forks_rid_and_meta():
    p, rows = _p(), np.arange(198)
    rid = rbc.chunk_rid(p, 0, rac.ATT_DB_PER_KM, (0.0118, 0.551))
    assert rid == "low_full_dgn_rssnr_proc_c00_srough_sr0.0118_0.551"
    m = rbc.chunk_meta(p, 0, rows, 17, 3365, rac.ATT_DB_PER_KM,
                       (0.0118, 0.551))
    assert m["surf_rough"] is True
    assert m["surf_rough_sigma_l"] == [0.0118, 0.551]
    cfg = rbc.sim_cfg(p["rc_sim"], 10.0, 15.0, (0.0118, 0.551))
    assert cfg.interfaces[0].roughness.sigma_m == 0.0118
    assert cfg.interfaces[0].roughness.corr_length_m == 0.551
    assert (rbc.chunk_rid(p, 0, rac.ATT_DB_PER_KM, False)
            == "low_full_dgn_rssnr_proc_c00")


def test_tangent_rule_matches_value_and_slope():
    S = b1.spectrum({"family": "powerlaw", "A_m4": 1e-4, "beta": 2.7})
    kb = b1.bragg_k(195e6, 30.0)
    sig, l = b1.tangent_pair(S, kb)
    assert b1.gaussian_psd(kb, sig, l) == pytest.approx(S(kb), rel=1e-6)
    for f in (0.9, 1.1):        # tangent: within ~0.15 dB at +-10 % in k
        r = 10 * np.log10(b1.gaussian_psd(kb * f, sig, l) / S(kb * f))
        assert abs(r) < 0.15
    assert l == pytest.approx(np.sqrt(2 * 2.7) / kb)


def test_atm_table_resolves_one_spectrum_per_line():
    tab = b1.load_table()
    s17, l17, inf = b1.resolve("greenland_westcoast", "p3_2017", 195e6,
                               30.0, tab)
    assert inf["spectrum"] == "westcoast_2017"
    assert 0.005 < s17 < 0.02 and 0.4 < l17 < 0.7
    s16, l16, inf16 = b1.resolve("greenland_westcoast", "p3_2016", 195e6,
                                 30.0, tab)
    assert inf16["spectrum"] == "westcoast_2017"
    assert (s16, l16) == pytest.approx((s17, l17))
    _, _, inf_h = b1.resolve("greenland_westcoast", "haps_14km", 60e6,
                             30.0, tab)
    assert inf_h["spectrum"] == "westcoast_2017"
    _, lg, infg = b1.resolve("greenland_geikie01_transit", "p3_2017_high", 195e6,
                             30.0, tab)
    assert infg["family"] == "exponential" and 0.5 < lg < 0.7
    with pytest.raises(KeyError):
        b1.resolve("nowhere", "x", 195e6, 30.0, tab)
    # a smaller clutter angle -> smaller k_B -> longer l
    _, l20, _ = b1.resolve("antarctica_getz", "dc8_2016_11km", 190e6, 20.0, tab)
    _, l40, _ = b1.resolve("antarctica_getz", "dc8_2016_11km", 190e6, 40.0, tab)
    assert l20 > l17 > l40


def test_resolve_surf_rough_in_runner():
    p, line0 = _p(), rbc.LINE
    rbc.activate_line("greenland_westcoast")
    try:
        pair, inf = rbc.resolve_surf_rough({"source": "atm_b1"},
                                           {**p, "key": "p3_2017"}, info=True)
        pair40 = rbc.resolve_surf_rough({"source": "atm_b1",
                                         "theta_c_deg": 40.0},
                                        {**p, "key": "p3_2017"})
    finally:
        rbc.activate_line(line0)
    assert inf["spectrum"] == "westcoast_2017" and inf["theta_c_deg"] == 30.0
    assert pair40[1] < pair[1]
    assert rbc.resolve_surf_rough(True, p) == FIX
    assert rbc.resolve_surf_rough(False, p) is None
    assert rbc.resolve_surf_rough([0.01, 0.5], p) == (0.01, 0.5)
    with pytest.raises(ValueError):
        rbc.resolve_surf_rough({"source": "other"}, p)


def test_spec_surface_roughness_forms():
    base = load_spec(ROOT / "config/experiments/pilot.yaml")
    doc = base.model_dump()
    doc["run"]["physics"]["surface_roughness"] = {"source": "atm_b1"}
    assert RunSpec.model_validate(doc).to_run_kwargs()["surf_rough"] == {
        "source": "atm_b1"}
    doc["run"]["physics"]["surface_roughness"] = {"sigma_m": 0.01,
                                                  "corr_length_m": 0.5}
    kw = RunSpec.model_validate(doc).to_run_kwargs()
    assert kw["surf_rough"] == [0.01, 0.5]
    doc["run"]["physics"]["surface_roughness"] = False
    assert RunSpec.model_validate(doc).to_run_kwargs()["surf_rough"] is False
    assert base.to_run_kwargs()["surf_rough"] == {
        "source": "atm_exponential"}
