"""Config-level tests for tools/run_cross_season.py (no simulation, no
network): the scout-note frame table, the pairwise difference-matrix math on
synthetic profiles, and the per-era window mapping the tool relies on."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("run_opr_comparison", "tools/run_opr_comparison.py")
_load("run_opr_coherent_bed", "tools/run_opr_coherent_bed.py")
rac = _load("run_altitude_comparison", "tools/run_altitude_comparison.py")
rcs = _load("run_cross_season", "tools/run_cross_season.py")


def test_frame_table_matches_scout():
    """The hardcoded common-window table must match the scout note
    (claude_notes/cross_season_line_scout.md)."""
    assert rcs.YEARS == ["2012", "2014", "2016", "2018"]
    exp = {"2012": ("2012_Antarctica_DC8", "20121023_04_008", (286, 1320)),
           "2014": ("2014_Antarctica_DC8", "20141029_05_013", (1, 2067)),
           "2016": ("2016_Antarctica_DC8", "20161104_05_008", (1058, 3124)),
           "2018": ("2018_Antarctica_DC8", "20181107_01_011", (1267, 3333))}
    for y, (season, fid, sl) in exp.items():
        f = rcs.FRAMES[y]
        assert f["season"] == season and f["frame"] == fid
        assert tuple(f["sl"]) == sl
        assert f["sl"][1] > f["sl"][0]
    # repeats share the same trace count over the common window; headline
    # pairs are 2012 vs each repeat
    n = {y: rcs.FRAMES[y]["sl"][1] - rcs.FRAMES[y]["sl"][0]
         for y in rcs.YEARS}
    assert n["2014"] == n["2016"] == n["2018"] == 2066 and n["2012"] == 1034
    assert rcs.HEADLINE == [("2012", "2014"), ("2012", "2016"),
                            ("2012", "2018")]
    # scout pitfalls recorded where the sim must not trust the provenance
    assert "ft_wind" in rcs.FRAMES["2014"]["quirks"]
    assert "20.202" in rcs.FRAMES["2016"]["quirks"]
    assert "roll" in rcs.FRAMES["2012"]["quirks"]


def test_window_mapping_per_era():
    """2012 tukey(0.2) -> 'none' (approximation recorded); repeats hanning ->
    'hann' (the 2014/2016 provenance-quirk fallback string maps the same)."""
    w, note = rac.map_window("tukeywin(N,0.2) (param_csarp ft_wind)")
    assert w == "none" and note and "APPROXIMATION" in note
    for s in ("hanning (param_sar ft_wind)",
              "hanning (ft_wind decode failed; CReSIS readme default)"):
        w, note = rac.map_window(s)
        assert w == "hann" and note is None


def test_pair_matrix_math():
    """Synthetic profiles with a known inter-flight difference: the matrix
    must recover the bed-minus-surface delta, a perfect capture fraction for
    an exact sim, the r^-2 surface expectation, and the profile-difference
    correlation."""
    z = np.arange(0.0, 1400.0, 2.0)

    def prof(slope_db_per_m):
        return z, slope_db_per_m * z

    analyses = {}
    bs = {"2012": -30.0, "2014": -20.0, "2016": -21.0, "2018": -22.0}
    slope = {"2012": -0.04, "2014": -0.06, "2016": -0.061, "2018": -0.062}
    for y in rcs.YEARS:
        analyses[y] = {
            "bedsurf_meas": bs[y], "bedsurf_sim": bs[y],       # exact sim
            "bedsurf_sim_fl": bs[y], "bed_snr_meas_db": 20.0,
            "surf_db_meas": -60.0, "surf_db_sim": -60.0,
            "prof_meas": prof(slope[y]), "prof_sim": prof(slope[y]),
            "prof_sim_fl": prof(slope[y])}
    agls = {"2012": 9217.0, "2014": 465.0, "2016": 446.0, "2018": 447.0}
    matrix, zz = rcs.pair_matrix(analyses, agls)
    assert set(matrix) == {"2012-2014", "2012-2016", "2012-2018",
                           "2014-2016", "2014-2018", "2016-2018"}
    m = matrix["2012-2014"]
    assert m["headline"] is True and matrix["2014-2016"]["headline"] is False
    assert m["bedsurf_delta_db"]["measured"] == -10.0
    assert m["bedsurf_delta_db"]["sim"] == -10.0
    assert m["bedsurf_delta_db"]["captured_frac"] == 1.0
    # exact sim -> perfect profile-difference agreement
    assert m["profile_diff_curve"]["corr"] > 0.9999
    assert m["profile_diff_curve"]["rms_db"] == 0.0
    # r^-2 expectation: -20 log10(9217/465) = -25.94 dB
    assert abs(m["surface_delta_db_UNCALIBRATED"]["r2_expectation"]
               - (-25.94)) < 0.01
    assert "UNCAL" in json_dumps_keys(m)
    # noise-aware rows: perfect sim + irrelevant floor -> perfect capture
    assert m["noise_aware"]["captured_frac"] == 1.0
    assert m["noise_aware"]["profile_diff_corr"] > 0.9999
    # a noise-limited frame gets flagged in the noise-aware note
    analyses["2012"]["bed_snr_meas_db"] = 0.5
    mx, _ = rcs.pair_matrix(analyses, agls)
    assert "NOISE-LIMITED" in mx["2012-2014"]["noise_aware"]["note"]
    # a wrong sim degrades the capture fraction, not the measured value
    analyses["2014"]["bedsurf_sim"] = -15.0
    matrix2, _ = rcs.pair_matrix(analyses, agls)
    m2 = matrix2["2012-2014"]
    assert m2["bedsurf_delta_db"]["measured"] == -10.0
    assert m2["bedsurf_delta_db"]["sim"] == -15.0
    assert m2["bedsurf_delta_db"]["captured_frac"] == 0.5
    # the diff band stays above the shallowest bed (533 m, scout note)
    assert rcs.DIFF_Z[1] <= 533.0


def json_dumps_keys(d):
    import json
    return json.dumps(d)


def test_peak_db():
    """peak_db picks the max power in the window and returns dB; NaN guess
    -> NaN out."""
    tw = np.arange(0.0, 10e-6, 0.1e-6)
    P = np.full((2, len(tw)), 1e-12)
    P[0, 50] = 1e-6                       # peak at 5 us
    t_guess = np.array([5.2e-6, np.nan])
    out = rcs.peak_db(P, tw, t_guess, 0.1e-6, win_us=0.8)
    assert abs(out[0] - (-60.0)) < 1e-9
    assert np.isnan(out[1])
