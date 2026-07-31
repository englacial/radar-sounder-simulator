"""Config-level integration cover for tools/run_basal_clutter.py (NO
simulation, NO network): the pass table must match the basal-clutter scout
note exactly, and the derived cross-track reaches must reproduce the scout's
worked geometry (surface reach closed form, refracted bed reach limits)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("run_opr_comparison", "tools/run_opr_comparison.py")
_load("run_opr_coherent_bed", "tools/run_opr_coherent_bed.py")
_load("run_altitude_comparison", "tools/run_altitude_comparison.py")
rbc = _load("run_basal_clutter", "tools/run_basal_clutter.py")


def test_pass_table_matches_scout():
    """Frames, slices, direction and altitudes from claude_notes/
    basal_clutter_scout.md (pilot s=30-40 km; full segment s=18-68 km)."""
    assert rbc.SEASON == "2016_Antarctica_DC8"
    assert list(rbc.PASSES) == ["low", "mid", "high"] == rbc.ORDER

    p = rbc.PASSES["low"]
    assert not p["rev"]
    assert p["pilot"] == [("20161105_05_005", (2020, 2693))]
    assert p["full"] == [("20161105_05_005", (1212, 3333)),
                         ("20161105_05_006", (0, 1244))]
    p = rbc.PASSES["mid"]
    assert p["rev"]
    assert p["pilot"] == [("20161028_05_006", (858, 1532))]
    assert p["full"] == [("20161028_05_006", (0, 2341)),
                         ("20161028_05_005", (2308, 3337))]
    p = rbc.PASSES["high"]
    assert p["rev"]
    assert p["pilot"] == [("20161031_07_005", (337, 1011))]
    assert p["full"] == [("20161031_07_005", (0, 1820)),
                         ("20161031_07_004", (1786, 3336))]

    # scout AGL medians and pilot trace counts (673/674/674)
    assert [rbc.PASSES[k]["agl_med_m"] for k in rbc.ORDER] == \
        [442.0, 9150.0, 10684.0]
    counts = [sum(b - a for _, (a, b) in rbc.PASSES[k]["pilot"])
              for k in rbc.ORDER]
    assert counts == [673, 674, 674]
    # full-segment trace counts match the scout (3365 / 3370 / 3370)
    fcounts = [sum(b - a for _, (a, b) in rbc.PASSES[k]["full"])
               for k in rbc.ORDER]
    assert fcounts == [3365, 3370, 3370]
    # cached param provenance exists for each pass's param frame
    from soundersim.opr import CACHE_DIR
    for k in rbc.ORDER:
        fid = rbc.PASSES[k]["param_frame"]
        assert any(fid == part[0] for part in rbc.PASSES[k]["pilot"])
        assert (CACHE_DIR / f"mcords_params_{rbc.SEASON}_{fid}.json").exists()


def test_surface_reach_matches_scout_numbers():
    """Scout quirk 3 worked examples: surface clutter reaching the nadir-bed
    delay of 720 m ice (8.53 us) needs +-1.66 km at 442 m and +-5.40 km at
    10.8 km AGL."""
    assert abs(rbc.surface_reach(442.0, 8.53e-6) - 1663.0) < 25.0
    assert abs(rbc.surface_reach(10684.0, 8.53e-6) - 5383.0) < 40.0
    # monotone in altitude and in delay
    assert rbc.surface_reach(9150.0, 8.53e-6) < rbc.surface_reach(
        10684.0, 8.53e-6)
    assert rbc.surface_reach(442.0, 8.53e-6) < rbc.surface_reach(
        442.0, 11.53e-6)


def test_bed_reach_refraction_limits():
    """The refracted bed reach must (a) reduce to the homogeneous closed
    form when n_ice = 1, (b) shrink when the medium slows (n > 1), and (c)
    never bind: the surface interface's reach target includes the whole ice
    column, so ct == surface reach."""
    h, d, m = 10684.0, 468.0, 3.0e-6
    homog = rbc.surface_reach(h + d, m)     # n=1: one uniform medium
    assert abs(rbc.bed_reach(h, d, 1.0, m) - homog) / homog < 0.01
    n_ice = float(np.sqrt(3.17))
    assert rbc.bed_reach(h, d, n_ice, m) < homog
    doc = rbc.derive_reach(h_max=10826.0, dbs_max=10.58e-6, d_min=468.0)
    assert doc["ct_m"] == doc["surface_reach_m"] > doc["bed_reach_m"]
    assert not doc["capped"]
    # the derived high-pass reach: nadir-bed delay + 3 us at max AGL ~6.9 km
    assert 6500.0 < doc["ct_m"] < 7500.0
    lo = rbc.derive_reach(h_max=547.0, dbs_max=10.58e-6, d_min=468.0)
    assert 2200.0 < lo["ct_m"] < 2800.0     # low pass ~2.5 km
    assert lo["ct_m"] < doc["ct_m"]


def test_windows_and_chunking():
    """The fast-time window must extend past the clutter margin, and the
    chunking must make the 50 km segment ~5 pilot-sized chunks (linear
    wall-time projection)."""
    assert rbc.POST_BED_US >= rbc.MARGIN_US + 0.4
    assert rbc.MID_LO_US == 1.0 and rbc.MID_HI_US == 0.5
    assert max(1, int(round(10000.0 / rbc.CHUNK_M))) == 1
    assert max(1, int(round(50000.0 / rbc.CHUNK_M))) == 5
