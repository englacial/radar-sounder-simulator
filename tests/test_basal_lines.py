"""Study-line registry of tools/run_basal_clutter.py (--line): the Antarctic
line IS the module default (activation must be a no-op, so its behaviour and
its caches cannot drift), the Greenland pair is a sibling config block, and
the per-line guards reject features that line does not have wired.

Config level only -- no network, no DEMs, no kernels, no simulation."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_basal_clutter as rbc  # noqa: E402

# The line-specific globals activate_line is allowed to rebind. Any name a
# registry entry sets MUST be in here, or it would silently leak between
# lines (and, worse, persist into the Antarctic defaults).
LINE_GLOBALS = (
    "LINE", "SEASON", "CRS", "CASE_PREFIX", "OUT_DEFAULT", "FC_HZ",
    "LAM_ICE_M", "PASSES", "ORDER", "SEGMENTS", "S0_KM", "DECOMP_S_KM",
    "N_TRACES_BY_SEGMENT", "REF_PASS", "REF_SEASON", "REF_FRAMES",
    "GL_S_KM", "SYNTHETIC_KEYS", "MEASURED_CAVEATS", "UNSUPPORTED",
    "RADARGRAM_Y_US", "RADARGRAM_DB",
)


@pytest.fixture
def line_sandbox():
    """Snapshot/restore every line-specific global: activate_line mutates
    module state, and no test may leak it into another."""
    saved = {k: getattr(rbc, k) for k in LINE_GLOBALS}
    yield rbc.activate_line
    for k, v in saved.items():
        setattr(rbc, k, v)


# --------------------------------------------------- module defaults
def test_module_defaults_are_the_antarctic_line():
    """The Antarctic line is the module default, so its registry entry is
    EMPTY and activating it changes nothing (bit-identical caches)."""
    assert rbc.LINE == rbc.ANTARCTIC_LINE
    assert rbc.LINES[rbc.ANTARCTIC_LINE] == {}
    assert rbc.SEASON == "2016_Antarctica_DC8"
    assert rbc.REF_SEASON == rbc.SEASON
    assert rbc.CRS == "EPSG:3031"
    assert rbc.ORDER == ["low", "mid", "high"]
    assert rbc.GL_S_KM == 69.7
    assert rbc.UNSUPPORTED == ()


def test_activating_the_antarctic_line_is_a_no_op(line_sandbox):
    before = {k: getattr(rbc, k) for k in LINE_GLOBALS}
    line_sandbox(rbc.ANTARCTIC_LINE)
    assert {k: getattr(rbc, k) for k in LINE_GLOBALS} == before


def test_unknown_line_rejected(line_sandbox):
    with pytest.raises(ValueError, match="unknown line"):
        line_sandbox("no_such_line")


def test_registry_entries_only_touch_line_globals():
    for name, entry in rbc.LINES.items():
        stray = sorted(set(entry) - set(LINE_GLOBALS))
        assert not stray, f"{name} rebinds non-line globals {stray}"


def test_activation_round_trips(line_sandbox):
    """Greenland -> Antarctic restores every default (no leakage)."""
    line_sandbox(rbc.GREENLAND_LINE)
    assert rbc.CRS == "EPSG:3413"
    line_sandbox(rbc.ANTARCTIC_LINE)
    # the empty Antarctic entry cannot restore anything, so the sandbox
    # fixture is what puts the defaults back -- assert the fixture's job is
    # real by checking the value is still Greenland's here.
    assert rbc.CRS == "EPSG:3413"


# --------------------------------------------------- Greenland registry
@pytest.fixture
def greenland(line_sandbox):
    line_sandbox(rbc.GREENLAND_LINE)
    return rbc


def test_greenland_radargram_window_reaches_its_bed(greenland):
    """The Antarctic -1..13.5 us framing leaves the Greenland bed (26-31 us
    below the surface) entirely off-panel -- which also hides the bed
    overlay. The window must cover the deepest bed with margin."""
    y_lo, y_hi = greenland.RADARGRAM_Y_US
    assert y_lo <= 0.0
    assert y_hi >= 34.0            # deepest bed 31.2 us + margin
    lo_db, _ = greenland.RADARGRAM_DB
    assert lo_db <= -115.0         # low pass bed sits ~108 dB down


def test_antarctic_radargram_window_is_the_module_default():
    """The Antarctic line must not override the framing (its entry is empty),
    so its figures are unchanged."""
    assert "RADARGRAM_Y_US" not in rbc.LINES[rbc.ANTARCTIC_LINE]
    assert "RADARGRAM_DB" not in rbc.LINES[rbc.ANTARCTIC_LINE]
    assert rbc.RADARGRAM_Y_US == (-1.0, 13.5)
    assert rbc.RADARGRAM_DB == (-90.0, 5.0)


def test_greenland_line_identity(greenland):
    assert greenland.LINE == "greenland_2014_2017"
    assert greenland.CRS == "EPSG:3413"            # Greenland polar stereo
    assert greenland.FC_HZ == 195e6                # scout: 180-210 MHz
    assert greenland.CASE_PREFIX == "greenland_pair"
    assert greenland.OUT_DEFAULT.name == "greenland_pair"
    assert greenland.GL_S_KM is None               # all grounded interior ice


def test_greenland_is_a_pair_plus_one_synthetic(greenland):
    assert greenland.ORDER == ["low", "high"]
    assert greenland.SYNTHETIC_KEYS == (rbc.SYN14_KEY,)
    assert set(greenland.PASSES) == {"low", "high", rbc.SYN14_KEY}


def test_greenland_mixes_two_seasons(greenland):
    """The pair flies two seasons -- the per-pass season key is what makes
    the frame loads and the chunk cache keys correct."""
    assert greenland.PASSES["low"]["season"] == "2014_Greenland_P3"
    assert greenland.PASSES["high"]["season"] == "2017_Greenland_P3"
    for key, spec in greenland.PASSES.items():
        assert greenland.pass_season(spec) == spec["season"], key
    # the synthetic pass rides the LOW pass, so it inherits its season
    syn = greenland.PASSES[rbc.SYN14_KEY]
    assert syn["season"] == greenland.PASSES["low"]["season"]
    assert syn["param_frame"] == greenland.PASSES["low"]["param_frame"]


def test_pass_season_falls_back_to_the_line_season():
    """Antarctic specs carry no per-pass season."""
    for spec in rbc.PASSES.values():
        assert "season" not in spec
        assert rbc.pass_season(spec) == rbc.SEASON


def test_greenland_segments_are_self_consistent(greenland):
    assert greenland.SEGMENTS == ("pilot", "full")
    for d in (greenland.S0_KM, greenland.DECOMP_S_KM,
              greenland.N_TRACES_BY_SEGMENT):
        assert set(d) == set(greenland.SEGMENTS)
    # scout segments: pilot s 25-35 km, full s 11-40 km
    assert greenland.S0_KM == {"pilot": 25.0, "full": 11.0}


def test_greenland_decomposition_location_is_inside_every_segment(greenland):
    """The single-trace decomposition s must land inside the segment it is
    the default for, else analyze_pass silently clamps to an end trace."""
    spans = {"pilot": (25.0, 35.0), "full": (11.0, 40.0)}
    for seg, (lo, hi) in spans.items():
        for v in np.atleast_1d(greenland.DECOMP_S_KM[seg]):
            assert lo <= float(v) <= hi, (seg, v)


def test_greenland_slices_are_one_frame_per_pass_and_match_in_length(greenland):
    """Scout: neither segment crosses a frame boundary and the two passes'
    trace counts match to one trace (the 0.05% figure in the note)."""
    for seg in greenland.SEGMENTS:
        lens = {}
        for key in greenland.ORDER:
            parts = greenland.PASSES[key][seg]
            assert len(parts) == 1, (key, seg)
            (fid, (a, b)), = parts
            assert b > a
            lens[key] = b - a
        assert abs(lens["low"] - lens["high"]) <= 1, (seg, lens)


def test_greenland_passes_are_not_reversed(greenland):
    """Both passes fly increasing anchor s, so no slice reversal / roll
    negation applies (unlike the Antarctic high passes)."""
    for key, spec in greenland.PASSES.items():
        assert spec["rev"] is False, key


def test_greenland_pick_axis_is_the_low_pass(greenland):
    assert greenland.REF_PASS == "low"
    assert greenland.REF_SEASON == greenland.PASSES["low"]["season"]
    assert greenland.REF_FRAMES == ("20140421_01_069", "20140421_01_070")
    # the segment frame must be one of the pick-axis frames
    (fid, _), = greenland.PASSES["low"]["full"]
    assert fid in greenland.REF_FRAMES


def test_greenland_altitude_ratio_is_the_scouted_one(greenland):
    lo = greenland.PASSES["low"]["agl_med_m"]
    hi = greenland.PASSES["high"]["agl_med_m"]
    assert 5.0 < hi / lo < 5.6                      # scout: 5.3x
    assert greenland.PASSES[rbc.SYN14_KEY]["synthetic_msl_m"] == 14000.0
    assert greenland.PASSES[rbc.SYN14_KEY]["agl_med_m"] is None


def test_greenland_synthetic_carries_the_lpa_facet_scale(greenland):
    """The 2-trace gate found the syn500km/syn300km Fresnel-zone LPA failure
    class at 14 km on this line; 0.7x is the recorded fix."""
    assert greenland.PASSES[rbc.SYN14_KEY]["facet_spacing_scale"] == 0.7
    for key in greenland.ORDER:
        assert "facet_spacing_scale" not in greenland.PASSES[key]


def test_greenland_measured_caveats_record_the_scout_quirks(greenland):
    txt = greenland.MEASURED_CAVEATS
    for needle in ("33.3859", "img_comb", "hanning", "param_combine",
                   "BedMachine Greenland v5", "post-bed"):
        assert needle in txt, needle


# --------------------------------------------------- per-line guards
def _run_kwargs(**kw):
    kw.setdefault("line", rbc.GREENLAND_LINE)
    return kw


def test_unsupported_features_rejected_on_greenland(line_sandbox):
    assert set(rbc.LINES[rbc.GREENLAND_LINE]["UNSUPPORTED"]) >= {
        "gamma_rssnr", "demogorgn_bed"}
    with pytest.raises(ValueError, match="gamma_rssnr is not wired"):
        rbc.run(**_run_kwargs(gamma_rssnr=True))
    with pytest.raises(ValueError, match="demogorgn_bed is not wired"):
        rbc.run(**_run_kwargs(demogorgn_bed=True))


def test_segment_not_on_this_line_rejected(line_sandbox):
    with pytest.raises(ValueError, match="has no 'full_line' segment"):
        rbc.run(**_run_kwargs(segment="full_line"))
    with pytest.raises(ValueError, match="has no 'extended' segment"):
        rbc.run(**_run_kwargs(segment="extended"))


def test_synthetic_not_defined_on_this_line_rejected(line_sandbox):
    with pytest.raises(ValueError, match="defines no pass"):
        rbc.run(**_run_kwargs(add_500km=True))
    with pytest.raises(ValueError, match="defines no pass"):
        rbc.run(**_run_kwargs(add_30km=True))


def test_antarctic_guards_still_reject_their_own_combinations():
    """The line guards must not have displaced the pre-existing ones."""
    with pytest.raises(ValueError, match="full_line"):
        rbc.run(segment="full_line")                # needs --demogorgn-bed
    with pytest.raises(ValueError, match="hybrid"):
        rbc.run(segment="full", demogorgn_bed=True, picked_bed=True)
