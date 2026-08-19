"""Study-wide measurement conventions (config/analysis.yaml).

These define what the metrics MEAN, so the tests here are mostly about who is
allowed to change them: a LINE may override a subset where the data genuinely
differs, an EXPERIMENT may not (that would be metric shopping)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_basal_clutter as rbc  # noqa: E402
from clutter_analysis import load_analysis  # noqa: E402
from clutter_spec import RunSpec  # noqa: E402


def test_shipped_analysis_binds_every_declared_global():
    g = load_analysis().to_globals()
    assert set(g) == set(rbc.ANALYSIS_GLOBALS)
    for k, v in g.items():
        assert getattr(rbc, k) == v, k


def test_science_critical_values_are_the_recorded_ones():
    """Spot-check the numbers the findings notes quote, so a stray edit to
    the YAML shows up as a test failure rather than as a moved result."""
    a = load_analysis()
    assert a.coverage.clutter_margin_us == 3.0     # THE reach parameter
    assert a.windows.midcolumn.after_surface_us == 1.0
    assert a.windows.bed.before_us == 0.5 and a.windows.bed.after_us == 1.5
    assert tuple(a.bed_tail.fit_us) == (0.5, 3.5)
    assert a.bed_tail.guard_db == 10.0
    assert a.compute.chunk_m == 10500.0            # part of the cache key


def test_unknown_override_key_is_refused():
    """A typo in a line's analysis block must fail, not silently add a
    convention nothing reads."""
    with pytest.raises(ValueError, match="unknown key"):
        load_analysis().merged({"bed_tail": {"fit_uss": [0.5, 3.5]}})


def test_line_override_merges_and_reports_what_changed():
    base = load_analysis()
    merged, changed = base.merged(
        {"noise_floor": {"record_end_window_us": [6.0, 4.0]}})
    assert tuple(merged.noise_floor.record_end_window_us) == (6.0, 4.0)
    # everything else survives the merge
    assert merged.bed_tail.fit_us == base.bed_tail.fit_us
    assert merged.coverage.clutter_margin_us == base.coverage.clutter_margin_us
    # and the deviation is reported, not silent
    assert "noise_floor.record_end_window_us" in changed
    assert tuple(changed["noise_floor.record_end_window_us"]["default"]) \
        == (12.0, 8.0)


def test_shipped_lines_declare_no_overrides():
    """Both study lines currently measure by the study conventions. If that
    changes, the override must be deliberate -- this test is the prompt."""
    for name, (_resolved, changed) in rbc.LINE_ANALYSIS.items():
        assert changed == {}, f"{name} overrides {sorted(changed)}"


def test_an_experiment_cannot_set_measurement_conventions():
    """The whole point of the split: per-run windows would let someone move
    the bed window until the residual looked right."""
    doc = {"schema_version": 1, "meta": {"name": "demo"},
           "run": {"line": "antarctic_2016", "segment": "full",
                   "out_name": "demo", "analysis": {"bed_tail": {}},
                   "physics": {"att_db_per_km": 20.0}}}
    with pytest.raises(ValueError, match="[Ee]xtra"):
        RunSpec.model_validate(doc)


def test_line_scoped_constants_resolve_at_call_time():
    """ROUGH_WIN_M / CORR_WIN_M are line-overridable, so binding them as
    default arguments would freeze the study value at import."""
    import inspect
    for fn in (rbc.roughness_rms, rbc._smooth_db):
        assert inspect.signature(fn).parameters["win_m"].default is None


# ------------------------------------------------- the level-anchor rule
def test_level_anchor_rule_is_declared_once():
    a = load_analysis().level_anchor
    assert a.method == "contamination_aware"
    assert a.min_bed_over_surface_db == 10.0
    assert a.combine == "median"
    assert rbc.LEVEL_ANCHOR_RULE == a.model_dump()


# Recorded bed-window levels (dB rel own surface peak) of the constant-gamma
# runs the two adopted deficits were solved from.
_ANT = {"real_low":  dict(measured=-54.28, sim_bed=-53.00, sim_surface=-89.83),
        "real_9km":  dict(measured=-45.95, sim_bed=-50.08, sim_surface=-68.86),
        "real_10km": dict(measured=-46.11, sim_bed=-49.68, sim_surface=-71.95)}
_GL = {"low":  dict(measured=-107.76, sim_bed=-99.87, sim_surface=-110.20),
       "high": dict(measured=-83.95,  sim_bed=-94.33, sim_surface=-90.19)}


def test_one_rule_reproduces_the_antarctic_deficit():
    """The Antarctic line already solved the contamination-aware form, so
    unifying must leave its number untouched."""
    d, rec = rbc.solve_level_deficit(_ANT)
    assert d == pytest.approx(3.56, abs=0.01)
    assert rec["n_qualifying"] == 3           # all margins are +18 dB or more


def test_the_same_rule_moves_the_greenland_deficit():
    """Greenland used a PLAIN difference, which credits the bed with power
    the surface supplied. Invisible on the Antarctic line, worth 3.67 dB
    here, where the bed stands only 10.3 dB clear."""
    d, rec = rbc.solve_level_deficit(_GL)
    assert d == pytest.approx(-11.56, abs=0.01)
    plain, _ = rbc.solve_level_deficit(
        _GL, {**rbc.LEVEL_ANCHOR_RULE, "method": "plain_difference"})
    assert plain == pytest.approx(-7.89, abs=0.01)      # the retired value
    assert d - plain == pytest.approx(-3.67, abs=0.02)


def test_surface_dominated_passes_are_excluded_automatically():
    """The exclusion is DERIVED from the decomposition, not a hand-written
    pass list -- one threshold reproduces both lines' hand-made choices."""
    _, rec = rbc.solve_level_deficit(_GL)
    assert rec["per_pass"]["low"]["qualifies"] is True
    assert rec["per_pass"]["high"]["qualifies"] is False
    assert rec["per_pass"]["high"]["bed_over_surface_db"] < 0
    _, ant = rbc.solve_level_deficit(_ANT)
    assert all(v["qualifies"] for v in ant["per_pass"].values())


def test_no_qualifying_pass_fails_loudly():
    """If attenuation dims the bed until nothing clears the threshold, D is
    unsolvable -- it must say so, not fall back to a contaminated pass."""
    only_high = {"high": _GL["high"]}
    with pytest.raises(ValueError, match="unsolvable"):
        rbc.solve_level_deficit(only_high)


def test_committed_specs_state_the_rule_solved_value():
    """A spec's D must equal what the rule yields; otherwise the declared
    convention and the number in use disagree."""
    from clutter_spec import load_spec
    exp = ROOT / "config" / "experiments"
    want = {"ant_full_line": 3.56, "gl_full_pbed_rssnr": -11.56}
    for name, d in want.items():
        fp = exp / f"{name}.yaml"
        if not fp.exists():
            pytest.skip(f"{name} not present")
        assert load_spec(fp).run.reflectivity.deficit_db == pytest.approx(d)
