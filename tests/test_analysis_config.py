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
