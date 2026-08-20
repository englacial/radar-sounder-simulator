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


@pytest.fixture(autouse=True)
def _restore_line():
    """activate_line mutates module state; tests here switch lines to read
    their rules and must not leak the last line into other modules."""
    saved = {k: getattr(rbc, k) for k in rbc.LINE_GLOBALS}
    saved_a = {k: getattr(rbc, k) for k in rbc.ANALYSIS_GLOBALS}
    yield
    for k, v in {**saved, **saved_a}.items():
        setattr(rbc, k, v)
from clutter_analysis import load_analysis  # noqa: E402
from clutter_spec import RunSpec  # noqa: E402


def test_shipped_analysis_binds_every_declared_global():
    """The bound values are the ACTIVE line's resolved conventions -- the
    study defaults with that line's declared overrides merged -- so the
    comparison target is LINE_ANALYSIS, not the bare study file."""
    g = load_analysis().to_globals()
    assert set(g) == set(rbc.ANALYSIS_GLOBALS)
    resolved, _ = rbc.LINE_ANALYSIS[rbc.LINE]
    for k, v in resolved.to_globals().items():
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


def test_an_experiment_cannot_set_measurement_conventions():
    """The whole point of the split: per-run windows would let someone move
    the bed window until the residual looked right."""
    doc = {"schema_version": 1, "meta": {"name": "demo"},
           "run": {"line": "antarctica_getz", "segment": "full",
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


# ------------------------------------------ calibration-era conventions

def test_line_overrides_declare_nothing_now():
    """The getz attenuation pin moved from an analysis override into the
    line's calibration block, so no line overrides the study conventions
    today -- the first future override must be deliberate again."""
    for name, (_resolved, changed) in rbc.LINE_ANALYSIS.items():
        assert changed == {}, f"{name} overrides {sorted(changed)}"


def test_regression_settings_are_declared_once():
    a = load_analysis()
    assert a.attenuation_regression.min_samples == 20
    assert a.attenuation_regression.min_thickness_span_m == 200.0
    assert rbc.ATTENUATION_REGRESSION == \
        a.attenuation_regression.model_dump()


def test_gamma_solve_settings_are_declared_once():
    """The gamma-surface solver's knobs live in analysis.yaml, nowhere
    else: seed (the evaluation gamma; -10 matches the retired uniform
    manual default so historic chunk caches stay warm), the verify
    tolerance, and the qualifying bed-over-surface margin."""
    a = load_analysis()
    s = a.gamma_surface_solve
    assert s.seed_db == -10.0
    assert s.tolerance_db == 0.5
    assert s.min_bed_over_surface_db == 10.0
    assert rbc.GAMMA_SURFACE_SOLVE == s.model_dump()
    assert "GAMMA_SURFACE_SOLVE" in rbc.ANALYSIS_GLOBALS


def test_every_line_declares_its_calibration():
    """gamma_surface is manual-with-why or 'solve' (the study default:
    resolved in-run by zeroing the qualifying-median bed-level residual --
    it cannot come from the regression intercept, which is degenerate with
    the mean bed reflectivity); A is manual-with-why or 'solve'."""
    from clutter_lines import load_all as _load
    for name, sp in _load().items():
        c = sp.calibration
        if c.gamma_surface_db != "solve":
            assert c.gamma_surface_db.why.strip(), name
        if c.att_db_per_km != "solve":
            assert c.att_db_per_km.why.strip(), name
    # the study default IS solve (user decision 2026-08-20): every current
    # line uses it
    for name, sp in _load().items():
        assert sp.calibration.gamma_surface_db == "solve", name
    # the two documented manual-A lines and their reasons
    lines = _load()
    getz = lines["antarctica_getz"].calibration
    assert getz.att_db_per_km.value == 20.0
    geikie = lines["greenland_geikie01_transit"].calibration
    assert geikie.att_db_per_km.value == 14.0
    assert "REJECTED" in geikie.att_db_per_km.why


def test_manual_value_requires_a_why():
    from clutter_lines import ManualValue
    with pytest.raises(ValueError):
        ManualValue(value=20.0)


def test_benchmark_and_pilot_specs_carry_no_numbers():
    from clutter_spec import load_spec
    for name in ("gl_std_benchmark", "pilot_smoke"):
        sp = load_spec(ROOT / "config" / "experiments" / f"{name}.yaml")
        assert sp.run.physics.att_db_per_km == "solve", name
        assert sp.run.reflectivity.gamma_from_rssnr is True
