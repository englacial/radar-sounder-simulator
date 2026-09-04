"""Declarative run specs (tools/clutter_spec.py + config/experiments/*.yaml).

Schema: the spec must reject the malformed and the physically contradictory
rather than silently running something else. Shipped specs: exactly two
(full, pilot), identical apart from the segment, valid on every line.

Config level only -- no network, no kernels, no simulation."""

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_basal_clutter as rbc  # noqa: E402
from clutter_spec import SCHEMA_VERSION, RunSpec, load_spec  # noqa: E402

EXPERIMENTS = ROOT / "config" / "experiments"

LINE = rbc.DEFAULT_LINE
SEG = next(s for s in rbc.LINES[LINE]["SEGMENTS"]
           if s not in rbc.LINES[LINE]["SEGMENTS_CROSSING_GL"])


def _doc(**over):
    """Minimal valid spec document."""
    d = {"schema_version": SCHEMA_VERSION,
         "meta": {"name": "demo"},
         "run": {"line": LINE, "segment": SEG,
                 "physics": {"att_db_per_km": 20.0}}}
    for k, v in over.items():
        d["run"][k] = v
    return d


# ------------------------------------------------------------------ schema
def test_attenuation_has_no_default():
    """A silent default is how a run reproduced a REJECTED attenuation.
    Every spec must state the number or 'solve'."""
    d = _doc()
    del d["run"]["physics"]["att_db_per_km"]
    with pytest.raises(ValueError, match="att_db_per_km"):
        RunSpec.model_validate(d)


def test_typos_are_rejected_not_ignored():
    d = _doc()
    d["run"]["physics"]["att_db_per_kmm"] = 20.0
    with pytest.raises(ValueError, match="[Ee]xtra"):
        RunSpec.model_validate(d)


def test_unknown_schema_version_rejected():
    d = _doc()
    d["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema_version"):
        RunSpec.model_validate(d)


def test_analysis_conventions_cannot_be_set_per_experiment():
    with pytest.raises(ValueError, match="[Ee]xtra"):
        RunSpec.model_validate(_doc(analysis={"bed": {"after_us": 3.0}}))


def test_posting_div_requires_the_matched_chain():
    with pytest.raises(ValueError, match="posting_div"):
        RunSpec.model_validate(_doc(processing={"posting_div": 2}))


def test_fixed_angle_focus_carries_its_half_angle():
    kw = RunSpec.model_validate(_doc(processing={
        "chain": "standard", "focus_aperture": "fixed_angle",
        "focus_half_angle_deg": 4.0})).to_run_kwargs()
    assert kw["focus_aperture"] == "fixed_angle"
    assert kw["focus_half_angle_deg"] == 4.0
    with pytest.raises(ValueError):
        RunSpec.model_validate(_doc(processing={
            "chain": "standard", "focus_aperture": "fixed_angle",
            "focus_half_angle_deg": 0.0}))


def test_focus_aperture_requires_the_matched_chain():
    with pytest.raises(ValueError, match="focus_aperture"):
        RunSpec.model_validate(
            _doc(processing={"focus_aperture": "first_fresnel"}))
    kw = RunSpec.model_validate(_doc(processing={
        "chain": "standard", "focus_aperture": "first_fresnel",
    })).to_run_kwargs()
    assert kw["focus_aperture"] == "first_fresnel"
    kw = RunSpec.model_validate(_doc(processing={
        "chain": "standard", "focus_aperture": "product_resolution",
    })).to_run_kwargs()
    assert kw["focus_aperture"] == "product_resolution"


# ------------------------------------------------------- kwargs conversion
def test_to_run_kwargs_matches_run_signature():
    """A renamed or dropped kwarg has to fail here, not at minute 40."""
    accepted = set(inspect.signature(rbc.run).parameters)
    produced = set(RunSpec.model_validate(_doc()).to_run_kwargs())
    assert produced <= accepted, produced - accepted


def test_bed_method_maps_to_run_flags_and_leaves_the_dem_to_the_line():
    """bed.nadir is the experiment's method; the DEM (bedmachine or
    DEMOGORGN) is the line's data, so the spec passes demogorgn_bed=None
    and run() resolves it from the line's bed_dem."""
    kw = RunSpec.model_validate(_doc()).to_run_kwargs()
    assert kw["picked_bed"] is True and kw["demogorgn_bed"] is None
    kw = RunSpec.model_validate(_doc(bed={"nadir": "dem"})).to_run_kwargs()
    assert kw["picked_bed"] is False
    with pytest.raises(ValueError):
        RunSpec.model_validate(_doc(bed={"floating": "dem"}))
    with pytest.raises(ValueError):
        RunSpec.model_validate(_doc(bed={"source": "hybrid"}))


def test_out_name_is_the_meta_name():
    kw = RunSpec.model_validate(_doc()).to_run_kwargs()
    assert kw["out_name"] == "demo"


def test_companion_is_a_flag():
    off = RunSpec.model_validate(
        _doc(processing={"companion": False})).to_run_kwargs()
    assert off["companion"] is False
    with pytest.raises(ValueError):
        RunSpec.model_validate(_doc(processing={"companion": "sibling_dir"}))


def test_unknown_line_rejected_at_load():
    d = _doc()
    d["run"]["line"] = "greenland_2014_2018"
    with pytest.raises(ValueError, match="unknown line"):
        RunSpec.model_validate(d)


def test_multi_line_protocol_requires_exactly_one_of_line_or_lines():
    d = _doc()
    d["run"]["lines"] = [LINE]
    with pytest.raises(ValueError, match="exactly one"):
        RunSpec.model_validate(d)
    del d["run"]["line"]
    ok = RunSpec.model_validate(d)
    assert ok.to_run_kwargs()["line"] is None    # resolved by --line at run
    del d["run"]["lines"]
    with pytest.raises(ValueError, match="exactly one"):
        RunSpec.model_validate(d)


def test_solve_sentinel_reaches_run_kwargs_verbatim():
    d = _doc(reflectivity={"gamma_from_rssnr": True})
    d["run"]["physics"]["att_db_per_km"] = "solve"
    assert RunSpec.model_validate(d).to_run_kwargs()["att"] == "solve"


def test_reference_carrier_is_accepted_by_the_schema():
    d = _doc(extra_passes={"haps_14km": {
        "carrier": "reference", "altitude_m": 14000.0,
        "instrument": "haps_60mhz"}})
    sp = RunSpec.model_validate(d)
    assert sp.to_run_kwargs()["extra_passes"]["haps_14km"]["carrier"] \
        == "reference"


# ------------------------------------------------------- shipped specs
def test_exactly_full_and_pilot_are_shipped():
    files = sorted(EXPERIMENTS.glob("*.yaml"))
    assert [f.stem for f in files] == ["full", "pilot"]
    for fp in files:
        spec = load_spec(fp)
        assert fp.stem == spec.meta.name


def test_full_and_pilot_differ_only_in_segment():
    """One protocol, two extents: any other difference would make the pilot
    a different experiment rather than the cheap end of the same loop."""
    full = load_spec(EXPERIMENTS / "full.yaml").run.model_dump()
    pilot = load_spec(EXPERIMENTS / "pilot.yaml").run.model_dump()
    assert full.pop("segment") == "full" and pilot.pop("segment") == "pilot"
    assert full == pilot


def test_shipped_specs_cover_every_line_with_the_same_haps_points():
    for name in ("full", "pilot"):
        sp = load_spec(EXPERIMENTS / f"{name}.yaml")
        assert set(sp.run.lines) == set(rbc.LINES), name
        for line in sp.run.lines:
            assert sp.run.segment in rbc.LINES[line]["SEGMENTS"], (name, line)
        ep = sp.run.extra_passes
        assert set(ep) == {"haps_14km_halflambda", "haps_14km_lambda"}
        for key, inst in (("haps_14km_halflambda", "haps_60mhz_6el_halflambda"),
                          ("haps_14km_lambda", "haps_60mhz_6el_lambda")):
            assert ep[key].carrier == "reference"
            assert ep[key].altitude_m == 14000.0
            assert ep[key].instrument == inst
        assert sp.run.passes is None          # real passes + extras
        assert sp.run.physics.att_db_per_km == "solve"
        assert sp.run.reflectivity.gamma_from_rssnr is True
