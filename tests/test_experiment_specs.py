"""Declarative run specs (tools/clutter_spec.py + config/experiments/*.yaml).

Two jobs. (1) Schema: the spec must reject the malformed and the physically
contradictory rather than silently running something else. (2) Round trip:
every committed experiment spec must reproduce the run_config.json of the
directory it claims to build -- the check that makes "this file reproduces
that result" a fact instead of a hope.

Config level only -- no network, no kernels, no simulation."""

import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_basal_clutter as rbc  # noqa: E402
from clutter_spec import (SCHEMA_VERSION, BedSource,  # noqa: E402
                          RunSpec, load_spec)

EXPERIMENTS = ROOT / "config" / "experiments"


# Read from the registry rather than hardcoded: the lines get renamed, and a
# schema test should not break when they do.
LINE = rbc.DEFAULT_LINE
SEG = next(s for s in rbc.LINES[LINE]["SEGMENTS"] if s != "full_line")
HYBRID_LINE = next((n for n in rbc.LINES
                    if "full_line" in rbc.LINES[n]["SEGMENTS"]), None)


def _doc(**over):
    """Minimal valid spec document, overridable by dotted-free kwargs."""
    d = {"schema_version": SCHEMA_VERSION,
         "meta": {"name": "demo"},
         "run": {"line": LINE, "segment": SEG,
                 "out_name": "demo", "physics": {"att_db_per_km": 20.0}}}
    for k, v in over.items():
        d["run"][k] = v
    return d


# ------------------------------------------------------------------ schema
def test_bed_source_is_an_enum_covering_every_wired_topography():
    assert {b.value for b in BedSource} == {
        "bedmachine", "picked", "demogorgn", "hybrid"}


def test_attenuation_has_no_default():
    """A silent default is how a run reproduced a REJECTED attenuation
    (foundations review A2). Every spec must state the number."""
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


def test_meta_name_must_equal_out_name():
    d = _doc()
    d["meta"]["name"] = "something_else"
    with pytest.raises(ValueError, match="must equal"):
        RunSpec.model_validate(d)


@pytest.mark.skipif(HYBRID_LINE is None, reason="no line has a full_line segment")
def test_hybrid_and_full_line_imply_each_other():
    """run() infers the hybrid bed from the segment; the spec states it, so a
    disagreement must fail rather than silently building a different bed."""
    with pytest.raises(ValueError, match="imply each other"):
        RunSpec.model_validate(_doc(bed={"source": "hybrid"}))   # not full_line
    d = _doc(line=HYBRID_LINE, segment="full_line",
             bed={"source": "demogorgn"})
    with pytest.raises(ValueError, match="imply each other"):
        RunSpec.model_validate(d)
    ok = RunSpec.model_validate(_doc(line=HYBRID_LINE, segment="full_line",
                                     bed={"source": "hybrid"}))
    assert ok.to_run_kwargs()["demogorgn_bed"] is True


def test_level_anchor_requires_rssnr_gamma():
    with pytest.raises(ValueError, match="requires gamma_from_rssnr"):
        RunSpec.model_validate(_doc(reflectivity={"anchor": "level"}))


def test_posting_div_requires_the_matched_chain():
    with pytest.raises(ValueError, match="posting_div"):
        RunSpec.model_validate(_doc(processing={"posting_div": 2}))


# ------------------------------------------------------- kwargs conversion
def test_to_run_kwargs_matches_run_signature():
    """The conversion must not drift from run(): a renamed or dropped kwarg
    has to fail here, not at minute 40 of a simulation."""
    accepted = set(inspect.signature(rbc.run).parameters)
    produced = set(RunSpec.model_validate(_doc()).to_run_kwargs())
    assert produced <= accepted, produced - accepted


def test_bed_enum_expands_to_the_historical_booleans():
    """Cache keys are built from these booleans, so the expansion is the
    thing that keeps chunk_rid/chunk_meta byte-identical."""
    cases = {"bedmachine": (False, False), "picked": (True, False),
             "demogorgn": (False, True)}
    for src, (pb, dgn) in cases.items():
        kw = RunSpec.model_validate(_doc(bed={"source": src})).to_run_kwargs()
        assert (kw["picked_bed"], kw["demogorgn_bed"]) == (pb, dgn), src


def test_synthetics_are_requested_by_naming_them_in_passes():
    """The line definition declares which synthetics exist, so naming one in
    passes: is the whole request -- the old --add-<N>km flags are gone."""
    want = list(rbc.LINES[LINE]["ORDER"]) + list(
        rbc.LINES[LINE]["SYNTHETIC_KEYS"])
    kw = RunSpec.model_validate(_doc(passes=want)).to_run_kwargs()
    assert kw["passes"] == want
    assert not any(k.startswith("add_") for k in kw)
    assert RunSpec.model_validate(_doc()).to_run_kwargs()["passes"] is None


def test_companion_is_a_flag_not_another_experiments_name():
    """The constant-gamma arm runs INSIDE this experiment now, in its own
    cache directory, so there is no sibling run to name and no cross-
    experiment dependency to resolve."""
    off = RunSpec.model_validate(
        _doc(processing={"companion": False})).to_run_kwargs()
    assert off["companion"] is False
    on = RunSpec.model_validate(
        _doc(processing={"companion": True})).to_run_kwargs()
    assert on["companion"] is True
    assert "companion_name" not in on
    with pytest.raises(ValueError):          # a directory name is no longer legal
        RunSpec.model_validate(_doc(processing={"companion": "sibling_dir"}))


def test_derived_number_carries_its_provenance():
    """`how` is free prose. It was briefly {value, from, how}, but `from`
    implied a resolvable link that nothing resolved -- a name that could go
    stale while looking authoritative."""
    d = _doc(reflectivity={"gamma_from_rssnr": True, "anchor": "level",
                           "level_deficit_db": {
                               "value": 3.56,
                               "how": "contamination-aware, from att20"}})
    spec = RunSpec.model_validate(d)
    assert spec.to_run_kwargs()["level_deficit_db"] == 3.56
    assert "att20" in spec.to_run_kwargs()["level_deficit_note"]
    with pytest.raises(ValueError, match="[Ee]xtra"):
        RunSpec.model_validate(_doc(reflectivity={
            "gamma_from_rssnr": True, "anchor": "level",
            "level_deficit_db": {"value": 1.0, "from": "somewhere"}}))
    # a bare float still works for the trivial cases
    d2 = _doc(reflectivity={"gamma_from_rssnr": True, "anchor": "level",
                            "level_deficit_db": -7.89})
    assert RunSpec.model_validate(d2).to_run_kwargs()[
        "level_deficit_db"] == -7.89


# ------------------------------------------------------------- round trip
# spec name -> the output directory it claims to build
RECORDED = {
    "ant_full_line": "outputs/antarctica_getz/full_line",
}
# ant_full_line was built STAGED (one --passes per invocation), so its
# recorded config holds only the last pass -- see foundations review A3.
STAGED = {"ant_full_line"}


def test_every_committed_spec_loads():
    files = sorted(EXPERIMENTS.glob("*.yaml"))
    assert files, "no experiment specs found"
    for fp in files:
        spec = load_spec(fp)
        assert spec.meta.name == (spec.run.out_name or spec.meta.name)
        assert fp.stem.endswith(spec.meta.name), (
            f"{fp.name}: filename must end with meta.name {spec.meta.name!r}")


def _norm_ts(v):
    return None if v is None else [float(x) for x in
                                   (v if isinstance(v, list) else [v])]


@pytest.mark.parametrize("name,rel", sorted(RECORDED.items()))
def test_spec_reproduces_its_recorded_run(name, rel):
    cfg_path = ROOT / rel / "run_config.json"
    if not cfg_path.exists():
        pytest.skip(f"{rel} not present (outputs are not committed)")
    kw = load_spec(EXPERIMENTS / f"{name}.yaml").to_run_kwargs()
    spec = load_spec(EXPERIMENTS / f"{name}.yaml")
    cfg = json.loads(cfg_path.read_text())
    rg = cfg.get("rssnr_gamma", {})

    assert kw["segment"] == cfg["segment"]
    assert kw["att"] == cfg["att_db_per_km"]
    assert kw["surf_rough"] == cfg["surf_rough"]
    assert kw["picked_bed"] == cfg["picked_bed"]
    assert kw["gamma_rssnr"] == cfg["gamma_rssnr"]
    assert kw["demogorgn_bed"] == cfg["demogorgn_bed"]
    assert kw["antenna"] == cfg["antenna"]
    assert kw["posting_div"] == cfg.get("posting_div", 1)
    assert kw["per_pass_figs"] == cfg.get("per_pass_figs", False)
    assert kw["plot_s_max_km"] == cfg.get("plot_s_max_km")
    assert spec.run.figures.width_scale == cfg.get("fig_width_scale", 1.0)
    if "line" in cfg:                      # older records predate the key
        assert kw["line"] == cfg["line"]
    if kw["gamma_rssnr"]:
        assert kw["anchor"] == rg.get("anchor")
        assert kw["level_deficit_db"] == rg.get("level_anchor", {}).get(
            "deficit_db")
    if cfg.get("trace_decomp_s_km") is not None:
        assert _norm_ts(kw["trace_decomp_s_km"]) == _norm_ts(
            cfg["trace_decomp_s_km"])
    if name in STAGED:
        assert set(cfg["passes"]) <= set(kw["passes"])
    else:
        assert sorted(kw["passes"]) == sorted(cfg["passes"])


def test_unknown_line_rejected_at_load():
    """A line name is a key into run_basal_clutter.LINES. Typing it wrong
    used to load clean and fail only after the scene prep."""
    d = _doc()
    d["run"]["line"] = "greenland_2014_2018"
    with pytest.raises(ValueError, match="unknown line"):
        RunSpec.model_validate(d)


def test_every_spec_names_a_registered_line():
    for fp in sorted(EXPERIMENTS.glob("*.yaml")):
        r = load_spec(fp).run
        for name in ([r.line] if r.line else r.lines):
            assert name in rbc.LINES, fp.name


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


def test_solve_sentinels_reach_run_kwargs_verbatim():
    d = _doc(reflectivity={"gamma_from_rssnr": True, "anchor": "level",
                           "level_deficit_db": "solve"})
    d["run"]["physics"]["att_db_per_km"] = "solve"
    kw = RunSpec.model_validate(d).to_run_kwargs()
    assert kw["level_deficit_db"] == "solve"
    assert kw["att"] == "solve"
    assert kw["level_deficit_note"] is None      # the solve writes its own
