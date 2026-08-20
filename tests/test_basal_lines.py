"""Study-line definitions (config/lines/*.yaml) and the activation mechanism.

Lines are DATA now, so this file asserts structural invariants that must hold
for whatever lines are shipped, parametrised over the registry, rather than
hardcoding two line names. The previous version pinned 'antarctic_2016' and
'greenland_2014_2017' throughout and broke wholesale the moment the lines
were renamed -- which is the failure mode a data-driven registry is supposed
to remove.

Line-SPECIFIC facts (frame slices, altitudes, scout quirks) live in the YAML
with their provenance; what is checked here is that the YAML is coherent and
that activation is total and reversible.

Config level only -- no network, no DEMs, no kernels."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_basal_clutter as rbc  # noqa: E402
from clutter_lines import LINES_DIR, load_all  # noqa: E402

LINE_NAMES = sorted(rbc.LINES)
SPECS = load_all()


@pytest.fixture(autouse=True)
def _restore_line():
    """AUTOUSE: activate_line mutates module state and run() activates without
    restoring, so any test here can leak a line into the next MODULE. It did:
    the line tests left Greenland bound and test_basal_processing then read
    Greenland's REAL_CHAIN."""
    saved = {k: getattr(rbc, k) for k in rbc.LINE_GLOBALS}
    yield
    for k, v in saved.items():
        setattr(rbc, k, v)


@pytest.fixture
def line_sandbox():
    return rbc.activate_line


# ------------------------------------------------------- the registry itself
def test_at_least_one_line_is_defined():
    assert LINE_NAMES, "no line definitions found in config/lines/"
    assert set(LINE_NAMES) == {fp.stem for fp in LINES_DIR.glob("*.yaml")}


def test_default_line_is_derived_from_the_registry():
    """Two hardcoded line names went stale the moment the lines were
    renamed. The default must come from what is actually defined."""
    assert rbc.DEFAULT_LINE in rbc.LINES


@pytest.mark.parametrize("name", LINE_NAMES)
def test_every_line_supplies_every_global(name):
    """Activation is TOTAL: a line that omitted a name would leave the
    previous line's value bound."""
    assert set(rbc.LINES[name]) == set(rbc.LINE_GLOBALS)


@pytest.mark.parametrize("name", LINE_NAMES)
def test_activation_round_trips(name, line_sandbox):
    """Switching to a line and back restores every global, with no help from
    the sandbox fixture -- otherwise any process driving two lines (a batch
    runner, a notebook) is quietly wrong."""
    before = {k: getattr(rbc, k) for k in rbc.LINE_GLOBALS}
    for other in LINE_NAMES:
        line_sandbox(other)
    line_sandbox(name)
    line_sandbox(name)                                    # idempotent
    after = {k: getattr(rbc, k) for k in rbc.LINE_GLOBALS}
    assert after == rbc.LINES[name]
    if name == before["LINE"]:
        assert after == before


def test_unknown_line_rejected(line_sandbox):
    with pytest.raises(ValueError, match="unknown line"):
        line_sandbox("no_such_line")


# ------------------------------------------------------ internal coherence
@pytest.mark.parametrize("name", LINE_NAMES)
def test_pass_and_segment_tables_agree(name):
    spec = SPECS[name]
    segs = set(spec.segments)
    assert set(spec.order) <= set(spec.passes)
    assert spec.reference.pass_key in spec.passes
    ref = spec.passes[spec.reference.pass_key]
    for key, ps in spec.passes.items():
        # a pass may omit windows it does not reach, but never invent one
        assert set(ps.segments) <= segs, f"{key} invents a segment"
    for name in segs:
        covering = [k for k, ps in spec.passes.items() if name in ps.segments]
        assert len(covering) >= 2, f"{name} covered by {covering}"
        assert name in ref.segments, f"{name} missing from the reference pass"
    for key, syn in spec.synthetic_passes.items():
        assert syn.carrier in spec.passes, f"{key} carrier undefined"
    for sname, seg in spec.segments.items():
        if seg.k_anchor:
            assert seg.k_anchor in segs


@pytest.mark.parametrize("name", LINE_NAMES)
def test_frame_ids_are_strings_not_yaml_integers(name):
    """YAML 1.1 reads '_' as a digit separator, so an unquoted 20161105_05_006
    silently becomes an integer."""
    spec = SPECS[name]
    for f in spec.reference.frames:
        assert isinstance(f, str), f
    for ps in spec.passes.values():
        assert isinstance(ps.param_frame, str)
        for parts in ps.segments.values():
            for p in parts:
                assert isinstance(p.frame, str), p


@pytest.mark.parametrize("name", LINE_NAMES)
def test_synthetics_inherit_their_carrier(name, line_sandbox):
    """A synthetic pass re-flies a real pass's line: same frames, same system
    params, same season -- only the altitude changes."""
    line_sandbox(name)
    spec = SPECS[name]
    for key, syn in spec.synthetic_passes.items():
        p, carrier = rbc.PASSES[key], rbc.PASSES[syn.carrier]
        assert p["synthetic_msl_m"] == syn.altitude_m
        assert p["agl_med_m"] is None            # no measured AGL exists
        assert p["param_frame"] == carrier["param_frame"]
        assert p.get("season") == carrier.get("season")
        assert p["instrument"] == carrier["instrument"]
        for s in spec.segments:
            assert p[s] == carrier[s]


@pytest.mark.parametrize("name", LINE_NAMES)
def test_profile_plot_window_stays_inside_the_data_window(name):
    """PROFILE_REL_US is a DATA extent: plotting beyond it shows nothing,
    because the values were never computed."""
    lg = rbc.LINES[name]
    assert lg["PROFILE_X_US"][1] <= lg["PROFILE_REL_US"][1]
    assert lg["PROFILE_X_US"][0] >= lg["PROFILE_REL_US"][0]


@pytest.mark.parametrize("name", LINE_NAMES)
def test_derived_globals_are_consistent(name):
    lg = rbc.LINES[name]
    assert set(lg["SEGMENTS"]) == set(lg["S0_KM"])
    assert set(lg["SEGMENTS"]) == set(lg["N_TRACES_BY_SEGMENT"])
    assert set(lg["SEGMENTS"]) == set(lg["DECOMP_S_KM"])
    assert lg["OUT_DEFAULT"].name == lg["CASE_PREFIX"]
    assert lg["RSSNR_CACHE"].parent == lg["OUT_DEFAULT"]
    ref = lg["REF_PASS"]
    assert lg["REF_SEASON"] == rbc.LINES[name]["PASSES"][ref].get(
        "season", lg["SEASON"])
    lam = 299792458.0 / (lg["FC_HZ"] * float(np.sqrt(3.17)))
    assert lg["LAM_ICE_M"] == pytest.approx(lam, rel=1e-12)


# ------------------------------------------------- call-time global lookups
def test_rel_mean_profile_extent_follows_the_active_line(line_sandbox):
    """Default arguments must resolve at CALL time, not bind at import, or
    activating a line silently keeps the previous line's extent."""
    dt, n = 33.3333e-9, 4000
    twtt = np.arange(n) * dt
    P, t_ref, norm = np.ones((3, n)), np.full(3, 400 * dt), np.ones(3)
    seen = {}
    for name in LINE_NAMES:
        line_sandbox(name)
        rel, _ = rbc.rel_mean_profile(P, twtt, dt, t_ref, norm)
        seen[name] = rel[-1]
        assert rel[-1] == pytest.approx(rbc.PROFILE_REL_US[1], abs=0.05)
    assert len(seen) == len(LINE_NAMES)


def test_panel_scale_follows_the_active_line(line_sandbox):
    rng = np.random.default_rng(0)
    img = rng.normal(-90.0, 6.0, size=(200, 300))
    for name in LINE_NAMES:
        line_sandbox(name)
        lo, hi, note = rbc._panel_scale(img, -120.0, 5.0)
        if rbc.RADARGRAM_SCALE == "shared":
            assert (lo, hi) == (-120.0, 5.0) and note == ""
        else:
            assert -120.0 < lo < hi < 5.0 and "dB" in note


def test_bed_roughness_guard_uses_the_active_carrier(line_sandbox):
    """bed_rough_nadir_db defaulted f0 to a module constant bound at import,
    so it computed the guard at the WRONG carrier after a line switch."""
    vals = {}
    for name in LINE_NAMES:
        line_sandbox(name)
        vals[rbc.FC_HZ] = rbc.bed_rough_nadir_db(0.1)
    for fc, v in vals.items():
        assert v == pytest.approx(
            -(0.1 * 2 * (2 * np.pi * fc * np.sqrt(3.17) / 299792458.0)) ** 2
            * 10.0 / np.log(10.0), rel=1e-9)


# ----------------------------------------------------- cross-line isolation
def test_real_chain_is_not_shared_between_lines():
    """It used to be ONE constant named for 2016, so every line's runs
    recorded 2016 DC-8 provenance."""
    chains = [rbc.LINES[n]["REAL_CHAIN"] for n in LINE_NAMES]
    for c in chains:
        assert {"product", "sar", "combine", "window"} <= set(c)
    if len(chains) > 1:
        assert any(a != b for a, b in zip(chains, chains[1:]))


def test_line_globals_covers_every_registry_key():
    for name, entry in rbc.LINES.items():
        assert set(entry) <= set(rbc.LINE_GLOBALS), name


def test_level_anchor_deficit_is_not_a_line_property():
    """D is solved against a particular run at a particular attenuation, so
    it belongs to that run pair -- a line default is how the Antarctic 14.8
    stayed wired in after the adopted config moved on."""
    assert "LEVEL_ANCHOR_DEFICIT_DB" not in rbc.LINE_GLOBALS


# ------------------------------------------------------------ run() guards
def test_segment_not_on_this_line_rejected():
    line = rbc.DEFAULT_LINE
    with pytest.raises(ValueError, match="has no .* segment"):
        rbc.run(line=line, segment="no_such_segment")


def test_pass_not_on_this_line_rejected():
    # a GL-crossing segment fails earlier (hybrid guard), so pick any line
    # with a non-crossing segment (the default line may have none:
    # antarctica_david's single window crosses)
    line, seg = next(
        (n, s) for n in sorted(rbc.LINES)
        for s in rbc.LINES[n]["SEGMENTS"]
        if s not in rbc.LINES[n]["SEGMENTS_CROSSING_GL"])
    with pytest.raises(ValueError, match="defines no pass"):
        rbc.run(line=line, segment=seg, passes=["no_such_pass"])


@pytest.mark.parametrize(
    "name", [n for n in LINE_NAMES if rbc.LINES[n]["UNSUPPORTED"]])
def test_unsupported_features_rejected(name):
    for feat in rbc.LINES[name]["UNSUPPORTED"]:
        if feat != "demogorgn_bed":
            continue
        with pytest.raises(ValueError, match="not wired"):
            rbc.run(line=name, segment=rbc.LINES[name]["SEGMENTS"][0],
                    demogorgn_bed=True)
