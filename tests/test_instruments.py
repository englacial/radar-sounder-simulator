"""Radar system definitions (config/instruments/*.yaml) and the line/
instrument split.

The split exists so the two mission-design axes vary independently: swap the
radar at a fixed altitude, or fly the same radar higher. These tests cover
the invariants that make a swap safe -- above all that swapping the
instrument cannot silently reuse the real instrument's cached chunks."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_altitude_comparison as rac  # noqa: E402
import run_basal_clutter as rbc  # noqa: E402
from clutter_instruments import (InstrumentSpec, load_all,  # noqa: E402
                                 segment_of, validate_line_instruments)
from clutter_lines import load_all as load_lines  # noqa: E402


def _inst(**over):
    d = {"schema_version": 1, "name": "demo",
         "source": {"kind": "stated"},
         "simulated": {"frequency_Hz": 60e6, "bandwidth_Hz": 15e6,
                       "pulse_length_s": 20e-6, "window": "hann"}}
    d.update(over)
    return InstrumentSpec.model_validate(d)


# ------------------------------------------------------------- the schema
def test_segment_of_splits_the_frame_id():
    assert segment_of("20161105_05_006") == "20161105_05"
    assert segment_of("20140421_01_069") == "20140421_01"


def test_stated_instrument_must_be_complete():
    """There is no OPR frame to defer to, so a null is unresolvable."""
    with pytest.raises(ValueError, match="leaves"):
        _inst(simulated={"frequency_Hz": 60e6})


def test_real_instrument_must_declare_its_segments():
    with pytest.raises(ValueError, match="no segments"):
        InstrumentSpec.model_validate(
            {"schema_version": 1, "name": "d",
             "source": {"kind": "opr_frame"}})


def test_unquoted_segment_ids_fail_with_an_explanation():
    """YAML 1.1 reads '_' as a digit separator, so an unquoted 20161105_05
    silently becomes the integer 2016110505."""
    with pytest.raises(ValueError, match="QUOTED"):
        InstrumentSpec.model_validate(
            {"schema_version": 1, "name": "d",
             "source": {"kind": "opr_frame"}, "segments": [2016110505]})


# ------------------------------------------------------- resolution rules
def test_real_instrument_defers_to_the_opr_frame():
    """Every simulated parameter comes from the frame the pass was flown on,
    which is what keeps a measured-vs-simulated comparison honest."""
    inst = load_all()["mcords3_dc8_2016"]
    frame = {"center_frequency_Hz": 190e6, "bandwidth_Hz": 50e6,
             "bed_waveform_pulse_length_s": 10e-6,
             "pulse_compression_freq_window": "hanning (whatever)"}
    wf, ant, dev = inst.resolve(frame)
    assert wf == frame                      # verbatim, no substitution
    assert dev == {}                        # nothing deviates
    assert (ant.n_elements, ant.spacing_lam) == (rac.N_ELEMENTS,
                                                 rac.SPACING_LAM)


def test_stated_instrument_ignores_the_frame_and_flags_nothing():
    wf, ant, dev = _inst().resolve(None)
    assert wf["center_frequency_Hz"] == 60e6
    assert wf["bed_waveform_pulse_length_s"] == 20e-6
    assert dev == {}


def test_overriding_a_real_parameter_is_recorded_as_a_deviation():
    """Stating a value on an opr_frame instrument is legal but must never be
    silent: the run config has to show it departed from the flown system."""
    inst = _inst(source={"kind": "opr_frame"}, segments=["20161105_05"],
                 simulated={"frequency_Hz": 150e6, "bandwidth_Hz": None,
                            "pulse_length_s": None, "window": None})
    frame = {"center_frequency_Hz": 190e6, "bandwidth_Hz": 50e6,
             "bed_waveform_pulse_length_s": 10e-6,
             "pulse_compression_freq_window": "hanning"}
    wf, _, dev = inst.resolve(frame)
    assert wf["center_frequency_Hz"] == 150e6
    assert dev["frequency_Hz"] == {"recorded_system": 190e6, "used": 150e6}
    assert wf["bandwidth_Hz"] == 50e6       # the nulls still defer
    assert "deviations_from_recorded_system" in inst.provenance_block(dev)


# ------------------------------------------------------- line cross-check
def test_shipped_lines_pin_instruments_that_cover_their_segments():
    assert validate_line_instruments(load_lines(), load_all())


def test_a_pass_pinned_outside_its_instrument_segment_is_refused():
    lines = load_lines()
    ln = lines[rbc.DEFAULT_LINE]
    next(iter(ln.passes.values())).param_frame = "19990101_99_001"
    with pytest.raises(ValueError, match="does not cover"):
        validate_line_instruments(lines, load_all())


def test_a_synthetic_instrument_cannot_be_a_line_default():
    """Swapping one in is an experiment's job; pinning one as what FLEW the
    line would be a false provenance claim."""
    lines = load_lines()
    synth = next(n for n, i in load_all().items()
                 if i.source.kind == "stated")
    next(iter(lines[rbc.DEFAULT_LINE].passes.values())).instrument = synth
    with pytest.raises(ValueError, match="SYNTHETIC"):
        validate_line_instruments(lines, load_all())


# --------------------------------------------------------- cache identity
def _p(**over):
    p = {"key": "low", "segment": "full", "picked_bed": True,
         "gamma_rssnr": False, "proc": True, "dgn": False, "rev": False,
         "parts": [("20140421_01_069", (736, 2675))], "spacing": 10.6712,
         "reach": {"ct_m": 5625.0}, "window": "hann",
         "instrument": "mcords3_p3_greenland",
         "instrument_default": "mcords3_p3_greenland",
         "rc_sim": rbc.RadarConfig(dt=1e-9, n_samples=64, t0=0.0, f0=195e6),
         "aux": {}}
    p.update(over)
    return p


def test_default_instrument_leaves_the_cache_key_untouched():
    """Introducing the instrument indirection must not move a single key, or
    every existing cached chunk silently re-simulates."""
    p = _p()
    rid = rbc.chunk_rid(p, 0, 14.0, True)
    meta = rbc.chunk_meta(p, 0, np.arange(198), 5, 1939, 14.0, True)
    assert rid == "low_full_pbed_proc_c00_srough_att14"
    assert "instrument" not in meta


def test_swapping_the_instrument_forks_the_cache_key():
    """A different radar is a different simulation. If the key did not move,
    a swap would silently serve the real instrument's chunks."""
    base, swap = _p(), _p(instrument="haps_60mhz")
    assert rbc.chunk_rid(swap, 0, 14.0, True) != rbc.chunk_rid(base, 0, 14.0,
                                                               True)
    assert "haps_60mhz" in rbc.chunk_rid(swap, 0, 14.0, True)
    m = rbc.chunk_meta(swap, 0, np.arange(198), 5, 1939, 14.0, True)
    assert m["instrument"] == "haps_60mhz"


# ------------------------------------------------------------ swap wiring
def test_extra_pass_rides_its_carrier_and_carries_its_own_radar():
    """Both swap axes at once: a carrier pass's geometry re-flown at a new
    altitude with a different radar. Built inline rather than read from a
    shipped spec -- the capability must hold whether or not any committed
    experiment currently exercises it."""
    from clutter_spec import RunSpec
    line = rbc.DEFAULT_LINE
    seg = next(s for s in rbc.LINES[line]["SEGMENTS"] if s != "full_line")
    carrier = rbc.LINES[line]["ORDER"][0]
    synth = next(n for n, i in load_all().items()
                 if i.source.kind == "stated")
    kw = RunSpec.model_validate({
        "schema_version": 1, "meta": {"name": "demo"},
        "run": {"line": line, "segment": seg, "out_name": "demo",
                "passes": [carrier, "swapped"],
                "extra_passes": {"swapped": {
                    "carrier": carrier, "altitude_m": 14000.0,
                    "instrument": synth}},
                "physics": {"att_db_per_km": 20.0}}}).to_run_kwargs()
    ep = kw["extra_passes"]["swapped"]
    assert ep["carrier"] == carrier          # real geometry, real picks
    assert ep["altitude_m"] == 14000.0       # new altitude
    assert ep["instrument"] == synth         # new radar
    assert kw["passes"] == [carrier, "swapped"]
