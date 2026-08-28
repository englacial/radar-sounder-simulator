"""Along-track projection (tools/line_geometry.py) and the survey tool.

The projection is what turns "these flights overlap" into numbers, so it is
worth pinning against cases with a known answer. No network: the geometry is
synthetic and the line files are read as config."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import line_geometry as lg  # noqa: E402
from clutter_lines import load_all  # noqa: E402


def _straight(n=200, step=10.0):
    xy = np.column_stack([np.arange(n) * step, np.zeros(n)])
    return xy, lg.arc_length(xy)


def test_arc_length_is_cumulative_distance():
    xy, s = _straight()
    assert s[0] == 0.0
    assert s[-1] == pytest.approx(1990.0)
    assert np.allclose(np.diff(s), 10.0)


def test_projection_recovers_along_track_and_offset():
    xy, s = _straight()
    q = np.array([[105.0, 25.0], [500.0, -10.0], [1990.0, 0.0]])
    ss, lat = lg.project(q, xy, s)
    assert np.allclose(ss, [105.0, 500.0, 1990.0], atol=1e-6)
    assert np.allclose(lat, [25.0, -10.0, 0.0], atol=1e-6)


def test_offset_sign_is_consistent():
    """Left of track and right of track must not both come back positive, or
    a track that weaves would average to a falsely small offset."""
    xy, s = _straight()
    _, lat = lg.project(np.array([[500.0, 30.0], [500.0, -30.0]]), xy, s)
    assert lat[0] == pytest.approx(-lat[1])
    assert lat[0] != 0.0


def test_a_pass_that_never_enters_the_span_is_reported_not_guessed():
    xy, s = _straight()
    q = np.column_stack([np.arange(5) * 10.0 + 5000.0, np.zeros(5)])
    ss, _ = lg.project(q, xy, s)
    sl, frac = lg.slice_to_span(ss, 0.0, 1990.0)
    assert sl is None and frac == 0.0


def test_slice_to_span_is_contiguous_and_reports_coverage():
    """A track that leaves and re-enters is returned WHOLE with a coverage
    fraction below 1, rather than silently split into pieces."""
    s = np.concatenate([np.linspace(0, 100, 50),
                        np.linspace(400, 500, 50),      # excursion
                        np.linspace(100, 200, 50)])
    sl, frac = lg.slice_to_span(s, 0.0, 200.0)
    assert sl == (0, 150)
    assert frac == pytest.approx(100 / 150, abs=0.01)


# --------------------------------------------------------- the new lines
LINES = load_all()


def test_geikie_transit_is_one_path_through_all_three_frames():
    """The extension is a single continuous path, not a stitched pair of
    clean windows. It contains a known misalignment -- the two aircraft flew
    the s 40-60 km TURN on different radii, up to 1.3 km apart -- which is
    accepted so that the line is one path."""
    ln = LINES["greenland_geikie01_transit"]
    assert set(ln.segments) == {"pilot", "full"}
    assert "20140421_01_071" in ln.reference.frames      # axis must reach it
    tr = {k: [p.frame for p in ps.segments["full"]]
          for k, ps in ln.passes.items()}
    assert tr["p3_2014_low"] == ["20140421_01_069", "20140421_01_070",
                         "20140421_01_071"]
    assert tr["p3_2017_high"] == ["20170424_01_067", "20170424_01_068",
                          "20170424_01_069"]
    # whole frames (slice omitted) resolved from recorded lengths
    # (verified against the OPR products 2026-08-21)
    flen = {"20140421_01_070": 3332, "20140421_01_071": 3332,
            "20170424_01_067": 3335, "20170424_01_068": 3335}
    n = {k: sum((p.slice[1] - p.slice[0]) if p.slice is not None
                else flen[p.frame] for p in ps.segments["full"])
         for k, ps in ln.passes.items()}
    assert abs(n["p3_2014_low"] - n["p3_2017_high"]) / max(n.values()) < 0.01
    # the decomposition locations must sit in the WELL-ALIGNED stretches,
    # not in the turn
    for v in ln.segments["full"].decomp_s_km:
        assert not (40.0 < v < 80.0), v


def test_westcoast_is_an_instrument_line_not_an_altitude_line():
    """Every pass flies within ~75 m of 460 m AGL; what varies is the radar.
    The line carries BOTH comparison axes: an instrument swap (200/100 vs
    195/30 MHz) and a same-instrument repeat (the 195/30 flown in 2017 and
    2019), whose spread is the product-to-product noise floor any
    cross-instrument difference must clear. No C-130 pass may return: the
    2015 C-130 season's products are radiometrically miscalibrated
    (img_comb combines the waveform images incorrectly)."""
    ln = LINES["greenland_westcoast"]
    agl = [p.agl_med_m for p in ln.passes.values()]
    assert max(agl) - min(agl) < 150.0
    assert len(ln.passes) == 3
    assert not any("c130" in k for k in ln.passes)
    # the same-instrument repeat: two passes read from OPR frames whose
    # simulated params are identical 195/30 (the 2017 and 2019 P-3s)
    fc = {}
    for k, ps in ln.passes.items():
        import json
        cache = ROOT / "outputs" / "cache" /             f"mcords_params_{ps.season}_{ps.param_frame}.json"
        if not cache.exists():
            pytest.skip("param cache not primed")
        wf = json.loads(cache.read_text())["waveform"]
        fc[k] = (wf["center_frequency_Hz"], wf["bandwidth_Hz"])
    assert fc["p3_2017"] == fc["p3_2019"] == (195e6, 30e6)   # the repeat
    assert fc["p3_2016"] == (200e6, 100e6)                   # the swap


def test_a_window_needs_at_least_two_passes_and_the_reference():
    """A window with one pass has nothing to compare; a window the reference
    pass misses has no axis to project the others onto."""
    from clutter_lines import LineSpec
    d = LINES["greenland_westcoast"].model_dump(by_alias=True)
    for k, ps in d["passes"].items():
        if k != d["reference"]["pass"]:
            ps["segments"] = {}
    with pytest.raises(ValueError, match="at least two passes"):
        LineSpec.model_validate(d)


def test_westcoast_has_rssnr_coverage():
    """Initially declared unsupported on an unchecked assertion; verified
    2026-08-19 against the pinned snapshot -- both reference frames are
    present (32/42 and 42/42 usable traces)."""
    ln = LINES["greenland_westcoast"]
    assert ln.rssnr is not None
    assert ln.rssnr.snapshot == "GEAMAHQ7BRVPG9SQPK20"
    assert "gamma_rssnr" not in ln.unsupported


def test_a_line_without_an_rssnr_store_must_say_so():
    """Otherwise a run could ask for a reflectivity mapping there is no data
    for, and find out only after the scene prep."""
    from clutter_lines import LineSpec
    d = LINES["greenland_westcoast"].model_dump(by_alias=True)
    d["rssnr"] = None
    with pytest.raises(ValueError, match="gamma_rssnr"):
        LineSpec.model_validate(d)
