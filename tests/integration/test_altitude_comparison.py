"""Integration: platform-altitude comparison of a coherent surface+bed frame.

Runs tools/run_altitude_comparison.py on a tiny cache-only config (few traces,
coarse facets clamped via --min-spacing, levels real + one synthetic) and
asserts the artifact schema plus the real-nav surface-alignment gate. Uses the
same cached frame/DEM/BedMachine windows as the other xOPR integration tests;
skips if the frame cache is absent and the network is unavailable.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# run_altitude_comparison imports run_opr_coherent_bed + run_opr_comparison.
_load("run_opr_comparison", "tools/run_opr_comparison.py")
_load("run_opr_coherent_bed", "tools/run_opr_coherent_bed.py")
rac = _load("run_altitude_comparison", "tools/run_altitude_comparison.py")

from soundersim.opr import CACHE_DIR  # noqa: E402

SEASON, FRAME = "2019_Greenland_P3", "20190418_01_009"


def _frame_cached():
    return (CACHE_DIR /
            f"frame_{SEASON}_{FRAME}_CSARP_standard.nc").exists()


@pytest.mark.integration
def test_altitude_comparison(tmp_path):
    try:
        metrics, out = rac.run(
            SEASON, FRAME, levels="real,5000msl", n_traces=8, along_m=10000.0,
            ct_cap=6000.0, out_root=tmp_path, min_spacing=80.0)
    except Exception as e:
        if not _frame_cached():
            pytest.skip(f"no local cache for {FRAME} and remote access "
                        f"failed: {type(e).__name__}: {e}")
        raise

    # Real-nav surface-alignment gate (offset-removed leading edge vs pick).
    sa = metrics["real_surface_alignment"]
    assert sa["pass"], (f"real-nav surface leading edge misaligned: "
                        f"{sa['value']:.2f} bins > {sa['threshold']}")
    assert abs(sa["offset_bins"]) < 40, "implausible constant twtt offset"

    # r^-2 scaling + per-level LPA error are recorded and finite.
    r2 = metrics["surface_r2_scaling"]
    assert r2["value"] == r2["value"] and len(r2["pairs"]) == 1
    for spec in ("real", "5000msl"):
        key = f"lpa_nadir_error_{spec}"
        assert key in metrics and metrics[key]["value"] == metrics[key]["value"]

    # Artifact schema for the report builder.
    written = json.loads((out / "metrics.json").read_text())
    assert written["case"] == f"altitude_{FRAME}"
    assert written["group"] == "xOPR clutter"
    for fig in ("radargrams.png", "nadir_profiles.png"):
        assert (out / fig).exists()
    assert (out / "report.html").exists()

    cfg = json.loads((out / "run_config.json").read_text())
    specs = {row["level"] for row in cfg["level_table"]}
    assert specs == {"real", "5000msl"}
    # Alias-free decimation factor is searched (not hard-coded /4) + recorded;
    # the 2019 P3 grid must still select k=4. The modeled compression window
    # is recorded alongside the product's own window string (hanning -> hann,
    # no approximation on this frame).
    assert cfg["oversample"] == 4
    assert cfg["chirp"]["window_modeled"] == "hann"
    assert "window_product" in cfg["chirp"]
    for row in cfg["level_table"]:
        for field in ("agl_range_m", "facet_spacing_m", "n_facets_per_interface",
                      "n_samples", "wall_s", "surface_peak_db",
                      "pulse_limited_footprint_m", "cross_track_cap_bound"):
            assert field in row
