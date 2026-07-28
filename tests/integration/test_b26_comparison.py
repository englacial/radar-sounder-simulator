"""Integration: B26 firn-core measured-vs-simulated comparison (light).

Runs tools/run_b26_comparison.py with a tiny configuration (few traces, N=10
firn layers only, coarse 64 m facets) from the outputs/cache/ frame + DEM +
BedMachine caches (the along-track window and cross-track reaches match the
main run so the cached windows are reused; network is touched only to
populate a missing cache -- if that fails the test skips).

Gates: the artifact schema (metrics.json case/group, figures, report) and the
surface-pick alignment sanity gate (median, offset-removed, <= 5 frame bins).
Everything else is recorded, not gated (real-frame convention). The alias
warning must be silent (asserted inside the tool on every simulate() call and
recorded in the metrics).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

_spec = importlib.util.spec_from_file_location(
    "run_b26_comparison", ROOT / "tools" / "run_b26_comparison.py")
rb = importlib.util.module_from_spec(_spec)
sys.modules["run_b26_comparison"] = rb
_spec.loader.exec_module(rb)

from soundersim.opr import CACHE_DIR  # noqa: E402


def _cached():
    return (CACHE_DIR /
            f"frame_{rb.SEASON}_{rb.FRAME_ID}_CSARP_standard.nc").exists()


@pytest.mark.integration
def test_b26_comparison_tiny(tmp_path):
    try:
        metrics, out = rb.run_all(
            out_root=tmp_path, n_traces=8, layer_counts=(10,), spacing=64.0,
            do_pilot=False)
    except Exception as e:
        if not _cached():
            pytest.skip(f"no local cache for {rb.FRAME_ID} and remote access "
                        f"failed: {type(e).__name__}: {e}")
        raise

    # Surface sanity gate: coherent surface-layer leading edge vs the frame's
    # Surface pick, constant offset removed, median <= 5 frame bins.
    sa = metrics["surface_pick_alignment"]
    assert sa["pass"], (f"surface leading edge misaligned: median "
                        f"{sa['value']:.2f} bins > {sa['threshold']}")
    assert abs(sa["offset_bins"]) < 40, "implausible constant twtt offset"

    # Alias rule: warning asserted silent inside the tool, recorded here.
    assert metrics["alias_free_dt"]["alias_warning_fired"] is False

    # Recorded diagnostics present and finite (x == x rejects NaN).
    for k in ("bed_alignment", "lpa_nadir_error", "bed_depth_at_site",
              "firn_seam_check", "closest_approach_m", "profile_correlation"):
        assert k in metrics and metrics[k]["value"] == metrics[k]["value"]

    # The frame passes within tens of meters of the borehole.
    assert metrics["closest_approach_m"]["value"] < 100.0

    # The field-sum seam must be tight where the strip covers all arrivals.
    assert metrics["firn_seam_check"]["value"] < 0.05

    # Artifact schema for the report builder.
    doc = json.loads((out / "metrics.json").read_text())
    assert doc["case"] == "b26_comparison"
    assert doc["group"] == "xOPR clutter"
    assert isinstance(doc["metrics"], dict) and doc["notes"]
    for fig in ("radargrams_full.png", "radargrams_nearsurface.png",
                "depth_profile.png"):
        assert (out / fig).exists()
    assert (out / "report.html").exists()
    assert (out / "run_config.json").exists()
