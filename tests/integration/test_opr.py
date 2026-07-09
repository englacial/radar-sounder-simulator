"""Integration: real OPR frames vs soundersim + simc (M7 / stage 1.1).

Runs tools/run_opr_comparison.py cases from the outputs/cache/ frame + DEM
caches (network is used only to populate a missing cache; if the cache is
absent and the network fails, the test skips with the reason).

Gates are the loose real-frame sanity checks: the simulated surface leading
edge must sit within a few range bins of the frame's Surface pick after
removing the measured constant twtt offset. The soundersim-vs-simc metrics are
recorded in metrics.json but not gated (per plan, real-frame thresholds are
set only after observing residuals).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "run_opr_comparison", ROOT / "tools" / "run_opr_comparison.py")
roc = importlib.util.module_from_spec(_spec)
sys.modules["run_opr_comparison"] = roc
_spec.loader.exec_module(roc)

# run_opr_coherent_bed imports from run_opr_comparison (registered above).
_spec_cb = importlib.util.spec_from_file_location(
    "run_opr_coherent_bed", ROOT / "tools" / "run_opr_coherent_bed.py")
rocb = importlib.util.module_from_spec(_spec_cb)
sys.modules["run_opr_coherent_bed"] = rocb
_spec_cb.loader.exec_module(rocb)

from soundersim.opr import CACHE_DIR  # noqa: E402


def _cached(case):
    return (CACHE_DIR /
            f"frame_{case['season']}_{case['frame_id']}_CSARP_standard.nc").exists()


@pytest.mark.integration
@pytest.mark.parametrize("case", roc.CASES, ids=lambda c: c["frame_id"])
def test_opr_frame(case):
    try:
        metrics, out = roc.run_case(case)
    except Exception as e:
        if not _cached(case):
            pytest.skip(f"no local cache for {case['frame_id']} and remote "
                        f"access failed: {type(e).__name__}: {e}")
        raise

    # Sanity gate: surface leading edge vs Surface pick, constant offset removed.
    sa = metrics["surface_alignment"]
    assert sa["pass"], (
        f"surface leading edge misaligned: p90 {sa['value']:.2f} bins > "
        f"{sa['threshold']} (offset {sa['offset_bins']:+.2f} bins)")
    assert abs(sa["offset_bins"]) < 20, "implausible constant twtt offset"

    # simc comparison values are recorded (not gated) and finite.
    for k in ("peak_alignment", "first_return_bin", "profile_pearson",
              "power_ratio_cv", "db_residual_rms"):
        assert k in metrics and metrics[k]["value"] == metrics[k]["value"]

    # Artifacts exist for the report builder.
    written = json.loads((out / "metrics.json").read_text())
    assert written["case"] == f"opr_{case['frame_id']}"
    for fig in ("radargram_vs_cluttergram.png", "soundersim_vs_simc.png",
                "map_context.png"):
        assert (out / fig).exists()


@pytest.mark.integration
@pytest.mark.parametrize("case", rocb.CASES, ids=lambda c: c["frame_id"])
def test_opr_frame_coherent_bed(case, tmp_path):
    """Coherent surface+bed xOPR case (M18 structure, M24 processing level):
    coherent multilayer kernel with the MCoRDS-like chirp + 7-element array
    (alias-free dt/4 grid, decimated back onto the frame axis) on a subdivided
    facet grid with a BedMachine bed under the PGC surface DEM. Gated against
    BOTH the frame's Surface pick (coherent surface-layer leading edge) and its
    Bottom pick (bed-layer nadir timing, floor-aware). Reduced traces/reach, a
    small dense sub-segment, and a tmp out_root keep the test fast and leave
    the report artifacts from tools/run_opr_coherent_bed.py untouched."""
    try:
        metrics, out = rocb.run_case(
            case, n_traces=25, ct_dist=1200.0, out_root=tmp_path, spacing=64.0,
            dense_traces=60, dense_spacing=1.5, dense_ct=800.0)
    except Exception as e:
        if not _cached(case):
            pytest.skip(f"no local cache for {case['frame_id']} and remote "
                        f"access failed: {type(e).__name__}: {e}")
        raise

    # Surface gate: smoothed-coherent surface-layer leading edge tracks the
    # frame's Surface pick after constant-offset removal.
    le = metrics["surface_leading_edge"]
    assert le["pass"], (f"coherent surface leading edge vs Surface pick "
                        f"misaligned: median {le['value']:.1f} bins > "
                        f"{le['threshold']}")

    # Bed gate: median |sim bed nadir - Bottom pick| after constant-offset
    # removal, thresholded at max(5, input floor + 5) bins -- the INPUT bed
    # model's own disagreement with the picks is a residual no simulator can
    # beat, so the gate checks the machinery's contribution beyond that floor.
    ba = metrics["bed_alignment"]
    floor = metrics["input_bed_error_floor_bins"]["value"]
    assert ba["pass"], (
        f"bed timing off beyond the input-bed floor: median {ba['value']:.1f} "
        f"bins vs threshold {ba['threshold']:.1f} (input floor {floor:.1f})")
    assert ba["pick_coverage"] > 0.2, "too few Bottom picks to gate against"
    assert abs(ba["offset_bins"]) < 40, "implausible constant twtt offset"

    # Recorded diagnostics are present and finite.
    for k in ("input_bed_error_floor_bins", "speckle_contrast",
              "lpa_nadir_error", "bed_surface_power_ratio_db",
              "dropped_power_fraction", "alias_free_dt",
              "clutter_to_surface_db", "speckle_contrast_multilooked",
              "unfocused_surface_gain_db"):
        assert k in metrics and metrics[k]["value"] == metrics[k]["value"]

    # M24: the chirped run must be alias-free (simulate() warning silent) and
    # the dense unfocused processing must not trip the Doppler guard.
    assert metrics["alias_free_dt"]["alias_warning_fired"] is False
    assert metrics["unfocused_surface_gain_db"]["doppler_guard_warned"] is False

    # Artifacts exist for the report builder.
    written = json.loads((out / "metrics.json").read_text())
    assert written["case"] == f"opr_{case['frame_id']}_coherent_bed"
    assert written["group"] == "xOPR clutter"
    for fig in ("radargram_vs_coherent_bed.png", "per_layer_split.png",
                "speckle.png", "unfocused_dense.png"):
        assert (out / fig).exists()
