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

# run_opr_coherent imports from run_opr_comparison (registered above).
_spec_c = importlib.util.spec_from_file_location(
    "run_opr_coherent", ROOT / "tools" / "run_opr_coherent.py")
rocoh = importlib.util.module_from_spec(_spec_c)
sys.modules["run_opr_coherent"] = rocoh
_spec_c.loader.exec_module(rocoh)

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
@pytest.mark.parametrize("case", rocoh.CASES, ids=lambda c: c["frame_id"])
def test_opr_frame_coherent(case, tmp_path):
    """Coherent xOPR case (M13, simplified): coherent kernel on a subdivided
    facet grid, gated against the frame's Surface pick. Reduced traces/reach and
    a tmp out_root so the test stays fast and never clobbers the report
    artifacts from tools/run_opr_coherent.py."""
    try:
        metrics, out = rocoh.run_coherent_case(
            case, n_traces=30, ct_dist=1500.0, out_root=tmp_path, spacing=64.0)
    except Exception as e:
        if not _cached(case):
            pytest.skip(f"no local cache for {case['frame_id']} and remote "
                        f"access failed: {type(e).__name__}: {e}")
        raise

    # Sanity gate: smoothed-coherent surface leading edge tracks the frame's
    # Surface pick after constant-offset removal.
    le = metrics["surface_leading_edge"]
    assert le["pass"], (f"coherent leading edge vs Surface pick misaligned: "
                        f"p90 {le['value']:.1f} bins > {le['threshold']}")

    # The recorded speckle/LPA diagnostics are finite.
    for k in ("speckle_contrast", "lpa_nadir_error"):
        assert k in metrics and metrics[k]["value"] == metrics[k]["value"]

    written = json.loads((out / "metrics.json").read_text())
    assert written["case"] == f"opr_{case['frame_id']}_coherent"
    assert written["group"] == "xOPR clutter"
    for fig in ("radargram_vs_coherent.png", "speckle.png"):
        assert (out / fig).exists()
