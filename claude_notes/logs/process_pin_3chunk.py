"""Run the normal pilot runner on antarctica_pineisland_north with the
pre-fc47ed6 chunk budget (1.1e9 -> 3 chunks/pass) so the 18 cloud chunks in
outputs/ hit [skip-exists] and only processing/analysis/figures run
locally. Verification helper, not part of the tool."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import run_basal_clutter as rbc  # noqa: E402

rbc.CHUNK_TRACE_FACETS = 1.1e9
sys.argv = ["run_basal_clutter", "--config", "config/experiments/pilot.yaml",
            "--line", "antarctica_pineisland_north"]
rbc.main_config()
