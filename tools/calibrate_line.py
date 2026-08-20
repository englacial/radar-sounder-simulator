"""Report a line's calibration: gamma_surface and A, with the fit evidence.

    uv run python tools/calibrate_line.py <line> [<line> ...]

Resolves each line exactly as a run would (resolve_calibration):
gamma_surface manual with its provenance or 'solve' (resolved IN-RUN by the
--config driver -- it needs simulations, so this dataset-only report can
only name the seed and the latest resolved value if a run has recorded
one), A manual or solved by the RSSNR-vs-2H Theil-Sen regression, and the
regression diagnostics computed on every line regardless -- so a pinned A
always sits next to what the data would have said. No simulation here.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_basal_clutter as rbc  # noqa: E402


def _latest_solved(line):
    """Newest run_config.json under this line's outputs with a SOLVED
    gamma_surface in its calibration record, if any."""
    root = ROOT / "outputs" / rbc.LINES[line]["CASE_PREFIX"]
    best = None
    for fp in root.glob("*/run_config.json"):
        try:
            cr = json.loads(fp.read_text()).get("calibration_resolution", {})
        except Exception:
            continue
        if "gamma_surface_solve_history" not in cr:
            continue
        if best is None or fp.stat().st_mtime > best[0]:
            best = (fp.stat().st_mtime, {
                "gamma_surface_db": cr["gamma_surface_db"],
                "surface_anomaly_db": cr["surface_anomaly_db"],
                "run_config": str(fp.relative_to(ROOT))})
    return best[1] if best else None


def main():
    lines = sys.argv[1:] or sorted(rbc.LINES)
    out = {}
    for line in lines:
        gamma, att, rec = rbc.resolve_calibration(line)
        out[line] = rec
        f = rec.get("regression") or {}
        print(f"\n=== {line}")
        if gamma == "solve":
            st = rec["gamma_surface_solve_settings"]
            print(f"  gamma_surface = solve   (resolved in-run: seed "
                  f"{st['seed_db']:+.1f} dB, tolerance "
                  f"{st['tolerance_db']} dB, qualifying headroom >= "
                  f"{st['min_headroom_db']} dB)")
            solved = _latest_solved(line)
            if solved:
                print(f"    latest resolved: {solved['gamma_surface_db']:+.2f}"
                      f" dB (anomaly {solved['surface_anomaly_db']:+.2f} dB)"
                      f" from {solved['run_config']}")
        else:
            print(f"  gamma_surface = {gamma:+.2f} dB   "
                  f"(anomaly vs smooth Fresnel {rec['surface_anomaly_db']:+.2f} "
                  "dB)")
            print(f"    why: {rec['gamma_surface_why'].strip()[:150]}")
        print(f"  A = {att:g} dB/km   [{rec['att_source'].split(':')[0]}]")
        if "error" in f:
            print(f"  regression: FAILED -- {f['error']}")
        else:
            print(f"  regression (diagnostic on every line): "
                  f"A = {f['att_db_per_km']} "
                  f"[{f['slope_ci95_db_per_km'][0]}, "
                  f"{f['slope_ci95_db_per_km'][1]}] dB/km, "
                  f"r = {f['r_rssnr_vs_2h']}, n = {f['n_used']}"
                  + (f", {f['n_floating_excluded']} floating excluded"
                     if f.get("n_floating_excluded") else "")
                  + f", H span {f['thickness_span_m']:.0f} m")
    fp = ROOT / "outputs" / "line_reports" / "calibrations.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
