"""Report a line's calibration: gamma_surface and A, with the fit evidence.

    uv run python tools/calibrate_line.py <line> [<line> ...]

Resolves each line exactly as a run would (resolve_calibration): the manual
gamma_surface with its provenance, A manual or solved by the RSSNR-vs-2H
Theil-Sen regression, and the regression diagnostics computed on every line
regardless -- so a pinned A always sits next to what the data would have
said. Dataset-only: no simulation.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_basal_clutter as rbc  # noqa: E402


def main():
    lines = sys.argv[1:] or sorted(rbc.LINES)
    out = {}
    for line in lines:
        gamma, att, rec = rbc.resolve_calibration(line)
        out[line] = rec
        f = rec.get("regression") or {}
        print(f"\n=== {line}")
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
