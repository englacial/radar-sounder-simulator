"""Solve the level-anchor deficit D from a constant-gamma run's metrics.

D is not a free parameter: it is the dB gap between the bed level a
constant-gamma simulation produces and the one the radar measured, solved
under the study-wide rule in config/analysis.yaml. It used to be copied by
hand into an experiment spec with a note about where it came from, which is
how a stale value survives a configuration change.

    uv run python tools/derive_level_deficit.py <run_dir> [--json]

<run_dir> is a completed CONSTANT-gamma run (a directory with metrics.json).
Prints D, the per-pass working, and which passes were excluded for having a
surface-dominated bed window.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_basal_clutter as rbc  # noqa: E402


def levels_from_metrics(doc):
    """{pass: {measured, sim_bed, sim_surface}} from a run's metrics.json."""
    out = {}
    for key, entry in doc.get("metrics", {}).items():
        if not key.startswith("clutter_"):
            continue
        dec = entry.get("decomposition_db") or {}
        if "bed" not in dec or "surface" not in dec:
            continue
        out[key[len("clutter_"):]] = {
            "measured": (entry.get("measured") or {}).get("bed_rel_surf_db"),
            "sim_bed": dec["bed"]["bed_rel_surf_db"],
            "sim_surface": dec["surface"]["bed_rel_surf_db"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="a completed constant-gamma run")
    ap.add_argument("--json", action="store_true", help="emit the record")
    args = ap.parse_args()
    mp = Path(args.run_dir) / "metrics.json"
    if not mp.exists():
        raise SystemExit(f"no metrics.json in {args.run_dir}")
    doc = json.loads(mp.read_text())
    cfg = Path(args.run_dir) / "run_config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        if c.get("gamma_rssnr"):
            print("WARNING: this run has gamma_rssnr on. D must be solved "
                  "against a CONSTANT-gamma run, or it double-counts the "
                  "mapping it is meant to calibrate.", file=sys.stderr)
        # a line may have been RENAMED since the run; the rule is study-wide
        # unless that line overrides it, so fall back rather than refuse
        if c.get("line"):
            try:
                rbc.activate_line(c["line"])
            except ValueError:
                print(f"note: run names line {c['line']!r}, which no longer "
                      "exists; using the study-wide rule", file=sys.stderr)
    d, rec = rbc.solve_level_deficit(levels_from_metrics(doc))
    if args.json:
        print(json.dumps(rec, indent=1))
        return
    print(f"rule: {rec['rule']}")
    for k, v in rec["per_pass"].items():
        mark = "used" if v["qualifies"] else "EXCLUDED (surface-dominated)"
        print(f"  {k:12s} bed-over-surface {v['bed_over_surface_db']:+7.2f} dB"
              f"   D {v['d_db']}   {mark}")
    print(f"\nD = {d:+.2f} dB  (from {rec['n_qualifying']} qualifying pass"
          f"{'es' if rec['n_qualifying'] != 1 else ''})")


if __name__ == "__main__":
    main()
