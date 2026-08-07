"""Level-anchored family comparison table (session artifact).

Per member: per-pass bed-window residual vs measured, bed-return tail slope
vs measured, excess at bed+2 us, guard, plus the implied-reflectivity
diagnostics (median |Gamma|^2, fraction > 0 dB, p95) and the 30 km margin.

    uv run python claude_notes/klevel_family.py [--md]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "outputs" / "basal_clutter" / "hypothesis_tests"
MEMBERS = [("att20", "A20 median (reference)"),
           ("att20_klevel", "A20 level"),
           ("att26_klevel", "A26 level"),
           ("att31_klevel", "A31 level")]
PASSES = ["low", "mid", "high"]
SLUG = "demogorgn"


def load(name):
    p = HYP / name / "metrics.json"
    return json.loads(p.read_text())["metrics"] if p.exists() else None


def rows(m):
    out = []
    for k in PASSES:
        c, t = m[f"clutter_{k}"], m[f"bed_return_tail_{k}"]
        s = t["sim"].get(SLUG) or next(iter(t["sim"].values()))
        out.append({
            "pass": k,
            "res": c["sim"]["bed_rel_surf_db"]
            - c["measured"]["bed_rel_surf_db"],
            "slope": s["bed_returns_slope_db_per_us"],
            "meas": t["measured"]["slope_db_per_us"],
            "exc": t["bed_return_tail_excess_db"][SLUG]["+2us"],
            "guard": s["guard"]["pass"],
            "guard_db": s["guard"]["min_bed_minus_surface_returns_db"]})
    return out


def gamma_diag(m):
    g = m["rssnr_gamma_mapping"]
    la = m.get("rssnr_level_anchor")
    return {"k": g["k_db"], "med": g["g2_seg_db"]["med"],
            "p95": g["g2_seg_db"]["p95"], "pos": g["g2_pos_frac_seg"],
            "d": (la or {}).get("deficit_db"),
            "k_med": (la or {}).get("k_median_db", g["k_db"])}


def main(md):
    print("| member | D dB | K dB | med G2 | p95 G2 | G2>0 frac | pass | "
          "bed-win residual | slope sim/meas | excess +2 us | guard |"
          if md else "")
    if md:
        print("|---|---|---|---|---|---|---|---|---|---|---|")
    syn = []
    for name, label in MEMBERS:
        m = load(name)
        if m is None:
            print(f"| {label} | (missing) |" if md else f"{label}: missing")
            continue
        g = gamma_diag(m)
        rr = rows(m)
        ms = float(np.mean([abs(r["slope"] - r["meas"]) for r in rr]))
        mres = float(np.median([r["res"] for r in rr]))
        for i, r in enumerate(rr):
            head = (f"| **{label}** | {g['d'] if g['d'] is not None else '--'} "
                    f"| {g['k']:+.2f} | {g['med']:+.1f} | {g['p95']:+.1f} | "
                    f"{g['pos']:.3f} " if i == 0 else "| | | | | | ")
            line = (f"| {r['pass']} | {r['res']:+.2f} | "
                    f"{r['slope']:+.2f} / {r['meas']:+.2f} | {r['exc']:+.1f} "
                    f"| {'ok' if r['guard'] else 'FAIL'} {r['guard_db']:+.0f} |")
            print(head + line if md
                  else f"{label:22s} {r['pass']:5s} res {r['res']:+6.2f} "
                       f"slope {r['slope']:+6.2f}/{r['meas']:+6.2f} "
                       f"exc {r['exc']:+6.1f} "
                       f"guard {'ok' if r['guard'] else 'FAIL'}"
                       f"{r['guard_db']:+5.0f}")
        tail = (f"| | | | | | | **summary** | median {mres:+.2f} | "
                f"mean abs {ms:.2f} | | |")
        print(tail if md
              else f"{label:22s} SUMMARY median residual {mres:+.2f} dB, "
                   f"mean |dSlope| {ms:.2f} dB/us")
        v = m.get("syn30km_bed_visibility")
        if v:
            syn.append((label, v["bed_over_surface_clutter_in_bed_window_db"],
                        v["bedpeak_over_midcol_db"]))
    print()
    if md:
        print("| member | 30 km bed - surface returns | bed peak over "
              "mid-column |")
        print("|---|---|---|")
    for label, a, b in syn:
        print(f"| {label} | **{a:+.2f}** | {b:+.2f} |" if md
              else f"{label:22s} 30 km bed-surface {a:+6.2f} dB, "
                   f"bedpeak-midcol {b:+6.2f}")


if __name__ == "__main__":
    main("--md" in sys.argv)
