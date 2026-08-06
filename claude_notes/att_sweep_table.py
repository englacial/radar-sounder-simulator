"""Attenuation-sweep summary table (session artifact).

Reads the four identically-configured value-dirs and emits the markdown
table for the findings note.

    uv run python claude_notes/att_sweep_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "outputs" / "basal_clutter" / "hypothesis_tests"
VALUES = [("baseline", 15), ("att20", 20), ("att26", 26), ("t2_att31", 31)]
PASSES = ["low", "mid", "high"]
SLUG = "demogorgn"


def load(name):
    p = HYP / name / "metrics.json"
    return json.loads(p.read_text())["metrics"] if p.exists() else None


def main():
    print("| att dB/km | K dB | K-K_phys | G2>0 frac | pass | bed window "
          "sim / meas | tail slope sim / meas | excess +2 us | guard |")
    print("|---|---|---|---|---|---|---|---|---|")
    syn = []
    for name, att in VALUES:
        m = load(name)
        if m is None:
            print(f"| {att} | (missing {name}) |")
            continue
        g = m["rssnr_gamma_mapping"]
        first = True
        for k in PASSES:
            t, c = m[f"bed_return_tail_{k}"], m[f"clutter_{k}"]
            s = t["sim"].get(SLUG) or next(iter(t["sim"].values()))
            exc = (t["bed_return_tail_excess_db"] or {}).get(SLUG, {}).get(
                "+2us")
            head = (f"| **{att}** | {g['k_db']:+.2f} | "
                    f"{g['k_minus_kphys_db']:+.2f} | "
                    f"{g['g2_pos_frac_seg']:.3f} " if first else "| | | | ")
            first = False
            print(head
                  + f"| {k} | {c['sim']['bed_rel_surf_db']:+.1f} / "
                  f"{c['measured']['bed_rel_surf_db']:+.1f} "
                  f"| {s['bed_returns_slope_db_per_us']:+.2f} / "
                  f"{t['measured']['slope_db_per_us']:+.2f} "
                  f"| {exc:+.1f} | "
                  f"{'ok' if s['guard']['pass'] else 'FAIL'} "
                  f"{s['guard']['min_bed_minus_surface_returns_db']:+.0f} |")
        v = m.get("syn30km_bed_visibility")
        if v:
            syn.append((att, v["bed_over_surface_clutter_in_bed_window_db"],
                        v["bedpeak_over_midcol_db"], v["bed_rel_surf_db"],
                        v["midcol_rel_surf_db"]))
    print("\n| att dB/km | 30 km bed - surface returns (bed window) | "
          "bed peak over mid-column | bed window | mid-column |")
    print("|---|---|---|---|---|")
    for a, b, p_, bw, mc in syn:
        print(f"| {a} | **{b:+.2f}** | {p_:+.2f} | {bw:+.1f} | {mc:+.1f} |")


if __name__ == "__main__":
    main()
