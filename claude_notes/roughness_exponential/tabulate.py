"""Measured vs fixture vs B1 tables from the pilot metrics.json files.

    uv run python claude_notes/roughness_exponential/tabulate.py [case ...]

Default cases: pilot_smoke (fixture), pilot_smoke_b1, pilot_smoke_exp (acf exponential).
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LINES = {"greenland_westcoast": "greenland_westcoast",
         "greenland_geikie01_transit": "greenland_geikie",
         "antarctica_getz": "antarctica_getz"}
CASES = sys.argv[1:] or ["pilot_smoke", "pilot_smoke_b1", "pilot_smoke_exp"]


def load(line, case):
    p = ROOT / "outputs" / LINES[line] / case / "metrics.json"
    if not p.exists():
        return None, None
    cfg = json.loads((p.parent / "run_config.json").read_text())
    return json.loads(p.read_text())["metrics"], cfg


def main():
    for line, ldir in LINES.items():
        ly = yaml.safe_load((ROOT / "config/lines" / f"{line}.yaml").read_text())
        agl = {k: v.get("agl_med_m") for k, v in ly["passes"].items()}
        runs = {c: load(line, c) for c in CASES}
        runs = {c: v for c, v in runs.items() if v[0]}
        if not runs:
            continue
        print(f"\n## {line}\n")
        cols = list(runs)
        # resolved roughness
        print("| pass | AGL m | f0 MHz | " + " | ".join(f"{c}: sigma cm / l m" for c in cols) + " |")
        print("|---|---|---|" + "---|" * len(cols))
        passes = [k for k in ly["passes"]]
        for k in passes:
            cells = []
            for c in cols:
                sr = (runs[c][1].get("surface_roughness") or {}).get("passes", {}).get(k)
                if sr is None:
                    sr = {"sigma_m": 0.049474, "corr_length_m": 2.982179}
                cells.append(f"{sr['sigma_m'] * 100:.2f} / {sr['corr_length_m']:.3f}" + (" exp" if sr.get("acf") == "exponential" else ""))
            f0 = ly["identity"]["fc_hz"] / 1e6
            print(f"| {k} | {agl[k]} | {f0:.0f} | " + " | ".join(cells) + " |")
        # mid-column clutter
        print("\n| pass | AGL m | measured midcol rel surf dB | " + " | ".join(f"{c} sim (err)" for c in cols) + " |")
        print("|---|---|---|" + "---|" * len(cols))
        for k in passes:
            m0 = runs[cols[0]][0].get(f"clutter_{k}")
            if not m0:
                continue
            meas = m0["measured"]["midcol_rel_surf_db"]
            cells = []
            for c in cols:
                m = runs[c][0].get(f"clutter_{k}")
                if m:
                    s = m["sim"]["midcol_rel_surf_db"]
                    cells.append(f"{s:.1f} ({s - meas:+.1f}) [{m['midcol_verdict']}]")
                else:
                    cells.append("-")
            print(f"| {k} | {agl[k]} | {meas:.1f} | " + " | ".join(cells) + " |")
        # altitude trend
        at = {c: runs[c][0].get("altitude_trend") for c in cols}
        if any(at.values()):
            print("\n| pair | measured delta dB | " + " | ".join(f"{c} sim (err)" for c in cols) + " |")
            print("|---|---|" + "---|" * len(cols))
            for pair, v in next(a for a in at.values() if a)["pairs"].items():
                cells = [(f"{at[c]['pairs'][pair]['sim_db']:.1f} ({at[c]['pairs'][pair]['error_db']:+.1f})"
                          if at[c] and pair in at[c]["pairs"] else "-") for c in cols]
                print(f"| {pair} | {v['measured_db']:.1f} | " + " | ".join(cells) + " |")
        # bed tail + alignment + bed level
        print("\n| pass | measured bed rel surf dB | " + " | ".join(f"{c}: bed rel surf / tail excess +1,+2,+3 us / surf align p90 bins" for c in cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for k in passes:
            m0 = runs[cols[0]][0].get(f"clutter_{k}")
            if not m0:
                continue
            cells = []
            for c in cols:
                mm, bt, sa = (runs[c][0].get(f"clutter_{k}"), runs[c][0].get(f"bed_return_tail_{k}"),
                              runs[c][0].get(f"surface_alignment_{k}"))
                if not mm:
                    cells.append("-"); continue
                ex = bt["bed_return_tail_excess_db"]["picked_bed"] if bt else {}
                cells.append(f"{mm['sim']['bed_rel_surf_db']:.1f} / " + ", ".join(f"{ex.get(u, float('nan')):+.1f}" for u in ("+1us", "+2us", "+3us"))
                             + f" / {sa['p90_bins']:.2f}" if sa else "")
            print(f"| {k} | {m0['measured']['bed_rel_surf_db']:.1f} | " + " | ".join(cells) + " |")
        # wall
        print("\nwall s per pass: " + "; ".join(
            f"{c}: " + ", ".join(f"{k} {v:.0f}" for k, v in runs[c][0]["simulation_wall_s"]["per_pass_s"].items()) for c in cols))


if __name__ == "__main__":
    main()
