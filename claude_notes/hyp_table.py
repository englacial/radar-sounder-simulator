"""Master table for the basal-clutter hypothesis campaign (session artifact).

Reads metrics.json from outputs/basal_clutter/hypothesis_tests/<test>/ (plus
the baseline copy) and prints, per test x per pass: bed-return tail slope,
excess at +2 us, nadir bed-window level, guard status -- as deltas vs the
BASELINE and vs MEASURED.

    uv run python claude_notes/hyp_table.py [test ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "outputs" / "basal_clutter" / "hypothesis_tests"
PASSES = ["low", "mid", "high", "syn30km"]
SLUG = "demogorgn"          # every campaign run uses the DEMOGORGN bed


def load(name):
    p = HYP / name / "metrics.json"
    return json.loads(p.read_text())["metrics"] if p.exists() else None


def row(m, key):
    """(slope, level+2us, excess+2us, nadir bed window level, guard)."""
    t = m.get(f"bed_return_tail_{key}")
    c = m.get(f"clutter_{key}") or m.get("syn30km_bed_visibility")
    if t is None:
        return None
    s = t["sim"].get(SLUG) or next(iter(t["sim"].values()))
    exc = (t["bed_return_tail_excess_db"] or {}).get(SLUG)
    if exc is None and t["bed_return_tail_excess_db"]:
        exc = next(iter(t["bed_return_tail_excess_db"].values()))
    bed = ((c or {}).get("sim") or {}).get("bed_rel_surf_db")
    if bed is None:
        bed = (c or {}).get("bed_rel_surf_db")
    return {"slope": s["slope_db_per_us"], "lvl2": s["level_rel_surf_db"]["+2us"],
            "exc2": None if not exc else exc["+2us"],
            "bed": bed, "guard": s["guard"]["pass"],
            "guard_db": s["guard"]["min_bed_minus_surface_returns_db"],
            "meas_slope": (t["measured"] or {}).get("slope_db_per_us"),
            "meas_lvl2": ((t["measured"] or {}).get("level_rel_surf_db") or
                          {}).get("+2us"),
            "meas_bed": ((c or {}).get("measured") or {}).get(
                "bed_rel_surf_db")}


def _d(a, c):
    return None if a is None or c is None else a - c


def fmt(v, s="{:+.2f}"):
    return "  --  " if v is None else s.format(v)


def main(tests):
    base = load("baseline")
    print(f"{'test':14s} {'pass':8s} {'slope':>7s} {'dSlope':>7s} "
          f"{'exc+2':>7s} {'dExc':>7s} {'bedwin':>7s} {'dBed':>7s} "
          f"{'guard':>6s} {'meas slope':>10s} {'meas bed':>9s}")
    for name in ["baseline"] + tests:
        m = load(name)
        if m is None:
            print(f"{name:14s} (no metrics.json yet)")
            continue
        for k in PASSES:
            r, b = row(m, k), row(base, k)
            if r is None:
                continue
            print(f"{name:14s} {k:8s} {fmt(r['slope'])} "
                  f"{fmt(_d(r['slope'], b['slope']))} {fmt(r['exc2'])} "
                  f"{fmt(_d(r['exc2'], b['exc2']))} {fmt(r['bed'])} "
                  f"{fmt(_d(r['bed'], b['bed']))} "
                  f"{('ok' if r['guard'] else 'FAIL'):>6s} "
                  f"{fmt(r['meas_slope']):>10s} {fmt(r['meas_bed']):>9s}")
        print()


def _all_tests():
    return [d.name for d in sorted(HYP.iterdir())
            if d.is_dir() and d.name != "baseline"]


def markdown(tests):
    """Markdown master table for the findings note."""
    base = load("baseline")
    print("| test | pass | bed-return slope dB/us | d base | excess +2 us | "
          "d base | bed window dB | d base | guard dB | measured slope/bed |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in ["baseline"] + tests:
        m = load(name)
        if m is None:
            continue
        for k in PASSES:
            if f"bed_return_tail_{k}" not in m:
                continue
            r, b = row(m, k), row(base, k)
            s_ = m[f"bed_return_tail_{k}"]["sim"]
            bs_ = base[f"bed_return_tail_{k}"]["sim"]
            sl = (s_.get(SLUG) or next(iter(s_.values())))[
                "bed_returns_slope_db_per_us"]
            bsl = (bs_.get(SLUG) or next(iter(bs_.values())))[
                "bed_returns_slope_db_per_us"]
            exc = "--" if r["exc2"] is None else f"{r['exc2']:+.1f}"
            dexc = ("--" if r["exc2"] is None or b["exc2"] is None
                    else f"{r['exc2'] - b['exc2']:+.1f}")
            meas = ("--" if r["meas_slope"] is None
                    else f"{r['meas_slope']:+.2f} / {r['meas_bed']:+.1f}")
            print(f"| {name} | {k} | {sl:+.2f} | {sl - bsl:+.2f} | {exc} | "
                  f"{dexc} | {r['bed']:+.1f} | {r['bed'] - b['bed']:+.1f} | "
                  f"{'ok' if r['guard'] else 'FAIL'} {r['guard_db']:+.0f} | "
                  f"{meas} |")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--md"]
    (markdown if "--md" in sys.argv else main)(args or _all_tests())
