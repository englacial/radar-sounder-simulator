"""Score the T5 specular/diffuse f_s scan (session artifact).

Two objectives, both averaged over the three measured passes:

  J_abs   = mean(|excess(bed+2 us)| + |slope_sim - slope_meas| * 1 us)
            -- the brief's objective. Its first term carries the ABSOLUTE
            bed level, which at att = 31 dB/km is knowingly off (the RSSNR
            median-anchoring caveat), so it cannot be driven to zero by f_s.
  J_shape = mean(|dR2| + |slope_sim - slope_meas| * 1 us), where
            R2 = tail(bed+2 us) - bed-window level, i.e. the tail measured
            AGAINST THE RUN'S OWN BED PEAK. R2 and the slope are both
            invariant to any constant reflectivity/attenuation level error,
            so J_shape scores exactly what the model is meant to fix: the
            SHAPE of the bed-return tail. J_shape selects the winner.

    uv run python claude_notes/t5_score.py [name ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "outputs" / "basal_clutter" / "hypothesis_tests"
PASSES = ["low", "mid", "high"]
SLUG = "demogorgn"


def load(name):
    p = HYP / name / "metrics.json"
    return json.loads(p.read_text())["metrics"] if p.exists() else None


def per_pass(m, key):
    t = m.get(f"bed_return_tail_{key}")
    c = m.get(f"clutter_{key}")
    if t is None or t.get("measured") is None:
        return None
    s = t["sim"].get(SLUG) or next(iter(t["sim"].values()))
    sim_bed = c["sim"]["bed_rel_surf_db"]
    meas_bed = c["measured"]["bed_rel_surf_db"]
    sim_t2 = s["level_rel_surf_db"]["+2us"]
    meas_t2 = t["measured"]["level_rel_surf_db"]["+2us"]
    exc = (t["bed_return_tail_excess_db"] or {}).get(SLUG, {}).get("+2us")
    return {
        "slope_sim": s["slope_db_per_us"],
        "slope_meas": t["measured"]["slope_db_per_us"],
        "d_slope": s["slope_db_per_us"] - t["measured"]["slope_db_per_us"],
        "excess2": exc,
        "bed_sim": sim_bed, "bed_meas": meas_bed,
        "r2_sim": sim_t2 - sim_bed, "r2_meas": meas_t2 - meas_bed,
        "d_r2": (sim_t2 - sim_bed) - (meas_t2 - meas_bed),
        "guard": s["guard"]["pass"],
        "guard_db": s["guard"]["min_bed_minus_surface_returns_db"],
    }


def score(name):
    m = load(name)
    if m is None:
        return None
    rows = {k: per_pass(m, k) for k in PASSES}
    if any(v is None for v in rows.values()):
        return None
    j_abs = sum(abs(v["excess2"]) + abs(v["d_slope"]) for v in rows.values())
    j_shape = sum(abs(v["d_r2"]) + abs(v["d_slope"]) for v in rows.values())
    return {"rows": rows, "J_abs": j_abs / 3.0, "J_shape": j_shape / 3.0}


def main(names):
    print(f"{'config':14s} {'pass':5s} {'slope':>7s} {'meas':>7s} {'dSlope':>7s}"
          f" {'R2 sim':>7s} {'R2 meas':>8s} {'dR2':>7s} {'exc+2':>7s} "
          f"{'guard':>6s}")
    best = []
    for n in names:
        sc = score(n)
        if sc is None:
            print(f"{n:14s} (incomplete)")
            continue
        for k in PASSES:
            v = sc["rows"][k]
            print(f"{n:14s} {k:5s} {v['slope_sim']:+7.2f} "
                  f"{v['slope_meas']:+7.2f} {v['d_slope']:+7.2f} "
                  f"{v['r2_sim']:+7.2f} {v['r2_meas']:+8.2f} {v['d_r2']:+7.2f} "
                  f"{v['excess2']:+7.1f} "
                  f"{('ok' if v['guard'] else 'FAIL'):>6s}"
                  f"{v['guard_db']:+7.1f}")
        print(f"{n:14s} {'JOINT':5s} J_shape {sc['J_shape']:6.2f}   "
              f"J_abs {sc['J_abs']:6.2f}")
        best.append((sc["J_shape"], sc["J_abs"], n))
    if best:
        best.sort()
        print("\nranking by J_shape (dB):")
        for j, ja, n in best:
            print(f"  {n:14s} J_shape {j:6.2f}   J_abs {ja:6.2f}")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(d.name for d in HYP.iterdir()
                                if d.is_dir() and d.name.startswith("t5p_")))
