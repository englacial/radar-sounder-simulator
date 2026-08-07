"""Where the EXTENDED segment's bed-return tail changed (session artifact).

Splits the bed-referenced tail statistics of the extended run into the
LEGACY 50 km sub-range (anchor s 18-68 km, the att20_klevel extent) and the
two NEW pieces (s < 18 km up-track, s > 68 km down-track), for measured and
simulated alike, and reports how concentrated the ensemble mean is (share of
the mean carried by the brightest 5 % of traces). Pure cache replay of the
extended run -- no new simulations.

    uv run python claude_notes/extended_tail_split.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_basal_clutter as rbc          # noqa: E402

SEG, ATT, DEFICIT = "extended", 20.0, 3.56
RUNS = rbc.OUT_DEFAULT / "extended" / "runs"
ZONES = [("up-track (s < 18 km, NEW)", -1e9, 18.0),
         ("legacy 50 km (s 18-68 km)", 18.0, 68.0),
         ("down-track (s > 68 km, NEW)", 68.0, 1e9),
         ("whole extended segment", -1e9, 1e9)]


def stats(P, tw, dt, t_ref, norm, s_km, lo, hi):
    m = np.isfinite(t_ref) & np.isfinite(norm) & (norm > 0) \
        & (s_km >= lo) & (s_km < hi)
    if m.sum() < 20:
        return None
    rows = np.where(m)[0]
    rel, db = rbc.rel_mean_profile(P[rows], tw, dt, t_ref[rows], norm[rows],
                                   *rbc.TAIL_PROF_US)
    v = rbc._wmean(P[rows], tw, dt, t_ref[rows] + 1.8e-6,
                   t_ref[rows] + 2.2e-6) / norm[rows]
    x = np.sort(v[np.isfinite(v)])[::-1]
    k = max(1, int(0.05 * len(x)))
    return {"n": int(m.sum()),
            "slope": rbc.tail_slope_db_per_us(rel, db),
            "+1us": rbc._at_us(rel, db, 1.0),
            "+2us": rbc._at_us(rel, db, 2.0),
            "top5": float(x[:k].sum() / x.sum())}


def main():
    axis = rbc.ref_bed_picks()
    gmap = rbc.build_rssnr_gamma(axis, SEG, ATT, anchor="level",
                                 level_deficit_db=DEFICIT,
                                 k_anchor_segment=rbc.K_ANCHOR_SEGMENT[SEG])
    s0 = rbc.S0_KM[SEG]
    for key in rbc.ORDER:
        p = rbc.prep_pass(key, SEG, rbc.N_TRACES_EXT, gmap=gmap, axis=axis,
                          fine_posting=True, dgn_seed=0)
        sim = rbc.simulate_pass(p, RUNS, ATT, True, False)
        proc = rbc.process_standard(p, sim)
        a = rbc.analyze_pass(p, sim, proc=proc)
        rc = p["rc_frame"]
        tw = rc.t0 + np.arange(rc.n_samples) * rc.dt
        spk = rbc._wpeak(a["P"], tw, rc.dt, a["t_s"], rbc.SURF_WIN_US)
        mspk = rbc._wpeak(a["meas_arr"], p["tw_m"], p["dt"], p["surf"],
                          rbc.SURF_WIN_US)
        print(f"\n== {key} ({p['h_med']:.0f} m AGL) ==")
        print(f"{'zone':30s} {'n':>5s} | {'sim slope':>9s} {'+1us':>7s} "
              f"{'+2us':>7s} {'top5%':>6s} | {'meas slope':>10s} {'+1us':>7s} "
              f"{'+2us':>7s} {'top5%':>6s} | {'exc+2':>6s}")
        for name, lo, hi in ZONES:
            ss = stats(a["P"], tw, rc.dt, sim["nadir"][:, 1], spk,
                       s0 + p["s_sim"] / 1e3, lo, hi)
            mm = stats(a["meas_arr"], p["tw_m"], p["dt"], p["bot"], mspk,
                       s0 + p["s_m"] / 1e3, lo, hi)
            if ss is None or mm is None:
                continue
            print(f"{name:30s} {ss['n']:5d} | {ss['slope']:+9.2f} "
                  f"{ss['+1us']:+7.1f} {ss['+2us']:+7.1f} {ss['top5']:6.2f} "
                  f"| {mm['slope']:+10.2f} {mm['+1us']:+7.1f} "
                  f"{mm['+2us']:+7.1f} {mm['top5']:6.2f} | "
                  f"{ss['+2us'] - mm['+2us']:+6.1f}")


if __name__ == "__main__":
    main()
