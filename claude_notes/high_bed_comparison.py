"""HIGH-pass bed-source comparison: picked bed vs DEMOGORGN (session artifact).

Composition only -- replays both simulations from their chunk caches
(``high_pbed_klevel`` and ``att20_klevel``), which were produced with an
IDENTICAL reflectivity mapping (A = 20 dB/km, RSSNR gamma, level-anchored
K = +7.92 dB) and matched CSARP_standard processing, so the only difference
between them is the BED TOPOGRAPHY.

Writes outputs/basal_clutter/hypothesis_tests/high_bed_comparison/
(radargrams.png, bed_tail.png, metrics.json) and mirrors to
outputs/verification/basal_clutter_high_bed_comparison/.

    uv run python claude_notes/high_bed_comparison.py
"""
import datetime
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_basal_clutter as rbc      # noqa: E402

KEY = "high"
SEGMENT = "full"
ATT = 20.0
DEFICIT_DB = 3.56          # the recorded att20 contamination-aware D
K_EXPECT_DB = 7.92         # the recorded level-anchored K, reused verbatim
HYP = rbc.OUT_DEFAULT / "hypothesis_tests"
OUT = HYP / "high_bed_comparison"
CASE = "basal_clutter_high_bed_comparison"
SOURCES = [("picked_bed", "picked bed", HYP / "high_pbed_klevel"),
           ("demogorgn", "DEMOGORGN bed (seed 0)", HYP / "att20_klevel")]
STYLE = {"picked_bed": dict(color="tab:green"),
         "demogorgn": dict(color="tab:purple")}


def build():
    """Replay both bed sources of the high pass from cache."""
    axis = rbc.ref_bed_picks()
    gmap = rbc.build_rssnr_gamma(axis, SEGMENT, ATT, anchor="level",
                                 level_deficit_db=DEFICIT_DB)
    if abs(gmap["k_db"] - K_EXPECT_DB) > 1e-9:
        raise RuntimeError(f"K {gmap['k_db']} != recorded {K_EXPECT_DB} dB")
    print(f"reflectivity mapping: K {gmap['k_db']} dB (K_median "
          f"{gmap['level_anchor']['k_median_db']} + D "
          f"{gmap['level_anchor']['deficit_db']}), shared by both bed "
          "sources", flush=True)
    out = {}
    for slug, _, dirp in SOURCES:
        pb = slug == "picked_bed"
        p = rbc.prep_pass(KEY, SEGMENT, rbc.N_TRACES_FULL,
                          ref=axis if pb else None, gmap=gmap, axis=axis,
                          fine_posting=True, dgn_seed=None if pb else 0)
        sim = rbc.simulate_pass(p, dirp / "runs", ATT, True, False)
        a = rbc.analyze_pass(p, sim, proc=rbc.process_standard(p, sim))
        out[slug] = {"p": p, "sim": sim, "a": a}
        print(f"  {slug}: bed window {a['sim']['bed_rel_surf_db']:+.2f} dB "
              f"(measured {a['meas']['bed_rel_surf_db']:+.2f}), wall "
              f"{sim['wall_s']:.1f} s", flush=True)
    return gmap, out


def fig_radargrams(res):
    """Triptych: measured high pass / sim picked bed / sim DEMOGORGN, one
    shared surface-referenced twtt axis and one shared dB scale."""
    y_lo, y_hi, vmin, vmax = -1.0, 13.5, -90.0, 5.0
    s0 = rbc.S0_KM[SEGMENT]
    fig, axs = plt.subplots(3, 1, figsize=(13.0, 11.0), sharex=True,
                            sharey=True)
    p0, a0 = res["picked_bed"]["p"], res["picked_bed"]["a"]
    ref_m = 10.0 * np.log10(max(float(np.nanmedian(rbc._wpeak(
        a0["meas_arr"], p0["tw_m"], p0["dt"], p0["surf"],
        rbc.SURF_WIN_US))), 1e-300))
    rel = (p0["tw_m"] - float(np.nanmedian(p0["surf"]))) * 1e6
    m = (rel >= y_lo) & (rel <= y_hi)
    s_km = s0 + p0["s_m"] / 1e3
    axs[0].imshow(rbc._db(a0["meas_arr"])[:, m].T - ref_m, aspect="auto",
                  cmap="gray", vmin=vmin, vmax=vmax,
                  extent=[s_km[0], s_km[-1], rel[m][-1], rel[m][0]])
    axs[0].set_title(f"measured high pass 20161031_07 "
                     f"({p0['h_med']:.0f} m AGL, CSARP_standard) -- surface "
                     "returns at 0 us, bed returns along the bright lower "
                     "band", fontsize=10)
    for ax, (slug, label, _) in zip(axs[1:], SOURCES):
        r = res[slug]
        rbc._sim_radargram_panel(ax, r["p"], r["a"], KEY, f"({label})", s0,
                                 y_lo, y_hi, vmin, vmax)
        ax.set_title(f"simulated high pass -- {label} (ct "
                     f"±{r['p']['reach']['ct_m'] / 1e3:.1f} km, "
                     f"{r['p']['spacing']:.1f} m facets)", fontsize=10)
    for ax in axs:
        ax.set_ylabel("twtt below surface returns (us)")
    axs[-1].set_xlabel("anchor along-track s (km)")
    fig.suptitle("HIGH pass bed-source comparison: measured vs simulated "
                 "surface+bed returns\nidentical reflectivity mapping "
                 f"(A = {ATT:.0f} dB/km, K = {K_EXPECT_DB:+.2f} dB); "
                 f"dB rel own surface-return peak, [{vmin:.0f}, {vmax:.0f}] "
                 "dB grey scale", fontsize=11)
    fig.tight_layout()
    fp = OUT / "radargrams.png"
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


def fig_bed_tail(res, entry):
    """(a) measured vs both sims' bed-return tails; (b) the per-interface
    guard context (sim bed returns vs sim surface returns)."""
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    a0 = res["picked_bed"]["a"]
    meas_rel, meas_db = a0["bed_profs"]["measured"]
    floor = entry["measured"]["noise_floor_caveat"]["floor_rel_surf_db"]
    axs[0].plot(meas_rel, meas_db, color="black", lw=1.9,
                label=f"measured bed returns "
                      f"({entry['measured']['slope_db_per_us']:+.2f} dB/us)")
    axs[0].axhline(floor, color="0.5", lw=0.9, ls=":",
                   label=f"measured noise floor ({floor:.1f} dB)")
    for slug, label, _ in SOURCES:
        st = entry["sim"][slug]
        axs[0].plot(*res[slug]["a"]["bed_profs"]["sim_total"], lw=1.5,
                    label=f"sim {label} ({st['slope_db_per_us']:+.2f} dB/us, "
                          f"excess at +2 us "
                          f"{entry['bed_return_tail_excess_db'][slug]['+2us']:+.1f}"
                          " dB)", **STYLE[slug])
        axs[1].plot(*res[slug]["a"]["bed_profs"]["sim_bed"], lw=1.5,
                    label=f"sim bed returns ({label})", **STYLE[slug])
        axs[1].plot(*res[slug]["a"]["bed_profs"]["sim_surface"], lw=1.1,
                    ls="--", label=f"sim surface returns ({label}), guard "
                    f"{st['guard']['min_bed_minus_surface_returns_db']:+.1f}"
                    f" dB {'ok' if st['guard']['pass'] else 'FAIL'}",
                    **STYLE[slug])
    axs[1].plot(meas_rel, meas_db, color="black", lw=1.2, alpha=0.5,
                label="measured (context)")
    ang = entry["bed_return_angle_map_deg"]
    for ax, ttl, loc in zip(axs, [
            "(a) total field vs measured -- the tail metric",
            "(b) guard context: per-interface bed vs surface returns"],
            ["lower left", "upper right"]):
        ax.axvspan(*rbc.TAIL_FIT_US, color="tab:blue", alpha=0.07)
        ax.axvline(0.0, color="0.5", lw=0.8)
        ax.set_xlim(*rbc.TAIL_PROF_US)
        ax.grid(alpha=0.3)
        ax.set_xlabel("delay past own bed reference (us)")
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=7.5, loc=loc)
    axs[0].set_ylabel("dB rel own surface-return peak (mean power)")
    fig.suptitle("HIGH pass bed-return tail by bed source (each dataset on "
                 "its OWN bed reference)\nbed incidence at +1/+2/+3 us = "
                 f"{ang['+1us']:.0f}/{ang['+2us']:.0f}/{ang['+3us']:.0f} deg;"
                 " shaded = Theil-Sen fit window", fontsize=11)
    fig.tight_layout()
    fp = OUT / "bed_tail.png"
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


def _tail_pixels_db(P, twtt, dt, t_ref, norm):
    """All samples in bed+0.5 -> bed+3.5 us, dB rel each trace's own
    surface-return peak (the arc-texture population)."""
    out = []
    for t in range(P.shape[0]):
        if not (np.isfinite(t_ref[t]) and np.isfinite(norm[t])
                and norm[t] > 0):
            continue
        a = int(round((t_ref[t] + rbc.TAIL_FIT_US[0] * 1e-6 - twtt[0]) / dt))
        b = int(round((t_ref[t] + rbc.TAIL_FIT_US[1] * 1e-6 - twtt[0]) / dt))
        a, b = max(a, 0), min(b, len(twtt))
        if b > a:
            out.append(P[t, a:b] / norm[t])
    v = np.concatenate(out)
    q = np.percentile(10.0 * np.log10(np.maximum(v, 1e-30)), [50, 90, 99])
    return {"p50_db": round(float(q[0]), 2), "p90_db": round(float(q[1]), 2),
            "p99_db": round(float(q[2]), 2),
            "p90_minus_p50_db": round(float(q[1] - q[0]), 2),
            "p99_minus_p50_db": round(float(q[2] - q[0]), 2)}


def _concentration(P, twtt, dt, t_ref, norm, s_km):
    """How much of the bed-referenced ENSEMBLE MEAN at bed+1/+2/+3 us comes
    from the brightest 5 % of traces, and where the brightest trace sits."""
    out = {}
    for t in rbc.TAIL_EXCESS_US:
        v = rbc._wmean(P, twtt, dt, t_ref + (t - 0.2) * 1e-6,
                       t_ref + (t + 0.2) * 1e-6) / norm
        ok = np.isfinite(v) & (norm > 0)
        x = np.sort(v[ok])[::-1]
        k = max(1, int(0.05 * len(x)))
        out[f"+{t:g}us"] = {
            "top5pct_trace_share_of_mean": round(
                float(x[:k].sum() / x.sum()), 3),
            "brightest_trace_s_km": round(
                float(s_km[np.nanargmax(np.where(ok, v, -np.inf))]), 2)}
    return out


def tail_concentration(res):
    """Is the post-bed ensemble mean a broad tail or a few bright arcs?"""
    p0, a0 = res["picked_bed"]["p"], res["picked_bed"]["a"]
    s0 = rbc.S0_KM[SEGMENT]
    out = {"measured": _concentration(
        a0["meas_arr"], p0["tw_m"], p0["dt"], p0["bot"],
        rbc._wpeak(a0["meas_arr"], p0["tw_m"], p0["dt"], p0["surf"],
                   rbc.SURF_WIN_US), s0 + p0["s_m"] / 1e3)}
    for slug, _, _ in SOURCES:
        p, a, sim = res[slug]["p"], res[slug]["a"], res[slug]["sim"]
        rc = p["rc_frame"]
        tw = rc.t0 + np.arange(rc.n_samples) * rc.dt
        out[slug] = _concentration(
            a["P"], tw, rc.dt, sim["nadir"][:, 1],
            rbc._wpeak(a["P"], tw, rc.dt, a["t_s"], rbc.SURF_WIN_US),
            s0 + p["s_sim"] / 1e3)
    return out


def arc_texture(res):
    """Post-bed sample-level contrast: a numeric proxy for 'arc texture'
    (how peaked the post-bed field is above its own median), measured and
    simulated, over the same bed-referenced fit window as the tail slope."""
    p0, a0 = res["picked_bed"]["p"], res["picked_bed"]["a"]
    out = {"measured": _tail_pixels_db(
        a0["meas_arr"], p0["tw_m"], p0["dt"], p0["bot"],
        rbc._wpeak(a0["meas_arr"], p0["tw_m"], p0["dt"], p0["surf"],
                   rbc.SURF_WIN_US))}
    for slug, _, _ in SOURCES:
        p, a, sim = res[slug]["p"], res[slug]["a"], res[slug]["sim"]
        rc = p["rc_frame"]
        tw = rc.t0 + np.arange(rc.n_samples) * rc.dt
        out[slug] = _tail_pixels_db(
            a["P"], tw, rc.dt, sim["nadir"][:, 1],
            rbc._wpeak(a["P"], tw, rc.dt, a["t_s"], rbc.SURF_WIN_US))
    return out


def brightness_corr(res):
    """Along-track bed-return brightness correlation vs measured, 1 km
    smoothed (the established acceptance metric)."""
    p0, a0 = res["picked_bed"]["p"], res["picked_bed"]["a"]
    meas = np.interp(p0["s_sim"], p0["s_m"],
                     rbc._smooth_db(p0["s_m"], a0["meas_bed_prof_db"]))
    out = {"smooth_win_m": rbc.CORR_WIN_M,
           "measured_med_db": round(float(np.nanmedian(meas)), 2)}
    for slug, _, _ in SOURCES:
        a, s = res[slug]["a"], res[slug]["p"]["s_sim"]
        tot = rbc._smooth_db(s, a["sim_bed_prof_db"])
        out[slug] = {
            "r_total_field_vs_measured": round(
                rbc._pearson(tot, meas), 3),
            "r_bed_layer_vs_measured": round(rbc._pearson(
                rbc._smooth_db(s, a["sim_bedlayer_prof_db"]), meas), 3),
            "sim_med_db": round(float(np.nanmedian(tot)), 2)}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gmap, res = build()
    p0 = res["picked_bed"]["p"]
    entry = rbc.bed_tail_entry(KEY, p0, res["picked_bed"]["a"],
                               [(slug, res[slug]["a"])
                                for slug, _, _ in SOURCES])
    corr = brightness_corr(res)
    tex = arc_texture(res)
    conc = tail_concentration(res)
    meas = res["picked_bed"]["a"]["meas"]
    metrics = {
        "bed_window_level_high": {
            "value": round(res["demogorgn"]["a"]["sim"]["bed_rel_surf_db"]
                           - meas["bed_rel_surf_db"], 2),
            "threshold": None, "op": "record", "pass": True,
            "measured_db": meas["bed_rel_surf_db"],
            **{slug: {
                "sim_db": res[slug]["a"]["sim"]["bed_rel_surf_db"],
                "residual_db": round(
                    res[slug]["a"]["sim"]["bed_rel_surf_db"]
                    - meas["bed_rel_surf_db"], 2),
                "midcol_db": res[slug]["a"]["sim"]["midcol_rel_surf_db"],
                "decomposition_db": res[slug]["a"]["decomposition"],
                "nadir_bed_offset_vs_picks": rbc.nadir_bed_offset(
                    res[slug]["p"], res[slug]["sim"]),
                "verdict": res[slug]["a"]["verdict"]}
               for slug, _, _ in SOURCES},
            "measured_midcol_db": meas["midcol_rel_surf_db"],
            "note": "bed-return window level (bed-0.5 -> bed+1.5 us mean "
            "power, dB rel own surface-return peak, median over traces) for "
            "both bed sources against the measured high pass, plus the "
            "mid-column clutter level and the per-interface decomposition. "
            "value = the DEMOGORGN residual. recorded only"},
        "bed_return_tail_high": {**entry, "tail_concentration": {
            **conc,
            "note": "share of the bed-referenced ENSEMBLE MEAN at "
            "bed+1/+2/+3 us carried by the brightest 5 % of traces (window "
            "+-0.2 us), with the brightest trace's along-track position. A "
            "share near 1 means the level at that delay is ONE bright "
            "off-nadir bed-return arc, not a broad tail -- read the robust "
            "slope and the +1/+2 us excesses instead."}},
        "bed_brightness_correlation_high": {
            "value": corr["demogorgn"]["r_total_field_vs_measured"],
            "threshold": None, "op": "record", "pass": True, **corr,
            "note": "KEY DELIVERABLE (acceptance): along-track Pearson r of "
            "the ~1 km-smoothed bed-return window power profile (dB rel own "
            "surface-return peak) between sim and measured, on the sim "
            "trace grid. r_total_field is the established metric (the "
            "bed-source ablation's r_bed_brightness_vs_measured); "
            "r_bed_layer uses the bed-borne field only. recorded only"},
        "post_bed_arc_texture_high": {
            "value": tex["demogorgn"]["p90_minus_p50_db"],
            "threshold": None, "op": "record", "pass": True, **tex,
            "window_us": list(rbc.TAIL_FIT_US),
            "note": "sample-level contrast of the post-bed field (all "
            "samples in bed+0.5 -> bed+3.5 us, dB rel each trace's own "
            "surface-return peak): p90-p50 and p99-p50 are numeric proxies "
            "for how PEAKED the off-nadir bed-return arcs are above their "
            "own background, complementing the visual arc texture. The "
            "measured product's looks (6-11) differ from the sim's (3), so "
            "compare the sims to each other first and to measured only as a "
            "direction. recorded only"},
        "reflectivity_mapping_check": {
            "value": gmap["k_db"], "threshold": K_EXPECT_DB, "op": "==",
            "pass": bool(abs(gmap["k_db"] - K_EXPECT_DB) < 1e-9),
            "k_db": gmap["k_db"], "att_db_per_km": ATT,
            "level_anchor": gmap["level_anchor"],
            "g2_seg_db": gmap["g2_seg_db"],
            "g2_pos_frac_seg": gmap["g2_pos_frac_seg"],
            "note": "the SAME level-anchored RSSNR reflectivity field (one "
            "shared along-track |Gamma_bed|^2, independent of the bed DEM) "
            "drives both simulations, so this comparison isolates bed "
            "TOPOGRAPHY. recorded only"},
        "simulation_wall_s": {
            "value": round(sum(res[s]["sim"]["wall_s"]
                               for s, _, _ in SOURCES), 1),
            "threshold": None, "op": "record", "pass": True,
            "per_source_s": {slug: round(res[slug]["sim"]["wall_s"], 1)
                             for slug, _, _ in SOURCES},
            "note": "recorded only"}}
    figs = [fig_radargrams(res), fig_bed_tail(res, entry)]
    doc = {"case": CASE, "group": "xOPR clutter",
           "created": datetime.datetime.now(
               datetime.timezone.utc).isoformat(),
           "metrics": metrics,
           "notes": "HIGH-PASS-ONLY bed-source comparison (20161031_07, "
           f"{p0['h_med']:.0f} m AGL) on the winning att20_klevel "
           f"configuration: A = {ATT:.0f} dB/km, RSSNR bed reflectivity "
           f"level-anchored to K = {K_EXPECT_DB:+.2f} dB (reused verbatim, "
           "not re-derived), matched CSARP_standard processing, identical "
           "geometry/reach/facets. ONLY the bed DEM differs: the radar-"
           "picked bed (low-pass picks as an along-track residual on "
           "BedMachine -- cross-track INERT, the known ridge artifact) vs a "
           "plain DEMOGORGN realization (seed 0, isotropic 2-D texture, its "
           "own bed line). Terminology: 'surface returns' = the ice-surface "
           "interface field, 'bed returns' = the ice-bed interface field; "
           "the measured product carries both. Composition replays both "
           "chunk caches -- see claude_notes/high_bed_comparison.py."}
    (OUT / "metrics.json").write_text(json.dumps(doc, indent=1) + "\n")
    ver = rbc.VER_ROOT / CASE
    ver.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "metrics.json", ver / "metrics.json")
    for f in figs:
        shutil.copy2(f, ver / f.name)
    print(f"wrote {OUT} (+ mirror {ver})", flush=True)


if __name__ == "__main__":
    main()
