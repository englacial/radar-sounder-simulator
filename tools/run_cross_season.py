"""Cross-season repeat-line comparison at the 2012 DC-8 high-altitude anchor.

Four real flights of the SAME 30.65 km line (claude_notes/
cross_season_line_scout.md): 2012_Antarctica_DC8 / 20121023_04_008 (9217 m
AGL, 9.5 MHz) and its ~450 m-AGL 50 MHz repeats 20141029_05_013 /
20161104_05_008 / 20181107_01_011. Each frame's common-window segment is
simulated at its REAL altitude and system parameters (frame's own chirp /
compression window / fast-time grid / nav incl. roll) and compared against
its own measured CSARP_standard; the KEY deliverable is the CROSS-FLIGHT
MATRIX: for every season pair, (sim_i - sim_j) vs (measured_i - measured_j)
for the surface peak level, the bed-minus-surface level, and the
surface-referenced nadir profile difference curve -- i.e. do the sims
reproduce the real altitude(20x)/bandwidth(5x)/processing differences
between the flights, with 2012-vs-repeats as the headline pairs.

CALIBRATION CAVEAT (drives the metric choices): CReSIS products are not
radiometrically cross-calibrated season to season, so measured ABSOLUTE
level differences between flights mix geometry with unknown per-season
gains. The trustworthy cross-season currencies are (a) the within-frame
bed-minus-surface level (per-season gain cancels) and (b) the
surface-peak-normalized mean-power depth profile; raw surface-level pair
deltas are recorded with the caveat, against the r^-2 expectation.

Machinery reused from tools/run_altitude_comparison.py (imported): per-era
param loading (param_sar/param_csarp layouts, tukey->none window mapping,
alias-free oversample search), REMA+BedMachine scene building, cached
surface+bed and chunked firn-strip runs, --surf-rough (validated
representative surface roughness), soundersim.firn effective contrasts (B25
REPRESENTATIVE Antarctic proxy, N=10, 15 dB/km), mean-power profiles.

Scout pitfalls honored: per-frame surface registration (constant-offset
leading-edge gate, never shared); metre-domain profile smoothing (105.21 vs
20 ns grids); 2016's 20.202 ns lattice (generic dt handling); 2014/2016
ft_wind provenance quirk (fallback string happens to be the scout-verified
'hanning' -- recorded, not quoted as measured); 2014+ img_comb 3-waveform
composite vs our single 10 us bed waveform (recorded); 2012 roll (nav roll
drives the array pattern); qlook posting incomparability (qlook unused).

Run: uv run python tools/run_cross_season.py            # all four frames
     uv run python tools/run_cross_season.py --n-traces 2 --out <scratch>
                                                        # pilot
"""

import argparse
import base64
import datetime
import html
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pyproj import Transformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_altitude_comparison as rac  # noqa: E402  shared machinery
from run_opr_comparison import _db  # noqa: E402

from soundersim import firn  # noqa: E402
from soundersim.config import FacetConfig, SimConfig  # noqa: E402
from soundersim.opr import load_bottom_pick, load_frame  # noqa: E402

C = 299792458.0
OUT_DEFAULT = ROOT / "outputs" / "cross_season"
VER_OUT = ROOT / "outputs" / "verification" / "cross_season"
CASE = "cross_season_20121023_04_008"

# Common window (scout note): anchor s = 8.46-39.11 km, 30.65 km; slices are
# half-open slow_time indices into each FULL frame.
FRAMES = {
    "2012": {"season": "2012_Antarctica_DC8", "frame": "20121023_04_008",
             "sl": (286, 1320), "quirks": "anchor; 9217 m AGL, 9.5 MHz, "
             "tukey(0.2)->none window approx; 105.21 ns critically-sampled "
             "grid; |roll| p95 2.7 deg (nav roll in the array pattern)"},
    "2014": {"season": "2014_Antarctica_DC8", "frame": "20141029_05_013",
             "sl": (1, 2067), "quirks": "ft_wind provenance = decode-fallback "
             "string; scout-verified hanning via param_csarp.csarp.ft_wind. "
             "img_comb 3-waveform composite (sim = 10 us bed waveform only)"},
    "2016": {"season": "2016_Antarctica_DC8", "frame": "20161104_05_008",
             "sl": (1058, 3124), "quirks": "20.202 ns grid (not 20.000); "
             "ft_wind provenance quirk as 2014; img_comb composite"},
    "2018": {"season": "2018_Antarctica_DC8", "frame": "20181107_01_011",
             "sl": (1267, 3333), "quirks": "img_comb composite; params decode "
             "cleanly via param_sar"},
}
YEARS = list(FRAMES)
HEADLINE = [("2012", y) for y in YEARS[1:]]

CT_CAP = 6000.0
CT_FIRN = 600.0
FIRN_N = 10
# Effective one-way column loss for the BED runs' ice medium (--att),
# CALIBRATED 2026-07-30 on the three low-altitude repeats' measured
# bed-minus-surface (-70 dB rel surface, 21-26 dB bed SNR): 15 dB/km (the
# b26/altitude value) left the sim bed ~28 dB hot; +16 dB/km closes all
# three frames to within 0.5-2.3 dB AND drops the 2012 sim bed below that
# frame's own measured noise floor (-42 dB rel surface) exactly as observed
# (its bed SNR is ~0 dB). An EFFECTIVE value: absorbs true attenuation
# (warm West Antarctic ice), bed-roughness scattering loss and any other
# unmodeled column loss -- not a temperature-derived attenuation. The firn
# strip keeps the b26-validated 15 dB/km (its 0-178 m path sees <= ~6 dB of
# the difference; recorded inconsistency).
ATT_EFF = 31.0
EPS_COL = 3.17          # full-column twtt->depth conversion for profiles
SMOOTH_M = 10.0         # metre-domain profile smoothing (>= the 2012 bin)
DIFF_Z = (20.0, 500.0)  # profile-difference band (above the shallowest bed)
DIFF_DZ = 5.0
SURF_WIN_US, BED_WIN_US = 0.8, 1.0
PROF_LO_M = 20.0


# ========================================================================
# per-frame preparation + cached simulation
# ========================================================================
def prep(year, n_traces):
    """Everything needed to simulate + compare one frame's common window."""
    spec = FRAMES[year]
    season, fid = spec["season"], spec["frame"]
    params = rac.mcords_params(season, fid)
    wf = params["waveform"]
    f0, bw = wf["center_frequency_Hz"], wf["bandwidth_Hz"]
    window, win_note = rac.map_window(wf["pulse_compression_freq_window"])
    if year in ("2014", "2016"):  # scout pitfall 4: fallback happens correct
        win_note = ("ft_wind provenance is mcords_params' decode-fallback "
                    "string; the scout verified the real value IS hanning "
                    "(param_csarp.csarp.ft_wind) -- modeled window 'hann' is "
                    "correct but the provenance string is not a measurement")
    frame = load_frame(season, fid)
    bot_full = load_bottom_pick(frame)
    a, b = spec["sl"]
    fsub = frame.isel(slow_time=slice(a, b))
    bot_sub = bot_full[a:b]
    tw = frame.twtt.values
    dt = float((tw[-1] - tw[0]) / (len(tw) - 1))
    t0f = float(tw[0])
    oversample, f_alias = rac.pick_oversample(dt, f0, bw)

    surf = np.asarray(fsub.Surface.values, np.float64)
    r_min = float(np.nanmin(surf)) * C / 2.0
    thick = (bot_sub - surf) * C / (2.0 * np.sqrt(rac.EPS_ICE))
    thick_med = float(np.nanmedian(thick))
    c_hi = C * (float(np.nanmax(bot_sub)) + rac.POST_BED_US * 1e-6) / 2.0
    ct = min(float(np.sqrt(max(c_hi ** 2 - r_min ** 2, 0.0))), CT_CAP)

    base, aux = rac.base_scene(fsub, n_traces, ct)
    idx = aux["idx"]
    lam = C / f0
    spacing = rac.facet_spacing(lam, r_min, thick_med)
    rc_sim, rc_frame, b0 = rac.radar_grid(
        params, surf, np.where(np.isfinite(bot_sub), bot_sub,
                               np.nanmax(bot_sub)), dt, t0f, oversample,
        window)
    # along-track axis (EPSG:3031)
    lat, lon = rac._lonlat(fsub)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    px, py = tr.transform(lon, lat)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    return {
        "year": year, "season": season, "frame_id": fid, "params": params,
        "window": window, "win_note": win_note, "frame": frame, "fsub": fsub,
        "bot_sub": bot_sub, "surf": surf, "dt": dt, "t0f": t0f,
        "oversample": oversample, "f_alias": f_alias, "r_min": r_min,
        "thick_med": thick_med, "ct": ct, "base": base, "idx": idx,
        "aux": aux, "lam": lam, "spacing": spacing, "rc_sim": rc_sim,
        "rc_frame": rc_frame, "b0": b0, "s_m": s,
        "agl_med": float(np.nanmedian(surf)) * C / 2.0,  # nadir air range
    }


def bed_cfg(rc_sim, spacing, surf_rough, att):
    """Surface+bed config with the tool's effective ice attenuation (ATT_EFF
    provenance note) and the validated representative surface roughness."""
    from soundersim.config import (DemInterface, Medium, RoughnessConfig)
    rcg = (RoughnessConfig(sigma_m=rac.SURF_ROUGH_SIGMA_M,
                           corr_length_m=rac.SURF_ROUGH_CL_M)
           if surf_rough else None)
    return SimConfig(
        mode="coherent", split_sides=False, radar=rc_sim,
        facets=FacetConfig(spacing=spacing),
        media=[Medium(name="air", eps_r=1.0),
               Medium(name="ice", eps_r=rac.EPS_ICE,
                      attenuation_db_per_km=att),
               Medium(name="bed", eps_r=rac.EPS_BED)],
        interfaces=[DemInterface(name="surface", roughness=rcg),
                    DemInterface(name="bed")])


def simulate_frame(p, runs_dir, firn_n, surf_rough, force, att=ATT_EFF):
    """Cached surface+bed (+ optional firn strip) runs for one frame."""
    year = p["year"]
    rid = (f"{year}_bed" + ("_srough" if surf_rough else "")
           + (f"_att{att:g}" if att != rac.ATT_DB_PER_KM else ""))
    meta = {"season": p["season"], "frame_id": p["frame_id"],
            "n_traces": len(p["idx"]), "spacing_m": round(p["spacing"], 4),
            "ct_m": round(p["ct"], 1), "window": p["window"],
            "att_db_per_km": att,
            "dt_sim_ns": round(p["rc_sim"].dt * 1e9, 5),
            "t0_us": round(p["rc_sim"].t0 * 1e6, 5),
            "n_samples_sim": p["rc_sim"].n_samples}
    if surf_rough:
        meta["surf_rough"] = [rac.SURF_ROUGH_SIGMA_M, rac.SURF_ROUGH_CL_M]
    diag, arrs = rac.run_level(
        rid, p["base"], bed_cfg(p["rc_sim"], p["spacing"], surf_rough, att),
        meta, runs_dir, p["oversample"], force)
    out = {"bed": (diag, arrs)}
    if firn_n:
        core, region, label, note = rac.firn_core_for(-75.0)
        depths = core.equal_depths(firn_n)
        eps, _ = core.effective_contrast_eps(depths, p["lam"])
        sp_f = rac.firn_facet_spacing(p["lam"], p["r_min"], core)
        chunks = rac.firn_strip_scenes(p["base"], CT_FIRN,
                                       p["base"].nav_llh[:, 2])
        media, ifaces = firn.firn_stack(depths, eps, rac.ATT_DB_PER_KM)
        fcfg = SimConfig(mode="coherent", split_sides=False, radar=p["rc_sim"],
                         facets=FacetConfig(spacing=sp_f), media=media,
                         interfaces=ifaces)
        fmeta = {"season": p["season"], "frame_id": p["frame_id"],
                 "kind": f"firn{firn_n}_h1eff", "core": label,
                 "n_traces": len(p["idx"]), "spacing_m": round(sp_f, 4),
                 "ct_firn_m": CT_FIRN, "att_db_per_km": rac.ATT_DB_PER_KM,
                 "dt_sim_ns": round(p["rc_sim"].dt * 1e9, 5),
                 "t0_us": round(p["rc_sim"].t0 * 1e6, 5),
                 "n_samples_sim": p["rc_sim"].n_samples,
                 "depths_hash": round(float(depths.sum()), 4),
                 "eps_sum": round(float(np.sum(eps)), 6)}
        fdiag, farrs = rac.run_firn_level(f"{year}_firn{firn_n}", chunks,
                                          fcfg, fmeta, runs_dir,
                                          p["oversample"], force)
        out["firn"] = (fdiag, farrs)
        out["firn_label"], out["firn_note"] = label, note
        out["firn_spacing"] = sp_f
    return out


# ========================================================================
# per-frame analysis
# ========================================================================
def peak_db(P, twtt, t_guess, dt, win_us):
    """Per-trace 10log10(max power) within +-win_us of a guess (NaN where the
    guess is)."""
    n = len(twtt)
    out = np.full(P.shape[0], np.nan)
    for t in range(P.shape[0]):
        if not np.isfinite(t_guess[t]):
            continue
        a = int(np.clip((t_guess[t] - win_us * 1e-6 - twtt[0]) / dt, 0, n - 2))
        b = int(np.clip((t_guess[t] + win_us * 1e-6 - twtt[0]) / dt, a + 1, n))
        out[t] = 10.0 * np.log10(max(float(P[t, a:b].max()), 1e-300))
    return out


def frame_analysis(p, runs, firn_n):
    """Sim fields -> per-frame comparison against the frame's own measured
    CSARP_standard: surface gate, gain-free bed-minus-surface, mean-power
    profiles (surface-referenced, metre-smoothed), profile correlation."""
    _, arrs = runs["bed"]
    E = arrs["field"].sum(-1)
    if "firn" in runs:
        E = E + runs["firn"][1]["field"][..., 1:].sum(-1)
    tw = arrs["twtt"]
    dt = p["rc_frame"].dt
    nadir = arrs["nadir_twtt"]
    P = np.abs(E) ** 2
    surf_pick = p["surf"][p["idx"]]

    gate = rac.leading_edge_gate(np.abs(arrs["field"][..., 0]) ** 2,
                                 p["spacing"], dt, p["rc_frame"].t0, surf_pick)
    # gain-free within-frame scalar: bed peak minus surface peak (median)
    s_db = peak_db(P, tw, nadir[:, 0], dt, SURF_WIN_US)
    b_db = peak_db(P, tw, nadir[:, 1], dt, BED_WIN_US)
    meas = np.asarray(p["fsub"].Data.values, np.float64)
    tw_m = p["frame"].twtt.values
    dt_m = p["dt"]
    surf_all = p["surf"]
    ms_db = peak_db(meas, tw_m, surf_all, dt_m, SURF_WIN_US)
    mb_db = peak_db(meas, tw_m, p["bot_sub"], dt_m, BED_WIN_US)
    ok_s = np.isfinite(s_db) & np.isfinite(b_db)
    ok_m = np.isfinite(ms_db) & np.isfinite(mb_db)
    bs_sim = float(np.median((b_db - s_db)[ok_s]))
    bs_meas = float(np.median((mb_db - ms_db)[ok_m]))

    # measured noise floor rel surface (gain-free): per-trace median power in
    # the mid-column quiet window 2-3 us ABOVE the bed pick, over the trace's
    # own surface peak. The sims carry no receiver noise, so this floor is
    # applied to the sim (max(profile, floor)) for the noise-aware rows --
    # decisive here: the 2012 high-altitude frame's bed has ~0 dB SNR (its
    # measured "bed" IS the floor at ~-42 dB rel surface) while the repeats
    # detect the bed at ~-70 with >20 dB SNR.
    nf = []
    n_m = len(tw_m)
    for t in range(meas.shape[0]):
        if not (np.isfinite(p["bot_sub"][t]) and np.isfinite(ms_db[t])):
            continue
        bp = int(np.clip((p["bot_sub"][t] - tw_m[0]) / dt_m, 0, n_m - 1))
        n0 = int(np.clip(bp - int(3e-6 / dt_m), 0, n_m - 2))
        n1 = int(np.clip(bp - int(2e-6 / dt_m), n0 + 1, n_m - 1))
        nf.append(10.0 * np.log10(max(float(np.median(meas[t, n0:n1])),
                                      1e-300)) - ms_db[t])
    floor_db = float(np.median(nf)) if nf else float("nan")
    bed_snr_meas = bs_meas - floor_db

    prof_sim = rac.mean_power_profile(P, tw, nadir[:, 0], dt, EPS_COL,
                                      smooth_m=SMOOTH_M)
    prof_meas = rac.mean_power_profile(meas, tw_m, surf_all, dt_m, EPS_COL,
                                       smooth_m=SMOOTH_M)
    prof_sim_fl = (prof_sim[0], np.maximum(prof_sim[1], floor_db))
    z_hi = float(np.nanmax((p["bot_sub"] - p["surf"]))
                 * C / (2.0 * np.sqrt(EPS_COL))) if np.isfinite(
        p["bot_sub"]).any() else 1200.0
    corr = rac.profile_corr(prof_meas, prof_sim, lo=PROF_LO_M,
                            hi=min(z_hi, 1200.0))
    corr_fl = rac.profile_corr(prof_meas, prof_sim_fl, lo=PROF_LO_M,
                               hi=min(z_hi, 1200.0))
    return {"E": E, "twtt": tw, "nadir": nadir, "meas": meas, "tw_m": tw_m,
            "gate": gate, "surf_db_sim": float(np.median(s_db[ok_s])),
            "surf_db_meas": float(np.median(ms_db[ok_m])),
            "bedsurf_sim": bs_sim, "bedsurf_meas": bs_meas,
            "floor_db": floor_db, "bed_snr_meas_db": bed_snr_meas,
            "bedsurf_sim_fl": max(bs_sim, floor_db),
            "prof_sim": prof_sim, "prof_meas": prof_meas,
            "prof_sim_fl": prof_sim_fl,
            "prof_corr": corr, "prof_corr_fl": corr_fl, "z_bed_max": z_hi}


# ========================================================================
# cross-flight difference matrix
# ========================================================================
def _interp(z, prof):
    d, db = prof
    return np.interp(z, d, db)


def pair_matrix(analyses, agls):
    """For every season pair: sim-vs-measured agreement of the INTER-FLIGHT
    differences. Gain-free rows: bed-minus-surface delta and the
    surface-normalized profile-difference curve; the raw surface-level delta
    is recorded with the calibration caveat (vs the r^-2 expectation)."""
    z = np.arange(DIFF_Z[0], DIFF_Z[1] + DIFF_DZ, DIFF_DZ)
    out = {}
    for yi, yj in itertools.combinations(YEARS, 2):
        ai, aj = analyses[yi], analyses[yj]
        real_bs = ai["bedsurf_meas"] - aj["bedsurf_meas"]
        sim_bs = ai["bedsurf_sim"] - aj["bedsurf_sim"]
        dm = _interp(z, ai["prof_meas"]) - _interp(z, aj["prof_meas"])
        ds = _interp(z, ai["prof_sim"]) - _interp(z, aj["prof_sim"])
        fin = np.isfinite(dm) & np.isfinite(ds)
        pc = float(np.corrcoef(dm[fin], ds[fin])[0, 1])
        rms = float(np.sqrt(np.mean((dm[fin] - ds[fin]) ** 2)))
        bias = float(np.mean((ds - dm)[fin]))
        real_surf = ai["surf_db_meas"] - aj["surf_db_meas"]
        sim_surf = ai["surf_db_sim"] - aj["surf_db_sim"]
        r2 = -20.0 * np.log10(agls[yi] / agls[yj])
        # noise-aware: sim floored at each frame's own measured floor
        fl_bs = (ai["bedsurf_sim_fl"] - aj["bedsurf_sim_fl"])
        dsf = (_interp(z, ai["prof_sim_fl"])
               - _interp(z, aj["prof_sim_fl"]))
        pcf = float(np.corrcoef(dm[fin], dsf[fin])[0, 1])
        rmsf = float(np.sqrt(np.mean((dm[fin] - dsf[fin]) ** 2)))
        snr_note = "; ".join(
            f"{y} bed SNR {a['bed_snr_meas_db']:+.1f} dB"
            + (" (NOISE-LIMITED: measured bedsurf is an upper bound)"
               if a["bed_snr_meas_db"] < 3.0 else "")
            for y, a in ((yi, ai), (yj, aj)))
        out[f"{yi}-{yj}"] = {
            "headline": (yi, yj) in HEADLINE,
            "bedsurf_delta_db": {
                "measured": round(real_bs, 2), "sim": round(sim_bs, 2),
                "error": round(sim_bs - real_bs, 2),
                "captured_frac": round(
                    1.0 - abs(sim_bs - real_bs) / max(abs(real_bs), 1e-9), 3)},
            "noise_aware": {
                "sim_floored_bedsurf_delta_db": round(fl_bs, 2),
                "captured_frac": round(
                    1.0 - abs(fl_bs - real_bs) / max(abs(real_bs), 1e-9), 3),
                "profile_diff_corr": round(pcf, 4),
                "profile_diff_rms_db": round(rmsf, 2),
                "note": "sim profiles/bed levels floored at each frame's own "
                "measured mid-column noise floor (rel surface, gain-free). "
                + snr_note},
            "profile_diff_curve": {
                "corr": round(pc, 4), "rms_db": round(rms, 2),
                "bias_db": round(bias, 2),
                "band_m": list(DIFF_Z)},
            "surface_delta_db_UNCALIBRATED": {
                "measured": round(real_surf, 2), "sim": round(sim_surf, 2),
                "r2_expectation": round(r2, 2),
                "note": "measured cross-season levels include unknown "
                "per-season gains -- record only"}}
    return out, z


# ========================================================================
# figures
# ========================================================================
def fig_radargrams(out, preps, analyses):
    """Measured vs sim panels per season, shared 'twtt below own median
    surface' axis (us)."""
    y_hi = max(a["z_bed_max"] for a in analyses.values()) \
        * 2.0 * np.sqrt(EPS_COL) / C * 1e6 + 2.0
    fig, axs = plt.subplots(2, len(YEARS), figsize=(5.6 * len(YEARS), 9.4),
                            sharey=True, squeeze=False)
    for k, year in enumerate(YEARS):
        p, a = preps[year], analyses[year]
        s_km = p["s_m"] / 1e3
        surf_med = float(np.nanmedian(p["surf"]))
        img = _db(a["meas"])
        rel = (a["tw_m"] - surf_med) * 1e6
        m = (rel >= -1.0) & (rel <= y_hi)
        fin = img[:, m][np.isfinite(img[:, m])]
        vmax = np.percentile(fin, 99.5)
        ax = axs[0, k]
        ax.imshow(img[:, m].T, aspect="auto", cmap="gray",
                  extent=[s_km[0], s_km[-1], rel[m][-1], rel[m][0]],
                  vmin=vmax - 100.0, vmax=vmax)
        ax.set_title(f"{year} measured ({p['agl_med']:.0f} m AGL, "
                     f"{p['params']['waveform']['bandwidth_Hz'] / 1e6:.1f}"
                     f" MHz)", fontsize=10)
        idx = p["idx"]
        s_sim = p["s_m"][idx] / 1e3
        surf_med_s = float(np.nanmedian(a["nadir"][:, 0]))
        comb = _db(np.abs(a["E"]) ** 2)
        rel_s = (a["twtt"] - surf_med_s) * 1e6
        ms = (rel_s >= -1.0) & (rel_s <= y_hi)
        fin = comb[:, ms][np.isfinite(comb[:, ms]) & (comb[:, ms] > -290)]
        vmax = np.percentile(fin, 99.5)
        ax = axs[1, k]
        ax.imshow(comb[:, ms].T, aspect="auto", cmap="gray",
                  extent=[s_sim[0], s_sim[-1], rel_s[ms][-1], rel_s[ms][0]],
                  vmin=vmax - 100.0, vmax=vmax)
        ax.set_title(f"{year} sim (spacing {p['spacing']:.1f} m)", fontsize=10)
        ax.set_xlabel("along-track (km)")
    for r in range(2):
        axs[r, 0].set_ylabel("twtt below median surface (us)")
    fig.suptitle("cross-season repeat line: measured (top) vs simulated "
                 "(bottom), common 30.65 km window")
    fig.tight_layout()
    fp = out / "radargrams.png"
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


def fig_profiles(out, analyses):
    fig, axs = plt.subplots(1, len(YEARS), figsize=(4.8 * len(YEARS), 4.6),
                            sharey=True, squeeze=False)
    for k, year in enumerate(YEARS):
        a = analyses[year]
        ax = axs[0, k]
        ax.plot(*a["prof_meas"], "k", lw=1.6, label="measured")
        ax.plot(*a["prof_sim"], "tab:blue", lw=1.2, label="sim")
        ax.plot(*a["prof_sim_fl"], "tab:orange", lw=1.0, ls="--",
                label="sim + meas. noise floor")
        ax.axhline(a["floor_db"], color="0.5", lw=0.7, ls=":",
                   label=f"floor {a['floor_db']:.0f} dB")
        ax.axvline(a["z_bed_max"], color="tab:red", ls=":", lw=0.8)
        ax.set_xlim(0, a["z_bed_max"] + 200)
        ax.set_ylim(-110, 3)
        ax.grid(alpha=0.3)
        ax.set_title(f"{year} (r={a['prof_corr']:.3f} / floored "
                     f"{a['prof_corr_fl']:.3f})", fontsize=10)
        ax.set_xlabel(f"depth (m, c/sqrt({EPS_COL}))")
        if k == 0:
            ax.set_ylabel("dB rel surface peak (mean power)")
            ax.legend(fontsize=8)
    fig.suptitle("nadir mean-power depth profiles, sim vs measured")
    fig.tight_layout()
    fp = out / "profiles.png"
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


def fig_diff_matrix(out, analyses, matrix, z):
    pairs = list(matrix)
    ncols = 3
    nrows = -(-len(pairs) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.2 * nrows),
                            sharex=True, squeeze=False)
    axs = axs.ravel()
    for ax in axs[len(pairs):]:
        ax.set_visible(False)
    for k, pair in enumerate(pairs):
        yi, yj = pair.split("-")
        ai, aj = analyses[yi], analyses[yj]
        dm = _interp(z, ai["prof_meas"]) - _interp(z, aj["prof_meas"])
        ds = _interp(z, ai["prof_sim"]) - _interp(z, aj["prof_sim"])
        dsf = _interp(z, ai["prof_sim_fl"]) - _interp(z, aj["prof_sim_fl"])
        ax = axs[k]
        ax.plot(z, dm, "k", lw=1.5, label="measured_i - measured_j")
        ax.plot(z, ds, "tab:blue", lw=1.2, label="sim_i - sim_j")
        ax.plot(z, dsf, "tab:orange", lw=1.0, ls="--",
                label="sim (noise-floored)")
        ax.axhline(0, color="0.6", lw=0.6)
        m = matrix[pair]["profile_diff_curve"]
        mf = matrix[pair]["noise_aware"]
        hl = "  [HEADLINE]" if matrix[pair]["headline"] else ""
        ax.set_title(f"{pair}{hl}  r={m['corr']:.2f} / floored "
                     f"{mf['profile_diff_corr']:.2f} "
                     f"rms={mf['profile_diff_rms_db']:.1f} dB", fontsize=10)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel("profile difference (dB)")
    for ax in axs[max(0, len(pairs) - ncols):len(pairs)]:
        ax.set_xlabel("depth (m)")
    fig.suptitle("inter-flight nadir-profile differences: measured vs "
                 "simulated (surface-peak-normalized, gain-free)")
    fig.tight_layout()
    fp = out / "diff_matrix.png"
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


# ========================================================================
# main
# ========================================================================
def run(out_root=None, n_traces=100, firn_n=FIRN_N, surf_rough=True,
        force=False, make_report=True, att=ATT_EFF):
    out = Path(out_root or OUT_DEFAULT)
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"
    t_all = time.perf_counter()
    preps, runs, analyses = {}, {}, {}
    for year in YEARS:
        print(f"== {year} ==", flush=True)
        p = prep(year, n_traces)
        if p["win_note"]:
            print(f"  [note] {year}: {p['win_note']}", flush=True)
        preps[year] = p
        runs[year] = simulate_frame(p, runs_dir, firn_n, surf_rough, force,
                                    att)
        analyses[year] = frame_analysis(p, runs[year], firn_n)
    agls = {y: preps[y]["agl_med"] for y in YEARS}
    matrix, z = pair_matrix(analyses, agls)
    wall = time.perf_counter() - t_all

    # ---- metrics ----
    rec = "recorded only"
    metrics = {}
    for year in YEARS:
        a, p = analyses[year], preps[year]
        g = a["gate"]
        metrics[f"surface_alignment_{year}"] = {
            "value": g["median_bins"], "threshold": rac.GATE_BINS, "op": "<=",
            "pass": bool(g["median_bins"] <= rac.GATE_BINS),
            "offset_bins": g["offset_bins"], "p90_bins": g["p90_bins"],
            "note": "per-frame constant-offset leading-edge gate (scout "
            "pitfall: season registrations differ by up to 23 m -- never "
            "share offsets across frames)"}
        metrics[f"frame_match_{year}"] = {
            "value": a["prof_corr"], "threshold": None, "op": "record",
            "pass": True,
            "corr_noise_floored": round(a["prof_corr_fl"], 4),
            "bedsurf_sim_db": round(a["bedsurf_sim"], 2),
            "bedsurf_meas_db": round(a["bedsurf_meas"], 2),
            "noise_floor_rel_surf_db": round(a["floor_db"], 2),
            "bed_snr_meas_db": round(a["bed_snr_meas_db"], 2),
            "bed_noise_limited": bool(a["bed_snr_meas_db"] < 3.0),
            "agl_m": round(p["agl_med"], 0),
            "bandwidth_mhz": p["params"]["waveform"]["bandwidth_Hz"] / 1e6,
            "note": "Pearson r of the mean-power depth profile vs the "
            "frame's own measured CSARP_standard "
            f"({PROF_LO_M:.0f} m-bed); bed-minus-surface medians (gain-free) "
            "alongside. bed_noise_limited: the measured bed peak sits at the "
            "frame's own mid-column noise floor (rel surface) -- its "
            "measured bedsurf is an UPPER BOUND, and the noise-aware matrix "
            "rows floor the sim there. " + rec}
    hl = {k: v for k, v in matrix.items() if v["headline"]}
    metrics["cross_flight_matrix"] = {
        "value": float(np.mean([v["noise_aware"]["profile_diff_corr"]
                                for v in hl.values()])),
        "value_unfloored": float(np.mean([v["profile_diff_curve"]["corr"]
                                          for v in hl.values()])),
        "threshold": None, "op": "record", "pass": True,
        "pairs": matrix,
        "note": "KEY DELIVERABLE: per season pair, sim-vs-measured agreement "
        "of the INTER-FLIGHT differences. value = mean profile-difference "
        "correlation over the headline 2012-vs-repeat pairs (20x altitude, "
        "5x bandwidth). bedsurf rows are gain-free; raw surface deltas are "
        "uncalibrated across seasons (recorded vs r^-2). " + rec}
    metrics["simulation_wall_s"] = {
        "value": round(sum(r["bed"][0]["wall_s"]
                           + (r["firn"][0]["wall_s"] if "firn" in r else 0.0)
                           for r in runs.values()), 1),
        "threshold": None, "op": "record", "pass": True,
        "wall_this_invocation_s": round(wall, 1), "note": rec}

    config = {
        "case": CASE, "common_window_km": 30.65,
        "frames": {y: {k: v for k, v in FRAMES[y].items() if k != "sl"}
                   | {"slice": list(FRAMES[y]["sl"]),
                      "n_traces": len(preps[y]["idx"]),
                      "agl_med_m": round(preps[y]["agl_med"], 0),
                      "spacing_m": round(preps[y]["spacing"], 3),
                      "ct_m": round(preps[y]["ct"], 0),
                      "oversample": preps[y]["oversample"],
                      "window_modeled": preps[y]["window"],
                      "wall_s_bed": runs[y]["bed"][0]["wall_s"],
                      "wall_s_firn": (runs[y]["firn"][0]["wall_s"]
                                      if "firn" in runs[y] else None)}
                   for y in YEARS},
        "firn": None if not firn_n else {
            "n_layers": firn_n, "core": runs[YEARS[0]]["firn_label"],
            "proxy_note": runs[YEARS[0]]["firn_note"],
            "ct_firn_m": CT_FIRN, "att_db_per_km": rac.ATT_DB_PER_KM},
        "surf_rough": None if not surf_rough else {
            "sigma_m": rac.SURF_ROUGH_SIGMA_M,
            "corr_length_m": rac.SURF_ROUGH_CL_M},
        "ice_attenuation": {
            "bed_runs_db_per_km_one_way": att,
            "firn_runs_db_per_km_one_way": rac.ATT_DB_PER_KM,
            "provenance": "EFFECTIVE column loss calibrated on the repeats' "
            "measured bed-minus-surface (-70 dB, >20 dB SNR): absorbs true "
            "attenuation + bed-roughness scattering + unmodeled losses; NOT "
            "a temperature-derived value. The firn strip keeps the "
            "b26-validated 15 (<= ~6 dB inconsistency over its 178 m)"},
        "profile": {"eps_col": EPS_COL, "smooth_m": SMOOTH_M,
                    "diff_band_m": list(DIFF_Z)},
        "measured_reference": "CSARP_standard (img_comb caveat: 2014+ is a "
        "1/3/10 us composite, sims carry only the 10 us bed waveform; qlook "
        "unused -- per-season decimation makes it incomparable)",
    }
    notes = (
        "Cross-season repeat line (claude_notes/cross_season_line_scout.md): "
        "the 2012 high-altitude 9.5 MHz anchor vs its 450 m 50 MHz repeats "
        "over the common 30.65 km, each simulated at its REAL altitude and "
        "system parameters and compared to its own measured frame; the key "
        "product is the cross-flight difference matrix (gain-free "
        "bed-minus-surface and surface-normalized profile differences; raw "
        "level deltas uncalibrated across seasons). Sims: coherent "
        "surface+bed (REMA 32 m + BedMachine) "
        + (f"+ B25-proxy firn N={firn_n} " if firn_n else "(no firn) ")
        + ("with representative surface roughness (--surf-rough), "
           if surf_rough else "smooth surface, ")
        + "frame's own chirp/window/grid, nav roll in the 7-element array "
        "pattern (the 2012 frame rolls to 5 deg), alias-free dt/k grids, "
        "metre-domain smoothing (105.21 vs 20 ns lattices).")
    doc = {"case": CASE, "group": "xOPR clutter",
           "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "metrics": metrics, "notes": notes}
    (out / "metrics.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out / "run_config.json").write_text(json.dumps(config, indent=1) + "\n")

    figs = [fig_radargrams(out, preps, analyses),
            fig_profiles(out, analyses),
            fig_diff_matrix(out, analyses, matrix, z)]
    if make_report:
        _report(out, config, metrics, notes, figs, matrix)
    VER_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / "metrics.json", VER_OUT / "metrics.json")
    for f in figs:
        shutil.copy2(f, VER_OUT / f.name)
    print("pairs: " + " | ".join(
        f"{k}: bs {v['bedsurf_delta_db']['measured']:+.1f}/"
        f"{v['bedsurf_delta_db']['sim']:+.1f} dB, "
        f"r {v['profile_diff_curve']['corr']:.2f}"
        for k, v in matrix.items()), flush=True)
    return metrics, out


def _report(out, config, metrics, notes, figs, matrix):
    def b64(p):
        return base64.b64encode(Path(p).read_bytes()).decode()

    css = ("body{font-family:system-ui,Arial,sans-serif;margin:2rem;"
           "max-width:1250px}table{border-collapse:collapse;margin:1rem 0;"
           "font-size:.82rem}th,td{border:1px solid #ccc;padding:.3rem .5rem}"
           "th{background:#f0f0f0}img{max-width:100%;border:1px solid #ddd}"
           ".note{background:#f6f6f6;border-left:3px solid #bbb;"
           "padding:.6rem 1rem}td.pass{background:#c8f7c5}"
           "td.fail{background:#f7c5c5}")
    prow = "".join(
        f"<tr><th>{html.escape(k)}{' *' if v['headline'] else ''}</th>"
        f"<td>{v['bedsurf_delta_db']['measured']:+.2f}</td>"
        f"<td>{v['bedsurf_delta_db']['sim']:+.2f}</td>"
        f"<td>{v['bedsurf_delta_db']['captured_frac']:.2f}</td>"
        f"<td>{v['profile_diff_curve']['corr']:.3f}</td>"
        f"<td>{v['profile_diff_curve']['rms_db']:.2f}</td>"
        f"<td>{v['surface_delta_db_UNCALIBRATED']['measured']:+.1f} / "
        f"{v['surface_delta_db_UNCALIBRATED']['sim']:+.1f} / "
        f"{v['surface_delta_db_UNCALIBRATED']['r2_expectation']:+.1f}</td>"
        f"</tr>" for k, v in matrix.items())
    mrows = "".join(
        f"<tr><th>{html.escape(k)}</th>"
        f"<td class='{'pass' if e.get('pass') else 'fail'}'>"
        f"{e.get('value'):.4g}</td>"
        f"<td>{html.escape(e.get('note', '')[:400])}</td></tr>"
        for k, e in metrics.items())
    figs_html = "".join(
        f"<h3>{html.escape(Path(f).stem)}</h3>"
        f"<img src='data:image/png;base64,{b64(f)}'>" for f in figs)
    body = f"""
<h1>Cross-season repeat line: 2012 anchor vs 2014/2016/2018 repeats</h1>
<p class="note">{html.escape(notes)}</p>
{figs_html}
<h2>Cross-flight difference matrix (* = headline 2012 pair)</h2>
<table><tr><th>pair</th><th>bed-surf delta meas (dB)</th><th>sim</th>
<th>captured</th><th>profile-diff r</th><th>rms (dB)</th>
<th>surf delta meas/sim/r^-2 (UNCAL)</th></tr>{prow}</table>
<h2>Metrics</h2>
<table><tr><th>metric</th><th>value</th><th>note</th></tr>{mrows}</table>
<h2>Configuration</h2>
<pre>{html.escape(json.dumps(config, indent=1))}</pre>
"""
    (out / "report.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{CASE}</title><style>{css}</style></head>"
        f"<body>{body}</body></html>")
    print(f"wrote {out / 'report.html'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-traces", type=int, default=100)
    ap.add_argument("--firn", type=int, default=FIRN_N,
                    help="firn layer count (0 = off)")
    ap.add_argument("--smooth-surface", action="store_true",
                    help="disable the validated --surf-rough default")
    ap.add_argument("--att", type=float, default=ATT_EFF,
                    help="effective one-way ice attenuation for the bed runs "
                    "(dB/km; ATT_EFF provenance note)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(out_root=args.out, n_traces=args.n_traces, firn_n=args.firn or None,
        surf_rough=not args.smooth_surface, force=args.force, att=args.att)


if __name__ == "__main__":
    main()
