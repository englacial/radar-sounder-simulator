"""Cross-year measured depth-power profiles at the B26 firn core.

A MEASUREMENT-SIDE reality check for the ~17 dB mid-band (20-70 m) deficit
of the simulated firn return in tools/run_b26_comparison.py: five OPR flight
lines pass within 1 km of B26 (77.2533 N, 49.2167 W; ngt37C95.2), spanning
2011-2019 and three radar/platform generations. If the measured ~-20 dB
plateau (dB rel own surface peak) is reproduced across years and instruments,
the target level is a property of the firn, not of one frame's processing.

Method: exactly run_b26_comparison's nadir profile, reused BY IMPORT
(sub_frame -> closest-approach trace; surface_peak_twtt -> per-trace surface
peak; profile_vs_depth -> 5 m boxcar, depth = (twtt - t_surf)*c/2/sqrt(
EPS_MEAN), dB rel that peak; band_levels; profile_corr over 5-200 m).

Deviations from the 2019 comparison, all forced by the data and recorded in
metrics.json: (1) the other four lines pass 356-929 m from the borehole, so
their closest trace samples a nearby -- not the same -- firn column; (2) the
product fast-time posting varies (16.7 ns in 2019, 20-33.4 ns earlier), so
profile_vs_depth's 5 m boxcar floors at 3 bins and the older frames are
effectively smoothed over ~9 m and range-resolution-limited; (3) the frame
datasets expose NO waveform parameters (see WAVEFORM_NOTE) -- dt/n_samples
are recorded as the only instrument-grid proxy available without parsing the
source .mat param structs.

No simulations are run: the firn_N40 reference curve is read from the cached
outputs/b26_comparison/runs/ fields.

Run: uv run python tools/run_b26_overflights.py
"""

import datetime
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_b26_comparison as b26  # noqa: E402  profile method, reused as-is
import run_firn_investigation as rfi  # noqa: E402  EPS_MEAN / core depth range

C = 299792458.0
OUT = ROOT / "outputs" / "b26_overflights"
SIM_RUNS = ROOT / "outputs" / "b26_comparison" / "runs"
SIM_CFG = ROOT / "outputs" / "b26_comparison" / "run_config.json"
SIM_RUN = "firn_N40"          # the headline sim run of the b26 comparison
NOTES = ROOT / "claude_notes" / "firn_core_flightlines.md"

# (season, frame_id, platform) -- the five lines within 1 km of B26
# (claude_notes/firn_core_flightlines.md); reference = the 2019 frame.
FRAMES = [
    ("2019_Greenland_P3", "20190418_01_009", "P-3"),
    ("2017_Greenland_P3", "20170328_01_055", "P-3"),
    ("2015_Greenland_C130", "20150515_03_021", "C-130"),
    ("2014_Greenland_P3", "20140508_01_061", "P-3"),
    ("2011_Greenland_P3", "20110506_01_026", "P-3"),
]
REF = ("2019_Greenland_P3", "20190418_01_009")
PRODUCTS = ("CSARP_standard", "CSARP_qlook")
ALONG_M = 2000.0              # only used to locate the closest-approach trace
MID_BAND, DEEP_BAND = "20-70m", "80-120m"
# 150-200 m: below the core and far above the bed here -- a per-frame NOISE
# FLOOR proxy, i.e. the dynamic range each product's processing delivers.
FLOOR_BAND = "150-200m"
BANDS_EXTRA = b26.EXTRA_BANDS + ((150.0, 200.0),)
WAVEFORM_NOTE = ("the cached frame datasets expose only geospatial/product "
                 "attrs -- no f0/bandwidth/pulse length; product dt and "
                 "sample count are recorded as the instrument-grid proxy "
                 "(parsing the source .mat param structs is out of scope "
                 "here; the 2019 values live in "
                 "outputs/cache/mcords_2019P3_params.json)")
# Okabe-Ito, assigned in fixed year order (colour follows the year, never
# its rank in the plot).
COLORS = {"2019": "#000000", "2017": "#0072B2", "2015": "#D55E00",
          "2014": "#009E73", "2011": "#CC79A7"}


def frame_profile(frame):
    """(depth_m, dB rel own surface peak, info) at the B26 closest-approach
    trace -- run_b26_comparison's measured-profile path, unchanged."""
    _, sinfo = b26.sub_frame(frame, ALONG_M)
    i = sinfo["i0_global"]
    tw = np.asarray(frame.twtt.values, np.float64)
    dt = float(tw[1] - tw[0])
    data = np.asarray(frame.Data.values[i], np.float64)
    t_s = b26.surface_peak_twtt(data[None], tw,
                                np.array([float(frame.Surface.values[i])]),
                                dt)[0]
    depth, db = b26.profile_vs_depth(data, tw, t_s, dt)
    bin_depth = C * dt / (2.0 * np.sqrt(rfi.EPS_MEAN))
    info = {
        "closest_approach_m": round(sinfo["closest_m"], 1),
        "trace_index": int(i), "n_traces": int(frame.sizes["slow_time"]),
        "n_samples": int(len(tw)), "dt_ns": round(dt * 1e9, 4),
        "t0_us": round(float(tw[0]) * 1e6, 4),
        "depth_bin_m": round(bin_depth, 3),
        "smooth_bins": int(max(int(round(5.0 / bin_depth)) | 1, 3)),
        "effective_smoothing_m": round(
            max(int(round(5.0 / bin_depth)) | 1, 3) * bin_depth, 2),
        "surface_peak_twtt_us": round(float(t_s) * 1e6, 4),
    }
    return depth, db, info


def load_frames():
    """{(season, frame_id): {...}} with per-product profiles; frames/products
    that fail to load are recorded with the reason instead of raising."""
    out = {}
    for season, fid, platform in FRAMES:
        rec = {"season": season, "frame_id": fid, "platform": platform,
               "year": fid[:4], "products": {}, "errors": {}}
        for prod in PRODUCTS:
            try:
                frame = b26.load_frame(season, fid, data_product=prod)
            except Exception as e:
                rec["errors"][prod] = f"{type(e).__name__}: {e}"
                print(f"  [skip] {fid} {prod}: {type(e).__name__}: {e}",
                      flush=True)
                continue
            depth, db, info = frame_profile(frame)
            rec["products"][prod] = {
                **info, "bands": b26.band_levels(depth, db, extra=BANDS_EXTRA),
                "attrs_waveform_fields": sorted(
                    k for k in frame.attrs
                    if any(s in k.lower() for s in
                           ("freq", "band", "wf", "wave", "chirp", "radar"))),
                "_profile": (depth, db)}
            print(f"  [ok] {fid} {prod}: {info['closest_approach_m']:.0f} m, "
                  f"dt {info['dt_ns']} ns, "
                  f"{MID_BAND} {b26.band_levels(depth, db)[MID_BAND]:+.1f} dB",
                  flush=True)
        out[(season, fid)] = rec
    return out


def sim_profile():
    """(depth, dB, meta) of the cached firn_N40 total field at the B26
    closest-approach trace -- the b26 comparison's own profile recipe, read
    from outputs/b26_comparison/runs/ (no simulation). None if unavailable."""
    try:
        wide = np.load(SIM_RUNS / "wide_surface_bed.npz")
        firn = np.load(SIM_RUNS / f"{SIM_RUN}.npz")
        cfg = json.loads(SIM_CFG.read_text())
    except Exception as e:
        print(f"  [warn] sim reference unavailable: {type(e).__name__}: {e}")
        return None
    j0 = int(cfg["closest_trace"]["sim_index"])
    tw = wide["twtt"]
    dt = float(tw[1] - tw[0])
    # total field = wide (surface + bed) + the firn run's INTERNAL layers
    # (layer 0 excluded: it is the firn run's own surface, already in wide)
    p = np.abs(wide["field"].sum(-1)[j0] + firn["field"][j0, :, 1:].sum(-1))**2
    t_s = b26.surface_peak_twtt(p[None], tw,
                                np.array([wide["nadir_twtt"][j0, 0]]), dt)[0]
    depth, db = b26.profile_vs_depth(p, tw, t_s, dt)
    bands = b26.band_levels(depth, db, extra=BANDS_EXTRA)
    rec_b = cfg["band_levels_db_rel_surface"].get(SIM_RUN, {})
    meta = {"run": SIM_RUN, "source": str(SIM_RUNS.relative_to(ROOT)),
            "bands": bands,
            "reproduces_run_config_bands": bool(
                all(abs(bands[k] - rec_b[k]) < 1e-9 for k in rec_b)),
            "note": "cached field, surface-peak guess taken from the run's "
                    "own nadir_twtt (the comparison used the frame Surface "
                    "pick; same local maximum -- band levels reproduce "
                    "run_config.json exactly)"}
    return depth, db, meta


def figure(recs, sim, path):
    """Two panels, same axes and same colour-per-year: the SAR-focused
    CSARP_standard product (what the b26 comparison quotes) and the unfocused
    CSARP_qlook product (the like-for-like processing for the sims)."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for ax, prod in zip(axs, PRODUCTS):
        ax.axvspan(20, 70, color="0.9", zorder=0)
        ax.text(45, 1.5, "mid-band 20-70 m", ha="center", fontsize=8,
                color="0.35")
        for (season, fid), r in recs.items():
            p = r["products"].get(prod)
            if p is None:
                lbl = f"{r['year']} {r['platform']}: no {prod}"
                ax.plot([], [], color=COLORS[r["year"]], lw=1.3, ls=":",
                        label=lbl)
                continue
            d, db = p["_profile"]
            m = (d >= -5) & (d <= b26.PROFILE_MAX_M)
            ref = (season, fid) == REF
            ax.plot(d[m], db[m], color=COLORS[r["year"]],
                    lw=2.2 if ref else 1.3, zorder=5 if ref else 3,
                    label=f"{r['year']} {r['platform']} "
                          f"({p['closest_approach_m']:.0f} m from B26)"
                          + (" - comparison frame" if ref else ""))
        if sim is not None:
            d, db, _ = sim
            m = (d >= -5) & (d <= b26.PROFILE_MAX_M)
            ax.plot(d[m], db[m], color="#56B4E9", lw=2.0, ls="--", zorder=4,
                    label=f"simulated {SIM_RUN} (2019 frame)")
        ax.axvline(rfi.ZMAX, color="k", ls=":", lw=1.0)
        ax.text(rfi.ZMAX + 1.5, -72, f"B26 core ends ({rfi.ZMAX:.1f} m)",
                fontsize=8, rotation=90, va="bottom")
        ax.set_xlim(-5, b26.PROFILE_MAX_M)
        ax.set_ylim(-75, 3)
        ax.set_xlabel("depth below surface peak (m; c/sqrt(eps_mean))")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.set_title(f"{prod}" + (" (f-k SAR + multilook)"
                                  if prod == "CSARP_standard"
                                  else " (unfocused: pulse compression"
                                       " + presums)"))
    axs[0].set_ylabel("power (dB rel own surface peak, 5 m smoothed)")
    fig.suptitle("B26 overflights: measured nadir depth-power at each frame's "
                 "closest approach to the borehole")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading frames ...", flush=True)
    recs = load_frames()
    sim = sim_profile()

    ref = recs[REF]["products"].get("CSARP_standard")
    if ref is None:
        raise RuntimeError("reference 2019 CSARP_standard profile missing")
    others = {f"{s}/{f}:{p}": v["_profile"]
              for (s, f), r in recs.items()
              for p, v in r["products"].items()}
    if sim is not None:
        others[f"sim:{SIM_RUN}"] = sim[:2]
    corr = b26.profile_corr(ref["_profile"], others, lo=5.0,
                            hi=b26.PROFILE_MAX_M)

    frames_out = []
    for (season, fid), r in recs.items():
        prods = {}
        for p, v in r["products"].items():
            prods[p] = {k: val for k, val in v.items() if k != "_profile"}
            prods[p]["corr_vs_2019_standard_5_200m"] = round(
                corr[f"{season}/{fid}:{p}"], 4)
        std = prods.get("CSARP_standard", {})
        frames_out.append({
            "season": season, "frame_id": fid, "year": r["year"],
            "platform": r["platform"],
            "instrument": f"MCoRDS on {r['platform']} ({season})",
            "closest_approach_m": (std.get("closest_approach_m")
                                   if std else None),
            "products_available": sorted(r["products"]),
            "products_missing": r["errors"],
            "mid_band_db": (round(std["bands"][MID_BAND], 2) if std else None),
            "deep_band_db": (round(std["bands"][DEEP_BAND], 2) if std else None),
            "floor_band_db": (round(std["bands"][FLOOR_BAND], 2) if std
                              else None),
            "mid_band_above_floor_db": (
                round(std["bands"][MID_BAND] - std["bands"][FLOOR_BAND], 2)
                if std else None),
            "mid_band_db_qlook": (
                round(prods["CSARP_qlook"]["bands"][MID_BAND], 2)
                if "CSARP_qlook" in prods else None),
            "corr_vs_2019_standard": std.get("corr_vs_2019_standard_5_200m"),
            "waveform_params": WAVEFORM_NOTE,
            "per_product": prods,
        })

    mids = [f["mid_band_db"] for f in frames_out if f["mid_band_db"] is not None]
    deeps = [f["deep_band_db"] for f in frames_out
             if f["deep_band_db"] is not None]
    mids_q = [f["mid_band_db_qlook"] for f in frames_out
              if f["mid_band_db_qlook"] is not None]

    def _stats(v):
        return {"values": v, "median": round(float(np.median(v)), 2),
                "spread_db": round(float(max(v) - min(v)), 2),
                "std_db": round(float(np.std(v)), 2)}

    summary = {
        "n_frames_loaded": len(mids),
        "mid_band_20_70m_db": _stats(mids),
        "mid_band_20_70m_db_qlook": {
            **_stats(mids_q), "n": len(mids_q),
            "note": "UNFOCUSED product: no per-season SAR/multilook gain "
                    "asymmetry between the surface peak and the volume, so "
                    "this is the tighter cross-instrument estimate of the "
                    "measured plateau"},
        "deep_band_80_120m_db": _stats(deeps),
        "floor_band_150_200m_db": _stats(
            [f["floor_band_db"] for f in frames_out
             if f["floor_band_db"] is not None]),
        "mid_band_above_floor_db": _stats(
            [f["mid_band_above_floor_db"] for f in frames_out
             if f["mid_band_above_floor_db"] is not None]),
        "sim_reference": None if sim is None else {
            **sim[2],
            "mid_band_db": round(sim[2]["bands"][MID_BAND], 2),
            "deep_band_db": round(sim[2]["bands"][DEEP_BAND], 2),
            "gap_vs_measured_median_db": round(
                sim[2]["bands"][MID_BAND] - float(np.median(mids)), 2),
            "gap_vs_measured_qlook_median_db": round(
                sim[2]["bands"][MID_BAND] - float(np.median(mids_q)), 2)},
    }
    doc = {"case": "b26_overflights", "group": "xOPR clutter",
           "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "core": {"name": "ngt37C95.2 (B26)", "lat_lon": b26.B26_LATLON},
           "method": ("run_b26_comparison's nadir depth-power profile "
                      "(imported): closest-approach trace, own surface peak, "
                      "dB rel that peak, 5 m boxcar, depth via "
                      f"c/sqrt(eps_mean={rfi.EPS_MEAN:.4f}); Pearson r over "
                      f"5-{b26.PROFILE_MAX_M:.0f} m vs the 2019 CSARP_standard "
                      "profile"),
           "caveats": [
               "the four non-2019 lines pass 356-929 m from the borehole: a "
               "nearby firn column, not the same one",
               "product fast-time posting varies (16.7-33.4 ns); the 5 m "
               "boxcar floors at 3 bins, so the older frames are effectively "
               "smoothed over ~9 m and are range-resolution-limited",
               "absolute product gain differs between products/seasons; every "
               "profile is normalized to its OWN surface peak, which cancels "
               "it but also makes the level a SURFACE-RELATIVE ratio -- a "
               "season with a different surface-return calibration shifts the "
               "whole curve",
               WAVEFORM_NOTE],
           "summary": summary, "frames": frames_out}
    (OUT / "metrics.json").write_text(json.dumps(doc, indent=1) + "\n")
    figure(recs, sim, OUT / "overflight_profiles.png")
    print(json.dumps(summary, indent=1))
    print(f"wrote {OUT}/metrics.json, {OUT}/overflight_profiles.png")
    return doc


if __name__ == "__main__":
    main()
