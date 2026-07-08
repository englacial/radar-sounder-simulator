"""Coherent surface + BedMachine-bed xOPR clutter cases (M18, combined).

For each cached OPR frame we run the COHERENT multilayer kernel on a two-media
ice stack -- air / ice (eps_r 3.17, attenuation) / bed (eps_r 8) -- with the
surface from the PGC 32 m mosaic (cached) and the bed from BedMachine (cached,
geoid-corrected inside fetch_bedmachine_window). The DEM is subdivided to a
single facet spacing meeting the Fresnel-zone criterion from the surface's
nadir r_min (spacing = beta * sqrt(lambda*r_min), beta=0.5 in check_facet_size).
That one spacing serves both interfaces: the bed's criterion uses the in-ice
wavelength (lambda/sqrt(eps_ice)) at a LARGER effective range, so it lands at a
similar limit -- verified via the simulate() facet-size warnings, recorded in
the notes.

This one case supersedes the two former split cases (coherent surface-only +
incoherent surface+bed): it is coherent AND carries the bed.

Deliverable per frame (case dir outputs/verification/opr_<frame_id>_coherent_bed/,
group "xOPR clutter"):
  * figure (a): measured radargram (dB) | coherent COMBINED cluttergram
    (|field summed over layer|^2, dB) with the Surface AND Bottom picks overlaid
    (offset-adjusted on the sim panel);
  * figure (b): per-layer split -- coherent surface layer | coherent bed layer;
  * figure (c): surface-return speckle panel;
  * metrics.json:
      - surface_leading_edge: median |smoothed-coherent surface-LAYER leading
        edge - Surface pick| after offset removal (gate <= 5 bins);
      - bed_alignment: median |bed-LAYER nadir_twtt - Bottom pick| after offset
        removal, floor-aware vs input_bed_error_floor_bins (gate <= max(5,
        floor+5) bins), raw 5-bin verdict recorded as abs_5bin_pass;
      - input_bed_error_floor_bins, speckle_contrast, lpa_nadir_error,
        bed_surface_power_ratio_db, dropped_power_fraction -- recorded.

Honesty (from both former cases): at 32 m DEM posting the surface phase is not
lambda-accurate, so the coherent product is meaningful for speckle/envelope
statistics but not deterministic phase; at these (necessarily large) facets the
coherent LPA is specular-dominated (bright leading edge, little diffuse off-
nadir clutter, cf. Gerekos 2023). BedMachine's effective resolution caps off-
nadir bed-clutter realism -- bed TIMING fidelity is the claim, not texture; the
bed is input-limited (input_bed_error_floor_bins). Geoid EIGEN-6C4; ice
attenuation ATT_DB_PER_KM one-way constant (recorded, not fitted; timing gate is
attenuation-free).

Run: uv run python tools/run_opr_coherent_bed.py
"""

import datetime
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from scipy.ndimage import uniform_filter1d

from soundersim.compare.brute_force import _contributions, flat_rectangle_samples
from soundersim.config import (DemInterface, FacetConfig, Medium, RadarConfig,
                               SimConfig)
from soundersim.kernels.coherent import lpa_contributions
from soundersim.opr import (fetch_bedmachine_window, fill_nodata_nearest,
                            frame_scene, load_bottom_pick, load_frame,
                            resample_to_grid)
from soundersim.output import combine
from soundersim.physics import fresnel_normal
from soundersim.simulate import simulate
from soundersim.synthetic import MultilayerScene

# Reuse the frame list, dB helper and output root from the stage-1 tool (DRY).
from run_opr_comparison import CASES, OUT_ROOT, _db  # noqa: E402

C = 299792458.0
F0 = 195e6                 # MCoRDS band center; sets lambda ~ 1.54 m
EPS_ICE = 3.17
EPS_BED = 8.0              # nominal bedrock (matches run_opr_bed); recorded only
ATT_DB_PER_KM = 15.0      # one-way, constant (warm marginal ice; recorded)
BETA = 0.5                # check_facet_size beta -> L <= 0.5*sqrt(lam*r_min)
GATE_BINS = 5.0
# Runtime trims (coherent bed = refracted LPA over subdivided facets, the bed
# layer dominates): trace count + cross-track reach kept so each frame stays
# well under ~30 min from cache.
N_TRACES_CB = 100
CT_DIST_CB = 3000.0


def _lpa_nadir_error(L, r, k, gamma):
    """Envelope-normalized |LPA - brute force| for one flat L x L facet at nadir
    range r (worst-case LPA validity point)."""
    lam = 2.0 * np.pi / k
    p = np.array([0.0, 0.0, r])
    pts, nrm, dA = flat_rectangle_samples(L, L, lam / 12.0)
    bf = _contributions(p, pts, nrm, dA, k, gamma)[0].sum()
    lp = lpa_contributions(
        p, np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), np.array([L * L]),
        np.array([[L, 0.0, 0.0]]), np.array([[0.0, L, 0.0]]), k, gamma, xp=np)[0][0]
    env = (k / (2.0 * np.pi)) * abs(gamma) * L * L / r ** 2
    return float(abs(lp - bf) / env)


def _leading_edge(power, drop_db=15.0):
    """Per-trace first fast-time bin exceeding (per-trace peak) - drop_db."""
    thr = power.max(axis=1, keepdims=True) * 10.0 ** (-drop_db / 10.0)
    over = power > thr
    has = over.any(axis=1)
    idx = np.where(has, over.argmax(axis=1), -1)
    return idx, has


def _bed_scene(frame, n_traces, ct_dist):
    """Frame surface + BedMachine bed on the same 32 m grid -> MultilayerScene.

    Reuses the CACHED PGC surface DEM window and the CACHED BedMachine window
    (fetch_* are cache-first; no refetch). rc carries f0 for the coherent run.
    """
    tw = frame.twtt.values
    dt = float((tw[-1] - tw[0]) / (len(tw) - 1))
    t0, n_samples = float(tw[0]), len(tw)
    rc = RadarConfig(dt=dt, n_samples=n_samples, t0=t0, f0=F0)

    scene, info = frame_scene(frame, n_traces=n_traces, ct_dist=ct_dist)
    idx = info["trace_idx"]

    lat = frame.Latitude.values[idx]
    lon = frame.Longitude.values[idx]
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    bounds = (lon.min(), lat.min(), lon.max(), lat.max())
    bed_native, tr_b, crs_b, meta = fetch_bedmachine_window(
        bounds, info["region"], pad_m=ct_dist + 500.0)
    bed = resample_to_grid(bed_native, tr_b, crs_b, scene.dem.shape,
                           scene.transform, scene.crs)
    bed, bed_fill = fill_nodata_nearest(bed)

    clamp = bed > scene.dem - 0.1
    clamp_frac = float(clamp.mean())
    bed = np.minimum(bed, scene.dem - 0.1).astype(np.float32)

    media = [Medium(name="air", eps_r=1.0),
             Medium(name="ice", eps_r=EPS_ICE,
                    attenuation_db_per_km=ATT_DB_PER_KM),
             Medium(name="bed", eps_r=EPS_BED)]
    mscene = MultilayerScene(scene.name + "_bed", [scene.dem, bed],
                             scene.transform, scene.crs, scene.nav_llh, media,
                             {**scene.params, "bed_product": meta["product"],
                              "bed_version": meta["version"]})
    aux = {"idx": idx, "rc": rc, "ct_dist": ct_dist, "bed_meta": meta,
           "bed_fill": bed_fill, "clamp_frac": clamp_frac,
           "surf_fill": info["fill_fraction"], "media": media}
    return mscene, aux


def run_case(case, n_traces=N_TRACES_CB, ct_dist=CT_DIST_CB, out_root=OUT_ROOT,
             spacing=None):
    frame = load_frame(case["season"], case["frame_id"])
    mscene, aux = _bed_scene(frame, n_traces, ct_dist)
    rc, idx = aux["rc"], aux["idx"]
    dt, t0, n_samples = rc.dt, rc.t0, rc.n_samples
    lam = rc.wavelength
    k = 2.0 * np.pi / lam
    gamma_surf = fresnel_normal(1.0, EPS_ICE)

    # Subdivision from the surface Fresnel criterion (same formula as the old
    # coherent tool); r_min = min platform->surface nadir range from the pick.
    r_min = float(np.nanmin(frame.Surface.values)) * C / 2.0
    if spacing is None:
        spacing = BETA * np.sqrt(lam * r_min)
    lpa_err = _lpa_nadir_error(spacing, r_min, k, gamma_surf)

    cfg = SimConfig(mode="coherent", split_sides=False, radar=rc,
                    facets=FacetConfig(spacing=spacing), media=aux["media"],
                    interfaces=[DemInterface(name="surface"),
                                DemInterface(name="bed")])

    # Capture the per-interface facet-size warnings from simulate() (records
    # whether the single spacing satisfies both interfaces' Fresnel criteria).
    t_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        ds = simulate(mscene, cfg)
    t_wall = time.perf_counter() - t_start
    facet_warnings = [str(w.message) for w in wlist
                      if "facet edge" in str(w.message)]

    range_bin = C * dt / 2.0

    # --- Surface gate: smoothed coherent SURFACE-LAYER leading edge vs pick ---
    p_surf = np.asarray(ds.power.sel(layer="surface"), np.float64)  # |field|^2
    w = max(1, int(round(spacing / range_bin)))
    sm_surf = uniform_filter1d(p_surf, w, axis=1, mode="nearest")
    le_c, hc = _leading_edge(sm_surf)
    surf_pick = frame.Surface.values[idx]
    surf_bin = (surf_pick - t0) / dt
    both = hc & np.isfinite(surf_bin)
    le_resid = le_c[both] - surf_bin[both]
    le_offset = float(np.median(le_resid))
    dle = np.abs(le_resid - le_offset)
    le_med = float(np.median(dle))
    le_p90 = float(np.percentile(dle, 90))
    le_max = float(dle.max())
    le_pass = bool(le_med <= GATE_BINS)

    # --- Speckle contrast on the coherent surface return ---
    band0, win = 3, 11
    keep = np.where(hc)[0]
    bandw = int(min(60, (n_samples - (le_c[keep] + band0)).min()))
    aligned = np.stack([p_surf[t, le_c[t] + band0: le_c[t] + band0 + bandw]
                        for t in keep])
    mu = uniform_filter1d(aligned, win, axis=0, mode="nearest")
    mu2 = uniform_filter1d(aligned ** 2, win, axis=0, mode="nearest")
    var = np.maximum(mu2 - mu ** 2, 0.0)
    h = win // 2
    valid = mu[h:-h] > 0
    cmap_ = np.sqrt(var[h:-h])[valid] / mu[h:-h][valid]
    speckle_contrast = float(np.median(cmap_))
    resid = (aligned[h:-h] / np.maximum(mu[h:-h], 1e-300))[valid]

    # --- Bed gate: bed-LAYER nadir twtt vs Bottom pick, floor-aware ---
    bot_pick = load_bottom_pick(frame)[idx]
    ok = np.isfinite(bot_pick) & np.isfinite(surf_pick)
    coverage = float(ok.mean())

    # Input floor: BedMachine ice thickness at nadir vs pick-derived thickness,
    # in bins (recomputed exactly as run_opr_bed).
    n_ice = np.sqrt(EPS_ICE)
    lon = np.where(frame.Longitude.values[idx] > 180.0,
                   frame.Longitude.values[idx] - 360.0,
                   frame.Longitude.values[idx])
    px, py = Transformer.from_crs("EPSG:4326", mscene.crs, always_xy=True
                                  ).transform(lon, frame.Latitude.values[idx])
    cols, rows = (~mscene.transform) * (px, py)
    r_i = np.clip(np.round(rows).astype(int), 0, mscene.dem.shape[0] - 1)
    c_i = np.clip(np.round(cols).astype(int), 0, mscene.dem.shape[1] - 1)
    thick_in = (mscene.dems[0] - mscene.dems[1])[r_i, c_i]
    thick_pk = (bot_pick - surf_pick) * rc.c / (2.0 * n_ice)
    d_in = (thick_in - thick_pk)[ok] * 2.0 * n_ice / rc.c / rc.dt
    d_in = d_in - np.median(d_in)
    in_med = float(np.median(np.abs(d_in)))
    in_p90 = float(np.percentile(np.abs(d_in), 90))
    in_corr = float(np.corrcoef(thick_in[ok], thick_pk[ok])[0, 1])

    nd_bed = ds.nadir_twtt.sel(layer="bed").values
    res = nd_bed[ok] - bot_pick[ok]
    off_b = float(np.median(res))
    rb = (res - off_b) / dt
    bed_med = float(np.median(np.abs(rb)))
    bed_p90 = float(np.percentile(np.abs(rb), 90))
    thr_eff = float(max(GATE_BINS, in_med + GATE_BINS))
    bed_pass = bool(bed_med <= thr_eff)
    # Surface nadir offset (figure overlay diagnostic).
    nd_surf = ds.nadir_twtt.sel(layer="surface").values
    oks = np.isfinite(surf_pick)
    off_s = float(np.median(nd_surf[oks] - surf_pick[oks]))

    # --- Recorded energy / dropped diagnostics ---
    p_bed = np.asarray(ds.power.sel(layer="bed"), np.float64)
    e_ratio_db = float(_db(p_bed.sum()) - _db(p_surf.sum()))
    pk_ratio_db = float(np.median(_db(p_bed.max(1)) - _db(p_surf.max(1))))
    drop = ds.dropped_power.values  # (slow_time, layer)
    tot = np.stack([p_surf.sum(1), p_bed.sum(1)], 1) + drop
    drop_frac = (drop.sum(0) / np.maximum(tot.sum(0), 1e-300)).tolist()

    # Subdivided facet count (surface grid; bed shares the grid).
    ny, nx = mscene.dem.shape
    f = 32.0 / spacing
    nrv = max(2, int(round((ny - 1) * f)) + 1)
    ncv = max(2, int(round((nx - 1) * f)) + 1)
    n_native = (ny - 1) * (nx - 1)
    n_facets = (nrv - 1) * (ncv - 1)
    lam_ice = lam / n_ice

    rec = ("recorded only; pass forced true (real-frame thresholds are set "
           "after observing residuals, per plan)")
    metrics = {
        "surface_leading_edge": {
            "value": le_med, "threshold": GATE_BINS, "pass": le_pass, "op": "<=",
            "p90_bins": le_p90, "max_abs_bins": le_max, "offset_bins": le_offset,
            "note": "median |smoothed-coherent surface-LAYER leading edge - "
            "frame Surface pick| in raw bins after removing the constant offset "
            "(recorded; absorbs system delay / DEM epoch). Median gate; p90/max "
            "recorded (tracker locks onto off-nadir clutter on rugged sections, "
            "heavy tail, as in stage 1)"},
        "bed_alignment": {
            "value": bed_med, "threshold": thr_eff, "op": "<=", "pass": bed_pass,
            "abs_5bin_pass": bool(bed_med <= GATE_BINS),
            "offset_s": off_b, "offset_bins": off_b / dt,
            "p90_abs_resid_bins": bed_p90,
            "max_abs_resid_bins": float(np.abs(rb).max()),
            "frac_within_5_bins": float(np.mean(np.abs(rb) <= GATE_BINS)),
            "pick_coverage": coverage, "n_gated_traces": int(ok.sum()),
            "surface_offset_bins": off_s / dt,
            "note": "median |bed-LAYER nadir_twtt - frame Bottom pick - median "
            f"offset| in bins, over traces with a pick. Threshold = max("
            f"{GATE_BINS:.0f}, input_bed_error_floor + {GATE_BINS:.0f}) bins: "
            "the input bed model caps achievable absolute timing; the raw 5-bin "
            "verdict is abs_5bin_pass. Median gate (heavy tracker tails on "
            "rugged terrain, p90/max recorded)"},
        "input_bed_error_floor_bins": {
            "value": in_med, "threshold": None, "pass": True, "op": "record",
            "p90_bins": in_p90, "thickness_corr": in_corr,
            "note": "recorded only: median |BedMachine-vs-pick ice thickness at "
            "nadir| in bins after offset removal -- the residual floor imposed "
            "by the INPUT bed model. bed_alignment cannot beat this"},
        "speckle_contrast": {
            "value": speckle_contrast, "threshold": None, "pass": True,
            "op": "~1", "band_bins": bandw, "window_traces": win,
            "note": "median std/mean of coherent surface-return intensity over "
            f"{win}-trace sliding slow-time windows, per fast-time row of a "
            f"surface-following band (leading edge +{band0}..+{band0+bandw} "
            "bins); ~1 for fully developed (exponential) speckle, <1 where a "
            "deterministic specular component remains (Rician). " + rec},
        "lpa_nadir_error": {
            "value": lpa_err, "threshold": None, "pass": True, "op": "record",
            "facet_size_m": float(spacing), "r_min_m": r_min,
            "note": "envelope-normalized single-facet LPA error at nadir (worst "
            "case); off-nadir clutter facets are sinc-suppressed and far more "
            "accurate. " + rec},
        "bed_surface_power_ratio_db": {
            "value": e_ratio_db, "threshold": None, "pass": True, "op": "record",
            "median_peak_ratio_db": pk_ratio_db,
            "attenuation_db_per_km_oneway": ATT_DB_PER_KM,
            "note": "recorded only: total bed/surface energy ratio (and median "
            "per-trace peak ratio) of the coherent per-layer |field|^2 under the "
            "constant attenuation assumption; incoherent-style, carries no "
            "target reflectivity (stage-1/simc convention)"},
        "dropped_power_fraction": {
            "value": drop_frac[1], "threshold": None, "pass": True,
            "op": "record", "surface_layer": drop_frac[0],
            "note": "recorded only: dropped (out-of-window + invalid refracted "
            "path) power fraction of the bed layer"},
    }

    out = out_root / f"opr_{case['frame_id']}_coherent_bed"
    out.mkdir(parents=True, exist_ok=True)
    meta = aux["bed_meta"]
    warn_txt = (f" simulate() facet-size warnings: {facet_warnings}"
                if facet_warnings else
                " simulate() emitted no facet-size warnings (single spacing "
                "satisfies both interfaces' Fresnel criteria).")
    notes = (
        f"{case['season']} {case['frame_id']} ({case['why']}); COHERENT "
        f"surface+bed cluttergram (supersedes the former coherent-surface and "
        f"incoherent-bed cases). f0 {F0/1e6:.0f} MHz (lambda {lam:.2f} m, "
        f"in-ice {lam_ice:.2f} m). Media air / ice (eps_r {EPS_ICE}, "
        f"{ATT_DB_PER_KM:.0f} dB/km one-way constant attenuation, warm-marginal "
        f"value; timing gate is attenuation-free) / bed (eps_r {EPS_BED}, "
        f"recorded only). Surface {mscene.params['dem_product']}; bed "
        f"{meta['product']} {meta['version']} at native {meta['posting_m']:.0f} "
        f"m, bilinearly oversampled to the 32 m grid -- BedMachine's effective "
        f"resolution caps off-nadir bed-clutter realism, so bed TIMING fidelity "
        f"is the claim, not texture. Geoid: product bed is EIGEN-6C4 geoid-"
        f"referenced (per file metadata), converted to WGS84-ellipsoidal as bed "
        f"+ geoid. Subdivision: single spacing {spacing:.1f} m = beta {BETA} * "
        f"sqrt(lambda*r_min) with surface r_min {r_min:.0f} m; the bed's "
        f"criterion uses the in-ice wavelength ({lam_ice:.2f} m) at a larger "
        f"effective range and lands at a similar limit, so one spacing serves "
        f"both.{warn_txt} Native 32 m grid {n_native} cells -> subdivided "
        f"{n_facets} facets ({n_facets/max(n_native,1):.1f}x). Estimated per-"
        f"facet nadir LPA envelope error {lpa_err*100:.0f}% (worst case). "
        f"Runtime trims: ct_dist {ct_dist:.0f} m, {len(idx)} traces; bed clamped "
        f"to surface-0.1 m on {aux['clamp_frac']*100:.1f}% of cells; nodata "
        f"fill fractions surface {aux['surf_fill']:.4f}, bed {aux['bed_fill']:.4f}"
        f". Bottom pick coverage {coverage*100:.1f}% of simulated traces. Input "
        f"bed-model floor: median {in_med:.1f} bins (p90 {in_p90:.1f}), "
        f"BedMachine-vs-pick thickness corr {in_corr:.2f}. Wall time "
        f"{t_wall:.1f} s. HONESTY: (1) at 32 m DEM posting surface phase is not "
        f"lambda-accurate, so the coherent product is meaningful for speckle/"
        f"envelope statistics but not deterministic phase; (2) at these "
        f"(necessarily large) facets the coherent LPA is specular-dominated "
        f"(bright leading edge, little diffuse off-nadir clutter, cf. Gerekos "
        f"2023); (3) the bed is INPUT-LIMITED (input_bed_error_floor_bins)."
        + ("" if bed_med <= GATE_BINS else
           f" ABSOLUTE 5-BIN BED GATE NOT MET AND INPUT-LIMITED: bed_alignment "
           f"{bed_med:.1f} bins vs input floor {in_med:.1f} bins; gate applied "
           f"as max(5, floor+5); simulator contribution beyond input "
           f"{bed_med - in_med:+.1f} bins."))
    (out / "metrics.json").write_text(json.dumps({
        "case": f"opr_{case['frame_id']}_coherent_bed",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metrics": metrics, "notes": notes, "group": "xOPR clutter",
    }, indent=1) + "\n")

    _figures(out, case, frame, idx, ds, surf_pick, bot_pick, off_s, off_b,
             resid, speckle_contrast)
    print(f"{case['frame_id']}_coherent_bed: {t_wall:.1f} s | spacing "
          f"{spacing:.1f}m {n_facets} facets LPA~{lpa_err*100:.0f}% | surf "
          f"med {le_med:.1f} bins (pass {le_pass}) | bed med {bed_med:.1f} vs "
          f"floor {in_med:.1f} thr {thr_eff:.1f} (pass {bed_pass}) | speckle "
          f"{speckle_contrast:.2f} | bed/surf {e_ratio_db:+.1f} dB | dropped "
          f"bed {drop_frac[1]*100:.1f}%")
    return metrics, out


def _figures(out, case, frame, idx, ds, surf_pick, bot_pick, off_s, off_b,
             resid, speckle_contrast):
    tw_us = ds.twtt.values * 1e6
    meas = _db(frame.Data.values[idx])
    comb = _db(np.asarray(combine(ds, "layer"), np.float64))  # |field.sum|^2
    p_surf_db = _db(np.asarray(ds.power.sel(layer="surface"), np.float64))
    p_bed_db = _db(np.asarray(ds.power.sel(layer="bed"), np.float64))
    x = np.arange(len(idx))
    ext = [0, len(idx), tw_us[-1], tw_us[0]]

    def _vmax(img):
        fin = img[np.isfinite(img) & (img > -290)]
        return np.percentile(fin, 99.5)

    # (a) measured | coherent combined cluttergram, both picks overlaid.
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    mv, cv = _vmax(meas), _vmax(comb)
    axs[0].imshow(meas.T, aspect="auto", extent=ext, cmap="gray",
                  vmin=mv - 60, vmax=mv)
    axs[0].plot(x, surf_pick * 1e6, "c", lw=0.7, label="Surface pick")
    axs[0].plot(x, bot_pick * 1e6, "r", lw=0.9, label="Bottom pick")
    axs[0].set_title("measured (CSARP_standard, dB)")
    axs[1].imshow(comb.T, aspect="auto", extent=ext, cmap="gray",
                  vmin=cv - 60, vmax=cv)
    axs[1].plot(x, (surf_pick + off_s) * 1e6, "c", lw=0.7,
                label="Surface pick + offset")
    axs[1].plot(x, (bot_pick + off_b) * 1e6, "r", lw=0.9,
                label="Bottom pick + offset")
    axs[1].set_title("coherent combined |field.sum(layer)|^2 (dB)")
    axs[0].set_ylabel("twtt (us)")
    for ax in axs:
        ax.set_xlabel("trace (subsampled)")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{case['frame_id']}: measured vs simulated coherent "
                 f"surface+bed clutter")
    fig.tight_layout()
    fig.savefig(out / "radargram_vs_coherent_bed.png", dpi=150)
    plt.close(fig)

    # (b) per-layer split: coherent surface layer | coherent bed layer.
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    sv = _vmax(p_surf_db)
    bfin = p_bed_db[np.isfinite(p_bed_db) & (p_bed_db > -290)]
    bv = np.percentile(bfin, 99.5) if bfin.size else sv
    axs[0].imshow(p_surf_db.T, aspect="auto", extent=ext, cmap="gray",
                  vmin=sv - 60, vmax=sv)
    axs[0].plot(x, (surf_pick + off_s) * 1e6, "c", lw=0.7,
                label="Surface pick + offset")
    axs[0].set_title("coherent surface layer |field|^2 (dB)")
    axs[1].imshow(p_bed_db.T, aspect="auto", extent=ext, cmap="gray",
                  vmin=bv - 60, vmax=bv)
    axs[1].plot(x, (bot_pick + off_b) * 1e6, "r", lw=0.9,
                label="Bottom pick + offset")
    axs[1].set_title("coherent bed layer |field|^2 (dB)")
    axs[0].set_ylabel("twtt (us)")
    for ax in axs:
        ax.set_xlabel("trace (subsampled)")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{case['frame_id']}: coherent per-layer split")
    fig.tight_layout()
    fig.savefig(out / "per_layer_split.png", dpi=150)
    plt.close(fig)

    # (c) speckle panel.
    fig, ax = plt.subplots(figsize=(7, 5))
    r = resid[resid < np.percentile(resid, 99.5)]
    ax.hist(r, bins=60, density=True, alpha=0.6, label="coherent (detrended)")
    xs = np.linspace(0, r.max(), 200)
    ax.plot(xs, np.exp(-xs), "r", lw=1.8, label="exponential (contrast 1)")
    ax.set_xlabel("intensity / local mean")
    ax.set_ylabel("pdf")
    ax.set_title(f"{case['frame_id']}: surface-return speckle\n"
                 f"contrast std/mean = {speckle_contrast:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "speckle.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    for case in CASES:
        run_case(case)
