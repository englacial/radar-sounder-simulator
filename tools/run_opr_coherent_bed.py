"""Coherent surface + BedMachine-bed xOPR clutter cases at MCoRDS-like
processing level (M18 structure, M24 upgrade).

For each cached OPR frame we run the COHERENT multilayer kernel on a two-media
ice stack -- air / ice (eps_r 3.17, attenuation) / bed (eps_r 8) -- with the
surface from the PGC 32 m mosaic (cached) and the bed from BedMachine (cached,
geoid-corrected inside fetch_bedmachine_window). The DEM is subdivided to a
single facet spacing meeting the Fresnel-zone criterion from the surface's
nadir r_min (spacing = beta * sqrt(lambda*r_min), beta=0.5 in check_facet_size).

M24: the simulation now carries the MCoRDS 3 (2017 P-3) instrument model,
parameters sourced from the frames' own param_records/param_combine structs and
the CReSIS rds readme (provenance: outputs/cache/mcords_2017P3_params.json):

  * chirp waveform: 180-210 MHz (f0 195 MHz, B 30 MHz), hann-weighted pulse
    compression (CReSIS ft_wind = hanning; the 20% transmit Tukey is an
    unmodeled second-order shape effect), pulse length = the frame's longest
    (bed) waveform (Antarctica: 3 us; Greenland: 10 us -- the 1 us surface
    waveform shares the compressed shape in our windowed-sinc model);
  * antenna: uniform unsteered 7-element 0.5-lambda cross-track array
    (P-3 center array, readme platform table), roll_source="nav" (frame Roll);
    the recorded tx amplitude taper / per-frame rx channel subsets / hanning
    array window are documented, unmodeled approximations;
  * ALIAS-FREE dt (M21 caveat): the frame grid dt = 33.333 ns puts the
    envelope-quantization alias at |f0 - round(f0*dt)/dt| = 15 MHz = B/2
    exactly (fragile hann band edge; simulate() warns). We simulate at
    dt/OVERSAMPLE = 8.333 ns (alias at 45 MHz = 3B/2, warning silent,
    measured in claude_notes/m24_alias_probe.py) and decimate [::OVERSAMPLE]
    exactly back onto the frame twtt grid (t0 = 0). interp_bins stays off --
    no multilayer kernel change needed (plan option (a)).

Along-track processing (the honesty decision, recorded): the measured
CSARP_standard product is motion-compensated f-k SAR (2.5 m SLC spacing) +
11-look hanning multilook, decimated to ~15 m posting. Our full-frame run
subsamples ~100 traces (spacing ~hundreds of m >> lambda/4), so COHERENT
along-track summation of those traces would be Doppler-aliased and physically
meaningless; the full frame is therefore compared at chirp+antenna level
(per-trace, no simulated along-track processing) with an incoherent
N_LOOKS_SIM-trace multilook as the speckle-statistics analog (NOT
resolution-matched: sim looks are ~hundreds of m apart vs 27.5 m measured).
A DENSE sub-segment (interpolated nav at DENSE_SPACING = 0.35 m < lambda/4 =
0.384 m, Doppler-unaliased for ALL scattering angles) demonstrates
properly-sampled UNFOCUSED processing (processing.unfocused_sar, ~20 m
aperture = the unfocused limit sqrt(lambda*r/2) at 500 m AGL), with the
coherent surface gain recorded. The dense scene reuses (crops) the cached
full-frame DEM/bed windows -- no extra network.

Deliverable per frame (case dir outputs/verification/opr_<frame_id>_coherent_bed/,
group "xOPR clutter"):
  * figure (a): measured radargram | simulated (chirp+antenna, frame grid) |
    median-profile panel (measured vs sim vs sim multilook);
  * figure (b): per-layer split; figure (c): surface speckle panel;
  * figure (d): dense sub-segment, raw vs unfocused-processed;
  * metrics.json: original gates (surface_leading_edge, bed_alignment,
    floor-aware) UNCHANGED in spirit, plus recorded M24 metrics
    (alias_free_dt, clutter_to_surface_db bands, speckle_contrast_multilooked,
    unfocused_surface_gain_db) -- record-first per repo convention.

Honesty (carried over + M24): (1) at 32 m DEM posting the surface phase is not
lambda-accurate -> speckle/envelope statistics, not deterministic phase; (2) at
these facets the coherent LPA is specular-dominated; (3) the bed is
input-limited (input_bed_error_floor_bins); (4) the processing levels are
matched only approximately -- no motion compensation, no focused SAR, no
waveform playlist/receiver-gain stitching in the sim.

Run: uv run python tools/run_opr_coherent_bed.py
"""

import datetime
import json
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from pyproj import Transformer
from scipy.ndimage import uniform_filter1d

from soundersim.compare.brute_force import _contributions, flat_rectangle_samples
from soundersim.config import (AntennaConfig, DemInterface, FacetConfig,
                               Medium, RadarConfig, SimConfig, WaveformConfig)
from soundersim.kernels.coherent import lpa_contributions
from soundersim.opr import (fetch_bedmachine_window, fill_nodata_nearest,
                            frame_scene, load_bottom_pick, load_frame,
                            resample_to_grid)
from soundersim.output import combine
from soundersim.physics import fresnel_normal
from soundersim.processing import multilook, unfocused_sar
from soundersim.simulate import simulate
from soundersim.synthetic import MultilayerScene

# Reuse the frame list, dB helper and output root from the stage-1 tool (DRY).
from run_opr_comparison import CASES, OUT_ROOT, _db  # noqa: E402

C = 299792458.0
# --- MCoRDS 3 / 2017 P-3 instrument model (outputs/cache/mcords_2017P3_params.json)
F0 = 195e6                 # 180-210 MHz chirp center (frame param_records)
B_CHIRP = 30e6             # bandwidth (f1 - f0)
CHIRP_WINDOW = "hann"      # CReSIS pulse-compression ft_wind = hanning
PULSE_LEN = {              # longest (bed) waveform per frame's param_records
    "20171121_03_005": 3e-6,   # Antarctica 2017 P3: Tpd in {1, 3} us
    "20170422_01_014": 10e-6,  # Greenland 2017 P3: Tpd in {1, 3, 10} us
}
N_ELEMENTS = 7             # P-3 center cross-track array (readme table)
SPACING_LAM = 0.5          # element spacing in carrier wavelengths (readme)
OVERSAMPLE = 4             # dt_sim = dt_frame/4 = 8.333 ns -> alias 45 MHz
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
# Along-track processing treatment (see module docstring).
N_LOOKS_SIM = 5            # incoherent multilook analog on the coarse traces
DENSE_TRACES = 220         # dense sub-segment trace count
DENSE_SPACING = 0.35       # m; < lambda/4 = 0.384 m at 195 MHz
DENSE_CT = 1500.0          # dense sub-segment cross-track reach (m)
APERTURE_M = 20.0          # unfocused aperture ~ sqrt(lambda*r/2) at 500 m AGL
# Clutter-to-surface dynamic-range bands (us after the per-trace Surface pick,
# clipped 1 us above the Bottom pick).
DR_BANDS = {"near": (1.0, 2.5), "mid": (2.5, 5.0)}


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


def _radar_config(frame, frame_id):
    """(rc_sim, rc_frame): the alias-free simulation grid (dt/OVERSAMPLE, same
    t0, window-covering n_samples) and the frame's native grid. Simulated
    traces decimate [::OVERSAMPLE] exactly onto the frame twtt axis."""
    tw = frame.twtt.values
    dt = float((tw[-1] - tw[0]) / (len(tw) - 1))
    t0, n_samples = float(tw[0]), len(tw)
    wf = WaveformConfig(kind="chirp", bandwidth=B_CHIRP,
                        pulse_length=PULSE_LEN[frame_id], window=CHIRP_WINDOW)
    ant = AntennaConfig(kind="array", n_elements=N_ELEMENTS,
                        spacing_lam=SPACING_LAM, roll_source="nav")
    rc_sim = RadarConfig(dt=dt / OVERSAMPLE,
                         n_samples=OVERSAMPLE * (n_samples - 1) + 1,
                         t0=t0, f0=F0, waveform=wf, antenna=ant)
    rc_frame = RadarConfig(dt=dt, n_samples=n_samples, t0=t0, f0=F0)
    return rc_sim, rc_frame


def _bed_scene(frame, n_traces, ct_dist):
    """Frame surface + BedMachine bed on the same 32 m grid -> MultilayerScene.

    Reuses the CACHED PGC surface DEM window and the CACHED BedMachine window
    (fetch_* are cache-first; no refetch). aux carries rc (simulation grid)
    and rc_frame (native frame grid).
    """
    rc, rc_frame = _radar_config(frame, frame.attrs.get("frame_id"))

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
    mscene.nav_roll = scene.nav_roll  # antenna roll_source="nav"
    aux = {"idx": idx, "rc": rc, "rc_frame": rc_frame, "ct_dist": ct_dist,
           "bed_meta": meta, "bed_fill": bed_fill, "clamp_frac": clamp_frac,
           "surf_fill": info["fill_fraction"], "media": media}
    return mscene, aux


def _dense_scene(frame, mscene, n_dense, spacing_m, ct_dist):
    """Dense sub-segment scene: nav interpolated to ``spacing_m`` along-track
    around the frame center, DEM/bed CROPPED from the already-built full-frame
    MultilayerScene (cache-only, no refetch). Returns (scene, info)."""
    lat = np.asarray(frame.Latitude.values, np.float64)
    lon = np.asarray(frame.Longitude.values, np.float64)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    elev = np.asarray(frame.Elevation.values, np.float64)
    roll = (np.asarray(frame.Roll.values, np.float64)
            if "Roll" in frame else np.zeros(len(lat)))
    fwd = Transformer.from_crs("EPSG:4326", mscene.crs, always_xy=True)
    inv = Transformer.from_crs(mscene.crs, "EPSG:4326", always_xy=True)
    px, py = fwd.transform(lon, lat)
    seg = np.hypot(np.diff(px), np.diff(py))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s_mid = s[len(s) // 2]
    st = s_mid + (np.arange(n_dense) - (n_dense - 1) / 2.0) * spacing_m
    xd = np.interp(st, s, px)
    yd = np.interp(st, s, py)
    zd = np.interp(st, s, elev)
    rd = np.interp(st, s, np.nan_to_num(roll, nan=0.0))
    lond, latd = inv.transform(xd, yd)
    nav = np.column_stack([latd, lond, zd])

    # Crop the full-frame grids around the dense track + reach.
    tr = mscene.transform
    pad = ct_dist + 500.0
    corners_x = np.array([xd.min() - pad, xd.max() + pad])
    corners_y = np.array([yd.min() - pad, yd.max() + pad])
    cols, rows = (~tr) * (corners_x, corners_y)
    ny, nx = mscene.dem.shape
    c0 = int(np.clip(np.floor(min(cols)), 0, nx - 2))
    c1 = int(np.clip(np.ceil(max(cols)) + 1, c0 + 2, nx))
    r0 = int(np.clip(np.floor(min(rows)), 0, ny - 2))
    r1 = int(np.clip(np.ceil(max(rows)) + 1, r0 + 2, ny))
    dems = [np.ascontiguousarray(d[r0:r1, c0:c1]) for d in mscene.dems]
    tr_c = tr * Affine.translation(c0, r0)
    scene = MultilayerScene(
        mscene.name + "_dense", dems, tr_c, mscene.crs, nav, mscene.media,
        {**mscene.params, "dense_spacing_m": spacing_m,
         "dense_n_traces": n_dense, "dense_ct_dist": ct_dist},
        grid_origin=tuple(
            np.asarray(getattr(mscene, "grid_origin", (0, 0))) + (r0, c0)))
    scene.nav_roll = rd
    info = {"track_len_m": float(st[-1] - st[0]),
            "crop_shape": dems[0].shape}
    return scene, info


def _band_drs(power, surf, bot, t0, dt):
    """Per-band clutter-to-surface dynamic range: median over traces of
    10*log10(median band power / surface peak), band twtt windows relative to
    the per-trace Surface pick (DR_BANDS), clipped 1 us above the Bottom pick.
    ``power`` is (traces, bins) on the frame grid."""
    out = {}
    n = power.shape[1]
    for name, (lo_us, hi_us) in DR_BANDS.items():
        vals = []
        for t in range(power.shape[0]):
            if not (np.isfinite(surf[t]) and np.isfinite(bot[t])):
                continue
            lo = surf[t] + lo_us * 1e-6
            hi = min(surf[t] + hi_us * 1e-6, bot[t] - 1.0e-6)
            if hi <= lo:
                continue
            b0 = int(np.clip((lo - t0) / dt, 0, n - 1))
            b1 = int(np.clip((hi - t0) / dt, b0 + 1, n))
            p0 = int(np.clip((surf[t] - 0.5e-6 - t0) / dt, 0, n - 1))
            p1 = int(np.clip((surf[t] + 1.0e-6 - t0) / dt, p0 + 1, n))
            pk = power[t, p0:p1].max()
            band = power[t, b0:b1]
            band = band[band > 0]
            if band.size and pk > 0:
                vals.append(10.0 * np.log10(np.median(band) / pk))
        out[name] = float(np.median(vals)) if vals else float("nan")
    return out


def _sliding_contrast(aligned, win=11):
    """Median std/mean of intensity over ``win``-trace sliding slow-time
    windows, per fast-time row of a surface-following band. Returns
    (contrast, residual intensities / local mean)."""
    if aligned.shape[0] < win + 2:
        return float("nan"), np.array([])
    mu = uniform_filter1d(aligned, win, axis=0, mode="nearest")
    mu2 = uniform_filter1d(aligned ** 2, win, axis=0, mode="nearest")
    var = np.maximum(mu2 - mu ** 2, 0.0)
    h = win // 2
    valid = mu[h:-h] > 0
    con = np.sqrt(var[h:-h])[valid] / mu[h:-h][valid]
    resid = (aligned[h:-h] / np.maximum(mu[h:-h], 1e-300))[valid]
    return float(np.median(con)), resid


def run_case(case, n_traces=N_TRACES_CB, ct_dist=CT_DIST_CB, out_root=OUT_ROOT,
             spacing=None, dense_traces=DENSE_TRACES,
             dense_spacing=DENSE_SPACING, dense_ct=DENSE_CT,
             aperture_m=APERTURE_M):
    frame = load_frame(case["season"], case["frame_id"])
    mscene, aux = _bed_scene(frame, n_traces, ct_dist)
    rc_sim, rc, idx = aux["rc"], aux["rc_frame"], aux["idx"]
    dt, t0, n_samples = rc.dt, rc.t0, rc.n_samples  # FRAME grid (gate math)
    lam = rc_sim.wavelength
    k = 2.0 * np.pi / lam
    gamma_surf = fresnel_normal(1.0, EPS_ICE)
    f_alias_native = abs(F0 - round(F0 * dt) / dt)
    f_alias_sim = abs(F0 - round(F0 * rc_sim.dt) / rc_sim.dt)

    # Subdivision from the surface Fresnel criterion (same formula as the old
    # coherent tool); r_min = min platform->surface nadir range from the pick.
    r_min = float(np.nanmin(frame.Surface.values)) * C / 2.0
    if spacing is None:
        spacing = BETA * np.sqrt(lam * r_min)
    lpa_err = _lpa_nadir_error(spacing, r_min, k, gamma_surf)

    cfg = SimConfig(mode="coherent", split_sides=False, radar=rc_sim,
                    facets=FacetConfig(spacing=spacing), media=aux["media"],
                    interfaces=[DemInterface(name="surface"),
                                DemInterface(name="bed")])

    # Capture warnings from simulate(): the per-interface facet-size warnings
    # (records whether the single spacing satisfies both Fresnel criteria) AND
    # the alias warning, which must be SILENT at the alias-free dt.
    t_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        ds_fine = simulate(mscene, cfg)
    t_wall = time.perf_counter() - t_start
    facet_warnings = [str(w.message) for w in wlist
                      if "facet edge" in str(w.message)]
    alias_warned = any("alias" in str(w.message) for w in wlist)

    # Decimate exactly onto the frame twtt grid (t0 = 0; every OVERSAMPLE-th
    # simulation bin IS a frame bin, no interpolation).
    ds = ds_fine.isel(twtt=slice(None, None, OVERSAMPLE))
    assert ds.sizes["twtt"] == n_samples

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

    # --- Speckle contrast on the coherent surface return (chirped, per-trace,
    # and after the N_LOOKS_SIM multilook analog) ---
    band0, win = 3, 11
    keep = np.where(hc)[0]
    bandw = int(min(60, (n_samples - (le_c[keep] + band0)).min()))
    aligned = np.stack([p_surf[t, le_c[t] + band0: le_c[t] + band0 + bandw]
                        for t in keep])
    speckle_contrast, resid = _sliding_contrast(aligned, win)
    n_ml = max(1, min(N_LOOKS_SIM, aligned.shape[0] // 3))
    n_blk = aligned.shape[0] // n_ml
    aligned_ml = aligned[: n_blk * n_ml].reshape(n_blk, n_ml, bandw).mean(1)
    speckle_ml, _ = _sliding_contrast(aligned_ml,
                                      win=min(win, max(3, n_blk - 2)))
    # Measured-frame contrast in the matched surface-following band (the
    # CSARP_standard product already carries 11 hanning looks; displayed traces
    # are ~independent, so the same estimator applies, terrain texture caveat).
    meas_p = np.asarray(frame.Data.values[idx], np.float64)
    ok_m = np.isfinite(surf_bin)
    le_m = np.clip(np.round(surf_bin[ok_m]).astype(int), 0,
                   n_samples - band0 - bandw - 1)
    aligned_m = np.stack([meas_p[t, b + band0: b + band0 + bandw]
                          for t, b in zip(np.where(ok_m)[0], le_m)])
    speckle_meas, _ = _sliding_contrast(aligned_m, win)

    # Multilooked Dataset for the record/figure (fixed-grid, processing layer;
    # the metric above uses the surface-following band -- see notes).
    ds_ml = multilook(ds, n_ml)

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

    # --- Peak-normalized dynamic range: measured vs simulated clutter-to-
    # surface ratio in twtt bands relative to the picks (frame grid) ---
    comb = np.asarray(combine(ds, "layer"), np.float64)  # |field.sum(layer)|^2
    # The sim knows nothing of the system delay: evaluate its bands against
    # offset-shifted picks so both products are surface-referenced.
    dr_meas = _band_drs(meas_p, surf_pick, bot_pick, t0, dt)
    dr_sim = _band_drs(comb, surf_pick + off_s, bot_pick + off_b, t0, dt)
    dr_diff = {b: dr_sim[b] - dr_meas[b] for b in DR_BANDS}

    # --- Recorded energy / dropped diagnostics ---
    p_bed = np.asarray(ds.power.sel(layer="bed"), np.float64)
    e_ratio_db = float(_db(p_bed.sum()) - _db(p_surf.sum()))
    pk_ratio_db = float(np.median(_db(p_bed.max(1)) - _db(p_surf.max(1))))
    drop = ds.dropped_power.values  # (slow_time, layer); kernel-level
    p_fine = np.asarray(ds_fine.power, np.float64)  # (T, twtt_fine, layer)
    tot = p_fine.sum(1) + drop
    drop_frac = (drop.sum(0) / np.maximum(tot.sum(0), 1e-300)).tolist()

    # --- Dense sub-segment: properly-sampled unfocused processing demo ---
    dscene, dinfo = _dense_scene(frame, mscene, dense_traces, dense_spacing,
                                 dense_ct)
    t_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as wl_d:
        warnings.simplefilter("always")
        ds_dense = simulate(dscene, cfg)
        ds_unf = unfocused_sar(ds_dense, aperture_m=aperture_m)
    t_dense = time.perf_counter() - t_start
    doppler_warned = any("alias in Doppler" in str(w.message) for w in wl_d)
    alias_warned_d = any("quantization alias" in str(w.message) for w in wl_d)
    step = json.loads(ds_unf.attrs["processing"])[-1]
    n_win = int(step["n_traces"])
    pk_raw = np.asarray(ds_dense.power.sel(layer="surface"), np.float64).max(1)
    pk_unf = np.asarray(ds_unf.power.sel(layer="surface"), np.float64).max(1)
    unf_gain_db = float(_db(np.median(pk_unf)) - _db(np.median(pk_raw)))

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
            "heavy tail, as in stage 1). M24: leading edge measured on the "
            "CHIRPED trace (hann main lobe 1.44/B ~ 1.4 frame bins; symmetric "
            "kernel, shift absorbed by the offset)"},
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
        "alias_free_dt": {
            "value": rc_sim.dt * 1e9, "threshold": None, "pass": True,
            "op": "record", "oversample": OVERSAMPLE,
            "f_alias_sim_mhz": f_alias_sim / 1e6,
            "f_alias_native_mhz": f_alias_native / 1e6,
            "alias_warning_fired": bool(alias_warned or alias_warned_d),
            "note": "recorded: simulation fast-time grid (ns). The frame's "
            "native dt (33.333 ns) puts the envelope-quantization alias at "
            "15 MHz = B/2 exactly (fragile hann band edge; simulate() warns); "
            "simulating at dt/4 moves it to 45 MHz = 3B/2 (warning silent, "
            "measured in claude_notes/m24_alias_probe.py: native-dt quiet-band "
            "floor +0.2 dB median / +3 dB p90 vs alias-free) and decimation "
            "[::4] lands exactly on the frame grid. interp_bins stays off; no "
            "multilayer kernel change (plan option (a))"},
        "clutter_to_surface_db": {
            "value": dr_diff["near"], "threshold": None, "pass": True,
            "op": "record",
            "measured_near_db": dr_meas["near"], "sim_near_db": dr_sim["near"],
            "measured_mid_db": dr_meas["mid"], "sim_mid_db": dr_sim["mid"],
            "diff_mid_db": dr_diff["mid"],
            "bands_us_after_surface": json.dumps(DR_BANDS),
            "note": "recorded only: sim-minus-measured clutter-to-surface "
            "dynamic range (median over traces of median band power rel the "
            "per-trace surface peak, dB) in surface-referenced twtt bands "
            "clipped 1 us above the Bottom pick. Value = 'near' band diff. "
            "The sim carries surface+bed clutter only (no volume scatter, "
            "internal layers, receiver noise floor, or waveform-playlist gain "
            "stitching), so bands where the measured product is noise- or "
            "layer-dominated will show large negative diffs. " + rec},
        "speckle_contrast": {
            "value": speckle_contrast, "threshold": None, "pass": True,
            "op": "~1", "band_bins": bandw, "window_traces": win,
            "note": "median std/mean of coherent surface-return intensity over "
            f"{win}-trace sliding slow-time windows, per fast-time row of a "
            f"surface-following band (leading edge +{band0}..+{band0+bandw} "
            "bins); ~1 for fully developed (exponential) speckle, <1 where a "
            "deterministic specular component remains (Rician). " + rec},
        "speckle_contrast_multilooked": {
            "value": speckle_ml, "threshold": None, "pass": True, "op": "record",
            "n_looks": n_ml, "theory_contrast": 1.0 / float(np.sqrt(n_ml)),
            "measured_frame_contrast": speckle_meas,
            "note": "recorded only: same estimator after an incoherent "
            f"{n_ml}-look average of the surface-following band -- the honest "
            "coarse-spacing analog of CSARP_standard's 11-look multilook "
            "(NOT resolution-matched: sim looks are ~hundreds of m apart vs "
            "27.5 m measured). Theory 1/sqrt(n) for fully developed speckle. "
            "measured_frame_contrast = same estimator on the measured product "
            "(11 hanning looks + terrain texture) in the matched band"},
        "unfocused_surface_gain_db": {
            "value": unf_gain_db, "threshold": None, "pass": True,
            "op": "record", "n_window_traces": n_win,
            "coherent_gain_db": float(20.0 * np.log10(n_win)),
            "incoherent_gain_db": float(10.0 * np.log10(n_win)),
            "aperture_m": aperture_m, "trace_spacing_m": dense_spacing,
            "n_dense_traces": dense_traces,
            "doppler_guard_warned": bool(doppler_warned),
            "note": "recorded only: median surface-layer peak-power gain of "
            "unfocused_sar over the raw dense traces (dense sub-segment, nav "
            f"interpolated to {dense_spacing} m trace spacing; < lambda/4 = "
            f"{lam/4:.3f} m makes the summation Doppler-unaliased for ALL "
            "scattering angles; processing.py guard silent). Coherent "
            "(specular) reference 20*log10(n), incoherent (speckle) "
            "10*log10(n): partial coherence of the rough-surface return lands "
            "between"},
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
            "path) power fraction of the bed layer (kernel-level, "
            "pre-convolution)"},
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
        f"surface+bed cluttergram at MCoRDS-like processing level (M24). "
        f"INSTRUMENT MODEL (provenance outputs/cache/mcords_2017P3_params.json: "
        f"frame param_records/param_combine + CReSIS rds readme): chirp "
        f"{F0/1e6:.0f} MHz center / {B_CHIRP/1e6:.0f} MHz bandwidth "
        f"(180-210 MHz), pulse {PULSE_LEN[case['frame_id']]*1e6:.0f} us (the "
        f"frame's longest/bed waveform; the 1 us surface waveform shares the "
        f"compressed shape in our model), hann compression window (CReSIS "
        f"ft_wind=hanning; 20% transmit Tukey unmodeled); antenna = uniform "
        f"unsteered {N_ELEMENTS}-element {SPACING_LAM}-lambda cross-track "
        f"array (P-3 center array), roll from frame nav; recorded-but-"
        f"unmodeled: tx amplitude taper, per-frame rx channel subset, hanning "
        f"array window. ALIAS-FREE GRID: simulated at dt/{OVERSAMPLE} = "
        f"{rc_sim.dt*1e9:.3f} ns (alias {f_alias_sim/1e6:.0f} MHz = 3B/2, "
        f"vs {f_alias_native/1e6:.0f} MHz = B/2 exactly at the native "
        f"33.333 ns; simulate() alias warning silent: {not alias_warned}), "
        f"decimated [::{OVERSAMPLE}] exactly onto the frame twtt grid. "
        f"PROCESSING LEVEL (asymmetry recorded): measured = motion-compensated "
        f"f-k SAR (2.5 m SLC) + 11-look hanning multilook -> ~15 m posting; "
        f"sim full frame = chirp+antenna PER TRACE at {len(idx)}-trace "
        f"subsampling (spacing >> lambda/4: coherent along-track summation "
        f"would be Doppler-aliased, so none is applied) + {n_ml}-look "
        f"incoherent multilook analog for speckle statistics; a DENSE "
        f"sub-segment ({dense_traces} traces at {dense_spacing} m < lambda/4, "
        f"{dinfo['track_len_m']:.0f} m of track, cropped cached DEM windows, "
        f"wall {t_dense:.1f} s) demonstrates properly-sampled UNFOCUSED "
        f"processing: {aperture_m:.0f} m aperture ({n_win} traces), surface "
        f"peak gain {unf_gain_db:.1f} dB (coherent ref "
        f"{20*np.log10(n_win):.1f}, incoherent {10*np.log10(n_win):.1f}), "
        f"Doppler guard silent: {not doppler_warned}. Media air / ice (eps_r "
        f"{EPS_ICE}, {ATT_DB_PER_KM:.0f} dB/km one-way constant attenuation, "
        f"warm-marginal value; timing gate is attenuation-free) / bed (eps_r "
        f"{EPS_BED}, recorded only). Surface {mscene.params['dem_product']}; "
        f"bed {meta['product']} {meta['version']} at native "
        f"{meta['posting_m']:.0f} m, bilinearly oversampled to the 32 m grid "
        f"-- BedMachine's effective resolution caps off-nadir bed-clutter "
        f"realism, so bed TIMING fidelity is the claim, not texture. Geoid: "
        f"product bed is EIGEN-6C4 geoid-referenced (per file metadata), "
        f"converted to WGS84-ellipsoidal as bed + geoid. Subdivision: single "
        f"spacing {spacing:.1f} m = beta {BETA} * sqrt(lambda*r_min) with "
        f"surface r_min {r_min:.0f} m; the bed's criterion uses the in-ice "
        f"wavelength ({lam_ice:.2f} m) at a larger effective range and lands "
        f"at a similar limit, so one spacing serves both.{warn_txt} Native "
        f"32 m grid {n_native} cells -> subdivided {n_facets} facets "
        f"({n_facets/max(n_native,1):.1f}x). Estimated per-facet nadir LPA "
        f"envelope error {lpa_err*100:.0f}% (worst case). Runtime trims: "
        f"ct_dist {ct_dist:.0f} m, {len(idx)} traces; bed clamped to "
        f"surface-0.1 m on {aux['clamp_frac']*100:.1f}% of cells; nodata fill "
        f"fractions surface {aux['surf_fill']:.4f}, bed {aux['bed_fill']:.4f}"
        f". Bottom pick coverage {coverage*100:.1f}% of simulated traces. "
        f"Input bed-model floor: median {in_med:.1f} bins (p90 {in_p90:.1f}), "
        f"BedMachine-vs-pick thickness corr {in_corr:.2f}. Wall time main "
        f"{t_wall:.1f} s + dense {t_dense:.1f} s. HONESTY: (1) at 32 m DEM "
        f"posting surface phase is not lambda-accurate, so the coherent "
        f"product is meaningful for speckle/envelope statistics but not "
        f"deterministic phase; (2) at these (necessarily large) facets the "
        f"coherent LPA is specular-dominated (bright leading edge, little "
        f"diffuse off-nadir clutter, cf. Gerekos 2023); (3) the bed is "
        f"INPUT-LIMITED (input_bed_error_floor_bins); (4) processing levels "
        f"are matched only approximately (no motion comp, no focused SAR, no "
        f"waveform-playlist gain stitching in the sim)."
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

    _figures(out, case, frame, idx, ds, ds_ml, surf_pick, bot_pick, off_s,
             off_b, resid, speckle_contrast, speckle_ml)
    _dense_figure(out, case, ds_dense, ds_unf, aperture_m, n_win, unf_gain_db,
                  dense_spacing)
    print(f"{case['frame_id']}_coherent_bed: {t_wall:.1f} s (+dense "
          f"{t_dense:.1f} s) | spacing {spacing:.1f}m {n_facets} facets "
          f"LPA~{lpa_err*100:.0f}% | surf med {le_med:.1f} bins (pass "
          f"{le_pass}) | bed med {bed_med:.1f} vs floor {in_med:.1f} thr "
          f"{thr_eff:.1f} (pass {bed_pass}) | speckle {speckle_contrast:.2f} "
          f"-> ml({n_ml}) {speckle_ml:.2f} (meas {speckle_meas:.2f}) | C/S "
          f"near sim {dr_sim['near']:.1f} meas {dr_meas['near']:.1f} dB | "
          f"unfoc gain {unf_gain_db:.1f} dB (n={n_win}) | alias warn "
          f"{alias_warned} | bed/surf {e_ratio_db:+.1f} dB | dropped bed "
          f"{drop_frac[1]*100:.1f}%")
    return metrics, out


def _figures(out, case, frame, idx, ds, ds_ml, surf_pick, bot_pick, off_s,
             off_b, resid, speckle_contrast, speckle_ml):
    tw_us = ds.twtt.values * 1e6
    meas = _db(frame.Data.values[idx])
    comb = _db(np.asarray(combine(ds, "layer"), np.float64))  # |field.sum|^2
    # multilook drops the field (phase destroyed): layers add in POWER there.
    comb_ml = _db(np.asarray(ds_ml.power.sum("layer"), np.float64))
    p_surf_db = _db(np.asarray(ds.power.sel(layer="surface"), np.float64))
    p_bed_db = _db(np.asarray(ds.power.sel(layer="bed"), np.float64))
    x = np.arange(len(idx))
    ext = [0, len(idx), tw_us[-1], tw_us[0]]

    def _vmax(img):
        fin = img[np.isfinite(img) & (img > -290)]
        return np.percentile(fin, 99.5)

    # (a) measured | simulated (chirp + antenna, matched grid) | median-profile
    # panel (measured vs sim vs sim multilook, peak-normalized dB).
    fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharey=True,
                            gridspec_kw={"width_ratios": [5, 5, 3]})
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
    axs[1].set_title("simulated: chirp + 7-el array (dB)")
    for ax in axs[:2]:
        ax.set_xlabel("trace (subsampled)")
        ax.legend(loc="lower right", fontsize=8)
    axs[0].set_ylabel("twtt (us)")
    prof_m = _db(np.nanmedian(10 ** (meas / 10.0), axis=0))
    prof_c = _db(np.nanmedian(10 ** (comb / 10.0), axis=0))
    prof_l = _db(np.nanmedian(10 ** (comb_ml / 10.0), axis=0))
    axs[2].plot(prof_m - np.nanmax(prof_m), tw_us, "k", lw=1.0,
                label="measured")
    axs[2].plot(prof_c - np.nanmax(prof_c), tw_us, "tab:blue", lw=1.0,
                label="sim")
    axs[2].plot(prof_l - np.nanmax(prof_l), tw_us, "tab:orange", lw=1.0,
                label="sim multilook")
    axs[2].set_xlim(-90, 3)
    axs[2].set_xlabel("median profile (dB rel peak)")
    axs[2].grid(alpha=0.3)
    axs[2].legend(loc="lower left", fontsize=8)
    axs[2].set_title("median power profile")
    fig.suptitle(f"{case['frame_id']}: measured vs simulated coherent "
                 f"surface+bed clutter (MCoRDS-like level)")
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
    fig.suptitle(f"{case['frame_id']}: coherent per-layer split "
                 f"(chirp + array)")
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
                 f"contrast std/mean = {speckle_contrast:.2f} "
                 f"(multilooked {speckle_ml:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "speckle.png", dpi=130)
    plt.close(fig)


def _dense_figure(out, case, ds_dense, ds_unf, aperture_m, n_win, gain_db,
                  spacing_m):
    """(d) dense sub-segment: raw traces vs unfocused-processed, combined dB."""
    tw_us = ds_dense.twtt.values * 1e6
    raw = _db(np.asarray(combine(ds_dense, "layer"), np.float64))
    prc = _db(np.asarray(combine(ds_unf, "layer"), np.float64))
    # Crop fast time to the populated part (dense reach is small).
    fin = np.isfinite(raw) & (raw > -290)
    rows = np.where(fin.any(axis=0))[0]
    b0 = int(max(rows[0] - 20, 0))
    b1 = int(min(rows[-1] + 20, raw.shape[1]))
    vmax = np.percentile(raw[fin], 99.5)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, img, ttl, vm in [
            (axs[0], raw, "raw dense traces (chirp+antenna)", vmax),
            (axs[1], prc,
             f"unfocused SAR ({aperture_m:.0f} m aperture, {n_win} traces)",
             vmax + gain_db)]:
        ax.imshow(img[:, b0:b1].T, aspect="auto", cmap="gray",
                  extent=[0, img.shape[0], tw_us[b1 - 1], tw_us[b0]],
                  vmin=vm - 60, vmax=vm)
        ax.set_title(ttl)
        ax.set_xlabel("trace")
    axs[0].set_ylabel("twtt (us)")
    fig.suptitle(f"{case['frame_id']}: dense sub-segment ({spacing_m} m "
                 f"spacing < lambda/4), surface peak gain {gain_db:.1f} dB")
    fig.tight_layout()
    fig.savefig(out / "unfocused_dense.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    for case in CASES:
        run_case(case)
