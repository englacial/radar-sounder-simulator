"""Basal-clutter altitude triplet on the 2016 DC-8 anchor line.

Three real flights of the SAME 148.5 km grounded-ice line (claude_notes/
basal_clutter_scout.md) at 442 / 9150 / 10684 m AGL with IDENTICAL systems
(190 MHz / 50 MHz / hann / 20.202 ns): the measured radargrams show ~20 dB
more mid-column ("basal") clutter power at altitude. Each pass's common
segment is simulated COHERENT SURFACE+BED ONLY (REMA 32 m + BedMachine
500 m; NO firn, NO internal layers) at its real altitude/nav/params, and the
simulated clutter is DECOMPOSED per interface (the kernel returns per-layer
fields) into SURFACE-borne vs BED-borne energy -- the discriminator for what
the high-altitude clutter actually is.

Cross-track reach is the science-critical parameter and is DERIVED per pass:
for BOTH interfaces, off-nadir arrivals are covered out to the nadir-bed
delay plus MARGIN_US (surface: closed form; bed: Snell ray sweep with in-ice
refraction). No cap unless compute forces one (none applied; reaches are
recorded).

Scout pitfalls honored: the two high passes fly the line BACKWARDS (slices
reversed; nav roll NEGATED because the kernel derives the along-track axis
from trace order, so reversed nav flips u_at and roll must flip with it);
per-pass surface registration fitted (leading-edge gate; never shared);
BedMachine's 500 m posting means simulated basal clutter is systematically
smoother/weaker in fine texture than measured (recorded, not tuned away;
--picked-bed corrects the NADIR bed onto the anchor radar picks while
keeping BedMachine's cross-track relief -- see PICKED_BED_NOTE);
params from each pass's own cached param frame; identical 20.202 ns lattice
across passes (shared surface-referenced fast-time comparison).

Machinery reused from tools/run_altitude_comparison.py: param loading,
window mapping, alias-safe oversampling, REMA+BedMachine scene building,
cached runs, facet spacing, surface gate. Runs are chunked ~10 km along
track so the 50 km segment projects ~linearly from the 10 km pilot.

Run:  uv run python tools/run_basal_clutter.py                # 10 km pilot
      uv run python tools/run_basal_clutter.py --segment full # 50 km (STOP:
      report pilot timings first; full run only on explicit go-ahead)
      uv run python tools/run_basal_clutter.py --segment full --picked-bed
      uv run python tools/run_basal_clutter.py --segment full --picked-bed \
          --gamma-from-rssnr   # + required-surface-SNR-driven bed gamma
"""

import argparse
import base64
import datetime
import html
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_altitude_comparison as rac  # noqa: E402  shared machinery
from run_opr_comparison import _db  # noqa: E402

from soundersim.config import (AntennaConfig, DemInterface, FacetConfig,  # noqa: E402
                               Medium, RadarConfig, RoughnessConfig, SimConfig,
                               WaveformConfig)
from soundersim.opr import load_bottom_pick, load_frame  # noqa: E402
from soundersim.physics import fresnel_normal  # noqa: E402

C = 299792458.0
SEASON = "2016_Antarctica_DC8"
CASE_PREFIX = "basal_clutter"
OUT_DEFAULT = ROOT / "outputs" / "basal_clutter"
VER_ROOT = ROOT / "outputs" / "verification"

# Clutter coverage: off-nadir arrivals covered out to the nadir-bed delay
# plus MARGIN_US, for both interfaces (the measured clutter fills the column
# and hugs past the bed peak; scout: nadir bed at median 8.09 us below
# surface). The fast-time window extends slightly further (POST_BED_US).
MARGIN_US = 3.0
POST_BED_US = 3.5
PRE_SURF_US = 0.8
CHUNK_M = 10500.0          # along-track chunk target; pilot (10 km) = 1 chunk

# Analysis windows (twtt, relative to each dataset's OWN picks/geometry).
SURF_WIN_US = 0.8          # surface peak search half-width
MID_LO_US, MID_HI_US = 1.0, 0.5     # mid-column: surf+1.0 -> bed-0.5 us
BED_LO_US, BED_HI_US = 0.5, 1.5     # bed window:  bed-0.5 -> bed+1.5 us
SCOUT_LO_US, SCOUT_HI_US = 3.0, 0.6  # scout contrast: mean(bed-3..bed-0.6)
SCOUT_PK_US = 0.3                    # ... over peak within +-0.3 us of bed
# Measured noise floor: record end - [12, 8] us. Probed 2026-07-31: the
# last ~4 us of every record are processing-rolled-off (reads 15-25 dB
# below adjacent windows), and the PRE-surface region is unusable on the
# low pass (TX leakage / img_comb shallow zone reads ~26 dB ABOVE its
# mid-column); end-12..-8 sits >=26 us past the deepest bed on all passes
# and agrees with end-30..-25 to ~1 dB (low), so it is a floor estimate
# with at most a few dB of residual high-pass clutter tail (upper bound).
FLOOR_TAIL_LO_US, FLOOR_TAIL_HI_US = 12.0, 8.0

N_TRACES_PILOT = 48
N_TRACES_FULL = 240        # same ~210 m sim trace spacing as the pilot

# Pass table (claude_notes/basal_clutter_scout.md). Slices are half-open
# slow_time indices into each FULL frame; "rev" passes fly the line backwards
# (slices reversed to align with increasing anchor s). "full" parts are
# listed in increasing-s order after reversal. param_frame: cached
# mcords_params provenance (identical system within a segment).
PASSES = {
    "low": {
        "agl_med_m": 442.0, "rev": False, "param_frame": "20161105_05_005",
        "pilot": [("20161105_05_005", (2020, 2693))],
        "full": [("20161105_05_005", (1212, 3333)),
                 ("20161105_05_006", (0, 1244))]},
    "mid": {
        "agl_med_m": 9150.0, "rev": True, "param_frame": "20161028_05_006",
        "pilot": [("20161028_05_006", (858, 1532))],
        "full": [("20161028_05_006", (0, 2341)),
                 ("20161028_05_005", (2308, 3337))]},
    "high": {
        "agl_med_m": 10684.0, "rev": True, "param_frame": "20161031_07_005",
        "pilot": [("20161031_07_005", (337, 1011))],
        "full": [("20161031_07_005", (0, 1820)),
                 ("20161031_07_004", (1786, 3336))]},
}
ORDER = ["low", "mid", "high"]
S0_KM = {"pilot": 30.0, "full": 18.0}   # anchor s at segment start (display)

MEASURED_CAVEATS = (
    "Measured references are CSARP_standard. Scout pitfalls recorded: the "
    "low pass composites 1/3/10 us waveforms vs 3/10 us on the high passes "
    "(do not compare the first ~3 us below the surface across passes as one "
    "instrument); PRF differs (12000 vs 7500 Hz) though the posting does "
    "not; BedMachine's 500 m bed reproduces only ~55% of the radar-pick "
    "along-track bed roughness rms, so simulated basal clutter is expected "
    "systematically smoother and weaker in fine texture than measured.")


# ========================================================================
# cross-track reach derivation (the science-critical parameter)
# ========================================================================
def surface_reach(h, dt_below_surf):
    """Cross-track distance where a SURFACE scatterer's delay exceeds the
    nadir-surface delay by ``dt_below_surf`` (s), platform at ``h`` m AGL.
    Closed form: y = sqrt((h + c*dt/2)^2 - h^2)."""
    r = h + C * dt_below_surf / 2.0
    return float(np.sqrt(max(r * r - h * h, 0.0)))


def bed_reach(h, d, n_ice, dt_extra):
    """Cross-track distance where a BED scatterer's delay exceeds the
    nadir-bed delay by ``dt_extra`` (s): platform h m above the surface, bed
    d m of ice below it, Snell refraction at the (locally flat) surface.
    Sweeps the air incidence angle (each Snell ray IS the Fermat path to the
    bed point it hits), then inverts the monotone y(t_extra) relation."""
    theta = np.linspace(0.0, np.deg2rad(89.5), 4000)[1:]
    sin_i = np.sin(theta) / n_ice
    phi = np.arcsin(np.clip(sin_i, 0.0, 1.0 - 1e-12))
    y = h * np.tan(theta) + d * np.tan(phi)
    t = 2.0 * (h / np.cos(theta) + n_ice * d / np.cos(phi)) / C
    t_extra = t - 2.0 * (h + n_ice * d) / C
    if dt_extra >= t_extra[-1]:
        raise ValueError("bed_reach: sweep did not cover dt_extra")
    return float(np.interp(dt_extra, t_extra, y))


def derive_reach(h_max, dbs_max, d_min):
    """Per-pass reach doc: surface reach out to (max nadir-bed delay below
    surface + MARGIN_US) and bed reach out to (nadir bed + MARGIN_US), both
    at the pass's max AGL (worst case: reach grows with h). ct = max of the
    two (the surface interface always binds: its target delay includes the
    whole ice column)."""
    m = MARGIN_US * 1e-6
    r_surf = surface_reach(h_max, dbs_max + m)
    r_bed = bed_reach(h_max, d_min, float(np.sqrt(rac.EPS_ICE)), m)
    return {"ct_m": max(r_surf, r_bed), "surface_reach_m": r_surf,
            "bed_reach_m": r_bed, "h_max_m": h_max,
            "bed_delay_max_us": dbs_max * 1e6, "d_min_m": d_min,
            "margin_us": MARGIN_US, "capped": False}


# ========================================================================
# picked-bed correction (--picked-bed): radar bed picks as an along-track
# residual on BedMachine
# ========================================================================
# ONE reference pass supplies the picks for ALL THREE simulations: per-pass
# beds would make the three scenes different and confound the altitude
# comparison with a scene change. The reference is the LOW pass
# (20161105_05_005-007, 442 m AGL) because its picks are the cleanest of the
# triplet -- scout registration table: 2.45 m surface-pick scatter (sigma)
# vs 10.80 / 10.92 m for mid / high, p5..p95 spread 7.7 m vs ~30 m -- and at
# 442 m the bed echo sits ~20 dB above the mid-column clutter (measured
# midcol/bed-peak -36.7 dB) whereas at altitude off-nadir arrivals crowd the
# bed to within a few dB (-17.7 / -16.1 dB), so the high passes' picks are
# both noisier and more likely to have followed a clutter arc. It is also
# the anchor line's own flight, i.e. the axis everything is registered to.
REF_PASS = "low"
REF_FRAMES = ("20161105_05_005", "20161105_05_006", "20161105_05_007")
ROUGH_WIN_M = 5000.0        # scout's along-track bed-roughness detrend window
PBED_TAG = "_pbed"          # output/cache suffix; BedMachine runs stay cached
PICKED_BED_NOTE = (
    "bed = BedMachine + resid(s), resid(s) = picked_bed(s) - BedMachine at "
    "nadir(s) on the anchor along-track axis, picks from the LOW pass only "
    "(20161105_05_005-007) and applied IDENTICALLY to all three passes. The "
    "nadir bed therefore matches the radar picks exactly while BedMachine's "
    "CROSS-TRACK structure -- the relief that actually drives off-nadir "
    "clutter -- is preserved; extending the 1-D picks cross-track as a "
    "constant would have erased it. Pick gaps fall back to zero residual "
    "(pure BedMachine). Caveat: the residual is constant along the "
    "cross-track normal, so along-track pick detail is replicated as "
    "cross-track ridges out to +-ct (an unavoidable consequence of "
    "correcting a 2-D DEM with a 1-D profile); the fast-time grid, reaches "
    "and facet spacings are left at their BedMachine-run values so the two "
    "runs are directly comparable.")


def case_tag(picked_bed, gamma_rssnr=False):
    return ((PBED_TAG if picked_bed else "")
            + (GRSSNR_TAG if gamma_rssnr else ""))


def ref_bed_picks():
    """Radar-picked bed elevation along the anchor line (the reference LOW
    pass), on the anchor along-track axis (EPSG:3031, s=0 at _005 trace 0).

    Elevation convention is the one the tool's registration fits already use
    (run_altitude_comparison): ellipsoidal ice surface = Elevation -
    c*Surface/2, ice thickness = (Bottom - Surface)*c/(2*sqrt(EPS_ICE)), bed
    = surface - thickness, with the same rac.EPS_ICE and the same
    WGS84-ellipsoidal datum as the REMA + BedMachine scene stack (no geoid
    term), so the residual against BedMachine is datum-consistent. Pick gaps
    stay NaN."""
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    xs, ys, beds = [], [], []
    for fid in REF_FRAMES:
        frame = _retry(f"ref frame {fid}", lambda f=fid: load_frame(SEASON, f))
        lat, lon = rac._lonlat(frame)
        surf = np.asarray(frame.Surface.values, np.float64)
        elev = np.asarray(frame.Elevation.values, np.float64)
        bot = _retry(f"ref picks {fid}", lambda f=frame: load_bottom_pick(f))
        x, y = tr.transform(lon, lat)
        xs.append(x)
        ys.append(y)
        beds.append(elev - surf * C / 2.0
                    - (bot - surf) * C / (2.0 * np.sqrt(rac.EPS_ICE)))
    x, y = np.concatenate(xs), np.concatenate(ys)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    bed = np.concatenate(beds)
    return {"pass": REF_PASS, "frames": list(REF_FRAMES), "x": x, "y": y,
            "s": s, "bed": bed, "eps_ice": rac.EPS_ICE,
            "frame_len": [int(len(b)) for b in beds],
            "n": int(len(s)), "line_len_km": round(float(s[-1]) / 1e3, 2),
            "gap_frac_line": round(float((~np.isfinite(bed)).mean()), 5)}


def project_to_track(px, py, tx, ty, s_ref):
    """Along-track coordinate of map points (px, py) on the polyline sampled
    at (tx, ty) with along-track coordinate s_ref: nearest sample plus its
    tangential offset (exact for a straight track; the anchor line is smooth
    at its 14.85 m posting)."""
    ux, uy = np.gradient(tx), np.gradient(ty)
    nrm = np.hypot(ux, uy)
    ux, uy = ux / nrm, uy / nrm
    _, i = cKDTree(np.column_stack([tx, ty])).query(
        np.column_stack([np.asarray(px), np.asarray(py)]))
    return s_ref[i] + (px - tx[i]) * ux[i] + (py - ty[i]) * uy[i]


def roughness_rms(s, z, win_m=ROUGH_WIN_M):
    """rms of z about a running mean of width win_m -- the scout's along-track
    bed roughness metric (BedMachine 33.3 m vs radar picks 60.5 m over the
    50 km segment). NaNs are linearly interpolated first."""
    ok = np.isfinite(z)
    z = np.interp(s, s[ok], z[ok])
    n = max(3, int(round(win_m / float(np.median(np.diff(s))))))
    return float(np.sqrt(np.mean(
        (z - ndimage.uniform_filter1d(z, n, mode="nearest")) ** 2)))


def sample_dem(dem, transform, px, py):
    """Bilinear sample of a map-referenced grid at (px, py), edge-clamped."""
    cols, rows = (~transform) * (np.asarray(px), np.asarray(py))
    return ndimage.map_coordinates(np.asarray(dem, np.float64),
                                   [rows - 0.5, cols - 0.5], order=1,
                                   mode="nearest")


def apply_picked_bed(base, ref):
    """Rewrite the base scene's bed DEM in place as BedMachine + the anchor
    -line pick residual (PICKED_BED_NOTE). Returns the recorded stats."""
    dem, bed = base.dems[0], np.asarray(base.dems[1], np.float64)
    tr = Transformer.from_crs("EPSG:3031", base.crs, always_xy=True)
    rx, ry = tr.transform(ref["x"], ref["y"])
    ny, nx = bed.shape
    xa, ya = base.transform * (0.0, 0.0)
    xb, yb = base.transform * (float(nx), float(ny))
    keep = ((rx >= min(xa, xb)) & (rx <= max(xa, xb))
            & (ry >= min(ya, yb)) & (ry <= max(ya, yb)))
    kk = np.where(keep)[0]
    if len(kk) < 100 or not (np.diff(kk) == 1).all():
        raise RuntimeError("picked-bed: anchor picks do not cover the scene "
                           "contiguously")
    rx, ry, s_ref = rx[kk], ry[kk], ref["s"][kk]
    pick = ref["bed"][kk]
    bm = sample_dem(bed, base.transform, rx, ry)
    gap = ~np.isfinite(pick)
    resid = np.where(gap, 0.0, pick - bm)

    cols, rows = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
    px, py = base.transform * (cols.ravel(), rows.ravel())
    s_pix = project_to_track(px, py, rx, ry, s_ref)
    bed_new = bed + np.interp(s_pix, s_ref, resid).reshape(bed.shape)
    clamp = float((bed_new > dem - 0.1).mean())
    base.dems[1] = np.minimum(bed_new, dem - 0.1).astype(np.float32)
    base.params["bed_correction"] = PICKED_BED_NOTE

    # stats over the simulated traces' own along-track span
    tr4 = Transformer.from_crs("EPSG:4326", base.crs, always_xy=True)
    nx_, ny_ = tr4.transform(base.nav_llh[:, 1], base.nav_llh[:, 0])
    s_nav = project_to_track(nx_, ny_, rx, ry, s_ref)
    seg = (s_ref >= s_nav.min()) & (s_ref <= s_nav.max())
    r_seg = resid[seg]
    return {"reference_pass": ref["pass"], "reference_frames": ref["frames"],
            "eps_ice": ref["eps_ice"],
            "anchor_s_km": [round(float(s_nav.min()) / 1e3, 2),
                            round(float(s_nav.max()) / 1e3, 2)],
            "n_picks_segment": int(seg.sum()),
            "gap_frac_segment": round(float(gap[seg].mean()), 5),
            "residual_rms_m": round(float(np.sqrt(np.mean(r_seg ** 2))), 1),
            "residual_mean_m": round(float(r_seg.mean()), 1),
            "residual_absmax_m": round(float(np.abs(r_seg).max()), 1),
            "bed_roughness_rms_m": {
                "bedmachine": round(roughness_rms(s_ref[seg], bm[seg]), 1),
                "picked": round(roughness_rms(s_ref[seg], pick[seg]), 1),
                "scout_reference": {"bedmachine": 33.3, "radar_picks": 60.5}},
            "bed_clamp_frac_after": round(clamp, 6),
            "note": PICKED_BED_NOTE}


# ========================================================================
# RSSNR-driven bed reflectivity (--gamma-from-rssnr): required-surface-SNR
# along the anchor line -> per-facet bed gamma
# ========================================================================
# Dataset + mapping: claude_notes/required_snr_dataset.md. The store's main
# branch was mid-rebuild at scouting time, so the completed 5,646-frame
# version is PINNED by snapshot id. RSSNR removes exactly the differential
# geometric spreading the simulator re-applies (r_bed_eff = r_surf + H/n ==
# the kernel's refracted nadir spreading), so the mapping
#   |Gamma_bed|^2 dB = 2*A*H(s) - RSSNR(s) + K
# double-counts nothing; H(s) from the DATASET's own twtts (self-consistent
# with its RSSNR), A = the run's --att. K is MEDIAN-ANCHORED: the segment
# median |Gamma|^2 equals the constant run's Fresnel ice->bed value, so the
# dataset supplies along-track RELATIVE structure while the absolute level
# stays continuous with the constant-gamma results (RSSNR is surface-
# referenced and attenuation-inclusive, so a physical K would transfer the
# attenuation/surface-model uncertainty straight into the bed level; the
# K - K_phys diagnostic records that gap). ONE anchor-derived gamma field is
# shared by all three passes (same reasons as the picked bed: per-pass fields
# would confound the altitude comparison; the low pass's RSSNR is the
# cleanest). The 1-D profile extends CROSS-TRACK AS A CONSTANT -- same caveat
# class as the picked-bed residual.
RSSNR_SNAPSHOT = "3YH47013745B2T5ZZR50"   # antarctica store, 2026-07-29
RSSNR_STORE = {"bucket": "opr-radar-metrics", "prefix": "icechunk/antarctica",
               "region": "us-west-2"}
RSSNR_CACHE = OUT_DEFAULT / "rssnr_anchor.npz"
GRSSNR_TAG = "_rssnr"
RSSNR_GAMMA_NOTE = (
    "bed reflectivity driven along-track by required_surface_snr_dB "
    "(claude_notes/required_snr_dataset.md): |Gamma_bed|^2(s) dB = 2*A*H(s) "
    "- RSSNR(s) + K on the anchor along-track axis, H from the dataset's own "
    "surface/bed twtts, A = the run's --att, K median-anchored so the "
    "segment-median |Gamma|^2 equals the constant Fresnel ice->bed value "
    "(the dataset supplies RELATIVE structure; K - K_phys records the "
    "absolute-chain gap). Samples are ~1.4 km apart (10 s decimation), "
    "linearly interpolated along-track onto the bed grid and extended "
    "cross-track as a constant (the picked-bed residual's caveat class). "
    "Censored samples (qc fail / RSSNR NaN: bed too dim to pick) take the "
    "segment's dimmest mapped value -- a brightness floor, not "
    "missing-at-random. ONE anchor-derived field is shared by all three "
    "passes.")


def fetch_rssnr_anchor(cache_path=None):
    """RSSNR per decimated trace along the anchor frames (REF_FRAMES), from
    the pinned antarctica icechunk snapshot. Cache-first (RSSNR_CACHE);
    live-fetches once and caches with provenance. Returns (arrays, prov)."""
    cache = Path(cache_path or RSSNR_CACHE)
    keys = ("lat", "lon", "rssnr", "qc", "stw", "btw")
    if cache.exists():
        z = np.load(cache)
        prov = json.loads(str(z["provenance"]))
        if prov.get("snapshot_id") != RSSNR_SNAPSHOT:
            raise RuntimeError(
                f"RSSNR cache {cache} pins snapshot "
                f"{prov.get('snapshot_id')}, tool wants {RSSNR_SNAPSHOT}: "
                "delete the cache to re-fetch")
        prov["source"] = f"cache:{cache}"
        return {k: np.asarray(z[k]) for k in keys}, prov
    import icechunk
    import zarr
    storage = icechunk.s3_storage(anonymous=True, **RSSNR_STORE)
    repo = icechunk.Repository.open(storage=storage)
    root = zarr.open_group(
        repo.readonly_session(snapshot_id=RSSNR_SNAPSHOT).store, mode="r")
    fid = root["frame_id"][:].astype(str)
    m = np.isin(fid, [f"Data_{f}" for f in REF_FRAMES])
    if m.sum() < 50:
        raise RuntimeError(f"pinned snapshot holds only {m.sum()} anchor "
                           "traces")
    d = {"lat": root["latitude"][m], "lon": root["longitude"][m],
         "rssnr": root["required_surface_snr_dB"][m],
         "qc": root["qc_pass"][m].astype(bool),
         "stw": root["surface_twtt"][m], "btw": root["bed_twtt"][m]}
    prov = {"snapshot_id": RSSNR_SNAPSHOT, "store": dict(RSSNR_STORE),
            "frames": list(REF_FRAMES), "n_traces": int(m.sum()),
            "fetched_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "schema_note": "pre-2026-07-29 schema: qc_pass masks all "
            "metrics; no censoring columns"}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, provenance=json.dumps(prov),
             **{k: (v.astype(np.uint8) if v.dtype == bool else v)
                for k, v in d.items()})
    d["qc"] = d["qc"].astype(bool)
    prov["source"] = "s3-live"
    return d, prov


def k_phys_db(eps_ice=None):
    """Physical anchoring constant |Gamma_surf|^2_dB - T2_dB (the dataset's
    surface reference and two-way transmission): what K would be if the
    absolute chain (Fresnel surface, --att attenuation) were trusted."""
    g = fresnel_normal(1.0, eps_ice or rac.EPS_ICE)
    return float(20.0 * np.log10(abs(g)) - 20.0 * np.log10(1.0 - g * g))


def segment_s_range(ref, segment):
    """Anchor-axis s range (m) of the study segment, from the LOW pass's
    trace slices (the axis's own frames)."""
    off = dict(zip(ref["frames"],
                   np.concatenate([[0], np.cumsum(ref["frame_len"])[:-1]])))
    ss = []
    for fid, (a, b) in PASSES["low"][segment]:
        ss += [ref["s"][off[fid] + a], ref["s"][off[fid] + b - 1]]
    return float(min(ss)), float(max(ss))


def rssnr_gamma_profile(s, rssnr, thick_m, qc, att_db_per_km, seg_lo, seg_hi):
    """Median-anchored |Gamma_bed|^2(s) profile (module-section comment).

    Pure mapping math (unit-tested): G2 = 2*A*H - RSSNR + K with K set so
    median(G2) over QC-passing segment samples equals the constant run's
    Fresnel ice->bed power reflectivity. Censored samples (qc fail / NaN
    RSSNR) get the segment's minimum mapped G2 -- their RSSNR is a FLOOR
    (bed too dim to pick), never interpolated across. Returns the s-sorted
    profile + recorded stats."""
    s = np.asarray(s, np.float64)
    rssnr = np.asarray(rssnr, np.float64)
    thick_m = np.asarray(thick_m, np.float64)
    ok = (np.asarray(qc, bool) & np.isfinite(rssnr) & np.isfinite(thick_m)
          & (thick_m > 0))
    seg = ok & (s >= seg_lo) & (s <= seg_hi)
    if seg.sum() < 5:
        raise RuntimeError(f"only {seg.sum()} usable RSSNR samples in the "
                           "segment")
    base = 2.0 * att_db_per_km * thick_m / 1e3 - rssnr        # G2 - K
    g2_const = float(20.0 * np.log10(abs(
        fresnel_normal(rac.EPS_ICE, rac.EPS_BED))))
    k = g2_const - float(np.median(base[seg]))
    g2 = base + k
    floor = float(np.nanmin(g2[seg]))
    g2 = np.where(ok, g2, floor)
    o = np.argsort(s)
    kp = k_phys_db()
    gs = g2[seg]
    return {"s": s[o], "g2_db": g2[o], "thick_m": thick_m[o],
            "ok": ok[o], "k_db": round(k, 2), "k_phys_db": round(kp, 2),
            "k_minus_kphys_db": round(k - kp, 2),
            "g2_const_db": round(g2_const, 2),
            "att_db_per_km": att_db_per_km,
            "n_samples": int(len(s)), "n_censored": int((~ok).sum()),
            "censored_floor_db": round(floor, 2),
            "seg_s_km": [round(seg_lo / 1e3, 2), round(seg_hi / 1e3, 2)],
            "n_seg": int(seg.sum()),
            # G2 > 0 dB is unphysical reflectivity: the price of holding A
            # fixed while median-anchoring on a dim-bed-dominated segment.
            # K - K_phys / (2 * H_med) estimates the attenuation the
            # anchoring absorbed (recorded, not tuned away).
            "g2_pos_frac_seg": round(float((gs > 0).mean()), 3),
            "implied_eff_att_db_per_km": round(
                att_db_per_km + (k - kp)
                / (2.0 * float(np.median(thick_m[seg])) / 1e3), 1),
            "g2_seg_db": {kk: round(float(vv), 1) for kk, vv in
                          [("min", gs.min()), ("p5", np.percentile(gs, 5)),
                           ("med", np.median(gs)),
                           ("p95", np.percentile(gs, 95)),
                           ("max", gs.max())]},
            "med_sample_spacing_m": round(float(np.median(np.diff(s[o]))), 0)}


def build_rssnr_gamma(axis, segment, att):
    """Fetch + map: the shared anchor G2(s) profile dict (rssnr_gamma_profile
    output + fetch provenance), on the anchor along-track axis ``axis``
    (ref_bed_picks)."""
    d, prov = fetch_rssnr_anchor()
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    sx, sy = tr.transform(d["lon"], d["lat"])
    s_smp = project_to_track(sx, sy, axis["x"], axis["y"], axis["s"])
    thick = C / (2.0 * np.sqrt(axis["eps_ice"])) * (d["btw"] - d["stw"])
    seg_lo, seg_hi = segment_s_range(axis, segment)
    prof = rssnr_gamma_profile(s_smp, d["rssnr"], thick, d["qc"], att,
                               seg_lo, seg_hi)
    prof["provenance"] = prov
    prof["note"] = RSSNR_GAMMA_NOTE
    return prof


def apply_rssnr_gamma(base, axis, gmap):
    """Attach ``base.gamma_bed``: per-map-pixel signed FIELD reflection
    coefficient -10^(G2(s_pix)/20) from the shared profile, constant along
    the cross-track normal (apply_picked_bed's projection). Returns recorded
    stats."""
    bed = base.dems[1]
    tr = Transformer.from_crs("EPSG:3031", base.crs, always_xy=True)
    rx, ry = tr.transform(axis["x"], axis["y"])
    ny, nx = bed.shape
    xa, ya = base.transform * (0.0, 0.0)
    xb, yb = base.transform * (float(nx), float(ny))
    keep = ((rx >= min(xa, xb)) & (rx <= max(xa, xb))
            & (ry >= min(ya, yb)) & (ry <= max(ya, yb)))
    kk = np.where(keep)[0]
    if len(kk) < 100:
        raise RuntimeError("rssnr gamma: anchor axis does not cover the "
                           "scene")
    rx, ry, s_ref = rx[kk], ry[kk], axis["s"][kk]
    cols, rows = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
    px, py = base.transform * (cols.ravel(), rows.ravel())
    s_pix = project_to_track(px, py, rx, ry, s_ref)
    g2 = np.interp(s_pix, gmap["s"], gmap["g2_db"])
    base.gamma_bed = (-(10.0 ** (g2 / 20.0))).reshape(bed.shape).astype(
        np.float32)
    return {"k_db": gmap["k_db"], "k_phys_db": gmap["k_phys_db"],
            "k_minus_kphys_db": gmap["k_minus_kphys_db"],
            "g2_seg_db": gmap["g2_seg_db"],
            "n_censored": gmap["n_censored"],
            "grid_g2_db_range": [round(float(v), 1) for v in
                                 (20.0 * np.log10(np.abs(
                                     base.gamma_bed)).min(),
                                  20.0 * np.log10(np.abs(
                                      base.gamma_bed)).max())],
            "snapshot_id": gmap["provenance"]["snapshot_id"],
            "source": gmap["provenance"]["source"]}


# ========================================================================
# per-pass preparation
# ========================================================================
def _retry(what, fn, tries=3, delay_s=20.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # fsspec timeouts are a known flake
            if i == tries - 1:
                raise
            print(f"  [retry {i + 1}/{tries - 1}] {what}: {e}", flush=True)
            time.sleep(delay_s * (i + 1))


def radar_grid(params, surf_tw, bed_tw, dt, t0f, oversample, window):
    """rac.radar_grid with this study's margins (post-bed window POST_BED_US
    > clutter margin MARGIN_US): alias-free dt/oversample grid anchored on a
    frame-dt bin so decimating [::oversample] lands on the frame lattice."""
    lo = float(np.nanmin(surf_tw)) - PRE_SURF_US * 1e-6
    hi = float(np.nanmax(bed_tw)) + POST_BED_US * 1e-6
    b0 = int(np.floor((lo - t0f) / dt))
    nb = int(np.ceil((hi - t0f) / dt)) - b0 + 1
    wf = params["waveform"]
    wave = WaveformConfig(kind="chirp", bandwidth=wf["bandwidth_Hz"],
                          pulse_length=wf["bed_waveform_pulse_length_s"],
                          window=window)
    ant = AntennaConfig(kind="array", n_elements=rac.N_ELEMENTS,
                        spacing_lam=rac.SPACING_LAM, roll_source="nav")
    f0 = wf["center_frequency_Hz"]
    t0 = t0f + b0 * dt
    rc_sim = RadarConfig(dt=dt / oversample, n_samples=oversample * (nb - 1) + 1,
                         t0=t0, f0=f0, waveform=wave, antenna=ant)
    rc_frame = RadarConfig(dt=dt, n_samples=nb, t0=t0, f0=f0)
    return rc_sim, rc_frame, b0


def prep_pass(key, segment, n_traces, ref=None, gmap=None, axis=None):
    """Slice (+reverse) the pass's frames onto the common window, derive the
    reach and grids, and build the base scene (REMA + BedMachine, cached).
    ``ref`` (ref_bed_picks) applies the picked-bed residual to that scene;
    ``gmap`` (build_rssnr_gamma, with ``axis`` = ref_bed_picks as the
    along-track axis) attaches the RSSNR-driven bed gamma grid."""
    spec = PASSES[key]
    parts = spec[segment]
    fsubs, bots, tw_ref = [], [], None
    for fid, (a, b) in parts:
        frame = load_frame(SEASON, fid)
        tw = np.asarray(frame.twtt.values, np.float64)
        if tw_ref is None:
            tw_ref = tw
        elif not np.allclose(tw, tw_ref):
            raise RuntimeError(f"{key}: twtt grid differs between frames")
        bot = load_bottom_pick(frame)[a:b]
        fs = frame.isel(slow_time=slice(a, b))
        if spec["rev"]:
            fs = fs.isel(slow_time=slice(None, None, -1))
            bot = bot[::-1]
        fsubs.append(fs)
        bots.append(bot)
    fsub = fsubs[0] if len(fsubs) == 1 else xr.concat(
        fsubs, dim="slow_time", combine_attrs="override")
    bot_sub = np.concatenate(bots)
    roll_note = None
    if spec["rev"]:
        # Reversed trace order flips the kernel's nav-derived along-track
        # axis u_at; roll is applied about u_at, so negate it to preserve
        # the PHYSICAL tilt direction of the array (scout pitfall 2).
        fsub = fsub.assign(Roll=-fsub.Roll)
        roll_note = ("pass flown backwards: slices reversed and nav roll "
                     "NEGATED (roll rotates about the nav-order along-track "
                     "axis, which reversal flips)")

    params = rac.mcords_params(SEASON, spec["param_frame"])
    wf = params["waveform"]
    f0, bw = wf["center_frequency_Hz"], wf["bandwidth_Hz"]
    window, win_note = rac.map_window(wf["pulse_compression_freq_window"])
    # Scout quirk 7: ft_wind decode falls back on all three 2016 passes; the
    # scout verified by hand the true value IS hanning (param_csarp.csarp).
    win_note = ("ft_wind provenance is the decode-fallback string on all "
                "three 2016 passes; scout-verified true value IS hanning, so "
                "the modeled 'hann' is correct (provenance, not measurement)")
    dt = float((tw_ref[-1] - tw_ref[0]) / (len(tw_ref) - 1))
    t0f = float(tw_ref[0])
    oversample, f_alias = rac.pick_oversample(dt, f0, bw)

    surf = np.asarray(fsub.Surface.values, np.float64)
    agl = surf * C / 2.0                       # nadir air range = AGL
    h_max, r_min = float(np.nanmax(agl)), float(np.nanmin(agl))
    dbs = bot_sub - surf                       # bed delay below surface (s)
    dbs_max = float(np.nanmax(dbs))
    thick = dbs * C / (2.0 * np.sqrt(rac.EPS_ICE))
    d_min, thick_med = float(np.nanmin(thick)), float(np.nanmedian(thick))
    reach = derive_reach(h_max, dbs_max, d_min)

    lam = C / f0
    spacing = rac.facet_spacing(lam, r_min, thick_med)
    bed_fill = np.where(np.isfinite(bot_sub), bot_sub, np.nanmax(bot_sub))
    rc_sim, rc_frame, b0 = radar_grid(params, surf, bed_fill, dt, t0f,
                                      oversample, window)

    base, aux = _retry(f"base_scene {key}",
                       lambda: rac.base_scene(fsub, n_traces, reach["ct_m"]))
    # Picked bed: the fast-time grid, reach and facet spacing above stay at
    # their BedMachine values (derived from each pass's OWN picks) so the two
    # runs share one lattice and are directly comparable.
    aux["picked_bed"] = apply_picked_bed(base, ref) if ref else None
    aux["rssnr_gamma"] = (apply_rssnr_gamma(base, axis or ref, gmap)
                          if gmap else None)
    idx = aux["idx"]
    lat, lon = rac._lonlat(fsub)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    px, py = tr.transform(lon, lat)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    return {"key": key, "segment": segment, "parts": parts, "rev": spec["rev"],
            "roll_note": roll_note, "params": params, "window": window,
            "win_note": win_note, "fsub": fsub, "bot": bot_sub, "surf": surf,
            "dt": dt, "t0f": t0f, "oversample": oversample, "f_alias": f_alias,
            "lam": lam, "spacing": spacing, "reach": reach, "rc_sim": rc_sim,
            "rc_frame": rc_frame, "b0": b0, "base": base, "aux": aux,
            "idx": idx, "s_m": s, "agl": agl, "r_min": r_min,
            "picked_bed": bool(ref), "gamma_rssnr": bool(gmap),
            "h_med": float(np.nanmedian(agl)), "thick_med": thick_med,
            "tw_m": tw_ref}


# ========================================================================
# chunked simulation (pilot = 1 chunk; 50 km segment = ~5 identical chunks)
# ========================================================================
def chunk_rows(p):
    """Split the sim trace indices into ~CHUNK_M along-track chunks."""
    s_sel = p["s_m"][p["idx"]]
    track = float(s_sel[-1] - s_sel[0])
    n_chunks = max(1, int(round(track / CHUNK_M)))
    edges = s_sel[0] + track * np.arange(1, n_chunks) / n_chunks
    which = np.searchsorted(edges, s_sel)
    return [np.where(which == c)[0] for c in range(n_chunks)]


def chunk_scene(base, rows, ct, gamma=False):
    """MultilayerScene for one chunk: DEM stack cropped to the chunk traces'
    bbox padded by ct + 100 m (every trace keeps full +-ct coverage in every
    direction), nav/roll subset. The rac.crop_scene pattern + trace subset.
    ``gamma`` attaches the cropped RSSNR bed-gamma grid (scene.gamma_maps,
    consumed by simulate's multilayer path)."""
    from affine import Affine

    from soundersim.synthetic import MultilayerScene

    tr = Transformer.from_crs("EPSG:4326", base.crs, always_xy=True)
    nav = base.nav_llh[rows]
    px, py = tr.transform(nav[:, 1], nav[:, 0])
    pad = ct + 100.0
    ny, nx = base.dem.shape
    cols, rws = (~base.transform) * (
        np.array([px.min() - pad, px.max() + pad]),
        np.array([py.min() - pad, py.max() + pad]))
    c0 = int(np.clip(np.floor(min(cols)), 0, nx - 2))
    c1 = int(np.clip(np.ceil(max(cols)) + 1, c0 + 2, nx))
    r0 = int(np.clip(np.floor(min(rws)), 0, ny - 2))
    r1 = int(np.clip(np.ceil(max(rws)) + 1, r0 + 2, ny))
    dems = [np.ascontiguousarray(d[r0:r1, c0:c1]) for d in base.dems]
    sc = MultilayerScene(f"{base.name}_r{rows[0]}", dems,
                         base.transform * Affine.translation(c0, r0),
                         base.crs, nav, base.media, dict(base.params))
    roll = getattr(base, "nav_roll", None)
    sc.nav_roll = None if roll is None else np.asarray(roll)[rows]
    if gamma:
        sc.gamma_maps = {"bed": (
            np.ascontiguousarray(base.gamma_bed[r0:r1, c0:c1]),
            sc.transform, sc.crs)}
    return sc


def sim_cfg(rc_sim, spacing, att, surf_rough):
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


def simulate_pass(p, runs_dir, att, surf_rough, force):
    """Chunked cached coherent surface+bed runs; assembled per-layer fields.
    Returns dict(field (T,nb,2), twtt, nadir (T,2), wall_s, facets, ...)."""
    chunks = chunk_rows(p)
    cfg = sim_cfg(p["rc_sim"], p["spacing"], att, surf_rough)
    n = len(p["idx"])
    field = twtt = nadir = None
    wall, facets, dropped = 0.0, [], []
    for ci, rows in enumerate(chunks):
        scene = chunk_scene(p["base"], rows, p["reach"]["ct_m"],
                            gamma=p["gamma_rssnr"])
        rid = (f"{p['key']}_{p['segment']}"
               f"{case_tag(p['picked_bed'], p['gamma_rssnr'])}"
               f"_c{ci:02d}"
               + ("_srough" if surf_rough else "")
               + (f"_att{att:g}" if att != rac.ATT_DB_PER_KM else ""))
        # gamma keys only when ON: constant-gamma metas stay byte-identical
        # to pre-feature caches (run_level skip-exists keys on meta json)
        meta = {"season": SEASON, "pass": p["key"], "segment": p["segment"],
                "picked_bed": p["picked_bed"],
                **({"gamma_rssnr": True, "rssnr_snapshot": RSSNR_SNAPSHOT,
                    "rssnr_k_db": p["aux"]["rssnr_gamma"]["k_db"]}
                   if p["gamma_rssnr"] else {}),
                "parts": [[fid, list(sl)] for fid, sl in p["parts"]],
                "reversed": p["rev"], "chunk": ci, "n_chunks": len(chunks),
                "rows": [int(rows[0]), int(rows[-1])], "n_traces_total": n,
                "spacing_m": round(p["spacing"], 4),
                "ct_m": round(p["reach"]["ct_m"], 1), "att_db_per_km": att,
                "window": p["window"], "surf_rough": bool(surf_rough),
                "dt_sim_ns": round(p["rc_sim"].dt * 1e9, 5),
                "t0_us": round(p["rc_sim"].t0 * 1e6, 5),
                "n_samples_sim": p["rc_sim"].n_samples}
        diag, arrs = rac.run_level(rid, scene, cfg, meta, runs_dir,
                                   p["oversample"], force)
        if field is None:
            field = np.zeros((n,) + arrs["field"].shape[1:], np.complex64)
            nadir = np.zeros((n, arrs["nadir_twtt"].shape[1]))
            twtt = arrs["twtt"]
        field[rows] = arrs["field"]
        nadir[rows] = arrs["nadir_twtt"]
        wall += diag["wall_s"]
        facets.append(diag["n_facets_per_interface"])
        dropped.append(diag["dropped_power_fraction"])
    if not np.isfinite(field).all():
        raise RuntimeError(f"{p['key']}: non-finite assembled field")
    return {"field": field, "twtt": twtt, "nadir": nadir, "wall_s": wall,
            "n_chunks": len(chunks), "facets_per_chunk": facets,
            "dropped_power_fraction": dropped}


# ========================================================================
# analysis
# ========================================================================
def _wpeak(P, twtt, dt, t_c, win_us):
    """Per-trace max power within +-win_us of t_c (NaN-guarded)."""
    n = len(twtt)
    out = np.full(P.shape[0], np.nan)
    for t in range(P.shape[0]):
        if not np.isfinite(t_c[t]):
            continue
        a = int(np.clip((t_c[t] - win_us * 1e-6 - twtt[0]) / dt, 0, n - 2))
        b = int(np.clip((t_c[t] + win_us * 1e-6 - twtt[0]) / dt, a + 1, n))
        out[t] = float(P[t, a:b].max())
    return out


def _wmean(P, twtt, dt, t_lo, t_hi):
    """Per-trace mean power in [t_lo[t], t_hi[t]] (NaN where empty)."""
    n = len(twtt)
    out = np.full(P.shape[0], np.nan)
    for t in range(P.shape[0]):
        if not (np.isfinite(t_lo[t]) and np.isfinite(t_hi[t])
                and t_hi[t] > t_lo[t]):
            continue
        a = int(np.clip((t_lo[t] - twtt[0]) / dt, 0, n - 2))
        b = int(np.clip((t_hi[t] - twtt[0]) / dt, a + 2, n))
        out[t] = float(P[t, a:b].mean())
    return out


def _med_db_rel(num, den):
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0) & (num > 0)
    if not ok.any():
        return float("nan")
    return float(np.median(10.0 * np.log10(num[ok] / den[ok])))


def clutter_metrics(P, twtt, dt, t_s, t_b):
    """The study's clutter currencies, per trace then median, all in dB rel
    the trace's OWN surface peak (gain-free): mid-column mean power
    (surf+1.0 -> bed-0.5 us), bed-window mean power (bed-0.5 -> bed+1.5 us),
    and the scout's contrast metric (mean(bed-3.0 .. bed-0.6 us) over the
    bed peak +-0.3 us) for direct comparison with the scout table."""
    spk = _wpeak(P, twtt, dt, t_s, SURF_WIN_US)
    mid = _wmean(P, twtt, dt, t_s + MID_LO_US * 1e-6, t_b - MID_HI_US * 1e-6)
    bed = _wmean(P, twtt, dt, t_b - BED_LO_US * 1e-6, t_b + BED_HI_US * 1e-6)
    sc_m = _wmean(P, twtt, dt, t_b - SCOUT_LO_US * 1e-6,
                  t_b - SCOUT_HI_US * 1e-6)
    bpk = _wpeak(P, twtt, dt, t_b, SCOUT_PK_US)
    return {"midcol_rel_surf_db": _med_db_rel(mid, spk),
            "bed_rel_surf_db": _med_db_rel(bed, spk),
            "scout_midcol_over_bedpeak_db": _med_db_rel(sc_m, bpk),
            "_spk": spk, "_mid": mid, "_bed": bed}


def rel_mean_profile(P, twtt, dt, t_ref, norm, lo_us=-1.5, hi_us=14.5):
    """(rel_us, dB): mean power vs twtt below each trace's own reference
    time, each trace normalized by ``norm`` (its own surface peak), integer
    bin shifts (every grid here shares the 20.202 ns lattice)."""
    k0 = int(round(lo_us * 1e-6 / dt))          # negative
    k1 = int(round(hi_us * 1e-6 / dt))
    nrel = k1 - k0 + 1
    acc, cnt = np.zeros(nrel), np.zeros(nrel)
    n = len(twtt)
    for t in range(P.shape[0]):
        if not (np.isfinite(t_ref[t]) and np.isfinite(norm[t])
                and norm[t] > 0):
            continue
        pk = int(round((t_ref[t] - twtt[0]) / dt))
        a, b = max(0, pk + k0), min(n, pk + k1 + 1)
        off = a - (pk + k0)
        acc[off:off + (b - a)] += P[t, a:b] / norm[t]
        cnt[off:off + (b - a)] += 1
    prof = acc / np.maximum(cnt, 1)
    rel_us = (np.arange(nrel) + k0) * dt * 1e6
    return rel_us, 10.0 * np.log10(np.maximum(prof, 1e-30))


def analyze_pass(p, sim):
    """Per-pass sim-vs-measured clutter metrics + per-interface (surface- vs
    bed-borne) decomposition + profiles for the figures."""
    tw, dtf = sim["twtt"], p["rc_frame"].dt
    F = sim["field"]
    P = np.abs(F.sum(-1)) ** 2
    Ps, Pb = np.abs(F[..., 0]) ** 2, np.abs(F[..., 1]) ** 2
    surf_pick = p["surf"][p["idx"]]

    # per-pass surface registration (scout pitfall 5: never shared)
    gate = rac.leading_edge_gate(Ps, p["spacing"], dtf, p["rc_frame"].t0,
                                 surf_pick)
    t_s = rac.surface_peak_twtt(P, tw, sim["nadir"][:, 0], dtf,
                                win_us=SURF_WIN_US)
    t_b = sim["nadir"][:, 1]

    m_sim = clutter_metrics(P, tw, dtf, t_s, t_b)
    spk = m_sim["_spk"]
    dec = {}
    bedlayer_bed = None
    for name, Pl in (("surface", Ps), ("bed", Pb)):
        mid = _wmean(Pl, tw, dtf, t_s + MID_LO_US * 1e-6,
                     t_b - MID_HI_US * 1e-6)
        bed = _wmean(Pl, tw, dtf, t_b - BED_LO_US * 1e-6,
                     t_b + BED_HI_US * 1e-6)
        if name == "bed":
            bedlayer_bed = bed  # per-trace, for the RSSNR sanity correlation
        dec[name] = {"midcol_rel_surf_db": _med_db_rel(mid, spk),
                     "bed_rel_surf_db": _med_db_rel(bed, spk)}
    dmid = (dec["surface"]["midcol_rel_surf_db"]
            - dec["bed"]["midcol_rel_surf_db"])
    verdict = ("surface-borne" if dmid > 3.0 else
               "bed-borne" if dmid < -3.0 else "mixed")

    # measured: ALL traces of the segment, windows on its OWN picks
    meas = np.asarray(p["fsub"].Data.values, np.float64)
    tw_m, dt_m = p["tw_m"], p["dt"]
    m_meas = clutter_metrics(meas, tw_m, dt_m, p["surf"], p["bot"])
    n_m = meas.shape[0]
    floor = _wmean(meas, tw_m, dt_m,
                   np.full(n_m, tw_m[-1] - FLOOR_TAIL_LO_US * 1e-6),
                   np.full(n_m, tw_m[-1] - FLOOR_TAIL_HI_US * 1e-6))
    floor_db = _med_db_rel(floor, m_meas["_spk"])
    noise_limited = bool(m_meas["midcol_rel_surf_db"] - floor_db < 3.0)

    profs = {
        "sim_total": rel_mean_profile(P, tw, dtf, t_s, spk),
        "sim_surface": rel_mean_profile(Ps, tw, dtf, t_s, spk),
        "sim_bed": rel_mean_profile(Pb, tw, dtf, t_s, spk),
        "measured": rel_mean_profile(meas, tw_m, dt_m, p["surf"],
                                     m_meas["_spk"]),
    }
    clean = {k: round(v, 2) for k, v in m_sim.items() if not k.startswith("_")}
    cleanm = {k: round(v, 2) for k, v in m_meas.items()
              if not k.startswith("_")}

    def _prof_db(m):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = 10.0 * np.log10(m["_bed"] / m["_spk"])
        return np.where(np.isfinite(r), r, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        blp = 10.0 * np.log10(bedlayer_bed / spk)
    return {"gate": gate, "sim": clean, "meas": cleanm,
            "sim_bed_prof_db": _prof_db(m_sim),
            "meas_bed_prof_db": _prof_db(m_meas),
            "sim_bedlayer_prof_db": np.where(np.isfinite(blp), blp, np.nan),
            "decomposition": {k: {kk: round(vv, 2) for kk, vv in v.items()}
                              for k, v in dec.items()},
            "verdict": verdict, "floor_db": round(floor_db, 2),
            "meas_noise_limited": noise_limited,
            "bed_delay_med_us": round(float(np.nanmedian(
                (p["bot"] - p["surf"]))) * 1e6, 2),
            "profs": profs, "P": P, "t_s": t_s, "meas_arr": meas}


# ========================================================================
# RSSNR-gamma acceptance analysis: bed-window brightness along-track
# ========================================================================
CORR_WIN_M = 1000.0    # profile smoothing scale (~ the RSSNR sampling)


def _smooth_db(s, v, win_m=CORR_WIN_M):
    """~win_m running mean of a per-trace dB profile (NaNs interpolated)."""
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.full_like(np.asarray(v, float), np.nan)
    vi = np.interp(s, s[ok], v[ok])
    n = max(1, int(round(win_m / max(float(np.median(np.diff(s))), 1e-6))))
    return ndimage.uniform_filter1d(vi, n, mode="nearest")


def _pearson(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def bed_profile_correlations(p, a, a_const, gmap, axis):
    """Acceptance metrics for one pass: along-track Pearson r of bed-window
    power profiles (dB rel own surface peak, ~1 km smoothed, on the sim trace
    grid). sim(RSSNR) vs the RSSNR-implied pattern is the by-construction
    sanity check (geometry/speckle-limited); sim vs MEASURED -- for both
    gamma models -- is the real test. Returns (stats, plot-series)."""
    s_meas, s_sim = p["s_m"], p["s_m"][p["idx"]]
    meas = np.interp(s_sim, s_meas,
                     _smooth_db(s_meas, a["meas_bed_prof_db"]))
    sim_r = _smooth_db(s_sim, a["sim_bed_prof_db"])
    sim_c = _smooth_db(s_sim, a_const["sim_bed_prof_db"])
    sim_rl = _smooth_db(s_sim, a["sim_bedlayer_prof_db"])  # bed-borne only
    # implied pattern -RSSNR(s) + K (== G2 - 2AH), at the sim traces'
    # anchor-axis position
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    px, py = tr.transform(p["base"].nav_llh[:, 1], p["base"].nav_llh[:, 0])
    s_anchor = project_to_track(px, py, axis["x"], axis["y"], axis["s"])
    implied = (np.interp(s_anchor, gmap["s"], gmap["g2_db"])
               - 2.0 * gmap["att_db_per_km"]
               * np.interp(s_anchor, gmap["s"], gmap["thick_m"]) / 1e3)
    stats = {
        "r_sim_rssnr_vs_implied": round(_pearson(sim_r, implied), 3),
        "r_bedlayer_rssnr_vs_implied": round(_pearson(sim_rl, implied), 3),
        "r_sim_rssnr_vs_measured": round(_pearson(sim_r, meas), 3),
        "r_sim_const_vs_measured": round(_pearson(sim_c, meas), 3),
        "r_implied_vs_measured": round(_pearson(implied, meas), 3),
        "smooth_win_m": CORR_WIN_M,
        "bed_rel_surf_med_db": {
            "measured": round(float(np.nanmedian(meas)), 2),
            "sim_const": round(float(np.nanmedian(sim_c)), 2),
            "sim_rssnr": round(float(np.nanmedian(sim_r)), 2)}}
    series = {"s_sim": s_sim, "measured": meas, "sim_const": sim_c,
              "sim_rssnr": sim_r, "implied": implied}
    return stats, series


def fig_bed_brightness(out, preps, corr_series, corr_stats, segment):
    """Per pass: bed-window power along-track (dB rel own surface peak,
    ~1 km smoothed) -- measured vs constant-gamma sim vs RSSNR-gamma sim vs
    the RSSNR-implied pattern (shape prediction, median-aligned to the RSSNR
    sim)."""
    s0 = S0_KM[segment]
    fig, axs = plt.subplots(1, len(ORDER), figsize=(5.4 * len(ORDER), 4.6),
                            sharey=True, squeeze=False)
    for k, key in enumerate(ORDER):
        ax = axs[0, k]
        se, st = corr_series[key], corr_stats[key]
        s_km = s0 + se["s_sim"] / 1e3
        imp = se["implied"] + (np.nanmedian(se["sim_rssnr"])
                               - np.nanmedian(se["implied"]))
        ax.plot(s_km, se["measured"], color="black", lw=1.8, label="measured")
        ax.plot(s_km, se["sim_const"], color="tab:blue", lw=1.3,
                label="sim constant gamma")
        ax.plot(s_km, se["sim_rssnr"], color="tab:red", lw=1.3,
                label="sim RSSNR gamma")
        ax.plot(s_km, imp, color="0.45", lw=1.0, ls="--",
                label="RSSNR-implied (median-aligned)")
        ax.set_title(
            f"{key} ({preps[key]['h_med']:.0f} m AGL)  r(meas): const "
            f"{st['r_sim_const_vs_measured']:+.2f} -> RSSNR "
            f"{st['r_sim_rssnr_vs_measured']:+.2f}", fontsize=9)
        ax.set_xlabel("anchor along-track s (km)")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("bed window mean power, dB rel own surface peak")
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("bed-window brightness along-track: measured vs sim "
                 f"(constant vs RSSNR-driven bed gamma), {CORR_WIN_M:.0f} m "
                 "smoothing")
    fig.tight_layout()
    fp = out / "bed_brightness.png"
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


# ========================================================================
# figures (grayscale radargrams = sequential magnitude; profile series in
# fixed categorical order with legend, one axis)
# ========================================================================
def fig_radargrams(out, preps, analyses, segment):
    """Measured (top) vs simulated (bottom) per pass, shared surface-
    referenced twtt axis and one shared dB-rel-surface color scale."""
    y_lo, y_hi = -1.0, 13.5
    vmin, vmax = -90.0, 5.0
    s0 = S0_KM[segment]
    fig, axs = plt.subplots(2, len(ORDER), figsize=(5.4 * len(ORDER), 8.8),
                            sharey=True, squeeze=False)
    for k, key in enumerate(ORDER):
        p, a = preps[key], analyses[key]
        # measured: dB rel per-pass median surface peak
        ref_m = 10.0 * np.log10(max(np.nanmedian(
            _wpeak(a["meas_arr"], p["tw_m"], p["dt"], p["surf"],
                   SURF_WIN_US)), 1e-300))
        surf_med = float(np.nanmedian(p["surf"]))
        rel = (p["tw_m"] - surf_med) * 1e6
        m = (rel >= y_lo) & (rel <= y_hi)
        s_km = s0 + p["s_m"] / 1e3
        ax = axs[0, k]
        ax.imshow(_db(a["meas_arr"])[:, m].T - ref_m, aspect="auto",
                  cmap="gray", vmin=vmin, vmax=vmax,
                  extent=[s_km[0], s_km[-1], rel[m][-1], rel[m][0]])
        ax.set_title(f"{key} measured ({p['h_med']:.0f} m AGL)", fontsize=10)
        # sim: dB rel per-pass median simulated surface peak
        twtt_s = p["rc_frame"].t0 + np.arange(
            p["rc_frame"].n_samples) * p["rc_frame"].dt
        ref_s = 10.0 * np.log10(max(float(np.nanmedian(
            _wpeak(a["P"], twtt_s, p["rc_frame"].dt, a["t_s"],
                   SURF_WIN_US))), 1e-300))
        surf_med_s = float(np.nanmedian(a["t_s"]))
        rel_s = (twtt_s - surf_med_s) * 1e6
        ms = (rel_s >= y_lo) & (rel_s <= y_hi)
        s_sim = s0 + p["s_m"][p["idx"]] / 1e3
        ax = axs[1, k]
        ax.imshow(_db(a["P"])[:, ms].T - ref_s, aspect="auto", cmap="gray",
                  vmin=vmin, vmax=vmax,
                  extent=[s_sim[0], s_sim[-1], rel_s[ms][-1], rel_s[ms][0]])
        ax.set_title(f"{key} sim (ct ±{p['reach']['ct_m'] / 1e3:.1f} km, "
                     f"{p['spacing']:.1f} m facets)", fontsize=10)
        ax.set_xlabel("anchor along-track s (km)")
    for r in range(2):
        axs[r, 0].set_ylabel("twtt below surface (us)")
    fig.suptitle("basal-clutter altitude triplet: measured (top) vs "
                 "simulated surface+bed (bottom), dB rel own surface peak")
    fig.tight_layout()
    fp = out / "radargrams.png"
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


def fig_decomposition(out, preps, analyses):
    """Per pass: measured vs sim total vs the sim's per-interface split
    (surface-borne vs bed-borne) mean-power profiles below the surface."""
    series = [("measured", "measured", dict(color="black", lw=1.8)),
              ("sim_total", "sim total", dict(color="tab:blue", lw=1.4)),
              ("sim_surface", "sim surface-borne",
               dict(color="tab:orange", lw=1.2, ls="--")),
              ("sim_bed", "sim bed-borne",
               dict(color="tab:green", lw=1.2, ls="-."))]
    fig, axs = plt.subplots(1, len(ORDER), figsize=(5.2 * len(ORDER), 4.8),
                            sharey=True, squeeze=False)
    for k, key in enumerate(ORDER):
        ax = axs[0, k]
        a = analyses[key]
        for pk, label, st in series:
            ax.plot(*a["profs"][pk], label=label, **st)
        tb = a["bed_delay_med_us"]
        ax.axvspan(1.0, tb - MID_HI_US, color="tab:blue", alpha=0.06,
                   label="mid-column window" if k == 0 else None)
        ax.axvline(tb, color="0.5", lw=0.8, ls=":")
        ax.text(tb, -108, " median bed", fontsize=7, color="0.4")
        ax.set_xlim(-1.0, 13.5)
        ax.set_ylim(-110, 5)
        ax.grid(alpha=0.3)
        ax.set_title(f"{key} ({preps[key]['h_med']:.0f} m AGL)  "
                     f"[{a['verdict']}]", fontsize=10)
        ax.set_xlabel("twtt below surface (us)")
        if k == 0:
            ax.set_ylabel("dB rel own surface peak (mean power)")
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("clutter decomposition: which interface supplies the "
                 "mid-column energy (per-layer coherent fields)")
    fig.tight_layout()
    fp = out / "decomposition.png"
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


# ========================================================================
# main
# ========================================================================
def run(segment="pilot", n_traces=None, att=rac.ATT_DB_PER_KM,
        surf_rough=True, out_root=None, force=False, make_report=True,
        picked_bed=False, gamma_rssnr=False):
    n_traces = n_traces or (N_TRACES_PILOT if segment == "pilot"
                            else N_TRACES_FULL)
    tag = case_tag(picked_bed, gamma_rssnr)
    out = Path(out_root or OUT_DEFAULT) / (segment + tag)
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"
    case = f"{CASE_PREFIX}_{segment}{tag}"
    axis = ref_bed_picks() if (picked_bed or gamma_rssnr) else None
    ref = axis if picked_bed else None
    if picked_bed:
        print(f"picked bed: reference pass {ref['pass']} "
              f"({'/'.join(ref['frames'])}), {ref['n']} picks over "
              f"{ref['line_len_km']} km, line gap frac "
              f"{ref['gap_frac_line']:.4f}", flush=True)
    gmap = None
    if gamma_rssnr:
        gmap = build_rssnr_gamma(axis, segment, att)
        print(f"rssnr gamma: {gmap['n_samples']} samples "
              f"({gmap['provenance']['source']}), snapshot {RSSNR_SNAPSHOT}, "
              f"K {gmap['k_db']} dB (K - K_phys "
              f"{gmap['k_minus_kphys_db']} dB), segment G2 "
              f"[{gmap['g2_seg_db']['min']} .. {gmap['g2_seg_db']['max']}] "
              f"dB (med {gmap['g2_seg_db']['med']}), censored "
              f"{gmap['n_censored']}/{gmap['n_samples']}", flush=True)
    preps, sims, analyses = {}, {}, {}
    for key in ORDER:
        print(f"== {key} ({segment}{tag}) ==", flush=True)
        p = prep_pass(key, segment, n_traces, ref=ref, gmap=gmap, axis=axis)
        if p["aux"]["picked_bed"]:
            pb = p["aux"]["picked_bed"]
            print(f"  picked bed: residual rms {pb['residual_rms_m']} m "
                  f"(mean {pb['residual_mean_m']}, |max| "
                  f"{pb['residual_absmax_m']}), gaps "
                  f"{pb['gap_frac_segment']:.4f}; along-track bed roughness "
                  f"{pb['bed_roughness_rms_m']['bedmachine']} -> "
                  f"{pb['bed_roughness_rms_m']['picked']} m rms", flush=True)
        print(f"  reach: surface {p['reach']['surface_reach_m']:.0f} m, bed "
              f"{p['reach']['bed_reach_m']:.0f} m -> ct "
              f"±{p['reach']['ct_m']:.0f} m; spacing {p['spacing']:.2f} m; "
              f"n_samples_sim {p['rc_sim'].n_samples}", flush=True)
        preps[key] = p
        sims[key] = simulate_pass(p, runs_dir, att, surf_rough, force)
        analyses[key] = analyze_pass(p, sims[key])

    # ---- RSSNR-gamma acceptance: vs the constant-gamma companion run ----
    corr_stats = corr_series = None
    if gamma_rssnr:
        runs_const = (Path(out_root or OUT_DEFAULT)
                      / (segment + case_tag(picked_bed)) / "runs")
        corr_stats, corr_series = {}, {}
        for key in ORDER:
            print(f"== {key} constant-gamma companion (cache-first) ==",
                  flush=True)
            p_const = dict(preps[key])
            p_const["gamma_rssnr"] = False
            sim_c = simulate_pass(p_const, runs_const, att, surf_rough, False)
            a_const = analyze_pass(preps[key], sim_c)
            corr_stats[key], corr_series[key] = bed_profile_correlations(
                preps[key], analyses[key], a_const, gmap, axis)
            st = corr_stats[key]
            print(f"  bed-brightness r vs measured: const "
                  f"{st['r_sim_const_vs_measured']:+.3f} -> RSSNR "
                  f"{st['r_sim_rssnr_vs_measured']:+.3f} (sanity vs implied: "
                  f"total {st['r_sim_rssnr_vs_implied']:+.3f}, bed-layer "
                  f"{st['r_bedlayer_rssnr_vs_implied']:+.3f}; "
                  f"implied-vs-meas "
                  f"{st['r_implied_vs_measured']:+.3f})", flush=True)

    # ---- metrics ----
    rec = "recorded only"
    metrics = {}
    for key in ORDER:
        p, a, s = preps[key], analyses[key], sims[key]
        g = a["gate"]
        metrics[f"surface_alignment_{key}"] = {
            "value": g["median_bins"], "threshold": rac.GATE_BINS, "op": "<=",
            "pass": bool(g["median_bins"] <= rac.GATE_BINS),
            "offset_bins": g["offset_bins"], "p90_bins": g["p90_bins"],
            "note": "per-pass constant-offset leading-edge gate vs the "
            "frame's own Surface pick (scout pitfall: registrations differ "
            "across passes by ~1.5 bins bias and 4x scatter -- never shared)"}
        metrics[f"clutter_{key}"] = {
            "value": a["sim"]["midcol_rel_surf_db"], "threshold": None,
            "op": "record", "pass": True,
            "sim": a["sim"], "measured": a["meas"],
            "decomposition_db": a["decomposition"],
            "midcol_verdict": a["verdict"],
            "measured_floor_rel_surf_db": a["floor_db"],
            "measured_midcol_noise_limited": a["meas_noise_limited"],
            "agl_med_m": round(p["h_med"], 0),
            "note": "mid-column mean power (surf+1.0 -> bed-0.5 us) rel own "
            "surface peak, median over traces; bed window bed-0.5 -> "
            "bed+1.5 us; scout_midcol_over_bedpeak matches the scout table "
            "metric (mean bed-3.0..bed-0.6 us over bed peak +-0.3 us). "
            "decomposition_db: same windows on the per-interface coherent "
            "fields (surface-borne vs bed-borne). measured floor: deep "
            "record tail (last 0.2-3.2 us; pre-surface is TX-leakage/"
            "img_comb-contaminated on the low pass). " + rec}
    # headline: altitude trend of mid-column clutter, sim vs measured
    trend = {}
    for hi in ("mid", "high"):
        trend[f"{hi}-low"] = {
            "measured_db": round(analyses[hi]["meas"]["midcol_rel_surf_db"]
                                 - analyses["low"]["meas"]["midcol_rel_surf_db"], 2),
            "sim_db": round(analyses[hi]["sim"]["midcol_rel_surf_db"]
                            - analyses["low"]["sim"]["midcol_rel_surf_db"], 2)}
        trend[f"{hi}-low"]["error_db"] = round(
            trend[f"{hi}-low"]["sim_db"] - trend[f"{hi}-low"]["measured_db"], 2)
    metrics["altitude_trend"] = {
        "value": trend["high-low"]["sim_db"], "threshold": None,
        "op": "record", "pass": True, "pairs": trend,
        "note": "KEY DELIVERABLE: mid-column clutter power delta (dB, rel "
        "own surface peaks -- gain-free) high/mid pass minus low pass, sim "
        "vs measured; the scout's measured whole-line value is ~+20 dB. "
        "If the low pass's measured mid-column is noise-limited its "
        "measured delta is a LOWER bound. " + rec}
    metrics["simulation_wall_s"] = {
        "value": round(sum(s["wall_s"] for s in sims.values()), 1),
        "threshold": None, "op": "record", "pass": True,
        "per_pass_s": {k: round(sims[k]["wall_s"], 1) for k in ORDER},
        "note": rec}
    if gamma_rssnr:
        metrics["rssnr_gamma_mapping"] = {
            "value": gmap["k_db"], "threshold": None, "op": "record",
            "pass": True,
            **{k: gmap[k] for k in
               ("k_db", "k_phys_db", "k_minus_kphys_db", "g2_const_db",
                "g2_seg_db", "n_samples", "n_seg", "n_censored",
                "censored_floor_db", "seg_s_km", "med_sample_spacing_m",
                "att_db_per_km", "g2_pos_frac_seg",
                "implied_eff_att_db_per_km")},
            "snapshot_id": RSSNR_SNAPSHOT,
            "note": "median-anchored K (dB): |Gamma_bed|^2 = 2*A*H - RSSNR "
            "+ K; K - K_phys is the absolute-chain gap the anchoring "
            "absorbs (attenuation + surface-model uncertainty). " + rec}
        metrics["bed_brightness_correlation"] = {
            "value": round(float(np.mean(
                [corr_stats[k]["r_sim_rssnr_vs_measured"]
                 for k in ORDER])), 3),
            "threshold": None, "op": "record", "pass": True,
            "per_pass": corr_stats,
            "note": "KEY DELIVERABLE (acceptance): along-track Pearson r of "
            "the ~1 km-smoothed bed-window power profile (dB rel own "
            "surface peak) between sim and MEASURED, RSSNR-driven vs "
            "constant bed gamma (same picked-bed geometry). "
            "r_bedlayer_rssnr_vs_implied is the by-construction sanity "
            "check (bed-borne layer only -- geometry/speckle-limited); "
            "r_sim_rssnr_vs_implied uses the TOTAL field, whose bed window "
            "is surface-clutter-crowded at altitude (the study's own "
            "finding), so it is expected to degrade low->mid->high; "
            "r_implied_vs_measured is the data-only ceiling estimate. "
            + rec}

    config = {
        "case": case, "segment": segment, "n_traces": n_traces,
        "att_db_per_km": att, "surf_rough": bool(surf_rough),
        "margin_us": MARGIN_US, "post_bed_window_us": POST_BED_US,
        "chunk_m": CHUNK_M, "picked_bed": bool(picked_bed),
        "gamma_rssnr": bool(gamma_rssnr),
        "passes": {}, "measured_caveats": MEASURED_CAVEATS}
    if gamma_rssnr:
        config["rssnr_gamma"] = {
            k: gmap[k] for k in
            ("provenance", "k_db", "k_phys_db", "k_minus_kphys_db",
             "g2_const_db", "g2_seg_db", "n_samples", "n_seg", "n_censored",
             "censored_floor_db", "seg_s_km", "med_sample_spacing_m",
             "att_db_per_km", "g2_pos_frac_seg",
             "implied_eff_att_db_per_km", "note")}
        config["rssnr_gamma"]["interpolation"] = (
            "linear in anchor along-track s (np.interp, edge-clamped), "
            "cross-track constant; H(x) from the DATASET's surface/bed "
            "twtts (self-consistent with its RSSNR), not the DEM")
        config["rssnr_gamma"]["shared_field"] = (
            "ONE anchor-derived gamma field applied identically to all "
            "three passes (per-pass fields would confound the altitude "
            "comparison)")
    if picked_bed:
        config["picked_bed_reference"] = {
            k: ref[k] for k in ("pass", "frames", "eps_ice", "n",
                                "line_len_km", "gap_frac_line")}
        config["picked_bed_reference"]["why_low_pass"] = (
            "cleanest bed of the triplet: scout registration sigma 2.45 m vs "
            "10.80/10.92 m (mid/high) and measured mid-column/bed-peak "
            "-36.7 dB vs -17.7/-16.1 dB, i.e. at 442 m the bed echo stands "
            "~20 dB clear of the clutter the high passes' picks sit in; it "
            "is also the anchor line's own flight. ONE reference pass is "
            "applied identically to all three simulations -- never per-pass "
            "beds.")
    for key in ORDER:
        p, s = preps[key], sims[key]
        config["passes"][key] = {
            "parts": [[fid, list(sl)] for fid, sl in p["parts"]],
            "reversed": p["rev"], "roll_note": p["roll_note"],
            "param_frame": PASSES[key]["param_frame"],
            "n_traces_measured": int(len(p["surf"])),
            "n_traces_sim": int(len(p["idx"])),
            "agl_med_m": round(p["h_med"], 0),
            "reach": {k: round(v, 1) if isinstance(v, float) else v
                      for k, v in p["reach"].items()},
            "facet_spacing_m": round(p["spacing"], 3),
            "facets_per_interface_per_chunk": s["facets_per_chunk"],
            "n_chunks": s["n_chunks"], "wall_s": round(s["wall_s"], 1),
            "oversample": p["oversample"],
            "n_samples_sim": p["rc_sim"].n_samples,
            "dt_ns": round(p["dt"] * 1e9, 4),
            "window_modeled": p["window"], "window_note": p["win_note"],
            "dropped_power_fraction": s["dropped_power_fraction"],
            "surf_fill_frac": p["aux"]["surf_fill"],
            "bed_clamp_frac": p["aux"]["clamp_frac"],
            "picked_bed": p["aux"]["picked_bed"]}
    if segment == "pilot":
        config["full_projection"] = {
            k: {"wall_s_projected": round(sims[k]["wall_s"] * 5.0, 1),
                "basis": "5x pilot wall (50/10 km at fixed trace spacing; "
                "5 chunks of the pilot's exact geometry; per-chunk JAX "
                "recompile risk if chunk shapes differ -- pilot wall "
                "already includes one compile)"} for k in ORDER}
        config["full_projection"]["total_s"] = round(
            5.0 * sum(s["wall_s"] for s in sims.values()), 1)

    notes = (
        "Basal-clutter altitude triplet (claude_notes/basal_clutter_scout"
        ".md): three 2016_Antarctica_DC8 flights of the same grounded 148.5 "
        "km line at 442/9150/10684 m AGL, identical 190 MHz/50 MHz/hann/"
        "20.202 ns systems; measured mid-column clutter is ~20 dB stronger "
        "at altitude. COHERENT SURFACE+BED ONLY (no firn/internal layers by "
        "design): the study asks whether surface+bed geometric clutter "
        "reproduces the altitude trend, and the per-interface field "
        "decomposition identifies which interface supplies it. Cross-track "
        "reach derived per pass out to nadir-bed delay + "
        f"{MARGIN_US:.0f} us for both interfaces (bed reach includes Snell "
        "refraction); reversed high passes' roll negated; per-pass surface "
        "registration; BedMachine 500 m texture caveat applies. "
        + MEASURED_CAVEATS
        + (" PICKED BED: " + PICKED_BED_NOTE if picked_bed else "")
        + (" RSSNR GAMMA: " + RSSNR_GAMMA_NOTE if gamma_rssnr else ""))
    doc = {"case": case, "group": "xOPR clutter",
           "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "metrics": metrics, "notes": notes}
    (out / "metrics.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out / "run_config.json").write_text(json.dumps(config, indent=1) + "\n")

    figs = [fig_radargrams(out, preps, analyses, segment),
            fig_decomposition(out, preps, analyses)]
    if gamma_rssnr:
        figs.insert(0, fig_bed_brightness(out, preps, corr_series,
                                          corr_stats, segment))
    if make_report:
        _report(out, case, config, metrics, notes, figs)
    ver = VER_ROOT / case
    ver.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / "metrics.json", ver / "metrics.json")
    for f in figs:
        shutil.copy2(f, ver / f.name)
    print("clutter (midcol rel surf, meas/sim dB): " + " | ".join(
        f"{k}: {analyses[k]['meas']['midcol_rel_surf_db']:+.1f}/"
        f"{analyses[k]['sim']['midcol_rel_surf_db']:+.1f} "
        f"[{analyses[k]['verdict']}]" for k in ORDER), flush=True)
    return metrics, config, out


def _report(out, case, config, metrics, notes, figs):
    def b64(fp):
        return base64.b64encode(Path(fp).read_bytes()).decode()

    css = ("body{font-family:system-ui,Arial,sans-serif;margin:2rem;"
           "max-width:1250px}table{border-collapse:collapse;margin:1rem 0;"
           "font-size:.82rem}th,td{border:1px solid #ccc;padding:.3rem .5rem}"
           "th{background:#f0f0f0}img{max-width:100%;border:1px solid #ddd}"
           ".note{background:#f6f6f6;border-left:3px solid #bbb;"
           "padding:.6rem 1rem}td.pass{background:#c8f7c5}"
           "td.fail{background:#f7c5c5}")
    rows = "".join(
        f"<tr><th>{html.escape(k)}</th>"
        f"<td class='{'pass' if e.get('pass') else 'fail'}'>"
        f"{e.get('value'):.4g}</td>"
        f"<td>{html.escape(e.get('note', '')[:420])}</td></tr>"
        for k, e in metrics.items())
    figs_html = "".join(
        f"<h3>{html.escape(Path(f).stem)}</h3>"
        f"<img src='data:image/png;base64,{b64(f)}'>" for f in figs)
    body = f"""
<h1>Basal-clutter altitude triplet ({html.escape(config['segment'])})</h1>
<p class="note">{html.escape(notes)}</p>
{figs_html}
<h2>Metrics</h2>
<table><tr><th>metric</th><th>value</th><th>note</th></tr>{rows}</table>
<h2>Configuration</h2>
<pre>{html.escape(json.dumps(config, indent=1))}</pre>
"""
    (out / "report.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{case}</title><style>{css}</style></head>"
        f"<body>{body}</body></html>")
    print(f"wrote {out / 'report.html'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segment", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--n-traces", type=int, default=None,
                    help=f"sim traces (default {N_TRACES_PILOT} pilot / "
                    f"{N_TRACES_FULL} full)")
    ap.add_argument("--att", type=float, default=rac.ATT_DB_PER_KM,
                    help="one-way ice attenuation dB/km (default the b26/"
                    "altitude 15; run_cross_season calibrated an EFFECTIVE "
                    "31 on a different West Antarctic line -- affects only "
                    "bed-borne levels, not the surface-borne geometry)")
    ap.add_argument("--smooth-surface", action="store_true",
                    help="disable the representative sub-facet surface "
                    "roughness (default ON: off-nadir surface scattering is "
                    "central to this study)")
    ap.add_argument("--picked-bed", action="store_true",
                    help="use the radar-picked bed (LOW pass 20161105_05_"
                    "005-007, applied identically to all three passes) as an "
                    "along-track residual on BedMachine, preserving "
                    "BedMachine's cross-track relief; outputs and cached "
                    f"runs get the {PBED_TAG} suffix")
    ap.add_argument("--gamma-from-rssnr", action="store_true",
                    help="drive the bed reflectivity along-track from the "
                    "required-surface-SNR dataset (anchor line, pinned "
                    f"icechunk snapshot {RSSNR_SNAPSHOT}): |Gamma|^2 = "
                    "2*A*H - RSSNR + K, median-anchored K, one shared field "
                    "for all passes; adds the acceptance analysis vs the "
                    "constant-gamma companion run; outputs/caches get the "
                    f"{GRSSNR_TAG} suffix")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(segment=args.segment, n_traces=args.n_traces, att=args.att,
        surf_rough=not args.smooth_surface, out_root=args.out,
        force=args.force, picked_bed=args.picked_bed,
        gamma_rssnr=args.gamma_from_rssnr)


if __name__ == "__main__":
    main()
