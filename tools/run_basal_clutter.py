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
smoother/weaker in fine texture than measured (recorded, not tuned away);
params from each pass's own cached param frame; identical 20.202 ns lattice
across passes (shared surface-referenced fast-time comparison).

Machinery reused from tools/run_altitude_comparison.py: param loading,
window mapping, alias-safe oversampling, REMA+BedMachine scene building,
cached runs, facet spacing, surface gate. Runs are chunked ~10 km along
track so the 50 km segment projects ~linearly from the 10 km pilot.

Run:  uv run python tools/run_basal_clutter.py                # 10 km pilot
      uv run python tools/run_basal_clutter.py --segment full # 50 km (STOP:
      report pilot timings first; full run only on explicit go-ahead)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_altitude_comparison as rac  # noqa: E402  shared machinery
from run_opr_comparison import _db  # noqa: E402

from soundersim.config import (AntennaConfig, DemInterface, FacetConfig,  # noqa: E402
                               Medium, RadarConfig, RoughnessConfig, SimConfig,
                               WaveformConfig)
from soundersim.opr import load_bottom_pick, load_frame  # noqa: E402

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


def prep_pass(key, segment, n_traces):
    """Slice (+reverse) the pass's frames onto the common window, derive the
    reach and grids, and build the base scene (REMA + BedMachine, cached)."""
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


def chunk_scene(base, rows, ct):
    """MultilayerScene for one chunk: DEM stack cropped to the chunk traces'
    bbox padded by ct + 100 m (every trace keeps full +-ct coverage in every
    direction), nav/roll subset. The rac.crop_scene pattern + trace subset."""
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
        scene = chunk_scene(p["base"], rows, p["reach"]["ct_m"])
        rid = (f"{p['key']}_{p['segment']}_c{ci:02d}"
               + ("_srough" if surf_rough else "")
               + (f"_att{att:g}" if att != rac.ATT_DB_PER_KM else ""))
        meta = {"season": SEASON, "pass": p["key"], "segment": p["segment"],
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
    for name, Pl in (("surface", Ps), ("bed", Pb)):
        mid = _wmean(Pl, tw, dtf, t_s + MID_LO_US * 1e-6,
                     t_b - MID_HI_US * 1e-6)
        bed = _wmean(Pl, tw, dtf, t_b - BED_LO_US * 1e-6,
                     t_b + BED_HI_US * 1e-6)
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
    return {"gate": gate, "sim": clean, "meas": cleanm,
            "decomposition": {k: {kk: round(vv, 2) for kk, vv in v.items()}
                              for k, v in dec.items()},
            "verdict": verdict, "floor_db": round(floor_db, 2),
            "meas_noise_limited": noise_limited,
            "bed_delay_med_us": round(float(np.nanmedian(
                (p["bot"] - p["surf"]))) * 1e6, 2),
            "profs": profs, "P": P, "t_s": t_s, "meas_arr": meas}


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
        surf_rough=True, out_root=None, force=False, make_report=True):
    n_traces = n_traces or (N_TRACES_PILOT if segment == "pilot"
                            else N_TRACES_FULL)
    out = Path(out_root or OUT_DEFAULT) / segment
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"
    case = f"{CASE_PREFIX}_{segment}"
    preps, sims, analyses = {}, {}, {}
    for key in ORDER:
        print(f"== {key} ({segment}) ==", flush=True)
        p = prep_pass(key, segment, n_traces)
        print(f"  reach: surface {p['reach']['surface_reach_m']:.0f} m, bed "
              f"{p['reach']['bed_reach_m']:.0f} m -> ct "
              f"±{p['reach']['ct_m']:.0f} m; spacing {p['spacing']:.2f} m; "
              f"n_samples_sim {p['rc_sim'].n_samples}", flush=True)
        preps[key] = p
        sims[key] = simulate_pass(p, runs_dir, att, surf_rough, force)
        analyses[key] = analyze_pass(p, sims[key])

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

    config = {
        "case": case, "segment": segment, "n_traces": n_traces,
        "att_db_per_km": att, "surf_rough": bool(surf_rough),
        "margin_us": MARGIN_US, "post_bed_window_us": POST_BED_US,
        "chunk_m": CHUNK_M,
        "passes": {}, "measured_caveats": MEASURED_CAVEATS}
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
            "bed_clamp_frac": p["aux"]["clamp_frac"]}
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
        + MEASURED_CAVEATS)
    doc = {"case": case, "group": "xOPR clutter",
           "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "metrics": metrics, "notes": notes}
    (out / "metrics.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out / "run_config.json").write_text(json.dumps(config, indent=1) + "\n")

    figs = [fig_radargrams(out, preps, analyses, segment),
            fig_decomposition(out, preps, analyses)]
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
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(segment=args.segment, n_traces=args.n_traces, att=args.att,
        surf_rough=not args.smooth_surface, out_root=args.out,
        force=args.force)


if __name__ == "__main__":
    main()
