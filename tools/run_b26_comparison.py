"""Measured-vs-simulated comparison at the B26 firn core (ngt37C95.2).

OPR frame 2019_Greenland_P3 / 20190418_01_009 passes 10 m from the B26 core
(77.2533 N, 49.2167 W; claude_notes/firn_core_flightlines.md). This tool runs
the coherent simulator on a ~10 km sub-segment centered on the closest
approach and compares four aligned radargram panels on shared twtt /
along-track axes:

  1. measured CSARP_standard (SAR-FOCUSED: f-k SAR + 11-look) with
     Surface/Bottom picks + B26 marker;
  2. measured CSARP_qlook (UNFOCUSED: pulse compression + presums only) --
     the like-for-like target for our unfocused per-trace sims;
  3. simulated coherent, surface + BedMachine bed only (M24 machinery:
     ArcticDEM 32 m surface, BedMachine bed, MCoRDS chirp + 7-element array);
  4+. simulated, surface + N firn layers + bed (equal + random placements,
      equal-placement runs whose INTERNAL layer interfaces carry the measured
      C&S 2020 Fig. 11 sub-facet roughness, and an equal-placement run whose
      layer permittivities are SYNTHETIC effective contrasts reproducing the
      full-resolution density profile's per-segment reflectivity -- H1);

plus a nadir depth-power profile at the closest-approach trace (both measured
products vs the sims, upper ~200 m, B26-style) and the run-configuration table.

The two measured products share the frame's fast-time grid EXACTLY (t0 =
0.2667 us, dt = 16.667 ns, 3044 samples) but differ in along-track posting
(standard 3335 traces / ~14.7 m vs qlook 1265 traces / ~39 m) and in absolute
gain (qlook runs ~10 dB hotter here: different presum/multilook normalization).
Every depth profile is normalized to its OWN surface peak, so the product gain
cancels; the radargram panels get per-product colour limits (99.5th pct).

Instrument model: the SEASON'S OWN MCoRDS parameters, read from the frame's
param_records/param_sar/param_array structs in the CSARP_standard .mat
(provenance outputs/cache/mcords_2019P3_params.json -- NOT the 2017 values;
the 2019 product grid is dt = 16.667 ns, not 33.333 ns). Chirp 180-210 MHz
(f0 195 MHz, B 30 MHz), hann compression window (ft_wind = @hanning), pulse =
the longest/bed waveform (10 us), 7-element 0.5-lambda cross-track array,
roll from nav.

Alias-free fast-time grid (the firn findings' alias rule): simulate at
dt_frame/OVERSAMPLE = 16.667/4 = 4.167 ns -- |f0 - round(f0*dt)/dt| = 45 MHz
> B/2 = 15 MHz, simulate()'s in-band-alias warning asserted SILENT on every
call -- then decimate [::4] exactly back onto the frame twtt grid (the M24
convention; t0 anchored on a frame bin).

Firn layers: OFFSET interfaces of the ArcticDEM surface (surface - depth_i),
depths + per-layer permittivities from the B26 point-sampled pipeline
(run_firn_investigation: equal placement over the core's 1-119.66 m range,
Kovacs eps(rho) point-sampled from the 0.1 m-smoothed density profile).
Because off-nadir firn returns are sinc-suppressed, the firn CONTRIBUTION is
computed on a NARROWER cross-track strip (+-CT_FIRN) and field-summed with
the wide (+-CT_WIDE) surface+bed run -- the kernel and the pulse convolution
are linear, and the firn run's own surface layer (layer 0) is EXCLUDED from
the sum so the surface field is not double-counted. The strip is cropped from
the wide scene's DEM in ALONG-TRACK CHUNKS on the same facet lattice
(firn_scenes/facet_spacing docstrings): one bbox around the whole diagonal
segment would carry ~5x the strip area. The seam is verified
numerically: the firn run's layer-0 field, scaled by the surface-gamma ratio
(air->firn0 vs air->ice), must agree with the wide run's surface-layer field
in the early post-surface window where the narrow strip covers all arrivals.

Compute budget: HARD CEILING ~100 min total simulation wall time. A 1-trace
N=20 pilot (plus a 1-trace wide pilot) is run first and extrapolated; if the
projection exceeds the budget the configuration shrinks IN ORDER: traces
100 -> 60, firn strip +-600 -> +-400 m, along-track 10 -> 7 km (recorded).

Deliverables: outputs/b26_comparison/ (self-contained report.html, figures,
metrics.json with group "xOPR clutter" / case "b26_comparison", resumable
runs/ cache) and a copy of metrics.json + figures under
outputs/verification/b26_comparison/ so tools/make_report.py picks it up.

Honesty notes (carried into the report): (1) 32 m DEM posting -> statistical
(speckle/envelope), not phase-deterministic; (2) ~11 m facets -> recorded LPA
nadir-error estimate; (3) equal-placement point-sampled layers -> morphology
comparison, not calibrated absolute levels (see the firn findings'
random-vs-equal caveat); (4) CSARP_standard is SAR-processed
(f-k SAR + 11-look multilook) while the sims are unfocused per-trace raw --
CSARP_qlook is carried alongside precisely to bound how much of the gap that
processing asymmetry explains; compare structure/relative levels, not
resolution.

Run: uv run python tools/run_b26_comparison.py            # pilot + runs + report
     uv run python tools/run_b26_comparison.py --no-pilot  # add runs to an
                                # existing outputs/b26_comparison (reuses its
                                # recorded SCENE config; no shrink risk)
     uv run python tools/run_b26_comparison.py --only firn_N40_h1eff
                                # simulate ONLY these keys; every other run is
                                # assembled from its cached npz AS-IS (stale
                                # ones are flagged in run_provenance), and a
                                # missing one is an error. Implies --no-pilot
     uv run python tools/run_b26_comparison.py --report-only
"""

import argparse
import base64
import dataclasses
import datetime
import html
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_firn_investigation as rfi  # noqa: E402  B26 density->eps pipeline
import run_opr_coherent_bed as rocb  # noqa: E402  M24 frame+DEM+bed machinery
from run_opr_comparison import _db  # noqa: E402

from soundersim import firn  # noqa: E402
from soundersim.config import (AntennaConfig, DemInterface, FacetConfig,  # noqa: E402
                               Medium, RadarConfig, SimConfig, WaveformConfig)
from soundersim.opr import (CACHE_DIR, fetch_bedmachine_window,  # noqa: E402
                            fill_nodata_nearest, frame_scene, load_bottom_pick,
                            load_frame, resample_to_grid)
from affine import Affine  # noqa: E402
from soundersim.physics import fresnel_normal  # noqa: E402
from soundersim.simulate import _joint_pad_to, simulate  # noqa: E402
from soundersim.synthetic import MultilayerScene, SyntheticScene  # noqa: E402

C = 299792458.0
SEASON, FRAME_ID = "2019_Greenland_P3", "20190418_01_009"
B26_LATLON = (77.2533, -49.2167)  # ngt37C95.2 (PANGAEA header)
OUT_DEFAULT = ROOT / "outputs" / "b26_comparison"
VER_OUT = ROOT / "outputs" / "verification" / "b26_comparison"
MAT_SRC = CACHE_DIR / f"Data_{FRAME_ID}_source.mat"
MAT_URL = (f"https://data.cresis.ku.edu/data/rds/{SEASON}/CSARP_standard/"
           f"20190418_01/Data_{FRAME_ID}.mat")
PARAMS_JSON = CACHE_DIR / "mcords_2019P3_params.json"

# --- default configuration (pilot may shrink it, in this order) -------------
ALONG_M = 10_000.0            # along-track sub-segment centered on B26
N_TRACES = 100                # evenly sampled sim traces
CT_WIDE = 3000.0              # surface+bed cross-track reach (m)
CT_FIRN = 600.0               # firn-strip cross-track reach (m)
LAYER_COUNTS = (10, 20, 40, 80)
# Random layer placements (n, seed): tests the equal-placement hypothesis for
# the residual mid-band gap (firn findings: random placements yield materially
# stronger returns). N=40 = the converged-mid-band, affordable point.
RANDOM_RUNS = ((40, 0), (40, 1), (40, 2))
# Rough-layer runs (n_layers, inversion source): equal-placement N=40 exactly
# like firn_N40, but every INTERNAL layer interface carries the measured
# sub-facet roughness of Culberg & Schroeder 2020 Fig. 11 (docs/roughness.md,
# Gerekos 2023 rough-facet response). The air-firn SURFACE stays smooth (as
# does the bed, which lives in the wide run), so the seam check and the
# own-surface-peak profile normalization stay comparable across runs. Tests
# the remaining hypothesis for the mid-band (20-70 m) deficit: diffuse
# scattering from sub-wavelength layer roughness.
ROUGH_RUNS = ((40, "mcords"), (40, "ar"))
# Effective-contrast runs (hypothesis H1, claude_notes/b26_gap_hypotheses.md):
# equal-placement N=40 exactly like firn_N40, but the layer permittivities are
# SYNTHETIC -- built so each interface's plain Fresnel contrast equals the
# transfer-matrix aggregate |r| of the RAW full-resolution (1 mm) B26 density
# profile over that layer's segment (effective_contrast_eps). Tests whether the
# ~15 dB mid-band deficit is point-sampling discarding the 0.1-0.5 m Bragg-scale
# density strata: the 1-D 20-70 m band level goes -28.3 -> -17.2 dB rel surface.
# N-LADDER: the segment-aggregated BAND LEVEL is N-independent by construction
# (the segments tile the profile, so reflectivity is conserved) -- the 1-D scan
# gives -16.5 / -19.4 / -18.9 / -17.2 dB in 20-70 m at N = 5 / 10 / 20 / 40 vs
# -16.7 for the full-res profile, i.e. scatter, not trend. So the ladder's
# question is profile SHAPE, not level: the correlation against the measured
# depth profile should climb while the layer spacing (119.7/N m) is coarser
# than the ~4.4 m in-firn range cell and plateau once it is finer (N ~ 27).
EFF_RUNS = (5, 10, 20, 40)
EFF_METHOD = "tmm_segment_aggregate_v1"
ROUGH_COST_FACTOR = 1.05      # rough/smooth simulate() wall ratio: MEASURED
# 1.017 on a 2-trace N=40 real crop (85.1 s -> 86.5 s steady, 44469 facets);
# sigma/lambda_firn ~ 0.05 needs only the 10-term D_Phi series, and the joint
# refraction solve dominates the cost. 1.05 keeps a small margin.
SHRINK_STEPS = (("n_traces", 60), ("ct_firn", 400.0), ("along_m", 7000.0))
BUDGET_S = 12 * 3600.0        # hard ceiling, total simulation wall time
# (raised from 100 min when N=40/80 were added, 2026-07-09, user-authorized:
# long runs OK; the shrink loop must NOT trigger, or the scene configuration
# would diverge from the cached N=10/20 runs and break panel comparability)

OVERSAMPLE = 4                # dt_sim = dt_frame/4 = 4.167 ns (alias 45 MHz)
EPS_ICE, EPS_BED = 3.17, 8.0  # M24 media (bed eps recorded only)
ATT_DB_PER_KM = 15.0          # one-way, constant (M24 value; recorded)
BETA = 0.5                    # facet Fresnel criterion L <= beta*sqrt(lam*r)
N_ELEMENTS, SPACING_LAM = 7, 0.5
GATE_BINS = 5.0               # frame bins (16.667 ns), median, offset-removed
PRE_SURF_US, POST_BED_US = 0.8, 2.0  # twtt window margins around the picks
SEAM_WIN_US = 1.5             # seam-check window after the surface peak
PROFILE_MAX_M = 200.0         # nadir depth-power comparison depth range
BAND_EDGES_M = (5.0, 20.0, 60.0, 120.0)  # firn band-level diagnostics
# Extra (overlapping) bands for the focused-vs-unfocused diagnostic: the
# mid-band where the sims sit below the measured, and the deep firn band.
EXTRA_BANDS = ((20.0, 70.0), (80.0, 120.0))
GAP_BAND = "20-70m"            # headline band for the band-delta metric
MEAS = {"standard": "measured", "qlook": "measured_qlook"}  # profile keys


# ========================================================================
# season parameters from the frame's own product file (M24 method)
# ========================================================================
def mcords_2019_params():
    """Read/caches the 2019_Greenland_P3 MCoRDS params from the frame's own
    param structs (param_records/param_sar/param_array in the CSARP_standard
    .mat). Returns the provenance dict; writes PARAMS_JSON on first call."""
    if PARAMS_JSON.exists():
        return json.loads(PARAMS_JSON.read_text())
    if not MAT_SRC.exists():
        import urllib.request
        print(f"downloading {MAT_URL} ...", flush=True)
        urllib.request.urlretrieve(MAT_URL, MAT_SRC)

    import h5py

    def deref(f, r):
        return f[h5py.h5r.get_name(r, f.id)]

    def tostr(v):
        return "".join(chr(int(c)) for c in np.asarray(v).ravel())

    with h5py.File(MAT_SRC, "r") as f:
        def vals(ds):
            """Numeric dataset values, dereferencing MATLAB object refs."""
            v = ds[()]
            if v.dtype == object:
                v = np.concatenate([np.asarray(deref(f, r)[()]).ravel()
                                    for r in v.ravel()])
            return np.asarray(v, np.float64).ravel()

        wfs = f["param_records/radar/wfs"]
        f0 = np.unique(vals(wfs["f0"])).tolist()
        f1 = np.unique(vals(wfs["f1"])).tolist()
        tpd = np.unique(vals(wfs["Tpd"])).tolist()
        tukey = float(np.unique(vals(wfs["tukey"]))[0])
        prf = float(vals(f["param_records/radar/prf"])[0])
        fs_raw = float(vals(f["param_records/radar/fs"])[0])
        try:  # ft_wind is a MATLAB function handle (group)
            node = deref(f, f["param_sar/radar/wfs/ft_wind"][()].ravel()[0])
            ft_wind = tostr(node["function_handle"]["function"][()])
        except Exception:
            ft_wind = "hanning (deref failed; CReSIS readme default)"
        try:
            rx = [np.asarray(deref(f, r)[()]).ravel().tolist()
                  for r in f["param_sar/radar/wfs/rx_paths"][()].ravel()][0]
        except Exception:
            rx = None
        sar = f["param_sar/sar"]
        sigma_x = float(np.asarray(sar["sigma_x"][()]).ravel()[0])
        sar_type = tostr(sar["sar_type"][()])
        arr = f["param_array/array"]
        line_rng = np.asarray(arr["line_rng"][()]).ravel().tolist()
        dline = float(np.asarray(arr["dline"][()]).ravel()[0])
        img_comb = np.asarray(arr["img_comb"][()]).ravel().tolist()

    doc = {
        "purpose": ("B26 comparison parameter provenance: MCoRDS on P-3, "
                    "2019_Greenland_P3, read from the frame's own param "
                    "structs (M24 method; do NOT assume the 2017 values)"),
        "source": f"outputs/cache/{MAT_SRC.name} (downloaded from {MAT_URL})",
        "waveform": {
            "f0_f1_Hz": [f0, f1],
            "center_frequency_Hz": float((f0[0] + f1[0]) / 2.0),
            "bandwidth_Hz": float(f1[0] - f0[0]),
            "pulse_lengths_s": tpd,
            "bed_waveform_pulse_length_s": float(max(tpd)),
            "tukey_time_window": tukey,
            "pulse_compression_freq_window": ft_wind + " (param_sar ft_wind)",
            "prf_Hz": prf, "fs_raw_Hz": fs_raw,
            "product_dt_s": 16.667e-9,
            "img_comb_s": img_comb,
        },
        "antenna": {
            "rx_paths": rx,
            "modeled_as": ("uniform unsteered 7-element 0.5-lambda "
                           "cross-track array (P-3 center array), "
                           "roll_source=nav; tx taper / hanning array window "
                           "recorded-but-unmodeled (as M24)"),
        },
        "processing_CSARP_standard": {
            "sar": f"{sar_type} SAR, sigma_x = {sigma_x} m SLC",
            "multilook": f"line_rng {line_rng[0]:.0f}..{line_rng[-1]:.0f} = "
                         f"{len(line_rng)} looks, dline {dline:.0f} "
                         f"-> ~15 m posting",
        },
    }
    PARAMS_JSON.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def load_qlook_frame():
    """CSARP_qlook (UNFOCUSED quick-look: pulse compression + presums, no SAR
    focusing) for the same frame -- the like-for-like measured target for our
    unfocused sims. Returns None if it cannot be loaded (diagnostic only; the
    rest of the comparison does not depend on it)."""
    try:
        return load_frame(SEASON, FRAME_ID, data_product="CSARP_qlook")
    except Exception as e:  # offline / product missing for this season
        print(f"  [warn] CSARP_qlook unavailable: {type(e).__name__}: {e}",
              flush=True)
        return None


# ========================================================================
# geometry: sub-segment centered on the B26 closest approach
# ========================================================================
def _lonlat(frame):
    lon = np.asarray(frame.Longitude.values, np.float64)
    return (np.asarray(frame.Latitude.values, np.float64),
            np.where(lon > 180.0, lon - 360.0, lon))


def sub_frame(frame, along_m):
    """(fsub, info): frame sliced to ~along_m centered on the B26 closest
    approach; info has the closest trace (global/local), distance, s-axis."""
    lat, lon = _lonlat(frame)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    px, py = tr.transform(lon, lat)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    bx, by = tr.transform(B26_LATLON[1], B26_LATLON[0])
    dist = np.hypot(px - bx, py - by)
    i0 = int(np.argmin(dist))
    a = int(np.searchsorted(s, s[i0] - along_m / 2.0))
    b = int(np.searchsorted(s, s[i0] + along_m / 2.0))
    fsub = frame.isel(slow_time=slice(a, b))
    info = {"i0_global": i0, "i0_local": i0 - a, "closest_m": float(dist[i0]),
            "s_rel_m": s[a:b] - s[i0], "track_len_m": float(s[b - 1] - s[a]),
            "slice": (a, b), "b26_xy": (bx, by)}
    return fsub, info


def radar_grids(frame, fsub, bot_sub, params):
    """(rc_sim, rc_frame, b0): alias-free simulation grid (dt_frame/OVERSAMPLE,
    t0 anchored on frame bin b0) covering [min surface pick - PRE_SURF_US,
    max bottom pick + POST_BED_US], and the matching frame-grid subwindow."""
    tw = frame.twtt.values
    dt = float((tw[-1] - tw[0]) / (len(tw) - 1))
    t0f = float(tw[0])
    surf = fsub.Surface.values
    lo = max(t0f, float(np.nanmin(surf)) - PRE_SURF_US * 1e-6)
    hi = min(float(tw[-1]), float(np.nanmax(bot_sub)) + POST_BED_US * 1e-6)
    b0 = int(np.floor((lo - t0f) / dt))
    b1 = int(np.ceil((hi - t0f) / dt))
    nb = b1 - b0 + 1
    wf_doc = params["waveform"]
    f0 = wf_doc["center_frequency_Hz"]
    wf = WaveformConfig(kind="chirp", bandwidth=wf_doc["bandwidth_Hz"],
                        pulse_length=wf_doc["bed_waveform_pulse_length_s"],
                        window="hann")
    ant = AntennaConfig(kind="array", n_elements=N_ELEMENTS,
                        spacing_lam=SPACING_LAM, roll_source="nav")
    rc_sim = RadarConfig(dt=dt / OVERSAMPLE, n_samples=OVERSAMPLE * (nb - 1) + 1,
                         t0=t0f + b0 * dt, f0=f0, waveform=wf, antenna=ant)
    rc_frame = RadarConfig(dt=dt, n_samples=nb, t0=t0f + b0 * dt, f0=f0)
    return rc_sim, rc_frame, b0


def facet_spacing(rc_sim, r_min, thickness):
    """Single spacing = BETA * sqrt(lam_j * r_j) minimized over the stack: the
    surface (lam, r_min), the deepest firn layer (in-firn lam, r_min + z) and
    the bed (in-ice lam, r_min + thickness). The deepest firn layer binds.

    The result is SNAPPED DOWN to an integer divisor of the 32 m DEM posting
    (e.g. 11.3 -> 32/3 = 10.67 m): build_facets' bilinear subdivision then
    places facet vertices at fixed fractions of each DEM cell, so the firn
    strips (whole-cell crops of the wide window, see firn_scenes) tessellate
    on EXACTLY the same facet lattice as the wide run -- required for the
    field-sum seam check to be tessellation-noise-free."""
    lam = rc_sim.wavelength
    cands = [lam * r_min,
             lam / np.sqrt(rfi.point_eps(rfi.ZMAX)) * (r_min + rfi.ZMAX),
             lam / np.sqrt(EPS_ICE) * (r_min + thickness)]
    s = float(BETA * np.sqrt(min(cands)))
    return 32.0 / np.ceil(32.0 / s) if s < 32.0 else s


def _crop_period_cells(spacing):
    """DEM-cell period of the facet lattice: crops whose origin is a multiple
    of this reproduce the parent's facet positions (subdividing spacings snap
    to 32/k -> period 1; coarser spacings stride round(spacing/32) cells)."""
    return 1 if spacing < 32.0 else max(1, int(round(spacing / 32.0)))


# ========================================================================
# scenes
# ========================================================================
def wide_scene(fsub, n_traces, ct_dist):
    """Surface (ArcticDEM 32 m) + BedMachine bed MultilayerScene, +-ct_dist,
    the M24 _bed_scene structure (cache-first fetches)."""
    scene, info = frame_scene(fsub, n_traces=n_traces, ct_dist=ct_dist)
    idx = info["trace_idx"]
    lat, lon = _lonlat(fsub)
    lat, lon = lat[idx], lon[idx]
    bounds = (lon.min(), lat.min(), lon.max(), lat.max())
    bed_native, tr_b, crs_b, meta = fetch_bedmachine_window(
        bounds, info["region"], pad_m=ct_dist + 500.0)
    bed = resample_to_grid(bed_native, tr_b, crs_b, scene.dem.shape,
                           scene.transform, scene.crs)
    bed, bed_fill = fill_nodata_nearest(bed)
    clamp_frac = float((bed > scene.dem - 0.1).mean())
    bed = np.minimum(bed, scene.dem - 0.1).astype(np.float32)
    media = [Medium(name="air", eps_r=1.0),
             Medium(name="ice", eps_r=EPS_ICE,
                    attenuation_db_per_km=ATT_DB_PER_KM),
             Medium(name="bed", eps_r=EPS_BED)]
    ms = MultilayerScene(scene.name + "_b26bed", [scene.dem, bed],
                         scene.transform, scene.crs, scene.nav_llh, media,
                         {**scene.params, "bed_product": meta["product"],
                          "bed_version": meta["version"]})
    ms.nav_roll = scene.nav_roll
    aux = {"idx": idx, "bed_meta": meta, "bed_fill": bed_fill,
           "clamp_frac": clamp_frac, "surf_fill": info["fill_fraction"]}
    return ms, aux


def firn_scenes(wscene, ct_dist, spacing, n_chunks=None):
    """Narrow-strip scenes for the firn contribution, CROPPED from the wide
    scene's surface DEM in ALONG-TRACK CHUNKS.

    The crop shares the wide scene's 32 m lattice and values (so the
    field-sum seam is exact up to the extent difference; no extra network),
    and chunking keeps the strip area near the ideal track-following strip
    for a diagonal track: one bbox around the whole 10 km diagonal segment
    would carry ~5x the intended +-ct_dist strip area. Each chunk's DEM is
    its OWN traces' bbox padded by ct_dist + 100 m, so every trace keeps full
    +-ct_dist coverage in all directions (no chunk-boundary seam within the
    strip; beyond +-ct_dist the extents differ between neighboring chunks,
    which is the same sinc-suppressed region the strip already truncates).

    Returns a list of (scene, trace_row_indices).
    """
    n_traces = len(wscene.nav_llh)
    n_chunks = n_chunks or max(1, round(n_traces / 17))
    period = _crop_period_cells(spacing)
    tr = Transformer.from_crs("EPSG:4326", wscene.crs, always_xy=True)
    px, py = tr.transform(wscene.nav_llh[:, 1], wscene.nav_llh[:, 0])
    pad = ct_dist + 100.0
    ny, nx = wscene.dem.shape
    out = []
    for rows_idx in np.array_split(np.arange(n_traces), n_chunks):
        x, y = px[rows_idx], py[rows_idx]
        cols, rows = (~wscene.transform) * (
            np.array([x.min() - pad, x.max() + pad]),
            np.array([y.min() - pad, y.max() + pad]))
        # origins snapped to the facet-lattice period so the crop's facets
        # coincide with the wide window's (facet_spacing docstring)
        c0 = int(np.clip(np.floor(min(cols) / period) * period, 0, nx - 2))
        c1 = int(np.clip(np.ceil(max(cols)) + 1, c0 + 2, nx))
        r0 = int(np.clip(np.floor(min(rows) / period) * period, 0, ny - 2))
        r1 = int(np.clip(np.ceil(max(rows)) + 1, r0 + 2, ny))
        dem = np.ascontiguousarray(wscene.dem[r0:r1, c0:c1])
        tr_c = wscene.transform * Affine.translation(c0, r0)
        roll = getattr(wscene, "nav_roll", None)
        sc = SyntheticScene(
            f"{wscene.name}_firnstrip{rows_idx[0]}", dem, tr_c, wscene.crs,
            wscene.nav_llh[rows_idx],
            {**wscene.params, "ct_dist_firn": ct_dist},
            nav_roll=None if roll is None else np.asarray(roll)[rows_idx])
        out.append((sc, rows_idx))
    return out


def bed_cfg(rc_sim, spacing):
    return SimConfig(mode="coherent", split_sides=False, radar=rc_sim,
                     facets=FacetConfig(spacing=spacing),
                     media=[Medium(name="air", eps_r=1.0),
                            Medium(name="ice", eps_r=EPS_ICE,
                                   attenuation_db_per_km=ATT_DB_PER_KM),
                            Medium(name="bed", eps_r=EPS_BED)],
                     interfaces=[DemInterface(name="surface"),
                                 DemInterface(name="bed")])


def _rough_table(fname):
    """Column dict of a Culberg & Schroeder 2020 Fig. 11 digitization CSV
    (tests/fixtures/firn/, '#' comment block then a header row)."""
    lines = [ln for ln in (rfi.FIXDIR / fname).read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    cols = [c.strip() for c in lines[0].split(",")]
    data = np.array([[float(x) for x in ln.split(",")] for ln in lines[1:]])
    return dict(zip(cols, data.T))


def layer_roughness(depths, source):
    """(sigma_m, corr_length_m) arrays at ``depths`` for inversion ``source``
    ('ar' | 'mcords' | 'joint'), linearly interpolated from the C&S 2020
    Fig. 11 inverted layer-roughness profiles (0-90 m at 5 m posting) and
    CLAMPED to the profile ends outside that range (np.interp semantics) --
    the B26 stack runs to 119.7 m, deeper than the published inversion."""
    a, b = _rough_table("fig11a_rms_height.csv"), \
        _rough_table("fig11b_correlation_length.csv")
    d = np.asarray(depths, np.float64)
    return (np.interp(d, a["depth_m"], a[f"rms_height_{source}_m"]),
            np.interp(d, b["depth_m"], b[f"corr_length_{source}_m"]))


# Effective-contrast construction: promoted to soundersim.firn (FirnCore);
# this module keeps thin delegates on the B26 fixture so its public API (and
# the cached runs' meta keys) are unchanged. FirnCore replicates
# rfi.load_b26/point_eps exactly (0.1 m edge-normalized boxcar), so the
# delegates are byte-identical to the pre-refactor local implementations.
B26_CORE = firn.FirnCore(rfi.FIXDIR / "ngt37C95.2_density.tab")


def segment_reflectivity(depths, lam, complex_r=False):
    """B26 segment-aggregate TMM reflectivity (FirnCore.segment_reflectivity
    on the raw 1 mm fixture; the profile ABOVE depths[0] is what the air-firn
    surface interface already represents -- its aggregate |r| of -13.7 dB
    would demand a spurious eps ~3.9 super-ice reflector at 4 m)."""
    return B26_CORE.segment_reflectivity(depths, lam, complex_r)


def effective_contrast_eps(depths, lam):
    """H1 synthetic permittivities for the B26 core
    (FirnCore.effective_contrast_eps: plain Fresnel contrasts reproduce
    segment_reflectivity, firn0 point-sampled, sign tracks the Kovacs trend)."""
    return B26_CORE.effective_contrast_eps(depths, lam)


def firn_cfg(rc_sim, spacing, depths, rough=None, eps=None):
    """Coherent B26 firn stack config (the investigation's layered_cfg on the
    real surface DEM): offset interfaces at surface - depth_i, point-sampled
    Kovacs permittivities, substrate = eps(deepest + 1 m), every firn medium
    (and the substrate) attenuating at ATT_DB_PER_KM = 15 dB/km one-way -- the
    same constant the wide run's ice medium uses.

    This corrects the earlier "no attenuation in the firn media (< 0.2 dB
    one-way at 120 m for any plausible cold-firn value)" convention, which was
    ~10x low: 15 dB/km over 120 m is 1.8 dB ONE-way, 3.6 dB two-way, growing
    with depth and therefore biasing the deep firn band specifically. The
    analytic bracket in claude_notes/b26_gap_hypotheses.md (attenuation
    addendum) puts the optimal uniform value at 8-12 dB/km; 15 dB/km is adopted
    for consistency with the ice medium rather than tuned to the residual.

    ``rough`` = (sigma_m[], corr_length_m[]) per layer attaches Gerekos-2023
    sub-facet roughness to every INTERNAL layer interface (the air-firn
    surface stays smooth); None -> the exact smooth path.

    ``eps`` = len(depths)+1 permittivities (firn0..firn_{N-1}, substrate)
    replacing the point-sampled ones (effective_contrast_eps, H1); None -> the
    exact point-sampled path."""
    e = (np.array([rfi.point_eps(d) for d in depths]
                  + [rfi.point_eps(float(depths[-1]) + 1.0)])
         if eps is None else np.asarray(eps, np.float64))
    media, ifaces = firn.firn_stack(depths, e, ATT_DB_PER_KM, roughness=rough)
    return SimConfig(mode="coherent", split_sides=False, radar=rc_sim,
                     facets=FacetConfig(spacing=spacing), media=media,
                     interfaces=ifaces)


def _n_facets(dem_shape, spacing):
    """Subdivided facet count of one interface (rocb formula)."""
    ny, nx = dem_shape
    f = 32.0 / spacing
    nrv = max(2, int(round((ny - 1) * f)) + 1)
    ncv = max(2, int(round((nx - 1) * f)) + 1)
    return (nrv - 1) * (ncv - 1)


PILOT_TRACES = 2  # nav_to_frame needs >= 2 traces for the along-track axis


def _pilot_scene(scene, j):
    """Copy of ``scene`` with nav (and roll) sliced to PILOT_TRACES traces
    starting at ``j``."""
    sl = slice(j, j + PILOT_TRACES)
    roll = getattr(scene, "nav_roll", None)
    roll = None if roll is None else np.asarray(roll)[sl]
    if isinstance(scene, MultilayerScene):
        s = MultilayerScene(scene.name + "_pilot", scene.dems, scene.transform,
                            scene.crs, scene.nav_llh[sl], scene.media,
                            scene.params)
        s.nav_roll = roll
        return s
    return dataclasses.replace(scene, name=scene.name + "_pilot",
                               nav_llh=scene.nav_llh[sl], nav_roll=roll)


def _padwork(n_layers):
    """Relative refracted-solver work of an N-layer firn stack: sum over
    target layers j of the (power-of-two padded) crossing count."""
    return sum(1 if j == 1 else _joint_pad_to(j, n_layers)
               for j in range(1, n_layers + 1))


# ========================================================================
# cached simulation runs (resumable)
# ========================================================================
def _simulate_checked(scene, cfg):
    """simulate() with the in-band-alias warning asserted SILENT."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = simulate(scene, cfg)
    msgs = [str(w.message) for w in caught]
    if any("alias" in m for m in msgs):
        raise RuntimeError(f"in-band-alias warning fired: {msgs}")
    return ds, msgs


def run_sim(rid, scene_chunks, cfg, meta, runs_dir, force=False,
            allow_sim=True):
    """Cached simulate() over one or more (scene, trace_rows) chunks:
    decimated per-layer complex field + nadir_twtt + diagnostics assembled
    over all traces into runs/<rid>.npz/.json, keyed on ``meta`` (config
    identity). Returns (diag dict, arrays dict); diag["provenance"] is
    "simulated" / "cache" / "cache-stale".

    ``allow_sim=False`` (the --only flag) forbids simulating: the existing
    runs/<rid>.npz is returned AS-IS even when its recorded ``meta`` no longer
    matches the current configuration ("cache-stale" -- the point of the flag),
    and a missing one is an error rather than a silent multi-hour run."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    jp, npz_p = runs_dir / f"{rid}.json", runs_dir / f"{rid}.npz"
    key = json.dumps(meta, sort_keys=True)
    have = jp.exists() and npz_p.exists()
    cached = json.loads(jp.read_text()) if have else None
    usable = have and not force and cached.get("meta_key") == key
    if not allow_sim:
        if not have:
            raise RuntimeError(
                f"{rid}: excluded from simulation by --only but "
                f"{npz_p.relative_to(runs_dir.parent)} does not exist -- add "
                f"'{rid}' to --only to simulate it, or drop --only")
        cached["provenance"] = "cache" if usable else "cache-stale"
        print(f"  [only-{cached['provenance']}] {rid} "
              f"({cached['wall_s']:.1f} s recorded)", flush=True)
        return cached, dict(np.load(npz_p))
    if usable:
        cached["provenance"] = "cache"
        print(f"  [skip-exists] {rid} ({cached['wall_s']:.1f} s recorded)",
              flush=True)
        return cached, dict(np.load(npz_p))
    n_traces = sum(len(rows) for _, rows in scene_chunks)
    field = twtt = nadir = None
    drop_sum = tot_sum = None
    msgs_all, facets, wall = [], [], 0.0
    layers = None
    for scene, rows in scene_chunks:
        t = time.perf_counter()
        ds, msgs = _simulate_checked(scene, cfg)
        wall += time.perf_counter() - t
        msgs_all += msgs
        ds_dec = ds.isel(twtt=slice(None, None, OVERSAMPLE))
        f = np.asarray(ds_dec.field.values, np.complex64)  # (t, nb, L)
        if field is None:
            field = np.zeros((n_traces,) + f.shape[1:], np.complex64)
            nadir = np.zeros((n_traces, f.shape[-1]))
            twtt = ds_dec.twtt.values
            layers = [str(x) for x in np.asarray(ds.layer.values)]
            drop_sum = np.zeros(f.shape[-1])
            tot_sum = np.zeros(f.shape[-1])
        field[rows] = f
        nadir[rows] = np.asarray(ds.nadir_twtt.values, np.float64)
        drop = np.asarray(ds.dropped_power.values, np.float64)
        drop_sum += drop.sum(0)
        tot_sum += np.asarray(ds.power.values, np.float64).sum((0, 1)) \
            + drop.sum(0)
        facets.append(_n_facets(scene.dem.shape, cfg.facets.spacing))
    if not np.isfinite(field).all():
        # Fail before writing the cache: a multi-hour run that produced NaN
        # must not be recorded as done (and NaN spreads over the whole trace
        # through the pulse convolution, so the damage is total).
        bad = [layers[k] for k in range(field.shape[-1])
               if not np.isfinite(field[..., k]).all()]
        raise RuntimeError(f"{rid}: non-finite field in layers {bad}")
    drop_frac = (drop_sum / np.maximum(tot_sum, 1e-300)).tolist()
    diag = {"rid": rid, "wall_s": round(wall, 2), "meta_key": key,
            "meta": meta, "warnings": msgs_all, "alias_warning_fired": False,
            "dropped_power_fraction": drop_frac, "layers": layers,
            "n_chunks": len(scene_chunks),
            "n_facets_per_interface_per_chunk": facets}
    arrs = dict(field=field, twtt=twtt, nadir_twtt=nadir)
    np.savez_compressed(npz_p, **arrs)
    jp.write_text(json.dumps(diag, indent=1) + "\n")
    print(f"  [ok] {rid}  {wall:.1f} s  chunks {len(scene_chunks)} "
          f"facets/iface/chunk ~{int(np.mean(facets))}", flush=True)
    return {**diag, "provenance": "simulated"}, arrs


# ========================================================================
# pilot + budget
# ========================================================================
def _firn_work(chunks, spacing):
    """Sum over chunks of n_traces * n_facets (the padwork-independent part
    of the firn cost model)."""
    return sum(len(rows) * _n_facets(sc.dem.shape, spacing)
               for sc, rows in chunks)


def pilot_and_budget(fsub, cfg_dict, rc_sim, spacing, out):
    """1-trace pilots (firn N_max chunk run + wide surface+bed), projection,
    and the shrink loop. Returns (cfg_dict, budget_log, resliced)."""
    n_max = max(cfg_dict["layer_counts"])
    log = {"pilot": {}, "projection_s": {}, "shrink_steps": [],
           "budget_s": BUDGET_S}

    def _build():
        ws, _ = wide_scene(fsub, cfg_dict["n_traces"], cfg_dict["ct_wide"])
        return ws, firn_scenes(ws, cfg_dict["ct_firn"], spacing)

    wscene, chunks = _build()
    depths = rfi.equal_depths(n_max)
    sc_mid, _ = chunks[len(chunks) // 2]
    jmid = len(sc_mid.nav_llh) // 2

    t = time.perf_counter()
    _simulate_checked(_pilot_scene(sc_mid, jmid),
                      firn_cfg(rc_sim, spacing, depths))
    t_first = time.perf_counter() - t
    t = time.perf_counter()
    _simulate_checked(_pilot_scene(sc_mid, jmid),
                      firn_cfg(rc_sim, spacing, depths))
    t_firn = time.perf_counter() - t

    t = time.perf_counter()
    _simulate_checked(_pilot_scene(wscene, 0), bed_cfg(rc_sim, spacing))
    t_wide1 = time.perf_counter() - t
    t = time.perf_counter()
    _simulate_checked(_pilot_scene(wscene, 0), bed_cfg(rc_sim, spacing))
    t_wide = time.perf_counter() - t

    nf_pilot = _n_facets(sc_mid.dem.shape, spacing)
    nf_wide = _n_facets(wscene.dem.shape, spacing)
    log["pilot"] = {"firn_first_s": round(t_first, 1),
                    "firn_steady_s": round(t_firn, 1),
                    "wide_first_s": round(t_wide1, 1),
                    "wide_steady_s": round(t_wide, 1),
                    "pilot_traces": PILOT_TRACES,
                    "n_facets_pilot_chunk": nf_pilot,
                    "n_facets_wide": nf_wide, "n_layers_pilot": n_max,
                    "n_chunks": len(chunks)}
    # per-(trace * facet [* padwork]) rates from the steady pilot calls
    rate_firn = t_firn / (PILOT_TRACES * nf_pilot * _padwork(n_max))
    rate_wide = t_wide / (PILOT_TRACES * nf_wide)
    compile_pad = (max(t_first - t_firn, 0.0)
                   + max(t_wide1 - t_wide, 0.0)) * (1 + len(chunks))

    def project(chunks_c, nf_wide_c, n_traces):
        work = _firn_work(chunks_c, spacing)
        proj = {"wide": rate_wide * nf_wide_c * n_traces}
        for n in cfg_dict["layer_counts"]:
            proj[f"firn_N{n}"] = rate_firn * work * _padwork(n)
        for n, s in cfg_dict.get("random_runs", ()):
            proj[f"firn_N{n}_s{s}"] = rate_firn * work * _padwork(n)
        for n, src in cfg_dict.get("rough_runs", ()):
            proj[f"firn_N{n}_rough_{src}"] = (rate_firn * work * _padwork(n)
                                              * ROUGH_COST_FACTOR)
        for n in cfg_dict.get("eff_runs", ()):  # same cost as the smooth run
            proj[f"firn_N{n}_h1eff"] = rate_firn * work * _padwork(n)
        proj["compile_pad_est"] = compile_pad
        proj["total"] = sum(proj.values())
        return proj

    proj = project(chunks, nf_wide, cfg_dict["n_traces"])
    log["projection_s"]["initial"] = {k: round(v, 1) for k, v in proj.items()}
    resliced = False
    for name, val in SHRINK_STEPS:
        if proj["total"] <= BUDGET_S:
            break
        ratio = val / cfg_dict[name]
        cfg_dict[name] = val
        log["shrink_steps"].append({name: val})
        if name in ("n_traces", "ct_firn"):
            wscene, chunks = _build()
            nf_wide = _n_facets(wscene.dem.shape, spacing)
        else:  # along_m: caller re-slices; scale the projection by the ratio
            resliced = True
            nf_wide = int(nf_wide * ratio)
        w_scale = ratio if name == "along_m" else 1.0
        proj = project(chunks, nf_wide, cfg_dict["n_traces"])
        if name == "along_m":
            for k in list(proj):
                if k.startswith("firn"):
                    proj[k] *= w_scale
            proj["total"] = sum(v for k, v in proj.items() if k != "total")
        log["projection_s"][f"after_{name}={val}"] = {
            k: round(v, 1) for k, v in proj.items()}
    log["projection_s"]["final"] = log["projection_s"][
        list(log["projection_s"])[-1]]
    (out / "budget_log.json").write_text(json.dumps(log, indent=1) + "\n")
    return cfg_dict, log, resliced


# ========================================================================
# analysis helpers
# ========================================================================
def leading_edge_gate(p_surf, spacing, rc_frame, surf_pick):
    """M24-style surface gate: smoothed surface-layer leading edge vs the
    Surface pick, constant offset removed, FRAME bins."""
    dt, t0 = rc_frame.dt, rc_frame.t0
    range_bin = C * dt / 2.0
    w = max(1, int(round(spacing / range_bin)))
    sm = uniform_filter1d(p_surf, w, axis=1, mode="nearest")
    le, has = rocb._leading_edge(sm)
    surf_bin = (surf_pick - t0) / dt
    both = has & np.isfinite(surf_bin)
    resid = le[both] - surf_bin[both]
    off = float(np.median(resid))
    d = np.abs(resid - off)
    return {"median_bins": float(np.median(d)),
            "p90_bins": float(np.percentile(d, 90)),
            "max_bins": float(d.max()), "offset_bins": off,
            "offset_s": off * dt, "n_traces": int(both.sum())}


def profile_vs_depth(power, twtt, t_surf, dt, smooth_m=5.0):
    """(depth_m, dB rel surface peak): twtt below the surface peak converted
    with c/sqrt(EPS_MEAN) (the investigation's convention), 5 m boxcar."""
    bin_depth = C * dt / (2.0 * np.sqrt(rfi.EPS_MEAN))
    w = max(int(round(smooth_m / bin_depth)) | 1, 3)
    ps = np.convolve(power, np.ones(w) / w, mode="same")
    i0 = int(np.clip(np.searchsorted(twtt, t_surf) - int(0.3e-6 / dt), 0,
                     len(twtt) - 2))
    i1 = int(np.clip(np.searchsorted(twtt, t_surf + 1.0e-6), i0 + 1, len(twtt)))
    pk = ps[i0:i1].max()
    depth = (twtt - t_surf) * C / (2.0 * np.sqrt(rfi.EPS_MEAN))
    db = 10.0 * np.log10(np.maximum(ps / max(pk, 1e-300), 1e-12))
    return depth, db


def band_levels(depth, db, edges=BAND_EDGES_M, extra=EXTRA_BANDS):
    out = {}
    for lo, hi in list(zip(edges[:-1], edges[1:])) + list(extra):
        m = (depth >= lo) & (depth < hi)
        out[f"{lo:.0f}-{hi:.0f}m"] = (float(np.median(db[m])) if m.any()
                                      else float("nan"))
    return out


def profile_corr(ref, others, lo=5.0, hi=PROFILE_MAX_M):
    """Pearson r of dB depth profiles vs ``ref`` (name -> (depth, db)), the
    others interpolated onto ref's depth axis over [lo, hi] m."""
    d_r, db_r = ref
    m = (d_r >= lo) & (d_r <= hi)
    return {k: float(np.corrcoef(db_r[m], np.interp(d_r[m], d, db))[0, 1])
            for k, (d, db) in others.items()}


def surface_peak_twtt(power, twtt, t_guess, dt):
    """Per-trace surface peak time within +-0.8 us of a guess."""
    n = len(twtt)
    out = np.full(power.shape[0], np.nan)
    for t in range(power.shape[0]):
        if not np.isfinite(t_guess[t]):
            continue
        a = int(np.clip((t_guess[t] - 0.8e-6 - twtt[0]) / dt, 0, n - 2))
        b = int(np.clip((t_guess[t] + 0.8e-6 - twtt[0]) / dt, a + 1, n))
        out[t] = twtt[a + int(np.argmax(power[t, a:b]))]
    return out


def run_keys(cfg_dict):
    """Every run key this configuration builds, in construction order (the
    --only vocabulary). Must stay in step with run_all's run_list."""
    return (["wide_surface_bed"]
            + [f"firn_N{n}" for n in sorted(cfg_dict["layer_counts"])]
            + [f"firn_N{n}_s{s}" for n, s in cfg_dict.get("random_runs", ())]
            + [f"firn_N{n}_rough_{src}"
               for n, src in cfg_dict.get("rough_runs", ())]
            + [f"firn_N{n}_h1eff" for n in cfg_dict.get("eff_runs", ())])


# ========================================================================
# main runner
# ========================================================================
def run_all(out_root=None, n_traces=N_TRACES, ct_wide=CT_WIDE, ct_firn=CT_FIRN,
            along_m=ALONG_M, layer_counts=LAYER_COUNTS,
            random_runs=RANDOM_RUNS, rough_runs=ROUGH_RUNS, eff_runs=EFF_RUNS,
            spacing=None, do_pilot=True, force=False, report=True, only=None):
    out = Path(out_root or OUT_DEFAULT)
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"
    params = mcords_2019_params()

    frame = load_frame(SEASON, FRAME_ID)
    qframe = load_qlook_frame()
    bot_full = load_bottom_pick(frame)
    cfg_dict = {"n_traces": n_traces, "ct_wide": ct_wide, "ct_firn": ct_firn,
                "along_m": along_m, "layer_counts": tuple(layer_counts),
                "random_runs": tuple(tuple(r) for r in random_runs),
                "rough_runs": tuple(tuple(r) for r in rough_runs),
                "eff_runs": tuple(int(n) for n in eff_runs)}
    only = None if only is None else set(only)
    if only is not None and not only <= set(run_keys(cfg_dict)):
        # before ANY simulation: a typo'd key must not cost a run first
        raise ValueError(
            f"--only names unknown runs {sorted(only - set(run_keys(cfg_dict)))}"
            f"; this configuration has {run_keys(cfg_dict)}")

    def _slice(along):
        fsub, sinfo = sub_frame(frame, along)
        a, b = sinfo["slice"]
        return fsub, sinfo, bot_full[a:b]

    fsub, sinfo, bot_sub = _slice(cfg_dict["along_m"])
    rc_sim, rc_frame, b0 = radar_grids(frame, fsub, bot_sub, params)
    r_min = float(np.nanmin(fsub.Surface.values)) * C / 2.0
    thick_est = float(np.nanmedian(
        (bot_sub - fsub.Surface.values))) * C / (2.0 * np.sqrt(EPS_ICE))
    if spacing is None:
        spacing = facet_spacing(rc_sim, r_min, thick_est)
    lam = rc_sim.wavelength
    gamma_surf = fresnel_normal(1.0, EPS_ICE)
    lpa_err = rocb._lpa_nadir_error(spacing, r_min, 2 * np.pi / lam, gamma_surf)

    budget_log = {"pilot": None, "note": "pilot skipped"}
    if not do_pilot and (out / "budget_log.json").exists():
        budget_log = json.loads((out / "budget_log.json").read_text())
    if do_pilot:
        cfg_dict, budget_log, resliced = pilot_and_budget(
            fsub, cfg_dict, rc_sim, spacing, out)
        if resliced:  # along-track shrank: rebuild the sub-segment + grids
            fsub, sinfo, bot_sub = _slice(cfg_dict["along_m"])
            rc_sim, rc_frame, b0 = radar_grids(frame, fsub, bot_sub, params)

    # --- scenes (final configuration) ---
    wscene, waux = wide_scene(fsub, cfg_dict["n_traces"], cfg_dict["ct_wide"])
    chunks = firn_scenes(wscene, cfg_dict["ct_firn"], spacing)
    idx = waux["idx"]

    meta_common = {"season": SEASON, "frame_id": FRAME_ID,
                   "along_m": cfg_dict["along_m"],
                   "n_traces": int(len(idx)), "spacing_m": round(spacing, 3),
                   "dt_sim_ns": round(rc_sim.dt * 1e9, 4),
                   "t0_us": round(rc_sim.t0 * 1e6, 4),
                   "n_samples_sim": rc_sim.n_samples}

    # --- simulations ---
    t_all = time.perf_counter()
    diag_bed, run_bed = run_sim(
        "wide_surface_bed", [(wscene, np.arange(len(idx)))],
        bed_cfg(rc_sim, spacing),
        {**meta_common, "ct": cfg_dict["ct_wide"], "kind": "surface+bed"},
        runs_dir, force, allow_sim=only is None or "wide_surface_bed" in only)
    run_list = [(f"firn_N{n}", rfi.equal_depths(n), None, None)
                for n in sorted(cfg_dict["layer_counts"])]
    run_list += [(f"firn_N{n}_s{s}", rfi.random_depths(n, s), None, None)
                 for n, s in cfg_dict.get("random_runs", ())]
    rough_spec = {}
    for n, src in cfg_dict.get("rough_runs", ()):
        d = rfi.equal_depths(n)
        sig, cl = layer_roughness(d, src)
        rough_spec[f"firn_N{n}_rough_{src}"] = {
            "source": src, "n_layers": int(n),
            "csv": [f"tests/fixtures/firn/fig11{c}" for c in
                    ("a_rms_height.csv", "b_correlation_length.csv")],
            "columns": [f"rms_height_{src}_m", f"corr_length_{src}_m"],
            "roughness_seed": SimConfig.model_fields["roughness_seed"].default,
            "smooth_interfaces": ["surface (air-firn)", "bed (wide run)"],
            "depth_m": [round(float(x), 3) for x in d],
            "sigma_m": [round(float(x), 6) for x in sig],
            "corr_length_m": [round(float(x), 4) for x in cl],
            "note": "Culberg & Schroeder 2020 Fig. 11 inverted layer "
                    "roughness, linearly interpolated to the equal-placement "
                    "layer depths and clamped beyond the profile's 0-90 m "
                    "range; applied to every INTERNAL layer interface only "
                    "(Gerekos 2023 rough-facet response, docs/roughness.md)"}
        run_list.append((f"firn_N{n}_rough_{src}", d, (sig, cl), None))
    eff_spec = {}
    for n in cfg_dict.get("eff_runs", ()):
        d = rfi.equal_depths(n)
        eps_eff, r_eff = effective_contrast_eps(d, rc_sim.wavelength)
        trend = [rfi.point_eps(x) for x in d] + [rfi.point_eps(float(d[-1]) + 1)]
        eff_spec[f"firn_N{n}_h1eff"] = {
            "method": EFF_METHOD, "n_layers": int(n),
            "source": "tests/fixtures/firn/ngt37C95.2_density.tab (RAW 1 mm, "
                      "Kovacs eps; NOT the 0.1 m-smoothed pipeline profile)",
            "lambda_m": round(float(rc_sim.wavelength), 6),
            "depth_m": [round(float(x), 3) for x in d],
            "segment_abs_r_db": [round(float(20 * np.log10(max(x, 1e-30))), 3)
                                 for x in r_eff],
            "eps_r": [round(float(x), 6) for x in eps_eff],
            "eps_r_point_sampled": [round(float(x), 6) for x in trend],
            "eps_range": [round(float(eps_eff.min()), 4),
                          round(float(eps_eff.max()), 4)],
            "max_abs_eps_minus_trend": round(float(np.abs(
                eps_eff - np.array(trend)).max()), 4),
            "note": "H1 (claude_notes/b26_gap_hypotheses.md): media "
                    "firn0..firn_{N-1} + substrate whose PLAIN Fresnel "
                    "interface contrasts equal the transfer-matrix aggregate "
                    "|r| of the raw full-resolution B26 profile over each "
                    "layer's segment (segment_reflectivity); firn0 keeps its "
                    "point-sampled value so the surface interface and the "
                    "seam check are unchanged, and the per-step sign tracks "
                    "the Kovacs trend. Same equal-placement geometry as "
                    f"firn_N{n}; everything else identical."}
        run_list.append((f"firn_N{n}_h1eff", d, None, eps_eff))
    assert [k for k, *_ in run_list] == run_keys(cfg_dict)[1:]
    firn_runs = {}
    for key, depths, rough, eps_eff in run_list:
        rmeta = {} if rough is None else {
            "roughness": [key.split("_rough_")[-1],
                          round(float(rough[0].sum()), 6),
                          round(float(rough[1].sum()), 4)]}
        if eps_eff is not None:
            rmeta["eps"] = [EFF_METHOD, round(float(np.sum(eps_eff)), 6)]
        diag, arrs = run_sim(
            key, chunks, firn_cfg(rc_sim, spacing, depths, rough, eps_eff),
            {**meta_common, "ct": cfg_dict["ct_firn"], "kind": key,
             "n_chunks": len(chunks), "att_db_per_km": ATT_DB_PER_KM,
             "depths_hash": round(float(depths.sum()), 4), **rmeta},
            runs_dir, force, allow_sim=only is None or key in only)
        firn_runs[key] = (diag, arrs, depths)
    wall_actual = time.perf_counter() - t_all
    prov = {"wide_surface_bed": diag_bed.get("provenance"),
            **{k: d[0].get("provenance") for k, d in firn_runs.items()}}
    stale = sorted(k for k, v in prov.items() if v == "cache-stale")

    # --- assemble fields on the frame subwindow grid ---
    tw = run_bed["twtt"]  # decimated == frame bins b0..b0+nb-1
    dt, t0 = rc_frame.dt, rc_frame.t0
    E_bed = run_bed["field"]                      # (T, nb, 2)
    E2 = E_bed.sum(-1)                            # surface + bed
    surf_pick = fsub.Surface.values[idx]
    totals = {"surface+bed": E2}
    seam = {}
    for key, (diag, arrs, depths) in firn_runs.items():
        Ef = arrs["field"]
        totals[key] = E2 + Ef[..., 1:].sum(-1)
        # seam check: firn run's surface layer (air->firn0), gamma-scaled,
        # vs the wide run's surface layer (air->ice), early window
        ratio = (fresnel_normal(1.0, EPS_ICE)
                 / fresnel_normal(1.0, rfi.point_eps(float(depths[0]))))
        rel = []
        for t in range(E2.shape[0]):
            if not np.isfinite(surf_pick[t]):
                continue
            a = int(np.clip((surf_pick[t] - 0.3e-6 - t0) / dt, 0, len(tw) - 2))
            b = int(np.clip((surf_pick[t] + SEAM_WIN_US * 1e-6 - t0) / dt,
                            a + 1, len(tw)))
            wide_s = E_bed[t, a:b, 0]
            firn_s = Ef[t, a:b, 0] * ratio
            den = np.abs(wide_s).max()
            if den > 0:
                rel.append(float(np.abs(firn_s - wide_s).max() / den))
        seam[key.removeprefix("firn_")] = (float(np.median(rel)) if rel
                                           else float("nan"))

    # --- gates / alignment ---
    bot_pick = bot_sub[idx]
    p_surf = np.abs(E_bed[..., 0]) ** 2
    le = leading_edge_gate(p_surf, spacing, rc_frame, surf_pick)
    nadir = run_bed["nadir_twtt"]                 # (T, 2)
    ok = np.isfinite(bot_pick) & np.isfinite(surf_pick)
    res_b = nadir[:, 1][ok] - bot_pick[ok]
    off_b = float(np.median(res_b))
    rb = np.abs(res_b - off_b) / dt
    bed_med, bed_p90 = float(np.median(rb)), float(np.percentile(rb, 90))
    off_s = float(np.median((nadir[:, 0] - surf_pick)[np.isfinite(surf_pick)]))

    # input bed floor (BedMachine vs picks), M24 formula
    n_ice = np.sqrt(EPS_ICE)
    trp = Transformer.from_crs("EPSG:4326", wscene.crs, always_xy=True)
    lat, lon = _lonlat(fsub)
    px, py = trp.transform(lon[idx], lat[idx])
    cols, rows = (~wscene.transform) * (px, py)
    r_i = np.clip(np.round(rows).astype(int), 0, wscene.dem.shape[0] - 1)
    c_i = np.clip(np.round(cols).astype(int), 0, wscene.dem.shape[1] - 1)
    thick_in = (wscene.dems[0] - wscene.dems[1])[r_i, c_i]
    thick_pk = (bot_pick - surf_pick) * C / (2.0 * n_ice)
    d_in = (thick_in - thick_pk)[ok] * 2.0 * n_ice / C / dt
    d_in = d_in - np.median(d_in)
    in_med = float(np.median(np.abs(d_in)))

    # closest-approach trace (sim + native)
    s_sim = sinfo["s_rel_m"][idx]
    j0 = int(np.argmin(np.abs(s_sim)))
    i0 = sinfo["i0_local"]
    bed_depth_bm = float(thick_in[j0])
    bed_depth_pick = float(thick_pk[j0]) if np.isfinite(thick_pk[j0]) else None

    # --- nadir depth-power profiles (closest trace, upper PROFILE_MAX_M) ---
    # Both measured products are treated identically: own closest-approach
    # trace, own surface peak, dB rel that peak -- so the ~10 dB product gain
    # difference (different presum/multilook normalization) cancels.
    meas = np.asarray(fsub.Data.values, np.float64)  # (T_native, twtt_full)
    tw_full = frame.twtt.values
    prof = {}

    def _meas_profile(sub, tw_p, i):
        dtp = float(tw_p[1] - tw_p[0])
        t_s = surface_peak_twtt(np.asarray(sub.Data.values[[i]], np.float64),
                                tw_p, np.array([sub.Surface.values[i]]),
                                dtp)[0]
        return profile_vs_depth(np.asarray(sub.Data.values[i], np.float64),
                                tw_p, t_s, dtp)

    prof["measured"] = _meas_profile(fsub, tw_full, i0)
    qsub, qinfo, qlook = None, None, None
    if qframe is not None:
        qsub, qinfo = sub_frame(qframe, cfg_dict["along_m"])
        tw_q = qframe.twtt.values
        prof["measured_qlook"] = _meas_profile(qsub, tw_q, qinfo["i0_local"])
        qlook = {"sub": qsub, "info": qinfo, "twtt": tw_q,
                 "same_fast_time_grid": bool(len(tw_q) == len(tw_full)
                                             and np.allclose(tw_q, tw_full))}
    for name, E in totals.items():
        p = np.abs(E[j0]) ** 2
        t_s = surface_peak_twtt(p[None], tw, np.array([surf_pick[j0]]), dt)[0]
        prof[name] = profile_vs_depth(p, tw, t_s, dt)
    bands = {k: band_levels(d, db) for k, (d, db) in prof.items()}

    # Correlations against BOTH measured products; each reference also scores
    # the other measured product (the qlook-vs-standard row).
    def _targets(ref_key):
        t = {k: prof[k] for k in totals}
        other = MEAS["qlook"] if ref_key == MEAS["standard"] else MEAS["standard"]
        if other in prof:
            t[other] = prof[other]
        return t

    corr = profile_corr(prof[MEAS["standard"]], _targets(MEAS["standard"]))
    corr_q = (profile_corr(prof[MEAS["qlook"]], _targets(MEAS["qlook"]))
              if MEAS["qlook"] in prof else {})
    # band deltas (sim - measured, dB) in the two diagnostic bands
    gap_bands = [f"{lo:.0f}-{hi:.0f}m" for lo, hi in EXTRA_BANDS]
    gap_run = ("firn_N40" if "firn_N40" in totals
               else f"firn_N{max(cfg_dict['layer_counts'])}")
    deltas = {}
    for tag, mkey in MEAS.items():
        if mkey not in bands:
            continue
        deltas[tag] = {k: {b: round(bands[k][b] - bands[mkey][b], 2)
                           for b in gap_bands}
                       for k in list(totals) + [m for m in MEAS.values()
                                                if m != mkey and m in bands]}

    # --- metrics ---
    rec = "recorded only"
    f_alias = abs(rc_sim.f0 - round(rc_sim.f0 * rc_sim.dt) / rc_sim.dt)
    metrics = {
        "surface_pick_alignment": {
            "value": le["median_bins"], "threshold": GATE_BINS, "op": "<=",
            "pass": bool(le["median_bins"] <= GATE_BINS),
            "p90_bins": le["p90_bins"], "max_bins": le["max_bins"],
            "offset_bins": le["offset_bins"],
            "note": "median |smoothed coherent surface-layer leading edge - "
            "frame Surface pick| in FRAME bins (16.667 ns) after removing the "
            "constant offset (absorbs system delay / DEM epoch), M24 method"},
        "bed_alignment": {
            "value": bed_med, "threshold": max(GATE_BINS, in_med + GATE_BINS),
            "op": "<=", "pass": bool(bed_med <= max(GATE_BINS,
                                                    in_med + GATE_BINS)),
            "p90_bins": bed_p90, "offset_bins": off_b / dt,
            "input_floor_bins": in_med,
            "note": "median |bed-layer nadir twtt - Bottom pick| frame bins, "
            "offset removed; floor-aware threshold max(5, input floor + 5) "
            "(BedMachine's own disagreement with the picks caps this)"},
        "alias_free_dt": {
            "value": rc_sim.dt * 1e9, "threshold": None, "pass": True,
            "op": "record", "oversample": OVERSAMPLE,
            "f_alias_sim_mhz": f_alias / 1e6,
            "alias_warning_fired": False,
            "note": "simulation dt (ns) = frame dt/4; alias at 45 MHz = 3B/2 "
            "(out of band, warning asserted silent on every simulate() call); "
            "decimation [::4] exact onto the frame grid. " + rec},
        "lpa_nadir_error": {
            "value": lpa_err, "threshold": None, "pass": True, "op": "record",
            "facet_size_m": float(spacing), "r_min_m": r_min,
            "note": "envelope-normalized single-facet LPA error at nadir "
            "(worst case); off-nadir facets are sinc-suppressed. " + rec},
        "bed_depth_at_site": {
            "value": bed_depth_bm, "threshold": None, "pass": True,
            "op": "record", "pick_derived_m": bed_depth_pick,
            "note": "BedMachine ice thickness (m) at the closest-approach "
            "nadir vs the pick-derived thickness (eps 3.17). " + rec},
        "firn_seam_check": {
            "value": max(seam.values()), "threshold": None, "pass": True,
            "op": "record", **{f"rel_{k}": v for k, v in seam.items()},
            "note": "field-sum seam: the firn run's surface (layer-0) field, "
            "scaled by the air->firn0 / air->ice gamma ratio, vs the wide "
            "run's surface-layer field over the first "
            f"{SEAM_WIN_US} us after the pick (where the +-"
            f"{cfg_dict['ct_firn']:.0f} m strip covers all arrivals); median "
            "over traces of the max relative deviation. The firn layer-0 "
            "field is EXCLUDED from the sum (no double count); this checks "
            "the two scenes agree where they must. " + rec},
        "closest_approach_m": {
            "value": sinfo["closest_m"], "threshold": None, "pass": True,
            "op": "record",
            "note": "frame ground-track distance to the B26 borehole"},
        "profile_correlation": {
            "value": corr.get("firn_N%d" % max(cfg_dict["layer_counts"])),
            "threshold": None, "pass": True, "op": "record",
            **{f"corr_standard_{k.replace('+', '_')}": v
               for k, v in corr.items()},
            **{f"corr_qlook_{k.replace('+', '_')}": v
               for k, v in corr_q.items()},
            "note": "Pearson r of the nadir depth-power dB profiles "
            f"(measured vs sim, 5-{PROFILE_MAX_M:.0f} m below the surface "
            "peak) at each product's own closest-approach trace, against BOTH "
            "measured products: corr_standard_* = vs CSARP_standard (f-k SAR "
            "+ 11 looks), corr_qlook_* = vs CSARP_qlook (unfocused, the "
            "like-for-like processing). corr_standard_measured_qlook / "
            "corr_qlook_measured are the two products against each other. "
            "Morphology diagnostic. " + rec},
        "band_delta_vs_measured": {
            "value": (deltas.get("qlook", deltas.get("standard", {}))
                      .get(gap_run, {}).get(GAP_BAND, float("nan"))),
            "gap_run": gap_run,
            "threshold": None, "pass": True, "op": "record",
            "bands": gap_bands,
            "measured_standard_db": {b: round(bands["measured"][b], 2)
                                     for b in gap_bands},
            "measured_qlook_db": ({b: round(bands["measured_qlook"][b], 2)
                                   for b in gap_bands}
                                  if "measured_qlook" in bands else None),
            "delta_vs_standard": deltas.get("standard"),
            "delta_vs_qlook": deltas.get("qlook"),
            "note": "median depth-power level (dB rel own surface peak) minus "
            "the measured product's level, in the mid-band "
            f"{gap_bands[0]} and deep firn band {gap_bands[1]}; value = "
            f"firn_N40 vs CSARP_qlook in {GAP_BAND}. Negative = the sim sits "
            "BELOW the measurement. The vs_qlook column removes the SAR-"
            "focusing asymmetry from the comparison. " + rec},
        "simulation_wall_s": {
            "value": round(diag_bed["wall_s"] + sum(
                d[0]["wall_s"] for d in firn_runs.values()), 1),
            "threshold": None, "pass": True, "op": "record",
            "wall_this_invocation_s": round(wall_actual, 1),
            "projected_s": (budget_log.get("projection_s", {})
                            .get("final", {}).get("total")),
            "budget_s": BUDGET_S,
            "note": "recorded simulate() wall of the three runs (resumable "
            "cache); pilot projection vs actual in budget_log.json / "
            "run_config.json. " + rec},
    }
    if rough_spec:
        smooth_ref = f"firn_N{rough_spec[next(iter(rough_spec))]['n_layers']}"
        gain = {k: (round(bands[k][GAP_BAND] - bands[smooth_ref][GAP_BAND], 2)
                    if smooth_ref in bands else float("nan"))
                for k in rough_spec}
        metrics["roughness_band_delta"] = {
            "value": gain[next(iter(gain))], "threshold": None, "pass": True,
            "op": "record", "smooth_reference": smooth_ref, "band": GAP_BAND,
            "gain_vs_smooth_db": gain,
            **{f"sigma_cm_range_{k}": [round(100 * min(v["sigma_m"]), 2),
                                       round(100 * max(v["sigma_m"]), 2)]
               for k, v in rough_spec.items()},
            **{f"corr_length_m_range_{k}": [round(min(v["corr_length_m"]), 2),
                                            round(max(v["corr_length_m"]), 2)]
               for k, v in rough_spec.items()},
            "note": "change in the median "
            f"{GAP_BAND} depth-power level (dB rel own surface peak) of the "
            "rough-layer runs vs the SMOOTH equal-placement run of the same "
            "layer count -- i.e. how much of the mid-band deficit the "
            "measured (C&S 2020 Fig. 11) sub-facet layer roughness recovers. "
            "Positive = roughness raises the mid-band. The per-product "
            "deltas of these runs vs the measured products are in "
            "band_delta_vs_measured / profile_correlation like any other "
            "run. " + rec}
    metrics["run_provenance"] = {
        "value": len(stale), "threshold": None, "pass": True, "op": "record",
        "only": sorted(only) if only else None, "provenance": prov,
        "simulated_this_invocation": sorted(
            k for k, v in prov.items() if v == "simulated"),
        "cache_stale": stale,
        "note": "how each assembled run was obtained: 'simulated' this "
        "invocation, 'cache' (recorded config matches), or 'cache-stale' "
        "(--only excluded it from simulation and its cached npz was reused "
        "AS-IS although its recorded config no longer matches the current "
        "module defaults). value = number of cache-stale runs: NON-ZERO MEANS "
        "THIS REPORT MIXES PROVENANCES and the listed runs are not directly "
        "comparable to the freshly simulated ones. " + rec}

    config = {
        **cfg_dict, "spacing_m": round(spacing, 2),
        "dt_sim_ns": round(rc_sim.dt * 1e9, 4),
        "n_samples_sim": rc_sim.n_samples,
        "frame_bins_window": [int(b0), int(b0 + rc_frame.n_samples - 1)],
        "chirp": {"f0_hz": rc_sim.f0,
                  "bandwidth_hz": params["waveform"]["bandwidth_Hz"],
                  "pulse_length_s":
                      params["waveform"]["bed_waveform_pulse_length_s"],
                  "window": "hann"},
        "antenna": f"{N_ELEMENTS}-element {SPACING_LAM}-lambda array, "
                   "roll_source=nav",
        "media_bed_run": f"air / ice(eps {EPS_ICE}, {ATT_DB_PER_KM} dB/km "
                         f"one-way) / bed(eps {EPS_BED})",
        "firn_attenuation_db_per_km": ATT_DB_PER_KM,
        "run_provenance": prov,
        "only": sorted(only) if only else None,
        "firn_pipeline": "B26 point-sampled Kovacs eps, equal placement "
                         f"1-{rfi.ZMAX:.1f} m (run_firn_investigation); "
                         f"random placements (n, seed): "
                         f"{cfg_dict.get('random_runs', ())}; rough runs "
                         f"(n, source): {cfg_dict.get('rough_runs', ())}; "
                         f"effective-contrast (H1) runs (n): "
                         f"{cfg_dict.get('eff_runs', ())}",
        "roughness_runs": rough_spec,
        "effective_contrast_runs": eff_spec,
        "band_levels_db_rel_surface": bands,
        "band_delta_db_sim_minus_measured": deltas,
        "profile_correlation_r": corr,
        "profile_correlation_r_qlook": corr_q,
        "measured_products": {
            "standard": {"n_traces": int(frame.sizes["slow_time"]),
                         "note": "f-k SAR sigma_x 2.5 m + 11-look, dline 6"},
            "qlook": (None if qframe is None else {
                "n_traces": int(qframe.sizes["slow_time"]),
                "same_fast_time_grid": qlook["same_fast_time_grid"],
                "gain_ratio_p999_vs_standard_db": round(float(
                    10.0 * np.log10(
                        np.percentile(np.abs(qframe.Data.values), 99.9)
                        / np.percentile(np.abs(frame.Data.values), 99.9))), 2),
                "note": "unfocused: pulse compression + presums, no SAR"})},
        "trace_spacing_m": float(np.median(np.diff(s_sim))),
        "closest_trace": {"sim_index": j0, "native_index": int(i0),
                          "s_rel_m": float(s_sim[j0])},
        "budget": {**budget_log, "actual_s": {
            "wide": diag_bed["wall_s"],
            **{k: d[0]["wall_s"] for k, d in firn_runs.items()},
            "total_this_invocation": round(wall_actual, 1)}},
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=1) + "\n")

    qnote = ""
    if MEAS["qlook"] in bands:
        mp = config["measured_products"]["qlook"]
        qnote = (
            f" MEASURED PRODUCTS: CSARP_standard (f-k SAR + 11 looks, "
            f"{config['measured_products']['standard']['n_traces']} traces) AND "
            f"CSARP_qlook (UNFOCUSED: pulse compression + presums, "
            f"{mp['n_traces']} traces, identical fast-time grid: "
            f"{mp['same_fast_time_grid']}; ~{mp['gain_ratio_p999_vs_standard_db']:+.0f}"
            f" dB product gain difference, cancelled by the per-profile "
            f"surface-peak normalization). The two measured profiles correlate "
            f"r={corr_q.get('measured', float('nan')):.2f} and differ by "
            f"{deltas['qlook']['measured'][GAP_BAND]:+.1f} dB (standard minus "
            f"qlook) in the {GAP_BAND} band. {gap_run} sits "
            f"{deltas['standard'][gap_run][GAP_BAND]:+.1f} dB vs standard "
            f"and {deltas['qlook'][gap_run][GAP_BAND]:+.1f} dB vs qlook "
            f"there, so the SAR-focusing asymmetry accounts for only part of "
            f"the mid-band deficit.")
    rnote = ""
    if rough_spec:
        rm = metrics["roughness_band_delta"]
        rnote = (
            " ROUGH-LAYER RUNS: every INTERNAL firn layer interface of the "
            "equal-placement stack carries Gerekos-2023 sub-facet Gaussian "
            "roughness (docs/roughness.md) with sigma / correlation length "
            "interpolated at that layer's depth from the Culberg & Schroeder "
            "2020 Fig. 11 inverted layer-roughness profiles (0-90 m, clamped "
            "below; tests/fixtures/firn/fig11a,b), "
            + "; ".join(
                f"{k}: sigma {100 * min(v['sigma_m']):.1f}-"
                f"{100 * max(v['sigma_m']):.1f} cm, l "
                f"{min(v['corr_length_m']):.1f}-"
                f"{max(v['corr_length_m']):.1f} m"
                for k, v in rough_spec.items())
            + f". The air-firn surface and the bed stay SMOOTH so the seam "
            f"check and the own-surface-peak normalization stay comparable. "
            f"roughness_seed = {SimConfig.model_fields['roughness_seed'].default}"
            f" (one frozen speckle realization). sigma/lambda_firn ~ 0.05 -- "
            f"the easy regime (<= ~0.3 dB, docs/roughness.md); l up to "
            f"{max(max(v['corr_length_m']) for v in rough_spec.values()):.1f} m"
            f" stays below the {spacing:.1f} m facet size as required. "
            f"Mid-band ({GAP_BAND}) level change vs the SMOOTH run of the same "
            f"layer count: {rm['gain_vs_smooth_db']} dB.")
    enote = ""
    if eff_spec:
        enote = (
            " EFFECTIVE-CONTRAST (H1) RUNS: same equal-placement geometry, but "
            "the layer permittivities are SYNTHETIC -- each interface's plain "
            "Fresnel contrast equals the transfer-matrix aggregate |r| of the "
            "RAW full-resolution (1 mm) B26 density profile over that layer's "
            "segment, so the 0.1-0.5 m Bragg-scale strata that point sampling "
            "discards are represented as an effective contrast "
            "(claude_notes/b26_gap_hypotheses.md H1; 1-D 20-70 m band level "
            "-28.3 -> -17.2 dB rel surface). "
            + "; ".join(
                f"{k}: eps {v['eps_range'][0]:.2f}-{v['eps_range'][1]:.2f} "
                f"(max {v['max_abs_eps_minus_trend']:.2f} off the "
                f"point-sampled trend), segment |r| "
                f"{min(v['segment_abs_r_db']):.1f} to "
                f"{max(v['segment_abs_r_db']):.1f} dB"
                for k, v in eff_spec.items())
            + ". firn0 keeps its point-sampled value (surface interface, seam "
            "check and surface-peak normalization unchanged); the full eps "
            "arrays are in run_config.json effective_contrast_runs. The "
            "segment-aggregated band LEVEL is N-independent by construction, "
            "so the N-ladder tests profile SHAPE: correlation should plateau "
            "once the layer spacing beats the ~4.4 m in-firn range cell "
            "(N ~ 27).")
    pnote = ""
    if stale:
        pnote = (
            f" MIXED PROVENANCE: this report was assembled with --only "
            f"{sorted(only)}, so {stale} were NOT re-simulated -- their cached "
            f"fields are reused AS-IS even though their recorded configuration "
            f"no longer matches the current module defaults (e.g. the firn "
            f"attenuation default). Freshly simulated here: "
            f"{sorted(k for k, v in prov.items() if v == 'simulated')}. Treat "
            f"cross-run comparisons involving the stale runs with care; "
            f"metrics.json run_provenance has the per-run detail.")
    notes = _notes(cfg_dict, params, spacing, lpa_err, le, bed_med, in_med,
                   off_s, off_b, dt, seam, sinfo, bed_depth_bm, bed_depth_pick,
                   wall_actual, budget_log, waux,
                   qnote + rnote + enote + pnote)
    doc = {"case": "b26_comparison", "group": "xOPR clutter",
           "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "metrics": metrics, "notes": notes}
    (out / "metrics.json").write_text(json.dumps(doc, indent=1) + "\n")

    figs = _figures(out, fsub, sinfo, idx, tw, tw_full, totals, meas,
                    surf_pick, bot_pick, off_s, off_b, prof, cfg_dict, dt, t0,
                    firn_runs, qlook)
    if report:
        _report(out, config, metrics, notes, figs, params)
    # mirror into outputs/verification/ so tools/make_report.py picks it up
    if Path(out) == OUT_DEFAULT:
        VER_OUT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "metrics.json", VER_OUT / "metrics.json")
        for f in figs:
            shutil.copy2(f, VER_OUT / f.name)
    print(f"b26_comparison: wall {wall_actual:.1f} s | surf med "
          f"{le['median_bins']:.2f} bins | bed med {bed_med:.1f} vs floor "
          f"{in_med:.1f} | seam {max(seam.values()):.3g} | corr {corr}\n"
          f"  corr_qlook {corr_q}\n  band delta ({GAP_BAND}) "
          f"{ {t: {k: v[GAP_BAND] for k, v in d.items()} for t, d in deltas.items()} }",
          flush=True)
    return metrics, out


def _notes(cfg, params, spacing, lpa_err, le, bed_med, in_med, off_s, off_b,
           dt, seam, sinfo, bm, pk, wall, budget_log, waux, qnote=""):
    wf = params["waveform"]
    shr = (", ".join(f"{k}={v}" for s in budget_log.get("shrink_steps", [])
                     for k, v in s.items()) or "none")
    return (
        f"{SEASON} {FRAME_ID}: measured (CSARP_standard) vs simulated coherent "
        f"cluttergrams on a {cfg['along_m']/1e3:.0f} km sub-segment centered "
        f"on the B26 firn core closest approach ({sinfo['closest_m']:.0f} m "
        f"from the borehole), {cfg['n_traces']} traces. INSTRUMENT (2019 "
        f"season's own product params, outputs/cache/mcords_2019P3_params.json"
        f" -- NOT the 2017 values; note the 2019 product grid is dt = 16.667 "
        f"ns): chirp {wf['center_frequency_Hz']/1e6:.0f} MHz / "
        f"{wf['bandwidth_Hz']/1e6:.0f} MHz (180-210), pulse "
        f"{wf['bed_waveform_pulse_length_s']*1e6:.0f} us (longest/bed "
        f"waveform of Tpd {sorted(wf['pulse_lengths_s'])}), hann compression "
        f"(ft_wind=@hanning; 20% tx Tukey unmodeled), 7-element 0.5-lambda "
        f"array, roll from nav. ALIAS-FREE grid: dt_sim = dt/4 = "
        f"{1e9*dt/4:.3f} ns (alias 45 MHz = 3B/2, warning asserted silent), "
        f"decimated [::4] exactly onto the frame twtt grid. Surface ArcticDEM "
        f"32 m; bed {waux['bed_meta']['product']} (150 m, bilinear to 32 m); "
        f"media air / ice(3.17, {ATT_DB_PER_KM:.0f} dB/km one-way constant -- "
        f"the M24 warm-ice value, generous for this cold interior site, "
        f"recorded) / bed(eps 8); every FIRN medium (and the substrate) also "
        f"attenuates at {ATT_DB_PER_KM:.0f} dB/km one-way = "
        f"{2 * ATT_DB_PER_KM * rfi.ZMAX / 1000:.1f} dB two-way at the bottom "
        f"of the core (the earlier zero-attenuation firn convention was ~10x "
        f"low and biased the deep firn band specifically; the analytic bracket "
        f"is 8-12 dB/km, 15 adopted for consistency with the ice medium). "
        f"Firn: B26 point-sampled Kovacs eps, EQUAL "
        f"placement over 1-{rfi.ZMAX:.1f} m, N in {cfg['layer_counts']}, as "
        f"OFFSET interfaces of the ArcticDEM surface, computed on a +-"
        f"{cfg['ct_firn']:.0f} m strip and FIELD-SUMMED (layer 0 excluded) "
        f"with the +-{cfg['ct_wide']:.0f} m surface+bed run; seam check "
        f"(gamma-scaled surface fields, early window) median rel dev "
        f"{max(seam.values()):.2e}. Facets: single spacing {spacing:.1f} m = "
        f"beta {BETA} Fresnel criterion minimized over the stack (deepest "
        f"firn layer binds); LPA nadir envelope error ~{lpa_err*100:.0f}% "
        f"(worst case). Gates: surface leading edge median "
        f"{le['median_bins']:.2f} frame bins (offset {le['offset_bins']:+.1f} "
        f"bins removed); bed nadir median {bed_med:.1f} bins vs input "
        f"BedMachine-vs-pick floor {in_med:.1f} bins. Bed depth at site: "
        f"BedMachine {bm:.0f} m vs pick-derived "
        f"{pk if pk is None else round(pk)} m. Compute budget: pilot-projected"
        f" vs actual in budget_log.json; shrink steps applied: {shr}; wall "
        f"this invocation {wall:.1f} s. HONESTY: (1) 32 m DEM posting -> "
        f"speckle/envelope statistics, not deterministic phase; (2) "
        f"{spacing:.0f} m facets -> specular-dominated coherent LPA (recorded "
        f"error above); (3) equal-placement point-sampled layers are a "
        f"MORPHOLOGY comparison, not calibrated absolute levels -- the firn "
        f"findings showed random placement (the physical case) shifts levels "
        f"and plateau structure vs equal placement; (4) the measured frame is "
        f"f-k SAR processed + 11-look multilooked (~25 m along-track "
        f"resolution) while the sims are unfocused per-trace raw at "
        f"~{cfg['along_m']/cfg['n_traces']:.0f} m trace spacing -- compare "
        f"structure and relative levels, not resolution or absolute texture "
        f"(CSARP_qlook is carried alongside to bound this); "
        f"(5) the sim carries no volume scatter, no receiver noise floor, no "
        f"waveform-playlist gain stitching." + qnote)


# ========================================================================
# figures + report
# ========================================================================
def _figures(out, fsub, sinfo, idx, tw, tw_full, totals, meas, surf_pick,
             bot_pick, off_s, off_b, prof, cfg, dt, t0, firn_runs, qlook=None):
    s_km = sinfo["s_rel_m"] / 1e3
    s_sim = s_km[idx]
    tw_us = tw * 1e6

    def _meas_panel(sub, tw_p, s_axis, label, data=None):
        """Measured-product panel spec. Colour limits are PER PRODUCT (99.5th
        pct over the sim window): standard and qlook differ by ~10 dB of
        processing gain, so a shared scale would black out one of them."""
        db = _db(np.asarray(sub.Data.values if data is None else data,
                            np.float64))
        a = int(np.searchsorted(tw_p, tw[0] - 1e-12))
        win = db[:, a:a + len(tw)]
        return {"db": db, "twtt": tw_p, "s": s_axis, "label": label,
                "surf": np.asarray(sub.Surface.values, np.float64),
                "bot": _bot_native(sub),
                "vmax": float(np.percentile(win[np.isfinite(win)], 99.5))}

    meas_panels = [_meas_panel(fsub, tw_full, s_km,
                               "measured (CSARP_standard: f-k SAR + 11 looks)",
                               data=meas)]
    if qlook is not None:
        meas_panels.append(_meas_panel(
            qlook["sub"], qlook["twtt"], qlook["info"]["s_rel_m"] / 1e3,
            "measured (CSARP_qlook: unfocused, pulse compression + presums)"))

    # Radargram panels: equal placements + one representative random seed
    # (all random seeds appear in the depth profile).
    # (one representative random seed; of the H1 effective-contrast ladder only
    # the finest N -- the whole ladder appears in the depth profile.)
    eff_keys = [k for k in firn_runs if k.endswith("_h1eff")]
    eff_show = ({max(eff_keys, key=lambda k: int(k.split("_N")[1].split("_")[0]))}
                if eff_keys else set())
    names = ["surface+bed"] + [
        k for k in sorted(firn_runs)
        if ("_s" not in k or k.endswith("_s0"))
        and (not k.endswith("_h1eff") or k in eff_show)]
    sims_db = {k: _db(np.abs(totals[k]) ** 2) for k in names}
    vs = np.percentile(np.concatenate(
        [v[np.isfinite(v) & (v > -290)] for v in sims_db.values()]), 99.5)

    def _panels(fig_path, t_lo, t_hi, title, dyn_db=60.0):
        # Measured panels are re-windowed from the full frame so an extended
        # t_hi (below the sim window) still shows real data.
        ext_s = [s_sim[0], s_sim[-1], tw_us[-1], tw_us[0]]
        n_meas = len(meas_panels)
        n_panels = n_meas + len(names)
        ncols = 2 if n_panels <= 4 else 3
        nrows = -(-n_panels // ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5.5 * nrows),
                                sharex=True, sharey=True)
        axs = np.atleast_1d(axs).ravel()
        for num, (ax, mp) in enumerate(zip(axs, meas_panels), start=1):
            twp = mp["twtt"]
            b_lo = max(0, int(np.searchsorted(twp, t_lo * 1e-6)) - 1)
            b_hi = min(len(twp), int(np.searchsorted(twp, t_hi * 1e-6)) + 1)
            ext_m = [mp["s"][0], mp["s"][-1], twp[b_hi - 1] * 1e6,
                     twp[b_lo] * 1e6]
            im = ax.imshow(mp["db"][:, b_lo:b_hi].T, aspect="auto",
                           extent=ext_m, cmap="gray",
                           vmin=mp["vmax"] - dyn_db, vmax=mp["vmax"])
            fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01, label="dB")
            ax.plot(mp["s"], mp["surf"] * 1e6, "c", lw=0.7,
                    label="Surface pick")
            ax.plot(mp["s"], mp["bot"] * 1e6, "r", lw=0.7, label="Bottom pick")
            ax.axvline(0.0, color="y", ls="--", lw=1.0, label="B26 core")
            ax.set_title(f"{num}. {mp['label']}")
        for ax in axs[n_panels:]:  # unused panels (fewer layer counts)
            ax.set_visible(False)
        for num, (ax, name) in enumerate(zip(axs[n_meas:], names),
                                         start=n_meas + 1):
            im = ax.imshow(sims_db[name].T, aspect="auto", extent=ext_s,
                           cmap="gray", vmin=vs - dyn_db, vmax=vs)
            fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01, label="dB")
            ax.plot(s_sim, (surf_pick + off_s) * 1e6, "c", lw=0.6,
                    label="Surface pick + offset")
            ax.plot(s_sim, (bot_pick + off_b) * 1e6, "r", lw=0.6,
                    label="Bottom pick + offset")
            ax.axvline(0.0, color="y", ls="--", lw=1.0)
            ax.set_title(f"{num}. simulated: {name}")
        for ax in axs:
            ax.set_ylim(t_hi, t_lo)
            if ax.get_visible() and ax.has_data():
                ax.legend(loc="center right", fontsize=7)
                ax.set_xlabel("along-track from B26 (km)")
        for ax in axs[::ncols]:
            ax.set_ylabel("twtt (us)")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)

    f1 = out / "radargrams_full.png"
    # Extend past the sim window so the bed pick and echo sit clear of the
    # frame edge (measured panel carries real data there; sims end at ext_s).
    t_bed_max = np.nanmax([np.nanmax(meas_panels[0]["bot"]),
                           np.nanmax(bot_pick + off_b)])
    # 100 dB range so the (weak) bed reflection is visible above the floor;
    # the near-surface figure keeps 60 dB for firn-zone contrast.
    _panels(f1, tw_us[0], max(tw_us[-1], t_bed_max * 1e6 + 1.5),
            f"{FRAME_ID} at B26: measured vs simulated (full window)",
            dyn_db=100.0)
    t_srf = float(np.nanmedian(surf_pick)) * 1e6
    f2 = out / "radargrams_nearsurface.png"
    _panels(f2, t_srf - 0.4, t_srf + 2.4,
            f"{FRAME_ID} at B26: near-surface / firn zone "
            f"(upper ~{(2.4e-6 - 0.2e-6) * C / (2 * np.sqrt(rfi.EPS_MEAN)):.0f} m)")

    # depth-power profile at the closest trace
    f3 = out / "depth_profile.png"
    fig, ax = plt.subplots(figsize=(8.5, 6))
    styles = {"measured": ("k", 1.6, "-"),
              "measured_qlook": ("0.45", 1.6, "--"),
              "surface+bed": ("C7", 1.1, "-"),
              "firn_N10": ("C0", 1.2, "-"), "firn_N20": ("C3", 1.2, "-"),
              "firn_N40": ("C1", 1.2, "-"), "firn_N80": ("C2", 1.4, "-")}
    # rough runs: keyed on the inversion SOURCE (any layer count)
    rough_styles = {"mcords": ("m", 1.7, "-"), "ar": ("c", 1.7, "-.")}
    # H1 effective-contrast ladder: a dark-red ramp, darkest at the finest N
    eff_styles = {5: ("#e8837a", 1.2, "--"), 10: ("#d4544a", 1.3, "--"),
                  20: ("#a81f18", 1.5, "-"), 40: ("darkred", 1.8, "-"),
                  80: ("#3d0000", 1.8, "-")}
    labels = {"measured": "measured (CSARP_standard, focused)",
              "measured_qlook": "measured (CSARP_qlook, unfocused)"}
    rand_labeled = False
    for name, (d, db) in prof.items():
        m = (d >= -5) & (d <= PROFILE_MAX_M)
        if "_s" in name and name.startswith("firn"):  # random-placement seeds
            lbl = None if rand_labeled else \
                name.split("_s")[0] + " random (3 seeds)"
            rand_labeled = True
            ax.plot(d[m], db[m], color="C4", lw=0.9, alpha=0.8, label=lbl)
            continue
        if name.endswith("_h1eff"):    # synthetic effective-contrast eps (H1)
            base = name[:-len("_h1eff")]
            c, lw, ls = eff_styles.get(int(base.split("_N")[1]),
                                       ("darkred", 1.5, "-"))
            lbl = f"{base} eff-contrast (H1)"
        elif "_rough_" in name:  # measured C&S20 Fig. 11 layer roughness
            base, src = name.split("_rough_")
            c, lw, ls = rough_styles.get(src, ("C6", 1.4, ":"))
            lbl = f"{base} + rough layers (C&S20 Fig.11 {src} sigma/l)"
        else:
            c, lw, ls = styles.get(name, ("C5", 1.0, "-"))
            lbl = labels.get(name, name)
        ax.plot(d[m], db[m], color=c, lw=lw, ls=ls, label=lbl)
    ax.axvline(rfi.ZMAX, color="k", ls=":", lw=1.2)
    ax.text(rfi.ZMAX + 1.5, -72, f"B26 core ends ({rfi.ZMAX:.1f} m)",
            fontsize=8, rotation=90, va="bottom")
    ax.set_xlim(-5, PROFILE_MAX_M)
    ax.set_ylim(-75, 3)
    ax.set_xlabel("depth below surface peak (m; c/sqrt(eps_mean))")
    ax.set_ylabel("power (dB rel own surface peak, 5 m smoothed)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"{FRAME_ID}: nadir depth-power at the B26 closest approach "
                 f"({sinfo['closest_m']:.0f} m from borehole)")
    fig.tight_layout()
    fig.savefig(f3, dpi=140)
    plt.close(fig)
    return [f1, f2, f3]


def _bot_native(fsub):
    """Bottom pick on the native sub-frame grid (cached load)."""
    return load_bottom_pick(fsub)


def _report(out, config, metrics, notes, figs, params):
    def b64(p):
        return base64.b64encode(Path(p).read_bytes()).decode()

    css = ("body{font-family:system-ui,Arial,sans-serif;margin:2rem;"
           "max-width:1250px;color:#1a1a1a}h1{margin-bottom:.2rem}"
           "table{border-collapse:collapse;margin:1rem 0;font-size:.85rem}"
           "th,td{border:1px solid #ccc;padding:.3rem .5rem;text-align:left}"
           "th{background:#f0f0f0}img{max-width:100%;border:1px solid #ddd}"
           ".note{color:#333;background:#f6f6f6;border-left:3px solid #bbb;"
           "padding:.6rem 1rem;margin:1rem 0}td.pass{background:#c8f7c5}"
           "td.fail{background:#f7c5c5}code{background:#eee;padding:0 .2rem}")
    mrows = []
    for k, e in metrics.items():
        cls = "pass" if e.get("pass") else "fail"
        thr = e.get("threshold")
        crit = "" if thr is None else f"{e.get('op', '<=')} {thr:.4g}"
        mrows.append(f"<tr><th>{html.escape(k)}</th>"
                     f"<td class='{cls}'>{e.get('value'):.4g}</td>"
                     f"<td>{html.escape(crit)}</td>"
                     f"<td>{html.escape(e.get('note', ''))}</td></tr>")
    crows = []
    flat = {k: v for k, v in config.items()
            if not isinstance(v, (dict, list))}
    for k, v in flat.items():
        crows.append(f"<tr><th>{html.escape(str(k))}</th>"
                     f"<td>{html.escape(str(v))}</td></tr>")
    for k in ("chirp", "closest_trace", "frame_bins_window"):
        crows.append(f"<tr><th>{k}</th><td>{html.escape(json.dumps(config[k]))}"
                     f"</td></tr>")
    bands = config["band_levels_db_rel_surface"]
    bhdr = "".join(f"<th>{b}</th>" for b in next(iter(bands.values())))
    brows = "".join(
        f"<tr><th>{html.escape(k)}</th>"
        + "".join(f"<td>{v:.1f}</td>" for v in bands[k].values()) + "</tr>"
        for k in bands)
    # correlation / band-delta table against BOTH measured products
    cs, cq = config["profile_correlation_r"], config["profile_correlation_r_qlook"]
    dl = config["band_delta_db_sim_minus_measured"]
    dbands = list(next(iter(dl["standard"].values()))) if dl else []
    prows = []
    def _c(v, fmt="{:.3f}"):
        return "-" if v is None else fmt.format(v)

    for k in list(cs) + (["measured"] if "measured" in cq else []):
        cells = [f"<td>{_c(cs.get(k))}</td>", f"<td>{_c(cq.get(k))}</td>"]
        for tag in ("standard", "qlook"):
            for b in dbands:
                cells.append(
                    f"<td>{_c(dl.get(tag, {}).get(k, {}).get(b), '{:+.1f}')}</td>")
        prows.append(f"<tr><th>{html.escape(k)}</th>{''.join(cells)}</tr>")
    dhdr = "".join(f"<th>{t}<br>{b}</th>" for t in ("std", "qlook")
                   for b in dbands)
    prof_tbl = (f"<table><tr><th>profile</th><th>r vs standard</th>"
                f"<th>r vs qlook</th>{dhdr}</tr>{''.join(prows)}</table>")

    figs_html = "".join(
        f"<h3>{html.escape(Path(f).stem)}</h3>"
        f"<img src='data:image/png;base64,{b64(f)}' alt='{Path(f).name}'>"
        for f in figs)
    budget = json.dumps(config.get("budget", {}), indent=1)
    body = f"""
<h1>B26 firn core: measured vs simulated ({SEASON} {FRAME_ID})</h1>
<p class="note">{html.escape(notes)}</p>
<h2>Radargram panels + nadir profile</h2>
{figs_html}
<h2>Metrics</h2>
<table><tr><th>metric</th><th>value</th><th>criterion</th><th>note</th></tr>
{''.join(mrows)}</table>
<h2>Nadir band levels (dB rel own surface peak, closest-approach trace)</h2>
<table><tr><th>profile</th>{bhdr}</tr>{brows}</table>
<h2>Profile correlation and band deltas vs BOTH measured products</h2>
<p>r = Pearson correlation of the dB depth profile (5-{PROFILE_MAX_M:.0f} m);
the delta columns are profile level minus that measured product's level (dB;
negative = below the measurement). CSARP_qlook is the UNFOCUSED product, the
like-for-like target for the unfocused sims.</p>
{prof_tbl}
<h2>Run configuration</h2>
<table>{''.join(crows)}</table>
<h2>2019 season instrument parameters (from the frame's own product file)</h2>
<pre>{html.escape(json.dumps(params, indent=1))}</pre>
<h2>Compute budget (pilot projection vs actual)</h2>
<pre>{html.escape(budget)}</pre>
"""
    (out / "report.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>B26 comparison</title><style>{css}</style></head>"
        f"<body>{body}</body></html>")
    print(f"wrote {out / 'report.html'}")


SCENE_KEYS = ("n_traces", "ct_wide", "ct_firn", "along_m")
RUNSET_KEYS = ("layer_counts", "random_runs", "rough_runs", "eff_runs")


def _recorded_cfg(out, runsets=False):
    """Configuration of the CACHED runs (out/run_config.json).

    The SCENE keys are always restored: the pilot may have shrunk n_traces
    below the module default, and any mismatch changes every run cache key, so
    the tool would silently re-simulate everything instead of re-assembling.

    The RUN-SET keys (which runs exist) are restored only for --report-only,
    which reproduces exactly what is cached. --no-pilot deliberately takes them
    from the module defaults -- that is what makes it "the way to ADD a run"."""
    p = Path(out) / "run_config.json"
    if not p.exists():
        return {}
    c = json.loads(p.read_text())
    keys = SCENE_KEYS + (RUNSET_KEYS if runsets else ())
    return {k: c[k] for k in keys if k in c}


def _parse_only(s):
    """--only value -> set of run keys, or None when the flag is absent."""
    if not s:
        return None
    keys = {k.strip() for k in s.split(",") if k.strip()}
    if not keys:
        raise ValueError("--only given but empty")
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild figures/report from cached runs (reuses the "
                         "recorded run_config.json scene configuration)")
    ap.add_argument("--no-pilot", action="store_true",
                    help="skip the pilot/budget projection and reuse the "
                         "recorded run_config.json scene configuration -- the "
                         "way to ADD a run to an existing output directory "
                         "without risking a shrink step that would invalidate "
                         "every cached run")
    ap.add_argument("--only", default=None, metavar="KEY[,KEY...]",
                    help="restrict SIMULATION to these run keys (e.g. "
                         "firn_N40_h1eff,firn_N20_h1eff). Every other run is "
                         "assembled from its existing runs/<key>.npz AS-IS, "
                         "even if its recorded configuration is stale against "
                         "the current module defaults (recorded as "
                         "cache-stale in run_provenance); a missing one is an "
                         "error, never a silent multi-hour run. Implies "
                         "--no-pilot (the pilot itself simulates)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-traces", type=int, default=None)
    args = ap.parse_args()
    only = _parse_only(args.only)
    no_pilot = args.no_pilot or only is not None
    kw = (_recorded_cfg(OUT_DEFAULT, runsets=args.report_only)
          if (args.report_only or no_pilot) else {})
    if args.n_traces:
        kw["n_traces"] = args.n_traces
    run_all(do_pilot=not (args.report_only or no_pilot), force=args.force,
            only=only, **kw)


if __name__ == "__main__":
    main()
