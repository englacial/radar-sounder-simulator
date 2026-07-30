"""Platform-altitude comparison of a coherent surface+bed cluttergram.

Runs the COHERENT multilayer kernel (surface + BedMachine bed only -- NO firn /
internal layers) on one xOPR frame at several platform altitudes, holding the
frame's real MCoRDS chirp/window params fixed so ONLY the flight geometry
changes. It answers "how does the surface+bed radargram trade against platform
altitude" for instrument-engineering altitude studies.

Altitude levels (``--levels``, comma list; each item is one of):
  * ``real``   -- the recorded nav (frame Elevation), unchanged;
  * ``<N>agl`` -- terrain-following: same horizontal track, platform height =
                  (along-track ~1 km-smoothed ICE-SURFACE elevation) + N metres;
  * ``<N>msl`` -- constant ellipsoidal height N metres.

IMPORTANT DATUM NOTE: every height in this codebase is WGS84-ELLIPSOIDAL (PGC
DEMs, CReSIS Elevation, BedMachine bed+geoid). "MSL" here is therefore
APPROXIMATED by ellipsoidal height -- the geoid offset (tens of m in Greenland)
is neglected, which is fine for altitude trades but is NOT a true orthometric
MSL. The guard requires the platform to clear the ice surface everywhere; a
level that dips into the surface errors out.

Instrument model (M24 / run_b26_comparison surface+bed physics): the frame's
OWN chirp (f0/f1/Tpd from its param structs, read through xopr), pulse = the
frame's longest/bed waveform, its recorded pulse-compression window mapped
onto soundersim's supported set -- tukey(alpha<=0.3) is modeled as UNWINDOWED
(near-rectangular: ~-15 dB sidelobes / ~1.05x rect main lobe, far closer than
hann's -31.5 dB / 1.44x), tukey(alpha>0.3) as hann; any mapping is recorded
prominently (metrics + report + console warning). Uniform unsteered 7-element
0.5-lambda cross-track array with roll from nav; media air / ice(eps 3.17,
15 dB/km one-way) / bed(eps 8). Fast time is simulated alias-free at
dt_frame/k with k the FIRST of (4, 5, 6, 8, 10) putting the envelope-
quantization alias out of band (|f0 - round(f0*dt_sim)/dt_sim| > B/2; k=4 on
the 2019 P3 grid, k=6 on the 2012 DC8 grid), the simulate() alias warning
asserted silent, then decimated [::k] exactly onto a frame-dt grid.

PER ALTITUDE the geometry is re-derived: the twtt window (surface ~2h/c through
bed + margin), the facet spacing (beta=0.5 Fresnel criterion at that altitude's
nadir r_min -- higher altitude -> coarser facets -> cheaper; LPA nadir error
recorded), and the cross-track reach (sized to cover returns landing in the
window, capped at --ct-cap). Per-level resumable npz caches live in the out dir.

Deliverables (out dir, default outputs/altitude_comparison/<frame_id>/):
report.html + metrics.json (group "xOPR clutter", case "altitude_<frame_id>"),
radargram panels on a shared surface-referenced twtt axis, and a nadir
depth-power overlay. A copy of metrics.json + figures is mirrored under
outputs/verification/altitude_<frame_id>/ for tools/make_report.py.

Optional firn layers (``--firn N``): N effective-contrast layers from the
region-appropriate density core (B26 Greenland / B25 Antarctic REPRESENTATIVE
proxy), the run_b26_comparison standard methodology (soundersim.firn): segment
transfer-matrix reflectivities of the raw core profile as plain Fresnel
contrasts, conformal DEM offsets over [1 m, core zmax], 15 dB/km firn
attenuation, simulated on a narrow cross-track strip and field-summed with the
surface+bed run (firn surface layer excluded). Without --firn the surface+bed
path (and its caches) is untouched.

Run:
  uv run python tools/run_altitude_comparison.py \
      --season 2019_Greenland_P3 --frame 20190418_01_009 [--firn 10]
"""

import argparse
import base64
import datetime
import html
import json
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from affine import Affine  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_opr_coherent_bed as rocb  # noqa: E402  M24 constants + helpers
from run_opr_comparison import _db  # noqa: E402

from soundersim.config import (AntennaConfig, DemInterface, FacetConfig,  # noqa: E402
                               Medium, RadarConfig, SimConfig, WaveformConfig)
from soundersim.opr import (CACHE_DIR, fetch_bedmachine_window,  # noqa: E402
                            fill_nodata_nearest, frame_scene, load_bottom_pick,
                            load_frame, resample_to_grid)
from soundersim import firn  # noqa: E402
from soundersim.physics import fresnel_normal  # noqa: E402
from soundersim.simulate import simulate  # noqa: E402
from soundersim.synthetic import MultilayerScene, SyntheticScene  # noqa: E402

C = 299792458.0
# Physics / instrument constants (shared with the M24 surface+bed run).
EPS_ICE, EPS_BED = rocb.EPS_ICE, rocb.EPS_BED
ATT_DB_PER_KM = rocb.ATT_DB_PER_KM
BETA = rocb.BETA
N_ELEMENTS, SPACING_LAM = rocb.N_ELEMENTS, rocb.SPACING_LAM
OVERSAMPLE_CANDS = (4, 5, 6, 8, 10)   # dt_sim = dt_frame/k, first alias-free k
GATE_BINS = rocb.GATE_BINS

PRE_SURF_US, POST_BED_US = 0.8, 2.0   # twtt window margins around the geometry
CLEAR_MARGIN_M = 50.0                 # min platform clearance above the surface
SMOOTH_M = 1000.0                     # AGL surface-elevation smoothing window
DEFAULT_LEVELS = "real,1000agl,5000msl"
DYN_DB = 120.0                        # radargram colorscale span below reference
PROF_FLOOR_DB = -150.0                # nadir-profile y floor (dB rel surf peak)

OUT_DEFAULT = ROOT / "outputs" / "altitude_comparison"
VER_ROOT = ROOT / "outputs" / "verification"

# ---- optional firn layers (--firn N; the b26 effective-contrast standard) ---
# Model layers carry the segment-aggregate transfer-matrix reflectivity of the
# region-appropriate core's RAW density profile (soundersim.firn; point-sampled
# eps is deprecated, ~12 dB weak in the 20-70 m band), placed as conformal
# OffsetInterfaces of the surface DEM over [1 m, core zmax], firn media
# attenuating at the ice medium's 15 dB/km. The firn contribution runs on a
# NARROW cross-track strip (off-nadir firn returns are sinc-suppressed) in
# along-track chunks and is field-summed with the wide surface+bed run,
# EXCLUDING the firn run's own surface layer (no double count) -- the
# run_b26_comparison construction. Unlike b26 the firn strip has its OWN facet
# spacing (the surface+bed caches predate --firn and must stay valid), so the
# exact-lattice seam check is replaced by a recorded gamma-scaled surface-layer
# agreement diagnostic.
CT_FIRN = 600.0                       # firn-strip cross-track reach (m)
FIRN_EFF_METHOD = "tmm_segment_aggregate_v1"
FIXDIR = ROOT / "tests" / "fixtures" / "firn"
FIRN_CORES = {  # hemisphere -> (fixture, label, proxy caveat or None)
    "greenland": ("ngt37C95.2_density.tab", "B26 (ngt37C95.2)", None),
    "antarctica": ("BER11C95_25_density.tab", "B25 (BER11C95_25)",
                   "B25 (Berkner Island summit) is a REPRESENTATIVE Antarctic "
                   "firn proxy: this frame does not pass a cored site, so the "
                   "firn stack is a plausible stand-in, not site truth"),
}
# Optional sub-facet SURFACE roughness (--surf-rough): Gerekos-2023 rough-facet
# response on the wide run's surface interface, at the C&S 2020 Fig. 11 mcords
# inversion's shallow (0 m) clamp -- tests/fixtures/firn/fig11a/b first row --
# a REPRESENTATIVE cm-scale snow-surface roughness the 32 m DEM lacks, not a
# site measurement. Iteration-validated 2026-07-29 on both frames (mean-power
# metric, with-firn): 2012 real corr 0.867 -> 0.917, 20-70 m delta -13.1 ->
# -9.6 dB; 2019 real corr 0.921 -> 0.930, -9.8 -> -7.7 dB. Default OFF: the
# smooth path and its caches are byte-identical; rough runs cache under their
# own rids (level_<spec>_srough[_firnN]).
SURF_ROUGH_SIGMA_M = 0.049474             # fig11a rms_height_mcords_m @ 0 m
SURF_ROUGH_CL_M = 2.982179                # fig11b corr_length_mcords_m @ 0 m
BAND_EDGES_M = (5.0, 20.0, 60.0, 120.0)   # firn band-level diagnostics (b26)
EXTRA_BANDS = ((20.0, 70.0), (80.0, 120.0))
GAP_BAND = "20-70m"
PROFILE_MAX_M = 200.0
SEAM_WIN_US = 1.5


# ========================================================================
# per-frame MCoRDS parameters (M24 method: the frame's own product structs)
# ========================================================================
def _wfs_field(wfs, key):
    """One waveform field as a flat float array, across the two product
    encodings: dict-of-arrays (2017/2019) or list-of-dicts (2012-era)."""
    if isinstance(wfs, dict):
        return np.asarray(wfs[key], np.float64).ravel()
    return np.concatenate([np.asarray(w[key], np.float64).ravel()
                           for w in wfs])


def _window_string(fw):
    """Compression-window string from an xopr-decoded ft_wind value: MATLAB
    function handles ({'function_handle': {'function': 'hanning'}}) or inline
    functions ({'expr': 'tukeywin(N,0.2)'}), possibly wrapped in nested
    object ndarrays (one per waveform; the first is taken)."""
    while isinstance(fw, np.ndarray):
        fw = fw.ravel()[0]
    if isinstance(fw, dict):
        if "function_handle" in fw:
            return str(fw["function_handle"]["function"])
        if "expr" in fw:
            return str(fw["expr"])
    return str(fw)


# (label, radar-struct getter, ft_wind getter) tried IN ORDER: 2017/2019-era
# products carry params under param_records/param_sar; 2012-era CSARP products
# under param_csarp (radar.wfs is then a list of per-waveform dicts).
_PARAM_LAYOUTS = (
    ("param_sar",
     lambda a: a["param_records"]["radar"],
     lambda a: a["param_sar"]["radar"]["wfs"]["ft_wind"]),
    ("param_csarp",
     lambda a: a["param_csarp"]["radar"],
     lambda a: a["param_csarp"]["csarp"]["ft_wind"]),
)


def mcords_params(season, frame_id):
    """Read the frame's MCoRDS chirp/window params from its own param structs,
    obtained THROUGH XOPR (no direct .mat download in this tool): xopr's
    ``load_frame_url`` decodes every MATLAB param struct of the product file
    into ``ds.attrs`` as nested dicts. Two product layouts are tried in order
    (_PARAM_LAYOUTS): param_records.radar + param_sar ft_wind (2017/2019-era)
    then param_csarp.radar + param_csarp.csarp.ft_wind (2012-era). An
    already-cached source .mat is handed to xopr as a local path (offline);
    otherwise the frame is resolved via the STAC catalog (query_frames +
    load_frame, xopr's own fsspec file cache does the transfer). Cached as a
    provenance JSON; the xopr-derived waveform values were verified IDENTICAL
    to the previous direct-h5py (M24) reader's on the 2019 frame."""
    prov = CACHE_DIR / f"mcords_params_{season}_{frame_id}.json"
    if prov.exists():
        return json.loads(prov.read_text())

    import xopr

    conn = xopr.OPRConnection(cache_dir=str(CACHE_DIR / "xopr"))
    mat = CACHE_DIR / f"Data_{frame_id}_source.mat"
    if mat.exists():  # cached product file: parse offline through xopr
        ds = conn.load_frame_url(str(mat))
        src = f"outputs/cache/{mat.name} (parsed via xopr.load_frame_url)"
    else:
        date, seg, num = frame_id.split("_")
        items = conn.query_frames(collections=[season],
                                  segment_paths=[f"{date}_{seg}"],
                                  properties={"opr:frame": int(num)})
        if items is None or len(items) == 0:
            raise LookupError(f"frame {frame_id} not found in {season}")
        ds = conn.load_frame(items.iloc[0], data_product="CSARP_standard")
        src = ("CSARP_standard STAC asset via xopr.load_frame "
               "(xopr fsspec file cache)")

    a = ds.attrs
    radar = wfs = None
    for label, get_radar, get_ftw in _PARAM_LAYOUTS:
        try:
            radar = get_radar(a)
            wfs = radar["wfs"]
            _wfs_field(wfs, "f0")  # probe the layout end-to-end
        except (KeyError, TypeError, IndexError):
            radar = wfs = None
            continue
        try:
            ft_wind = _window_string(get_ftw(a))
        except Exception:
            ft_wind = "hanning (ft_wind decode failed; CReSIS readme default)"
        layout = label
        break
    if radar is None:
        raise LookupError(
            f"no known param layout in {frame_id}: tried "
            + ", ".join(lay[0] for lay in _PARAM_LAYOUTS)
            + f"; product attrs: {sorted(k for k in a if 'param' in k)}")

    def uniq(key):
        return np.unique(_wfs_field(wfs, key)).tolist()

    f0, f1, tpd = uniq("f0"), uniq("f1"), uniq("Tpd")
    tukey = float(uniq("tukey")[0])
    prf = float(np.asarray(radar["prf"], np.float64).ravel()[0])
    tw = np.asarray(ds.twtt.values, np.float64)
    dt_prod = float(f"{(tw[-1] - tw[0]) / (len(tw) - 1) * 1e9:.3f}e-9")
    doc = {
        "purpose": (f"altitude-comparison instrument provenance for {season} "
                    f"{frame_id}, from the frame's own param structs "
                    f"(M24 fields, obtained through xopr)"),
        "source": src,
        "waveform": {
            "f0_f1_Hz": [f0, f1],
            "center_frequency_Hz": float((f0[0] + f1[0]) / 2.0),
            "bandwidth_Hz": float(f1[0] - f0[0]),
            "pulse_lengths_s": tpd,
            "bed_waveform_pulse_length_s": float(max(tpd)),
            "tukey_time_window": tukey,
            "pulse_compression_freq_window": ft_wind + f" ({layout} ft_wind)",
            "prf_Hz": prf, "product_dt_s": dt_prod,
        },
        "antenna": {"modeled_as": (
            f"uniform unsteered {N_ELEMENTS}-element {SPACING_LAM}-lambda "
            "cross-track array, roll_source=nav (M24 convention)")},
    }
    prov.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def map_window(ft_wind_str):
    """(soundersim window, approximation note or None) for a recorded
    pulse-compression-window string. soundersim.waveform supports only
    none/hann/hamming: hanning/hamming map directly; tukey(alpha<=0.3) maps
    to "none" (tukey 0.2 is near-rectangular: ~-15 dB sidelobes, ~1.05x rect
    main lobe -- far closer to rect than to hann's -31.5 dB / 1.44x) and
    tukey(alpha>0.3) to "hann". Proper tukey support in soundersim.waveform
    is future work; the mapping is recorded prominently."""
    s = ft_wind_str.lower()
    m = re.search(r"tukeywin\s*\([^,)]*,\s*([0-9.]+)\s*\)", s)
    if m:
        alpha = float(m.group(1))
        if alpha <= 0.3:
            return "none", (
                f"MODELED-WINDOW APPROXIMATION: the product's compression "
                f"window is tukey(alpha={alpha:g}) but soundersim.waveform "
                f"supports only none/hann/hamming; modeled as UNWINDOWED "
                f"(rect). tukey({alpha:g}) is near-rectangular (~-15 dB "
                f"sidelobes, ~1.05x rect main-lobe width), far closer to "
                f"rect than to hann (-31.5 dB, 1.44x); expect the simulated "
                f"range sidelobes to be modestly optimistic (rect -13.3 dB)")
        return "hann", (
            f"MODELED-WINDOW APPROXIMATION: tukey(alpha={alpha:g}) modeled "
            f"as hann (nearest supported window at this alpha)")
    if "hamming" in s:
        return "hamming", None
    if "hann" in s:  # 'hanning' / 'hann'
        return "hann", None
    return "hann", (f"MODELED-WINDOW APPROXIMATION: unrecognized compression "
                    f"window {ft_wind_str!r}; modeled as hann")


def pick_oversample(dt, f0, bandwidth, cands=OVERSAMPLE_CANDS):
    """First k with the envelope-quantization alias of dt_sim = dt/k out of
    band: |f0 - round(f0*dt_sim)/dt_sim| > B/2 (the M21/M24 rule; e.g. k=4
    on the 2019 P3 16.667 ns grid, k=6 on the 2012 DC8 105.21 ns grid).
    Returns (k, f_alias_hz)."""
    for k in cands:
        dts = dt / k
        f_alias = abs(f0 - round(f0 * dts) / dts)
        if f_alias > bandwidth / 2.0:
            return k, f_alias
    raise SystemExit(
        f"no alias-free oversample in {cands} for dt={dt*1e9:.4f} ns, "
        f"f0={f0/1e6:.2f} MHz, B={bandwidth/1e6:.2f} MHz")


# ========================================================================
# geometry helpers
# ========================================================================
def _lonlat(frame):
    lon = np.asarray(frame.Longitude.values, np.float64)
    return (np.asarray(frame.Latitude.values, np.float64),
            np.where(lon > 180.0, lon - 360.0, lon))


def sub_frame(frame, along_m):
    """(fsub, info): frame sliced to ~along_m of track centered on the frame
    middle; info has the along-track arc-length axis and slice bounds. The
    arc length uses the hemisphere's polar stereographic CRS."""
    lat, lon = _lonlat(frame)
    crs = "EPSG:3413" if float(lat.mean()) > 0 else "EPSG:3031"
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    px, py = tr.transform(lon, lat)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    i0 = len(s) // 2
    a = int(np.searchsorted(s, s[i0] - along_m / 2.0))
    b = int(np.searchsorted(s, s[i0] + along_m / 2.0))
    fsub = frame.isel(slow_time=slice(a, b))
    info = {"i0_local": i0 - a, "slice": (a, b),
            "s_rel_m": s[a:b] - s[i0], "track_len_m": float(s[b - 1] - s[a])}
    return fsub, info


def base_scene(fsub, n_traces, ct_dist):
    """Surface (ArcticDEM 32 m) + BedMachine bed MultilayerScene at ``ct_dist``
    (cache-first fetches); the M24 _bed_scene structure. Nav uses the REAL
    elevation -- per-level altitudes override nav z downstream."""
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
    ms = MultilayerScene(scene.name + "_bed", [scene.dem, bed], scene.transform,
                         scene.crs, scene.nav_llh, media,
                         {**scene.params, "bed_product": meta["product"]})
    ms.nav_roll = scene.nav_roll
    aux = {"idx": idx, "bed_meta": meta, "bed_fill": bed_fill,
           "clamp_frac": clamp_frac, "surf_fill": info["fill_fraction"]}
    return ms, aux


def crop_scene(base, ct_dist, nav_z, name):
    """Crop the base scene's DEM stack to the track bbox padded by ct_dist, and
    override the platform height (nav z) -- one MultilayerScene per level with
    NO extra network (all levels share the base's cached windows)."""
    tr = Transformer.from_crs("EPSG:4326", base.crs, always_xy=True)
    px, py = tr.transform(base.nav_llh[:, 1], base.nav_llh[:, 0])
    pad = ct_dist + 100.0
    ny, nx = base.dem.shape
    cols, rows = (~base.transform) * (
        np.array([px.min() - pad, px.max() + pad]),
        np.array([py.min() - pad, py.max() + pad]))
    c0 = int(np.clip(np.floor(min(cols)), 0, nx - 2))
    c1 = int(np.clip(np.ceil(max(cols)) + 1, c0 + 2, nx))
    r0 = int(np.clip(np.floor(min(rows)), 0, ny - 2))
    r1 = int(np.clip(np.ceil(max(rows)) + 1, r0 + 2, ny))
    dems = [np.ascontiguousarray(d[r0:r1, c0:c1]) for d in base.dems]
    tr_c = base.transform * Affine.translation(c0, r0)
    nav = base.nav_llh.copy()
    nav[:, 2] = nav_z
    sc = MultilayerScene(name, dems, tr_c, base.crs, nav, base.media,
                         {**base.params, "ct_dist": ct_dist})
    sc.nav_roll = getattr(base, "nav_roll", None)
    return sc


# ========================================================================
# per-level radar grid, facet spacing, cross-track
# ========================================================================
def facet_spacing(lam, r_min, thickness):
    """beta=0.5 Fresnel spacing minimized over the surface (lam_air, r_min) and
    the bed (in-ice lam, r_min + thickness), snapped down to a 32 m divisor."""
    cands = [lam * r_min, (lam / np.sqrt(EPS_ICE)) * (r_min + thickness)]
    s = float(BETA * np.sqrt(min(cands)))
    return 32.0 / np.ceil(32.0 / s) if s < 32.0 else s


def radar_grid(params, surf_tw, bed_tw, dt, t0f, oversample, window):
    """(rc_sim, rc_frame, b0) for a level: alias-free dt_frame/oversample grid
    anchored on a frame-dt bin, covering [min surface twtt - margin, max bed
    twtt + margin]. Decimating the sim [::oversample] lands on this frame-dt
    grid. ``window`` is the mapped soundersim compression window."""
    lo = float(np.nanmin(surf_tw)) - PRE_SURF_US * 1e-6
    hi = float(np.nanmax(bed_tw)) + POST_BED_US * 1e-6
    b0 = int(np.floor((lo - t0f) / dt))
    nb = int(np.ceil((hi - t0f) / dt)) - b0 + 1
    t0 = t0f + b0 * dt
    wf = params["waveform"]
    wave = WaveformConfig(kind="chirp", bandwidth=wf["bandwidth_Hz"],
                          pulse_length=wf["bed_waveform_pulse_length_s"],
                          window=window)
    ant = AntennaConfig(kind="array", n_elements=N_ELEMENTS,
                        spacing_lam=SPACING_LAM, roll_source="nav")
    f0 = wf["center_frequency_Hz"]
    rc_sim = RadarConfig(dt=dt / oversample,
                         n_samples=oversample * (nb - 1) + 1,
                         t0=t0, f0=f0, waveform=wave, antenna=ant)
    rc_frame = RadarConfig(dt=dt, n_samples=nb, t0=t0, f0=f0)
    return rc_sim, rc_frame, b0


def bed_cfg(rc_sim, spacing, surf_rough=False):
    """Surface+bed config; ``surf_rough`` attaches the representative C&S
    Fig. 11 shallow-clamp sub-facet roughness to the SURFACE interface only
    (the bed stays smooth; see the SURF_ROUGH_* provenance note)."""
    from soundersim.config import RoughnessConfig
    rc = (RoughnessConfig(sigma_m=SURF_ROUGH_SIGMA_M,
                          corr_length_m=SURF_ROUGH_CL_M) if surf_rough
          else None)
    return SimConfig(mode="coherent", split_sides=False, radar=rc_sim,
                     facets=FacetConfig(spacing=spacing),
                     media=[Medium(name="air", eps_r=1.0),
                            Medium(name="ice", eps_r=EPS_ICE,
                                   attenuation_db_per_km=ATT_DB_PER_KM),
                            Medium(name="bed", eps_r=EPS_BED)],
                     interfaces=[DemInterface(name="surface", roughness=rc),
                                 DemInterface(name="bed")])


def _n_facets(dem_shape, spacing):
    ny, nx = dem_shape
    f = 32.0 / spacing
    nrv = max(2, int(round((ny - 1) * f)) + 1)
    ncv = max(2, int(round((nx - 1) * f)) + 1)
    return (nrv - 1) * (ncv - 1)


# ========================================================================
# firn layers (--firn N)
# ========================================================================
def firn_core_for(lat_mean):
    """(FirnCore, region, label, proxy_note) for the frame's hemisphere."""
    region = "greenland" if lat_mean > 0 else "antarctica"
    fname, label, note = FIRN_CORES[region]
    return firn.FirnCore(FIXDIR / fname), region, label, note


def firn_facet_spacing(lam, r_min, core):
    """beta=0.5 Fresnel spacing for the firn strip, minimized over the surface
    (lam, r_min) and the deepest firn layer (in-firn lam, r_min + zmax; it
    binds), snapped down to a 32 m divisor. Independent of the surface+bed
    run's spacing (whose caches predate --firn and must stay valid)."""
    cands = [lam * r_min,
             lam / np.sqrt(core.point_eps(core.zmax)) * (r_min + core.zmax)]
    s = float(BETA * np.sqrt(min(cands)))
    return 32.0 / np.ceil(32.0 / s) if s < 32.0 else s


def firn_strip_scenes(base, ct_firn, nav_z, n_chunks=None):
    """Narrow-strip SURFACE-ONLY scenes for the firn contribution, cropped
    from the base scene's surface DEM in ALONG-TRACK CHUNKS (the b26
    firn_scenes construction: one bbox around a long diagonal track would
    carry many times the intended +-ct_firn strip area; each chunk's DEM is
    its own traces' bbox padded by ct_firn + 100 m so every trace keeps full
    +-ct_firn coverage). Platform height overridden to ``nav_z`` per level.
    Returns [(SyntheticScene, trace_rows), ...]."""
    n_traces = len(base.nav_llh)
    n_chunks = n_chunks or max(1, round(n_traces / 17))
    tr = Transformer.from_crs("EPSG:4326", base.crs, always_xy=True)
    px, py = tr.transform(base.nav_llh[:, 1], base.nav_llh[:, 0])
    pad = ct_firn + 100.0
    ny, nx = base.dem.shape
    out = []
    for rows_idx in np.array_split(np.arange(n_traces), n_chunks):
        x, y = px[rows_idx], py[rows_idx]
        cols, rows = (~base.transform) * (
            np.array([x.min() - pad, x.max() + pad]),
            np.array([y.min() - pad, y.max() + pad]))
        c0 = int(np.clip(np.floor(min(cols)), 0, nx - 2))
        c1 = int(np.clip(np.ceil(max(cols)) + 1, c0 + 2, nx))
        r0 = int(np.clip(np.floor(min(rows)), 0, ny - 2))
        r1 = int(np.clip(np.ceil(max(rows)) + 1, r0 + 2, ny))
        dem = np.ascontiguousarray(base.dems[0][r0:r1, c0:c1])
        tr_c = base.transform * Affine.translation(c0, r0)
        nav = base.nav_llh[rows_idx].copy()
        nav[:, 2] = nav_z[rows_idx]
        roll = getattr(base, "nav_roll", None)
        sc = SyntheticScene(
            f"{base.name}_firnstrip{rows_idx[0]}", dem, tr_c, base.crs, nav,
            {**base.params, "ct_dist_firn": ct_firn},
            nav_roll=None if roll is None else np.asarray(roll)[rows_idx])
        out.append((sc, rows_idx))
    return out


def run_firn_level(rid, chunks, cfg, meta, runs_dir, oversample, force=False):
    """Cached chunked firn simulate() for one level (the b26 run_sim shape):
    decimated per-layer complex fields assembled over all traces into
    runs/<rid>.npz/.json, keyed on ``meta``. Alias warning asserted silent;
    a non-finite field fails BEFORE the cache is written."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    jp, npz_p = runs_dir / f"{rid}.json", runs_dir / f"{rid}.npz"
    key = json.dumps(meta, sort_keys=True)
    if jp.exists() and npz_p.exists() and not force:
        diag = json.loads(jp.read_text())
        if diag.get("meta_key") == key:
            print(f"  [skip-exists] {rid} ({diag['wall_s']:.1f} s recorded)",
                  flush=True)
            return diag, dict(np.load(npz_p))
    n_traces = sum(len(rows) for _, rows in chunks)
    field = twtt = None
    msgs_all, facets, wall = [], [], 0.0
    for scene, rows in chunks:
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = simulate(scene, cfg)
        wall += time.perf_counter() - t
        msgs = [str(w.message) for w in caught]
        if any("alias" in m for m in msgs):
            raise RuntimeError(f"in-band-alias warning fired for {rid}: {msgs}")
        msgs_all += msgs
        ds_dec = ds.isel(twtt=slice(None, None, oversample))
        f = np.asarray(ds_dec.field.values, np.complex64)   # (t, nb, L)
        if field is None:
            field = np.zeros((n_traces,) + f.shape[1:], np.complex64)
            twtt = ds_dec.twtt.values
        field[rows] = f
        facets.append(_n_facets(scene.dem.shape, cfg.facets.spacing))
    if not np.isfinite(field).all():
        raise RuntimeError(f"{rid}: non-finite field")
    diag = {"rid": rid, "wall_s": round(wall, 2), "meta_key": key, "meta": meta,
            "n_chunks": len(chunks), "n_facets_per_interface_per_chunk": facets,
            "warnings": msgs_all}
    arrs = dict(field=field, twtt=twtt)
    np.savez_compressed(npz_p, **arrs)
    jp.write_text(json.dumps(diag, indent=1) + "\n")
    print(f"  [ok] {rid}  {wall:.1f} s  chunks {len(chunks)} "
          f"facets/iface/chunk ~{int(np.mean(facets))}", flush=True)
    return diag, arrs


def firn_seam(E_wide_surf, E_firn_surf, twtt, t_surf, dt, eps_firn0):
    """Recorded diagnostic (NOT the b26 exact-lattice gate): median over traces
    of the max relative deviation between the firn run's gamma-scaled surface
    field and the wide run's surface field in [t_surf-0.3us, t_surf+1.5us].
    The two runs use different facet spacings here, so agreement is
    approximate away from the specular peak."""
    ratio = fresnel_normal(1.0, EPS_ICE) / fresnel_normal(1.0, eps_firn0)
    rel = []
    for t in range(E_wide_surf.shape[0]):
        if not np.isfinite(t_surf[t]):
            continue
        a = int(np.clip((t_surf[t] - 0.3e-6 - twtt[0]) / dt, 0, len(twtt) - 2))
        b = int(np.clip((t_surf[t] + SEAM_WIN_US * 1e-6 - twtt[0]) / dt,
                        a + 1, len(twtt)))
        den = np.abs(E_wide_surf[t, a:b]).max()
        if den > 0:
            rel.append(float(np.abs(E_firn_surf[t, a:b] * ratio
                                    - E_wide_surf[t, a:b]).max() / den))
    return float(np.median(rel)) if rel else float("nan")


def profile_vs_depth(power, twtt, t_surf, dt, eps_mean, smooth_m=5.0):
    """(depth_m, dB rel surface peak): twtt below the surface peak converted
    with c/sqrt(eps_mean) (the b26 convention), 5 m boxcar."""
    bin_depth = C * dt / (2.0 * np.sqrt(eps_mean))
    w = max(int(round(smooth_m / bin_depth)) | 1, 3)
    ps = np.convolve(power, np.ones(w) / w, mode="same")
    i0 = int(np.clip(np.searchsorted(twtt, t_surf) - int(0.3e-6 / dt), 0,
                     len(twtt) - 2))
    i1 = int(np.clip(np.searchsorted(twtt, t_surf + 1.0e-6), i0 + 1, len(twtt)))
    pk = ps[i0:i1].max()
    depth = (twtt - t_surf) * C / (2.0 * np.sqrt(eps_mean))
    db = 10.0 * np.log10(np.maximum(ps / max(pk, 1e-300), 1e-15))
    return depth, db


def band_levels(depth, db, edges=BAND_EDGES_M, extra=EXTRA_BANDS):
    out = {}
    for lo, hi in list(zip(edges[:-1], edges[1:])) + list(extra):
        m = (depth >= lo) & (depth < hi)
        out[f"{lo:.0f}-{hi:.0f}m"] = (float(np.median(db[m])) if m.any()
                                      else float("nan"))
    return out


def profile_corr(ref, other, lo=5.0, hi=PROFILE_MAX_M):
    """Pearson r of dB depth profiles, ``other`` interpolated onto ref's depth
    axis over [lo, hi] m."""
    d_r, db_r = ref
    m = (d_r >= lo) & (d_r <= hi)
    d, db = other
    return float(np.corrcoef(db_r[m], np.interp(d_r[m], d, db))[0, 1])


def mean_power_profile(P, twtt, t_guess, dt, eps_mean, pre_us=0.5,
                       smooth_m=5.0):
    """(depth_m, dB rel surface peak): MEAN POWER over all traces, each
    aligned on its own surface-peak bin and normalized to its own surface
    peak, then 5 m boxcar. The b26 findings' fair-metric convention:
    speckle-averaged, so band medians and correlations are stable across
    levels (a single trace's deep inter-layer nulls are realization noise).
    Kept alongside the single-trace profile_vs_depth for comparability."""
    tpk = surface_peak_twtt(P, twtt, t_guess, dt)
    k0 = int(round(pre_us * 1e-6 / dt))
    n_rel = k0 + P.shape[1]
    acc, cnt = np.zeros(n_rel), np.zeros(n_rel)
    for t in range(P.shape[0]):
        if not np.isfinite(tpk[t]):
            continue
        pk = int(round((tpk[t] - twtt[0]) / dt))
        w = float(P[t, max(0, pk - 31):pk + 32].max())
        if w <= 0:
            continue
        a = max(0, pk - k0)
        seg = P[t, a:] / w
        off = k0 - (pk - a)
        acc[off:off + len(seg)] += seg
        cnt[off:off + len(seg)] += 1
    prof = acc / np.maximum(cnt, 1)
    bin_d = C * dt / (2.0 * np.sqrt(eps_mean))
    wln = max(int(round(smooth_m / bin_d)) | 1, 3)
    ps = np.convolve(prof, np.ones(wln) / wln, mode="same")
    depth = (np.arange(n_rel) - k0) * bin_d
    pk0 = ps[np.abs(depth) < 3.0].max()
    db = 10.0 * np.log10(np.maximum(ps / max(pk0, 1e-300), 1e-15))
    return depth, db


def _firn_analysis(fctx, results, frame, fsub, sinfo, j0):
    """Per-level firn assembly + nadir depth-power comparison vs measured.
    PRIMARY profile/metric: trace-averaged mean power (mean_power_profile,
    the b26 fair-metric convention -- adopted 2026-07-29 after it raised the
    with-firn correlation at every level and stabilized the band deltas);
    the single representative-trace profile (b26 profile_vs_depth pattern)
    is recorded alongside as repr_trace. Mutates each result: E_comb =
    surface+bed + firn internal layers (firn layer 0 excluded -- no surface
    double count); firn_profiles for the figure. Returns the firn_doc dict."""
    em = fctx["eps_mean"]
    tw_full = frame.twtt.values
    dt_full = float(tw_full[1] - tw_full[0])
    i0 = sinfo["i0_local"]
    surf_nat = np.asarray(fsub.Surface.values, np.float64)
    meas_all = np.asarray(fsub.Data.values, np.float64)
    prof_meas = mean_power_profile(meas_all, tw_full, surf_nat, dt_full, em)
    t_s = surface_peak_twtt(meas_all[[i0]], tw_full,
                            np.array([surf_nat[i0]]), dt_full)[0]
    prof_meas_1 = profile_vs_depth(meas_all[i0], tw_full, t_s, dt_full, em)
    mb, mb1 = band_levels(*prof_meas), band_levels(*prof_meas_1)
    doc = {"measured_bands": {b: round(v, 2) for b, v in mb.items()},
           "prof_measured": prof_meas, "per_level": {}}
    for spec, r in results.items():
        tw, dt_l = r["arrs"]["twtt"], r["rc_frame"].dt
        E2 = r["arrs"]["field"].sum(-1)
        Ef = r["firn"]["arrs"]["field"]
        E3 = E2 + Ef[..., 1:].sum(-1)
        r["E_comb"] = E3
        nad0 = r["arrs"]["nadir_twtt"][:, 0]
        seam = firn_seam(r["arrs"]["field"][..., 0], Ef[..., 0], tw, nad0,
                         dt_l, float(fctx["eps"][0]))
        pr, pr1 = {}, {}
        for name, E in (("surface+bed", E2), ("with_firn", E3)):
            P = np.abs(E) ** 2
            pr[name] = mean_power_profile(P, tw, nad0, dt_l, em)
            ts = surface_peak_twtt(P[[j0]], tw, np.array([nad0[j0]]), dt_l)[0]
            pr1[name] = profile_vs_depth(P[j0], tw, ts, dt_l, em)
        r["firn_profiles"] = pr
        bands = {k: band_levels(*v) for k, v in pr.items()}
        bands1 = {k: band_levels(*v) for k, v in pr1.items()}
        doc["per_level"][spec] = {
            "seam_rel": round(seam, 5),
            "spacing_firn_m": round(r["firn"]["spacing"], 3),
            "wall_s_firn": r["firn"]["diag"]["wall_s"],
            "bands_db": {k: {b: round(x, 2) for b, x in v.items()}
                         for k, v in bands.items()},
            "delta_vs_measured_db": {
                k: {b: round(v[b] - mb[b], 2) for b in v}
                for k, v in bands.items()},
            "corr_vs_measured": {k: round(profile_corr(prof_meas, v), 4)
                                 for k, v in pr.items()},
            "repr_trace": {
                "delta_vs_measured_db": {
                    k: {b: round(v[b] - mb1[b], 2) for b in v}
                    for k, v in bands1.items()},
                "corr_vs_measured": {k: round(profile_corr(prof_meas_1, v), 4)
                                     for k, v in pr1.items()}}}
    return doc


def _firn_metric(firn_doc, firn_n, fctx, order):
    pl = firn_doc["per_level"]
    first = order[0]
    return {
        "value": pl[first]["delta_vs_measured_db"]["with_firn"][GAP_BAND],
        "threshold": None, "op": "record", "pass": True,
        "band": GAP_BAND, "n_layers": firn_n, "core": fctx["label"],
        "measured_bands_db": firn_doc["measured_bands"],
        "per_level_delta_vs_measured_db": {
            s: v["delta_vs_measured_db"] for s, v in pl.items()},
        "per_level_corr_vs_measured": {
            s: v["corr_vs_measured"] for s, v in pl.items()},
        "per_level_seam_rel": {s: v["seam_rel"] for s, v in pl.items()},
        "note": f"nadir depth-power (dB rel own surface peak, MEAN POWER over "
        f"all traces each aligned/normalized on its own surface peak -- the "
        f"b26 fair-metric convention; single-trace values recorded under "
        f"repr_trace in run_config.json) minus measured, in the {GAP_BAND} "
        f"firn band, for the {first} level WITH the N={firn_n} "
        f"effective-contrast firn stack ({fctx['label']}); per-level before "
        f"(surface+bed) / after (with_firn) deltas, Pearson r "
        f"(5-{PROFILE_MAX_M:.0f} m) and the approximate gamma-scaled seam "
        f"diagnostic recorded alongside. "
        + (f"CAVEAT: {fctx['proxy_note']}. " if fctx["proxy_note"] else "")
        + "recorded only"}


def _firn_figure(out, results, order, fctx, firn_doc):
    """Per-level nadir depth-power panels: measured vs surface+bed vs
    +firn (the firn zone is where the stack should lift the sim toward the
    measured near-surface falloff)."""
    zmax = fctx["core"].zmax
    x_hi = min(PROFILE_MAX_M, zmax + 40.0)
    ncols = min(2, len(order))
    nrows = -(-len(order) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.6 * nrows),
                            sharex=True, sharey=True, squeeze=False)
    axs = axs.ravel()
    for ax in axs[len(order):]:
        ax.set_visible(False)
    d_m, db_m = firn_doc["prof_measured"]
    for k, spec in enumerate(order):
        ax = axs[k]
        r = results[spec]
        ax.plot(d_m, db_m, "k", lw=1.8, label="measured")
        d, db = r["firn_profiles"]["surface+bed"]
        ax.plot(d, db, color="0.55", lw=1.2, ls="--", label="surface+bed")
        d, db = r["firn_profiles"]["with_firn"]
        ax.plot(d, db, color="tab:blue", lw=1.4,
                label=f"+firn N={len(fctx['depths'])}")
        ax.axvline(zmax, color="tab:red", lw=0.7, ls=":",
                   label=f"core end {zmax:.0f} m")
        ax.set_xlim(0, x_hi)
        ax.set_ylim(-90, 3)
        ax.grid(alpha=0.3)
        ax.set_title(f"{spec} (AGL {r['h_med']:.0f} m)", fontsize=10)
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    for c in range(0, len(order), ncols):
        axs[c].set_ylabel("dB rel surface peak (mean power)")
    for ax in axs[max(0, len(order) - ncols):len(order)]:
        ax.set_xlabel(f"depth (m, c/sqrt({fctx['eps_mean']:.2f}))")
    fig.suptitle(f"nadir depth-power (trace-averaged) vs measured: "
                 f"{fctx['label']} effective-contrast firn stack", fontsize=11)
    fig.tight_layout()
    fp = out / "firn_profiles.png"
    fig.savefig(fp, dpi=140)
    plt.close(fig)
    return fp


# ========================================================================
# cached per-level simulation (resumable)
# ========================================================================
def run_level(rid, scene, cfg, meta, runs_dir, oversample, force=False):
    """Cached coherent surface+bed simulate() for one level: decimated per-layer
    complex field + nadir_twtt into runs/<rid>.npz/.json, keyed on ``meta``.
    The alias warning is asserted silent."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    jp, npz_p = runs_dir / f"{rid}.json", runs_dir / f"{rid}.npz"
    key = json.dumps(meta, sort_keys=True)
    if jp.exists() and npz_p.exists() and not force:
        diag = json.loads(jp.read_text())
        if diag.get("meta_key") == key:
            print(f"  [skip-exists] {rid} ({diag['wall_s']:.1f} s recorded)",
                  flush=True)
            return diag, dict(np.load(npz_p))
    t = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = simulate(scene, cfg)
    wall = time.perf_counter() - t
    msgs = [str(w.message) for w in caught]
    if any("alias" in m for m in msgs):
        raise RuntimeError(f"in-band-alias warning fired for {rid}: {msgs}")
    ds_dec = ds.isel(twtt=slice(None, None, oversample))
    field = np.asarray(ds_dec.field.values, np.complex64)   # (T, nb, 2)
    nadir = np.asarray(ds.nadir_twtt.values, np.float64)     # (T, 2)
    drop = np.asarray(ds.dropped_power.values, np.float64)
    tot = np.asarray(ds.power.values, np.float64).sum((0, 1)) + drop.sum(0)
    diag = {"rid": rid, "wall_s": round(wall, 2), "meta_key": key, "meta": meta,
            "n_facets_per_interface": _n_facets(scene.dem.shape, cfg.facets.spacing),
            "n_samples": int(field.shape[1]),
            "dropped_power_fraction":
                (drop.sum(0) / np.maximum(tot, 1e-300)).tolist(),
            "warnings": msgs}
    arrs = dict(field=field, twtt=ds_dec.twtt.values, nadir_twtt=nadir)
    np.savez_compressed(npz_p, **arrs)
    jp.write_text(json.dumps(diag, indent=1) + "\n")
    print(f"  [ok] {rid}  {wall:.1f} s  facets/iface "
          f"~{diag['n_facets_per_interface']}  n_samples {diag['n_samples']}",
          flush=True)
    return diag, arrs


# ========================================================================
# analysis helpers
# ========================================================================
def leading_edge_gate(p_surf, spacing, dt, t0, surf_pick):
    """M24 surface gate: smoothed surface-layer leading edge vs the Surface pick,
    constant offset removed, in FRAME bins."""
    range_bin = C * dt / 2.0
    w = max(1, int(round(spacing / range_bin)))
    sm = uniform_filter1d(p_surf, w, axis=1, mode="nearest")
    le, has = rocb._leading_edge(sm)
    surf_bin = (surf_pick - t0) / dt
    both = has & np.isfinite(surf_bin)
    resid = le[both] - surf_bin[both]
    off = float(np.median(resid))
    d = np.abs(resid - off)
    return {"median_bins": float(np.median(d)), "p90_bins": float(np.percentile(d, 90)),
            "max_bins": float(d.max()), "offset_bins": off,
            "n_traces": int(both.sum())}


def surface_peak_twtt(power, twtt, t_guess, dt, win_us=0.8):
    """Per-trace surface-peak twtt within +-win_us of a guess."""
    n = len(twtt)
    out = np.full(power.shape[0], np.nan)
    for t in range(power.shape[0]):
        if not np.isfinite(t_guess[t]):
            continue
        a = int(np.clip((t_guess[t] - win_us * 1e-6 - twtt[0]) / dt, 0, n - 2))
        b = int(np.clip((t_guess[t] + win_us * 1e-6 - twtt[0]) / dt, a + 1, n))
        out[t] = twtt[a + int(np.argmax(power[t, a:b]))]
    return out


def depth_power_profile(power, twtt, t_surf, dt, smooth_m=3.0):
    """(twtt_below_surface_us, dB rel surface peak): boxcar-smoothed nadir power
    vs twtt below the surface peak."""
    bin_m = C * dt / 2.0
    w = max(int(round(smooth_m / bin_m)) | 1, 3)
    ps = np.convolve(power, np.ones(w) / w, mode="same")
    i0 = int(np.clip(np.searchsorted(twtt, t_surf) - int(0.3e-6 / dt), 0, len(twtt) - 2))
    i1 = int(np.clip(np.searchsorted(twtt, t_surf + 1.0e-6), i0 + 1, len(twtt)))
    pk = ps[i0:i1].max()
    rel_us = (twtt - t_surf) * 1e6
    db = 10.0 * np.log10(np.maximum(ps / max(pk, 1e-300), 1e-17))
    return rel_us, db


# ========================================================================
# level specification
# ========================================================================
def parse_level(spec):
    s = spec.strip().lower()
    if s == "real":
        return {"spec": "real", "kind": "real", "value": None}
    for suf in ("agl", "msl"):
        if s.endswith(suf):
            return {"spec": s, "kind": suf, "value": float(s[:-len(suf)])}
    raise ValueError(f"bad level {spec!r}: expected 'real', '<N>agl', '<N>msl'")


def platform_z(level, real_elev, surf_elev, trace_spacing_m):
    """Per-trace platform ellipsoidal height for a level."""
    if level["kind"] == "real":
        return real_elev.copy()
    if level["kind"] == "msl":
        return np.full_like(real_elev, level["value"])
    w = max(1, int(round(SMOOTH_M / max(trace_spacing_m, 1.0))))
    smooth = uniform_filter1d(surf_elev, w, mode="nearest")
    return smooth + level["value"]


# ========================================================================
# main runner
# ========================================================================
def run(season, frame_id, levels=DEFAULT_LEVELS, n_traces=100, along_m=10000.0,
        ct_cap=6000.0, out_root=None, min_spacing=None, force=False,
        make_report=True, firn_n=None, ct_firn=CT_FIRN, surf_rough=False):
    lvls = [parse_level(s) for s in levels.split(",") if s.strip()]
    out = Path(out_root or (OUT_DEFAULT / f"altitude_{frame_id}"))
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"
    params = mcords_params(season, frame_id)
    f0 = params["waveform"]["center_frequency_Hz"]
    lam = C / f0
    window, win_note = map_window(
        params["waveform"]["pulse_compression_freq_window"])
    if win_note:
        print(f"WARNING: {win_note}", flush=True)

    frame = load_frame(season, frame_id)
    bot_full = load_bottom_pick(frame)
    fsub, sinfo = sub_frame(frame, along_m)
    a, b = sinfo["slice"]
    bot_sub = bot_full[a:b]
    tw = frame.twtt.values
    dt = float((tw[-1] - tw[0]) / (len(tw) - 1))
    t0f = float(tw[0])
    oversample, f_alias = pick_oversample(
        dt, f0, params["waveform"]["bandwidth_Hz"])
    print(f"alias-free grid: dt_frame {dt*1e9:.4f} ns / k={oversample} -> "
          f"dt_sim {dt/oversample*1e9:.4f} ns, alias {f_alias/1e6:.2f} MHz "
          f"(> B/2 = {params['waveform']['bandwidth_Hz']/2e6:.2f} MHz)",
          flush=True)

    # Base scene at the cross-track cap (single fetch); levels crop + reheight.
    base, aux = base_scene(fsub, n_traces, ct_cap)
    idx = aux["idx"]
    n = len(idx)
    real_elev = np.asarray(fsub.Elevation.values[idx], np.float64)
    surf_pick = np.asarray(fsub.Surface.values[idx], np.float64)
    bot_pick = bot_sub[idx]
    # Ice-surface ellipsoidal elevation and thickness from the frame picks.
    surf_elev = real_elev - surf_pick * C / 2.0
    ok_s = np.isfinite(surf_elev)
    surf_elev = np.where(ok_s, surf_elev, np.nanmedian(surf_elev[ok_s]))
    thick = (bot_pick - surf_pick) * C / (2.0 * np.sqrt(EPS_ICE))  # ice, m
    thick_fill = np.where(np.isfinite(thick), thick, np.nanmax(thick))
    trace_spacing = float(np.median(np.abs(np.diff(sinfo["s_rel_m"][idx]))))
    s_sim = sinfo["s_rel_m"][idx] / 1e3
    j0 = int(np.argmin(np.abs(sinfo["s_rel_m"][idx])))  # representative trace

    # --- optional firn stack (region-appropriate core, effective contrasts) --
    fctx = None
    if firn_n:
        lat_mean = float(np.nanmean(_lonlat(fsub)[0]))
        core, region, core_label, proxy_note = firn_core_for(lat_mean)
        depths = core.equal_depths(firn_n)
        eps_eff, r_eff = core.effective_contrast_eps(depths, lam)
        fctx = {"core": core, "region": region, "label": core_label,
                "proxy_note": proxy_note, "depths": depths, "eps": eps_eff,
                "r": r_eff, "eps_mean": float(core.eps.mean()),
                "file": FIRN_CORES[region][0]}
        print(f"firn: N={firn_n} effective-contrast layers from {core_label} "
              f"({fctx['file']}), 1-{core.zmax:.1f} m, strip +-{ct_firn:.0f} m",
              flush=True)
        if proxy_note:
            print(f"WARNING: {proxy_note}", flush=True)

    results = {}          # spec -> dict(diag, arrs, rc_frame, level info)
    for level in lvls:
        pz = platform_z(level, real_elev, surf_elev, trace_spacing)
        agl = pz - surf_elev
        if float(np.nanmin(agl)) <= CLEAR_MARGIN_M:
            raise SystemExit(
                f"level {level['spec']}: platform does not clear the surface "
                f"(min AGL {np.nanmin(agl):.0f} m <= {CLEAR_MARGIN_M:.0f} m "
                f"margin). Heights are WGS84-ellipsoidal; pick a higher level.")
        r_min = float(np.nanmin(agl))                 # nadir air range
        spacing = facet_spacing(lam, r_min, float(np.nanmedian(thick_fill)))
        if min_spacing:
            spacing = max(spacing, min_spacing)
        surf_tw = 2.0 * agl / C
        bed_tw = surf_tw + 2.0 * thick_fill * np.sqrt(EPS_ICE) / C
        rc_sim, rc_frame, b0 = radar_grid(params, surf_tw, bed_tw, dt, t0f,
                                          oversample, window)
        # Cross-track to cover surface returns landing in the window, capped.
        c_hi = C * (rc_frame.t0 + (rc_frame.n_samples - 1) * dt) / 2.0
        ct_needed = float(np.sqrt(max(c_hi ** 2 - r_min ** 2, 0.0)))
        ct = min(ct_needed, ct_cap)
        cap_bound = ct_needed > ct_cap
        scene = crop_scene(base, ct, pz, f"{base.name}_{level['spec']}")
        rid = f"level_{level['spec']}" + ("_srough" if surf_rough else "")
        meta = {"season": season, "frame_id": frame_id, "spec": level["spec"],
                "n_traces": n, "along_m": along_m, "spacing_m": round(spacing, 4),
                "ct_m": round(ct, 1), "dt_sim_ns": round(rc_sim.dt * 1e9, 5),
                "t0_us": round(rc_sim.t0 * 1e6, 5),
                "n_samples_sim": rc_sim.n_samples}
        if window != "hann":  # keyed only when it deviates from the historical
            meta["window"] = window     # default, so pre-existing caches with
        if surf_rough:                  # hann stay valid
            meta["surf_rough"] = [SURF_ROUGH_SIGMA_M, SURF_ROUGH_CL_M]
        diag, arrs = run_level(rid, scene, bed_cfg(rc_sim, spacing, surf_rough),
                               meta, runs_dir, oversample, force)
        firn_res = None
        if fctx is not None:
            sp_f = firn_facet_spacing(lam, r_min, fctx["core"])
            if min_spacing:
                sp_f = max(sp_f, min_spacing)
            chunks = firn_strip_scenes(base, ct_firn, pz)
            media_f, ifaces_f = firn.firn_stack(fctx["depths"], fctx["eps"],
                                                ATT_DB_PER_KM)
            fcfg = SimConfig(mode="coherent", split_sides=False, radar=rc_sim,
                             facets=FacetConfig(spacing=sp_f), media=media_f,
                             interfaces=ifaces_f)
            fmeta = {"season": season, "frame_id": frame_id,
                     "spec": level["spec"], "kind": f"firn{firn_n}_h1eff",
                     "core": fctx["file"], "method": FIRN_EFF_METHOD,
                     "n_traces": n, "along_m": along_m,
                     "spacing_m": round(sp_f, 4), "ct_firn_m": ct_firn,
                     "dt_sim_ns": round(rc_sim.dt * 1e9, 5),
                     "t0_us": round(rc_sim.t0 * 1e6, 5),
                     "n_samples_sim": rc_sim.n_samples,
                     "att_db_per_km": ATT_DB_PER_KM,
                     "depths_hash": round(float(fctx["depths"].sum()), 4),
                     "eps_sum": round(float(np.sum(fctx["eps"])), 6)}
            # firn rid deliberately excludes _srough: the firn strip run does
            # not depend on the wide run's surface roughness (its own surface
            # layer is excluded from the sum), so the cache is shared.
            fdiag, farrs = run_firn_level(
                f"level_{level['spec']}_firn{firn_n}", chunks, fcfg, fmeta,
                runs_dir, oversample, force)
            firn_res = {"diag": fdiag, "arrs": farrs, "spacing": sp_f}
        lpa_err = rocb._lpa_nadir_error(spacing, r_min, 2 * np.pi / lam,
                                        fresnel_normal(1.0, EPS_ICE))
        pl_res_m = C / (2.0 * params["waveform"]["bandwidth_Hz"])   # range res
        h_med = float(np.median(agl))
        pl_foot = 2.0 * np.sqrt(2.0 * h_med * pl_res_m)             # diameter
        results[level["spec"]] = {
            "level": level, "diag": diag, "arrs": arrs, "rc_frame": rc_frame,
            "pz": pz, "agl": agl, "spacing": spacing, "ct": ct,
            "cap_bound": cap_bound, "lpa_err": lpa_err, "r_min": r_min,
            "h_med": h_med, "pl_foot": pl_foot, "pl_res_m": pl_res_m,
            "firn": firn_res}

    # ---- analysis: real-level surface gate + r^-2 surface scaling ----
    real_spec = next((L["spec"] for L in lvls if L["kind"] == "real"), None)
    gate = None
    if real_spec is not None:
        r = results[real_spec]
        p_surf = np.abs(r["arrs"]["field"][..., 0]) ** 2
        gate = leading_edge_gate(p_surf, r["spacing"], r["rc_frame"].dt,
                                 r["rc_frame"].t0, surf_pick)

    for spec, r in results.items():
        f = r["arrs"]["field"]
        r["surf_peak"] = float(np.median(np.max(np.abs(f[..., 0]) ** 2, axis=1)))
        r["bed_peak"] = float(np.median(np.max(np.abs(f[..., 1]) ** 2, axis=1)))
        r["surf_db"] = 10.0 * np.log10(max(r["surf_peak"], 1e-300))
        r["bed_db"] = 10.0 * np.log10(max(r["bed_peak"], 1e-300))
    order = list(results)
    scaling = []
    for i in range(1, len(order)):
        a0, b1 = results[order[0]], results[order[i]]
        meas_db = b1["surf_db"] - a0["surf_db"]
        exp_db = -20.0 * np.log10(b1["h_med"] / a0["h_med"])  # r^-2 in power
        scaling.append({"pair": f"{order[i]}/{order[0]}",
                        "measured_db": round(meas_db, 2),
                        "expected_r2_db": round(exp_db, 2),
                        "deviation_db": round(meas_db - exp_db, 2)})

    firn_doc = None
    if fctx is not None:
        firn_doc = _firn_analysis(fctx, results, frame, fsub, sinfo, j0)

    metrics = _metrics(gate, scaling, results, params, win_note)
    config = _config(season, frame_id, levels, n_traces, along_m, ct_cap,
                     results, params, sinfo, trace_spacing, oversample, window)
    notes = _notes(season, frame_id, results, params, along_m, n, gate,
                   scaling, oversample, window, win_note)
    if firn_doc is not None:
        metrics["firn_comparison"] = _firn_metric(firn_doc, firn_n, fctx,
                                                  list(results))
        config["firn"] = {
            "n_layers": firn_n, "core": fctx["label"],
            "core_file": f"tests/fixtures/firn/{fctx['file']}",
            "region": fctx["region"], "proxy_note": fctx["proxy_note"],
            "method": FIRN_EFF_METHOD, "ct_firn_m": ct_firn,
            "depth_range_m": [1.0, round(fctx["core"].zmax, 2)],
            "att_db_per_km": ATT_DB_PER_KM,
            "eps_mean_depth_conversion": round(fctx["eps_mean"], 4),
            "per_level": firn_doc["per_level"]}
        config["firn_stack"] = {
            "depth_m": [round(float(x), 3) for x in fctx["depths"]],
            "eps_r": [round(float(x), 6) for x in fctx["eps"]],
            "segment_abs_r_db": [round(float(20 * np.log10(max(x, 1e-30))), 3)
                                 for x in fctx["r"]]}
        notes += (
            f" FIRN (--firn {firn_n}): effective-contrast layers from "
            f"{fctx['label']} ({FIRN_EFF_METHOD}: each interface carries the "
            f"segment-aggregate transfer-matrix |r| of the raw core density "
            f"profile; point-sampled eps deprecated), conformal surface "
            f"offsets 1-{fctx['core'].zmax:.1f} m, firn attenuation "
            f"{ATT_DB_PER_KM:.0f} dB/km one-way, narrow +-{ct_firn:.0f} m "
            f"strip field-summed with the surface+bed run (firn run's own "
            f"surface layer excluded; approximate gamma-scaled seam recorded)."
            + (f" CAVEAT: {fctx['proxy_note']}." if fctx["proxy_note"] else "")
            + (f" N={firn_n} resolves band-integrated power but leaves a "
               f"'picket-fence' profile shape (b26 ladder: r 0.907 at N=10 "
               f"vs 0.963 plateau at N=20)." if firn_n <= 10 else ""))
    if surf_rough:
        config["surface_roughness"] = {
            "sigma_m": SURF_ROUGH_SIGMA_M, "corr_length_m": SURF_ROUGH_CL_M,
            "source": "C&S 2020 Fig. 11 mcords inversion at its 0 m clamp "
                      "(tests/fixtures/firn/fig11*.csv): REPRESENTATIVE "
                      "cm-scale snow-surface roughness the 32 m DEM lacks, "
                      "not a site measurement",
            "applies_to": "wide-run surface interface only (bed smooth; firn "
                          "strip unchanged -- its surface layer is excluded "
                          "from the field sum)"}
        notes += (f" SURFACE ROUGHNESS ON (--surf-rough): sigma "
                  f"{SURF_ROUGH_SIGMA_M * 100:.1f} cm, l "
                  f"{SURF_ROUGH_CL_M:.1f} m (C&S 2020 Fig. 11 mcords 0 m "
                  f"clamp, representative not site-measured) on the surface "
                  f"interface via the Gerekos rough-facet response; "
                  f"iteration-validated to improve the measured match on "
                  f"both study frames.")
    doc = {"case": f"altitude_{frame_id}", "group": "xOPR clutter",
           "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "metrics": metrics, "notes": notes}
    (out / "metrics.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out / "run_config.json").write_text(json.dumps(config, indent=1) + "\n")

    figs = _figures(out, frame, fsub, sinfo, idx, s_sim, j0, results, order,
                    surf_pick, bot_pick, dt, t0f, firn_n=firn_n)
    if firn_doc is not None:
        figs.append(_firn_figure(out, results, order, fctx, firn_doc))
    if make_report:
        _report(out, config, metrics, notes, figs, results, order, params,
                scaling)
    # mirror for tools/make_report.py
    ver = VER_ROOT / f"altitude_{frame_id}"
    ver.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / "metrics.json", ver / "metrics.json")
    for fpath in figs:
        shutil.copy2(fpath, ver / fpath.name)
    print(f"altitude_{frame_id}: levels {order} | "
          + (f"real surf gate {gate['median_bins']:.2f} bins | " if gate else "")
          + " | ".join(f"{s['pair']} {s['measured_db']:+.1f} vs r2 "
                       f"{s['expected_r2_db']:+.1f} dB" for s in scaling),
          flush=True)
    return metrics, out


# ========================================================================
# metrics / config / notes
# ========================================================================
def _metrics(gate, scaling, results, params, win_note=None):
    rec = "recorded only"
    m = {}
    if win_note:
        m["window_approximation"] = {
            "value": 1.0, "threshold": None, "op": "record", "pass": True,
            "product_window":
                params["waveform"]["pulse_compression_freq_window"],
            "note": win_note + ". Proper tukey support in soundersim.waveform "
            "is a separate future work item. " + rec}
    if gate is not None:
        m["real_surface_alignment"] = {
            "value": gate["median_bins"], "threshold": GATE_BINS, "op": "<=",
            "pass": bool(gate["median_bins"] <= GATE_BINS),
            "p90_bins": gate["p90_bins"], "max_bins": gate["max_bins"],
            "offset_bins": gate["offset_bins"],
            "note": "median |smoothed coherent surface-layer leading edge - "
            "frame Surface pick| in FRAME bins, constant offset removed (M24). "
            "Real-nav level only; the synthetic altitudes have no measured "
            "reference to align to."}
    worst = max((abs(s["deviation_db"]) for s in scaling), default=0.0)
    m["surface_r2_scaling"] = {
        "value": worst, "threshold": None, "op": "record", "pass": True,
        "pairs": scaling,
        "note": "max |measured surface-peak power ratio - r^-2 expectation| in "
        "dB across level pairs (r = median AGL). A free physics check: the "
        "coherent quasi-specular surface return should scale ~(r2/r1)^-2 in "
        "power. Recorded, not gated for the synthetic levels. " + rec}
    for spec, r in results.items():
        m[f"lpa_nadir_error_{spec}"] = {
            "value": r["lpa_err"], "threshold": None, "op": "record",
            "pass": True, "facet_size_m": round(r["spacing"], 3),
            "r_min_m": round(r["r_min"], 1),
            "note": f"envelope-normalized single-facet nadir LPA error at the "
            f"{spec} level (worst case; off-nadir facets sinc-suppressed). "
            + rec}
    return m


def _config(season, frame_id, levels, n_traces, along_m, ct_cap, results,
            params, sinfo, trace_spacing, oversample, window):
    wf = params["waveform"]
    tbl = []
    for spec, r in results.items():
        agl = r["agl"]
        tbl.append({
            "level": spec,
            "agl_range_m": [round(float(np.nanmin(agl)), 0),
                            round(float(np.nanmax(agl)), 0)],
            "agl_median_m": round(r["h_med"], 0),
            "facet_spacing_m": round(r["spacing"], 3),
            "n_facets_per_interface": r["diag"]["n_facets_per_interface"],
            "n_samples": r["diag"]["n_samples"],
            "wall_s": r["diag"]["wall_s"],
            "lpa_nadir_error": round(r["lpa_err"], 4),
            "surface_peak_db": round(r["surf_db"], 2),
            "bed_peak_db": round(r["bed_db"], 2),
            "pulse_limited_footprint_m": round(r["pl_foot"], 1),
            "cross_track_m": round(r["ct"], 0),
            "cross_track_cap_bound": bool(r["cap_bound"]),
            "dropped_power_fraction_bed":
                round(r["diag"]["dropped_power_fraction"][1], 5)})
    return {
        "season": season, "frame_id": frame_id, "levels": levels,
        "n_traces": n_traces, "along_m": along_m, "ct_cap_m": ct_cap,
        "track_len_m": round(sinfo["track_len_m"], 0),
        "trace_spacing_m": round(trace_spacing, 1),
        "chirp": {"f0_hz": wf["center_frequency_Hz"],
                  "bandwidth_hz": wf["bandwidth_Hz"],
                  "pulse_length_s": wf["bed_waveform_pulse_length_s"],
                  "window_modeled": window,
                  "window_product": wf["pulse_compression_freq_window"]},
        "range_resolution_m": round(C / (2.0 * wf["bandwidth_Hz"]), 3),
        "antenna": f"{N_ELEMENTS}-element {SPACING_LAM}-lambda array, roll=nav",
        "media": f"air / ice(eps {EPS_ICE}, {ATT_DB_PER_KM} dB/km one-way) / "
                 f"bed(eps {EPS_BED})",
        "dt_sim_ns": round(1e9 * (results[next(iter(results))]["rc_frame"].dt
                                  / oversample), 5),
        "oversample": oversample,
        "level_table": tbl}


def _notes(season, frame_id, results, params, along_m, n, gate, scaling,
           oversample, window, win_note):
    wf = params["waveform"]
    specs = list(results)
    gate_txt = (f"Real-nav surface gate: median {gate['median_bins']:.2f} frame "
                f"bins (offset {gate['offset_bins']:+.1f} removed). "
                if gate else "No real-nav level requested. ")
    win_txt = (f" {win_note}." if win_note else "")
    return (
        f"{season} {frame_id}: coherent surface+bed cluttergram (NO firn / "
        f"internal layers) at platform altitudes {specs} on a "
        f"{along_m/1e3:.0f} km sub-segment ({n} traces), holding the frame's "
        f"real MCoRDS params fixed so ONLY the geometry changes. INSTRUMENT: "
        f"chirp {wf['center_frequency_Hz']/1e6:.1f} MHz / "
        f"{wf['bandwidth_Hz']/1e6:.1f} MHz, pulse "
        f"{wf['bed_waveform_pulse_length_s']*1e6:.0f} us, compression window "
        f"{wf['pulse_compression_freq_window']} modeled as '{window}',{win_txt}"
        f" {N_ELEMENTS}-element {SPACING_LAM}-lambda array (roll=nav); "
        f"alias-free dt_frame/{oversample} grid (first k of {OVERSAMPLE_CANDS} "
        f"with the envelope-quantization alias out of band) decimated "
        f"[::{oversample}]. Surface PGC 32 m mosaic, bed BedMachine (bilinear "
        f"to 32 m); media air / ice(3.17, "
        f"{ATT_DB_PER_KM:.0f} dB/km one-way) / bed(eps 8). ALTITUDE SEMANTICS: "
        f"real = recorded nav; <N>agl = ~1 km along-track-smoothed ice-surface "
        f"elevation + N (terrain-following); <N>msl = constant ellipsoidal "
        f"height N. NB: all heights are WGS84-ELLIPSOIDAL -- 'MSL' is "
        f"approximated by ellipsoid height (geoid offset ~tens of m "
        f"neglected; fine for altitude trades, not a true orthometric "
        f"datum). Per level the twtt window (surface ~2h/c through bed + "
        f"margin), facet spacing (beta {BETA} Fresnel at the level's nadir "
        f"r_min -- higher altitude -> coarser facets -> cheaper), and "
        f"cross-track (sized to cover in-window returns, capped) are all "
        f"re-derived; see the level table. {gate_txt}Surface r^-2 scaling "
        f"(power vs (r2/r1)^-2): "
        + "; ".join(f"{s['pair']} {s['measured_db']:+.1f} dB "
                    f"(expect {s['expected_r2_db']:+.1f}, dev "
                    f"{s['deviation_db']:+.1f})" for s in scaling)
        + ". Panels share a 'twtt relative to each panel's median surface "
        "return' axis so structure is comparable despite the raw-twtt offset "
        "between altitudes. HONESTY: (1) 32 m DEM posting -> the coherent "
        "product is statistical (speckle/envelope), not phase-deterministic; "
        "(2) the coherent LPA is specular-dominated at these facets, per-level "
        "nadir error recorded; (3) the measured frame is f-k SAR + multilooked "
        "while the sims are unfocused per-trace raw -- compare structure and "
        "relative levels, not resolution; (4) the sims carry no volume "
        "scatter, internal layers, or receiver noise floor"
        + ("; (5) MODELED-WINDOW APPROXIMATION in force -- see the "
           "window_approximation metric." if win_note else "."))


# ========================================================================
# figures
# ========================================================================
def _figures(out, frame, fsub, sinfo, idx, s_sim, j0, results, order,
             surf_pick, bot_pick, dt, t0f, firn_n=None):
    """Radargram panels + nadir overlay. When firn runs exist the sim panels
    show the FIRN-ENABLED composite (E_comb: surface+bed field + firn internal
    layers, no surface double count) and are labeled accordingly."""
    tw_full = frame.twtt.values
    meas_full = _db(np.asarray(fsub.Data.values, np.float64))
    bot_native = load_bottom_pick(fsub)
    surf_native = np.asarray(fsub.Surface.values, np.float64)
    s_km = sinfo["s_rel_m"] / 1e3

    # Shared relative-twtt axis: below-surface span from the sims + measured.
    def _med(v):
        return float(np.nanmedian(v[np.isfinite(v)]))

    surf_med_meas = _med(surf_pick)
    below_max = float(np.nanmax(bot_pick - surf_pick)) if np.isfinite(
        bot_pick).any() else 30e-6
    y_hi = (below_max + POST_BED_US * 1e-6) * 1e6      # us below surface
    y_lo = -0.6                                        # us above surface

    panels = [("measured", None)] + [(spec, results[spec]) for spec in order]
    ncols = min(3, len(panels))
    nrows = -(-len(panels) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 5.5 * nrows),
                            sharex=False, sharey=True, squeeze=False)
    axs = axs.ravel()
    for ax in axs[len(panels):]:
        ax.set_visible(False)

    for k, (name, r) in enumerate(panels):
        ax = axs[k]
        if name == "measured":
            rel_us = (tw_full - surf_med_meas) * 1e6
            m = (rel_us >= y_lo) & (rel_us <= y_hi)
            img = meas_full[:, m]
            ext = [s_km[0], s_km[-1], rel_us[m][-1], rel_us[m][0]]
            fin = img[np.isfinite(img)]
            vmax = np.percentile(fin, 99.5)
            im = ax.imshow(img.T, aspect="auto", extent=ext, cmap="gray",
                           vmin=vmax - DYN_DB, vmax=vmax)
            ax.plot(s_km, (surf_native - surf_med_meas) * 1e6, "c", lw=0.7,
                    label="Surface pick")
            ax.plot(s_km, (bot_native - surf_med_meas) * 1e6, "r", lw=0.7,
                    label="Bottom pick")
            ax.set_title("measured (CSARP_standard, dB)")
        else:
            tw = r["arrs"]["twtt"]
            nadir = r["arrs"]["nadir_twtt"]
            surf_med = _med(nadir[:, 0])
            E = r.get("E_comb", r["arrs"]["field"].sum(-1))  # +firn if run
            comb = _db(np.abs(E) ** 2)
            rel_us = (tw - surf_med) * 1e6
            ext = [s_sim[0], s_sim[-1], rel_us[-1], rel_us[0]]
            fin = comb[np.isfinite(comb) & (comb > -290)]
            vmax = np.percentile(fin, 99.5)
            im = ax.imshow(comb.T, aspect="auto", extent=ext, cmap="gray",
                           vmin=vmax - DYN_DB, vmax=vmax)
            ax.plot(s_sim, (nadir[:, 0] - surf_med) * 1e6, "c", lw=0.7,
                    label="sim surface nadir")
            ax.plot(s_sim, (nadir[:, 1] - surf_med) * 1e6, "r", lw=0.7,
                    label="sim bed nadir")
            firn_tag = f" +firn N={firn_n}" if "E_comb" in r else ""
            ax.set_title(f"{name}{firn_tag}  (median AGL {r['h_med']:.0f} m, "
                         f"spacing {r['spacing']:.1f} m)")
        fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01, label="dB")
        ax.set_ylim(y_hi, y_lo)
        ax.legend(loc="lower right", fontsize=7)
        ax.set_xlabel("along-track (km)")
    for c in range(0, len(panels), ncols):
        axs[c].set_ylabel("twtt below median surface return (us)")
    sim_desc = (f"surface+firn(N={firn_n})+bed" if firn_n
                else "surface+bed")
    fig.suptitle(f"{fsub.attrs.get('frame_id', '')}: coherent {sim_desc} vs "
                 f"platform altitude (shared surface-referenced twtt)")
    fig.tight_layout()
    f1 = out / "radargrams.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    # Nadir depth-power overlay at the representative trace.
    f2 = out / "nadir_profiles.png"
    fig, ax = plt.subplots(figsize=(8.5, 6))
    i0 = sinfo["i0_local"]
    dt_full = float(tw_full[1] - tw_full[0])
    meas_lin = np.asarray(fsub.Data.values[i0], np.float64)
    t_s = surface_peak_twtt(meas_lin[None], tw_full,
                            np.array([surf_native[i0]]), dt_full)[0]
    rel, db = depth_power_profile(meas_lin, tw_full, t_s, dt_full)
    ax.plot(rel, db, "k", lw=1.8, label="measured")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(order)))
    for c, spec in zip(colors, order):
        r = results[spec]
        tw = r["arrs"]["twtt"]
        E = r.get("E_comb", r["arrs"]["field"].sum(-1))
        p = np.abs(E[j0]) ** 2
        t_s = surface_peak_twtt(p[None], tw,
                                np.array([r["arrs"]["nadir_twtt"][j0, 0]]),
                                r["rc_frame"].dt)[0]
        rel, db = depth_power_profile(p, tw, t_s, r["rc_frame"].dt)
        ax.plot(rel, db, color=c, lw=1.2, label=f"{spec} (AGL {r['h_med']:.0f} m)")
    ax.set_xlim(-0.4, min(y_hi, 40))
    ax.set_ylim(PROF_FLOOR_DB, 3)
    ax.set_xlabel("twtt below surface peak (us)")
    ax.set_ylabel("power (dB rel own surface peak, 3 m smoothed)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"{fsub.attrs.get('frame_id', '')}: nadir depth-power "
                 f"(representative trace"
                 + (f", {sim_desc} sims" if firn_n else "")
                 + ") vs altitude")
    fig.tight_layout()
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    return [f1, f2]


# ========================================================================
# report
# ========================================================================
def _report(out, config, metrics, notes, figs, results, order, params, scaling):
    def b64(p):
        return base64.b64encode(Path(p).read_bytes()).decode()

    css = ("body{font-family:system-ui,Arial,sans-serif;margin:2rem;"
           "max-width:1250px;color:#1a1a1a}h1{margin-bottom:.2rem}"
           "table{border-collapse:collapse;margin:1rem 0;font-size:.82rem}"
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
    # level table
    tbl = config["level_table"]
    cols = ["level", "agl_range_m", "agl_median_m", "facet_spacing_m",
            "n_facets_per_interface", "n_samples", "wall_s", "lpa_nadir_error",
            "surface_peak_db", "bed_peak_db", "pulse_limited_footprint_m",
            "cross_track_m", "cross_track_cap_bound", "dropped_power_fraction_bed"]
    thead = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    trows = "".join("<tr>" + "".join(
        f"<td>{html.escape(json.dumps(row[c]) if isinstance(row[c], list) else str(row[c]))}</td>"
        for c in cols) + "</tr>" for row in tbl)
    srows = "".join(
        f"<tr><th>{html.escape(s['pair'])}</th>"
        f"<td>{s['measured_db']:+.2f}</td><td>{s['expected_r2_db']:+.2f}</td>"
        f"<td>{s['deviation_db']:+.2f}</td></tr>" for s in scaling)
    crows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(json.dumps(v) if isinstance(v, (dict, list)) else str(v))}</td></tr>"
        for k, v in config.items() if k not in ("level_table", "firn_stack"))
    figs_html = "".join(
        f"<h3>{html.escape(Path(f).stem)}</h3>"
        f"<img src='data:image/png;base64,{b64(f)}' alt='{Path(f).name}'>"
        for f in figs)
    body = f"""
<h1>Platform-altitude comparison: {html.escape(config['season'])} {html.escape(config['frame_id'])}</h1>
<p class="note">{html.escape(notes)}</p>
<h2>Radargram panels + nadir depth-power</h2>
{figs_html}
<h2>Level table</h2>
<table><tr>{thead}</tr>{trows}</table>
<h2>Surface r^-2 scaling (power ratio vs (r2/r1)^-2)</h2>
<table><tr><th>pair</th><th>measured (dB)</th><th>expected (dB)</th>
<th>deviation (dB)</th></tr>{srows}</table>
<h2>Metrics</h2>
<table><tr><th>metric</th><th>value</th><th>criterion</th><th>note</th></tr>
{''.join(mrows)}</table>
<h2>Run configuration</h2>
<table>{crows}</table>
<h2>Instrument parameters (from the frame's own product file)</h2>
<pre>{html.escape(json.dumps(params, indent=1))}</pre>
"""
    (out / "report.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>altitude comparison {html.escape(config['frame_id'])}</title>"
        f"<style>{css}</style></head><body>{body}</body></html>")
    print(f"wrote {out / 'report.html'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2019_Greenland_P3")
    ap.add_argument("--frame", default="20190418_01_009")
    ap.add_argument("--levels", default=DEFAULT_LEVELS,
                    help="comma list of real / <N>agl / <N>msl")
    ap.add_argument("--n-traces", type=int, default=100)
    ap.add_argument("--along-m", type=float, default=10000.0)
    ap.add_argument("--ct-cap", type=float, default=6000.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-spacing", type=float, default=None,
                    help="clamp per-level facet spacing up (test/speed knob)")
    ap.add_argument("--firn", type=int, default=None, metavar="N",
                    help="add N effective-contrast firn layers from the "
                    "region-appropriate core (B26 Greenland / B25 Antarctic "
                    "proxy); default off")
    ap.add_argument("--ct-firn", type=float, default=CT_FIRN,
                    help="firn-strip cross-track reach (m)")
    ap.add_argument("--surf-rough", action="store_true",
                    help="representative C&S Fig. 11 shallow-clamp sub-facet "
                    "roughness on the surface interface (default smooth)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(args.season, args.frame, levels=args.levels, n_traces=args.n_traces,
        along_m=args.along_m, ct_cap=args.ct_cap, out_root=args.out,
        min_spacing=args.min_spacing, force=args.force, firn_n=args.firn,
        ct_firn=args.ct_firn, surf_rough=args.surf_rough)


if __name__ == "__main__":
    main()
