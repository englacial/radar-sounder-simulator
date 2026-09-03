"""Basal-clutter altitude triplet on the 2016 DC-8 anchor line.

Three real flights of the SAME 148.5 km grounded-ice line (claude_notes/
basal_clutter_scout.md) at 442 / 9150 / 10684 m AGL with IDENTICAL systems
(190 MHz / 50 MHz / hann / 20.202 ns): the measured radargrams show ~20 dB
more mid-column ("basal") clutter power at altitude. Each pass's common
segment is simulated COHERENT SURFACE+BED ONLY (REMA 32 m + BedMachine
500 m; NO firn, NO internal layers) at its real altitude/nav/params, and the
simulated clutter is DECOMPOSED per interface (the kernel returns per-layer
fields) into SURFACE returns vs BED returns energy -- the discriminator for what
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

STUDY LINES (--line, see LINES / activate_line): the module defaults ARE
the Antarctic line above, so `--line antarctic_2016` is a no-op and that
path (and its caches) is untouched. `--line greenland_2014_2017` swaps in
the Greenland PAIR from claude_notes/greenland_altitude_scout.md --
20140421_01_069 (465 m AGL, 2014_Greenland_P3) and 20170424_01_067
(2483 m AGL, 2017_Greenland_P3) over the same central-west interior line,
a 5.3x altitude ratio with an IDENTICAL radar configuration -- plus the
synthetic 14 km pass as the third altitude. That line is EPSG:3413,
ArcticDEM 32 m + BedMachine Greenland v5 150 m (opr.py selects both from
the hemisphere), has no grounding line, and mixes two seasons (one per
pass, hence the per-pass "season" key).

Run:  uv run python tools/run_basal_clutter.py \
          --config config/experiments/pilot.yaml --line <line>   # PREFERRED: a named,
      # committed experiment spec (config/README.md). The flag front door
      # below is unchanged and builds the same RunSpec, so both produce
      # identical chunk cache keys.

      uv run python tools/run_basal_clutter.py                # 10 km pilot
      uv run python tools/run_basal_clutter.py --segment full # 50 km (STOP:
      report pilot timings first; full run only on explicit go-ahead)
      uv run python tools/run_basal_clutter.py --segment full --picked-bed
      uv run python tools/run_basal_clutter.py --segment full --picked-bed \
          --gamma-from-rssnr   # + required-surface-SNR-driven bed gamma
"""

import argparse
import base64
import hashlib
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
import xarray as xr  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy import ndimage, stats  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import clutter_analysis  # noqa: E402  measurement conventions
import clutter_instruments  # noqa: E402  radar system definitions
import clutter_lines  # noqa: E402  study-line definitions (YAML)
import run_altitude_comparison as rac  # noqa: E402  shared machinery
from run_opr_comparison import _db  # noqa: E402

from soundersim.kernels import KERNEL_VERSION
from soundersim.config import (AntennaConfig, DemInterface, FacetConfig,  # noqa: E402
                               GrazingFixConfig, Medium, RadarConfig,
                               RoughnessConfig, SimConfig, WaveformConfig)
from soundersim.opr import load_bottom_pick, load_frame  # noqa: E402
from soundersim.physics import fresnel_normal  # noqa: E402

C = 299792458.0
VER_ROOT = ROOT / "outputs" / "verification"


ANT_DEFAULT = "array"
SPEC_DIFFUSE_NOTE = (
    "Angle-dependent bed reflectivity: the RSSNR-mapped |Gamma_bed|^2(x) is "
    "split into a SPECULAR share f_s, weighted by the facet tilt "
    "G(psi) = exp(-tan^2(psi)/(2 s0^2)) (bright-because-flat: a facet tilted "
    "psi mirrors the nadir-looking radar to 2 psi off-nadir, so it must not "
    "inherit the nadir-calibrated brightness), and a DIFFUSE share 1 - f_s "
    "carried by the kernel's incoherent per-facet channel with a "
    "cos^n(theta_i) law. The split conserves total bed power at nadir over a "
    "flat interface BY CONSTRUCTION (kernels/multilayer.py normalization "
    "derivation; f_s = 1 with s0 = 0 traces the unsplit program "
    "bit-identically). ONE scene-constant f_s is fitted across all three "
    "measured altitudes -- the over-determination test.")


def bed_rough_validity():
    """Gerekos 2023 validity envelope for the ACTIVE line.

    A function, not a constant: the limits are quoted in lambda_ice, which is
    line-specific, and a module-level string would freeze the Antarctic
    190 MHz numbers into every Greenland run's recorded config."""
    return (
        "Gerekos 2023 sub-facet roughness: l <= facet size (10.7/46.0/49.8/"
        f"81.9 m on the Antarctic line) and up to a few lambda_ice "
        f"(lambda_ice = {LAM_ICE_M:.3f} m at {FC_HZ / 1e6:.0f} MHz); accuracy "
        "~0.3 dB below lambda/10, ~1 dB near sigma = lambda/4 = "
        f"{LAM_ICE_M / 4.0:.3f} m (the comfortable ceiling), degrading "
        "beyond ~0.4*lambda")


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


def bed_incidence_deg(h, d, n_ice, dt_extra):
    """Refracted IN-ICE incidence angle at the bed (deg) for a bed return
    arriving ``dt_extra`` (s, scalar or array) past the nadir-bed delay --
    the same Snell sweep as ``bed_reach``, inverted for the angle instead of
    the cross-track distance. This is the angular-backscatter reading of a
    post-bed delay: tail delay -> off-nadir bed incidence angle."""
    theta = np.linspace(0.0, np.deg2rad(89.5), 4000)[1:]
    phi = np.arcsin(np.clip(np.sin(theta) / n_ice, 0.0, 1.0 - 1e-12))
    t_extra = (2.0 * (h / np.cos(theta) + n_ice * d / np.cos(phi)) / C
               - 2.0 * (h + n_ice * d) / C)
    return np.degrees(np.interp(dt_extra, t_extra, phi))


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


PBED_TAG = "_pbed"          # output/cache suffix; BedMachine runs stay cached
PICKED_BED_NOTE = (
    "bed = DEM + resid(s), resid(s) = picked_bed(s) - DEM at "
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
    "runs are directly comparable. The DEM is the line's bed_dem (BedMachine "
    "or a DEMOGORGN realization; on a crosses_gl segment the hybrid grid).")


# ========================================================================
# STUDY-LINE REGISTRY (--line): definitions live in config/lines/*.yaml
# ========================================================================
# A line used to be ~250 lines of dict literal here, with the Antarctic one
# spelled as the module's own defaults and the Greenland one as a sibling
# override -- an asymmetry that made activation irreversible and let shared
# constants (REAL_CHAIN) leak across lines. Both are now validated YAML
# documents (tools/clutter_lines.py); this module holds only the mechanism.
#
# activate_line() still rebinds module globals, so none of the analysis code
# below knows a line is data rather than a literal.
INSTRUMENTS = clutter_instruments.load_all()
LINE_SPECS = clutter_lines.load_all()
clutter_instruments.validate_line_instruments(LINE_SPECS, INSTRUMENTS)
LINES = {name: spec.to_globals(ROOT) for name, spec in LINE_SPECS.items()}
# The names a line definition binds. Exported so the tests assert against ONE
# list; every entry sets all of them, so activation is total and reversible.
LINE_GLOBALS = tuple(sorted(next(iter(LINES.values()))))
# The flag front door needs SOME default; a spec always names its line
# explicitly, so this only affects a bare CLI invocation. Derived from the
# registry rather than hardcoded -- two hardcoded line names were exactly
# the kind of literal this migration removed, and they went stale the moment
# the lines were renamed.
DEFAULT_LINE = sorted(LINES)[0]


# ---- the contract config/analysis.yaml fills ---------------------------
# Study-wide measurement conventions: what the metrics MEAN. Bound at import
# and re-bound by activate_line() when a LINE overrides a subset. Declared
# explicitly for the same reason as the line contract above.
MARGIN_US: float = 0.0                 # clutter coverage margin (reach)
POST_BED_US: float = 0.0
PRE_SURF_US: float = 0.0
SURF_WIN_US: float = 0.0               # surface peak search half-width
MID_LO_US: float = 0.0                 # mid-column: surf+lo -> bed-hi
MID_HI_US: float = 0.0
BED_LO_US: float = 0.0                 # bed window: bed-lo -> bed+hi
BED_HI_US: float = 0.0
SCOUT_LO_US: float = 0.0               # scout contrast metric
SCOUT_HI_US: float = 0.0
SCOUT_PK_US: float = 0.0
FLOOR_TAIL_LO_US: float = 0.0          # measured noise-floor window
FLOOR_TAIL_HI_US: float = 0.0
FLOOR_BED_GUARD_US: float = 0.0
FLOOR_ROLLOFF_US: float = 0.0
FLOOR_MIN_WIDTH_US: float = 0.0
TAIL_PROF_US: tuple = ()               # bed-referenced tail profile extent
TAIL_FIT_US: tuple = ()                # robust slope fit window
TAIL_EXCESS_US: tuple = ()
TAIL_GUARD_DB: float = 0.0             # sim bed vs surface returns guard
TAIL_FLOOR_MARGIN_DB: float = 0.0
CALIBRATION: dict = {}                 # gamma_surface + A (manual or solve)
ATTENUATION_REGRESSION: dict = {}      # A-solver settings (analysis.yaml)
GAMMA_SURFACE_SOLVE: dict = {}         # gamma-solver settings (analysis.yaml)
GRAZING_FIX: dict = {}                 # facet-lattice fix s_eff (analysis.yaml)
CORR_WIN_M: float = 0.0                # bed-brightness smoothing scale
ROUGH_WIN_M: float = 0.0               # bed-roughness detrend window
GL_RAMP_KM: float = 0.0                # hybrid-bed blend ramp past the GL
EPS_SEAWATER: float = 0.0              # floating reflector: ice -> seawater
N_LOOKS_SIM: int = 0                   # incoherent looks in matched proc
RADARGRAM_PCT: tuple = ()              # per_panel robust scaling limits
CHUNK_M: float = 0.0                   # along-track chunk target
FACET_SPACING_SCALE: float = 0.0       # default lattice refinement (<= 1)
CHUNK_M_PROC: float = 0.0              # ... at fine posting
# Per-chunk memory guard. Measured 2026-09-03 at posting_div 8 (~1800
# traces/chunk): pineisland_north 0.56 M facets/interface/chunk ran,
# westcoast 1.35 M was OOM-killed at 110 GB, david 1.63 M at 100 GB -- the
# kernels' footprint grows as traces x facets (~50 B per pair). Chunks are
# split further until traces * facets_per_interface (estimated from the
# chunk_scene crop) fits this budget; 1.1e9 keeps pineisland_north at 3.
CHUNK_TRACE_FACETS: float = 1.1e9

ANALYSIS = clutter_analysis.load_analysis()
ANALYSIS_GLOBALS = tuple(sorted(ANALYSIS.to_globals()))
# Per-line resolved conventions: the study defaults with that line's declared
# overrides merged in, plus what changed (recorded in every run config).
LINE_ANALYSIS = {}
for _n, _s in LINE_SPECS.items():
    LINE_ANALYSIS[_n] = ANALYSIS.merged(_s.analysis)


# ---- the contract a line definition fills ------------------------------
# Every name here is (re)bound by activate_line() from config/lines/*.yaml,
# which runs at import below -- these placeholders are never observable. They
# are declared explicitly so the set is visible in one place to readers,
# editors and static analysis, which cannot see through globals().update().
LINE: str = ""                         # line key
SEASON: str = ""                       # reference pass season
BED_DEM: str = ""                      # grounded-ice DEM: bedmachine|demogorgn
CRS: str = ""                          # anchor along-track / pick-axis CRS
CASE_PREFIX: str = ""                  # output + verification case prefix
OUT_DEFAULT: Path = ROOT               # outputs/<case_prefix>
FC_HZ: float = 0.0                     # carrier of the line's instrument
LAM_ICE_M: float = 0.0                 # in-ice wavelength at FC_HZ
PASSES: dict = {}                      # pass table (geometry + instrument)
ORDER: list = []                       # real passes, altitude order
SEGMENTS: tuple = ()                   # study segments
SEGMENTS_CROSSING_GL: tuple = ()       # segments spanning the grounding line
S0_KM: dict = {}                       # per-segment display origin
DECOMP_S_KM: dict = {}                 # per-segment single-trace location(s)
N_TRACES_BY_SEGMENT: dict = {}         # per-segment simulated trace count
REF_PASS: str = ""                     # pick-axis / RSSNR anchor pass
REF_SEASON: str = ""                   # that pass's season
REF_FRAMES: tuple = ()                 # frames supplying the pick axis
GL_S_KM: float | None = None           # grounding line, None if all grounded
SYNTHETIC_KEYS: tuple = ()             # line-declared synthetic passes
MEASURED_CAVEATS: str = ""             # recorded product caveats
UNSUPPORTED: tuple = ()                # features not wired for this line
REAL_CHAIN: dict = {}                  # the measured product's own chain
RADARGRAM_Y_US: tuple = ()             # radargram framing
RADARGRAM_DB: tuple = ()
RADARGRAM_SCALE: str = "shared"
RADARGRAM_SOURCE_COLOR: bool = False   # colour sim panels by energy source
# Okabe-Ito (colour-blind safe). "layers" is reserved for a future
# internal-layer field; the kernel returns surface + bed today.
SOURCE_COLORS: dict = {"surface": "#56B4E9", "bed": "#E69F00",
                       "layers": "#009E73"}
PROFILE_REL_US: tuple = ()             # profile DATA window (not just axes)
PROFILE_X_US: tuple = ()
PROFILE_DB: tuple = ()
RSSNR_SNAPSHOT: str = ""               # pinned required-surface-SNR snapshot
RSSNR_STORE: dict = {}
RSSNR_CACHE: Path = ROOT


def activate_line(name):
    """Rebind the line-specific module globals to line definition ``name``.

    Total and reversible: every line supplies every name in LINE_GLOBALS, so
    switching lines cannot leave a previous line's value bound.
    """
    if name not in LINES:
        raise ValueError(f"unknown line {name!r}; have {sorted(LINES)}")
    missing = set(LINE_GLOBALS) - set(LINES[name])
    if missing:
        raise ValueError(f"line {name!r} defines no {sorted(missing)}")
    # study measurement conventions first, then this line's overrides of
    # them, then its geometry -- so a line override is visible but a line
    # can never redefine a name the study does not have.
    resolved, _changed = LINE_ANALYSIS[name]
    globals().update(resolved.to_globals())
    globals().update(LINES[name])
    return name


# bind the default line at import; every module global below is a line value
activate_line(DEFAULT_LINE)


def pass_season(spec):
    """Season of one pass spec: its own ``season`` if the line mixes seasons
    (the Greenland pair flies 2014 and 2017), else the line season."""
    return spec.get("season", SEASON)


def case_tag(picked_bed, gamma_rssnr=False, proc=False, dgn=False):
    return ((PBED_TAG if picked_bed else "")
            + (DGN_TAG if dgn else "")
            + (GRSSNR_TAG if gamma_rssnr else "")
            + (PROC_TAG if proc else ""))


def ref_bed_picks():
    """Radar-picked bed elevation along the anchor line (the reference LOW
    pass), on the anchor along-track axis (the line CRS, s = 0 at the first
    REF_FRAMES trace).

    Elevation convention is the one the tool's registration fits already use
    (run_altitude_comparison): ellipsoidal ice surface = Elevation -
    c*Surface/2, ice thickness = (Bottom - Surface)*c/(2*sqrt(EPS_ICE)), bed
    = surface - thickness, with the same rac.EPS_ICE and the same
    WGS84-ellipsoidal datum as the REMA + BedMachine scene stack (no geoid
    term), so the residual against BedMachine is datum-consistent. Pick gaps
    stay NaN."""
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    xs, ys, beds = [], [], []
    for fid in REF_FRAMES:
        frame = _retry(f"ref frame {fid}",
                       lambda f=fid: load_frame(REF_SEASON, f))
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


def roughness_rms(s, z, win_m=None):
    """rms of z about a running mean of width win_m -- the scout's along-track
    bed roughness metric (BedMachine 33.3 m vs radar picks 60.5 m over the
    50 km segment). NaNs are linearly interpolated first."""
    win_m = ROUGH_WIN_M if win_m is None else win_m   # active line, call time
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
    tr = Transformer.from_crs(CRS, base.crs, always_xy=True)
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
                "dem": round(roughness_rms(s_ref[seg], bm[seg]), 1),
                "picked": round(roughness_rms(s_ref[seg], pick[seg]), 1),
                "scout_reference": {"bedmachine": 33.3, "radar_picks": 60.5}},
            "bed_clamp_frac_after": round(clamp, 6),
            "note": PICKED_BED_NOTE}


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


def gamma_surface_fresnel_db(eps_ice=None):
    """|Gamma_surf|^2 (dB) of a smooth air->ice interface at normal
    incidence: the PHYSICAL surface reflectivity a line's manual
    gamma_surface is judged against (~-11.03 dB at eps 3.17)."""
    g = fresnel_normal(1.0, eps_ice or rac.EPS_ICE)
    return float(20.0 * np.log10(abs(g)))


def t2_db(eps_ice=None):
    """Two-way power transmission through that interface (dB, negative):
    field (1 - g^2) each way (~-0.71 dB)."""
    g = fresnel_normal(1.0, eps_ice or rac.EPS_ICE)
    return float(20.0 * np.log10(1.0 - g * g))


def k_phys_db(eps_ice=None):
    """gamma_surface_fresnel - T^2: what the mapping constant is when the
    surface reference is a smooth Fresnel interface. Kept as the named
    diagnostic baseline; a line's (gamma_surface - T^2) minus this equals
    its surface-reference anomaly."""
    return gamma_surface_fresnel_db(eps_ice) - t2_db(eps_ice)


def manual_gamma_surface_db():
    """The active line's manual gamma_surface for a bare run(). A solve
    line has no number until the --config driver's seed/verify loop runs,
    so a bare run() must be given gamma_surface_db explicitly."""
    if CALIBRATION["gamma_surface_db"] == "solve":
        raise ValueError(
            f"line {LINE!r} calibrates gamma_surface by solve, which needs "
            "the --config driver (main_config owns the seed/verify loop); "
            "pass gamma_surface_db explicitly for a bare run()")
    return float(CALIBRATION["gamma_surface_db"]["value"])


def gamma_solve_required(gamma0_db, meas_db, clutter_db, bed_db,
                         min_headroom_db):
    """Per-pass gamma_surface reproducing the measured bed-window level.

    The simulated bed window is S + B(gamma): surface returns S fixed, bed
    returns B moving dB-for-dB with the mapping constant. Inverting
    M = S + B gives gamma_required = gamma0 + (M (-) S) - B (power
    subtraction) -- exact at ANY clutter contamination and gamma0-invariant,
    so the solve is one evaluation plus a verification. A pass qualifies
    when the measured level stands >= min_headroom_db above the modeled
    clutter floor; below that (including M < S, where NO gamma can
    reproduce the window) the bed information is inside the clutter-model
    error and the pass must not vote. Returns (per_pass, qualifying keys,
    median, spread); NaN median = nothing qualifies, the solve refuses."""
    per = {}
    for k in meas_db:
        h = meas_db[k] - clutter_db[k]
        g = None
        if h > 0.0:
            g = round(gamma0_db + 10.0 * np.log10(
                10.0 ** (meas_db[k] / 10.0)
                - 10.0 ** (clutter_db[k] / 10.0)) - bed_db[k], 2)
        per[k] = {"headroom_db": round(h, 2), "gamma_required_db": g}
    qual = [k for k in per if per[k]["headroom_db"] >= min_headroom_db]
    vals = [per[k]["gamma_required_db"] for k in qual]
    med = float(np.median(vals)) if qual else float("nan")
    spread = (round(max(vals) - min(vals), 2) if qual else float("nan"))
    return per, qual, med, spread


def solve_attenuation_regression(s_m, rssnr, thick_m, qc, gl_km=None,
                                 gl_aware=True, settings=None):
    """Solve A by robust linear regression of RSSNR on two-way thickness.

    RSSNR(x) = 2 A H(x) + const, ASSUMING (declared, testable, and rejected
    on the geikie line) that gamma_bed is uncorrelated with H and that
    gamma_surface and A are constant over the line. Theil-Sen (repo
    convention: one bright reach cannot set the slope). Censored samples are
    EXCLUDED, never floored -- flooring would manufacture slope. GL-aware by
    default: floating samples see a different reflector and are excluded
    when the line has a grounding line.

    Returns (A_db_per_km, diagnostics)."""
    settings = settings or ATTENUATION_REGRESSION
    s_m = np.asarray(s_m, float)
    rssnr = np.asarray(rssnr, float)
    h_km = np.asarray(thick_m, float) / 1e3
    ok = np.asarray(qc, bool) & np.isfinite(rssnr) & np.isfinite(h_km)         & (h_km > 0)
    n_floating = 0
    if gl_aware and gl_km is not None:
        floating = s_m > gl_km * 1e3
        n_floating = int((ok & floating).sum())
        ok &= ~floating
    if ok.sum() < settings["min_samples"]:
        raise ValueError(
            f"attenuation regression: only {int(ok.sum())} usable samples "
            f"(< {settings['min_samples']})")
    span = float(h_km[ok].max() - h_km[ok].min()) * 1e3
    if span < settings["min_thickness_span_m"]:
        raise ValueError(
            f"attenuation regression: thickness span {span:.0f} m < "
            f"{settings['min_thickness_span_m']} m -- no leverage on A")
    x, y = 2.0 * h_km[ok], rssnr[ok]
    slope, intercept, lo, hi = stats.theilslopes(y, x)
    r = float(np.corrcoef(x, y)[0, 1])
    resid = y - (slope * x + intercept)
    # residual-vs-H structure: a trend here means the gamma_bed-H
    # independence assumption is failing (the geikie confounder signature)
    rr = float(np.corrcoef(x, np.abs(resid))[0, 1]) if len(x) > 3 else np.nan
    return float(slope), {
        "att_db_per_km": round(float(slope), 2),
        "slope_ci95_db_per_km": [round(float(lo), 2), round(float(hi), 2)],
        "intercept_db": round(float(intercept), 2),
        "r_rssnr_vs_2h": round(r, 3),
        "resid_rms_db": round(float(np.sqrt(np.mean(resid ** 2))), 2),
        "r_absresid_vs_2h": round(rr, 3),
        "n_used": int(ok.sum()), "n_floating_excluded": n_floating,
        "thickness_span_m": round(span, 0),
        "gl_aware": bool(gl_aware and gl_km is not None),
        "assumption": "gamma_bed uncorrelated with H; gamma_surface and A "
                      "constant over the line. REJECTED on the geikie line "
                      "(thawed-bed Gamma-H confounder); the slope CI and "
                      "residual structure are the tests.",
        "note": "Theil-Sen regression of RSSNR on 2H over the line's own "
                "store samples; censored samples excluded, floating samples "
                "excluded when gl_aware."}


def segment_s_range(ref, segment):
    """Anchor-axis s range (m) of the study segment, from the LOW pass's
    trace slices (the axis's own frames)."""
    off = dict(zip(ref["frames"],
                   np.concatenate([[0], np.cumsum(ref["frame_len"])[:-1]])))
    n_by = dict(zip(ref["frames"], ref["frame_len"]))
    ss = []
    # the REFERENCE pass, not a hardcoded name: this is the pass whose picks
    # define the axis, and 'low' stopped existing when the passes were renamed
    for fid, (a, b) in PASSES[REF_PASS][segment]:
        a, b = (a or 0), (n_by[fid] if b is None else b)
        ss += [ref["s"][off[fid] + a], ref["s"][off[fid] + b - 1]]
    return float(min(ss)), float(max(ss))


def rssnr_gamma_profile(s, rssnr, thick_m, qc, att_db_per_km,
                        gamma_surface_db, seg_lo, seg_hi, g2_offset_db=0.0):
    """The physical reflectivity mapping (unit-tested, anchoring-free):

        |Gamma_bed|^2(x) = 2*A*H(x) - RSSNR(x) + (gamma_surface - T^2)

    gamma_surface is the line's MANUAL effective surface reference and A its
    calibrated attenuation, so the constant is direct -- no median anchoring,
    no level deficit, and therefore no segment dependence: the same
    calibration produces the same field on any window of the line. The old
    K survives as a recorded diagnostic, k_db = gamma_surface - T^2, with
    k_minus_kphys_db == the surface-reference anomaly (gamma_surface -
    Fresnel). Censored samples (qc fail / RSSNR NaN: bed too dim to pick)
    take the segment's dimmest mapped value -- a brightness floor, not
    missing-at-random. ``g2_offset_db`` carries the bed-roughness
    double-count guard only."""
    s = np.asarray(s, np.float64)
    rssnr = np.asarray(rssnr, np.float64)
    thick_m = np.asarray(thick_m, np.float64)
    ok = (np.asarray(qc, bool) & np.isfinite(rssnr) & np.isfinite(thick_m)
          & (thick_m > 0))
    seg = ok & (s >= seg_lo) & (s <= seg_hi)
    if seg.sum() < 5:
        raise RuntimeError(f"only {seg.sum()} usable RSSNR samples in the "
                           "segment")
    k = float(gamma_surface_db) - t2_db() + g2_offset_db
    g2 = 2.0 * att_db_per_km * thick_m / 1e3 - rssnr + k
    g2_const = float(20.0 * np.log10(abs(
        fresnel_normal(rac.EPS_ICE, rac.EPS_BED))))
    floor = float(np.nanmin(g2[seg]))
    g2 = np.where(ok, g2, floor)
    o = np.argsort(s)
    kp = k_phys_db()
    gs = g2[seg]
    return {"s": s[o], "g2_db": g2[o], "thick_m": thick_m[o],
            "ok": ok[o], "k_db": round(k, 2), "k_phys_db": round(kp, 2),
            "k_minus_kphys_db": round(k - kp, 2),
            "gamma_surface_db": round(float(gamma_surface_db), 2),
            "gamma_surface_fresnel_db": round(gamma_surface_fresnel_db(), 2),
            "surface_anomaly_db": round(
                float(gamma_surface_db) - gamma_surface_fresnel_db(), 2),
            "g2_const_db": round(g2_const, 2),
            "att_db_per_km": att_db_per_km,
            "n_samples": int(len(s)), "n_censored": int((~ok).sum()),
            "censored_floor_db": round(floor, 2),
            "seg_s_km": [round(seg_lo / 1e3, 2), round(seg_hi / 1e3, 2)],
            "n_seg": int(seg.sum()),
            "g2_pos_frac_seg": round(float((gs > 0).mean()), 3),
            "g2_over_fresnel_frac_seg": round(float((gs > g2_const).mean()),
                                              3),
            "g2_max_over_fresnel_db": round(float(gs.max() - g2_const), 2),
            "g2_seg_db": {kk: round(float(vv), 1) for kk, vv in
                          [("min", gs.min()), ("p5", np.percentile(gs, 5)),
                           ("med", np.median(gs)),
                           ("p95", np.percentile(gs, 95)),
                           ("max", gs.max())]},
            "med_sample_spacing_m": round(float(np.median(np.diff(s[o]))), 0)}


def bed_rough_nadir_db(sigma_m, f0=None, eps_ice=None):
    """Gerekos coherent-term mean-POWER attenuation at NADIR on a buried bed
    facet, in dB: 10*log10(exp(-sigma^2 K^2)) with K = 2*k_ice*cos(0) (the
    facet's LOCAL medium is ice -- docs/roughness.md). Returned NEGATIVE."""
    # f0 defaults to the ACTIVE line's carrier, resolved here rather than
    # bound at import (the Greenland pair is 195 MHz, not 190).
    k_ice = (2.0 * np.pi * (f0 or FC_HZ)
             * np.sqrt(eps_ice or rac.EPS_ICE) / C)
    return float(-(sigma_m * 2.0 * k_ice) ** 2 * 10.0 / np.log(10.0))


def zone_g2_stats(gmap, run_lo, run_hi, gl_km=None):
    """ZONE-AWARE implied-reflectivity physicality (full_line): the mapped
    |Gamma_bed|^2 judged against each zone's OWN Fresnel ceiling -- grounded
    traces vs the ice->rock anchor (the -12.9 dB Fresnel constant the median
    anchoring used) with 0 dB as the hard physical bound, floating traces vs
    the ice->SEAWATER coefficient (~-3.5 dB), which is a genuine CEILING for
    a specular ice-ocean interface (nothing at the shelf base can beat it)."""
    gl_km = GL_S_KM if gl_km is None else gl_km    # active line, call time
    ceil_f = float(20.0 * np.log10(abs(
        fresnel_normal(rac.EPS_ICE, EPS_SEAWATER))))
    gl_m = gl_km * 1e3
    out = {"gl_s_km": gl_km, "floating_ceiling_db": round(ceil_f, 2),
           "grounded_fresnel_anchor_db": gmap["g2_const_db"],
           "note": "implied |Gamma_bed|^2 per zone under the run's K. "
           "Grounded: fraction above 0 dB is the unphysical fraction "
           "(established diagnostic), the ice->rock Fresnel anchor is the "
           "reference level. Floating: the ice->seawater Fresnel "
           "coefficient IS the physical ceiling (specular ice-ocean "
           "interface); fractions above it and above 0 dB are both "
           "recorded. recorded only"}
    for name, lo, hi in (("grounded", run_lo, min(gl_m, run_hi)),
                         ("floating", max(gl_m, run_lo), run_hi)):
        m_all = (gmap["s"] >= lo) & (gmap["s"] <= hi)
        m = gmap["ok"] & m_all
        if m.sum() < 3:
            out[name] = {"n": int(m.sum()), "note": "too few samples"}
            continue
        g = gmap["g2_db"][m]
        out[name] = {
            "n": int(m.sum()), "n_total": int(m_all.sum()),
            "qc_pass_frac": round(float(gmap["ok"][m_all].mean()), 3),
            "s_km": [round(lo / 1e3, 2), round(hi / 1e3, 2)],
            **{k: round(float(v), 1) for k, v in
               [("min", g.min()), ("p5", np.percentile(g, 5)),
                ("med", np.median(g)), ("p95", np.percentile(g, 95)),
                ("max", g.max())]},
            "frac_above_0db": round(float((g > 0.0).mean()), 3),
            "frac_above_seawater_ceiling": round(
                float((g > ceil_f).mean()), 3),
            "med_minus_zone_ceiling_db": round(float(
                np.median(g) - (0.0 if name == "grounded" else ceil_f)), 1)}
    return out


def build_rssnr_gamma(axis, segment, att, gamma_surface_db,
                      bed_rough_sigma=None, extra_db=0.0, zone_gl_km=None):
    """Fetch + map: the shared G2(s) profile dict (rssnr_gamma_profile output
    + fetch provenance) on the anchor along-track axis (ref_bed_picks),
    under the line's calibration (gamma_surface manual or solved, A
    calibrated).
    Anchoring-free: the mapping constant is gamma_surface - T^2 directly, so
    the same calibration yields the same field on any segment -- the old
    per-segment K anchoring (K_ANCHOR_SEGMENT) is obsolete.

    DOUBLE-COUNT GUARD (``bed_rough_sigma``, T1): the RSSNR-derived G2 is
    calibrated against the MEASURED bed echo, which already contains the
    real bed's roughness loss; enabling sub-facet bed roughness would apply
    it twice, so G2 is raised by the nadir coherent attenuation (+ the
    recorded empirical residual)."""
    d, prov = fetch_rssnr_anchor()
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    sx, sy = tr.transform(d["lon"], d["lat"])
    s_smp = project_to_track(sx, sy, axis["x"], axis["y"], axis["s"])
    thick = C / (2.0 * np.sqrt(axis["eps_ice"])) * (d["btw"] - d["stw"])
    seg_lo, seg_hi = segment_s_range(axis, segment)
    shift = ((-bed_rough_nadir_db(bed_rough_sigma) + extra_db)
             if bed_rough_sigma else 0.0)
    prof = rssnr_gamma_profile(s_smp, d["rssnr"], thick, d["qc"], att,
                               gamma_surface_db, seg_lo, seg_hi,
                               g2_offset_db=shift)
    if bed_rough_sigma:
        prof["bed_rough_guard"] = {
            "sigma_m": bed_rough_sigma,
            "nadir_coherent_attenuation_db":
                round(bed_rough_nadir_db(bed_rough_sigma), 2),
            "empirical_extra_db": round(extra_db, 2),
            "g2_shift_db": round(shift, 2),
            "note": "G2 raised by the nadir coherent-term roughness "
            "attenuation so the nadir bed level is conserved (the RSSNR "
            "calibration already contains the real bed's roughness loss)."}
    if zone_gl_km is not None:
        run_lo, run_hi = segment_s_range(axis, segment)
        prof["g2_zones_db"] = zone_g2_stats(prof, run_lo, run_hi,
                                            gl_km=zone_gl_km)
    prof["provenance"] = prov
    prof["note"] = RSSNR_GAMMA_NOTE
    return prof


def apply_rssnr_gamma(base, axis, gmap, spec=None):
    """Attach ``base.gamma_bed``: per-map-pixel signed FIELD reflection
    coefficient -10^(G2(s_pix)/20) from the shared profile, constant along
    the cross-track normal (apply_picked_bed's projection). Returns recorded
    stats."""
    bed = base.dems[1]
    tr = Transformer.from_crs(CRS, base.crs, always_xy=True)
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
    g2_grid = g2.reshape(bed.shape)
    stats = {}
    if spec is None:
        base.gamma_bed = (-(10.0 ** (g2_grid / 20.0))).astype(np.float32)
    else:
        # SPECULAR/DIFFUSE split of the SAME mapped |Gamma|^2 (T5):
        #   specular  f_s * G2 * G(psi),  G = exp(-tan^2(psi)/(2 s0^2))
        #   diffuse   (1 - f_s) * G2,     shaped in-kernel by cos^n(theta)
        # G(psi) is "bright because flat": a facet tilted by psi mirrors the
        # nadir-looking radar to 2*psi off-nadir, so it must not inherit the
        # nadir-calibrated brightness. psi comes from the BED DEM gradient on
        # the scene grid (32 m) -- a proxy for the facet tilt (facets are
        # 10.7-49.8 m and are built from this same DEM); recorded, not tuned.
        f_s, s0_deg, n_exp = spec
        psi = bed_tilt_rad(bed, base.transform)
        gw = (np.ones_like(psi) if not s0_deg
              else np.exp(-np.tan(psi) ** 2
                          / (2.0 * np.tan(np.deg2rad(s0_deg)) ** 2)))
        # DOUBLE-COUNT GUARD (same logic as the T1 roughness guard): the
        # RSSNR-mapped |Gamma|^2 is calibrated against the MEASURED bed echo,
        # which already contains the real bed's tilt mix. G(psi) must
        # therefore act as a RELATIVE reweighting with unit scene mean, not
        # as an absolute loss -- otherwise the nadir bed level collapses (it
        # does: unnormalized s0 = 1 deg on this bed costs 20-38 dB, recorded
        # as trial A). With <G> = 1 the split conserves nadir bed power for
        # every (f_s, s0) by construction: <f_s*G_n> + (1 - f_s) = 1.
        gw_mean = float(gw.mean())
        gw = gw / max(gw_mean, 1e-300)
        amp = 10.0 ** (g2_grid / 20.0)
        base.gamma_bed = (-(np.sqrt(f_s * gw) * amp)).astype(np.float32)
        if f_s < 1.0:
            base.diffuse_bed = (np.sqrt(1.0 - f_s) * amp).astype(np.float32)
        pw = np.rad2deg(psi)
        stats = {"specular_fraction": f_s, "spec_tilt_s0_deg": s0_deg,
                 "diffuse_exponent": n_exp,
                 "bed_tilt_deg": {k: round(float(v), 2) for k, v in
                                  [("med", np.median(pw)),
                                   ("p90", np.percentile(pw, 90)),
                                   ("max", pw.max())]},
                 "specular_tilt_weight_db": {
                     k: round(float(10.0 * np.log10(max(v, 1e-300))), 1)
                     for k, v in [("med", np.median(gw)),
                                  ("p10", np.percentile(gw, 10)),
                                  ("p90", np.percentile(gw, 90)),
                                  ("max", gw.max())]},
                 "mean_normalization_db": round(
                     float(10.0 * np.log10(max(gw_mean, 1e-300))), 2),
                 "note": "G(psi) is a POWER weight on the SPECULAR channel "
                 "only, MEAN-NORMALIZED over the scene grid (double-count "
                 "guard: the RSSNR calibration already contains the bed's "
                 "tilt mix), so <G> = 1 and the nadir bed level is conserved "
                 "for every (f_s, s0). mean_normalization_db is the raw "
                 "<G> that was divided out. The diffuse channel keeps the "
                 "full (1-f_s) share at every tilt. psi from the 32 m "
                 "bed-DEM gradient (facets are 10.7-49.8 m and are built "
                 "from this same DEM)."}
    return {"k_db": gmap["k_db"], "k_phys_db": gmap["k_phys_db"],
            **({"spec_diffuse": stats} if stats else {}),
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
# DEMOGORGN bed (--demogorgn-bed): geostatistically simulated bed realization
# ========================================================================
# claude_notes/demogorgn_scout.md: SGS/MCMC 100-member ensemble on the
# Bedmap3 500 m grid, CONDITIONED on this very flight line (ensemble sd
# 0.5 m at nadir), isotropic 2-D texture (AT/CT 1.03 vs picked-bed's 1.90),
# 88% of the 500 m-resolvable pick roughness (50.3 vs BedMachine's 28.5 m
# rms). KNOWN, DOCUMENTED, NOT TUNED AWAY: its nadir bed sits ~+44 m raw /
# ~43.7 m rms off OUR picks (a thickness-convention disagreement with the
# Bedmap3-ingested version of this survey) -> expect a visible ~0.5 us bed-
# line offset vs the measured panels. PLAIN DEMOGORGN only (the clean
# three-way bed-source ablation); the picked-bed hybrid (residual drops
# 81.3 -> 43.7 m rms, anisotropy 1.90 -> 1.27) is a RECORDED FOLLOW-UP,
# deliberately not wired.
DGN_TAG = "_dgn"
DGN_NOTE = (
    "bed = DEMOGORGN-Antarctica realization (500 m Bedmap3 grid, EIGEN-6C4 "
    "geoid added from the BedMachine cache band 2), bilinearly resampled "
    "onto the 32 m scene grid; pinned snapshot, seed recorded. Conditioned "
    "on this line (ensemble sd 0.5 m at nadir) with isotropic 2-D texture; "
    "its nadir bed differs from our picks by ~43.7 m rms (+44 m median raw; "
    "thickness-convention disagreement, scout-documented); with the picked "
    "bed (bed.nadir: picked) the along-track pick residual is applied on top.")


def bed_tilt_rad(bed, transform):
    """Facet-scale tilt from horizontal (rad) of a DEM on an affine grid:
    arctan|grad z|, central differences with edge-replication."""
    dx, dy = abs(transform.a), abs(transform.e)
    gy, gx = np.gradient(np.asarray(bed, np.float64), dy, dx)
    return np.arctan(np.hypot(gx, gy))


def apply_demogorgn_bed(base, fsub, ct_m, seed):
    """Replace the base scene's bed DEM with the DEMOGORGN realization
    (ellipsoidal, via opr.fetch_demogorgn_window), resampled onto the scene
    grid and clamped below the surface. Returns recorded stats."""
    from soundersim.opr import (DEMOGORGN_SNAPSHOT, fetch_demogorgn_window,
                                fill_nodata_nearest)

    lat, lon = rac._lonlat(fsub)
    bounds = (float(lon.min()), float(lat.min()),
              float(lon.max()), float(lat.max()))
    dgn, tr_d, crs_d, meta = fetch_demogorgn_window(
        bounds, pad_m=ct_m + 600.0, seed=seed)
    bed = rac.resample_to_grid(dgn, tr_d, crs_d, base.dems[0].shape,
                               base.transform, base.crs)
    bed, fill = fill_nodata_nearest(bed)
    clamp = float((bed > base.dems[0] - 0.1).mean())
    base.dems[1] = np.minimum(bed, base.dems[0] - 0.1).astype(np.float32)
    base.params["bed_source"] = DGN_NOTE
    return {"seed_id": int(seed), "snapshot_id": DEMOGORGN_SNAPSHOT,
            "posting_m": meta["posting_m"], "datum": meta["returned_datum"],
            "nodata_fill_frac": round(fill, 6),
            "bed_clamp_frac": round(clamp, 6), "note": DGN_NOTE}


# ========================================================================
# HYBRID bed (full_line segment): grounded DEMOGORGN + floating radar-picked
# shelf base, blended across the grounding line
# ========================================================================
# Beyond the GL the radar's basal reflector is the ICE-OCEAN interface;
# BedMachine/DEMOGORGN report the SEAFLOOR under the cavity there (scout
# quirk 1), so neither may supply the floating "bed". The floating base is
# instead the LOW pass's radar basal picks (the established pick reference,
# load_bottom_pick machinery), nearest-neighbour-interpolated in anchor
# along-track s and extended cross-track as a constant. PLAINLY RECORDED
# APPROXIMATION: the shelf base is modeled as flat-ish cross-track (no
# basal crevasses/channels; the 1-D picks cannot supply cross-track relief,
# so unlike the grounded picked-bed residual there is no 2-D DEM to
# preserve). The grounded side (s < GL) is the DEMOGORGN realization
# EXACTLY as in the extended run; the two grids are blended over
# GL -> GL + GL_RAMP_KM so the documented ~10-20 m DEMOGORGN-vs-picks nadir
# offset and the texture change do not step at the GL. Chunks spanning the
# GL crop this scene-level hybrid, so their facets are built from the
# blended grid.
HYBRID_BED_NOTE = (
    "HYBRID bed (crosses_gl segment): s < GL = the line's grounded DEM "
    "(DEMOGORGN realization, seed recorded), "
    "s > GL + ramp = LOW-pass radar basal picks (ice-ocean interface), "
    "nearest-neighbour in anchor s and constant cross-track (flat-ish "
    "shelf-base approximation, recorded); linear blend over the ramp. "
    "BedMachine/DEMOGORGN report the SEAFLOOR beyond the GL, not the shelf "
    "base the radar sees, and are never used there.")


def picks_bed_nn(axis, s_q):
    """Nearest-neighbour (in anchor along-track s) radar-picked bed
    elevation at query positions ``s_q`` (m). Pick gaps are skipped (the
    nearest FINITE pick wins); edge-clamped."""
    ok = np.isfinite(axis["bed"])
    s_f, b_f = axis["s"][ok], axis["bed"][ok]
    j = np.clip(np.searchsorted(s_f, s_q), 1, len(s_f) - 1)
    left = (np.asarray(s_q) - s_f[j - 1]) < (s_f[j] - np.asarray(s_q))
    return b_f[np.where(left, j - 1, j)]


def apply_hybrid_bed(base, fsub, ct_m, seed, axis):
    """Replace the base scene's bed DEM with the HYBRID grounded-DEMOGORGN /
    floating-picks grid (module-section comment). DEMOGORGN is fetched over
    the grounded(+ramp) part of the track only -- its values carry zero
    weight beyond GL + ramp, so the floating stretch needs no seafloor
    fetch. Returns recorded stats (superset of apply_demogorgn_bed's)."""
    from soundersim.opr import (DEMOGORGN_SNAPSHOT, fetch_demogorgn_window,
                                fill_nodata_nearest)

    gl_m, ramp_m = GL_S_KM * 1e3, GL_RAMP_KM * 1e3
    dem = np.asarray(base.dems[0], np.float64)
    tr = Transformer.from_crs(CRS, base.crs, always_xy=True)
    rx, ry = tr.transform(axis["x"], axis["y"])
    ny, nx = dem.shape
    cols, rows = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
    px, py = base.transform * (cols.ravel(), rows.ravel())
    s_pix = project_to_track(px, py, rx, ry, axis["s"]).reshape(dem.shape)

    # DEMOGORGN window: grounded track + ramp + 2 km margin only
    lat, lon = rac._lonlat(fsub)
    tr4 = Transformer.from_crs("EPSG:4326", base.crs, always_xy=True)
    nx_, ny_ = tr4.transform(lon, lat)
    s_nav = project_to_track(nx_, ny_, rx, ry, axis["s"])
    m_g = s_nav <= gl_m + ramp_m + 2000.0
    if not m_g.any():
        raise RuntimeError("hybrid bed: no traces on the grounded side")
    bounds = (float(lon[m_g].min()), float(lat[m_g].min()),
              float(lon[m_g].max()), float(lat[m_g].max()))
    dgn, tr_d, crs_d, meta = fetch_demogorgn_window(
        bounds, pad_m=ct_m + 600.0, seed=seed)
    dgn_grid = rac.resample_to_grid(dgn, tr_d, crs_d, dem.shape,
                                    base.transform, base.crs)
    dgn_grid, fill = fill_nodata_nearest(dgn_grid)   # fills OUTSIDE the
    # grounded fetch window too -- those pixels carry zero blend weight

    pick_grid = picks_bed_nn(axis, s_pix.ravel()).reshape(dem.shape)
    w = np.clip((gl_m + ramp_m - s_pix) / ramp_m, 0.0, 1.0)  # 1 = DEMOGORGN
    bed_new = w * dgn_grid + (1.0 - w) * pick_grid

    grounded = s_pix < gl_m
    floating = ~grounded
    blend = (s_pix >= gl_m) & (s_pix <= gl_m + ramp_m)
    step = dgn_grid[blend] - pick_grid[blend]        # what the ramp absorbs
    # nadir-only step: the same difference sampled ON the anchor track in
    # the blend s-range (the cross-track-inclusive stat above also carries
    # DEMOGORGN's 2-D relief against the cross-track-constant picks)
    mb = (axis["s"] >= gl_m) & (axis["s"] <= gl_m + ramp_m)
    step_nad = (sample_dem(dgn_grid, base.transform, rx[mb], ry[mb])
                - picks_bed_nn(axis, axis["s"][mb]))
    clear = dem - bed_new
    clamp = float((bed_new > dem - 0.1).mean())
    base.dems[1] = np.minimum(bed_new, dem - 0.1).astype(np.float32)
    base.params["bed_source"] = HYBRID_BED_NOTE

    ok_f = np.isfinite(axis["bed"]) & (axis["s"] >= gl_m)
    return {"seed_id": int(seed), "snapshot_id": DEMOGORGN_SNAPSHOT,
            "posting_m": meta["posting_m"], "datum": meta["returned_datum"],
            "nodata_fill_frac": round(fill, 6),
            "bed_clamp_frac": round(clamp, 6),
            "hybrid": {
                "gl_s_km": GL_S_KM, "ramp_km": GL_RAMP_KM,
                "grounded_source": f"DEMOGORGN seed {int(seed)} "
                                   f"(snapshot {DEMOGORGN_SNAPSHOT})",
                "floating_source": "LOW-pass radar basal picks, NN in "
                                   "anchor s, constant cross-track",
                "demogorgn_fetch_track_max_s_km": round(
                    float(s_nav[m_g].max()) / 1e3, 2),
                "blend_zone_dgn_minus_picks_m": {
                    "med": round(float(np.median(step)), 1),
                    "rms": round(float(np.sqrt(np.mean(step ** 2))), 1),
                    "absmax": round(float(np.abs(step).max()), 1),
                    "note": "over the FULL cross-track blend zone: includes "
                    "DEMOGORGN's 2-D relief vs the cross-track-constant "
                    "picks, not just the nadir offset"},
                "blend_zone_nadir_dgn_minus_picks_m": {
                    "med": round(float(np.median(step_nad)), 1),
                    "rms": round(float(np.sqrt(np.mean(step_nad ** 2))), 1),
                    "absmax": round(float(np.abs(step_nad).max()), 1),
                    "n": int(mb.sum()),
                    "note": "sampled ON the anchor track in the blend "
                    "s-range: the nadir step the ramp absorbs"},
                "clearance_m": {
                    "min": round(float(clear.min()), 1),
                    "grounded_min": round(float(clear[grounded].min()), 1),
                    "floating_min": round(float(clear[floating].min()), 1),
                    "floating_med": round(float(
                        np.median(clear[floating])), 1),
                    "clamp_frac_grounded": round(float(
                        (bed_new > dem - 0.1)[grounded].mean()), 6),
                    "clamp_frac_floating": round(float(
                        (bed_new > dem - 0.1)[floating].mean()), 6)},
                "floating_picks": {"n": int(ok_f.sum()),
                                   "gap_frac": round(float(
                                       1.0 - ok_f.sum()
                                       / max((axis["s"] >= gl_m).sum(), 1)),
                                       5)},
                "note": HYBRID_BED_NOTE},
            "note": HYBRID_BED_NOTE}


# ========================================================================
# CSARP_standard-matching processing (--processing standard)
# ========================================================================
# THE REAL CHAIN (recorded provenance): motion-compensated f-k migration,
# sigma_x = 2.5 m SLC, start_eps 3.15 -- read from THIS season's own
# param_csarp/param_sar structs (scout table, hand-verified ft_wind
# hanning); then delay-and-sum channel combine and incoherent look
# averaging rline_rng [-5..5] = 11 looks with dline 6 -> ~14.85 m posting,
# ~25 m EFFECTIVE along-track resolution (the M24-verified CReSIS standard
# convention; 11/6 were verified on 2017/2019 P3 param structs and stated
# for 2016 by the coordinator -- the 2016 combine struct records only
# method=standard, so 11/6 is a RECORDED ASSUMPTION here, not a 2016
# param read).
#
# OUR CHAIN (as close as the compute-feasible along-track sampling allows):
# simulate at the PRODUCT posting (~14.85 m -- sim traces land on the
# measured columns), first-order nadir motion compensation to a smoothed
# reference track (field *= exp(+2jk dz); the real chain is motion
# compensated, our focuser is straight-track), then
# soundersim.processing.focused_sar (straight-track time-domain
# backprojection == f-k in its validity domain, hann aperture taper) with
# the ALIAS-LIMITED aperture at the pass's median optical bed range:
# theta = asin(lam/(4*ds)) is the largest unaliased Doppler half-angle at
# posting ds, giving ~1.44*ds ~ 21 m hann azimuth resolution -- close to
# the product's ~25 m effective POST-LOOK resolution (its 2.5 m SLC
# resolution would need 2.5 m posting: ~40x the compute). Then
# N_LOOKS_SIM-look stride-1 incoherent averaging (posting preserved; the
# product's dline-6 decimation is a no-op at matched posting).
#
# RECORDED GAPS (an honest list, not silent guesses):
#  g1 SLC resolution 2.5 m unreachable -> matched at the ~25 m effective
#     post-look level instead;
#  g2 fewer independent looks (~2-3 vs the product's ~6-11): speckle
#     contrast up to ~2x the product's;
#  g3 focusing is through AIR only (processing.py scope): in-ice migration
#     is absent -- bed arcs carry a residual quadratic phase (~1 rad at the
#     aperture edge at altitude, hann-tapered); the real chain migrates
#     with start_eps 3.15;
#  g4 motion compensation is first-order nadir-vertical only (dz to a
#     smoothed track); the real chain compensates full 3-D motion + channel
#     phase. Cross-track wander enters at second order (~dy^2/2r: mm);
#  g5 the surface range is inside the bed-range-sized aperture cone ->
#     specular near-surface Doppler mildly aliased (worst on the low pass,
#     theta_surf/theta_alias ~ 3.6; the guard warning is caught+recorded);
#  g6 delay-and-sum channel combine is already inside the sim's array
#     pattern (M22); no per-channel processing to combine.
PROC_TAG = "_proc"
FIG_WIDTH_SCALE = 1.0   # --fig-width-scale: radargram panel width multiplier
# Radargram bed overlay (--no-bed-overlay turns it off): the MEASURED Bottom
# pick, DOTTED, on MEASURED PANELS ONLY. It is drawn from the same array the
# measured bed-window metrics use, so a visible mismatch between the line and
# the echo IS a registration/pick problem, not a plotting artefact -- which
# is the point of having it on by default. Simulated panels carry NO bed
# line: the sim's bed nadir twtt is a model input, not an independent pick,
# so drawing it there invites reading a tautology as agreement.
BED_OVERLAY_STYLE = dict(color="tab:cyan", lw=1.1, ls=":", alpha=0.9)
BED_OVERLAY = True      # --no-bed-overlay


def alias_limited_aperture(lam, spacing_m, r_ref_m):
    """(aperture_m, half_angle_deg): the largest synthetic aperture at range
    ``r_ref_m`` whose Doppler stays unaliased at along-track sampling
    ``spacing_m`` -- sin(theta) = lam/(4*ds) (the lambda/4 criterion), L =
    2 r tan(theta). Azimuth resolution at that limit equals the posting
    (1.44x with the hann taper)."""
    st = min(lam / (4.0 * spacing_m), 1.0)
    theta = float(np.arcsin(st))
    return 2.0 * r_ref_m * np.tan(theta), float(np.degrees(theta))


def first_fresnel_aperture(lam, r_ref_m):
    """Full diameter and half-angle of the monostatic first Fresnel zone."""
    length = np.sqrt(2.0 * lam * r_ref_m)
    theta = np.arctan((length / 2.0) / r_ref_m)
    return float(length), float(np.degrees(theta))


def _proc_ds(F2, twtt, s, lam):
    """Minimal coherent Dataset wrapper for soundersim.processing: field
    (slow_time, twtt) + straight-track positions x = along-track arc length
    (the focuser uses inter-trace distances only)."""
    T = F2.shape[0]
    z = np.zeros(T)
    return xr.Dataset(
        {"field": (("slow_time", "twtt"), F2),
         "power": (("slow_time", "twtt"), np.abs(F2) ** 2)},
        coords={"slow_time": ("slow_time", np.arange(T, dtype=float)),
                "twtt": ("twtt", twtt),
                "x": ("slow_time", np.asarray(s, float)),
                "y": ("slow_time", z), "z": ("slow_time", z)},
        attrs={"mode": "coherent", "wavelength": float(lam),
               "processing": "[]"})


def straightness_stats(x, y, z_smooth_resid, win_n):
    """Straight-track check over sliding aperture-length windows: p95/max
    horizontal deviation from the window chord, plus the residual vertical
    deviation AFTER the first-order mocomp (what the focuser cannot see)."""
    n = len(x)
    win_n = max(2, min(int(win_n), n - 1))
    dev = []
    for a in range(0, n - win_n, max(1, win_n // 2)):
        b = a + win_n
        ux, uy = x[b] - x[a], y[b] - y[a]
        nrm = np.hypot(ux, uy)
        if nrm == 0:
            continue
        dev.append(np.max(np.abs(
            (x[a:b] - x[a]) * (-uy / nrm) + (y[a:b] - y[a]) * (ux / nrm))))
    dev = np.asarray(dev) if dev else np.zeros(1)
    return {"horiz_chord_dev_p95_m": round(float(np.percentile(dev, 95)), 2),
            "horiz_chord_dev_max_m": round(float(dev.max()), 2),
            "vert_resid_rms_m": round(float(np.sqrt(np.mean(
                z_smooth_resid ** 2))), 3)}


def process_standard(p, sim, focus_aperture="alias_limited"):
    """Apply the CSARP_standard-matching chain (section comment) to the
    assembled per-layer fields. Returns dict(P, Ps, Pb, twtt, chain)."""
    from soundersim import processing as proc

    F = sim["field"]                       # (T, nb, 2) complex64
    twtt = sim["twtt"]
    lam = p["lam"]
    s_sim = p["s_sim"]
    spacing = float(np.median(np.diff(s_sim)))
    r_bed = float(C * np.nanmedian(p["bot_sim"]) / 2.0)   # optical range
    if focus_aperture == "alias_limited":
        L, theta_deg = alias_limited_aperture(lam, spacing, r_bed)
    elif focus_aperture == "product_resolution":
        base_spacing = spacing * p.get("posting_div", 1)
        L, theta_deg = alias_limited_aperture(lam, base_spacing, r_bed)
    elif focus_aperture == "first_fresnel":
        L, theta_deg = first_fresnel_aperture(lam, r_bed)
    else:
        raise ValueError(f"unknown focus aperture {focus_aperture!r}")
    r_surf = float(C * np.nanmedian(p["surf_sim"]) / 2.0)

    # first-order nadir mocomp: dz to a ~2-aperture smoothed reference track
    z = np.asarray(p["base"].nav_llh[:, 2], np.float64)
    w = max(3, int(round(2.0 * L / spacing)))
    dz = z - ndimage.uniform_filter1d(z, w, mode="nearest")
    k0 = 2.0 * np.pi / lam
    ph = np.exp(2j * k0 * dz).astype(np.complex64)[:, None]

    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    px, py = tr.transform(p["base"].nav_llh[:, 1], p["base"].nav_llh[:, 0])
    straight = straightness_stats(np.asarray(px), np.asarray(py), dz,
                                  round(L / spacing))

    layers = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        for li in range(F.shape[-1]):
            ds = _proc_ds(F[..., li] * ph, twtt, s_sim, lam)
            layers.append(proc.focused_sar(ds, aperture_m=L, window="hann")
                          .field.values)
    caught = sorted({str(w.message)[:200] for w in wlist})
    Fs, Fb = layers

    def look(P):
        return ndimage.uniform_filter1d(P, N_LOOKS_SIM, axis=0,
                                        mode="nearest")

    P = look(np.abs(Fs + Fb) ** 2)
    Ps, Pb = look(np.abs(Fs) ** 2), look(np.abs(Fb) ** 2)
    # Fs/Fb (complex64, the focuser's native output) ride along so the
    # proc cache can persist the exact source of P/Ps/Pb.
    chain = {
        "real_chain": REAL_CHAIN,
        "focus_aperture": focus_aperture,
        "sim_posting_m": round(spacing, 3),
        "aperture_m": round(L, 1), "half_angle_deg": round(theta_deg, 3),
        "aperture_traces": int(round(L / spacing)) + 1,
        "azimuth_res_hann_m": round(1.44 * lam * r_bed / (2.0 * L), 1),
        "surface_alias_ratio": round(
            np.degrees(np.arctan((L / 2.0) / r_surf)) / theta_deg, 2),
        "n_looks_sim": N_LOOKS_SIM,
        "look_span_m": round(N_LOOKS_SIM * spacing, 1),
        "mocomp": {"kind": "first-order nadir (dz to smoothed track), "
                           "field *= exp(+2jk dz)",
                   "dz_rms_m": round(float(np.sqrt(np.mean(dz ** 2))), 3),
                   "smooth_win_m": round(w * spacing, 0)},
        "straight_track_check": straight,
        "focuser_warnings": caught,
        "gaps": "g1 SLC 2.5 m res unreachable (matched at ~25 m post-look "
                "level); g2 ~2-3 vs ~6-11 independent looks (speckle "
                "contrast up to ~2x); g3 air-only focusing (no in-ice "
                "migration, real chain start_eps 3.15); g4 first-order "
                "vertical mocomp only; g5 near-surface Doppler mildly "
                "aliased inside the bed-range aperture (see "
                "surface_alias_ratio); g6 channel combine inside the sim "
                "array pattern",
    }
    return {"P": P, "Ps": Ps, "Pb": Pb, "twtt": twtt, "chain": chain,
            "Fs": Fs, "Fb": Fb}


# ------------------------------------------------------------------------
# processed-stack cache (--proc-cache): seconds-fast plot iteration
# ------------------------------------------------------------------------
# The chunk cache stores RAW per-layer fields; the matched-processing
# focuser on top of them costs minutes-to-tens-of-minutes per pass (the
# syn300km aperture is 1078 traces), which made figure iterations ~30 min.
# This cache persists the FOCUSED per-interface complex64 stacks (the
# bit-exact source of the P/Ps/Pb powers: complex64 is the focuser's
# native output dtype, and the look averaging is recomputed identically on
# load) plus the light per-trace arrays a figure needs, under
# <out>/proc_cache/<rid>.npz|.json following the run-cache meta
# conventions. STALENESS: the meta key embeds a sha256 digest of every
# source chunk's cache meta_key -- the simulation is deterministic in its
# meta, so equal digests imply an identical assembled field; a changed or
# missing chunk cache fails the key comparison and the stack is REBUILT,
# never silently served.
PROC_CACHE_DIRNAME = "proc_cache"
PROC_CACHE_VERSION = 1


def proc_rid(p):
    return (f"{p['key']}_{p['segment']}"
            f"{case_tag(p['picked_bed'], p['gamma_rssnr'], p['proc'], p['dgn'])}"
            + ("_hyb" if p.get("hybrid") else ""))


def surface_roughness_record(surf_rough, preps):
    """run_config.json entry: the fixture and, per pass, the RESOLVED
    Gaussian pair the surface interface was simulated with."""
    rec = {"model": "Gerekos 2023 sub-facet roughness on the surface "
                    "interface (acf gaussian: exact finite-facet series; "
                    "exponential: area-only law, docs/roughness.md)",
           "fixture": {"sigma_m": SURF_ROUGH_FIXTURE[0],
                       "corr_length_m": SURF_ROUGH_FIXTURE[1],
                       "why": "C&S 2020 Fig. 11 mcords 0 m firn-layer "
                              "inversion (representative, not a site "
                              "measurement)"},
           "passes": {}}
    for key, p in preps.items():
        pair, inf = resolve_surf_rough(surf_rough, p, info=True)
        rec["passes"][key] = (None if pair is None else
                              {"sigma_m": pair[0], "corr_length_m": pair[1],
                               "acf": surf_rough_acf(pair),
                               "fixture": surf_rough_is_fixture(pair),
                               **({} if inf is None else inf)})
    return rec


def chunk_digests(p, runs_dir, n_chunks, att, surf_rough,
                  antenna=None, bed_rough=None, spec=None, gfix=None):
    """{chunk rid: sha256(meta_key)[:16]} over the pass's chunk caches.

    The hypothesis knobs MUST be forwarded: chunk_rid suffixes the file
    name with them, so reconstructing the rid without them looked for a
    chunk that never existed (--specular-fraction + --proc-cache crashed
    here, 2026-08-21)."""
    import hashlib
    out = {}
    for ci in range(n_chunks):
        rid = chunk_rid(p, ci, att, surf_rough,
                        antenna=antenna if antenna is not None
                        else ANT_DEFAULT,
                        bed_rough=bed_rough, spec=spec, gfix=gfix)
        d = json.loads((Path(runs_dir) / f"{rid}.json").read_text())
        out[rid] = hashlib.sha256(d["meta_key"].encode()).hexdigest()[:16]
    return out


def _digests_current(stored, runs_dir):
    """Re-derive the digests for the STORED chunk rid list from the chunk
    cache as it exists now (missing file -> None: stale)."""
    import hashlib
    out = {}
    for rid in stored:
        jp = Path(runs_dir) / f"{rid}.json"
        if not jp.exists():
            out[rid] = None
            continue
        d = json.loads(jp.read_text())
        out[rid] = hashlib.sha256(d["meta_key"].encode()).hexdigest()[:16]
    return out


def _proc_from_stacks(Fs, Fb, twtt, chain):
    def look(P):
        return ndimage.uniform_filter1d(P, N_LOOKS_SIM, axis=0,
                                        mode="nearest")
    return {"P": look(np.abs(Fs + Fb) ** 2), "Ps": look(np.abs(Fs) ** 2),
            "Pb": look(np.abs(Fb) ** 2), "twtt": twtt, "chain": chain,
            "Fs": Fs, "Fb": Fb}


def process_standard_cached(p, sim, out_dir, att, surf_rough,
                            force=False, antenna=None, bed_rough=None,
                            spec=None, gfix=None,
                            focus_aperture="alias_limited"):
    """Cache-first process_standard (module-section comment). Also stores
    the light per-trace arrays + scalars that let load_proc_pass rebuild a
    figure-ready (p_lite, sim_lite, proc) with NO scene prep or chunk
    replay."""
    cdir = Path(out_dir) / PROC_CACHE_DIRNAME
    cdir.mkdir(parents=True, exist_ok=True)
    surf_rough = resolve_surf_rough(surf_rough, p)
    rid = proc_rid(p)
    jp, npz_p = cdir / f"{rid}.json", cdir / f"{rid}.npz"
    meta = {"version": PROC_CACHE_VERSION, "pass": p["key"],
            "segment": p["segment"], "n_traces": int(len(p["idx"])),
            "n_looks": N_LOOKS_SIM, "posting_div": p.get("posting_div", 1),
            "focus_aperture": focus_aperture,
            "spacing_m": round(p["spacing"], 4),
            "att_db_per_km": att, "surf_rough": bool(surf_rough),
            "chunk_digests": chunk_digests(p, Path(out_dir) / "runs",
                                           sim["n_chunks"], att, surf_rough,
                                           antenna=antenna,
                                           bed_rough=bed_rough, spec=spec,
                                           gfix=gfix)}
    key = json.dumps(meta, sort_keys=True)
    if jp.exists() and npz_p.exists() and not force:
        doc = json.loads(jp.read_text())
        if doc.get("meta_key") == key:
            z = np.load(npz_p)
            print(f"  [proc-cache hit] {rid}", flush=True)
            return _proc_from_stacks(z["Fs"], z["Fb"], z["twtt"],
                                     doc["chain"])
        print(f"  [proc-cache STALE] {rid}: meta changed, rebuilding",
              flush=True)
    proc = process_standard(p, sim, focus_aperture=focus_aperture)
    np.savez(npz_p, Fs=proc["Fs"], Fb=proc["Fb"], twtt=proc["twtt"],
             nadir=sim["nadir"], s_sim=p["s_sim"], surf_sim=p["surf_sim"],
             bot_sim=np.asarray(p["bot_sim"], np.float64),
             s_m=p["s_m"], surf=p["surf"],
             bot=np.asarray(p["bot"], np.float64), tw_m=p["tw_m"])
    rc = p["rc_frame"]
    jp.write_text(json.dumps({
        "meta_key": key, "meta": meta, "chain": proc["chain"],
        "p_lite": {"key": p["key"], "segment": p["segment"],
                   "season": p.get("season", SEASON),
                   "parts": [[fid, list(sl)] for fid, sl in p["parts"]],
                   "rev": p["rev"], "synthetic": p["synthetic"],
                   "spacing": p["spacing"], "dt": p["dt"],
                   "posting_div": p.get("posting_div", 1),
                   "h_med": p["h_med"], "thick_med": p["thick_med"],
                   "reach_ct_m": p["reach"]["ct_m"],
                   "rc_frame": {"dt": rc.dt, "t0": rc.t0,
                                "n_samples": rc.n_samples, "f0": rc.f0}},
    }, indent=1) + "\n")
    print(f"  [proc-cache write] {rid} "
          f"({npz_p.stat().st_size / 1e6:.0f} MB)", flush=True)
    return proc


def load_proc_pass(key, out_dir, segment="full_line", picked_bed=False,
                   gamma_rssnr=True, proc_tag=True, dgn=True, hybrid=True):
    """SECONDS-fast figure path: (p_lite, sim_lite, proc) from the proc
    cache + the locally cached frames -- no scene prep, no chunk replay.
    The stored chunk fingerprints are re-derived from the chunk cache AS IT
    EXISTS NOW and must match, else returns None (caller falls back to the
    full path, which rebuilds the cache). p_lite carries exactly the
    fields analyze_pass and the trace figures consume; measured Data is
    re-sliced from the cached frames (synthetic passes skip it)."""
    cdir = Path(out_dir) / PROC_CACHE_DIRNAME
    rid = (f"{key}_{segment}"
           f"{case_tag(picked_bed, gamma_rssnr, proc_tag, dgn)}"
           + ("_hyb" if hybrid else ""))
    jp, npz_p = cdir / f"{rid}.json", cdir / f"{rid}.npz"
    if not (jp.exists() and npz_p.exists()):
        return None
    doc = json.loads(jp.read_text())
    stored = doc["meta"]["chunk_digests"]
    if (doc["meta"].get("version") != PROC_CACHE_VERSION
            or _digests_current(stored, Path(out_dir) / "runs") != stored):
        print(f"  [proc-cache STALE] {rid}: chunk cache changed", flush=True)
        return None
    z = np.load(npz_p)
    pl = doc["p_lite"]
    rc = pl["rc_frame"]
    fsub = None
    if not pl["synthetic"]:
        fsubs = []
        for fid, (a, b) in pl["parts"]:
            fs = load_frame(pl.get("season", SEASON),
                            fid).isel(slow_time=slice(a, b))
            if pl["rev"]:
                fs = fs.isel(slow_time=slice(None, None, -1))
            fsubs.append(fs)
        fsub = (fsubs[0] if len(fsubs) == 1
                else xr.concat(fsubs, dim="slow_time",
                               combine_attrs="override"))
    p_lite = {"key": pl["key"], "segment": pl["segment"],
              "season": pl.get("season", SEASON),
              "parts": [(fid, tuple(sl)) for fid, sl in pl["parts"]],
              "rev": pl["rev"], "synthetic": pl["synthetic"],
              "spacing": pl["spacing"], "dt": pl["dt"],
              "posting_div": pl["posting_div"], "h_med": pl["h_med"],
              "thick_med": pl["thick_med"],
              "reach": {"ct_m": pl["reach_ct_m"]},
              "rc_frame": RadarConfig(dt=rc["dt"], n_samples=rc["n_samples"],
                                      t0=rc["t0"], f0=rc["f0"]),
              "fsub": fsub, "tw_m": z["tw_m"], "surf": z["surf"],
              "bot": z["bot"], "s_m": z["s_m"], "s_sim": z["s_sim"],
              "surf_sim": z["surf_sim"], "bot_sim": z["bot_sim"]}
    sim_lite = {"twtt": z["twtt"], "nadir": z["nadir"]}
    proc = _proc_from_stacks(z["Fs"], z["Fb"], z["twtt"], doc["chain"])
    print(f"  [proc-cache load] {rid}", flush=True)
    return p_lite, sim_lite, proc


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


def synth_altitude_fsub(fsub, bot_sub, alt_m):
    """Rewrite a real pass's sliced frames as a SYNTHETIC smooth pass at
    constant ``alt_m`` ellipsoidal height: platform Elevation constant,
    Roll = 0 (smooth trajectory), surface twtt recomputed from the real
    surface ELEVATION (Elevation - c*Surface/2), and the bed pick's delay
    BELOW the surface preserved. Line geometry, picks and system params stay
    the real ones; the measured Data is geometrically meaningless for the
    synthetic pass and must not be compared. Returns (fsub, bot, note)."""
    surf = np.asarray(fsub.Surface.values, np.float64)
    elev = np.asarray(fsub.Elevation.values, np.float64)
    z_surf = elev - surf * C / 2.0
    new_surf = 2.0 * (alt_m - z_surf) / C
    if not (new_surf > 0).all():
        raise ValueError(f"synthetic altitude {alt_m} m is not above the "
                         "surface everywhere")
    new_bot = new_surf + (bot_sub - surf)
    fsub = fsub.assign(
        Elevation=xr.zeros_like(fsub.Elevation) + alt_m,
        Surface=xr.zeros_like(fsub.Surface) + new_surf,
        Roll=xr.zeros_like(fsub.Roll))
    note = {"synthetic_msl_m": alt_m,
            "convention": "constant ELLIPSOIDAL height (rac platform_z "
            "'msl' convention), roll = 0 smooth trajectory; surface/bed "
            "twtts recomputed from the real surface elevation and the real "
            "bed delay below surface; carrier pass: low (anchor line), "
            "shared 2016 system params (identical fc/B/window/dt across "
            "the triplet); measured Data untouched but MEANINGLESS here",
            "agl_med_m": round(float(np.nanmedian(alt_m - z_surf)), 0)}
    return fsub, new_bot, note


def upsample_fsub(fsub, bot, div):
    """T3: refine the along-track SIM grid by ``div`` (14.85 -> 7.43 m at
    div=2) so the alias-limited aperture (sin(theta) = lam/(4*ds)) doubles.

    Every per-trace GEOMETRY variable is linearly interpolated onto the
    refined index grid; the radargram ``Data`` is carried by nearest
    neighbour and is NEVER used on this grid (the measured side of every
    metric keeps the original frame). Endpoints are preserved exactly, so
    the refined traces land on the measured columns plus (div-1) new
    columns between each pair."""
    n = int(fsub.sizes["slow_time"])
    pos = np.linspace(0.0, n - 1.0, (n - 1) * div + 1)
    src = np.arange(n, dtype=np.float64)
    up = fsub.isel(slow_time=np.rint(pos).astype(int)).copy()
    for v in ("Latitude", "Longitude", "Elevation", "Roll", "Pitch",
              "Heading", "Surface", "GPS_time"):
        if v in up:
            up[v] = ("slow_time",
                     np.interp(pos, src, np.asarray(fsub[v].values,
                                                    np.float64)))
    return up, np.interp(pos, src, np.asarray(bot, np.float64))


def radar_grid(params, surf_tw, bed_tw, dt, t0f, oversample, window,
               antenna=None):
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
                          window=window,
                          construction=wf.get("pulse_compression_construction",
                                              "analytic"))
    ant = (AntennaConfig(kind="array", n_elements=rac.N_ELEMENTS,
                         spacing_lam=rac.SPACING_LAM, roll_source="nav")
           if antenna is None else
           # carry EVERY declared field (the old 4-field rebuild silently
           # dropped axis and would drop the tapered/finite-dipole params)
           AntennaConfig(**{k: v for k, v in antenna.model_dump().items()
                            if v is not None}))
    f0 = wf["center_frequency_Hz"]
    t0 = t0f + b0 * dt
    rc_sim = RadarConfig(dt=dt / oversample, n_samples=oversample * (nb - 1) + 1,
                         t0=t0, f0=f0, waveform=wave, antenna=ant)
    rc_frame = RadarConfig(dt=dt, n_samples=nb, t0=t0, f0=f0)
    return rc_sim, rc_frame, b0


def prep_pass(key, segment, n_traces, ref=None, gmap=None, axis=None,
              fine_posting=False, dgn_seed=None, posting_div=1,
              spec_diffuse=None, hybrid=False, instrument=None):
    """Slice (+reverse) the pass's frames onto the common window, derive the
    reach and grids, and build the base scene (REMA + BedMachine, cached).
    ``ref`` (ref_bed_picks) applies the picked-bed residual to that scene;
    ``gmap`` (build_rssnr_gamma, with ``axis`` = ref_bed_picks as the
    along-track axis) attaches the RSSNR-driven bed gamma grid.
    ``fine_posting`` (--processing standard) simulates EVERY measured trace
    (~14.85 m, the product posting: sim columns land on the measured
    columns). A ``synthetic_msl_m`` pass spec rewrites the geometry via
    synth_altitude_fsub."""
    spec = PASSES[key]
    season = pass_season(spec)
    fsubs, bots, tw_ref, parts = [], [], None, []
    for fid, (a, b) in spec[segment]:
        frame = load_frame(season, fid)
        # whole-frame parts ((None, None) from an omitted YAML slice) become
        # concrete HERE, where the frame length is known -- p['parts'] and
        # therefore chunk_meta stay byte-identical with existing caches
        a, b = (a or 0), (frame.sizes["slow_time"] if b is None else b)
        parts.append((fid, (a, b)))
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
        # Pre-2010 CSARP products carry no attitude at all: the scene then
        # gets nav_roll=None (opr.frame_scene), the kernel treats it as zero
        # roll, and there is nothing to negate.
        if "Roll" in fsub:
            fsub = fsub.assign(Roll=-fsub.Roll)
            roll_note = ("pass flown backwards: slices reversed and nav roll "
                         "NEGATED (roll rotates about the nav-order "
                         "along-track axis, which reversal flips)")
        else:
            roll_note = ("pass flown backwards: slices reversed. The product "
                         "carries NO Roll variable, so the antenna is "
                         "modelled unrolled (zero) and there is no roll sign "
                         "to flip")
    synth_note = None
    if spec.get("synthetic_msl_m"):
        fsub, bot_sub, synth_note = synth_altitude_fsub(
            fsub, bot_sub, spec["synthetic_msl_m"])
    if fine_posting:
        n_traces = int(fsub.sizes["slow_time"])
    # T3: simulate on a FINER along-track grid than the product posting. The
    # measured arrays (fsub/bot_sub/surf) are untouched -- only the SIM trace
    # grid is refined, so the measured side of every metric is unchanged.
    fsub_sim, bot_sim_full = fsub, bot_sub
    if posting_div > 1:
        if not fine_posting:
            raise ValueError("--posting-div refines the product-posting sim "
                             "grid: use it with --processing standard")
        fsub_sim, bot_sim_full = upsample_fsub(fsub, bot_sub, posting_div)
        n_traces = int(fsub_sim.sizes["slow_time"])

    # INSTRUMENT: the line pins the radar that actually flew this pass; an
    # experiment may swap it while keeping the geometry (that is the point of
    # the line/instrument split). A real instrument reads every simulated
    # parameter from the pass's OWN OPR frame, so the default path resolves
    # to exactly the values this tool has always used.
    inst_name = instrument or spec["instrument"]
    if inst_name not in INSTRUMENTS:
        raise ValueError(f"unknown instrument {inst_name!r}; have "
                         f"{sorted(INSTRUMENTS)}")
    inst = INSTRUMENTS[inst_name]
    params = rac.mcords_params(season, spec["param_frame"])
    wf, ant_cfg, inst_dev = inst.resolve(params["waveform"])
    params = {**params, "waveform": wf}
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
    # a pass may declare its own refinement but only FINER than the study
    # default (analysis.yaml compute.facet_spacing_scale; convergence
    # evidence 2026-08-21)
    spacing = rac.facet_spacing(lam, r_min, thick_med) * min(
        spec.get("facet_spacing_scale") or FACET_SPACING_SCALE,
        FACET_SPACING_SCALE)
    bed_fill = np.where(np.isfinite(bot_sub), bot_sub, np.nanmax(bot_sub))
    rc_sim, rc_frame, b0 = radar_grid(params, surf, bed_fill, dt, t0f,
                                      oversample, window, antenna=ant_cfg)

    base, aux = _retry(f"base_scene {key}",
                       lambda: rac.base_scene(fsub_sim, n_traces,
                                              reach["ct_m"]))
    # Picked bed: the fast-time grid, reach and facet spacing above stay at
    # their BedMachine values (derived from each pass's OWN picks) so the two
    # runs share one lattice and are directly comparable.
    if hybrid:
        if dgn_seed is None or axis is None:
            raise ValueError("the hybrid bed needs a DEMOGORGN seed AND the "
                             "anchor pick axis (--demogorgn-bed on a "
                             "crosses_gl segment)")
        aux["demogorgn"] = apply_hybrid_bed(base, fsub_sim, reach["ct_m"],
                                            dgn_seed, axis)
    else:
        aux["demogorgn"] = (apply_demogorgn_bed(base, fsub_sim,
                                                reach["ct_m"], dgn_seed)
                            if dgn_seed is not None else None)
    aux["picked_bed"] = apply_picked_bed(base, ref) if ref else None
    aux["rssnr_gamma"] = (apply_rssnr_gamma(base, axis or ref, gmap,
                                            spec_diffuse)
                          if gmap else None)
    idx = aux["idx"]
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

    def _s_of(fs):
        lat_, lon_ = rac._lonlat(fs)
        x_, y_ = tr.transform(lon_, lat_)
        return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x_),
                                                         np.diff(y_)))])

    s = _s_of(fsub)                                    # MEASURED trace axis
    # sim-trace views: identical to <measured>[idx] at posting_div == 1, and
    # the refined grid's own values when the sim grid is finer (T3).
    s_sim = (s[idx] if posting_div == 1 else _s_of(fsub_sim)[idx])
    surf_sim = (surf[idx] if posting_div == 1
                else np.asarray(fsub_sim.Surface.values, np.float64)[idx])
    bot_sim = bot_sub[idx] if posting_div == 1 else bot_sim_full[idx]
    return {"key": key, "segment": segment, "parts": parts,
            "season": season, "rev": spec["rev"],
            "roll_note": roll_note, "params": params, "window": window,
            "win_note": win_note, "fsub": fsub, "bot": bot_sub, "surf": surf,
            "dt": dt, "t0f": t0f, "oversample": oversample, "f_alias": f_alias,
            "lam": lam, "spacing": spacing, "reach": reach, "rc_sim": rc_sim,
            "rc_frame": rc_frame, "b0": b0, "base": base, "aux": aux,
            "idx": idx, "s_m": s, "agl": agl, "r_min": r_min,
            "s_sim": s_sim, "surf_sim": surf_sim, "bot_sim": bot_sim,
            "posting_div": posting_div,
            "picked_bed": bool(ref), "gamma_rssnr": bool(gmap),
            "proc": bool(fine_posting), "synthetic": synth_note,
            "dgn": dgn_seed is not None, "hybrid": bool(hybrid),
            "instrument": inst_name,
            "instrument_default": spec["instrument"],
            "instrument_provenance": inst.provenance_block(inst_dev),
            "h_med": float(np.nanmedian(agl)), "thick_med": thick_med,
            "tw_m": tw_ref}


# ========================================================================
# chunked simulation (pilot = 1 chunk; 50 km segment = ~5 identical chunks)
# ========================================================================
def chunk_rows(p):
    """Split the sim trace indices into along-track chunks (~CHUNK_M, or
    ~CHUNK_M_PROC at fine posting; the shorter DEM windows cut wasted facet
    work), then split further while traces x facets per interface would
    exceed CHUNK_TRACE_FACETS (kernel memory bound)."""
    s_sel = p["s_sim"]
    track = float(s_sel[-1] - s_sel[0])
    chunk_m = CHUNK_M_PROC if p.get("proc") else CHUNK_M
    n_chunks = max(1, int(round(track / chunk_m)))

    def split(n):
        edges = s_sel[0] + track * np.arange(1, n) / n
        which = np.searchsorted(edges, s_sel)
        return [np.where(which == c)[0] for c in range(n)]

    while True:   # memory guard (CHUNK_TRACE_FACETS)
        chunks = split(n_chunks)
        pairs = max(len(r) * _chunk_facets_estimate(p, r) for r in chunks)
        if pairs <= CHUNK_TRACE_FACETS or n_chunks >= len(s_sel):
            return chunks
        n_chunks += 1


def _chunk_facets_estimate(p, rows):
    """Facets per interface the chunk_scene crop will hold for ``rows``:
    the traces' bbox padded by ct + 100 m, in DEM cells (0 when the pass has
    no reach/base yet, e.g. in unit tests without a scene)."""
    base, reach = p.get("base"), p.get("reach")
    if base is None or reach is None:
        return 0
    tr = Transformer.from_crs("EPSG:4326", base.crs, always_xy=True)
    nav = base.nav_llh[rows]
    px, py = tr.transform(nav[:, 1], nav[:, 0])
    pad = reach["ct_m"] + 100.0
    cell = 0.5 * (abs(base.transform.a) + abs(base.transform.e))
    ncol = (px.max() - px.min() + 2 * pad) / cell + 2
    nrow = (py.max() - py.min() + 2 * pad) / cell + 2
    return int(ncol * nrow)


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
    origin = tuple(np.asarray(getattr(base, "grid_origin", (0, 0)))
                   + (r0, c0))
    sc = MultilayerScene(f"{base.name}_r{rows[0]}", dems,
                         base.transform * Affine.translation(c0, r0),
                         base.crs, nav, base.media, dict(base.params),
                         grid_origin=origin)
    roll = getattr(base, "nav_roll", None)
    sc.nav_roll = None if roll is None else np.asarray(roll)[rows]
    if gamma:
        sc.gamma_maps = {"bed": (
            np.ascontiguousarray(base.gamma_bed[r0:r1, c0:c1]),
            sc.transform, sc.crs)}
        if getattr(base, "diffuse_bed", None) is not None:
            sc.diffuse_maps = {"bed": (
                np.ascontiguousarray(base.diffuse_bed[r0:r1, c0:c1]),
                sc.transform, sc.crs)}
    return sc


SURF_ROUGH_FIXTURE = (rac.SURF_ROUGH_SIGMA_M, rac.SURF_ROUGH_CL_M)


def surf_rough_pair(surf_rough):
    """Normalise a surface-roughness setting to None (smooth), a
    (sigma_m, corr_length_m) Gaussian pair, or a (sigma_m, corr_length_m,
    acf) triple for a non-Gaussian ACF. True = the C&S fixture."""
    if not surf_rough:
        return None
    if surf_rough is True:
        return SURF_ROUGH_FIXTURE
    if isinstance(surf_rough, dict):
        raise ValueError("surface roughness source not resolved for this "
                         "pass (call resolve_surf_rough first)")
    pair = (float(surf_rough[0]), float(surf_rough[1]))
    acf = surf_rough[2] if len(surf_rough) > 2 else "gaussian"
    return pair if acf == "gaussian" else pair + (str(acf),)


def surf_rough_acf(pair):
    return "gaussian" if pair is None or len(pair) == 2 else pair[2]


def surf_rough_is_fixture(pair):
    return (pair is not None and surf_rough_acf(pair) == "gaussian"
            and all(abs(a - b) < 1e-9 for a, b in
                    zip(pair[:2], SURF_ROUGH_FIXTURE)))


def resolve_surf_rough(surf_rough, p, info=False):
    """Per-pass surface roughness: bool / pair / (sigma, l, acf) pass
    through; a {source: atm_b1} dict resolves the effective Gaussian pair
    from config/roughness/atm_b1.yaml for the line, the pass and its carrier
    (tools/surface_roughness_b1.py); {source: atm_exponential} takes the
    table's exponential-ACF entry directly as (sigma, l, 'exponential').
    ``info`` also returns the provenance dict."""
    if isinstance(surf_rough, dict):
        if surf_rough.get("source") not in ("atm_b1", "atm_exponential"):
            raise ValueError(f"unknown surface roughness source "
                             f"{surf_rough.get('source')!r}")
        import surface_roughness_b1 as b1
        theta = surf_rough.get("theta_c_deg")
        tab = b1.load_table()
        if surf_rough["source"] == "atm_exponential":
            sig, l, inf = b1.resolve_exponential(LINE, p["key"], tab)
            pair = (round(sig, 6), round(l, 4), "exponential")
            return (pair, inf) if info else pair
        sig, l, inf = b1.resolve(LINE, p["key"], p["rc_sim"].f0,
                                 tab["theta_c_deg"] if theta is None
                                 else theta, tab)
        pair = (round(sig, 6), round(l, 4))
        return (pair, inf) if info else pair
    pair = surf_rough_pair(surf_rough)
    return (pair, None) if info else pair


def sim_cfg(rc_sim, spacing, att, surf_rough, antenna=ANT_DEFAULT,
            bed_rough=None, diffuse_exponent=1.0, gfix=None):
    """``antenna``: 'array' (the MCoRDS-like default) or 'isotropic' (the
    T4 pattern-sensitivity bound). ``bed_rough`` = (sigma_m, corr_length_m)
    attaches Gerekos sub-facet roughness to the BED interface (T1).
    ``gfix``: None (legacy) or the grazing-fix taper s_eff
    (GrazingFixConfig)."""
    srp = surf_rough_pair(surf_rough)
    rcg = (RoughnessConfig(sigma_m=srp[0], corr_length_m=srp[1],
                           acf=surf_rough_acf(srp))
           if srp else None)
    rcb = (RoughnessConfig(sigma_m=bed_rough[0], corr_length_m=bed_rough[1])
           if bed_rough else None)
    ant = rc_sim.antenna
    if antenna == "isotropic":
        ant = AntennaConfig(kind="isotropic")
    elif antenna == "array8":
        # more-directive BRACKET: the same 0.5-lambda cross-track array with
        # 8 elements. NOT a claim about the real antenna -- the physical
        # element pattern (dipoles over structure) always makes the true
        # pattern MORE directive than the bare array factor the baseline
        # uses, and this brackets that direction the way 'isotropic' brackets
        # the other. The bracket is MILD: against the baseline's
        # rac.N_ELEMENTS the aperture ratio is (8-1)/(N-1), i.e. ~1.17x at
        # N = 7 -- earlier comments claimed 1.6x by computing against a
        # 5-element baseline this tool has never used.
        ant = AntennaConfig(kind="array", n_elements=8,
                            spacing_lam=rac.SPACING_LAM, roll_source="nav")
    return SimConfig(
        mode="coherent", split_sides=False,
        grazing_fix=(None if gfix is None else GrazingFixConfig(s_eff=gfix)),
        diffuse_exponent=diffuse_exponent,
        radar=rc_sim.model_copy(update={"antenna": ant}),
        facets=FacetConfig(spacing=spacing),
        media=[Medium(name="air", eps_r=1.0),
               Medium(name="ice", eps_r=rac.EPS_ICE,
                      attenuation_db_per_km=att),
               Medium(name="bed", eps_r=rac.EPS_BED)],
        interfaces=[DemInterface(name="surface", roughness=rcg),
                    DemInterface(name="bed", roughness=rcb)])


# The exact resolved instrument-antenna states every PRE-EXISTING chunk cache
# was built under when the CLI antenna is the default ('array' = "use the
# instrument YAML"): the david isotropic placeholders, the uniform MCoRDS
# 7-element array (all four mcords YAMLs), and the HAPS 8-element/no-roll
# array. These states contribute NO cache key (byte-identical rid + meta with
# every existing cache); any OTHER resolved antenna -- the realistic david
# models, or a future edit to an array YAML's numbers -- is fingerprinted
# into both the rid and the meta, so a YAML pattern change can never silently
# reuse a stale chunk. CLI antenna overrides ('isotropic'/'array8') bypass
# the instrument antenna entirely and stay keyed by name as before.
_LEGACY_INSTRUMENT_ANTS = (
    AntennaConfig(kind="isotropic"),
    AntennaConfig(kind="array", n_elements=7, spacing_lam=0.5,
                  roll_source="nav"),
    AntennaConfig(kind="array", n_elements=8, spacing_lam=0.5,
                  roll_source="none"),
)


def inst_ant_meta(p, antenna):
    """{'instrument_antenna': resolved params} or {} (legacy/overridden)."""
    rc = p.get("rc_sim")
    if antenna != ANT_DEFAULT or rc is None:
        return {}          # CLI override active: instrument antenna unused
    a = rc.antenna
    if any(a == la for la in _LEGACY_INSTRUMENT_ANTS):
        return {}
    fp = {"kind": a.kind, "roll_source": a.roll_source}
    if a.kind in ("dipole", "finite_dipole"):
        fp["axis"] = a.axis
    if a.kind == "finite_dipole":
        fp["length_lam"] = a.length_lam
    if a.kind == "array":
        fp.update(n_elements=a.n_elements, spacing_lam=a.spacing_lam)
    if a.kind == "array_tapered":
        fp.update(spacing_lam=a.spacing_lam,
                  tx_weights=list(a.tx_weights),
                  rx_weights=list(a.rx_weights))
    if a.kind in ("array", "array_tapered") \
            and a.element_directivity_db is not None:
        fp["element_directivity_db"] = a.element_directivity_db
        fp["element_pattern"] = "forward_cosine_power_integrated_directivity"
    if a.kind == "tabulated":
        fp.update(theta_deg=list(a.theta_deg), gain=list(a.gain))
    return {"instrument_antenna": fp}


def inst_ant_tag(p, antenna):
    """Chunk-file-name suffix for a fingerprinted instrument antenna: new
    patterns get NEW file names, so legacy chunk files (isotropic-placeholder
    baselines) survive on disk instead of being overwritten."""
    fpm = inst_ant_meta(p, antenna)
    if not fpm:
        return ""
    h = hashlib.sha1(json.dumps(fpm["instrument_antenna"],
                                sort_keys=True).encode()).hexdigest()[:8]
    return f"_ia{h}"


def wave_meta(p):
    """Chunk cache-key entry for a non-default compressed-pulse construction.
    The waveform convolution runs INSIDE the cached chunk (simulate.py ->
    apply_waveform), so the construction and the pulse length it now
    depends on must fork the key -- but only when construction != analytic,
    so every pre-existing key string stays byte-identical."""
    wv = p["rc_sim"].waveform
    if wv.construction == "analytic":
        return {}
    return {"waveform": {"construction": wv.construction,
                         "pulse_length_us": round(wv.pulse_length * 1e6, 4)}}


def surf_rough_tag(surf_rough):
    """Rid suffix for a NON-fixture surface roughness pair; the fixture
    (and the boolean) keep the pre-existing ``_srough`` names."""
    pair = surf_rough_pair(surf_rough)
    if pair is None or surf_rough_is_fixture(pair):
        return ""
    return (f"_sr{pair[0]:.4g}_{pair[1]:.4g}"
            + ("" if surf_rough_acf(pair) == "gaussian"
               else f"_{surf_rough_acf(pair)[:3]}"))


def chunk_rid(p, ci, att, surf_rough, antenna=ANT_DEFAULT, bed_rough=None,
              spec=None, gfix=None):
    """Cache file name for one chunk. Non-default hypothesis knobs append a
    suffix; the default case keeps the pre-campaign names (cache reuse)."""
    return (f"{p['key']}_{p['segment']}"
            f"{case_tag(p['picked_bed'], p['gamma_rssnr'], p['proc'], p['dgn'])}"
            f"_c{ci:02d}"
            + ("_hyb" if p.get("hybrid") else "")
            + ("_srough" if surf_rough else "")
            + surf_rough_tag(surf_rough)
            + (f"_att{att:g}" if att != rac.ATT_DB_PER_KM else "")
            + ("" if antenna == ANT_DEFAULT else f"_ant{antenna}")
            + ("" if not bed_rough
               else f"_brough{bed_rough[0]:g}_{bed_rough[1]:g}")
            + ("" if p.get("posting_div", 1) == 1
               else f"_pdiv{p['posting_div']:d}")
            + ("" if not spec
               else f"_fs{spec[0]:g}_s0{spec[1]:g}_n{spec[2]:g}")
            + ("" if gfix is None else f"_gfx{gfix:g}")
            + ("" if p.get("instrument") in (None,
                                             p.get("instrument_default"))
               else f"_i{p['instrument']}")
            + inst_ant_tag(p, antenna)
            + ("" if not wave_meta(p)
               else f"_w{p['rc_sim'].waveform.construction}"))


def chunk_meta(p, ci, rows, n_chunks, n, att, surf_rough,
               antenna=ANT_DEFAULT, bed_rough=None, spec=None, gfix=None):
    """run_level cache key for one chunk. Optional features (gamma,
    DEMOGORGN, and the hypothesis knobs) contribute keys ONLY when they are
    ON, so every pre-existing cache stays valid byte-for-byte."""
    return {"season": p.get("season", SEASON), "pass": p["key"],
            "segment": p["segment"],
            "picked_bed": p["picked_bed"],
            **({"gamma_rssnr": True, "rssnr_snapshot": RSSNR_SNAPSHOT,
                "rssnr_k_db": p["aux"]["rssnr_gamma"]["k_db"]}
               if p["gamma_rssnr"] else {}),
            **({"demogorgn_seed": p["aux"]["demogorgn"]["seed_id"],
                "demogorgn_snapshot": p["aux"]["demogorgn"]["snapshot_id"]}
               if p["dgn"] else {}),
            **({"hybrid_bed": {"gl_s_km": GL_S_KM, "ramp_km": GL_RAMP_KM,
                               "floating": "low-pass picks, NN in anchor s, "
                                           "constant cross-track"}}
               if p.get("hybrid") else {}),
            "parts": [[fid, list(sl)] for fid, sl in p["parts"]],
            "reversed": p["rev"], "chunk": ci, "n_chunks": n_chunks,
            "rows": [int(rows[0]), int(rows[-1])], "n_traces_total": n,
            "spacing_m": round(p["spacing"], 4),
            "ct_m": round(p["reach"]["ct_m"], 1), "att_db_per_km": att,
            **({} if antenna == ANT_DEFAULT else {"antenna": antenna}),
            **({} if not bed_rough
               else {"bed_rough": [float(bed_rough[0]), float(bed_rough[1])]}),
            **({} if p.get("posting_div", 1) == 1
               else {"posting_div": int(p["posting_div"])}),
            **({} if not spec else {"spec_diffuse": [float(v)
                                                     for v in spec]}),
            **({} if gfix is None
               else {"grazing_fix": {"s_eff": float(gfix)}}),
            **({} if p.get("instrument") in (None,
                                             p.get("instrument_default"))
               else {"instrument": p["instrument"]}),
            **inst_ant_meta(p, antenna),
            **wave_meta(p),
            "window": p["window"], "surf_rough": bool(surf_rough),
            **({} if not surf_rough_tag(surf_rough)
               else {"surf_rough_sigma_l":
                     list(surf_rough_pair(surf_rough)[:2])}),
            **({} if surf_rough_acf(surf_rough_pair(surf_rough)) == "gaussian"
               else {"surf_rough_acf":
                     surf_rough_acf(surf_rough_pair(surf_rough))}),
            "kernel": KERNEL_VERSION,
            "dt_sim_ns": round(p["rc_sim"].dt * 1e9, 5),
            "t0_us": round(p["rc_sim"].t0 * 1e6, 5),
            "n_samples_sim": p["rc_sim"].n_samples}


def simulate_pass(p, runs_dir, att, surf_rough, force, antenna=ANT_DEFAULT,
                  bed_rough=None, spec=None, gfix=None):
    """Chunked cached coherent surface+bed runs; assembled per-layer fields.
    Returns dict(field (T,nb,2), twtt, nadir (T,2), wall_s, facets, ...)."""
    chunks = chunk_rows(p)
    surf_rough = resolve_surf_rough(surf_rough, p)
    if surf_rough_tag(surf_rough):
        print(f"  surface roughness: sigma {surf_rough[0] * 100:.2f} cm, "
              f"l {surf_rough[1]:.3f} m (non-fixture; chunk rid"
              f"{surf_rough_tag(surf_rough)})", flush=True)
    cfg = sim_cfg(p["rc_sim"], p["spacing"], att, surf_rough, antenna,
                  bed_rough, diffuse_exponent=spec[2] if spec else 1.0,
                  gfix=gfix)
    n = len(p["idx"])
    field = twtt = nadir = None
    wall, facets, dropped = 0.0, [], []
    for ci, rows in enumerate(chunks):
        scene = chunk_scene(p["base"], rows, p["reach"]["ct_m"],
                            gamma=p["gamma_rssnr"])
        rid = chunk_rid(p, ci, att, surf_rough, antenna, bed_rough, spec,
                        gfix)
        meta = chunk_meta(p, ci, rows, len(chunks), n, att, surf_rough,
                          antenna, bed_rough, spec, gfix)
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


def floor_window(tw_m, bot):
    """(lo, hi, doc): absolute-twtt measured noise-floor window for ONE pass.

    Default is the tool's established record end -[12, 8] us. If the pass's
    deepest bed pick reaches into it, the window is slid to start at
    (deepest bed + FLOOR_BED_GUARD_US) and to stop FLOOR_ROLLOFF_US before
    the record end. ``doc["valid"]`` is False when what is left is narrower
    than FLOOR_MIN_WIDTH_US -- the floor (and everything derived from it:
    noise_limited, the tail floor caveat) must then be reported as unknown
    rather than as a number.
    """
    t_end = float(tw_m[-1])
    lo = t_end - FLOOR_TAIL_LO_US * 1e-6
    hi = t_end - FLOOR_TAIL_HI_US * 1e-6
    b_max = float(np.nanmax(bot))
    clear = b_max + FLOOR_BED_GUARD_US * 1e-6
    slid = lo < clear
    if slid:
        lo, hi = clear, t_end - FLOOR_ROLLOFF_US * 1e-6
    width = hi - lo
    doc = {"window_us": [round(lo * 1e6, 3), round(hi * 1e6, 3)],
           "width_us": round(width * 1e6, 3),
           "record_end_us": round(t_end * 1e6, 3),
           "deepest_bed_us": round(b_max * 1e6, 3),
           "margin_past_deepest_bed_us": round((lo - b_max) * 1e6, 3),
           "slid_off_the_bed": bool(slid),
           "valid": bool(width >= FLOOR_MIN_WIDTH_US * 1e-6),
           "note": ("measured noise floor. Default record end -[12, 8] us; "
                    "SLID to (deepest bed + "
                    f"{FLOOR_BED_GUARD_US:g} us) .. (record end - "
                    f"{FLOOR_ROLLOFF_US:g} us) when the default would land "
                    "on the bed (short post-bed tail). An UPPER bound: it "
                    "may still hold a few dB of clutter tail")}
    return lo, hi, doc


def _med_db_rel(num, den):
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0) & (num > 0)
    if not ok.any():
        return float("nan")
    return float(np.median(10.0 * np.log10(num[ok] / den[ok])))


def nadir_bed_offset(p, sim):
    """Median offset of the SIMULATED nadir bed vs the pass's own radar
    pick (us and in-ice meters) -- reported, not tuned away (the DEMOGORGN
    thickness-convention misfit shows up here)."""
    d = sim["nadir"][:, 1] - p["bot_sim"]
    med = float(np.nanmedian(d))
    return {"med_us": round(med * 1e6, 3),
            "med_m_ice": round(med * C / (2.0 * np.sqrt(rac.EPS_ICE)), 1),
            "note": "sim nadir bed twtt minus the pass's own Bottom pick; "
            "the per-pass registration gate aligns the SURFACE only and "
            "cannot absorb a bed offset"}


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


def rel_mean_profile(P, twtt, dt, t_ref, norm, lo_us=None, hi_us=None):
    """(rel_us, dB): mean power vs twtt below each trace's own reference
    time, each trace normalized by ``norm`` (its own surface peak), integer
    bin shifts (every grid here shares the 20.202 ns lattice).

    Defaults to the ACTIVE LINE's PROFILE_REL_US: this is the data extent,
    so a line whose bed sits deeper than the default simply would not have
    its bed in the profile at all."""
    if lo_us is None:
        lo_us = PROFILE_REL_US[0]
    if hi_us is None:
        hi_us = PROFILE_REL_US[1]
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


# ------------------------------------------------------------------------
# bed-return tail: how fast the power decays AFTER the bed echo
# ------------------------------------------------------------------------
# Complements (does not duplicate) the existing bed-WINDOW level in
# clutter_{key} / *_bed_ablation_{key} ("bed_rel_surf_db", a single mean over
# bed-0.5 -> bed+1.5 us): this measures the SHAPE of the decay past the bed,
# which is where the sim and the measurement visibly diverge.
def _at_us(rel_us, db, t):
    """Profile value (dB) at delay ``t`` us past the bed reference."""
    ok = np.isfinite(db)
    if ok.sum() < 2:
        return float("nan")
    return float(np.interp(t, rel_us[ok], db[ok]))


def tail_slope_db_per_us(rel_us, db, lo=None, hi=None):
    """Robust (Theil-Sen) slope of a mean-power profile (dB) vs delay (us)
    over the fit window. Negative = a decaying tail; ~0 = a flat pedestal.
    Theil-Sen (not least squares) so a single bright arc crossing the window
    cannot set the slope."""
    lo = TAIL_FIT_US[0] if lo is None else lo
    hi = TAIL_FIT_US[1] if hi is None else hi
    m = np.isfinite(db) & (rel_us >= lo) & (rel_us <= hi)
    if m.sum() < 3:
        return float("nan")
    return float(stats.theilslopes(db[m], rel_us[m])[0])


def tail_slope_db_per_deg(p, rel_us, db, lo=None, hi=None):
    """Same robust fit against the REFRACTED off-nadir bed incidence angle
    (deg) instead of delay -- the physically interpretable angular-backscatter
    view. Uses the pass's median geometry (median AGL, median ice thickness);
    the delay->angle map is nonlinear, so this is a window-average slope."""
    lo = TAIL_FIT_US[0] if lo is None else lo
    hi = TAIL_FIT_US[1] if hi is None else hi
    m = np.isfinite(db) & (rel_us >= lo) & (rel_us <= hi)
    if m.sum() < 3:
        return float("nan")
    ang = bed_incidence_deg(p["h_med"], p["thick_med"],
                            float(np.sqrt(rac.EPS_ICE)), rel_us[m] * 1e-6)
    return float(stats.theilslopes(db[m], ang)[0])


def tail_angle_map(p, delays_us=(0.5, 1.0, 2.0, 3.0, 3.5)):
    """Post-bed delay -> refracted bed incidence angle (deg) for this pass."""
    n_ice = float(np.sqrt(rac.EPS_ICE))
    return {f"+{t:g}us": round(float(bed_incidence_deg(
        p["h_med"], p["thick_med"], n_ice, t * 1e-6)), 2) for t in delays_us}


def sim_tail_stats(p, a):
    """Bed-return tail numbers for ONE simulated pass/bed source, from its
    bed-referenced ensemble mean-power profiles (dB rel own surface peak).
    Includes the fair-comparison guard: the sim SURFACE-return curve must sit
    >= TAIL_GUARD_DB below the sim BED-return curve everywhere in the fit
    window, else the total-field tail is surface clutter, not bed returns."""
    rel, tot = a["bed_profs"]["sim_total"]
    sur, bed = a["bed_profs"]["sim_surface"][1], a["bed_profs"]["sim_bed"][1]
    lo, hi = TAIL_FIT_US
    m = (rel >= lo) & (rel <= hi)
    marg = bed[m] - sur[m]
    j = int(np.argmin(marg))
    ok = bool(marg[j] >= TAIL_GUARD_DB)
    return {
        "slope_db_per_us": round(tail_slope_db_per_us(rel, tot), 3),
        "bed_returns_slope_db_per_us": round(tail_slope_db_per_us(rel, bed), 3),
        "slope_db_per_deg": round(tail_slope_db_per_deg(p, rel, tot), 3),
        "level_rel_surf_db": {f"+{t:g}us": round(_at_us(rel, tot, t), 2)
                              for t in TAIL_EXCESS_US},
        "bed_returns_level_rel_surf_db": {
            f"+{t:g}us": round(_at_us(rel, bed, t), 2) for t in TAIL_EXCESS_US},
        "guard": {"min_bed_minus_surface_returns_db": round(float(marg[j]), 2),
                  "at_us": round(float(rel[m][j]), 2),
                  "threshold_db": TAIL_GUARD_DB, "pass": ok,
                  "note": "sim bed returns minus sim surface returns "
                  "(per-interface decomposition), minimum over the fit "
                  "window; a FAIL means the total-field tail there is "
                  "surface-return clutter and the total-field slope/excess "
                  "must be read as an upper bound (use the bed-returns-only "
                  "slope instead)"},
        "record_coverage_frac": a["tail_cov"]["sim"]}


def meas_tail_stats(p, a):
    """Measured bed-return tail + the noise-floor caveat: is the measured
    decay genuinely bed returns at bed+3 us, or is it floor-limited there?"""
    rel, db = a["bed_profs"]["measured"]
    lev = {f"+{t:g}us": round(_at_us(rel, db, t), 2) for t in TAIL_EXCESS_US}
    floor = a["floor_db"]
    marg = (None if floor is None
            else round(lev[f"+{TAIL_EXCESS_US[-1]:g}us"] - floor, 2))
    return {"slope_db_per_us": round(tail_slope_db_per_us(rel, db), 3),
            "slope_db_per_deg": round(tail_slope_db_per_deg(p, rel, db), 3),
            "level_rel_surf_db": lev,
            "record_coverage_frac": a["tail_cov"]["measured"],
            "noise_floor_caveat": {
                "floor_rel_surf_db": floor,
                f"tail_minus_floor_at_+{TAIL_EXCESS_US[-1]:g}us_db": marg,
                "floor_limited": (None if marg is None
                                  else bool(marg < TAIL_FLOOR_MARGIN_DB)),
                "margin_threshold_db": TAIL_FLOOR_MARGIN_DB,
                "window": a["floor_doc"],
                "note": "measured floor from the PER-PASS floor window "
                "(see 'window': default record end -12..-8 us, slid clear "
                "of the deepest bed pick when the post-bed tail is short). "
                "An UPPER bound -- it may still hold a few dB of clutter "
                "tail. A small margin means the measured tail is "
                "floor-limited there and the sim-minus-measured excess is a "
                "LOWER bound on the real gap; a null floor means the record "
                "leaves no trustworthy floor window at all"}}


def bed_tail_entry(key, p, a, sources):
    """Assemble the bed-return-tail metric for one pass. ``sources`` =
    [(slug, analysis)] bed-source variants (the first is the headline).
    All curves are trace-ensemble mean power in dB rel each trace's own
    surface peak, referenced to that dataset's OWN bed reference."""
    sim = {slug: sim_tail_stats(p, a_s) for slug, a_s in sources}
    meas = meas_tail_stats(p, a) if a["bed_profs"].get("measured") else None
    slopes = {"measured": meas["slope_db_per_us"] if meas else None,
              **{slug: sim[slug]["slope_db_per_us"] for slug in sim}}
    slopes_deg = {"measured": meas["slope_db_per_deg"] if meas else None,
                  **{slug: sim[slug]["slope_db_per_deg"] for slug in sim}}
    excess = None
    if meas is not None:
        excess = {slug: {t: round(sim[slug]["level_rel_surf_db"][t]
                                  - meas["level_rel_surf_db"][t], 2)
                         for t in meas["level_rel_surf_db"]} for slug in sim}
    head = sources[0][0]
    value = (excess[head][f"+{TAIL_EXCESS_US[1]:g}us"] if excess
             else sim[head]["slope_db_per_us"])
    return {
        "value": value, "threshold": None, "op": "record", "pass": True,
        "bed_return_tail_slope_db_per_us": slopes,
        "bed_return_tail_slope_db_per_deg": slopes_deg,
        "bed_return_tail_excess_db": excess,
        "sim": sim, "measured": meas,
        "bed_return_angle_map_deg": tail_angle_map(p),
        "agl_med_m": round(p["h_med"], 0),
        "fit_window_us": list(TAIL_FIT_US),
        "excess_delays_us": list(TAIL_EXCESS_US),
        "note": "KEY DELIVERABLE (bed-return tail): robust Theil-Sen slope "
        "of the trace-ensemble mean power (dB rel own surface peak) vs "
        f"delay over bed+{TAIL_FIT_US[0]:g} -> bed+{TAIL_FIT_US[1]:g} us, "
        "each trace referenced to its OWN bed (measured: its Bottom pick; "
        "sim: the sim bed-layer nadir twtt), plus sim-minus-measured excess "
        "at bed+1/2/3 us. Negative slope = decaying tail. 'guard' is the "
        "fair-comparison check that the sim tail is bed returns and not "
        "surface returns; 'noise_floor_caveat' asks whether the MEASURED "
        "tail is floor-limited. The *_db_per_deg variants refit the same "
        "window against the REFRACTED off-nadir bed incidence angle "
        "(bed_return_angle_map_deg): the same post-bed delay probes very "
        "different bed angles at 0.4 vs 10 km AGL, so the angular slopes -- "
        "not the delay slopes -- are what compare across passes. "
        "Complements the single bed-window level in "
        f"clutter_{key} (bed-0.5 -> bed+1.5 us mean) -- this is the decay "
        "SHAPE past the bed, not the window level. recorded only"}


def analyze_pass(p, sim, proc=None, trace_s_km=None):
    """Per-pass sim-vs-measured clutter metrics + per-interface (surface- vs
    bed returns) decomposition + profiles for the figures. ``proc``
    (process_standard output) analyzes the PROCESSED powers on the same
    lattice; a synthetic pass (p['synthetic']) skips every measured-side
    quantity (no measured data exists at that geometry). ``trace_s_km``
    (anchor along-track km) additionally records the SINGLE-TRACE variant of
    the decomposition at the nearest trace -- same curves, one slow-time
    location instead of the ensemble average."""
    tw, dtf = sim["twtt"], p["rc_frame"].dt
    if proc is not None:
        P, Ps, Pb = proc["P"], proc["Ps"], proc["Pb"]
    else:
        F = sim["field"]
        P = np.abs(F.sum(-1)) ** 2
        Ps, Pb = np.abs(F[..., 0]) ** 2, np.abs(F[..., 1]) ** 2
    surf_pick = p["surf_sim"]

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
    verdict = ("surface returns" if dmid > 3.0 else
               "bed returns" if dmid < -3.0 else "mixed")

    profs = {
        "sim_total": rel_mean_profile(P, tw, dtf, t_s, spk),
        "sim_surface": rel_mean_profile(Ps, tw, dtf, t_s, spk),
        "sim_bed": rel_mean_profile(Pb, tw, dtf, t_s, spk),
    }
    # BED-referenced profiles (the tail metrics): sims on the SIM bed-layer
    # nadir twtt, measured on its own Bottom pick. record_coverage = fraction
    # of traces whose record actually reaches the end of the fit window (the
    # fast-time window is anchored on the DEEPEST bed + POST_BED_US).
    bed_profs = {
        "sim_total": rel_mean_profile(P, tw, dtf, t_b, spk, *TAIL_PROF_US),
        "sim_surface": rel_mean_profile(Ps, tw, dtf, t_b, spk, *TAIL_PROF_US),
        "sim_bed": rel_mean_profile(Pb, tw, dtf, t_b, spk, *TAIL_PROF_US),
    }
    tail_cov = {"sim": round(float(np.mean(
        tw[-1] - t_b >= TAIL_FIT_US[1] * 1e-6)), 3), "measured": None}
    clean = {k: round(v, 2) for k, v in m_sim.items() if not k.startswith("_")}

    def _prof_db(m):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = 10.0 * np.log10(m["_bed"] / m["_spk"])
        return np.where(np.isfinite(r), r, np.nan)

    if p.get("synthetic"):
        # no measured data exists at the synthetic geometry
        meas = m_meas = cleanm = floor_db = noise_limited = None
        meas_prof = None
    else:
        # measured: ALL traces of the segment, windows on its OWN picks
        meas = np.asarray(p["fsub"].Data.values, np.float64)
        tw_m, dt_m = p["tw_m"], p["dt"]
        m_meas = clutter_metrics(meas, tw_m, dt_m, p["surf"], p["bot"])
        n_m = meas.shape[0]
        f_lo, f_hi, floor_doc = floor_window(tw_m, p["bot"])
        floor = _wmean(meas, tw_m, dt_m, np.full(n_m, f_lo),
                       np.full(n_m, f_hi))
        floor_db = (round(_med_db_rel(floor, m_meas["_spk"]), 2)
                    if floor_doc["valid"] else None)
        noise_limited = (None if floor_db is None else
                         bool(m_meas["midcol_rel_surf_db"] - floor_db < 3.0))
        profs["measured"] = rel_mean_profile(meas, tw_m, dt_m, p["surf"],
                                             m_meas["_spk"])
        bed_profs["measured"] = rel_mean_profile(
            meas, tw_m, dt_m, p["bot"], m_meas["_spk"], *TAIL_PROF_US)
        tail_cov["measured"] = round(float(np.mean(
            tw_m[-1] - p["bot"] >= TAIL_FIT_US[1] * 1e-6)), 3)
        cleanm = {k: round(v, 2) for k, v in m_meas.items()
                  if not k.startswith("_")}
        meas_prof = _prof_db(m_meas)

    # ---- single-trace decomposition (same curves, ONE slow-time location
    # each; ``trace_s_km`` may be a scalar or a list -- the full_line
    # segment records a grounded AND a floating location)
    tprofs_l, tinfo_l = [], []
    for ts1 in ([] if trace_s_km is None else np.atleast_1d(trace_s_km)):
        ts1 = float(ts1)
        s0 = S0_KM[p["segment"]]
        i = int(np.argmin(np.abs(s0 + p["s_sim"] / 1e3 - ts1)))
        sl = slice(i, i + 1)
        tprofs = {k: rel_mean_profile(A[sl], tw, dtf, t_s[sl], spk[sl])
                  for k, A in (("sim_total", P), ("sim_surface", Ps),
                               ("sim_bed", Pb))}
        gb = _wmean(Pb[sl], tw, dtf, t_b[sl] - BED_LO_US * 1e-6,
                    t_b[sl] + BED_HI_US * 1e-6)
        gs = _wmean(Ps[sl], tw, dtf, t_b[sl] - BED_LO_US * 1e-6,
                    t_b[sl] + BED_HI_US * 1e-6)
        tinfo = {"requested_s_km": round(ts1, 3),
                 "sim_trace_index": i,
                 "sim_s_km": round(float(s0 + p["s_sim"][i] / 1e3), 3),
                 "agl_m": round(float(p["surf_sim"][i] * C / 2.0), 0),
                 "bed_below_surface_us": round(float(
                     (t_b[i] - t_s[i]) * 1e6), 2),
                 "bed_window_bed_minus_surface_returns_db": round(float(
                     10.0 * np.log10(max(float(gb[0]), 1e-300)
                                     / max(float(gs[0]), 1e-300))), 2),
                 "note": "single-trace decomposition location; the guard is "
                 "this trace's own bed-window sim bed returns minus sim "
                 "surface returns (>= 10 dB = the bed window is a bed "
                 "measurement here)"}
        if m_meas is not None:
            j = int(np.argmin(np.abs(s0 + p["s_m"] / 1e3 - ts1)))
            jl = slice(j, j + 1)
            tprofs["measured"] = rel_mean_profile(
                meas[jl], p["tw_m"], p["dt"], p["surf"][jl],
                m_meas["_spk"][jl])
            with np.errstate(divide="ignore", invalid="ignore"):
                mid_db = 10.0 * np.log10(m_meas["_mid"] / m_meas["_spk"])
            ok = np.isfinite(mid_db)
            tinfo.update({
                "measured_trace_index": j,
                "measured_s_km": round(float(s0 + p["s_m"][j] / 1e3), 3),
                "measured_midcol_rel_surf_db": round(float(mid_db[j]), 2),
                "measured_midcol_percentile": round(float(
                    (mid_db[ok] < mid_db[j]).mean()), 3)})
        tprofs_l.append(tprofs)
        tinfo_l.append(tinfo)
    tprofs = tprofs_l[0] if tprofs_l else None
    tinfo = tinfo_l[0] if tinfo_l else None
    with np.errstate(divide="ignore", invalid="ignore"):
        blp = 10.0 * np.log10(bedlayer_bed / spk)
    return {"gate": gate, "sim": clean, "meas": cleanm,
            "trace_profs": tprofs, "trace_info": tinfo,
            "trace_profs_list": tprofs_l or None,
            "trace_info_list": tinfo_l or None,
            "Ps": Ps, "Pb": Pb, "t_b": t_b, "twtt_sim": tw,
            "spk_sim": m_sim["_spk"], "spk_meas":
                (None if m_meas is None else m_meas["_spk"]),
            "sim_bed_prof_db": _prof_db(m_sim),
            "meas_bed_prof_db": meas_prof,
            "sim_bedlayer_prof_db": np.where(np.isfinite(blp), blp, np.nan),
            "decomposition": {k: {kk: round(vv, 2) for kk, vv in v.items()}
                              for k, v in dec.items()},
            "verdict": verdict, "floor_db": floor_db,
            "floor_doc": (floor_doc if meas is not None else None),
            "meas_noise_limited": noise_limited,
            "bed_delay_med_us": round(float(np.nanmedian(
                (p["bot"] - p["surf"]))) * 1e6, 2),
            "profs": profs, "bed_profs": bed_profs, "tail_cov": tail_cov,
            "P": P, "t_s": t_s, "meas_arr": meas}


# ========================================================================
# ZONE-SPLIT analysis (full_line): grounded vs floating sub-windows
# ========================================================================
# TERMINOLOGY (user-set): "surface returns" = the surface returns layer,
# "bed returns" = the basal-layer returns -- on the floating side the
# "bed" is the ice-ocean shelf base. All levels dB rel each trace's OWN
# surface-return peak, as everywhere in this tool.
def zone_analysis(p, a, gl_km=None):
    """Grounded/floating split of the standard per-pass metrics: clutter
    windows, decomposition, bed-referenced tail (slope/excess/guard) and the
    bed-window level residual vs measured -- each zone judged only against
    its own traces. Returns {zone: {"metrics": ..., "profs": ...,
    "bed_profs": ...}}; the key science number is the floating bed-window
    residual (does the fixed K reproduce the shelf-base brightness?)."""
    gl_km = GL_S_KM if gl_km is None else gl_km    # active line, call time
    s0 = S0_KM[p["segment"]]
    tw, dtf = a["twtt_sim"], p["rc_frame"].dt
    P, Ps, Pb = a["P"], a["Ps"], a["Pb"]
    t_s, t_b, spk = a["t_s"], a["t_b"], a["spk_sim"]
    meas = a["meas_arr"]
    s_sim_km = s0 + p["s_sim"] / 1e3
    s_m_km = s0 + p["s_m"] / 1e3
    lo_us, hi_us = TAIL_FIT_US
    out = {}
    for name, zlo, zhi in (("grounded", -1e9, gl_km),
                           ("floating", gl_km, 1e9)):
        ms = (s_sim_km >= zlo) & (s_sim_km < zhi)
        if ms.sum() < 5:
            out[name] = {"metrics": {"n_traces_sim": int(ms.sum()),
                                     "note": "too few traces"}}
            continue
        msim = clutter_metrics(P[ms], tw, dtf, t_s[ms], t_b[ms])
        dec = {}
        for lname, Pl in (("surface_returns", Ps), ("bed_returns", Pb)):
            mid = _wmean(Pl[ms], tw, dtf, t_s[ms] + MID_LO_US * 1e-6,
                         t_b[ms] - MID_HI_US * 1e-6)
            bed = _wmean(Pl[ms], tw, dtf, t_b[ms] - BED_LO_US * 1e-6,
                         t_b[ms] + BED_HI_US * 1e-6)
            dec[lname] = {
                "midcol_rel_surf_db": round(_med_db_rel(mid, msim["_spk"]), 2),
                "bed_rel_surf_db": round(_med_db_rel(bed, msim["_spk"]), 2)}
        profs = {
            "sim_total": rel_mean_profile(P[ms], tw, dtf, t_s[ms], spk[ms]),
            "sim_surface": rel_mean_profile(Ps[ms], tw, dtf, t_s[ms],
                                            spk[ms]),
            "sim_bed": rel_mean_profile(Pb[ms], tw, dtf, t_s[ms], spk[ms])}
        bed_profs = {
            "sim_total": rel_mean_profile(P[ms], tw, dtf, t_b[ms], spk[ms],
                                          *TAIL_PROF_US),
            "sim_surface": rel_mean_profile(Ps[ms], tw, dtf, t_b[ms],
                                            spk[ms], *TAIL_PROF_US),
            "sim_bed": rel_mean_profile(Pb[ms], tw, dtf, t_b[ms], spk[ms],
                                        *TAIL_PROF_US)}
        rel, tot = bed_profs["sim_total"]
        sur, bed_ = bed_profs["sim_surface"][1], bed_profs["sim_bed"][1]
        mfit = (rel >= lo_us) & (rel <= hi_us)
        marg = bed_[mfit] - sur[mfit]
        j = int(np.argmin(marg))
        met = {
            "n_traces_sim": int(ms.sum()),
            "s_km": [round(float(s_sim_km[ms].min()), 2),
                     round(float(s_sim_km[ms].max()), 2)],
            "sim": {k: round(v, 2) for k, v in msim.items()
                    if not k.startswith("_")},
            "decomposition_db": dec,
            "tail": {
                "sim_slope_db_per_us": round(
                    tail_slope_db_per_us(rel, tot), 3),
                "sim_bed_returns_slope_db_per_us": round(
                    tail_slope_db_per_us(rel, bed_), 3),
                "guard": {
                    "min_bed_minus_surface_returns_db": round(
                        float(marg[j]), 2),
                    "at_us": round(float(rel[mfit][j]), 2),
                    "threshold_db": TAIL_GUARD_DB,
                    "pass": bool(marg[j] >= TAIL_GUARD_DB)}}}
        if meas is not None:
            mm = (s_m_km >= zlo) & (s_m_km < zhi)
            mmeas = clutter_metrics(meas[mm], p["tw_m"], p["dt"],
                                    p["surf"][mm], p["bot"][mm])
            n_m = int(mm.sum())
            zf_lo, zf_hi, _ = floor_window(p["tw_m"], p["bot"][mm])
            floor = _wmean(meas[mm], p["tw_m"], p["dt"],
                           np.full(n_m, zf_lo), np.full(n_m, zf_hi))
            bed_profs["measured"] = rel_mean_profile(
                meas[mm], p["tw_m"], p["dt"], p["bot"][mm], mmeas["_spk"],
                *TAIL_PROF_US)
            profs["measured"] = rel_mean_profile(
                meas[mm], p["tw_m"], p["dt"], p["surf"][mm], mmeas["_spk"])
            relm, dbm = bed_profs["measured"]
            met["measured"] = {k: round(v, 2) for k, v in mmeas.items()
                               if not k.startswith("_")}
            met["measured"]["floor_rel_surf_db"] = round(
                _med_db_rel(floor, mmeas["_spk"]), 2)
            met["tail"]["meas_slope_db_per_us"] = round(
                tail_slope_db_per_us(relm, dbm), 3)
            met["tail"]["excess_db"] = {
                f"+{t:g}us": round(_at_us(rel, tot, t) - _at_us(relm, dbm, t),
                                   2) for t in TAIL_EXCESS_US}
            met["bed_window_residual_db"] = round(
                msim["bed_rel_surf_db"] - mmeas["bed_rel_surf_db"], 2)
            met["midcol_residual_db"] = round(
                msim["midcol_rel_surf_db"] - mmeas["midcol_rel_surf_db"], 2)
        out[name] = {"metrics": met, "profs": profs, "bed_profs": bed_profs}
    return out


def fig_decomposition_zones(out, key, zres, gl_km=None,
                            fname="decomposition_zones.png", src=None):
    """Trace-averaged decomposition of ONE pass split into the grounded and
    floating sub-windows: measured vs sim total vs surface/bed returns,
    surface-referenced, one panel per zone (fig_decomposition's series)."""
    gl_km = GL_S_KM if gl_km is None else gl_km    # active line, call time
    series = [("measured", "measured", dict(color="black", lw=1.8)),
              ("sim_total", "sim total", dict(color="tab:blue", lw=1.4)),
              ("sim_surface", "sim surface returns",
               dict(color="tab:orange", lw=1.2, ls="--")),
              ("sim_bed", "sim bed returns",
               dict(color="tab:green", lw=1.2, ls="-."))]
    zones = [z for z in ("grounded", "floating") if "profs" in zres.get(z, {})]
    if not zones:
        return None
    fig, axs = plt.subplots(1, len(zones), figsize=(5.4 * len(zones), 4.8),
                            sharey=True, squeeze=False)
    for k, zn in enumerate(zones):
        ax, z = axs[0, k], zres[zn]
        for pk, label, st in series:
            if pk in z["profs"]:
                ax.plot(*z["profs"][pk], label=label, **st)
        met = z["metrics"]
        res = met.get("bed_window_residual_db")
        ax.set_xlim(*PROFILE_X_US)
        ax.set_ylim(*PROFILE_DB)
        ax.grid(alpha=0.3)
        ax.set_title(
            f"{key} {zn.upper()} (s {met['s_km'][0]:.0f}-"
            f"{met['s_km'][1]:.0f} km, {met['n_traces_sim']} traces)"
            + ("" if res is None
               else f"\nbed-window sim - measured {res:+.2f} dB"),
            fontsize=10)
        ax.set_xlabel("twtt below surface returns (us)")
        if k == 0:
            ax.set_ylabel("dB rel own surface-return peak (mean power)")
            ax.legend(fontsize=8, loc="upper right")
    title = (f"zone-split decomposition, GL at s = {gl_km:g} km "
             "(grounded rock bed vs floating ice-ocean shelf base)")
    fig.suptitle((src + "\n" + title) if src else title)
    fig.tight_layout()
    fp = out / fname
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


def _smooth_db(s, v, win_m=None):
    """~win_m running mean of a per-trace dB profile (NaNs interpolated)."""
    win_m = CORR_WIN_M if win_m is None else win_m    # active line, call time
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
    s_meas, s_sim = p["s_m"], p["s_sim"]
    meas = np.interp(s_sim, s_meas,
                     _smooth_db(s_meas, a["meas_bed_prof_db"]))
    sim_r = _smooth_db(s_sim, a["sim_bed_prof_db"])
    sim_c = _smooth_db(s_sim, a_const["sim_bed_prof_db"])
    sim_rl = _smooth_db(s_sim, a["sim_bedlayer_prof_db"])  # bed returns only
    # implied pattern -RSSNR(s) + K (== G2 - 2AH), at the sim traces'
    # anchor-axis position
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
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


def fig_bed_brightness(out, preps, corr_series, corr_stats, segment,
                       syn=None):
    """Per measured pass: bed-window power along-track (dB rel own surface
    peak, ~1 km smoothed) -- measured vs constant-gamma sim vs RSSNR-gamma
    sim vs the RSSNR-implied pattern (shape prediction, median-aligned to
    the RSSNR sim). ``syn`` = (key, series) adds a PREDICTION panel
    (simulated only -- no measured curve exists)."""
    s0 = S0_KM[segment]
    keys = list(corr_series)
    ncol = len(keys) + (1 if syn else 0)
    fig, axs = plt.subplots(1, ncol, figsize=(5.4 * ncol, 4.6),
                            sharey=True, squeeze=False)
    for k, key in enumerate(keys):
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
    if syn:
        key, se = syn
        ax = axs[0, len(keys)]
        s_km = s0 + se["s_sim"] / 1e3
        imp = se["implied"] + (np.nanmedian(se["sim_rssnr"])
                               - np.nanmedian(se["implied"]))
        ax.plot(s_km, se["sim_rssnr"], color="tab:red", lw=1.3,
                label="sim RSSNR gamma (prediction)")
        ax.plot(s_km, imp, color="0.45", lw=1.0, ls="--",
                label="RSSNR-implied (median-aligned)")
        ax.set_title(f"{key} ({preps[key]['h_med']:.0f} m AGL) -- "
                     "PREDICTION (no measured)", fontsize=9)
        ax.set_xlabel("anchor along-track s (km)")
        ax.grid(alpha=0.3)
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
def _panel_scale(arr, vmin, vmax):
    """(vmin, vmax, note) for one panel: the shared RADARGRAM_DB, or this
    panel's own robust percentiles when RADARGRAM_SCALE == "per_panel"."""
    if RADARGRAM_SCALE != "per_panel":
        return vmin, vmax, ""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return vmin, vmax, ""
    lo, hi = np.percentile(finite, RADARGRAM_PCT)
    return float(lo), float(hi), f"  [{lo:.0f}, {hi:.0f}] dB"


def source_color_rgb(db_total, db_surf, db_bed, vmin, vmax):
    """Source-coloured radargram (docs/source_color_radargrams.md): pixel
    brightness is the TOTAL power on the panel's grey scale, unchanged; hue
    is the dominant source (SOURCE_COLORS) and saturation is its dominance,
    linear in |bed fraction - 0.5| * 2, so a 50/50 pixel stays grey."""
    from matplotlib.colors import to_rgb
    lum = np.clip((db_total - vmin) / (vmax - vmin), 0.0, 1.0)
    ps = np.nan_to_num(10.0 ** (db_surf / 10.0))   # NaN: no field there
    pb = np.nan_to_num(10.0 ** (db_bed / 10.0))
    fb = pb / (ps + pb + 1e-300)
    sat = (np.abs(fb - 0.5) * 2.0)[..., None]
    hue = np.where(fb[..., None] > 0.5,
                   np.array(to_rgb(SOURCE_COLORS["bed"])),
                   np.array(to_rgb(SOURCE_COLORS["surface"])))
    tint = (1.0 - sat) + sat * hue
    return np.clip(lum[..., None] * tint, 0.0, 1.0)


def _source_color_legend(ax):
    from matplotlib.patches import Patch
    h = [Patch(color=SOURCE_COLORS[k], label=f"{k} returns")
         for k in ("surface", "bed")]
    h.append(Patch(color="0.6", label="mixed (~50/50)"))
    ax.legend(handles=h, fontsize=7, loc="lower left", framealpha=0.7)


def _sim_radargram_panel(ax, p, a, key, label, s0, y_lo, y_hi, vmin, vmax):
    """One simulated-pass panel: dB rel per-pass median simulated surface
    peak, surface-referenced twtt axis. NO bed overlay (see
    BED_OVERLAY_STYLE): the sim's bed nadir twtt is a model input.
    RADARGRAM_SOURCE_COLOR tints the panel by energy source
    (source_color_rgb); the brightness is the same grey image."""
    twtt_s = p["rc_frame"].t0 + np.arange(
        p["rc_frame"].n_samples) * p["rc_frame"].dt
    ref_s = 10.0 * np.log10(max(float(np.nanmedian(
        _wpeak(a["P"], twtt_s, p["rc_frame"].dt, a["t_s"],
               SURF_WIN_US))), 1e-300))
    surf_med_s = float(np.nanmedian(a["t_s"]))
    rel_s = (twtt_s - surf_med_s) * 1e6
    ms = (rel_s >= y_lo) & (rel_s <= y_hi)
    s_sim = s0 + p["s_sim"] / 1e3
    img = _db(a["P"])[:, ms].T - ref_s
    vmin, vmax, note = _panel_scale(img, vmin, vmax)
    extent = [s_sim[0], s_sim[-1], rel_s[ms][-1], rel_s[ms][0]]
    if RADARGRAM_SOURCE_COLOR:
        rgb = source_color_rgb(img, _db(a["Ps"])[:, ms].T - ref_s,
                               _db(a["Pb"])[:, ms].T - ref_s, vmin, vmax)
        ax.imshow(rgb, aspect="auto", extent=extent)
        _source_color_legend(ax)
        note += "  [colour: source]"
    else:
        ax.imshow(img, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax,
                  extent=extent)
    ax.set_title(f"{key} sim {label} (ct ±{p['reach']['ct_m'] / 1e3:.1f} km,"
                 f" {p['spacing']:.1f} m facets){note}", fontsize=10)


def fig_radargrams(out, preps, analyses, segment, keys=None, ablation=None,
                   gl_s_km=None, w_scale=1.0, plot_s_max_km=None,
                   fname="radargrams.png", src=None, bed_overlay=True):
    """Measured (top) vs simulated per pass, shared surface-referenced twtt
    axis and one shared dB-rel-surface color scale. A synthetic pass has no
    measured data: its top panel is a placeholder. ``ablation`` = list of
    (preps, analyses, label) bed-source rows appended below the picked-bed
    row (row 2 is then labeled 'picked bed'): the clean bed-source
    ablation. ``gl_s_km`` marks the grounding line on every panel.
    ``plot_s_max_km`` crops the PLOTTED along-track range (the data and
    every metric keep the full segment); ``src`` prepends a source-data
    provenance line to the figure title. ``bed_overlay`` (default ON) draws
    the MEASURED Bottom pick, dotted, on measured panels only."""
    keys = keys or ORDER
    ablation = ablation or []
    y_lo, y_hi = RADARGRAM_Y_US
    vmin, vmax = RADARGRAM_DB
    s0 = S0_KM[segment]
    nrow = 2 + len(ablation)
    fig, axs = plt.subplots(nrow, len(keys),
                            figsize=(5.4 * len(keys) * w_scale, 4.4 * nrow),
                            sharey=True, squeeze=False)
    for k, key in enumerate(keys):
        p, a = preps[key], analyses[key]
        ax = axs[0, k]
        if a["meas_arr"] is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"no measured data\n({key}: simulated "
                    "prediction only)", ha="center", va="center",
                    fontsize=11, transform=ax.transAxes)
        else:
            # measured: dB rel per-pass median surface peak
            ref_m = 10.0 * np.log10(max(np.nanmedian(
                _wpeak(a["meas_arr"], p["tw_m"], p["dt"], p["surf"],
                       SURF_WIN_US)), 1e-300))
            surf_med = float(np.nanmedian(p["surf"]))
            rel = (p["tw_m"] - surf_med) * 1e6
            m = (rel >= y_lo) & (rel <= y_hi)
            s_km = s0 + p["s_m"] / 1e3
            img = _db(a["meas_arr"])[:, m].T - ref_m
            vmin_m, vmax_m, note_m = _panel_scale(img, vmin, vmax)
            ax.imshow(img, aspect="auto", cmap="gray",
                      vmin=vmin_m, vmax=vmax_m,
                      extent=[s_km[0], s_km[-1], rel[m][-1], rel[m][0]])
            if bed_overlay:
                ax.plot(s_km, (p["bot"] - surf_med) * 1e6,
                        label="measured Bottom pick", **BED_OVERLAY_STYLE)
                ax.set_ylim(y_hi, y_lo)
                if k == 0:
                    ax.legend(fontsize=7, loc="lower left", framealpha=0.7)
            ax.set_title(f"{key} measured ({p['h_med']:.0f} m AGL)"
                         f"{note_m}", fontsize=10)
        # sim row(s): picked-bed (labeled when ablation rows are present)
        _sim_radargram_panel(axs[1, k], p, a, key,
                             "(picked bed)" if ablation else "",
                             s0, y_lo, y_hi, vmin, vmax)
        for r, (pr, an, label) in enumerate(ablation):
            _sim_radargram_panel(axs[2 + r, k], pr[key], an[key], key,
                                 f"({label})", s0, y_lo, y_hi, vmin, vmax)
        if gl_s_km is not None:
            for r in range(nrow):
                ax_ = axs[r, k]
                if ax_.get_images():
                    ax_.axvline(gl_s_km, color="tab:red", lw=1.0, ls="--",
                                alpha=0.85)
                    ax_.text(gl_s_km, y_lo + 0.4, " GL", color="tab:red",
                             fontsize=8, va="top")
        if plot_s_max_km is not None:
            s_km_ = s0 + p["s_sim"] / 1e3
            for r in range(nrow):
                axs[r, k].set_xlim(float(s_km_[0]),
                                   min(float(s_km_[-1]), plot_s_max_km))
        axs[nrow - 1, k].set_xlabel("anchor along-track s (km)")
    for r in range(nrow):
        axs[r, 0].set_ylabel("twtt below surface (us)")
    title = ("basal-clutter altitude comparison: measured (top) vs simulated "
             "surface+bed, dB rel own surface peak"
             + (" [cyan dotted: measured Bottom pick]"
                if bed_overlay else "")
             + (" -- bed-source ablation: picked bed / "
                + " / ".join(label for _, _, label in ablation)
                + ", all else identical" if ablation else "")
             + (f"\n[plotted s <= {plot_s_max_km:g} km of the full segment]"
                if plot_s_max_km is not None else ""))
    fig.suptitle((src + "\n" + title) if src else title, fontsize=11)
    fig.tight_layout()
    fp = out / fname
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


def fig_decomposition(out, preps, analyses, keys=None, ablation=None,
                      fname="decomposition.png", src=None):
    """Per pass: measured vs sim total vs the sim's per-interface split
    (surface returns vs bed returns) mean-power profiles below the surface.

    ``ablation`` = list of (preps, analyses, label) bed-source variants
    (radargram row order): the panel then shows measured, ONE surface returns
    curve (verified bed-source-invariant; all variants drawn and flagged if
    they deviate beyond speckle/numerical noise) and one BED returns curve per
    bed source; sim totals are dropped for legibility."""
    series = [("measured", "measured", dict(color="black", lw=1.8)),
              ("sim_total", "sim total", dict(color="tab:blue", lw=1.4)),
              ("sim_surface", "sim surface returns",
               dict(color="tab:orange", lw=1.2, ls="--")),
              ("sim_bed", "sim bed returns",
               dict(color="tab:green", lw=1.2, ls="-."))]
    ab_styles = [dict(color="tab:red", lw=1.2, ls=":"),
                 dict(color="tab:purple", lw=1.2, ls=(0, (4, 2)))]
    keys = keys or ORDER
    if ablation:
        series = [s for s in series if s[0] != "sim_total"]
        series[-1] = ("sim_bed", "sim bed returns (picked bed)",
                      dict(color="tab:green", lw=1.2, ls="-."))
    fig, axs = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.8),
                            sharey=True, squeeze=False)
    for k, key in enumerate(keys):
        ax = axs[0, k]
        a = analyses[key]
        for pk, label, st in series:
            if pk in a["profs"]:
                ax.plot(*a["profs"][pk], label=label, **st)
        for (pr_v, an_v, label), st in zip(ablation or [], ab_styles):
            av = an_v[key]
            # surface returns must be bed-source-invariant (same surface DEM,
            # geometry, speckle seeds): verify, plot only if it deviates
            x0, y0 = a["profs"]["sim_surface"]
            xv, yv = av["profs"]["sim_surface"]
            dev = float(np.nanmax(np.abs(
                np.interp(x0, xv, yv) - y0)[(x0 >= PROFILE_X_US[0])
                                            & (x0 <= PROFILE_X_US[1])
                                            & (y0 > PROFILE_DB[0] + 5.0)]))
            print(f"  decomposition {key} [{label}]: surface returns max "
                  f"deviation {dev:.3f} dB vs picked-bed run", flush=True)
            if dev > 0.3:
                ax.plot(xv, yv, color=st["color"], lw=0.9, ls="--",
                        label=f"sim surface returns ({label}) DEVIATES "
                              f"{dev:.1f} dB")
            ax.plot(*av["profs"]["sim_bed"],
                    label=f"sim bed returns ({label})", **st)
        tb = a["bed_delay_med_us"]
        ax.axvspan(1.0, tb - MID_HI_US, color="tab:blue", alpha=0.06,
                   label="mid-column window" if k == 0 else None)
        ax.axvline(tb, color="0.5", lw=0.8, ls=":")
        ax.text(tb, PROFILE_DB[0] + 2.0, " median bed", fontsize=7,
                color="0.4")
        ax.set_xlim(*PROFILE_X_US)
        ax.set_ylim(*PROFILE_DB)
        ax.grid(alpha=0.3)
        ax.set_title(f"{key} ({preps[key]['h_med']:.0f} m AGL)  "
                     f"[{a['verdict']}]", fontsize=10)
        ax.set_xlabel("twtt below surface (us)")
        if k == 0:
            ax.set_ylabel("dB rel own surface peak (mean power)")
            ax.legend(fontsize=8, loc="upper right")
    title = ("clutter decomposition: which interface supplies the "
             "mid-column energy (per-layer coherent fields)")
    fig.suptitle((src + "\n" + title) if src else title)
    fig.tight_layout()
    fp = out / fname
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


def fig_decomposition_trace(out, preps, analyses, keys=None,
                            fname="decomposition_trace.png", src=None):
    """SINGLE-TRACE variant of fig_decomposition: the same measured / sim
    total / sim surface returns / sim bed returns curves at ONE slow-time
    location (``--trace-decomp-s``, recorded per pass in the config) instead
    of the trace ensemble average. Single-trace curves are speckly BY
    CONSTRUCTION -- that is the point: it shows what one sounding looks like
    rather than the ensemble mean the tail/level metrics are built on."""
    keys = [k for k in (keys or ORDER) if analyses[k]["trace_profs"]]
    if not keys:
        return None
    series = [("measured", "measured", dict(color="black", lw=1.2)),
              ("sim_total", "sim total", dict(color="tab:blue", lw=1.1)),
              ("sim_surface", "sim surface returns",
               dict(color="tab:orange", lw=1.0, ls="--")),
              ("sim_bed", "sim bed returns",
               dict(color="tab:green", lw=1.0, ls="-."))]
    # one panel per (pass, location): multi-location runs (full_line's
    # grounded + floating pair) fan out into extra columns
    panels = []
    for key in keys:
        a = analyses[key]
        tps = a.get("trace_profs_list") or [a["trace_profs"]]
        tis = a.get("trace_info_list") or [a["trace_info"]]
        panels += [(key, tp, ti) for tp, ti in zip(tps, tis)]
    fig, axs = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8),
                            sharey=True, squeeze=False)
    for k, (key, tprofs, ti) in enumerate(panels):
        ax = axs[0, k]
        for pk, label, st in series:
            if pk in tprofs:
                ax.plot(*tprofs[pk], label=label, **st)
        tb = ti["bed_below_surface_us"]
        ax.axvspan(1.0, tb - MID_HI_US, color="tab:blue", alpha=0.06,
                   label="mid-column window" if k == 0 else None)
        ax.axvspan(tb - BED_LO_US, tb + BED_HI_US, color="tab:green",
                   alpha=0.08, label="bed-return window" if k == 0 else None)
        ax.axvline(tb, color="0.5", lw=0.8, ls=":")
        ax.set_xlim(*PROFILE_X_US)
        ax.set_ylim(*PROFILE_DB)
        ax.grid(alpha=0.3)
        ax.set_title(f"{key} ({preps[key]['h_med']:.0f} m AGL)  s = "
                     f"{ti['sim_s_km']:.2f} km, trace {ti['sim_trace_index']}"
                     f"\nbed at {tb:.2f} us; bed-window bed - surface returns"
                     f" {ti['bed_window_bed_minus_surface_returns_db']:+.1f}"
                     " dB", fontsize=9)
        ax.set_xlabel("twtt below surface returns (us)")
        if k == 0:
            ax.set_ylabel("dB rel own surface-return peak (single trace)")
            ax.legend(fontsize=8, loc="upper right")
    s_list = sorted({round(ti["requested_s_km"], 2) for _, _, ti in panels})
    title = ("SINGLE-TRACE decomposition, anchor s = "
             + " / ".join(f"{v:.2f}" for v in s_list) + " km"
             "\n(one sounding per panel, not the trace ensemble mean)")
    fig.suptitle((src + "\n" + title) if src else title, fontsize=10)
    fig.tight_layout()
    fp = out / fname
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


def fig_bed_tail(out, preps, analyses, metrics, keys=None, ablation=None,
                 fname="bed_tail.png", src=None):
    """Per pass: BED-REFERENCED ensemble mean-power profiles -- measured vs
    each bed source's sim total, plus the sim surface-return curve (the
    fair-comparison guard) and the measured noise floor. The fit window is
    shaded; this is the figure behind bed_return_tail_*."""
    keys = keys or ORDER
    styles = {"picked_bed": dict(color="tab:green", lw=1.4),
              "bedmachine": dict(color="tab:red", lw=1.2, ls=":"),
              "demogorgn": dict(color="tab:purple", lw=1.2, ls=(0, (4, 2)))}
    ab_an = {("demogorgn" if "DEMOGORGN" in label else "bedmachine"): an
             for _, an, label in (ablation or [])}
    fig, axs = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.6),
                            sharey=True, squeeze=False)
    for k, key in enumerate(keys):
        ax, a = axs[0, k], analyses[key]
        e = metrics[f"bed_return_tail_{key}"]
        main = next(iter(e["sim"]))
        if "measured" in a["bed_profs"]:
            ax.plot(*a["bed_profs"]["measured"], color="black", lw=1.8,
                    label=f"measured ({e['measured']['slope_db_per_us']:+.1f}"
                          " dB/us)")
            _fl = e["measured"]["noise_floor_caveat"]["floor_rel_surf_db"]
            if _fl is not None:
                ax.axhline(_fl, color="0.5", lw=0.9, ls=":",
                           label="measured floor")
        for slug, an in [(main, analyses)] + list(ab_an.items()):
            av = an[key] if slug != main else a
            ax.plot(*av["bed_profs"]["sim_total"],
                    label=f"sim {slug.replace('_', ' ')} "
                          f"({e['sim'][slug]['slope_db_per_us']:+.1f} dB/us)",
                    **styles.get(slug, dict(lw=1.2)))
        ax.plot(*a["bed_profs"]["sim_surface"], color="tab:orange", lw=1.0,
                ls="--", label=f"sim surface returns ({main.replace('_', ' ')})")
        ax.axvspan(*TAIL_FIT_US, color="tab:blue", alpha=0.07,
                   label="slope fit window" if k == 0 else None)
        ax.axvline(0.0, color="0.5", lw=0.8)
        ang = e["bed_return_angle_map_deg"]
        ax.set_title(f"{key} ({preps[key]['h_med']:.0f} m AGL)  bed incidence "
                     f"+1/+2/+3 us = {ang['+1us']:.0f}/{ang['+2us']:.0f}/"
                     f"{ang['+3us']:.0f} deg", fontsize=9)
        ax.set_xlim(*TAIL_PROF_US)
        ax.grid(alpha=0.3)
        ax.set_xlabel("delay past own bed reference (us)")
        if k == 0:
            ax.set_ylabel("dB rel own surface peak (mean power)")
            ax.legend(fontsize=7, loc="lower left")
    title = ("bed-return tail: decay past the bed echo (each dataset on "
             "its OWN bed reference)")
    fig.suptitle((src + "\n" + title) if src else title)
    fig.tight_layout()
    fp = out / fname
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


# ========================================================================
# per-pass figure sets (--per-pass-figs): incremental delivery
# ========================================================================
def frame_span(parts):
    """Compact provenance id for a parts list: '20161031_07_002-005' when
    the frames share one flight prefix, else the '/'-joined full ids."""
    ids = [fid for fid, _ in parts]
    pre = ids[0].rsplit("_", 1)[0]
    if any(f.rsplit("_", 1)[0] != pre for f in ids):
        return "/".join(ids)
    nums = sorted(int(f.rsplit("_", 1)[1]) for f in ids)
    if len(ids) == 1:
        return ids[0]
    return f"{pre}_{nums[0]:03d}-{nums[-1]:03d}"


def source_label(key, p):
    """SOURCE-DATA provenance line for the figure tops: season + frame
    segments (+ altitude); synthetic passes name the carrier line."""
    span = frame_span(p["parts"])
    if p.get("synthetic"):
        alt = p["synthetic"].get("synthetic_msl_m",
                                 p["synthetic"].get("agl_med_m", 0.0))
        return (f"{p.get('season', SEASON)} - SYNTHETIC {alt / 1e3:g} km "
                f"constant-altitude pass on the {span} line "
                f"(no measured data)")
    return (f"{p.get('season', SEASON)} - measured {span} "
            f"({p['h_med'] / 1e3:.1f} km AGL)")


def emit_pass_figs(out, key, p, a, zres, segment, gl_s_km, plot_s_max_km,
                   main_slug):
    """Write ONE pass's complete figure set as separate suffixed files
    (radargrams_<key>.png, ...), each labeled with the source-data
    provenance, immediately after that pass's sim+processing+analysis --
    the staged-delivery contract. Returns the figure paths."""
    src = source_label(key, p)
    figs = [fig_radargrams(out, {key: p}, {key: a}, segment, keys=[key],
                           gl_s_km=gl_s_km, w_scale=FIG_WIDTH_SCALE,
                           plot_s_max_km=plot_s_max_km,
                           fname=f"radargrams_{key}.png", src=src,
                           bed_overlay=BED_OVERLAY),
            fig_decomposition(out, {key: p}, {key: a}, keys=[key],
                              fname=f"decomposition_{key}.png", src=src)]
    e = bed_tail_entry(key, p, a, [(main_slug, a)])
    figs.append(fig_bed_tail(out, {key: p}, {key: a},
                             {f"bed_return_tail_{key}": e}, keys=[key],
                             fname=f"bed_tail_{key}.png", src=src))
    ftr = fig_decomposition_trace(out, {key: p}, {key: a}, keys=[key],
                                  fname=f"decomposition_trace_{key}.png",
                                  src=src)
    if ftr is not None:
        figs.append(ftr)
    if zres is not None:
        fz = fig_decomposition_zones(out, key, zres,
                                     fname=f"decomposition_zones_{key}.png",
                                     src=src)
        if fz is not None:
            figs.append(fz)
    return figs


# ========================================================================
# main
# ========================================================================
def run(segment="pilot", n_traces=None, att=rac.ATT_DB_PER_KM,
        surf_rough=True, out_root=None, force=False, make_report=True,
        picked_bed=False, gamma_rssnr=False, processing="none",
        bed_ablation=False,
        demogorgn_bed=None,     # None -> the line's bed_dem decides
        demogorgn_seed=0, companion=True, out_name=None,
        antenna=ANT_DEFAULT, bed_rough=None, posting_div=1,
        focus_aperture="alias_limited",
        bed_rough_extra_db=0.0, passes=None, spec=None,
        grazing_fix=None,   # None -> ON at GRAZING_FIX['s_eff']; False -> off

        gamma_surface_db=None, trace_decomp_s_km=None,
        per_pass_figs=False,
        plot_s_max_km=None, proc_cache=False, line=None,
        spec_doc=None, spec_path=None,
        instruments=None, extra_passes=None):
    # Always re-activate: this resets PASSES to the line definition, so
    # extra_passes added below cannot leak into a later run in-process.
    activate_line(line or LINE)
    if demogorgn_bed is None:
        demogorgn_bed = BED_DEM == "demogorgn"
    # The grazing-angle facet-lattice fix is a BUG FIX and is ON by default
    # (analysis.yaml grazing_fix.s_eff); False (--no-grazing-fix) is the
    # legacy path, kept only for artifact demonstrations and A/B regression.
    if grazing_fix is None:
        grazing_fix = float(GRAZING_FIX["s_eff"])
    elif grazing_fix is False:
        grazing_fix = None
    instruments = dict(instruments or {})
    if extra_passes:
        extra, syn = {}, list(SYNTHETIC_KEYS)
        for name, ep in extra_passes.items():
            # "reference" resolves per line, so one multi-line spec can
            # invent the same observation on lines with different pass names
            carrier = REF_PASS if ep["carrier"] == "reference" \
                else ep["carrier"]
            if carrier not in PASSES:
                raise ValueError(f"extra pass {name!r}: carrier "
                                 f"{carrier!r} is not a pass of line "
                                 f"{LINE!r}")
            base = dict(PASSES[carrier])
            base["agl_med_m"] = None
            base["synthetic_msl_m"] = float(ep["altitude_m"])
            if ep.get("facet_spacing_scale"):
                base["facet_spacing_scale"] = ep["facet_spacing_scale"]
            if ep.get("instrument"):
                base["instrument"] = ep["instrument"]
                instruments.setdefault(name, ep["instrument"])
            extra[name] = base
            syn.append(name)
        globals()["PASSES"] = {**PASSES, **extra}
        globals()["SYNTHETIC_KEYS"] = tuple(syn)
    if segment not in SEGMENTS:
        raise ValueError(f"line {LINE!r} has no {segment!r} segment; "
                         f"have {list(SEGMENTS)}")
    for feat, on in (("gamma_rssnr", gamma_rssnr),
                     ("demogorgn_bed", demogorgn_bed)):
        if on and feat in UNSUPPORTED:
            raise ValueError(
                f"{feat} is not wired for line {LINE!r}: "
                + {"gamma_rssnr": "no required-surface-SNR mapping has "
                   "been calibrated on this line",
                   "demogorgn_bed": "DEMOGORGN is an ANTARCTIC-only bed "
                   "ensemble (Bedmap3 grid)"}[feat])
    proc = processing == "standard"
    hybrid = segment in SEGMENTS_CROSSING_GL
    if hybrid and not demogorgn_bed:
        raise ValueError(
            f"segment {segment!r} crosses the grounding line: the hybrid "
            "bed (grounded DEM blended into the floating reference-pass "
            "picks) is wired for bed_dem demogorgn only")
    if hybrid and bed_ablation:
        raise ValueError("--bed-ablation is not wired for a crosses_gl "
                         "hybrid segment")
    if out_name and bed_ablation:
        raise ValueError("--out-name relocates the case directory; the "
                         "ablation runs resolve their own sibling "
                         "directories, so run it without --bed-ablation")
    if spec and not gamma_rssnr:
        raise ValueError("--specular-fraction splits the RSSNR-mapped bed "
                         "reflectivity: use it with --gamma-from-rssnr")
    if bed_rough and not gamma_rssnr:
        raise ValueError("--bed-rough needs --gamma-from-rssnr: the "
                         "double-count guard is applied to the RSSNR-mapped "
                         "gamma (constant-gamma path not wired)")
    if bed_ablation and not picked_bed:
        raise ValueError("--bed-ablation adds bed-source rows to the "
                         "picked-bed case: run it WITH --picked-bed")
    # Naming a synthetic in ``passes`` is the only way to request it: the
    # line definition declares which synthetics exist, so the old
    # --add-14km/--add-30km/--add-300km/--add-500km flags carried no
    # information the pass list did not already have.
    if passes:
        unknown = [k for k in passes if k not in PASSES]
        if unknown:
            raise ValueError(
                f"line {LINE!r} defines no pass(es) {unknown}; real passes "
                f"{list(ORDER)}, synthetics {list(SYNTHETIC_KEYS)}")
        order = [k for k in list(ORDER) + list(SYNTHETIC_KEYS)
                 if k in passes]
    else:
        # extra passes are observations THIS experiment invented, so with no
        # explicit passes: list they run alongside the real ones; the line's
        # own synthetics stay opt-in (design studies, not defaults)
        order = list(ORDER) + list(extra_passes or {})
    # A segment is a WINDOW on the line, and not every flight reaches every
    # window. Passes that do not cover this one are dropped with a note
    # rather than failing: the alternative is a line that can only have
    # windows its shortest flight covers.
    absent = [k for k in order if segment not in PASSES[k]]
    if absent:
        print(f"  segment {segment!r}: no data for {absent} (that flight "
              "does not reach this window) -- skipped", flush=True)
        order = [k for k in order if k not in absent]
    if not order:
        raise ValueError(f"no pass covers segment {segment!r} on line "
                         f"{LINE!r}")
    n_traces = n_traces or N_TRACES_BY_SEGMENT[segment]
    ts_km = (DECOMP_S_KM[segment] if trace_decomp_s_km is None
             else trace_decomp_s_km)
    ts_km = [float(v) for v in np.atleast_1d(ts_km)]
    tag = case_tag(picked_bed, gamma_rssnr, proc, demogorgn_bed)
    out = Path(out_root or OUT_DEFAULT) / (out_name or (segment + tag))
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"
    # The RSSNR acceptance analysis compares against a CONSTANT-gamma arm of
    # the same scene. That arm now lives in THIS run's own cache directory
    # rather than in a separately-named experiment: its chunk rid differs by
    # the gamma tag, so the two cannot collide, and an experiment no longer
    # has to know another experiment's directory name to be runnable. On a
    # warm cache the arm costs nothing; on a cold one it is the same total
    # work that was previously paid by a separate run.
    runs_const = runs_dir if (gamma_rssnr and companion) else None
    case = f"{CASE_PREFIX}_{out_name or (segment + tag)}"
    axis = (ref_bed_picks() if (picked_bed or gamma_rssnr or hybrid)
            else None)
    ref = axis if picked_bed else None
    if picked_bed:
        print(f"picked bed: reference pass {ref['pass']} "
              f"({'/'.join(ref['frames'])}), {ref['n']} picks over "
              f"{ref['line_len_km']} km, line gap frac "
              f"{ref['gap_frac_line']:.4f}", flush=True)
    gmap = None
    if gamma_rssnr:
        if gamma_surface_db is None:
            gamma_surface_db = manual_gamma_surface_db()
            print(f"gamma_surface = {gamma_surface_db:+.2f} dB (line "
                  f"calibration; anomaly vs Fresnel "
                  f"{gamma_surface_db - gamma_surface_fresnel_db():+.2f} dB)",
                  flush=True)
        gmap = build_rssnr_gamma(axis, segment, att, gamma_surface_db,
                                 bed_rough_sigma=bed_rough[0] if bed_rough
                                 else None,
                                 extra_db=bed_rough_extra_db,
                                 zone_gl_km=GL_S_KM if hybrid else None)
        if bed_rough:
            g = gmap["bed_rough_guard"]
            print(f"bed roughness: sigma {g['sigma_m']} m, l "
                  f"{bed_rough[1]} m; nadir coherent attenuation "
                  f"{g['nadir_coherent_attenuation_db']} dB -> G2 raised "
                  f"{g['g2_shift_db']} dB (double-count guard)", flush=True)
        print(f"rssnr gamma: {gmap['n_samples']} samples "
              f"({gmap['provenance']['source']}), snapshot {RSSNR_SNAPSHOT}, "
              f"K {gmap['k_db']} dB (K - K_phys "
              f"{gmap['k_minus_kphys_db']} dB), segment G2 "
              f"[{gmap['g2_seg_db']['min']} .. {gmap['g2_seg_db']['max']}] "
              f"dB (med {gmap['g2_seg_db']['med']}), censored "
              f"{gmap['n_censored']}/{gmap['n_samples']}", flush=True)
    preps, sims, analyses, procs = {}, {}, {}, {}
    zone_results, pass_figs = {}, []
    main_slug = ("picked_bed" if picked_bed else
                 "demogorgn" if demogorgn_bed else "bedmachine")
    for key in order:
        print(f"== {key} ({segment}{tag}) ==", flush=True)
        p = prep_pass(key, segment, n_traces, ref=ref, gmap=gmap, axis=axis,
                      fine_posting=proc, posting_div=posting_div,
                      spec_diffuse=spec,
                      dgn_seed=demogorgn_seed if demogorgn_bed else None,
                      hybrid=hybrid, instrument=instruments.get(key))
        if p["aux"]["demogorgn"]:
            d = p["aux"]["demogorgn"]
            print(f"  DEMOGORGN bed: seed {d['seed_id']}, snapshot "
                  f"{d['snapshot_id']}, clamp {d['bed_clamp_frac']:.4f}",
                  flush=True)
            if "hybrid" in d:
                h = d["hybrid"]
                print(f"  HYBRID bed: GL {h['gl_s_km']} km, ramp "
                      f"{h['ramp_km']} km; blend-zone DGN-picks "
                      f"med {h['blend_zone_dgn_minus_picks_m']['med']} / rms "
                      f"{h['blend_zone_dgn_minus_picks_m']['rms']} m; "
                      f"clearance min grounded "
                      f"{h['clearance_m']['grounded_min']} / floating "
                      f"{h['clearance_m']['floating_min']} m (clamp "
                      f"g {h['clearance_m']['clamp_frac_grounded']:.5f} / "
                      f"f {h['clearance_m']['clamp_frac_floating']:.5f}); "
                      f"floating picks n {h['floating_picks']['n']}, gaps "
                      f"{h['floating_picks']['gap_frac']:.4f}", flush=True)
        if p["synthetic"]:
            print(f"  SYNTHETIC pass: {p['synthetic']['synthetic_msl_m']:.0f}"
                  f" m constant ellipsoidal height, roll 0, AGL med "
                  f"{p['synthetic']['agl_med_m']:.0f} m", flush=True)
        if p["aux"]["picked_bed"]:
            pb = p["aux"]["picked_bed"]
            print(f"  picked bed: residual rms {pb['residual_rms_m']} m "
                  f"(mean {pb['residual_mean_m']}, |max| "
                  f"{pb['residual_absmax_m']}), gaps "
                  f"{pb['gap_frac_segment']:.4f}; along-track bed roughness "
                  f"{pb['bed_roughness_rms_m']['dem']} -> "
                  f"{pb['bed_roughness_rms_m']['picked']} m rms", flush=True)
        print(f"  reach: surface {p['reach']['surface_reach_m']:.0f} m, bed "
              f"{p['reach']['bed_reach_m']:.0f} m -> ct "
              f"±{p['reach']['ct_m']:.0f} m; spacing {p['spacing']:.2f} m; "
              f"n_traces {len(p['idx'])}; "
              f"n_samples_sim {p['rc_sim'].n_samples}", flush=True)
        preps[key] = p
        sims[key] = simulate_pass(p, runs_dir, att, surf_rough, force,
                                  antenna=antenna, bed_rough=bed_rough,
                                  spec=spec, gfix=grazing_fix)
        if proc:
            procs[key] = (process_standard_cached(p, sims[key], out, att,
                                                  surf_rough, force=force,
                                                  antenna=antenna,
                                                  bed_rough=bed_rough,
                                                  spec=spec,
                                                  gfix=grazing_fix,
                                                  focus_aperture=focus_aperture)
                          if proc_cache else process_standard(
                              p, sims[key], focus_aperture=focus_aperture))
            ch = procs[key]["chain"]
            print(f"  processed: aperture {ch['aperture_m']:.0f} m "
                  f"({ch['aperture_traces']} traces, half-angle "
                  f"{ch['half_angle_deg']:.2f} deg), {ch['n_looks_sim']} "
                  f"looks, mocomp dz rms {ch['mocomp']['dz_rms_m']} m",
                  flush=True)
        analyses[key] = analyze_pass(p, sims[key], proc=procs.get(key),
                                     trace_s_km=ts_km)
        for ti in analyses[key]["trace_info_list"] or []:
            print(f"  single-trace decomposition at s = {ti['sim_s_km']:.2f}"
                  f" km (sim trace {ti['sim_trace_index']}"
                  + (f", measured trace {ti['measured_trace_index']}"
                     if "measured_trace_index" in ti else "")
                  + f"): bed-window bed - surface returns "
                  f"{ti['bed_window_bed_minus_surface_returns_db']:+.1f} dB",
                  flush=True)
        if hybrid:
            zone_results[key] = zone_analysis(preps[key], analyses[key])
        if per_pass_figs:
            # STAGED DELIVERY: this pass's complete figure set, written the
            # moment its sim+processing+analysis is done; the marker line is
            # the coordinator's pickup signal.
            pass_figs += emit_pass_figs(
                out, key, preps[key], analyses[key], zone_results.get(key),
                segment, GL_S_KM if hybrid else None, plot_s_max_km,
                main_slug)
            print(f"FIGSET_READY {key}", flush=True)

    # ---- RSSNR-gamma acceptance: vs the constant-gamma companion run ----
    corr_stats = corr_series = None
    if gamma_rssnr and companion:
        corr_stats, corr_series = {}, {}
        for key in [k for k in ORDER if k in order]:
            print(f"== {key} constant-gamma companion (cache-first) ==",
                  flush=True)
            p_const = dict(preps[key])
            p_const["gamma_rssnr"] = False
            sim_c = simulate_pass(p_const, runs_const, att, surf_rough,
                                  False, gfix=grazing_fix)
            proc_c = process_standard(preps[key], sim_c) if proc else None
            a_const = analyze_pass(preps[key], sim_c, proc=proc_c)
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

    # ---- bed-source ablation rows: BEDMACHINE and DEMOGORGN beds ----
    # (identical RSSNR gamma + processing; only the bed topography changes)
    ab_rows = None
    if bed_ablation:
        ab_rows = []
        for label, seed in (("BedMachine bed", None),
                            (f"DEMOGORGN bed, seed {demogorgn_seed}",
                             demogorgn_seed)):
            runs_ab = (Path(out_root or OUT_DEFAULT)
                       / (segment + case_tag(False, gamma_rssnr, proc,
                                             seed is not None)) / "runs")
            pr, an, sm = {}, {}, {}
            for key in order:
                print(f"== {key} {label} ablation (cache-first) ==",
                      flush=True)
                p_ab = prep_pass(key, segment, n_traces, ref=None, gmap=gmap,
                                 axis=axis, fine_posting=proc, dgn_seed=seed)
                s_ab = simulate_pass(p_ab, runs_ab, att, surf_rough, force,
                                     gfix=grazing_fix)
                proc_ab = process_standard(p_ab, s_ab) if proc else None
                pr[key], sm[key] = p_ab, s_ab
                an[key] = analyze_pass(p_ab, s_ab, proc=proc_ab)
            ab_rows.append((pr, an, label, sm))

    # ---- metrics ----
    rec = "recorded only"
    metrics = {}
    for key in order:
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
            "measured_floor_window": a.get("floor_doc"),
            "measured_midcol_noise_limited": a["meas_noise_limited"],
            "agl_med_m": round(p["h_med"], 0),
            "note": "mid-column mean power (surf+1.0 -> bed-0.5 us) rel own "
            "surface peak, median over traces; bed window bed-0.5 -> "
            "bed+1.5 us; scout_midcol_over_bedpeak matches the scout table "
            "metric (mean bed-3.0..bed-0.6 us over bed peak +-0.3 us). "
            "decomposition_db: same windows on the per-interface coherent "
            "fields (surface returns vs bed returns). measured floor: deep "
            "record tail (last 0.2-3.2 us; pre-surface is TX-leakage/"
            "img_comb-contaminated on the low pass). " + rec}
    # headline: altitude trend of mid-column clutter, sim vs measured.
    # Derived from the line's ALTITUDE ORDERING, not from pass names: the
    # old hardcoded pass-name literals silently skipped this metric --
    # the line's KEY DELIVERABLE -- the moment the passes were renamed. The
    # lowest measured pass present is the baseline; every higher measured
    # pass pairs against it. A line whose real passes all fly one altitude
    # (the westcoast instrument line) produces no pairs and no metric, which
    # is correct: there is no altitude trend to report there.
    meas_keys = [k for k in ORDER if k in order
                 and analyses[k]["meas"] is not None]
    meas_keys.sort(key=lambda k: preps[k]["h_med"])
    trend = {}
    if len(meas_keys) >= 2:
        lo_k = meas_keys[0]
        for hi in [k for k in meas_keys[1:]
                   if preps[k]["h_med"] > 2.0 * preps[lo_k]["h_med"]]:
            pair = f"{hi}-{lo_k}"
            trend[pair] = {
                "measured_db": round(
                    analyses[hi]["meas"]["midcol_rel_surf_db"]
                    - analyses[lo_k]["meas"]["midcol_rel_surf_db"], 2),
                "sim_db": round(
                    analyses[hi]["sim"]["midcol_rel_surf_db"]
                    - analyses[lo_k]["sim"]["midcol_rel_surf_db"], 2)}
            trend[pair]["error_db"] = round(
                trend[pair]["sim_db"] - trend[pair]["measured_db"], 2)
    if trend:
        head = f"{meas_keys[-1]}-{meas_keys[0]}"
        metrics["altitude_trend"] = {
            "value": trend[head]["sim_db"], "threshold": None,
            "op": "record", "pass": True, "pairs": trend,
            "baseline_pass": meas_keys[0],
            "note": "KEY DELIVERABLE: mid-column clutter power delta (dB, "
            "rel own surface peaks -- gain-free) high/mid pass minus low "
            "pass, sim vs measured; the scout's measured whole-line value is "
            "~+20 dB. If the low pass's measured mid-column is noise-limited "
            "its measured delta is a LOWER bound. " + rec}
    metrics["simulation_wall_s"] = {
        "value": round(sum(s["wall_s"] for s in sims.values()), 1),
        "threshold": None, "op": "record", "pass": True,
        "per_pass_s": {k: round(sims[k]["wall_s"], 1) for k in order},
        "note": rec}
    for skey in [k for k in SYNTHETIC_KEYS if k in order]:
        asy = analyses[skey]
        dsy = asy["decomposition"]
        bed_over_clutter = round(dsy["bed"]["bed_rel_surf_db"]
                                 - dsy["surface"]["bed_rel_surf_db"], 2)
        ps = preps[skey]
        metrics[f"{skey}_bed_visibility"] = {
            "value": bed_over_clutter, "threshold": None, "op": "record",
            "pass": True,
            "bed_over_surface_clutter_in_bed_window_db": bed_over_clutter,
            "bedpeak_over_midcol_db": round(
                -asy["sim"]["scout_midcol_over_bedpeak_db"], 2),
            "bed_rel_surf_db": asy["sim"]["bed_rel_surf_db"],
            "midcol_rel_surf_db": asy["sim"]["midcol_rel_surf_db"],
            "decomposition_db": dsy, "midcol_verdict": asy["verdict"],
            "agl_med_m": round(ps["h_med"], 0),
            "geometry": {"ct_reach_m": round(ps["reach"]["ct_m"], 0),
                         "facet_spacing_m": round(ps["spacing"], 2),
                         "t0_us": round(ps["rc_sim"].t0 * 1e6, 2),
                         "n_samples_sim": ps["rc_sim"].n_samples,
                         **({"aperture_m": procs[skey]["chain"]["aperture_m"],
                             "aperture_traces":
                                 procs[skey]["chain"]["aperture_traces"]}
                            if skey in procs else {})},
            "note": f"KEY DELIVERABLE ({skey} prediction, clutter-limited "
            "analysis -- no receiver-noise model and no link budget): "
            "bed-return minus surface-return power in the BED window "
            "(median dB; > 0 means the bed beats the surface clutter "
            "arriving at the same delay), plus bed peak over mid-column "
            "clutter (the scout contrast metric, sign-flipped). Same scene, "
            "reflectivity model and processing as the measured passes of "
            "this run. " + rec}
    if bed_ablation:
        for pr, an, label, sm in ab_rows:
            slug = "demogorgn" if "DEMOGORGN" in label else "bedmachine"
            for key in order:
                a_ab, a_pb = an[key], analyses[key]
                e = {"value": a_ab["sim"]["bed_rel_surf_db"],
                     "threshold": None, "op": "record", "pass": True,
                     f"sim_{slug}": a_ab["sim"],
                     "sim_picked_bed": a_pb["sim"],
                     f"decomposition_{slug}_db": a_ab["decomposition"],
                     "nadir_bed_offset_vs_picks":
                         nadir_bed_offset(pr[key], sm[key]),
                     "note": f"bed-source ABLATION row ({label}) with "
                     "IDENTICAL RSSNR gamma + processing; compare "
                     "sim_picked_bed (row 2) and the measured metrics in "
                     f"clutter_{key}. " + rec}
                if slug == "demogorgn":
                    e["provenance"] = pr[key]["aux"]["demogorgn"]
                if a_pb["meas_bed_prof_db"] is not None:
                    p = preps[key]
                    s_sim = p["s_sim"]
                    meas = np.interp(s_sim, p["s_m"], _smooth_db(
                        p["s_m"], a_pb["meas_bed_prof_db"]))
                    e["r_bed_brightness_vs_measured"] = {
                        slug: round(_pearson(_smooth_db(
                            pr[key]["s_sim"],
                            a_ab["sim_bed_prof_db"]), meas), 3),
                        "picked_bed": round(_pearson(_smooth_db(
                            s_sim, a_pb["sim_bed_prof_db"]), meas), 3)}
                metrics[f"{slug}_bed_ablation_{key}"] = e
    # ---- bed-return tail: decay past the bed echo, sim vs measured ----
    ab_by_slug = [("demogorgn" if "DEMOGORGN" in label else "bedmachine", an)
                  for _, an, label, _ in (ab_rows or [])]
    for key in order:
        sources = ([(main_slug, analyses[key])]
                   + [(slug, an[key]) for slug, an in ab_by_slug])
        e = bed_tail_entry(key, preps[key], analyses[key], sources)
        metrics[f"bed_return_tail_{key}"] = e
        sl = e["bed_return_tail_slope_db_per_us"]
        print(f"  bed-return tail {key}: slope dB/us " + ", ".join(
            f"{k} {v:+.2f}" for k, v in sl.items() if v is not None)
            + ("; excess at +2 us " + ", ".join(
                f"{k} {v['+2us']:+.1f}" for k, v in
                e["bed_return_tail_excess_db"].items())
               if e["bed_return_tail_excess_db"] else "")
            + "; guard " + ", ".join(
                f"{k} {'ok' if v['guard']['pass'] else 'FAIL'} "
                f"({v['guard']['min_bed_minus_surface_returns_db']:+.1f} dB)"
                for k, v in e["sim"].items()), flush=True)
    # ---- ZONE SPLIT (full_line): grounded vs floating sub-windows ----
    # (zone_results filled in the pass loop, right after each analysis)
    if hybrid:
        for key in order:
            zres = zone_results[key]
            zmet = {zn: zres[zn]["metrics"] for zn in zres
                    if "metrics" in zres[zn]}
            fres = zmet.get("floating", {}).get("bed_window_residual_db")
            metrics[f"zone_split_{key}"] = {
                "value": float("nan") if fres is None else fres,
                "threshold": None, "op": "record",
                "pass": True, "gl_s_km": GL_S_KM, **zmet,
                "note": "KEY DELIVERABLE (zone split): every standard metric "
                "computed separately over the grounded (s < GL) and "
                "floating (s > GL) traces -- windows, decomposition "
                "(surface returns vs bed returns), bed-referenced tail "
                "slope/excess/guard, and the bed-window level residual "
                "(sim - measured, dB). 'value' is the FLOATING bed-window "
                "residual: does the fixed K (+7.92 dB, anchored on the "
                "grounded 50 km segment) reproduce the shelf-base "
                "brightness -- the specular-regime test. recorded only"}
            for zn, zm in zmet.items():
                if "sim" not in zm:
                    continue        # too-few-traces zone: recorded, no print
                res = zm.get("bed_window_residual_db")
                g = zm["tail"]["guard"]
                print(f"  zone {key}/{zn}: bed window sim "
                      f"{zm['sim']['bed_rel_surf_db']:+.1f}"
                      + (f" meas {zm['measured']['bed_rel_surf_db']:+.1f} "
                         f"(residual {res:+.2f} dB)"
                         if "measured" in zm else "")
                      + f"; tail slope sim "
                      f"{zm['tail']['sim_slope_db_per_us']:+.2f}"
                      + (f" meas {zm['tail']['meas_slope_db_per_us']:+.2f}"
                         if "meas_slope_db_per_us" in zm["tail"] else "")
                      + f"; guard "
                      f"{'ok' if g['pass'] else 'FAIL'} "
                      f"({g['min_bed_minus_surface_returns_db']:+.1f} dB)",
                      flush=True)
        for key in order:
            h = preps[key]["aux"]["demogorgn"]
            metrics[f"hybrid_bed_{key}"] = {
                "value": h["hybrid"]["clearance_m"]["min"],
                "threshold": 0.0, "op": ">=",
                "pass": bool(h["hybrid"]["clearance_m"]["min"] >= 0.0
                             or h["bed_clamp_frac"] < 1e-3),
                **{k: h[k] for k in ("seed_id", "snapshot_id",
                                     "nodata_fill_frac", "bed_clamp_frac")},
                **h["hybrid"],
                "note": "HYBRID bed guard: min (REMA surface - hybrid bed) "
                "clearance over the scene, per zone; the existing clamp "
                "(bed <= surface - 0.1 m) is the enforcement and its "
                "fraction is recorded. blend_zone_dgn_minus_picks_m is the "
                "DEMOGORGN-vs-picks step the GL ramp absorbs. recorded"}
        if gamma_rssnr and "g2_zones_db" in gmap:
            metrics["rssnr_zone_physicality"] = {
                "value": gmap["g2_zones_db"]["floating"].get(
                    "med", float("nan")),
                "threshold": None, "op": "record", "pass": True,
                **gmap["g2_zones_db"]}
        cov = {}
        for key in [k for k in order if analyses[k]["meas"] is not None]:
            p = preps[key]
            skm = S0_KM[segment] + p["s_m"] / 1e3
            cov[key] = {
                zn: {"n_traces": int(m.sum()),
                     "bottom_pick_frac": round(float(
                         np.isfinite(p["bot"][m]).mean()), 5)}
                for zn, m in (("grounded", skm < GL_S_KM),
                              ("floating", skm >= GL_S_KM))}
        if hybrid and (cov or gamma_rssnr):
            metrics["zone_qc_coverage"] = {
                "value": (min([z["bottom_pick_frac"] for c in cov.values()
                               for z in c.values()], default=float("nan"))),
                "threshold": None, "op": "record",
                "pass": True, "picks_per_pass": cov,
                **({"rssnr_zones": {
                    zn: {k: gmap["g2_zones_db"][zn][k]
                         for k in ("n", "n_total", "qc_pass_frac", "s_km")
                         if k in gmap["g2_zones_db"][zn]}
                    for zn in ("grounded", "floating")}}
                   if gamma_rssnr and "g2_zones_db" in gmap else {}),
                "note": "QC coverage over the zones: bottom-pick coverage "
                "of each measured pass (the floating stretch's picks are "
                "the hybrid bed's floating source) and the QC-passing "
                "RSSNR sample count per zone (the reflectivity mapping's "
                "along-track support). recorded only"}
    if gamma_rssnr:
        # POST-RUN LEVEL RESIDUALS: the physical-chain closure evidence.
        # Under a manual gamma_surface these are pure diagnostics; under
        # 'solve' the QUALIFYING median is the one number main_config's
        # seed/verify loop feeds back -- everything else stays recorded,
        # never absorbed.
        res = {k: round(analyses[k]["sim"]["bed_rel_surf_db"]
                        - analyses[k]["meas"]["bed_rel_surf_db"], 2)
               for k in order if analyses[k]["meas"]}
        med = float(np.median(list(res.values()))) if res else float("nan")
        # GAMMA-SOLVE SUPPORT: per-pass gamma_required from the power-sum
        # inversion (clutter-aware, exact); the solve driver consumes the
        # qualifying median. The bed-over-surface margin stays recorded as
        # the window-domination diagnostic.
        marg = {k: round(analyses[k]["decomposition"]["bed"]["bed_rel_surf_db"]
                         - analyses[k]["decomposition"]["surface"][
                             "bed_rel_surf_db"], 2)
                for k in res}
        per, qual, greq_med, greq_spread = gamma_solve_required(
            gmap["gamma_surface_db"],
            {k: analyses[k]["meas"]["bed_rel_surf_db"] for k in res},
            {k: analyses[k]["decomposition"]["surface"]["bed_rel_surf_db"]
             for k in res},
            {k: analyses[k]["decomposition"]["bed"]["bed_rel_surf_db"]
             for k in res},
            GAMMA_SURFACE_SOLVE["min_headroom_db"])
        metrics["rssnr_level_residuals"] = {
            "value": round(med, 2), "threshold": None, "op": "record",
            "pass": True,
            "median_residual_db": round(med, 2),
            "per_pass_residual_db": res,
            "per_pass_bed_over_surface_margin_db": marg,
            "gamma_surface_db": gmap["gamma_surface_db"],
            "surface_anomaly_db": gmap["surface_anomaly_db"],
            # gamma_solve: the power-sum inversion a 'gamma_surface_db:
            # solve' calibration feeds back (main_config drives the loop);
            # under a manual calibration it stays a pure diagnostic. If the
            # required gamma is stable per line it is a candidate real
            # surface property; if it moves with configuration it is a
            # chain artifact; if the qualifying passes DISAGREE (spread)
            # the sim is missing physics in some regime -- e.g. englacial
            # scattering brightening the measured window.
            "gamma_solve": {
                "per_pass": per,
                "qualifying_passes": qual,
                "min_headroom_db": GAMMA_SURFACE_SOLVE["min_headroom_db"],
                "gamma_required_median_db": round(greq_med, 2),
                "gamma_required_spread_db": greq_spread},
            # the naive total-field zeroing gamma, kept as a diagnostic
            "gamma_surface_level_match_db": round(
                gmap["gamma_surface_db"] - med, 2),
            "caveat": "per-pass residuals use the TOTAL-field bed window, "
            "which is surface-clutter contaminated on high-altitude passes; "
            "the gamma_solve block subtracts the modeled clutter floor "
            "before reading the bed level",
            "implied_reflectivity": {
                "g2_seg_db": gmap["g2_seg_db"],
                "g2_pos_frac_seg": gmap["g2_pos_frac_seg"]},
            "note": "per-pass simulated minus measured bed-window level (dB "
            "rel own surface peak) under the line calibration. NOT a gated "
            "solve: the mapping is anchoring-free, so this residual is the "
            "physical-chain closure evidence. recorded only"}
    if gamma_rssnr:
        metrics["rssnr_gamma_mapping"] = {
            "value": gmap["k_db"], "threshold": None, "op": "record",
            "pass": True,
            **{k: gmap[k] for k in
               ("k_db", "k_phys_db", "k_minus_kphys_db", "g2_const_db",
                "g2_seg_db", "n_samples", "n_seg", "n_censored",
                "censored_floor_db", "seg_s_km", "med_sample_spacing_m",
                "att_db_per_km", "g2_pos_frac_seg",
                "gamma_surface_db", "gamma_surface_fresnel_db",
                "surface_anomaly_db")},
            "snapshot_id": RSSNR_SNAPSHOT,
            "note": "anchoring-free mapping |Gamma_bed|^2 = 2*A*H - RSSNR "
            "+ (gamma_surface - T^2), gamma_surface per the line calibration "
            "(manual or residual-solved), A per "
            "the line calibration. surface_anomaly_db = gamma_surface - "
            "smooth-Fresnel is the chain diagnostic (old K - K_phys). "
            + rec}
        if corr_stats is not None:
            metrics["bed_brightness_correlation"] = {
                "value": round(float(np.mean(
                    [corr_stats[k]["r_sim_rssnr_vs_measured"]
                     for k in corr_stats])), 3),
                "threshold": None, "op": "record", "pass": True,
                "per_pass": corr_stats,
                "note": "KEY DELIVERABLE (acceptance): along-track Pearson "
                "r of the ~1 km-smoothed bed-window power profile (dB rel "
                "own surface peak) between sim and MEASURED, RSSNR-driven "
                "vs constant bed gamma (same bed geometry). "
                "r_bedlayer_rssnr_vs_implied is the by-construction sanity "
                "check (bed returns layer only -- geometry/speckle-limited); "
                "r_sim_rssnr_vs_implied uses the TOTAL field, whose bed "
                "window is surface-clutter-crowded at altitude (the "
                "study's own finding), so it is expected to degrade "
                "low->mid->high; r_implied_vs_measured is the data-only "
                "ceiling estimate. " + rec}
    if demogorgn_bed:
        for key in order:
            metrics[f"dgn_nadir_bed_offset_{key}"] = {
                "value": nadir_bed_offset(preps[key],
                                          sims[key])["med_us"],
                "threshold": None, "op": "record", "pass": True,
                **nadir_bed_offset(preps[key], sims[key]),
                "provenance": preps[key]["aux"]["demogorgn"],
                "note": "DEMOGORGN nadir-bed offset vs this pass's own "
                "radar pick (thickness-convention misfit, "
                "scout-documented): reported, not tuned away. " + rec}

    config = {
        "case": case, "line": LINE, "segment": segment,
        "n_traces": n_traces,
        "att_db_per_km": att, "surf_rough": surf_rough,
        "surface_roughness": surface_roughness_record(surf_rough, preps),
        "margin_us": MARGIN_US, "post_bed_window_us": POST_BED_US,
        "chunk_m": CHUNK_M, "chunk_trace_facets": CHUNK_TRACE_FACETS,
        "picked_bed": bool(picked_bed),
        "gamma_rssnr": bool(gamma_rssnr),
        "demogorgn_bed": bool(demogorgn_bed),
        "antenna": antenna,
        "posting_div": posting_div,
        "focus_aperture": focus_aperture,
        "grazing_fix": (None if grazing_fix is None else {
            "s_eff": grazing_fix,
            "model": "Grazing-angle facet-lattice fix (GrazingFixConfig): "
                     "coherent specular facet FIELD tapered by "
                     "exp(-tan^2(alpha)/(2 s_eff^2)) off the facet normal "
                     "(sub-facet slope spread; removed sinc/grid-lobe "
                     "aliasing power is dropped, not re-booked), and the "
                     "sub-facet-roughness D_Phi reduced to its facet-area "
                     "term (per-facet infinite-surface PO law) -- effective "
                     "sigma0 is then facet-size invariant at grazing"}),
        "spec_diffuse": (None if not spec else {
            "specular_fraction": spec[0], "spec_tilt_s0_deg": spec[1],
            "diffuse_exponent": spec[2], "model": SPEC_DIFFUSE_NOTE,
            **((preps[order[0]]["aux"]["rssnr_gamma"] or {}).get(
                "spec_diffuse", {}) if preps else {})}),
        "bed_roughness": (None if not bed_rough else {
            "sigma_m": bed_rough[0], "corr_length_m": bed_rough[1],
            "interface": "bed only (the surface keeps its own "
                         "representative roughness)",
            "gerekos_validity": bed_rough_validity(),
            "gamma_double_count_guard": gmap["bed_rough_guard"]
            if gamma_rssnr else None}),
        "trace_decomp_s_km": ts_km,
        "per_pass_figs": bool(per_pass_figs),
        "plot_s_max_km": plot_s_max_km,
        "fig_width_scale": FIG_WIDTH_SCALE,
        "segment_s_km": [round(v / 1e3, 2)
                         for v in segment_s_range(axis, segment)]
        if axis else None,

        "hybrid_bed": ({**preps[order[0]]["aux"]["demogorgn"]["hybrid"],
                        "applies": "all passes of this run (one hybrid "
                        "construction per pass scene)"} if hybrid else None),
        "passes": {}, "measured_caveats": MEASURED_CAVEATS}
    _resolved, _overrides = LINE_ANALYSIS[LINE]
    config["analysis"] = {
        **_resolved.model_dump(),
        **({"line_overrides": _overrides,
            "note": f"line {LINE!r} overrides these study-wide conventions "
                    "(config/analysis.yaml); every other value is the study "
                    "default"} if _overrides else
           {"note": "study-wide conventions (config/analysis.yaml), "
                    "unmodified by this line"})}
    if demogorgn_bed:
        config["demogorgn"] = {**preps[order[0]]["aux"]["demogorgn"],
                               "license": "NONE FOUND -- internal "
                               "engineering use only until the Gator "
                               "Glaciology group provides one (scout "
                               "section 7)"}
    if gamma_rssnr:
        config["rssnr_gamma"] = {
            k: gmap[k] for k in
            ("provenance", "k_db", "k_phys_db", "k_minus_kphys_db",
             "gamma_surface_db", "gamma_surface_fresnel_db",
             "surface_anomaly_db",
             "g2_const_db", "g2_seg_db", "n_samples", "n_seg", "n_censored",
             "censored_floor_db", "seg_s_km", "med_sample_spacing_m",
             "att_db_per_km", "g2_pos_frac_seg", "note")}
        config["rssnr_gamma"]["interpolation"] = (
            "linear in anchor along-track s (np.interp, edge-clamped), "
            "cross-track constant; H(x) from the DATASET's surface/bed "
            "twtts (self-consistent with its RSSNR), not the DEM")
        config["rssnr_gamma"]["calibration"] = dict(CALIBRATION)
        if "g2_zones_db" in gmap:
            config["rssnr_gamma"]["g2_zones_db"] = gmap["g2_zones_db"]
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
    for key in order:
        p, s = preps[key], sims[key]
        config["passes"][key] = {
            "parts": [[fid, list(sl)] for fid, sl in p["parts"]],
            "reversed": p["rev"], "roll_note": p["roll_note"],
            "param_frame": PASSES[key]["param_frame"],
            "n_traces_measured": int(len(p["surf"])),
            "n_traces_sim": int(len(p["idx"])),
            "posting_div": p["posting_div"],
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
            "instrument": p["instrument_provenance"],
            "dropped_power_fraction": s["dropped_power_fraction"],
            "surf_fill_frac": p["aux"]["surf_fill"],
            "bed_clamp_frac": p["aux"]["clamp_frac"],
            "picked_bed": p["aux"]["picked_bed"],
            "synthetic": p["synthetic"]}
        if proc:
            config["passes"][key]["processing"] = procs[key]["chain"]
        if analyses[key]["trace_info"]:
            til = analyses[key]["trace_info_list"]
            config["passes"][key]["trace_decomposition"] = \
                til if len(til) > 1 else til[0]
    if segment == "pilot":
        config["full_projection"] = {
            k: {"wall_s_projected": round(sims[k]["wall_s"] * 5.0, 1),
                "basis": "5x pilot wall (50/10 km at fixed trace spacing; "
                "5 chunks of the pilot's exact geometry; per-chunk JAX "
                "recompile risk if chunk shapes differ -- pilot wall "
                "already includes one compile)"} for k in order}
        config["full_projection"]["total_s"] = round(
            5.0 * sum(s["wall_s"] for s in sims.values()), 1)

    notes = (
        f"Basal clutter on line {LINE} ({segment}): passes "
        f"{', '.join(order)}. COHERENT SURFACE+BED ONLY (no firn/internal "
        "layers by design); the per-interface field decomposition "
        "identifies which interface supplies the clutter. Cross-track "
        "reach derived per pass out to nadir-bed delay + "
        f"{MARGIN_US:.0f} us for both interfaces (bed reach includes Snell "
        "refraction); reversed high passes' roll negated; per-pass surface "
        "registration; BedMachine 500 m texture caveat applies. BED-RETURN "
        "TAIL (bed_return_tail_*): robust slope of the bed-referenced "
        "ensemble mean-power profile over bed+"
        f"{TAIL_FIT_US[0]:g}..+{TAIL_FIT_US[1]:g} us and the sim-minus-"
        "measured excess at bed+1/2/3 us, with a guard that the sim tail is "
        "bed returns (not surface returns) and a caveat on whether the "
        "measured tail is noise-floor-limited. "
        + MEASURED_CAVEATS
        + (" PICKED BED: " + PICKED_BED_NOTE if picked_bed else "")
        + ((" HYBRID BED: " + HYBRID_BED_NOTE
            + f" Blend ramp {GL_RAMP_KM:g} km past the GL at "
            f"s = {GL_S_KM:g} km; zone-split metrics (zone_split_*) judge "
            "the grounded and floating traces separately, and the implied "
            "reflectivity is checked against each zone's own Fresnel "
            "ceiling (rock anchor vs ice-seawater).") if hybrid
           else (" DEMOGORGN BED: " + DGN_NOTE if demogorgn_bed else ""))
        + (" RSSNR GAMMA: " + RSSNR_GAMMA_NOTE if gamma_rssnr else "")
        + (" PROCESSING: CSARP_standard-matching chain applied identically "
           "to every simulated pass (measured panels are already the "
           "standard product); the real chain, our chain and the recorded "
           "gap list g1-g6 are in each pass's config 'processing' block."
           if proc else "")
        + ("".join(
            f" {k.upper()}: synthetic constant-ellipsoidal-height pass at "
            f"{PASSES[k]['synthetic_msl_m'] / 1e3:g} km on the same line and "
            "system parameters; reach, facet spacing and alias-limited "
            "aperture all follow the geometry (prediction only -- no "
            "measured data exists there)."
            for k in order if k in SYNTHETIC_KEYS)))
    config["config_schema_version"] = CONFIG_SCHEMA_VERSION
    config["provenance"] = run_provenance(spec_doc=spec_doc,
                                          spec_path=spec_path)
    fp = config_fingerprint(config)
    config["config_fingerprint"] = fp
    doc = {"case": case, "group": "xOPR clutter",
           "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "config_schema_version": CONFIG_SCHEMA_VERSION,
           "config_fingerprint": fp,
           "metrics": metrics, "notes": notes}
    doc = merge_metrics_doc(out / "metrics.json", doc, order)
    (out / "metrics.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out / "run_config.json").write_text(json.dumps(config, indent=1) + "\n")

    if bed_ablation:
        for old, keep in (("radargrams.png", "radargrams_tworow.png"),
                          ("decomposition.png", "decomposition_pbed.png")):
            if (out / old).exists() and not (out / keep).exists():
                shutil.copy2(out / old, out / keep)  # pre-ablation version
    ab_fig = ([(pr, an, label) for pr, an, label, _ in ab_rows]
              if bed_ablation else None)
    if per_pass_figs:
        # staged-delivery mode: the suffixed per-pass sets (already emitted
        # in the loop, marker-lined) ARE the figure deliverable; the
        # unsuffixed combined figures are skipped so the two styles cannot
        # drift apart within one run.
        figs = list(pass_figs)
    else:
        figs = [fig_radargrams(out, preps, analyses, segment, keys=order,
                               ablation=ab_fig,
                               gl_s_km=GL_S_KM if hybrid else None,
                               w_scale=FIG_WIDTH_SCALE,
                               plot_s_max_km=plot_s_max_km,
                               bed_overlay=BED_OVERLAY),
                fig_decomposition(out, preps, analyses, keys=order,
                                  ablation=ab_fig),
                fig_bed_tail(out, preps, analyses, metrics, keys=order,
                             ablation=ab_fig)]
        ftr = fig_decomposition_trace(out, preps, analyses, keys=order)
        if ftr is not None:
            figs.insert(2, ftr)
        for key in [k for k in order if k in zone_results]:
            fz = fig_decomposition_zones(
                out, key, zone_results[key],
                fname=("decomposition_zones.png" if len(order) == 1
                       else f"decomposition_zones_{key}.png"))
            if fz is not None:
                figs.append(fz)
    if gamma_rssnr and corr_stats is not None:
        syn = None
        syn_key = next((k for k in order if k in SYNTHETIC_KEYS), None)
        if syn_key is not None:
            p30, a30 = preps[syn_key], analyses[syn_key]
            s30 = p30["s_sim"]
            tr = Transformer.from_crs("EPSG:4326", CRS,
                                      always_xy=True)
            px, py = tr.transform(p30["base"].nav_llh[:, 1],
                                  p30["base"].nav_llh[:, 0])
            s_anchor = project_to_track(px, py, axis["x"], axis["y"],
                                        axis["s"])
            syn = (syn_key, {
                "s_sim": s30,
                "sim_rssnr": _smooth_db(s30, a30["sim_bed_prof_db"]),
                "implied": (np.interp(s_anchor, gmap["s"], gmap["g2_db"])
                            - 2.0 * gmap["att_db_per_km"]
                            * np.interp(s_anchor, gmap["s"],
                                        gmap["thick_m"]) / 1e3)})
        figs.insert(0, fig_bed_brightness(out, preps, corr_series,
                                          corr_stats, segment, syn=syn))
    if make_report:
        _report(out, case, config, metrics, notes, figs)
    ver = VER_ROOT / case
    ver.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / "metrics.json", ver / "metrics.json")
    for f in figs:
        shutil.copy2(f, ver / f.name)
    print("clutter (midcol rel surf, meas/sim dB): " + " | ".join(
        (f"{k}: "
         + (f"{analyses[k]['meas']['midcol_rel_surf_db']:+.1f}"
            if analyses[k]["meas"] else "----")
         + f"/{analyses[k]['sim']['midcol_rel_surf_db']:+.1f} "
         f"[{analyses[k]['verdict']}]") for k in order), flush=True)
    return metrics, config, out


# ========================================================================
# run provenance + staged-metrics merge
# ========================================================================
CONFIG_SCHEMA_VERSION = 1
# Physics-relevant config keys. Their hash fingerprints a run so a STAGED
# build (one --passes per invocation) can accumulate into one metrics.json
# while a CHANGED configuration refuses to merge into the old one.
_FINGERPRINT_KEYS = ("line", "segment", "att_db_per_km", "surf_rough",
                     "picked_bed", "gamma_rssnr", "demogorgn_bed", "antenna",
                     "posting_div", "spec_diffuse", "bed_roughness",
                     "n_traces", "hybrid_bed")


def _git_provenance():
    """(short sha, dirty) of the working tree, or (None, None) outside git."""
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=5, check=True).stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            capture_output=True, text=True, timeout=10,
                            check=True).stdout.strip()
        return sha, bool(st)
    except Exception:
        return None, None


def run_provenance(spec_doc=None, spec_path=None):
    """What code, invoked how, produced this directory.

    Recorded because a resolved physics config alone cannot identify the
    program that resolved it: two generations of this tool write configs that
    differ in which keys exist at all."""
    import hashlib
    import platform

    import soundersim
    sha, dirty = _git_provenance()
    prov = {"argv": list(sys.argv),
            "git_sha": sha, "git_dirty": dirty,
            "soundersim_version": soundersim.__version__,
            "python": platform.python_version(),
            "host": platform.node(),
            "created_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat()}
    if spec_path is not None:
        raw = Path(spec_path).read_bytes()
        prov["spec_path"] = str(spec_path)
        prov["spec_sha256"] = hashlib.sha256(raw).hexdigest()[:16]
    if spec_doc is not None:
        # the output directory literally contains its runnable input
        prov["spec"] = spec_doc
    return prov


def config_fingerprint(config):
    import hashlib
    blob = json.dumps({k: config.get(k) for k in _FINGERPRINT_KEYS},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def merge_metrics_doc(path, doc, order):
    """Accumulate a STAGED build into one metrics.json.

    run() used to overwrite metrics.json per invocation, so building a
    segment one --passes at a time left a canonical file describing only the
    LAST pass (outputs/basal_clutter/full_line held syn300km alone while the
    directory carried figures for six passes). Metric keys are namespaced by
    pass, so merging is well defined: existing entries survive, this
    invocation's entries win, and nothing is ever silently dropped.

    Merging only happens when the stored fingerprint matches -- a changed
    configuration starts a fresh file rather than blending two runs."""
    doc = dict(doc)
    prior = None
    if path.exists():
        try:
            old = json.loads(path.read_text())
            if old.get("config_fingerprint") == doc.get("config_fingerprint"):
                prior = old
            else:
                print(f"  [metrics] {path.name}: fingerprint differs from the "
                      "stored run -- starting a fresh file", flush=True)
        except Exception as e:
            print(f"  [metrics] {path.name} unreadable ({e}); overwriting",
                  flush=True)
    if prior:
        merged = dict(prior.get("metrics", {}))
        merged.update(doc["metrics"])
        # the one additive aggregate: per-pass wall times accumulate
        pw = dict(prior.get("metrics", {}).get(
            "simulation_wall_s", {}).get("per_pass_s", {}))
        pw.update(doc["metrics"].get("simulation_wall_s", {})
                  .get("per_pass_s", {}))
        if "simulation_wall_s" in merged and pw:
            merged["simulation_wall_s"] = {**merged["simulation_wall_s"],
                                           "per_pass_s": pw,
                                           "value": round(sum(pw.values()), 1)}
        doc["metrics"] = merged
        was = set(prior.get("passes_present", []))
        print(f"  [metrics] merged into {len(merged)} entries "
              f"(had {len(prior.get('metrics', {}))})", flush=True)
    else:
        was = set()
    doc["passes_present"] = sorted(was | set(order))
    doc["passes_this_invocation"] = list(order)
    if set(doc["passes_present"]) != set(order):
        doc["staged_note"] = (
            "STAGED build: this file accumulates across invocations that each "
            "ran a subset of passes. Aggregates needing the whole set "
            "(altitude_trend) are present only if one invocation ran them "
            "together; per-pass entries are complete for passes_present.")
    return doc


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
<h1>Basal clutter: {html.escape(LINE)} / {html.escape(config['segment'])}</h1>
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


def resolve_calibration(line=None):
    """(gamma_surface_db, att_db_per_km, record) for the active line.

    gamma_surface is manual, or 'solve' (the default on the study lines):
    returned as the literal string -- it needs simulations, so main_config
    owns the loop (seed run -> shift by the qualifying-median residual ->
    verify run; exact by dB-for-dB linearity of the received bed level in
    the mapping constant). It cannot come from the RSSNR regression
    intercept, which is degenerate with the mean bed reflectivity. A is
    manual, or solved by the RSSNR-vs-2H Theil-Sen regression over the
    line's own store samples: dataset-only, no simulation, one shot,
    seed-free. The regression is ALSO computed as a diagnostic when A is
    manual, so every line's fit is on record next to whatever value it
    actually uses.
    """
    activate_line(line or LINE)
    cal = dict(CALIBRATION)
    if cal["gamma_surface_db"] == "solve":
        gamma = "solve"
        rec = {"line": LINE, "gamma_surface_db": "solve",
               "gamma_surface_solve_settings": dict(GAMMA_SURFACE_SOLVE),
               "gl_aware": cal.get("gl_aware", True)}
    else:
        gamma = float(cal["gamma_surface_db"]["value"])
        rec = {"line": LINE, "gamma_surface_db": gamma,
               "gamma_surface_why": cal["gamma_surface_db"]["why"],
               "surface_anomaly_db": round(
                   gamma - gamma_surface_fresnel_db(), 2),
               "gl_aware": cal.get("gl_aware", True)}
    fit = None
    try:
        d, prov = fetch_rssnr_anchor()
        axis = ref_bed_picks()
        tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
        sx, sy = tr.transform(d["lon"], d["lat"])
        s_smp = project_to_track(sx, sy, axis["x"], axis["y"], axis["s"])
        thick = C / (2.0 * np.sqrt(axis["eps_ice"])) * (d["btw"] - d["stw"])
        _, fit = solve_attenuation_regression(
            s_smp, d["rssnr"], thick, d["qc"], gl_km=GL_S_KM,
            gl_aware=rec["gl_aware"])
    except Exception as e:
        fit = {"error": f"{type(e).__name__}: {e}"}
    rec["regression"] = fit
    if cal["att_db_per_km"] == "solve":
        if fit is None or "error" in fit:
            raise ValueError(f"line {LINE!r} calibrates A by regression but "
                             f"the fit failed: {fit}")
        att = fit["att_db_per_km"]
        rec["att_db_per_km"] = att
        rec["att_source"] = "solved (RSSNR-vs-2H Theil-Sen regression)"
    else:
        att = float(cal["att_db_per_km"]["value"])
        rec["att_db_per_km"] = att
        rec["att_source"] = f"manual: {cal['att_db_per_km']['why']}"
    return gamma, float(att), rec


def main_config():
    """``--config`` front door: an config/experiments/*.yaml spec IS the run input.

    Only EXECUTION-level overrides are accepted alongside it -- every physics
    knob has to come from the file, so a published result and the spec that
    names it cannot drift apart. The spec builds exactly the keyword dict the
    flag front door builds (tools/clutter_spec.py), so run() -- and every
    chunk cache key -- is unaffected by which door was used.
    """
    from clutter_spec import load_spec

    ap = argparse.ArgumentParser(
        description="Run one declarative experiment spec.")
    ap.add_argument("--config", required=True, metavar="YAML",
                    help="path to an config/experiments/*.yaml run spec")
    ap.add_argument("--line", default=None,
                    help="which line to run a MULTI-line protocol on "
                    "(required when the spec states run.lines; an execution "
                    "selector, not physics)")
    ap.add_argument("--out", default=None,
                    help="override the spec's run.out_root (execution only)")
    ap.add_argument("--force", action="store_true",
                    help="re-simulate cached chunks")
    args = ap.parse_args()
    spec = load_spec(args.config)
    kw = spec.to_run_kwargs()
    if spec.run.lines:
        if not args.line or args.line not in spec.run.lines:
            raise SystemExit(
                f"spec {spec.meta.name!r} is a multi-line protocol: pass "
                f"--line, one of {spec.run.lines}")
        kw["line"] = args.line
    elif args.line and args.line != spec.run.line:
        raise SystemExit(f"spec pins line {spec.run.line!r}; --line may only "
                         "select within a multi-line protocol")
    if args.out:
        kw["out_root"] = args.out
    kw["force"] = args.force
    # the spec travels into run_config.json, so the output directory carries
    # the exact input that produced it
    kw["spec_doc"] = spec.model_dump(mode="json", by_alias=True)
    kw["spec_path"] = args.config
    # figure knobs are module globals rather than run() kwargs (a wart the
    # RunSpec inherits; folded into the spec object in a later phase)
    global FIG_WIDTH_SCALE, BED_OVERLAY
    FIG_WIDTH_SCALE = spec.run.figures.width_scale
    BED_OVERLAY = spec.run.figures.bed_overlay
    print(f"spec {spec.meta.name!r} <- {args.config}", flush=True)
    gamma, att, cal_rec = resolve_calibration(kw["line"])
    gtxt = ("solve" if gamma == "solve" else
            f"{gamma:+.2f} dB (anomaly {cal_rec['surface_anomaly_db']:+.2f})")
    print(f"calibration [{cal_rec['line']}]: gamma_surface {gtxt}, "
          f"A = {att:g} dB/km ({cal_rec['att_source'].split(':')[0]})",
          flush=True)
    if cal_rec.get("regression") and "error" not in cal_rec["regression"]:
        f = cal_rec["regression"]
        print(f"  regression: A {f['att_db_per_km']} "
              f"[{f['slope_ci95_db_per_km'][0]}, "
              f"{f['slope_ci95_db_per_km'][1]}] dB/km, r "
              f"{f['r_rssnr_vs_2h']}, n {f['n_used']}"
              + (f" ({f['n_floating_excluded']} floating excluded)"
                 if f["n_floating_excluded"] else ""), flush=True)
    if kw["att"] == "solve":
        kw["att"] = att
    if gamma != "solve":
        kw["gamma_surface_db"] = gamma
        out = run(**kw)
    else:
        # GAMMA-SURFACE SOLVE: match the measured bed-window level via the
        # power-sum inversion (metric rssnr_level_residuals.gamma_solve).
        # gamma_required is seed-invariant by linearity, so the loop is one
        # seed evaluation plus one verification at the solved median; the
        # tolerance absorbs the small nonlinearities (surface registration,
        # gating) a gamma change drags along. Requires a gamma_from_rssnr
        # spec with at least one qualifying measured pass.
        st = dict(GAMMA_SURFACE_SOLVE)
        g, history, out = float(st["seed_db"]), [], None
        for it in range(3):
            print(f"gamma-solve eval {it}: gamma_surface {g:+.2f} dB",
                  flush=True)
            kw["gamma_surface_db"] = g
            out = run(**kw)
            m = json.loads((Path(out[2]) / "metrics.json").read_text())
            lv = m["metrics"].get("rssnr_level_residuals")
            if lv is None:
                raise SystemExit(
                    "gamma_surface: solve needs the rssnr_level_residuals "
                    "metric -- is this a gamma_from_rssnr spec with "
                    "measured passes?")
            gs = lv["gamma_solve"]
            r = gs["gamma_required_median_db"]
            history.append({"gamma_surface_db": g, **gs,
                            "per_pass_residual_db":
                                lv["per_pass_residual_db"]})
            if r is None or not np.isfinite(r):
                raise SystemExit(
                    "gamma_surface: solve has no qualifying pass -- no "
                    "measured bed window stands >= "
                    f"{st['min_headroom_db']} dB above the modeled clutter "
                    f"floor (per pass: {gs['per_pass']}). Set a manual "
                    "gamma_surface_db for this line.")
            print(f"  gamma_required {r:+.2f} dB (spread "
                  f"{gs['gamma_required_spread_db']} dB over "
                  f"{gs['qualifying_passes']})", flush=True)
            if (np.isfinite(gs["gamma_required_spread_db"])
                    and gs["gamma_required_spread_db"]
                    > st["spread_warn_db"]):
                print(f"  WARNING: qualifying passes disagree by "
                      f"{gs['gamma_required_spread_db']} dB (> "
                      f"{st['spread_warn_db']}): the sim is missing "
                      "physics in some pass's regime; the median is "
                      "recorded but treat this line's solved gamma as "
                      "suspect", flush=True)
            if abs(r - g) <= st["tolerance_db"]:
                break
            g = round(r, 2)
        else:
            raise SystemExit(
                f"gamma_surface: solve did not converge to |gamma_required "
                f"- gamma| <= {st['tolerance_db']} dB in 3 evaluations "
                f"(history {history}) -- the linearity assumption is "
                "broken; inspect the qualifying passes.")
        cal_rec["gamma_surface_db"] = g
        cal_rec["gamma_surface_source"] = (
            "solved (measured bed-window level matched by power-sum "
            "inversion, clutter floor subtracted; seed run + verify run)")
        cal_rec["surface_anomaly_db"] = round(
            g - gamma_surface_fresnel_db(), 2)
        cal_rec["gamma_surface_solve_history"] = history
        print(f"gamma-solve done: gamma_surface {g:+.2f} dB (anomaly "
              f"{cal_rec['surface_anomaly_db']:+.2f}), verify "
              f"gamma_required "
              f"{history[-1]['gamma_required_median_db']:+.2f} dB",
              flush=True)
    cfg_path = Path(out[2]) / "run_config.json"
    doc = json.loads(cfg_path.read_text())
    doc["calibration_resolution"] = cal_rec
    cfg_path.write_text(json.dumps(doc, indent=1) + "\n")
    return out


def main():
    # --line is pre-parsed so the parser below advertises the ACTIVE
    # line's segments/defaults in its choices and help text. --config is
    # pre-parsed alongside it: a spec run takes a different, much smaller
    # parser (main_config) and never reaches the flag set below.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, metavar="YAML",
                     help="run a declarative config/experiments/*.yaml spec instead "
                     "of the flags below (accepts only --out / --force "
                     "alongside it)")
    pre.add_argument("--line", choices=sorted(LINES),
                     default=DEFAULT_LINE)
    known = pre.parse_known_args()[0]
    if known.config:
        return main_config()
    activate_line(known.line)
    ap = argparse.ArgumentParser(description=__doc__, parents=[pre])
    ap.add_argument("--segment", choices=list(SEGMENTS), default="pilot",
                    help="study segment: 'pilot' (10 km) or 'full' (the "
                    "line's whole overlap window; where it crosses the "
                    "grounding line the bed is the hybrid grounded-DEM / "
                    "floating-picks construction)")
    ap.add_argument("--n-traces", type=int, default=None,
                    help="sim traces (default: this line's per-segment value, "
                    f"{ {k: N_TRACES_BY_SEGMENT[k] for k in SEGMENTS} })")
    ap.add_argument("--att", type=float, default=31.0,
                    help="one-way ice attenuation dB/km. THE DEFAULT (31) IS "
                    "NOT AN ADOPTED VALUE: the family analysis rejected it in "
                    "favour of A = 20 (Antarctic, level-anchored); the "
                    "Greenland line currently runs A = 14 with A = 16 +/- 2 "
                    "under consideration. Use a spec from config/experiments/ rather "
                    "than the flag defaults for anything reproducible. "
                    "Historical note -- 31 was "
                    "confirmed independently by run_cross_season repeat-pass "
                    "calibration and by the RSSNR K-K_phys diagnostic, and "
                    "the only tested change that improved the bed-return "
                    "tail slope. KNOWN CONSEQUENCE under median anchoring: "
                    "absolute nadir bed levels sit 15-20 dB below measured "
                    "(received level goes as K - RSSNR); a value sweep and "
                    "the anchoring choice are recorded follow-ups. Affects "
                    "only bed returns, not the surface-return geometry; 15 "
                    "remains the b26/altitude tools' constant)")
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
    ap.add_argument("--processing", choices=["none", "standard"],
                    default="none",
                    help="'standard': simulate at the product posting "
                    "(~14.85 m, every measured trace) and apply the "
                    "CSARP_standard-matching chain (mocomp + straight-track "
                    "focused SAR at the alias-limited aperture + "
                    f"{N_LOOKS_SIM}-look averaging) identically to every "
                    "simulated pass; recorded real-chain/gap list in the "
                    f"config; outputs/caches get the {PROC_TAG} suffix")
    ap.add_argument("--per-pass-figs", action="store_true",
                    help="STAGED DELIVERY: write each pass's complete "
                    "figure set as separate suffixed files "
                    "(radargrams_<pass>.png, ...) immediately after that "
                    "pass completes, printing 'FIGSET_READY <pass>'; the "
                    "unsuffixed combined figures are skipped")
    ap.add_argument("--plot-s-max", type=float, default=None,
                    metavar="S_KM",
                    help="crop the PLOTTED radargram along-track range at "
                    "this anchor s (km); data, caches and metrics keep the "
                    "full segment (plot-iteration knob, cache-replay safe)")
    ap.add_argument("--proc-cache", action="store_true",
                    help="persist/reuse the FOCUSED per-interface stacks "
                    f"under <out>/{PROC_CACHE_DIRNAME}/ (bit-exact source "
                    "of the processed powers, keyed on chain provenance + "
                    "per-chunk cache digests; stale caches are rebuilt): "
                    "plot iterations drop from ~30 min to seconds via "
                    "load_proc_pass")
    ap.add_argument("--bed-ablation", action="store_true",
                    help="with --picked-bed: also simulate every pass with "
                    "the BEDMACHINE and DEMOGORGN beds (identical "
                    "gamma/processing; own cache suffixes) and add them as "
                    "radargram rows -- the clean bed-source ablation")
    ap.add_argument("--demogorgn-bed", action="store_const", const=True,
                    default=None,
                    help="use a DEMOGORGN realization as the bed "
                    f"(pinned snapshot, {DGN_TAG} suffix); PLAIN -- the "
                    "picked-bed hybrid is a recorded follow-up. Nadir bed "
                    "sits ~44 m off our picks (documented, reported)")
    ap.add_argument("--demogorgn-seed", type=int, default=0,
                    help="DEMOGORGN realization seed_id (default 0; "
                    "conditioning makes the seed nearly irrelevant at "
                    "nadir)")
    ap.add_argument("--no-companion", action="store_true",
                    help="skip the constant-gamma companion correlation "
                    "run (e.g. when only the sims/caches are needed)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-name", default=None,
                    help="override the case directory NAME under --out "
                    "(default '<segment><tag>'); hypothesis tests use it to "
                    "keep one directory + cache per variable. Requires "
                    "--no-companion and no --bed-ablation")
    _n = rac.N_ELEMENTS
    ap.add_argument("--antenna", choices=["array", "isotropic", "array8"],
                    default=ANT_DEFAULT,
                    help=f"antenna pattern (default array = the MCoRDS-like "
                    f"{_n}-element {rac.SPACING_LAM}-lambda cross-track "
                    "array); 'isotropic' is the pattern-sensitivity worst "
                    "case and 'array8' the more-directive bracket (8 "
                    f"elements, {(8 - 1) / (_n - 1):.2f}x aperture). "
                    f"PROVENANCE CAVEAT: {_n} elements is the 2017 P-3 "
                    "centre-array readme value; it is applied unchanged to "
                    "the 2016 DC-8 line, which is an unverified transfer")
    ap.add_argument("--grazing-fix", nargs="?", type=float,
                    const=None, default=None, metavar="S_EFF",
                    help="override the grazing-fix taper s_eff (default: "
                    "analysis.yaml grazing_fix.s_eff). The fix itself -- "
                    "coherent off-specular taper + area-term-only D_Phi -- "
                    "is ON by default (it removes a facet-lattice aliasing "
                    "artifact); s_eff is part of the chunk cache key")
    ap.add_argument("--no-grazing-fix", action="store_true",
                    help="DEBUG: run the legacy kernels with the "
                    "facet-lattice aliasing artifact (bit-identical to "
                    "pre-fix; matches pre-fix caches). For artifact "
                    "demonstrations and A/B regression only")
    ap.add_argument("--bed-rough", nargs=2, type=float, default=None,
                    metavar=("SIGMA_M", "CORR_LEN_M"),
                    help="Gerekos sub-facet roughness on the BED interface "
                    "(the surface keeps its own); the RSSNR gamma is raised "
                    "by the nadir coherent attenuation so the nadir bed "
                    "level is conserved. " + bed_rough_validity())
    ap.add_argument("--passes", nargs="+", default=None,
                    metavar="KEY",
                    help=f"passes to simulate (default: the real ones, "
                    f"{list(ORDER)}). Naming a synthetic is how you request "
                    f"it -- this line declares {list(SYNTHETIC_KEYS)}. The "
                    "altitude-trend metric needs the whole triplet and is "
                    "skipped otherwise")
    ap.add_argument("--bed-rough-extra-db", type=float, default=0.0,
                    help="extra dB added to the --bed-rough gamma guard (the "
                    "pilot-measured residual that also compensates the added "
                    "INCOHERENT term; recorded in the config)")
    ap.add_argument("--specular-fraction", type=float, default=None,
                    metavar="F_S",
                    help="split the RSSNR-mapped bed reflectivity into a "
                    "SPECULAR share f_s (tilt-weighted, coherent) and a "
                    "DIFFUSE share 1-f_s (cos^n, incoherent). Off unless "
                    "given; f_s = 1 with --spec-tilt-deg 0 is the unsplit "
                    "baseline")
    ap.add_argument("--spec-tilt-deg", type=float, default=1.0,
                    metavar="S0",
                    help="rms sub-facet slope s0 (deg) of the specular tilt "
                    "weight exp(-tan^2(psi)/(2 s0^2)); 0 disables the "
                    "weighting")
    ap.add_argument("--diffuse-exponent", type=float, default=1.0,
                    metavar="N", help="exponent of the diffuse cos^n(theta) "
                    "angular law (1-2 spans the usual near-Lambert range)")
    ap.add_argument("--posting-div", type=int, default=1,
                    help="refine the SIM along-track posting by this factor "
                    "(2 -> 7.43 m, doubling the alias-limited aperture and "
                    "the simulation cost); measured data untouched. "
                    "Requires --processing standard")
    ap.add_argument("--focus-aperture",
                    choices=("alias_limited", "product_resolution",
                             "first_fresnel"),
                    default="alias_limited",
                    help="focused-SAR aperture rule (default uses the full "
                    "unaliased Doppler band; product_resolution preserves "
                    "the original-posting aperture after refinement; "
                    "first_fresnel uses one monostatic Fresnel-zone diameter "
                    "at bed optical range)")
    ap.add_argument("--fig-width-scale", type=float, default=1.0,
                    help="radargram figure width multiplier "
                    "(plot-iteration knob; cache-replay safe)")
    ap.add_argument("--trace-decomp-s", type=float, default=None,
                    nargs="+", metavar="S_KM",
                    help="anchor along-track position(s) (km) of the "
                    "SINGLE-TRACE decomposition figure, one panel each "
                    "(default "
                    f"{'/'.join(f'{v:g}' for v in np.atleast_1d(DECOMP_S_KM['full']))}"
                    " km on full). The nearest trace of every pass "
                    "is used and recorded per pass in the config; changing "
                    "it only re-does the analysis, never the simulations")
    ap.add_argument("--no-bed-overlay", action="store_true",
                    help="drop the bed overlays from the radargram panels "
                    "(default ON: the measured Bottom pick on measured "
                    "panels, the sim bed-layer nadir twtt on sim panels)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    global FIG_WIDTH_SCALE, BED_OVERLAY
    FIG_WIDTH_SCALE = args.fig_width_scale
    BED_OVERLAY = not args.no_bed_overlay
    run(segment=args.segment, n_traces=args.n_traces, att=args.att,
        surf_rough=not args.smooth_surface, out_root=args.out,
        force=args.force, picked_bed=args.picked_bed,
        gamma_rssnr=args.gamma_from_rssnr, processing=args.processing,
        bed_ablation=args.bed_ablation,
        demogorgn_bed=args.demogorgn_bed, demogorgn_seed=args.demogorgn_seed,
        companion=not args.no_companion, out_name=args.out_name,
        antenna=args.antenna,
        bed_rough=tuple(args.bed_rough) if args.bed_rough else None,
        grazing_fix=(False if args.no_grazing_fix else args.grazing_fix),
        posting_div=args.posting_div, passes=args.passes,
        focus_aperture=args.focus_aperture,
        bed_rough_extra_db=args.bed_rough_extra_db,
        trace_decomp_s_km=args.trace_decomp_s,
        per_pass_figs=args.per_pass_figs, plot_s_max_km=args.plot_s_max,
        proc_cache=args.proc_cache, line=args.line,
        spec=(None if args.specular_fraction is None
              else (args.specular_fraction, args.spec_tilt_deg,
                    args.diffuse_exponent)))


if __name__ == "__main__":
    main()
