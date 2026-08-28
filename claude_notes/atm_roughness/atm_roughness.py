"""ATM L1B surface-roughness form and along-line statistics (plan steps 1-2).

    uv run claude_notes/atm_roughness/atm_roughness.py --line greenland_westcoast --date 2017-05-10

Estimator choice (see results note): the ATM point density is 0.08-0.23
shots/m^2, so a 1 m grid is <20 % filled and its 2-D FFT is dominated by the
conical-scan mask. Primary estimator is therefore the point-pair structure
function D(r) = <[h(x) - h(x+r)]^2> in log lag bins and azimuth sectors
(along / cross track / all), which is exact on gappy clouds, carries the
white-noise nugget 2 n0 as a free term, and has closed forms for all three
ACF families. The 1 m median grid + mask-corrected Welch PSD is kept as a
cross-check. PSD/ACF fitting follows the structure of
claude_notes/icesat2_roughness/roughness_from_atl.py (log-space curve_fit with
a free noise term), rewritten for 2-D isotropic forms.

Conventions: increments at lag r probe wavelengths ~2r, so octave
[L1, L2] in wavelength <-> lags [L1/2, L2/2] and the non-parametric octave
RMS is sqrt((D(L2/2) - D(L1/2)) / 2), noise-free by construction.
2-D two-sided PSDs, normalised so that int S d2k = sigma^2:
  Gaussian     S = sigma^2 l^2/(4 pi) exp(-k^2 l^2/4)
  exponential  S = sigma^2 l^2/(2 pi) (1 + k^2 l^2)^(-3/2)
  power law    S = A k^-beta, beta = 2H + 2, D(r) = c r^(2H), c = 4 pi A I(H)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.special import j0
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
import atm_common as ac  # noqa: E402
import line_geometry as lg  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ac.ROOT / "outputs" / "atm_roughness"
LAG_EDGES = 0.25 * 2 ** (np.arange(0, 38) / 4.0)          # 0.25 .. ~150 m, quarter-octave
LAG_MID = np.sqrt(LAG_EDGES[:-1] * LAG_EDGES[1:])
OCTAVES = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)]
BRAGG = {"60MHz": 5.0, "195MHz": 1.5, "300MHz": 1.0, "400MHz": 0.75}   # wavelength m at 30 deg
FIT_LAGS = (0.25, 50.0)     # lags; Lambda = 2r -> 0.5..100 m; the smallest populated bins pin the nugget
HP_SIGMA_M = 37.0           # Gaussian high-pass, half-power at Lambda = 5.34 sigma ~ 200 m
GRID_M = 1.0
FIX_SIGMA, FIX_L = 0.049, 2.98   # C&S fixture


# ------------------------------------------------------------- loading
def load_points(line, date):
    """All ILATM1B shots for a line/date: lat, lon, h, t (s of day), sigstr."""
    d = ac.ATM_CACHE / line / date
    files = sorted(glob.glob(str(d / "ILATM1B_*.h5")))
    cols = {k: [] for k in ("lat", "lon", "h", "t", "rcv", "pw")}
    for f in files:
        with h5py.File(f) as h:
            cols["lat"].append(h["latitude"][:]); cols["lon"].append(h["longitude"][:])
            cols["h"].append(h["elevation"][:].astype(np.float64))
            hh = h["instrument_parameters/time_hhmmss"][:]
            # stored as hhmmss.sss (float), despite the attribute's x1000 example
            hr = np.floor(hh / 1e4); mn = np.floor((hh - hr * 1e4) / 100); sc = hh - hr * 1e4 - mn * 100
            cols["t"].append(hr * 3600 + mn * 60 + sc)
            cols["rcv"].append(h["instrument_parameters/rcv_sigstr"][:].astype(np.float32))
            cols["pw"].append(h["instrument_parameters/pulse_width"][:].astype(np.float32))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["files"] = [Path(f).name for f in files]
    return out


def to_axis(pts, line, own_axis=False, gate_m=2000.0):
    """Project to the line CRS and the anchor axis -> s, lateral offset."""
    xy_ref, s_ref, crs = ac.anchor_axis(line)
    xy = lg.to_crs(pts["lat"], pts["lon"], crs)
    s, lat_off = lg.project(xy, xy_ref, s_ref)
    inside = np.abs(lat_off) < gate_m
    dist_km = float(np.median(np.abs(lat_off)) / 1e3)
    if own_axis or inside.mean() < 0.5:
        # ATM track far from the line: axis = the ATM flight track itself
        order = np.argsort(pts["t"]); t = pts["t"][order]
        tb = np.floor(t).astype(np.int64); u, first = np.unique(tb, return_index=True)
        cen = np.array([xy[order][a:b].mean(0) for a, b in zip(first, np.r_[first[1:], len(t)])])
        keep = np.r_[True, np.hypot(*np.diff(cen, axis=0).T) > 20]
        cen = cen[keep]
        s_c = lg.arc_length(cen)
        s, lat_off = lg.project(xy, cen, s_c)
        inside = np.abs(lat_off) < gate_m
        return s, lat_off, xy, crs, dict(own_axis=True, median_dist_from_line_km=dist_km)
    return s, lat_off, xy, crs, dict(own_axis=False, median_dist_from_line_km=dist_km)


# --------------------------------------------------------- preprocessing
def qc_and_detrend(s, y, h, rcv, pw, t, cell=5.0):
    """Drop weak/odd shots and blunders, remove >= ~200 m scales (normalised
    Gaussian convolution on a 5 m grid), return residual field per point,
    plus the low-pass surface for slope/elevation regimes."""
    ok = (rcv > np.percentile(rcv, 1)) & (pw < np.percentile(pw, 99.5)) & np.isfinite(h)
    s, y, h, t = s[ok], y[ok], h[ok], t[ok]
    i = np.floor((s - s.min()) / cell).astype(int); j = np.floor((y - y.min()) / cell).astype(int)
    ni, nj = i.max() + 1, j.max() + 1
    # median per 5 m cell (blunder-robust), then normalised Gaussian smooth
    key = i * nj + j
    order = np.argsort(key); ks = key[order]; hs = h[order]
    uk, first = np.unique(ks, return_index=True)
    med = np.array([np.median(hs[a:b]) for a, b in zip(first, np.r_[first[1:], len(hs)])])
    grid = np.full(ni * nj, np.nan); grid[uk] = med; grid = grid.reshape(ni, nj)
    m = np.isfinite(grid).astype(float); g0 = np.where(m > 0, grid, 0.0)
    sig = HP_SIGMA_M / cell
    lp = gaussian_filter(g0, sig) / np.maximum(gaussian_filter(m, sig), 1e-6)
    lp[gaussian_filter(m, sig) < 0.05] = np.nan
    trend = lp[i, j]
    r = h - trend
    mad = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
    good = np.isfinite(r) & (np.abs(r) < 6 * max(mad, 0.05))
    # slope of the low-pass surface (for regimes)
    gs, gy = np.gradient(np.nan_to_num(lp), cell)
    slope = np.hypot(gs, gy)[i, j]
    return dict(s=s[good], y=y[good], r=r[good], h=h[good], t=t[good], trend=trend[good], slope=slope[good],
                n_dropped_qc=int((~ok).sum()), n_dropped_blunder=int((~good).sum()), mad_m=float(mad))


# ------------------------------------------------------- structure function
SAME_DT, XSCAN_DT = 0.03, 0.5      # one scan rotation is ~0.05-0.07 s
CLASSES = ("same", "xscan")        # same-scan pairs (ranging noise only) / cross-scan (adds attitude noise)
SECTORS = ("all", "along", "cross")


def structure_function(s, y, r, t, max_lag=150.0, sub_frac=0.2, seed=0):
    """D(r) in log lag bins for pair classes x azimuth sectors.

    Full point set for lags <= 8 m, random subset for lags <= max_lag."""
    rng = np.random.default_rng(seed)
    pts = np.column_stack([s, y])
    nb = len(LAG_MID)
    keys = [f"{c}_{a}" for c in CLASSES for a in SECTORS]
    D = {k: np.zeros(nb) for k in keys}; N = {k: np.zeros(nb) for k in keys}
    for rmax, frac, lo_bin in ((8.0, 1.0, 0), (max_lag, sub_frac, np.searchsorted(LAG_EDGES, 8.0))):
        idx = np.arange(len(pts)) if frac >= 1 else rng.choice(len(pts), max(int(frac * len(pts)), 1), replace=False)
        if len(idx) < 50:
            continue
        p = pts[idx]
        pairs = cKDTree(p).query_pairs(rmax, output_type="ndarray")
        if len(pairs) == 0:
            continue
        d = p[pairs[:, 1]] - p[pairs[:, 0]]
        lag = np.hypot(d[:, 0], d[:, 1])
        dh2 = (r[idx][pairs[:, 1]] - r[idx][pairs[:, 0]]) ** 2
        dt = np.abs(t[idx][pairs[:, 1]] - t[idx][pairs[:, 0]])
        ang = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))   # 0 = along, 90 = cross
        b = np.digitize(lag, LAG_EDGES) - 1
        sel = (b >= lo_bin) & (b < nb) & (lag > 0)
        for c, cm in (("same", dt < SAME_DT), ("xscan", dt > XSCAN_DT)):
            for a, am in (("all", True), ("along", ang < 22.5), ("cross", ang > 67.5)):
                mask = sel & cm & am
                D[f"{c}_{a}"] += np.bincount(b[mask], dh2[mask], nb); N[f"{c}_{a}"] += np.bincount(b[mask], None, nb)
    for k in D:
        D[k] = np.where(N[k] > 0, D[k] / np.maximum(N[k], 1), np.nan)
    return D, N


# ------------------------------------------------------------------ fits
def _I_H(H):
    return quad(lambda u: (1 - j0(u)) * u ** (-2 * H - 1), 0, np.inf, limit=400)[0]


def D_gauss(r, s2, l, n0): return 2 * s2 * (1 - np.exp(-(r / l) ** 2)) + 2 * n0
def D_exp(r, s2, l, n0): return 2 * s2 * (1 - np.exp(-r / l)) + 2 * n0
def D_pl(r, c, H, n0): return c * r ** (2 * H) + 2 * n0


def S_gauss(k, s2, l): return s2 * l ** 2 / (4 * np.pi) * np.exp(-(k * l) ** 2 / 4)
def S_exp(k, s2, l): return s2 * l ** 2 / (2 * np.pi) * (1 + (k * l) ** 2) ** -1.5
def S_pl(k, A, beta): return A * k ** -beta


def runs_test_p(x):
    """Wald-Wolfowitz runs test on the sign of residuals -> p-value."""
    sgn = x > 0
    n1, n2 = sgn.sum(), (~sgn).sum()
    if n1 < 2 or n2 < 2:
        return np.nan
    runs = 1 + np.sum(sgn[1:] != sgn[:-1])
    mu = 1 + 2 * n1 * n2 / (n1 + n2)
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    return float(2 * norm.sf(abs((runs - mu) / np.sqrt(var))))


def fit_D(lag, D, N, min_pairs=30, fixed_n0=None):
    """Fit Gaussian / exponential / power-law + nugget to D(r) in log space.
    fixed_n0: nugget variance held at this value (2 free params) instead of free (3)."""
    m = np.isfinite(D) & (D > 0) & (N >= min_pairs) & (lag >= FIT_LAGS[0]) & (lag <= FIT_LAGS[1])
    if m.sum() < 8:
        return {}
    x, yv, n = lag[m], D[m], m.sum()
    ly = np.log(yv)
    n0_guess = max(yv[0] / 2 * 0.5, 1e-6)
    s2_guess = max(yv[-1] / 2 - n0_guess, 1e-5)
    res = {}
    if fixed_n0 is None:
        models = {
            "gaussian": (lambda r, a, b, c: np.log(D_gauss(r, np.exp(a), np.exp(b), np.exp(c))),
                         (np.log(s2_guess), np.log(5.0), np.log(n0_guess)), ([-30, np.log(0.1), -30], [10, np.log(500), 5])),
            "exponential": (lambda r, a, b, c: np.log(D_exp(r, np.exp(a), np.exp(b), np.exp(c))),
                            (np.log(s2_guess), np.log(5.0), np.log(n0_guess)), ([-30, np.log(0.1), -30], [10, np.log(500), 5])),
            "powerlaw": (lambda r, a, H, c: np.log(D_pl(r, np.exp(a), H, np.exp(c))),
                         (np.log(s2_guess / 10 ** (2 * 0.6)), 0.6, np.log(n0_guess)), ([-30, 0.02, -30], [10, 0.99, 5])),
        }
    else:
        f0 = float(fixed_n0)
        models = {
            "gaussian": (lambda r, a, b: np.log(D_gauss(r, np.exp(a), np.exp(b), f0)),
                         (np.log(s2_guess), np.log(5.0)), ([-30, np.log(0.1)], [10, np.log(500)])),
            "exponential": (lambda r, a, b: np.log(D_exp(r, np.exp(a), np.exp(b), f0)),
                            (np.log(s2_guess), np.log(5.0)), ([-30, np.log(0.1)], [10, np.log(500)])),
            "powerlaw": (lambda r, a, H: np.log(D_pl(r, np.exp(a), H, f0)),
                         (np.log(s2_guess / 10 ** (2 * 0.6)), 0.6), ([-30, 0.02], [10, 0.99])),
        }
    kpar = 3 if fixed_n0 is None else 2
    for name, (fn, p0, bounds) in models.items():
        try:
            pp, cov = curve_fit(fn, x, ly, p0=p0, bounds=bounds, maxfev=20000)
            resid = ly - fn(x, *pp)
            rss = float(np.sum(resid ** 2))
            bic = n * np.log(rss / n) + kpar * np.log(n)
            err = np.sqrt(np.abs(np.diag(cov)))
            n0v = np.exp(pp[2]) if fixed_n0 is None else f0
            d = dict(rss_log=rss, bic=float(bic), runs_p=runs_test_p(resid), n_bins=int(n),
                     noise_sigma_m=float(np.sqrt(n0v)), nugget_fixed=fixed_n0 is not None, rms_log_resid=float(np.sqrt(rss / n)))
            if name == "powerlaw":
                c, H = float(np.exp(pp[0])), float(pp[1])
                A = c / (4 * np.pi * _I_H(H))
                d.update(c=c, H=H, H_err=float(err[1]), beta=2 * H + 2, A=A)
            else:
                d.update(sigma_m=float(np.sqrt(np.exp(pp[0]))), l_m=float(np.exp(pp[1])), l_rel_err=float(err[1]))
            res[name] = d
        except Exception as e:  # noqa: BLE001
            res[name] = dict(error=str(e)[:80])
    ok = {k: v for k, v in res.items() if "bic" in v}
    if ok:
        best = min(ok, key=lambda k: ok[k]["bic"])
        res["best"] = best
        res["dbic_second"] = float(sorted(v["bic"] for v in ok.values())[1] - ok[best]["bic"]) if len(ok) > 1 else np.nan
    return res


def psd_from_fit(fit, k):
    if "gaussian" in fit and "sigma_m" in fit["gaussian"]:
        g = fit["gaussian"]; Sg = S_gauss(k, g["sigma_m"] ** 2, g["l_m"])
    else:
        Sg = np.nan
    e = fit.get("exponential", {}); Se = S_exp(k, e["sigma_m"] ** 2, e["l_m"]) if "sigma_m" in e else np.nan
    p = fit.get("powerlaw", {}); Sp = S_pl(k, p["A"], p["beta"]) if "A" in p else np.nan
    return dict(gaussian=Sg, exponential=Se, powerlaw=Sp)


def octave_rms(lag, D, N):
    """Non-parametric octave RMS from D differences at lags L/2 (nugget cancels)."""
    out = {}
    m = np.isfinite(D) & (N >= 30)
    for lo, hi in OCTAVES:
        if m.sum() < 4:
            out[f"{lo}-{hi}m"] = np.nan; continue
        d_lo = np.interp(np.log(lo / 2), np.log(lag[m]), D[m])
        d_hi = np.interp(np.log(hi / 2), np.log(lag[m]), D[m])
        out[f"{lo}-{hi}m"] = float(np.sqrt(max(d_hi - d_lo, 0) / 2))
    return out


# ---------------------------------------------------------- gridded PSD
def grid_block(s, y, r, s0, s1):
    """1 m median grid over the block with shot-count mask."""
    y0, y1 = np.percentile(y, [0.5, 99.5])
    i = np.floor((s - s0) / GRID_M).astype(int); j = np.floor((y - y0) / GRID_M).astype(int)
    ni, nj = int((s1 - s0) / GRID_M), int((y1 - y0) / GRID_M) + 1
    ok = (i >= 0) & (i < ni) & (j >= 0) & (j < nj)
    key = i[ok] * nj + j[ok]; order = np.argsort(key); ks = key[order]; rs = r[ok][order]
    uk, first, cnt = np.unique(ks, return_index=True, return_counts=True)
    med = np.array([np.median(rs[a:b]) for a, b in zip(first, np.r_[first[1:], len(rs)])])
    g = np.zeros(ni * nj); c = np.zeros(ni * nj, int); g[uk] = med; c[uk] = cnt
    return g.reshape(ni, nj), c.reshape(ni, nj)


def welch2d(g, c, tile=256):
    """Mask-corrected 2-D Welch PSD (Hann tiles): radial average + axis slices.
    Returns k (rad/m), S_radial, S_along (k_s axis), S_cross (k_y axis)."""
    ni, nj = g.shape
    tj = min(tile, nj); ti = min(tile, ni)
    if ti < 64 or tj < 32:
        return None
    win = np.outer(np.hanning(ti), np.hanning(tj)); wn = np.sum(win ** 2)
    acc = None; ntile = 0
    for a in range(0, ni - ti + 1, ti // 2):
        for b in range(0, nj - tj + 1, tj // 2):
            m = (c[a:a + ti, b:b + tj] > 0); p = m.mean()
            if p < 0.03:
                continue
            f = np.fft.fft2(g[a:a + ti, b:b + tj] * win)
            P = np.abs(f) ** 2 * GRID_M ** 2 / (wn * (2 * np.pi) ** 2) / p ** 2   # two-sided, m^4
            acc = P if acc is None else acc + P; ntile += 1
    if acc is None:
        return None
    P = np.fft.fftshift(acc / ntile)
    ks = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ti, GRID_M)); ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(tj, GRID_M))
    KS, KY = np.meshgrid(ks, ky, indexing="ij"); K = np.hypot(KS, KY)
    edges = np.geomspace(2 * np.pi / 200, np.pi / GRID_M, 30)
    kb = np.sqrt(edges[:-1] * edges[1:])
    rad = np.array([np.mean(P[(K >= lo) & (K < hi)]) if np.any((K >= lo) & (K < hi)) else np.nan for lo, hi in zip(edges[:-1], edges[1:])])
    ci, cj = ti // 2, tj // 2
    along = np.array([np.mean(P[(np.abs(KS[:, cj]) >= lo) & (np.abs(KS[:, cj]) < hi), cj]) if np.any((np.abs(ks) >= lo) & (np.abs(ks) < hi)) else np.nan for lo, hi in zip(edges[:-1], edges[1:])])
    cross = np.array([np.mean(P[ci, (np.abs(ky) >= lo) & (np.abs(ky) < hi)]) if np.any((np.abs(ky) >= lo) & (np.abs(ky) < hi)) else np.nan for lo, hi in zip(edges[:-1], edges[1:])])
    return dict(k=kb, radial=rad, along=along, cross=cross, fill=float((c > 0).mean()), ntile=ntile)


# ------------------------------------------------------------ per block
def primary_class(N):
    """same-scan if it has pairs below 2 m lag (9 kHz ATM6), else cross-scan."""
    small = LAG_MID < 2.0
    return "same" if N["same_all"][small].sum() > 2000 else "xscan"


def analyse_block(P, s0, s1, do_grid=True):
    m = (P["s"] >= s0) & (P["s"] < s1)
    if m.sum() < 2000:
        return None
    s, y, r, t = P["s"][m], P["y"][m], P["r"][m], P["t"][m]
    D, N = structure_function(s, y, r, t)
    fits = {}
    cls = primary_class(N)
    oct_ = {a: octave_rms(LAG_MID, D[f"{cls}_{a}"], N[f"{cls}_{a}"]) for a in SECTORS}
    aniso = {}
    for lo, hi in OCTAVES:
        a_, c_ = oct_["along"][f"{lo}-{hi}m"], oct_["cross"][f"{lo}-{hi}m"]
        aniso[f"{lo}-{hi}m"] = float(a_ / c_) if (np.isfinite(c_) and c_ > 0 and np.isfinite(a_)) else np.nan
    # scan-to-scan noise: lag-independent excess of cross-scan over same-scan D at 2-10 m
    mm = (LAG_MID >= 2) & (LAG_MID <= 10) & (N["same_all"] > 200) & (N["xscan_all"] > 200)
    scan_noise = float(np.sqrt(max(np.nanmean(D["xscan_all"][mm] - D["same_all"][mm]) / 2, 0))) if mm.sum() >= 3 else np.nan
    blk = dict(s0_km=s0 / 1e3, s1_km=s1 / 1e3, n_shots=int(m.sum()), swath_m=float(np.percentile(y, 99) - np.percentile(y, 1)),
               elev_m=float(np.median(P["h"][m])), slope=float(np.median(P["slope"][m])),
               rms_resid_m=float(r.std()), primary_class=cls,
               octave_rms=oct_["all"], octave_rms_along=oct_["along"], octave_rms_cross=oct_["cross"],
               aniso_along_over_cross=aniso, scan_noise_sigma_m=scan_noise, fits=fits,
               D=dict(lag=LAG_MID.tolist(), **{k: np.where(np.isfinite(v), v, None).tolist() for k, v in D.items()}),
               N={k: v.tolist() for k, v in N.items()})
    if do_grid:
        g, c = grid_block(s, y, r, s0, s1)
        w = welch2d(g, c)
        if w:
            blk["grid_psd"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in w.items()}
            blk["grid_fill_1m"] = w["fill"]
    return blk


def fit_blocks(blocks, noise):
    """Stage 3: fits per block/class/sector. Same-scan nugget fixed at the flight
    ranging noise (degenerate otherwise: no same-scan pairs below ~1 m); cross-scan free."""
    n0_same = noise.get("ranging_sigma_m", np.nan) ** 2
    for b in blocks:
        D = {k: np.array([np.nan if v is None else v for v in v_]) for k, v_ in b["D"].items() if k != "lag"}
        N = {k: np.array(v) for k, v in b["N"].items()}
        b["fits"] = {k: fit_D(LAG_MID, D[k], N[k], fixed_n0=(n0_same if (k.startswith("same") and np.isfinite(n0_same)) else None)) for k in D}
        fa = b["fits"].get(f"{b['primary_class']}_all", {})
        kB = {k: 2 * np.pi / v for k, v in BRAGG.items()}
        b["psd_bragg_m4"] = {name: psd_from_fit(fa, kk) for name, kk in kB.items()} if fa else {}


# ------------------------------------------------------------- driver
def run(line, date, block_m=1000.0, own_axis=False, quick=False):
    pts = load_points(line, date)
    s, y, xy, crs, axis_info = to_axis(pts, line, own_axis)
    gate = np.abs(y) < 2000
    P = qc_and_detrend(s[gate], y[gate], pts["h"][gate], pts["rcv"][gate], pts["pw"][gate], pts["t"][gate])
    t = P["t"]
    s_lo, s_hi = np.floor(P["s"].min() / block_m) * block_m, np.ceil(P["s"].max() / block_m) * block_m
    blocks = []
    starts = np.arange(s_lo, s_hi, block_m)
    if quick:
        starts = starts[:: max(1, len(starts) // 6)]
    for s0 in starts:
        blk = analyse_block(P, s0, s0 + block_m, do_grid=(block_m >= 1000))
        if blk is None:
            continue
        blocks.append(blk)
        print(f"  block {s0 / 1e3:6.1f} km n={blk['n_shots']:6d} {blk['primary_class']} rms={blk['rms_resid_m']:.3f} "
              f"oct={[round(v, 3) for v in blk['octave_rms'].values()]} scan={blk['scan_noise_sigma_m']:.3f} "
              f"aniso4-8={blk['aniso_along_over_cross'].get('4-8m', np.nan):.2f}", flush=True)
    noise = flight_noise(blocks)
    print("noise:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in noise.items()})
    fit_blocks(blocks, noise)
    for cls in CLASSES:
        nug = [b["fits"][f"{cls}_all"]["powerlaw"]["noise_sigma_m"] for b in blocks
               if "noise_sigma_m" in b["fits"].get(f"{cls}_all", {}).get("powerlaw", {}) and not b["fits"][f"{cls}_all"]["powerlaw"].get("nugget_fixed")]
        if nug:
            noise[f"{cls}_block_nugget_median_sigma_m"] = float(np.median(nug))
            noise[f"{cls}_block_nugget_iqr_m"] = [float(np.percentile(nug, 25)), float(np.percentile(nug, 75))]
    for b in blocks:
        f = b["fits"].get(f"{b['primary_class']}_all", {})
        print(f"  fit {b['s0_km']:6.1f} km best={f.get('best')} dBIC={f.get('dbic_second', np.nan):.1f} "
              + (f"G(s={f['gaussian']['sigma_m']:.3f},l={f['gaussian']['l_m']:.1f},p={f['gaussian']['runs_p']:.2f}) " if 'sigma_m' in f.get('gaussian', {}) else "")
              + (f"E(s={f['exponential']['sigma_m']:.3f},l={f['exponential']['l_m']:.1f},p={f['exponential']['runs_p']:.2f}) " if 'sigma_m' in f.get('exponential', {}) else "")
              + (f"PL(H={f['powerlaw']['H']:.2f},p={f['powerlaw']['runs_p']:.2f}) " if 'H' in f.get('powerlaw', {}) else ""), flush=True)
    meta = dict(line=line, date=date, files=pts["files"], n_shots_total=int(len(pts["h"])),
                n_in_gate=int(gate.sum()), n_used=int(len(P["s"])), dropped_qc=P["n_dropped_qc"],
                dropped_blunder=P["n_dropped_blunder"], crs=crs, block_m=block_m, axis=axis_info,
                s_range_km=[float(P["s"].min() / 1e3), float(P["s"].max() / 1e3)], noise=noise,
                hp_sigma_m=HP_SIGMA_M, fit_lags_m=FIT_LAGS)
    od = OUT / line; od.mkdir(parents=True, exist_ok=True)
    tag = f"{date}_{int(block_m)}m"
    (od / f"blocks_{tag}.json").write_text(json.dumps(dict(meta=meta, blocks=blocks), indent=None, default=float))
    write_csv(od / f"blocks_{tag}.csv", blocks)
    return meta, blocks


def flight_noise(blocks):
    """Per-flight white-noise floor.
    ranging: same-scan D(r) pooled over blocks, r -> 0 intercept of 2 n0 + c r^2H on the
             smallest populated lags (<= 3 m); cross-scan likewise (includes attitude noise).
    scan-to-scan: lag-independent excess of cross-scan over same-scan D at 2-10 m."""
    out = {}
    lag = LAG_MID
    for cls in CLASSES:
        Dsum = np.zeros(len(lag)); Nsum = np.zeros(len(lag))
        for b in blocks:
            D = np.array([np.nan if v is None else v for v in b["D"][f"{cls}_all"]]); N = np.array(b["N"][f"{cls}_all"])
            Dsum += np.nan_to_num(D * N); Nsum += N
        D = Dsum / np.maximum(Nsum, 1)
        m = (lag <= 3.0) & (Nsum > 500)
        if m.sum() >= 3:
            try:
                pp, _ = curve_fit(lambda r, a, H, c: np.log(np.exp(a) * r ** (2 * H) + 2 * np.exp(c)), lag[m], np.log(D[m]),
                                  p0=(np.log(D[m][-1] / 2), 0.5, np.log(D[m][0] / 4)), bounds=([-30, 0.02, -30], [10, 0.99, 5]))
                out[f"{cls}_intercept_sigma_m"] = float(np.sqrt(np.exp(pp[2])))
            except Exception as e:  # noqa: BLE001
                out[f"{cls}_fit_error"] = str(e)[:60]
            out[f"{cls}_smallest_bin_sigma_m"] = float(np.sqrt(D[m][0] / 2))
            out[f"{cls}_smallest_lag_m"] = float(lag[m][0])
        nug = [b["fits"][f"{cls}_all"]["powerlaw"]["noise_sigma_m"] for b in blocks
               if "noise_sigma_m" in b["fits"].get(f"{cls}_all", {}).get("powerlaw", {})]
        if nug:
            out[f"{cls}_block_nugget_median_sigma_m"] = float(np.median(nug))
            out[f"{cls}_block_nugget_iqr_m"] = [float(np.percentile(nug, 25)), float(np.percentile(nug, 75))]
    sn = [b["scan_noise_sigma_m"] for b in blocks if np.isfinite(b["scan_noise_sigma_m"])]
    if sn:
        out["scan_to_scan_sigma_median_m"] = float(np.median(sn)); out["scan_to_scan_sigma_iqr_m"] = [float(np.percentile(sn, 25)), float(np.percentile(sn, 75))]
    out["primary_class"] = blocks[0]["primary_class"] if blocks else None
    # ranging noise: crossover total (cross-scan, smallest lag) minus scan-to-scan, in quadrature
    sx, ss = out.get("xscan_smallest_bin_sigma_m", np.nan), out.get("scan_to_scan_sigma_median_m", np.nan)
    out["ranging_sigma_m"] = float(np.sqrt(max(sx ** 2 - ss ** 2, 0.0))) if np.isfinite(sx) and np.isfinite(ss) else np.nan
    if np.isfinite(ss) and "scan_to_scan_sigma_iqr_m" in out:
        lo, hi = out["scan_to_scan_sigma_iqr_m"]
        out["ranging_sigma_range_m"] = [float(np.sqrt(max(sx ** 2 - hi ** 2, 0))), float(np.sqrt(max(sx ** 2 - lo ** 2, 0)))]
    return out


def write_csv(path, blocks):
    cols = ["s0_km", "s1_km", "n_shots", "swath_m", "elev_m", "slope", "rms_resid_m", "primary_class", "scan_noise_sigma_m"]
    oct_cols = [f"rms_{lo}-{hi}m" for lo, hi in OCTAVES]
    an_cols = [f"aniso_{lo}-{hi}m" for lo, hi in OCTAVES]
    fit_cols = ["best", "dbic_second", "g_sigma", "g_l", "g_runs_p", "e_sigma", "e_l", "e_runs_p", "pl_H", "pl_beta", "pl_A", "pl_runs_p", "nugget_sigma",
                "along_best", "along_g_l", "cross_best", "cross_g_l", "along_pl_H", "cross_pl_H", "grid_fill_1m"]
    br_cols = [f"S_{k}_best_m4" for k in BRAGG]
    with open(path, "w") as f:
        f.write(",".join(cols + oct_cols + an_cols + fit_cols + br_cols) + "\n")
        for b in blocks:
            c = b["primary_class"]
            fa = b["fits"].get(f"{c}_all", {}); g = fa.get("gaussian", {}); e = fa.get("exponential", {}); p = fa.get("powerlaw", {})
            fl = b["fits"].get(f"{c}_along", {}); fc = b["fits"].get(f"{c}_cross", {})
            best = fa.get("best")
            row = [b[c] for c in cols] + [b["octave_rms"][f"{lo}-{hi}m"] for lo, hi in OCTAVES] + \
                  [b["aniso_along_over_cross"][f"{lo}-{hi}m"] for lo, hi in OCTAVES] + \
                  [best, fa.get("dbic_second"), g.get("sigma_m"), g.get("l_m"), g.get("runs_p"), e.get("sigma_m"), e.get("l_m"), e.get("runs_p"),
                   p.get("H"), p.get("beta"), p.get("A"), p.get("runs_p"), p.get("noise_sigma_m"),
                   fl.get("best"), fl.get("gaussian", {}).get("l_m"), fc.get("best"), fc.get("gaussian", {}).get("l_m"),
                   fl.get("powerlaw", {}).get("H"), fc.get("powerlaw", {}).get("H"), b.get("grid_fill_1m")] + \
                  [b["psd_bragg_m4"].get(k, {}).get(best) if best else None for k in BRAGG]
            f.write(",".join("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.5g}" if isinstance(v, float) else str(v)) for v in row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", required=True); ap.add_argument("--date", required=True)
    ap.add_argument("--block-m", type=float, nargs="*", default=[1000.0, 500.0])
    ap.add_argument("--own-axis", action="store_true"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    for bm in a.block_m:
        print(f"== {a.line} {a.date} block {bm:.0f} m")
        run(a.line, a.date, bm, a.own_axis, a.quick)


if __name__ == "__main__":
    main()
