"""Per site-visit ATM1B roughness: point-pair structure function on the 5 km site,
crossover noise budget, family fits over 1-30 m lag (Gaussian / exponential / power law /
Matern nu free) with a free nugget, Bragg-band S(k), exponential misfits and adequacy.
Reuses the study-line pipeline (claude_notes/atm_roughness/atm_roughness.py) for the
detrending, pair statistics, octave RMS and noise budget."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from pyproj import Transformer
from scipy.optimize import curve_fit
from scipy.special import gamma as Gamma, kv
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".." / "atm_roughness"))
import atm_roughness as ar  # noqa: E402
from common import CRS, SITE_RADIUS_M, read_granule  # noqa: E402

FIT_BAND = (1.0, 30.0)          # lag m (facet size ~30 m: larger scales are DEM tilt)
DIAG_LAGS = (1.0, 2.0, 5.0, 10.0, 20.0)
BRAGG = ar.BRAGG                # wavelength m at 30 deg: 60/195/300/400 MHz
OCTAVES = ar.OCTAVES + [(64, 128)]
LAG = ar.LAG_MID


# ------------------------------------------------------------ Matern
def D_matern(r, s2, l, nu, n0):
    a = np.sqrt(2 * nu) * np.asarray(r, float) / l
    with np.errstate(all="ignore"):
        c = 2 ** (1 - nu) / Gamma(nu) * a ** nu * kv(nu, a)
    c = np.where(np.isfinite(c), c, 0.0); c = np.clip(c, 0, 1)
    return 2 * s2 * (1 - c) + 2 * n0


def S_matern(k, s2, l, nu):
    """2-D Matern PSD, int S d2k = s2 (nu = 1/2 -> S_exp)."""
    return s2 * Gamma(nu + 1) * (2 * nu) ** nu / (np.pi * Gamma(nu) * l ** (2 * nu)) * (2 * nu / l ** 2 + k ** 2) ** (-(nu + 1))


# --------------------------------------------------------------- fits
def fit_families(lag, D, N, n0_max, min_pairs=50):
    """Log-space LSQ of the four families + free nugget (0 <= n0 <= n0_max) over FIT_BAND."""
    m = np.isfinite(D) & (D > 0) & (N >= min_pairs) & (lag >= FIT_BAND[0]) & (lag <= FIT_BAND[1])
    if m.sum() < 8:
        return {}
    x, y = lag[m], np.log(D[m]); n = int(m.sum())
    n0g = min(max(D[m][0] / 4, 1e-6), n0_max * 0.9) if n0_max > 0 else 1e-6
    s2g = max(D[m][-1] / 2 - n0g, 1e-5)
    ln0_hi = np.log(max(n0_max, 1e-8))
    mods = {
        "gaussian": (lambda r, a, b, c: np.log(ar.D_gauss(r, np.exp(a), np.exp(b), np.exp(c))), 3,
                     (np.log(s2g), np.log(5.0), np.log(n0g)), ([-30, np.log(0.2), -40], [10, np.log(300), ln0_hi])),
        "exponential": (lambda r, a, b, c: np.log(ar.D_exp(r, np.exp(a), np.exp(b), np.exp(c))), 3,
                        (np.log(s2g), np.log(5.0), np.log(n0g)), ([-30, np.log(0.2), -40], [10, np.log(300), ln0_hi])),
        "powerlaw": (lambda r, a, H, c: np.log(ar.D_pl(r, np.exp(a), H, np.exp(c))), 3,
                     (np.log(s2g / 10 ** (2 * 0.5)), 0.5, np.log(n0g)), ([-30, 0.02, -40], [10, 0.99, ln0_hi])),
        "matern": (lambda r, a, b, nu, c: np.log(D_matern(r, np.exp(a), np.exp(b), nu, np.exp(c))), 4,
                   (np.log(s2g), np.log(5.0), 0.5, np.log(n0g)), ([-30, np.log(0.2), 0.08, -40], [10, np.log(300), 2.5, ln0_hi])),
    }
    res = {}
    for name, (fn, k, p0, bounds) in mods.items():
        try:
            p0 = np.clip(p0, np.array(bounds[0]) + 1e-6, np.array(bounds[1]) - 1e-6)
            pp, cov = curve_fit(fn, x, y, p0=p0, bounds=bounds, maxfev=40000)
            resid = y - fn(x, *pp); rss = float(np.sum(resid ** 2))
            err = np.sqrt(np.abs(np.diag(cov)))
            d = dict(bic=float(n * np.log(max(rss, 1e-12) / n) + k * np.log(n)), runs_p=ar.runs_test_p(resid), n_bins=n,
                     rms_log_resid=float(np.sqrt(rss / n)), n0=float(np.exp(pp[-1])))
            if name == "powerlaw":
                c, H = float(np.exp(pp[0])), float(pp[1]); d.update(c=c, H=H, H_err=float(err[1]), beta=2 * H + 2, A=c / (4 * np.pi * ar._I_H(H)))
            elif name == "matern":
                d.update(sigma=float(np.sqrt(np.exp(pp[0]))), l=float(np.exp(pp[1])), nu=float(pp[2]), nu_err=float(err[2]), l_rel_err=float(err[1]))
            else:
                d.update(sigma=float(np.sqrt(np.exp(pp[0]))), l=float(np.exp(pp[1])), l_rel_err=float(err[1]), sigma_rel_err=float(err[0] / 2))
            res[name] = d
        except Exception as e:  # noqa: BLE001
            res[name] = dict(error=str(e)[:60])
    ok = {k: v["bic"] for k, v in res.items() if "bic" in v}
    if ok:
        best = min(ok, key=ok.get); res["best"] = best
        ok3 = {k: v for k, v in ok.items() if k != "matern"}   # 3-parameter families only
        res["best3"] = min(ok3, key=ok3.get) if ok3 else None
        res["dbic"] = {k: float(v - ok[best]) for k, v in ok.items()}
        # exponential fit with nugget fixed at the crossover budget (robustness check)
        if "exponential" in res and "sigma" in res["exponential"] and n0_max > 0:
            try:
                fn = lambda r, a, b: np.log(ar.D_exp(r, np.exp(a), np.exp(b), n0_max))
                pp, _ = curve_fit(fn, x, y, p0=(np.log(s2g), np.log(5.0)), bounds=([-30, np.log(0.2)], [10, np.log(300)]))
                res["exponential"]["sigma_fixn0"] = float(np.sqrt(np.exp(pp[0]))); res["exponential"]["l_fixn0"] = float(np.exp(pp[1]))
            except Exception:  # noqa: BLE001
                pass
    return res


def S_of(fit, name, k):
    f = fit.get(name, {})
    if name == "gaussian" and "sigma" in f: return ar.S_gauss(k, f["sigma"] ** 2, f["l"])
    if name == "exponential" and "sigma" in f: return ar.S_exp(k, f["sigma"] ** 2, f["l"])
    if name == "powerlaw" and "A" in f: return ar.S_pl(k, f["A"], f["beta"])
    if name == "matern" and "sigma" in f: return S_matern(k, f["sigma"] ** 2, f["l"], f["nu"])
    return np.nan


def D_of(fit, name, r):
    f = fit.get(name, {})
    if name == "gaussian" and "sigma" in f: return ar.D_gauss(r, f["sigma"] ** 2, f["l"], f["n0"])
    if name == "exponential" and "sigma" in f: return ar.D_exp(r, f["sigma"] ** 2, f["l"], f["n0"])
    if name == "powerlaw" and "c" in f: return ar.D_pl(r, f["c"], f["H"], f["n0"])
    if name == "matern" and "sigma" in f: return D_matern(r, f["sigma"] ** 2, f["l"], f["nu"], f["n0"])
    return np.nan


def dB(x): return float(10 * np.log10(x)) if np.isfinite(x) and x > 0 else np.nan


# ------------------------------------------------------------- site run
def load_site(paths, lat0, lon0, hemi, pad_m=500.0):
    tr = Transformer.from_crs("EPSG:4326", CRS[hemi], always_xy=True)
    x0, y0 = tr.transform(lon0, lat0)
    parts = []
    for p in paths:
        try:
            g = read_granule(p)
        except Exception as e:  # noqa: BLE001
            print(f"    read fail {Path(p).name}: {str(e)[:60]}", flush=True); continue
        x, y = tr.transform(g["lon"], g["lat"])
        m = np.hypot(x - x0, y - y0) < SITE_RADIUS_M + pad_m
        if m.sum() == 0: continue
        parts.append(dict(x=x[m], y=y[m], h=g["h"][m], t=g["t"][m], rcv=g["rcv"][m], pw=g["pw"][m]))
    if not parts:
        return None
    P = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    # track axis: principal direction of the cloud
    xy = np.c_[P["x"] - x0, P["y"] - y0]
    u, s_, vt = np.linalg.svd(xy - xy.mean(0), full_matrices=False)
    ax = vt[0]; s = xy @ ax; yy = xy @ np.array([-ax[1], ax[0]])
    P["s"], P["yc"] = s, yy
    P["heading_deg"] = float(np.degrees(np.arctan2(ax[1], ax[0])))
    if not np.isfinite(P["pw"]).any():
        P["pw"] = np.zeros_like(P["rcv"])
    return P


def analyse_site(P, block_m=1000.0):
    """Return dict: site-level pooled D(r) fits + per-block summaries."""
    Q = ar.qc_and_detrend(P["s"], P["yc"], P["h"], P["rcv"], P["pw"], P["t"])
    inside = np.abs(Q["s"]) <= SITE_RADIUS_M
    for k in ("s", "y", "r", "h", "t", "trend", "slope"):
        Q[k] = Q[k][inside]
    if len(Q["s"]) < 5000:
        return dict(status="too_few_points", n_used=int(len(Q["s"])))
    blocks = []
    for s0 in np.arange(-SITE_RADIUS_M, SITE_RADIUS_M, block_m):
        b = ar.analyse_block(Q, s0, s0 + block_m, do_grid=False)
        if b is not None:
            b.pop("N"); blocks.append(b)
    if not blocks:
        return dict(status="no_blocks", n_used=int(len(Q["s"])))
    # pooled D(r) over the site
    D, N = ar.structure_function(Q["s"], Q["y"], Q["r"], Q["t"], max_lag=110.0, sub_frac=0.15)
    noise = pooled_noise(D, N)
    cls = ar.primary_class(N)
    n0_max = noise["xover_total_sigma"] ** 2 if np.isfinite(noise["xover_total_sigma"]) else 0.05 ** 2
    if cls == "same":
        n0_max = max(noise["ranging_sigma"] ** 2 if np.isfinite(noise["ranging_sigma"]) else n0_max, 1e-6)
    fits = {a: fit_families(LAG, D[f"{cls}_{a}"], N[f"{cls}_{a}"], n0_max) for a in ar.SECTORS}
    fa = fits["all"]
    oct_ = {a: octaves(D[f"{cls}_{a}"], N[f"{cls}_{a}"]) for a in ar.SECTORS}
    out = dict(status="ok", n_used=int(len(Q["s"])), n_blocks=len(blocks), primary_class=cls,
               swath_m=float(np.percentile(Q["y"], 99) - np.percentile(Q["y"], 1)),
               density_per_m2=float(len(Q["s"]) / (2 * SITE_RADIUS_M * max(np.percentile(Q["y"], 99) - np.percentile(Q["y"], 1), 1))),
               elev_m=float(np.median(Q["h"])), slope_100m=float(np.median(Q["slope"])), rms_resid_m=float(Q["r"].std()),
               mad_m=Q["mad_m"], dropped_qc=Q["n_dropped_qc"], dropped_blunder=Q["n_dropped_blunder"], heading_deg=P["heading_deg"],
               noise=noise, fits=fa, fits_along=fits["along"], fits_cross=fits["cross"],
               octave_rms=oct_["all"], octave_rms_along=oct_["along"], octave_rms_cross=oct_["cross"],
               D_lag=LAG.tolist(), D_all=D[f"{cls}_all"].tolist(), N_all=N[f"{cls}_all"].tolist(),
               D_same=D["same_all"].tolist(), N_same=N["same_all"].tolist(), D_xscan=D["xscan_all"].tolist(), N_xscan=N["xscan_all"].tolist(),
               blocks=[block_summary(b) for b in blocks])
    out.update(derived(fa, D[f"{cls}_all"], N[f"{cls}_all"]))
    return out


def octaves(D, N):
    out = {}
    m = np.isfinite(D) & (N >= 30)
    for lo, hi in OCTAVES:
        if m.sum() < 4 or LAG[m].min() > lo / 2 or LAG[m].max() < hi / 2:
            out[f"{lo}-{hi}m"] = np.nan; continue
        d_lo = np.interp(np.log(lo / 2), np.log(LAG[m]), D[m]); d_hi = np.interp(np.log(hi / 2), np.log(LAG[m]), D[m])
        out[f"{lo}-{hi}m"] = float(np.sqrt(max(d_hi - d_lo, 0) / 2))
    return out


def pooled_noise(D, N):
    """Crossover total (xscan pairs, lag < 0.5 m), scan-to-scan (xscan-same excess 2-10 m), ranging (quadrature)."""
    m = (LAG < 0.5) & (N["xscan_all"] > 300)
    xo = float(np.sqrt(np.nansum(D["xscan_all"][m] * N["xscan_all"][m]) / N["xscan_all"][m].sum() / 2)) if m.any() else np.nan
    mm = (LAG >= 2) & (LAG <= 10) & (N["same_all"] > 200) & (N["xscan_all"] > 200)
    sc = float(np.sqrt(max(np.nanmean(D["xscan_all"][mm] - D["same_all"][mm]) / 2, 0))) if mm.sum() >= 3 else np.nan
    rg = float(np.sqrt(max(xo ** 2 - sc ** 2, 0))) if np.isfinite(xo) and np.isfinite(sc) else np.nan
    ms = (LAG < 1.5) & (N["same_all"] > 300)
    same_small = float(np.sqrt(np.nanmin(D["same_all"][ms]) / 2)) if ms.any() else np.nan
    return dict(xover_total_sigma=xo, scan_sigma=sc, ranging_sigma=rg, same_smallest_sigma=same_small,
                xover_lag_m=float(LAG[m].max()) if m.any() else np.nan, n_same_lt2m=int(N["same_all"][LAG < 2].sum()))


def derived(fa, D, N):
    """Bragg S(k) per family, exponential misfits, band-limited sigma, adequacy."""
    out = {}
    if not fa or "best" not in fa:
        return dict(bragg={}, misfit={}, adequacy=np.nan)
    kB = {k: 2 * np.pi / v for k, v in BRAGG.items()}
    br = {}
    for fam in ("gaussian", "exponential", "powerlaw", "matern"):
        for name, k in kB.items():
            br[f"S_{name}_{fam}_dB"] = dB(S_of(fa, fam, k))
    best, best3 = fa["best"], fa.get("best3")
    for name, k in kB.items():
        br[f"S_{name}_best_dB"] = dB(S_of(fa, best, k))
    mis = {}
    for name, k in kB.items():
        se = dB(S_of(fa, "exponential", k))
        mis[f"bragg_{name}_vs_best"] = se - dB(S_of(fa, best, k))
        mis[f"bragg_{name}_vs_matern"] = se - dB(S_of(fa, "matern", k))
        mis[f"bragg_{name}_vs_powerlaw"] = se - dB(S_of(fa, "powerlaw", k))
    m = np.isfinite(D) & (N >= 50)
    for r in DIAG_LAGS:
        if m.sum() >= 4 and LAG[m].min() <= r <= LAG[m].max():
            dm = np.interp(np.log(r), np.log(LAG[m]), D[m]); de = D_of(fa, "exponential", r)
            mis[f"D_{int(r)}m_exp_minus_meas_dB"] = dB(de) - dB(dm)
        else:
            mis[f"D_{int(r)}m_exp_minus_meas_dB"] = np.nan
    e = fa.get("exponential", {})
    if "sigma" in e and m.sum() >= 4 and LAG[m].max() >= 15:
        d15 = np.interp(np.log(15.0), np.log(LAG[m]), D[m])
        out["sigma_bl30_m"] = float(np.sqrt(max(d15 / 2 - e["n0"], 0)))     # band-limited RMS, wavelengths < 30 m
        out["sigma_bl30_expmodel_m"] = float(e["sigma"] * np.sqrt(1 - np.exp(-15.0 / e["l"])))
    else:
        out["sigma_bl30_m"] = np.nan; out["sigma_bl30_expmodel_m"] = np.nan
    white = e.get("runs_p", np.nan)
    b15, b10 = mis.get("bragg_195MHz_vs_best", np.nan), mis.get("bragg_300MHz_vs_best", np.nan)
    out["adequate"] = bool(np.isfinite(white) and white > 0.05 and abs(b15) < 3 and abs(b10) < 3) if np.isfinite(b15) and np.isfinite(b10) else None
    out["adequate_bragg_only"] = bool(abs(b15) < 3 and abs(b10) < 3) if np.isfinite(b15) and np.isfinite(b10) else None
    out["bragg"] = br; out["misfit"] = mis
    return out


def block_summary(b):
    c = b["primary_class"]
    return dict(s0_km=b["s0_km"], n_shots=b["n_shots"], swath_m=b["swath_m"], elev_m=b["elev_m"], slope=b["slope"], rms_resid_m=b["rms_resid_m"],
                cls=c, octave_rms=b["octave_rms"], aniso=b["aniso_along_over_cross"], scan_noise_sigma_m=b["scan_noise_sigma_m"])
