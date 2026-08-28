"""RSR prototype on one westcoast 195 MHz frame (2017_Greenland_P3 20170510_03_013).

Per-trace surface peak power from the cached CSARP_standard product (SAR
sigma_x 2.5 m, combine rline_rng -5..5 = 11-line incoherent average, dline 6
-> 14.9 m spacing) and from CSARP_qlook (50 presums x 10 incoherent averages,
174 m spacing).  Neither is single-look, so besides the textbook HK / Rice
amplitude fits (which assume one look) the power is fitted with the N-look
Rice law (noncentral chi-square, 2N dof) with N free or fixed.

Inversion (first-order SPM, nadir, derivation in the note):
    Pn/Pc = (2k sigma_cell)^2 exp((2k sigma_c)^2)
sigma_cell^2 = roughness variance in the band the resolution cell sees
(|k_B| <= k_max, Lambda >= ~5 m here); the ACF family only enters through
which band-limited variance you call sigma. Gaussian (rsr.spm "large l") and
power-law variants are both evaluated. Compared with the ATM 2017 1 km blocks.

Run: uv run python claude_notes/rsr_roughness/rsr_prototype.py
"""
from pathlib import Path
import json
import numpy as np
import xarray as xr
import h5py
from scipy import optimize, special, stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "rsr_roughness"
OUT.mkdir(parents=True, exist_ok=True)
C = 299792458.0
F0 = 195e6
K = 2 * np.pi * F0 / C
B = 30e6                      # chirp bandwidth -> 5 m range resolution
WIN, STEP = 1000, 500         # traces (15 km, 50 % overlap) on the standard product
PILOT = (2678, 3334)          # p3_2017 pilot slice = s 40-50 km


# ----------------------------------------------------------------- data ---
def surface_peaks_standard():
    ds = xr.open_dataset(ROOT / "outputs/cache/frame_2017_Greenland_P3_20170510_03_013_CSARP_standard.nc",
                         engine="h5netcdf")
    P = ds.Data.values
    tw = ds.twtt.values
    dt = tw[1] - tw[0]
    isurf = np.round((ds.Surface.values - tw[0]) / dt).astype(int)
    rows = np.arange(P.shape[0])[:, None]
    idx = np.clip(isurf[:, None] + np.arange(-6, 7)[None, :], 0, P.shape[1] - 1)
    pk = P[rows, idx].max(1)
    agl = ds.Elevation.values - 0.0  # ellipsoidal platform height
    h = C * ds.Surface.values / 2    # platform -> surface range
    import pyproj
    x, y = pyproj.Transformer.from_crs(4326, 3413, always_xy=True).transform(
        ds.Longitude.values, ds.Latitude.values)
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))]) / 1e3
    return dict(P=pk, h=h, s=s, roll=ds.Roll.values, pitch=ds.Pitch.values)


def surface_peaks_qlook():
    f = h5py.File(ROOT / "outputs/cache/Data_20170510_03_013_qlook.mat", "r")
    P = f["Data"][()]                      # (traces, bins) in this layout
    tw = f["Time"][()].ravel()
    surf = f["Surface"][()].ravel()
    dt = tw[1] - tw[0]
    isurf = np.round((surf - tw[0]) / dt).astype(int)
    rows = np.arange(P.shape[0])[:, None]
    idx = np.clip(isurf[:, None] + np.arange(-6, 7)[None, :], 0, P.shape[1] - 1)
    return dict(P=P[rows, idx].max(1), h=C * surf / 2)


# ------------------------------------------------------------- models ---
def nll_nlook(theta, P):
    """N-look Rice on power: P = (Pn/2N) * X, X ~ ncx2(2N, 2N Pc/Pn)."""
    lpc, lpn, lN = theta
    pc, pn, N = np.exp(lpc), np.exp(lpn), np.exp(lN)
    scale = pn / (2 * N)
    return -np.sum(stats.ncx2.logpdf(P / scale, 2 * N, 2 * N * pc / pn) - np.log(scale))


def fit_nlook(P, N_fixed=None):
    m, v = P.mean(), P.var()
    # moment start: mean = Pc+Pn, var = (Pn^2 + 2 Pc Pn)/N
    N0 = N_fixed or 5.0
    pn0 = max(m * 0.5, 1e-30)
    x0 = [np.log(m - pn0), np.log(pn0), np.log(N0)]
    if N_fixed is None:
        r = optimize.minimize(nll_nlook, x0, args=(P,), method="Nelder-Mead",
                              options=dict(xatol=1e-4, fatol=1e-4, maxiter=4000))
        lpc, lpn, lN = r.x
    else:
        f = lambda t: nll_nlook([t[0], t[1], np.log(N_fixed)], P)
        r = optimize.minimize(f, x0[:2], method="Nelder-Mead",
                              options=dict(xatol=1e-4, fatol=1e-4, maxiter=4000))
        lpc, lpn, lN = r.x[0], r.x[1], np.log(N_fixed)
    return dict(pc=np.exp(lpc), pn=np.exp(lpn), N=np.exp(lN), nll=r.fun)


def pdf_hk(A, a, s, mu, w=None):
    """Homodyned-K on amplitude, compound form: Rice(a, s*sqrt(w)) mixed with Gamma(mu, 1)."""
    if w is None:
        w = np.exp(np.linspace(np.log(1e-3), np.log(40), 160))
    gw = stats.gamma.pdf(w, mu)
    sw = s * np.sqrt(w)[None, :]
    A = A[:, None]
    z = a * A / sw**2
    logrice = np.log(A / sw**2) - (A**2 + a**2) / (2 * sw**2) + np.log(special.i0e(z)) + z
    return np.trapezoid(np.exp(logrice) * gw[None, :], w, axis=1)


def fit_hk(A):
    m2 = np.mean(A**2)
    def nll(t):
        a, s, mu = np.exp(t)
        p = pdf_hk(A, a, s, mu)
        return -np.sum(np.log(np.maximum(p, 1e-300)))
    best = None
    for fc in (0.2, 0.6):
        x0 = [0.5 * np.log(fc * m2), 0.5 * np.log((1 - fc) * m2 / 2), np.log(3.0)]
        r = optimize.minimize(nll, x0, method="Nelder-Mead",
                              options=dict(xatol=1e-3, fatol=1e-3, maxiter=3000))
        if best is None or r.fun < best.fun:
            best = r
    a, s, mu = np.exp(best.x)
    return dict(pc=a**2, pn=2 * s**2 * mu, mu=mu, nll=best.fun)


def fit_rice(A):
    def nll(t):
        a, s = np.exp(t)
        return -np.sum(stats.rice.logpdf(A, a / s, scale=s))
    m2 = np.mean(A**2)
    r = optimize.minimize(nll, [0.5 * np.log(0.5 * m2), 0.5 * np.log(0.25 * m2)],
                          method="Nelder-Mead")
    a, s = np.exp(r.x)
    return dict(pc=a**2, pn=2 * s**2, nll=r.fun)


# ---------------------------------------------------------- inversion ---
def sigma_from_ratio(ratio, sigma_c_over_cell=1.0):
    """Solve (2k s)^2 exp((2k s_c)^2) = ratio with s_c = f * s (SPM, nadir).
    Returns sigma_cell (m); f=1 is the rsr.spm 'large correlation length' form."""
    g = lambda s: (2 * K * s)**2 * np.exp((2 * K * sigma_c_over_cell * s)**2) - ratio
    return optimize.brentq(g, 1e-4, 2.0)


def sigma_gauss_band(sigma, l, kmax, kmin=0.0):
    """Gaussian ACF: variance inside kmin<|k|<kmax (S = s^2 l^2/(4pi) exp(-k^2 l^2/4))."""
    return sigma * np.sqrt(np.exp(-kmin**2 * l**2 / 4) - np.exp(-kmax**2 * l**2 / 4))


def sigma_pl_band(A, beta, kmin, kmax):
    """Power law S = A k^-beta: sqrt(2 pi A int k^(1-beta) dk) over the band."""
    return np.sqrt(2 * np.pi * A * (kmin**(2 - beta) - kmax**(2 - beta)) / (beta - 2))


# ---------------------------------------------------------------- main ---
def main():
    std = surface_peaks_standard()
    ql = surface_peaks_qlook()
    P, s, h = std["P"], std["s"], std["h"]
    h0 = np.median(h)
    # geometry -> band the cell sees
    dr = C / (2 * B)                         # 5 m range resolution
    r_pl = np.sqrt(2 * h0 * dr)              # pulse-limited radius
    th_max = r_pl / h0
    kmax_ct = 2 * K * th_max                 # cross-track Bragg limit
    kmax_at = np.pi / 2.5                    # SAR along-track (delta x 2.5 m)
    kmax = min(kmax_ct, kmax_at)
    L_strip = 11 * 2.5                       # 11-line average -> 27.5 m along-track cell
    kmin = 2 * np.pi / (2 * r_pl)            # scales larger than the cell are tilt
    geom = dict(h0_m=h0, r_pl_m=r_pl, theta_max_deg=np.degrees(th_max),
                Lambda_min_ct_m=2 * np.pi / kmax_ct, Lambda_min_at_m=2 * np.pi / kmax_at,
                Lambda_max_m=2 * np.pi / kmin, strip_m=L_strip)
    print("geometry:", {k: round(float(v), 2) for k, v in geom.items()})

    # normalise the coherent 1/(2h)^2 range dependence within windows (tiny here)
    Pn_ = P * (h / h0)**2
    rows = []
    starts = list(range(0, len(P) - WIN + 1, STEP)) + ["pilot"]
    for st in starts:
        a, b = PILOT if st == "pilot" else (st, st + WIN)
        Pw = Pn_[a:b]
        A = np.sqrt(Pw)
        nl_free = fit_nlook(Pw)
        nl11 = fit_nlook(Pw, 11.0)
        nl55 = fit_nlook(Pw, 5.5)
        nl1 = fit_nlook(Pw, 1.0)
        hk = fit_hk(A / A.std())          # scale-free fit for stability
        hk_pc, hk_pn = hk["pc"] * A.var(), hk["pn"] * A.var()
        rc = fit_rice(A)
        row = dict(s0=float(s[a]), s1=float(s[b - 1]), n=b - a,
                   mean_dB=10 * np.log10(Pw.mean()),
                   cv=float(Pw.std() / Pw.mean()),
                   lag1_corr=float(np.corrcoef(np.log(Pw[:-1]), np.log(Pw[1:]))[0, 1]),
                   N_free=nl_free["N"], ratio_free_dB=10 * np.log10(nl_free["pn"] / nl_free["pc"]),
                   ratio_N11_dB=10 * np.log10(nl11["pn"] / nl11["pc"]),
                   ratio_N55_dB=10 * np.log10(nl55["pn"] / nl55["pc"]),
                   ratio_N1_dB=10 * np.log10(nl1["pn"] / nl1["pc"]),
                   ratio_hk_dB=10 * np.log10(hk_pn / hk_pc), hk_mu=hk["mu"],
                   ratio_rice_dB=10 * np.log10(rc["pn"] / rc["pc"]),
                   dnll_free_vs_N11=nl11["nll"] - nl_free["nll"],
                   dnll_free_vs_N1=nl1["nll"] - nl_free["nll"])
        for tag in ("free", "N11", "N55", "hk"):
            r = 10**(row[f"ratio_{tag}_dB"] / 10)
            try:
                row[f"sigma_{tag}_cm"] = 100 * sigma_from_ratio(r)
            except ValueError:
                row[f"sigma_{tag}_cm"] = np.nan
        rows.append(row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})

    # qlook: 10 looks known, 286 traces, one window
    Pq = ql["P"] * (ql["h"] / np.median(ql["h"]))**2
    q10 = fit_nlook(Pq, 10.0); qf = fit_nlook(Pq)
    qrow = dict(n=len(Pq), N_free=qf["N"], ratio_N10_dB=10 * np.log10(q10["pn"] / q10["pc"]),
                ratio_free_dB=10 * np.log10(qf["pn"] / qf["pc"]),
                sigma_N10_cm=100 * sigma_from_ratio(q10["pn"] / q10["pc"]))
    print("qlook whole frame:", {k: round(float(v), 2) for k, v in qrow.items()})

    # ---- ATM 2017 blocks: band-limited sigma the cell should see
    import csv
    atm = list(csv.DictReader(open(ROOT / "outputs/atm_roughness/greenland_westcoast/blocks_2017-05-10_1000m.csv")))
    atm_rows = []
    for r in atm:
        g = lambda k: float(r[k]) if r[k] not in ("", "nan") else np.nan
        oct_ = {k: g(f"rms_{k}") for k in ("1-2m", "2-4m", "4-8m", "8-16m", "16-32m", "32-64m")}
        s_5_64 = np.sqrt(sum(oct_[k]**2 for k in ("4-8m", "8-16m", "16-32m", "32-64m")))
        s_5_32 = np.sqrt(sum(oct_[k]**2 for k in ("4-8m", "8-16m", "16-32m")))
        s_1_64 = np.sqrt(np.nansum([oct_[k]**2 for k in oct_]))
        A_, beta = g("pl_A"), g("pl_beta")
        s_pl = sigma_pl_band(A_, beta, kmin, kmax) if np.isfinite(A_) else np.nan
        s_g = sigma_gauss_band(g("g_sigma"), g("g_l"), kmax, kmin)
        ratio_pred = (2 * K * s_5_64)**2 * np.exp((2 * K * s_1_64)**2)
        atm_rows.append(dict(s0=g("s0_km"), s1=g("s1_km"), sigma_5_64_cm=100 * s_5_64,
                             sigma_5_32_cm=100 * s_5_32, sigma_1_64_cm=100 * s_1_64,
                             sigma_pl_band_cm=100 * s_pl, sigma_gauss_band_cm=100 * s_g,
                             g_sigma_cm=100 * g("g_sigma"), g_l_m=g("g_l"), H=g("pl_H"),
                             ratio_pred_dB=10 * np.log10(ratio_pred),
                             slope=g("slope")))
    json.dump(dict(geometry=geom, windows=rows, qlook=qrow, atm=atm_rows),
              open(OUT / "rsr_prototype_20170510_03_013.json", "w"), indent=1, default=float)

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    ax[0].plot(s, 10 * np.log10(Pn_), lw=0.4, color="0.5")
    ax[0].set_ylabel("surface peak (dB, product units)")
    ax[0].axvspan(s[PILOT[0]], s[PILOT[1] - 1], color="tab:orange", alpha=0.15, label="pilot s 40-50")
    ax[0].legend(loc="lower left", fontsize=8)
    wm = [0.5 * (r["s0"] + r["s1"]) for r in rows[:-1]]
    for key, lab, mk in (("ratio_free_dB", "N-look Rice, N free", "o"),
                         ("ratio_N11_dB", "N = 11 fixed", "s"),
                         ("ratio_N1_dB", "N = 1 (Rice)", "^"),
                         ("ratio_hk_dB", "HK (1-look)", "x")):
        ax[1].plot(wm, [r[key] for r in rows[:-1]], mk + "-", label=lab, ms=4)
    ax[1].plot([0.5 * (rows[-1]["s0"] + rows[-1]["s1"])], [rows[-1]["ratio_free_dB"]],
               "*", ms=12, color="tab:orange", label="pilot window, N free")
    am = [0.5 * (r["s0"] + r["s1"]) for r in atm_rows]
    ax[1].plot(am, [r["ratio_pred_dB"] for r in atm_rows], ".", color="k", alpha=0.5,
               label="SPM prediction from ATM (5-64 m band)")
    ax[1].set_ylabel("Pn / Pc (dB)"); ax[1].legend(fontsize=7, ncol=2)
    ax[2].plot(wm, [r["N_free"] for r in rows[:-1]], "o-", label="fitted N (looks)")
    ax[2].axhline(11, color="k", ls="--", lw=0.8, label="11 lines averaged")
    ax[2].set_ylabel("N"); ax[2].legend(fontsize=8); ax[2].set_yscale("log")
    ax[3].plot(am, [r["sigma_5_64_cm"] for r in atm_rows], ".", color="k", label="ATM sigma 4-64 m octaves")
    ax[3].plot(am, [r["sigma_5_32_cm"] for r in atm_rows], ".", color="0.6", label="ATM sigma 4-32 m")
    ax[3].plot(am, [r["sigma_pl_band_cm"] for r in atm_rows], "+", color="tab:green", label="ATM power law, cell band")
    for key, lab, mk in (("sigma_free_cm", "RSR N free", "o"), ("sigma_N11_cm", "RSR N=11", "s"),
                         ("sigma_hk_cm", "RSR HK 1-look", "x")):
        ax[3].plot(wm, [r[key] for r in rows[:-1]], mk + "-", label=lab, ms=4)
    ax[3].set_ylabel("sigma_cell (cm)"); ax[3].set_xlabel("s (km, 2017 axis)"); ax[3].legend(fontsize=7, ncol=2)
    ax[3].set_yscale("log")
    fig.suptitle("RSR prototype, 20170510_03_013 (CSARP_standard, 15 m traces, 1000-trace windows)")
    fig.tight_layout()
    fig.savefig(OUT / "rsr_prototype_20170510_03_013.png", dpi=130)

    # amplitude histogram of the pilot window with the fitted laws
    a, b = PILOT
    Pw = Pn_[a:b]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(0, Pw.max() * 1.05, 300)
    ax.hist(Pw, bins=40, density=True, color="0.8", label="surface peak power")
    for fit, lab in ((fit_nlook(Pw), "N-look Rice (N free)"), (fit_nlook(Pw, 11.0), "N = 11"),
                     (fit_nlook(Pw, 1.0), "N = 1 (Rice)")):
        sc = fit["pn"] / (2 * fit["N"])
        ax.plot(x, stats.ncx2.pdf(x / sc, 2 * fit["N"], 2 * fit["N"] * fit["pc"] / fit["pn"]) / sc,
                label=f"{lab}: N={fit['N']:.1f}, Pn/Pc={10*np.log10(fit['pn']/fit['pc']):.1f} dB")
    ax.set_xlabel("peak power (product units)"); ax.legend(fontsize=8)
    ax.set_title("pilot window s 40-50 km")
    fig.tight_layout(); fig.savefig(OUT / "rsr_prototype_pilot_hist.png", dpi=130)


if __name__ == "__main__":
    main()
