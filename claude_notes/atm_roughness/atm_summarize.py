"""Along-line summary of the ATM per-block results (questions 1 and 2).

    uv run claude_notes/atm_roughness/atm_summarize.py [--lines ...]

Reads outputs/atm_roughness/<line>/blocks_<date>_<block>m.json, writes
  <line>/summary_<date>.json, figures fig_{a,b,c,d}_<date>.png, fig_years.png
  (westcoast), and prints markdown tables for the results note.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import atm_roughness as ar  # noqa: E402

OUT = ar.OUT
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FAM = {"gaussian": PAL[0], "exponential": PAL[1], "powerlaw": PAL[2]}
OCT = [f"{lo}-{hi}m" for lo, hi in ar.OCTAVES]
DB = lambda x: 10 * np.log10(x)  # noqa: E731


def load(line, date, bm):
    p = OUT / line / f"blocks_{date}_{bm}m.json"
    return json.loads(p.read_text()) if p.exists() else None


def arr(blocks, fn):
    return np.array([fn(b) if fn(b) is not None else np.nan for b in blocks], float)


def fits_of(b, sector="all"):
    return b["fits"].get(f"{b['primary_class']}_{sector}", {})


def q(x, ps=(5, 25, 50, 75, 95)):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, p)) for p in ps] if len(x) else [np.nan] * len(ps)


def series_acf(x, max_lag):
    """Autocorrelation of a block series (NaN-tolerant) and the lag (in blocks)
    where it first drops below 1/e and 0.5."""
    x = x - np.nanmean(x)
    v = np.nanmean(x * x)
    ac = [1.0]
    for k in range(1, max_lag):
        a, b = x[:-k], x[k:]
        m = np.isfinite(a) & np.isfinite(b)
        ac.append(float(np.mean(a[m] * b[m]) / v) if m.sum() > 5 else np.nan)
    ac = np.array(ac)
    def first_below(th):
        i = np.where(ac < th)[0]
        return int(i[0]) if len(i) else None
    return ac, first_below(1 / np.e), first_below(0.5)


def binseg(x, min_len=5, pen_factor=4.0, max_segs=6):
    """Binary segmentation on the mean with a BIC-like penalty (log-space series)."""
    x = np.asarray(x, float)
    n = len(x)
    ok = np.isfinite(x)
    def cost(a, b):
        v = x[a:b][ok[a:b]]
        return float(np.sum((v - v.mean()) ** 2)) if len(v) else 0.0
    segs = [(0, n)]
    while len(segs) < max_segs:
        best = None
        for si, (a, b) in enumerate(segs):
            if b - a < 2 * min_len:
                continue
            c0 = cost(a, b)
            for k in range(a + min_len, b - min_len):
                gain = c0 - cost(a, k) - cost(k, b)
                if best is None or gain > best[0]:
                    best = (gain, si, k)
        if best is None:
            break
        var = np.nanvar(x)
        if best[0] < pen_factor * var * np.log(n):     # penalty per extra change point
            break
        _, si, k = best
        a, b = segs[si]; segs[si:si + 1] = [(a, k), (k, b)]
    return segs


def summarise(line, date):
    d1, d5 = load(line, date, 1000), load(line, date, 500)
    if d1 is None:
        return None
    B = d1["blocks"]; meta = d1["meta"]
    s = arr(B, lambda b: (b["s0_km"] + b["s1_km"]) / 2)
    out = dict(line=line, date=date, meta={k: v for k, v in meta.items() if k != "files"}, n_files=len(meta["files"]),
               n_blocks_1km=len(B), n_blocks_500m=len(d5["blocks"]) if d5 else 0)
    # ---- Q1: family
    fam = {}
    for f in ("gaussian", "exponential", "powerlaw"):
        best = np.array([fits_of(b).get("best") == f for b in B])
        conf = np.array([fits_of(b).get("best") == f and fits_of(b).get("dbic_second", 0) > 2 for b in B])
        white = arr(B, lambda b: fits_of(b).get(f, {}).get("runs_p"))
        rl = arr(B, lambda b: fits_of(b).get(f, {}).get("rms_log_resid"))
        fam[f] = dict(frac_best=float(best.mean()), frac_best_dbic_gt2=float(conf.mean()),
                      frac_white_p_gt_0p05=float(np.nanmean(white > 0.05)), rms_log_resid_med=float(np.nanmedian(rl)))
    g_s = arr(B, lambda b: fits_of(b).get("gaussian", {}).get("sigma_m")); g_l = arr(B, lambda b: fits_of(b).get("gaussian", {}).get("l_m"))
    e_s = arr(B, lambda b: fits_of(b).get("exponential", {}).get("sigma_m")); e_l = arr(B, lambda b: fits_of(b).get("exponential", {}).get("l_m"))
    H = arr(B, lambda b: fits_of(b).get("powerlaw", {}).get("H"))
    out["q1"] = dict(families=fam, gauss_sigma_q=q(g_s), gauss_l_q=q(g_l), exp_sigma_q=q(e_s), exp_l_q=q(e_l),
                     H_q=q(H), beta_q=q(2 * H + 2), primary_class=meta["noise"].get("primary_class"))
    # ---- Q2: octave series, variability, correlation length, regimes
    o1 = {k: arr(B, lambda b: b["octave_rms"][k]) for k in OCT}
    o5 = {k: arr(d5["blocks"], lambda b: b["octave_rms"][k]) for k in OCT} if d5 else {}
    var = {}
    for k in OCT:
        x = o1[k]; x5 = o5.get(k)
        lx = np.log10(np.where(x > 0, x, np.nan))
        ac, le, l50 = series_acf(lx, min(40, len(lx) // 2))
        var[k] = dict(q_1km_m=q(x), cv_1km=float(np.nanstd(x) / np.nanmean(x)), p95_over_p5_db=float(20 * np.log10(q(x)[4] / q(x)[0])) if q(x)[0] > 0 else np.nan,
                      corr_len_1e_km=le, corr_len_50_km=l50,
                      q_500m_m=q(x5) if x5 is not None else None, cv_500m=float(np.nanstd(x5) / np.nanmean(x5)) if x5 is not None else None)
    # estimator noise vs real variability: the two 500 m halves of each 1 km block
    if d5:
        s5 = arr(d5["blocks"], lambda b: b["s0_km"])
        for k in OCT:
            x5 = np.log10(np.where(o5[k] > 0, o5[k], np.nan))
            diffs, means = [], []
            for b, x1 in zip(B, o1[k]):
                m = (s5 >= b["s0_km"] - 1e-6) & (s5 < b["s1_km"] - 1e-6)
                if m.sum() == 2 and np.all(np.isfinite(x5[m])):
                    diffs.append(x5[m][1] - x5[m][0])
            diffs = np.array(diffs)
            lx1 = np.log10(np.where(o1[k] > 0, o1[k], np.nan))
            var[k]["within_block_std_db"] = float(20 * np.std(diffs) / np.sqrt(2)) if len(diffs) > 5 else None   # per-500 m estimate noise + sub-km variability
            var[k]["between_block_std_db"] = float(20 * np.nanstd(lx1))
            var[k]["real_fraction_of_variance"] = float(max(0.0, 1 - (np.std(diffs) ** 2 / 2 / 2) / np.nanvar(lx1))) if len(diffs) > 5 and np.nanvar(lx1) > 0 else None
    # regimes on the 4-8 m octave (Bragg band centre) and on H
    key = "4-8m"
    lx = np.log10(np.where(o1[key] > 0, o1[key], np.nan))
    segs = binseg(lx)
    elev = arr(B, lambda b: b["elev_m"]); slope = arr(B, lambda b: b["slope"])
    from scipy.stats import spearmanr
    def sp(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return [float(v) for v in spearmanr(a[m], b[m])] if m.sum() > 8 else [np.nan, np.nan]
    out["q2"] = dict(octave_variability=var,
                     regimes_4_8m=[dict(s0_km=float(s[a] - 0.5), s1_km=float(s[b - 1] + 0.5), n=b - a,
                                        rms_4_8m_med_m=float(np.nanmedian(o1[key][a:b])), H_med=float(np.nanmedian(H[a:b])),
                                        elev_med_m=float(np.nanmedian(elev[a:b]))) for a, b in segs],
                     spearman_rms48_vs_elev=sp(lx, elev), spearman_rms48_vs_slope=sp(lx, slope), spearman_rms48_vs_s=sp(lx, s),
                     spearman_H_vs_elev=sp(H, elev), spearman_H_vs_s=sp(H, s),
                     elev_range_m=[float(np.nanmin(elev)), float(np.nanmax(elev))], s_range_km=[float(s.min()), float(s.max())])
    # ---- anisotropy
    an = {k: arr(B, lambda b: b["aniso_along_over_cross"][k]) for k in OCT}
    Ha = arr(B, lambda b: fits_of(b, "along").get("powerlaw", {}).get("H")); Hc = arr(B, lambda b: fits_of(b, "cross").get("powerlaw", {}).get("H"))
    la = arr(B, lambda b: fits_of(b, "along").get("gaussian", {}).get("l_m")); lc = arr(B, lambda b: fits_of(b, "cross").get("gaussian", {}).get("l_m"))
    out["anisotropy"] = dict(ratio_q={k: q(an[k]) for k in OCT}, H_along_q=q(Ha), H_cross_q=q(Hc), l_along_q=q(la), l_cross_q=q(lc))
    # ---- Bragg PSD: per block best-fit, gaussian-fit, powerlaw-fit; pooled line fit; grid non-parametric
    br = {}
    for name, lam in ar.BRAGG.items():
        best = arr(B, lambda b: b["psd_bragg_m4"].get(name, {}).get(fits_of(b).get("best")) if fits_of(b).get("best") else None)
        gg = arr(B, lambda b: b["psd_bragg_m4"].get(name, {}).get("gaussian")); pl = arr(B, lambda b: b["psd_bragg_m4"].get(name, {}).get("powerlaw"))
        k = 2 * np.pi / lam
        grid, floors, above = [], [], []
        for b in B:
            gp = b.get("grid_psd")
            if gp and lam >= 2.0:
                kk = np.array(gp["k"]); rr = np.array(gp["radial"], float)
                floor = np.nanmedian(rr[2 * np.pi / kk < 3.0])          # white floor: cross-scan noise x mask
                v = float(np.interp(np.log(k), np.log(kk), rr))
                floors.append(floor); above.append(v > 2 * floor)
                grid.append(v - floor if v > 2 * floor else np.nan)
        grid = np.array(grid)
        br[name] = dict(wavelength_m=lam, best_db_q=q(DB(best)), gaussian_db_q=q(DB(gg)), powerlaw_db_q=q(DB(pl)),
                        grid_db_q=q(DB(grid)) if np.isfinite(grid).any() else None,
                        grid_floor_db_med=float(DB(np.median(floors))) if floors else None,
                        grid_frac_above_floor=float(np.mean(above)) if above else None,
                        fixture_db=float(DB(ar.S_gauss(k, ar.FIX_SIGMA ** 2, ar.FIX_L))),
                        extrapolated=lam < 1.8)
    out["bragg"] = br
    # pooled line-level D(r) -> single fit; misfit vs blocks at Bragg
    cls = meta["noise"].get("primary_class", "same")
    lag = np.array(B[0]["D"]["lag"]); Ds = np.zeros(len(lag)); Ns = np.zeros(len(lag))
    for b in B:
        D = np.array([np.nan if v is None else v for v in b["D"][f"{cls}_all"]]); N = np.array(b["N"][f"{cls}_all"])
        Ds += np.nan_to_num(D * N); Ns += N
    Dp = Ds / np.maximum(Ns, 1)
    n0 = meta["noise"].get("ranging_sigma_m", np.nan) ** 2 if cls == "same" else None
    pooled = ar.fit_D(lag, Dp, Ns, fixed_n0=n0)
    out["pooled_fit"] = {k: v for k, v in pooled.items() if k in ("best", "dbic_second", "gaussian", "exponential", "powerlaw")}
    if pooled.get("best"):
        pb = {name: ar.psd_from_fit(pooled, 2 * np.pi / lam)[pooled["best"]] for name, lam in ar.BRAGG.items()}
        out["pooled_bragg_db"] = {k: float(DB(v)) for k, v in pb.items()}
        # misfit: per-block best-fit S at Bragg minus pooled S, in dB
        out["pooled_misfit_db_q"] = {name: q(DB(arr(B, lambda b: b["psd_bragg_m4"].get(name, {}).get(fits_of(b).get("best")) if fits_of(b).get("best") else None)) - DB(pb[name]))
                                     for name in ar.BRAGG}
    # octave RMS misfit of a single line-level octave table (p95/p5 already in var)
    out["fixture_octave_rms_m"] = fixture_octaves()
    (OUT / line / f"summary_{date}.json").write_text(json.dumps(out, indent=1, default=float))
    figures(line, date, B, s, o1, an, H, g_l, segs)
    return out


def fixture_octaves():
    out = {}
    for lo, hi in ar.OCTAVES:
        k1, k2 = 2 * np.pi / hi, 2 * np.pi / lo
        v = ar.FIX_SIGMA ** 2 * (np.exp(-(k1 * ar.FIX_L) ** 2 / 4) - np.exp(-(k2 * ar.FIX_L) ** 2 / 4))
        out[f"{lo}-{hi}m"] = float(np.sqrt(v))
    return out


def figures(line, date, B, s, o1, an, H, g_l, segs):
    od = OUT / line
    # (a) octave RMS vs s + fitted params
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i, k in enumerate(OCT):
        ax[0].plot(s, 100 * o1[k], color=PAL[i], lw=1.2, label=k)
    ax[0].set(ylabel="octave RMS (cm)", yscale="log", title=f"{line} {date}: band-limited RMS per 1 km block (primary pair class)")
    for a, b in segs[1:]:
        ax[0].axvline(s[a] - 0.5, color="0.6", lw=0.8, ls=":")
    ax[0].legend(ncol=6, fontsize=8, frameon=False)
    ax[1].plot(s, H, color=PAL[2], lw=1.2, label="power-law H")
    ax[1].set(ylabel="Hurst H", ylim=(0, 1)); ax[1].legend(frameon=False, fontsize=8)
    ax[2].plot(s, g_l, color=PAL[0], lw=1.2, label="Gaussian l (m)")
    ax[2].set(ylabel="Gaussian l (m)", yscale="log", xlabel="s along radar line (km)"); ax[2].legend(frameon=False, fontsize=8)
    for a_ in ax:
        a_.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"fig_a_octaves_{date}.png", dpi=130); plt.close(fig)
    # (b) example D(r) + PSD with fits: typical (median rms 4-8) and extreme (max)
    r48 = o1["4-8m"]
    order = np.argsort(np.nan_to_num(r48, nan=-1))
    typ, ext = B[order[len(order) // 2]], B[order[-1]]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for col, (b, tag) in enumerate(((typ, "typical"), (ext, "roughest"))):
        cls = b["primary_class"]; lag = np.array(b["D"]["lag"])
        D = np.array([np.nan if v is None else v for v in b["D"][f"{cls}_all"]]); N = np.array(b["N"][f"{cls}_all"])
        Dx = np.array([np.nan if v is None else v for v in b["D"]["xscan_all"]])
        m = N > 30
        ax[0, col].loglog(lag[m], D[m], "o", ms=4, color="0.2", mfc="none", label=f"D(r) {cls}-scan pairs (primary)")
        ax[0, col].loglog(lag, Dx, ".", ms=3, color="0.6", label="D(r) cross-scan pairs")
        f = fits_of(b)
        rr = np.geomspace(0.25, 150, 200)
        for name, fn in (("gaussian", ar.D_gauss), ("exponential", ar.D_exp), ("powerlaw", ar.D_pl)):
            p = f.get(name, {})
            if "sigma_m" in p:
                ax[0, col].loglog(rr, fn(rr, p["sigma_m"] ** 2, p["l_m"], p["noise_sigma_m"] ** 2), color=FAM[name], lw=1.5,
                                  label=f"{name} s={p['sigma_m']*100:.1f}cm l={p['l_m']:.1f}m BIC={p['bic']:.0f}")
            elif "H" in p:
                ax[0, col].loglog(rr, fn(rr, p["c"], p["H"], p["noise_sigma_m"] ** 2), color=FAM[name], lw=1.5,
                                  label=f"power law H={p['H']:.2f} BIC={p['bic']:.0f}")
        ax[0, col].axvspan(ar.FIT_LAGS[0], ar.FIT_LAGS[1], color="0.93", zorder=0)
        ax[0, col].set(xlabel="lag r (m)", ylabel="D(r) (m$^2$)", title=f"{tag} block s={b['s0_km']:.0f}-{b['s1_km']:.0f} km")
        ax[0, col].legend(fontsize=7, frameon=False)
        # PSD panel: model 2-D PSDs + grid radial + fixture
        kk = np.geomspace(2 * np.pi / 200, 2 * np.pi / 0.5, 200)
        for name in FAM:
            p = f.get(name, {})
            S = ar.S_gauss(kk, p["sigma_m"] ** 2, p["l_m"]) if name == "gaussian" and "sigma_m" in p else \
                ar.S_exp(kk, p["sigma_m"] ** 2, p["l_m"]) if name == "exponential" and "sigma_m" in p else \
                ar.S_pl(kk, p["A"], p["beta"]) if name == "powerlaw" and "A" in p else None
            if S is not None:
                ax[1, col].loglog(2 * np.pi / kk, S, color=FAM[name], lw=1.5, label=name)
        ax[1, col].loglog(2 * np.pi / kk, ar.S_gauss(kk, ar.FIX_SIGMA ** 2, ar.FIX_L), color="0.3", ls="--", lw=1.2, label="C&S fixture 4.9 cm / 2.98 m")
        gp = b.get("grid_psd")
        if gp:
            kg = np.array(gp["k"]); ax[1, col].loglog(2 * np.pi / kg, np.array(gp["radial"], float), "s", ms=3, color="0.5", mfc="none",
                                                     label=f"1 m grid Welch (fill {gp['fill']:.2f}, cross-scan noise incl.)")
        for name, lam in ar.BRAGG.items():
            ax[1, col].axvline(lam, color="0.8", lw=0.8)
        ax[1, col].set(xlabel="wavelength (m)", ylabel="2-D PSD S(k) (m$^4$)", ylim=(1e-9, 1e2), xlim=(0.5, 200))
        ax[1, col].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{line} {date}: structure function and PSD fits; vertical lines = Bragg 5/1.5/1/0.75 m")
    fig.tight_layout(); fig.savefig(od / f"fig_b_fits_{date}.png", dpi=130); plt.close(fig)
    # (c) family strip
    fig, ax = plt.subplots(figsize=(11, 1.8))
    for b in B:
        f = fits_of(b); best = f.get("best")
        if best:
            ax.bar(b["s0_km"], 1, width=b["s1_km"] - b["s0_km"], color=FAM[best], alpha=1 if f.get("dbic_second", 0) > 2 else 0.4, align="edge", lw=0)
    for name, c in FAM.items():
        ax.bar([], [], color=c, label=name)
    ax.set(yticks=[], xlabel="s (km)", title="best family by BIC per 1 km block (faint = dBIC < 2)"); ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(od / f"fig_c_family_{date}.png", dpi=130); plt.close(fig)
    # (d) anisotropy
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for i, k in enumerate(OCT[1:]):
        ax.plot(s, an[k], color=PAL[i + 1], lw=1.2, label=k)
    ax.axhline(1, color="0.5", lw=0.8); ax.set(ylabel="along / cross octave RMS", xlabel="s (km)", yscale="log", ylim=(0.3, 3),
                                                title=f"{line} {date}: anisotropy ratio per block"); ax.legend(ncol=5, fontsize=8, frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(od / f"fig_d_aniso_{date}.png", dpi=130); plt.close(fig)


def years_figure(line, dates):
    data = [(d, load(line, d, 1000)) for d in dates]
    data = [(d, x) for d, x in data if x]
    if len(data) < 2:
        return None
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    res = {}
    for i, (d, x) in enumerate(data):
        B = x["blocks"]; s = arr(B, lambda b: (b["s0_km"] + b["s1_km"]) / 2)
        ax[0].plot(s, 100 * arr(B, lambda b: b["octave_rms"]["4-8m"]), color=PAL[i], lw=1.2, label=d)
        ax[1].plot(s, 100 * arr(B, lambda b: b["octave_rms"]["16-32m"]), color=PAL[i], lw=1.2, label=d)
        ax[2].plot(s, arr(B, lambda b: fits_of(b).get("powerlaw", {}).get("H")), color=PAL[i], lw=1.2, label=d)
        res[d] = dict(s=s, r48=arr(B, lambda b: b["octave_rms"]["4-8m"]), r16=arr(B, lambda b: b["octave_rms"]["16-32m"]),
                      H=arr(B, lambda b: fits_of(b).get("powerlaw", {}).get("H")),
                      S195=arr(B, lambda b: b["psd_bragg_m4"].get("195MHz", {}).get(fits_of(b).get("best")) if fits_of(b).get("best") else None),
                      S60=arr(B, lambda b: b["psd_bragg_m4"].get("60MHz", {}).get(fits_of(b).get("best")) if fits_of(b).get("best") else None))
    ax[0].set(ylabel="4-8 m octave RMS (cm)", yscale="log", title=f"{line}: year-to-year, 1 km blocks on the shared axis")
    ax[1].set(ylabel="16-32 m octave RMS (cm)", yscale="log"); ax[2].set(ylabel="power-law H", xlabel="s (km)", ylim=(0, 1))
    for a in ax:
        a.legend(fontsize=8, frameon=False); a.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / line / "fig_years.png", dpi=130); plt.close(fig)
    # pairwise comparisons on a common s grid
    sg = np.arange(0, 120, 1.0)
    def on_grid(v, s):
        m = np.isfinite(v); return np.interp(sg, s[m], v[m], left=np.nan, right=np.nan) if m.sum() > 2 else np.full(len(sg), np.nan)
    pairs = {}
    ds = list(res)
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            a, b = res[ds[i]], res[ds[j]]
            m = (sg >= max(a["s"].min(), b["s"].min())) & (sg <= min(a["s"].max(), b["s"].max()))
            out = {}
            for key in ("r48", "r16", "S195", "S60"):
                va, vb = on_grid(a[key], a["s"])[m], on_grid(b[key], b["s"])[m]
                ok = np.isfinite(va) & np.isfinite(vb) & (va > 0) & (vb > 0)
                ratio_db = (20 if key.startswith("r") else 10) * np.log10(vb[ok] / va[ok])
                out[key] = dict(median_db=float(np.median(ratio_db)), iqr_db=[float(np.percentile(ratio_db, 25)), float(np.percentile(ratio_db, 75))],
                                corr_log=float(np.corrcoef(np.log(va[ok]), np.log(vb[ok]))[0, 1]) if ok.sum() > 5 else np.nan, n=int(ok.sum()))
            va, vb = on_grid(a["H"], a["s"])[m], on_grid(b["H"], b["s"])[m]; ok = np.isfinite(va) & np.isfinite(vb)
            out["H"] = dict(median_diff=float(np.median(vb[ok] - va[ok])), corr=float(np.corrcoef(va[ok], vb[ok])[0, 1]) if ok.sum() > 5 else np.nan)
            pairs[f"{ds[i]} -> {ds[j]}"] = out
    (OUT / line / "summary_years.json").write_text(json.dumps(pairs, indent=1, default=float))
    return pairs


def md_tables(S):
    """Markdown tables for the note."""
    L = []
    L.append("| line | date | class | blocks | best: G / E / PL (frac, dBIC>2) | white p>0.05: G / E / PL | Gauss sigma cm (p5/50/95) | Gauss l m (p5/50/95) | H (p5/50/95) | beta |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for o in S:
        f = o["q1"]["families"]; g = o["q1"]
        L.append(f"| {o['line']} | {o['date']} | {g['primary_class']} | {o['n_blocks_1km']} | "
                 + " / ".join(f"{f[k]['frac_best']:.2f} ({f[k]['frac_best_dbic_gt2']:.2f})" for k in ("gaussian", "exponential", "powerlaw"))
                 + " | " + " / ".join(f"{f[k]['frac_white_p_gt_0p05']:.2f}" for k in ("gaussian", "exponential", "powerlaw"))
                 + f" | {g['gauss_sigma_q'][0]*100:.1f}/{g['gauss_sigma_q'][2]*100:.1f}/{g['gauss_sigma_q'][4]*100:.1f}"
                 + f" | {g['gauss_l_q'][0]:.1f}/{g['gauss_l_q'][2]:.1f}/{g['gauss_l_q'][4]:.1f}"
                 + f" | {g['H_q'][0]:.2f}/{g['H_q'][2]:.2f}/{g['H_q'][4]:.2f} | {g['beta_q'][2]:.2f} |")
    L.append("")
    L.append("Octave RMS (cm), 1 km blocks, median [p5-p95]; CV; along-track correlation length of the block series (1/e, km); 500 m CV")
    L.append("| line | date | " + " | ".join(OCT) + " |")
    L.append("|---|---|" + "---|" * len(OCT))
    for o in S:
        v = o["q2"]["octave_variability"]
        L.append(f"| {o['line']} | {o['date']} | " + " | ".join(
            f"{v[k]['q_1km_m'][2]*100:.1f} [{v[k]['q_1km_m'][0]*100:.1f}-{v[k]['q_1km_m'][4]*100:.1f}] CV {v[k]['cv_1km']:.2f} Lc {v[k]['corr_len_1e_km']} between {v[k].get('between_block_std_db', float('nan')):.1f} dB within {v[k].get('within_block_std_db') or float('nan'):.1f} dB real {v[k].get('real_fraction_of_variance') if v[k].get('real_fraction_of_variance') is None else round(v[k]['real_fraction_of_variance'], 2)}" for k in OCT) + " |")
    L.append("")
    L.append("Anisotropy along/cross octave-RMS ratio, median [p5-p95]; H along / cross (median)")
    L.append("| line | date | " + " | ".join(OCT[1:]) + " | H along/cross |")
    L.append("|---|---|" + "---|" * len(OCT))
    for o in S:
        a = o["anisotropy"]
        L.append(f"| {o['line']} | {o['date']} | " + " | ".join(f"{a['ratio_q'][k][2]:.2f} [{a['ratio_q'][k][0]:.2f}-{a['ratio_q'][k][4]:.2f}]" for k in OCT[1:])
                 + f" | {a['H_along_q'][2]:.2f} / {a['H_cross_q'][2]:.2f} |")
    L.append("")
    L.append("2-D PSD at the Bragg wavelength, dB re 1 m^4: per-block best-family median [p5-p95]; pooled single line fit; per-block-vs-pooled misfit p5-p95 (dB); Gaussian-fit median; 1 m grid non-parametric median; fixture")
    L.append("| line | date | band | best med [p5-p95] | pooled | misfit p5..p95 | Gauss fit | PL fit | grid | fixture |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for o in S:
        for name, b in o["bragg"].items():
            mis = o.get("pooled_misfit_db_q", {}).get(name, [np.nan] * 5)
            L.append(f"| {o['line']} | {o['date']} | {name} ({b['wavelength_m']} m{'*' if b['extrapolated'] else ''}) | {b['best_db_q'][2]:.1f} [{b['best_db_q'][0]:.1f}, {b['best_db_q'][4]:.1f}] "
                     f"| {o.get('pooled_bragg_db', {}).get(name, np.nan):.1f} | {mis[0]:.1f}..{mis[4]:.1f} | {b['gaussian_db_q'][2]:.1f} | {b['powerlaw_db_q'][2]:.1f} | "
                     + (f"{b['grid_db_q'][2]:.1f} (floor {b['grid_floor_db_med']:.0f}, {b['grid_frac_above_floor']:.0%} above)" if b['grid_db_q'] else (f"below floor {b['grid_floor_db_med']:.0f}" if b.get('grid_floor_db_med') is not None else "-")) + f" | {b['fixture_db']:.1f} |")
    L.append("")
    L.append("Regimes (binary segmentation of log 4-8 m octave RMS, 1 km blocks) and Spearman correlations")
    for o in S:
        q2 = o["q2"]
        L.append(f"- {o['line']} {o['date']}: elev {q2['elev_range_m'][0]:.0f}-{q2['elev_range_m'][1]:.0f} m; rho(rms48, elev)={q2['spearman_rms48_vs_elev'][0]:.2f}, "
                 f"rho(rms48, slope)={q2['spearman_rms48_vs_slope'][0]:.2f}, rho(rms48, s)={q2['spearman_rms48_vs_s'][0]:.2f}, rho(H, elev)={q2['spearman_H_vs_elev'][0]:.2f}; segments: "
                 + "; ".join(f"s {r['s0_km']:.0f}-{r['s1_km']:.0f} km rms48 {r['rms_4_8m_med_m']*100:.1f} cm H {r['H_med']:.2f} elev {r['elev_med_m']:.0f}" for r in q2["regimes_4_8m"]))
    L.append("")
    L.append("Noise per flight (m): " + "; ".join(f"{o['line']} {o['date']}: ranging {o['meta']['noise'].get('ranging_sigma_m', float('nan')):.3f} "
                                                  f"[{', '.join(f'{v:.3f}' for v in o['meta']['noise'].get('ranging_sigma_range_m', [np.nan, np.nan]))}], "
                                                  f"scan-to-scan {o['meta']['noise'].get('scan_to_scan_sigma_median_m', float('nan')):.3f}, "
                                                  f"crossover total {o['meta']['noise'].get('xscan_smallest_bin_sigma_m', float('nan')):.3f} at {o['meta']['noise'].get('xscan_smallest_lag_m', float('nan')):.2f} m, "
                                                  f"cross-scan block nugget {o['meta']['noise'].get('xscan_block_nugget_median_sigma_m', float('nan')):.3f}" for o in S))
    L.append("")
    L.append("Fixture octave RMS (cm): " + ", ".join(f"{k} {v*100:.2f}" for k, v in S[0]["fixture_octave_rms_m"].items()))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lines", nargs="*"); a = ap.parse_args()
    lines = a.lines or sorted(p.name for p in OUT.iterdir() if p.is_dir())
    S = []
    for line in lines:
        dates = sorted({Path(p).stem.split("_")[1] for p in glob.glob(str(OUT / line / "blocks_*_1000m.json"))})
        for d in dates:
            o = summarise(line, d)
            if o:
                S.append(o); print(f"summarised {line} {d}: {o['n_blocks_1km']} blocks")
        if len(dates) > 1:
            yp = years_figure(line, dates)
            if yp:
                print(f"\nYear-to-year {line}:"); print(json.dumps(yp, indent=1))
    md = md_tables(S)
    (OUT / "summary_tables.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
