"""Spatial structure, distributions, covariate relations, GMM regime candidates on the
10 km grids and the platelet series. Run: uv run --with scikit-learn claude_notes/atm_regional/atm2_analyze.py
Writes outputs/atm_regional/{variogram,hist,relations,clusters}_*.png, analysis.json,
grid_<hemi>_clusters.csv (cell -> regime candidate).
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "cache" / "atm2"
OUT = ROOT / "outputs" / "atm_regional"
RNG = np.random.default_rng(0)
NAME = {"gl": "Greenland", "aa": "Antarctica"}
R_FLOOR = 0.5  # cm; log r floored here for cells at the noise floor


def along_track_variogram(df, hemi, lags_km=np.logspace(-1, 2.3, 18)):
    """Semivariance of log10(rms) (observable, floor-inclusive) and log10(max(r,0.5)) along
    centre-platelet series per flight day; along-track distance from consecutive platelet spacing."""
    acc = {k: [np.zeros(len(lags_km) - 1), np.zeros(len(lags_km) - 1)] for k in ("rms", "r")}
    for d, g in df[df.centre & (df.hemi == hemi)].groupby("date"):
        g = g.sort_values("utc_s")
        if len(g) < 500: continue
        dx = np.hypot(np.diff(g.x.values), np.diff(g.y.values)) / 1e3
        dx[dx > 1.0] = np.nan  # gaps between flight segments break the series
        s = np.concatenate([[0], np.nancumsum(np.nan_to_num(dx))]); bad = np.concatenate([[False], np.isnan(dx)])
        seg = np.cumsum(bad)
        vals = {"rms": np.log10(g.rms_cm.values), "r": np.log10(np.maximum(g.r_cm.values, R_FLOOR))}
        n = len(g)
        for k in np.unique(np.round(np.logspace(0, np.log10(n / 4), 40)).astype(int)):
            h = s[k:] - s[:-k]; same = seg[k:] == seg[:-k]
            for key, v in vals.items():
                dv2 = 0.5 * (v[k:] - v[:-k]) ** 2
                idx = np.digitize(h[same], lags_km) - 1; ok = (idx >= 0) & (idx < len(lags_km) - 1)
                np.add.at(acc[key][0], idx[ok], dv2[same][ok]); np.add.at(acc[key][1], idx[ok], 1)
    mid = np.sqrt(lags_km[1:] * lags_km[:-1])
    return mid, {k: np.where(c > 100, sv / np.maximum(c, 1), np.nan) for k, (sv, c) in acc.items()}


def grid_variogram(agg, col, lags_km=np.logspace(1, 3.3, 24), npairs=3_000_000):
    v = agg[col].values; x = agg.xc.values / 1e3; y = agg.yc.values / 1e3
    i = RNG.integers(0, len(v), npairs); j = RNG.integers(0, len(v), npairs)
    h = np.hypot(x[i] - x[j], y[i] - y[j]); dv2 = 0.5 * (v[i] - v[j]) ** 2
    idx = np.digitize(h, lags_km) - 1; ok = (idx >= 0) & (idx < len(lags_km) - 1)
    sv = np.bincount(idx[ok], dv2[ok], minlength=len(lags_km) - 1); c = np.bincount(idx[ok], minlength=len(lags_km) - 1)
    return np.sqrt(lags_km[1:] * lags_km[:-1]), np.where(c > 200, sv / np.maximum(c, 1), np.nan), float(np.var(v))


def corr_length(h, gam, sill):
    """first lag where semivariance reaches (1 - 1/e) of the sill."""
    ok = np.isfinite(gam)
    if ok.sum() < 3: return np.nan
    thr = (1 - np.exp(-1)) * sill
    above = np.where(ok & (gam >= thr))[0]
    return float(h[above[0]]) if len(above) else np.inf


def gmm_bic(X, kmax=8):
    bics, models = [], []
    for k in range(1, kmax + 1):
        m = GaussianMixture(k, covariance_type="full", n_init=3, random_state=0).fit(X)
        bics.append(m.bic(X)); models.append(m)
    return np.array(bics), models


def main():
    df = pd.read_parquet(CACHE / "platelets_qc.parquet")
    res = {}
    for hemi in ("gl", "aa"):
        agg = pd.read_csv(OUT / f"grid_{hemi}.csv")
        if agg.empty: continue
        agg["logr"] = np.log10(np.maximum(agg.r_med, R_FLOOR))
        agg["above"] = agg.at_floor < 0.5
        r = {"n_cells": int(len(agg)), "frac_cells_at_floor": float((agg.at_floor >= 0.5).mean())}
        plat = df[(df.hemi == hemi) & df.centre]
        r["frac_platelets_at_floor"] = float(plat.at_floor.mean())
        r["frac_platelets_at_floor_by_elev"] = {f"{lo}-{hi}": float(plat.at_floor[(plat.h >= lo) & (plat.h < hi)].mean())
                                                for lo, hi in ((0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 4000))}
        r["floor_cm_by_year"] = plat.groupby("year").floor_cm.median().round(2).to_dict()
        r["floor_cm_by_year_p10_p90"] = {int(y): [round(float(np.percentile(g, 10)), 2), round(float(np.percentile(g, 90)), 2)]
                                          for y, g in plat.drop_duplicates("date").groupby("year").floor_cm}

        # --- variograms
        hk, gam = along_track_variogram(df, hemi)
        hg, gg, sill = grid_variogram(agg, "logr")
        hg2, gg2, sill2 = grid_variogram(agg[agg.above], "logr") if agg.above.sum() > 50 else (hg, gg * np.nan, np.nan)
        r["variogram_track"] = {"lag_km": hk.round(3).tolist(), **{k: np.round(v, 4).tolist() for k, v in gam.items()}}
        r["variogram_grid"] = {"lag_km": hg.round(1).tolist(), "gamma_logr": np.round(gg, 4).tolist(), "sill": round(sill, 4),
                               "gamma_logr_above_floor": np.round(gg2, 4).tolist(), "sill_above": round(float(sill2), 4) if np.isfinite(sill2) else None}
        r["corr_length_km_grid"] = corr_length(hg, gg, sill)
        r["corr_length_km_grid_above_floor"] = corr_length(hg2, gg2, sill2) if np.isfinite(sill2) else None
        r["corr_length_km_track_logrms"] = corr_length(hk, gam["rms"], np.nanmax(gam["rms"]))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].loglog(hk, gam["rms"], "o-", label="log10 RMS_Fit (platelet series)"); ax[0].loglog(hk, gam["r"], "s-", label="log10 r (floored 0.5 cm)")
        ax[0].set_xlabel("along-track lag (km)"); ax[0].set_ylabel("semivariance (dex^2)"); ax[0].legend(); ax[0].set_title(f"{NAME[hemi]}: platelet series")
        ax[1].semilogx(hg, gg, "o-", label=f"all cells (var {sill:.3f})"); ax[1].semilogx(hg2, gg2, "s-", label="cells above floor")
        ax[1].set_xlabel("lag (km)"); ax[1].set_ylabel("semivariance of log10 r_med"); ax[1].legend(); ax[1].set_title("10 km grid")
        fig.tight_layout(); fig.savefig(OUT / f"variogram_{hemi}.png", dpi=120); plt.close(fig)

        # --- histogram / 1-D mixture
        lr = np.log10(np.maximum(plat.r_cm.values, R_FLOOR)); lrms = np.log10(plat.rms_cm.values)
        sub = RNG.choice(lr[plat.r_cm.values > R_FLOOR], min(200_000, int((plat.r_cm.values > R_FLOOR).sum())), replace=False)
        b1, _ = gmm_bic(sub[:, None], 5)
        bc, _ = gmm_bic(agg.logr.values[agg.above][:, None], 5) if agg.above.sum() > 100 else (np.array([0]), None)
        r["gmm1d_platelets_bic_rel"] = (b1 - b1.min()).round(0).tolist(); r["gmm1d_cells_above_bic_rel"] = (bc - bc.min()).round(0).tolist()
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].hist(lrms, 80, color="0.5"); ax[0].set_title("platelet log10 RMS_Fit (cm)")
        ax[1].hist(lr[plat.r_cm.values > R_FLOOR], 80, color="C0"); ax[1].set_title(f"platelet log10 r (cm), above floor ({(plat.r_cm.values > R_FLOOR).mean():.0%})")
        ax[2].hist(agg.logr[agg.above], 50, color="C1"); ax[2].set_title("10 km cells log10 r_med, cells not at floor")
        fig.suptitle(NAME[hemi]); fig.tight_layout(); fig.savefig(OUT / f"hist_{hemi}.png", dpi=120); plt.close(fig)

        # --- relations
        rel = {}
        for c in ("h", "dist_km", "slope", "lat"):
            rho = spearmanr(agg[c], agg.logr).correlation
            rho_ab = spearmanr(agg[c][agg.above], agg.logr[agg.above]).correlation if agg.above.sum() > 30 else np.nan
            rel[c] = {"spearman_all": round(float(rho), 3), "spearman_above_floor": round(float(rho_ab), 3)}
        rel["month_medians"] = agg.groupby("month").r_med.median().round(2).to_dict()
        rel["month_at_floor"] = agg.groupby("month").at_floor.mean().round(2).to_dict()
        if hemi == "aa": rel["shelf_vs_grounded_r_med"] = agg.groupby(agg.shelf > 0.5).r_med.median().round(2).to_dict()
        r["relations"] = rel
        fig, axs = plt.subplots(1, 4, figsize=(18, 4))
        for ax, c, lg in zip(axs, ("h", "dist_km", "slope", "lat"), (False, True, True, False)):
            ax.scatter(agg[c], agg.r_med, s=2, c=np.where(agg.above, "C0", "0.7"))
            ax.set_yscale("log"); ax.set_xlabel(c); ax.set_ylabel("r_med (cm)")
            if lg: ax.set_xscale("log")
            q = pd.qcut(agg[c].rank(method="first"), 12, labels=False); m = agg.groupby(q).agg(x=(c, "median"), y=("r_med", "median"))
            ax.plot(m.x, m.y, "r-o", ms=3); ax.set_title(f"rho={rel[c]['spearman_all']:.2f} (above floor {rel[c]['spearman_above_floor']:.2f})")
        fig.suptitle(f"{NAME[hemi]}: 10 km cells (grey = at floor)"); fig.tight_layout(); fig.savefig(OUT / f"relations_{hemi}.png", dpi=120); plt.close(fig)

        # --- clustering: GMM on (log r, log slope, h, log dist), BIC over k
        feats = pd.DataFrame({"logr": agg.logr, "logslope": np.log10(np.maximum(agg.slope, 1e-4)), "h_km": agg.h / 1e3,
                              "logdist": np.log10(np.maximum(agg.dist_km, 1))})
        X = StandardScaler().fit(feats).transform(feats)
        bics, models = gmm_bic(X, 6)
        # BIC keeps falling with k on 10^3-10^4 cells; stop where adding a component gains < 10 % of the total drop
        gain = -np.diff(bics); tot = bics[0] - bics.min()
        k = 1
        for g_ in gain:
            if g_ < 0.10 * tot: break
            k += 1
        kbic = int(np.argmin(bics) + 1)
        m = models[k - 1]; lab = m.predict(X)
        order = np.argsort([agg.logr[lab == i].median() for i in range(k)]); remap = {o: i for i, o in enumerate(order)}
        agg["regime"] = [remap[l] for l in lab]
        sig = agg.groupby("regime").agg(n=("logr", "size"), r_med=("r_med", "median"), r_p10=("r_med", lambda s: s.quantile(.1)), r_p90=("r_med", lambda s: s.quantile(.9)),
                                        at_floor=("at_floor", "mean"), slope=("slope", "median"), h=("h", "median"), dist_km=("dist_km", "median"),
                                        lat=("lat", "median"), shelf=("shelf", "mean"), n_years_med=("n_years", "median"))
        r["gmm4d"] = {"bic_rel": (bics - bics.min()).round(0).tolist(), "k_bic_min": kbic, "k_elbow_used": k,
                      "signatures": sig.round(3).reset_index().to_dict(orient="records")}
        agg[["cell", "xc", "yc", "xm", "ym", "lat", "lon", "h", "slope", "dist_km", "r_med", "at_floor", "regime", "logr", "above", "n_years", "years"]].to_csv(OUT / f"grid_{hemi}_clusters.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        sc = ax[0].scatter(agg.xc / 1e3, agg.yc / 1e3, c=agg.regime, s=3, cmap="tab10", vmin=-0.5, vmax=9.5); ax[0].set_aspect("equal")
        plt.colorbar(sc, ax=ax[0], ticks=range(k)); ax[0].set_title(f"{NAME[hemi]}: GMM regime candidates (k={k}, ordered by r)")
        for i in range(k):
            s = agg[agg.regime == i]; ax[1].scatter(s.h, s.r_med, s=3, label=f"{i}: n={len(s)} r={s.r_med.median():.1f} slope={s.slope.median():.3f} d={s.dist_km.median():.0f} km")
        ax[1].set_yscale("log"); ax[1].set_xlabel("elevation (m)"); ax[1].set_ylabel("r_med (cm)"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / f"clusters_{hemi}.png", dpi=120); plt.close(fig)
        res[hemi] = r
        print(hemi, json.dumps({k: v for k, v in r.items() if "variogram" not in k}, indent=None, default=str)[:1500])
    json.dump(res, open(OUT / "analysis.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
