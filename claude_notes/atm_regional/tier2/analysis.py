"""Tier 2 analysis: family verdicts, exponential (sigma, l) tables/maps, adequacy, year
partition, variograms, GMM/GBT grouping. Reads rows.parquet (+ covariates.csv if present).
  uv run --with scikit-learn --with tabulate claude_notes/atm_regional/tier2/analysis.py
Writes outputs/atm_regional/tier2/{tables_*.md, analysis.json, map_*.png, fig_*.png, site_medians.csv}"""
from __future__ import annotations
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT, ATM2
warnings.filterwarnings("ignore")
STRATUM_NAME = {"gl_0": "GL interior", "gl_1": "GL percolation belt", "gl_2": "GL margin", "aa_0": "AA shelf/sea level", "aa_1": "AA grounded"}
HB = {"h0": "<500 m", "h1": "500-1500", "h2": "1500-2500", "h3": ">2500"}
BR = ["60MHz", "195MHz", "300MHz", "400MHz"]


def q(x, p): return float(np.nanpercentile(x, p)) if np.isfinite(x).sum() else np.nan


def load():
    df = pd.read_parquet(OUT / "rows.parquet")
    df = df[df.status == "ok"].copy()
    cov = OUT / "covariates.csv"
    if cov.exists():
        c = pd.read_csv(cov).drop(columns=["hemi"], errors="ignore"); df = df.merge(c, on="site", how="left")
    else:
        df["facies_proxy"] = facies_fallback(df)
    df["facies_proxy"] = facies_rule(df)
    df["stratum_name"] = df.apply(lambda r: stratum_name(r), axis=1)
    df["log_sigma"] = np.log10(df.e_sigma); df["log_l"] = np.log10(df.e_l)
    df["l_capped"] = df.e_l >= 250          # exponential l at the fit bound: no outer scale within 30 m
    df["e_sigma_cm"] = df.e_sigma * 100; df["sigma_bl30_cm"] = df.sigma_bl30_m * 100
    df["e_white"] = df.e_runs_p > 0.05
    # reference family = best BIC among exponential / power law / Matern (Gaussian excluded: unphysical Bragg tail)
    bics = df[["e_bic", "pl_bic", "m_bic"]].values
    ref = np.array(["exponential", "powerlaw", "matern"])[np.nanargmin(np.where(np.isfinite(bics), bics, np.inf), axis=1)]
    df["ref"] = ref
    for b in BR:
        df[f"S_{b}_ref_dB"] = [r[f"S_{b}_{f}_dB"] for f, (_, r) in zip(ref, df.iterrows())]
        df[f"bragg_{b}_vs_best"] = df[f"S_{b}_exponential_dB"] - df[f"S_{b}_ref_dB"]
        df[f"S_{b}_best_dB"] = df[f"S_{b}_ref_dB"]
    df["mis15"] = df["bragg_195MHz_vs_best"]; df["mis10"] = df["bragg_300MHz_vs_best"]
    df["mis15_m"] = df["bragg_195MHz_vs_matern"]; df["mis10_m"] = df["bragg_300MHz_vs_matern"]
    df["adequate_bragg_only"] = ((df.mis15.abs() < 3) & (df.mis10.abs() < 3)).astype(float)
    df["adequate"] = (df.adequate_bragg_only.astype(bool) & df.e_white).astype(float)
    df["best3"] = np.where(df.best3 == "gaussian", df.best3, df.best3)
    df["lowq"] = (df.n_used < 20000) | (df.rms_105 if "rms_105" in df else False)
    return df


def facies_rule(df):
    """GL: elevation/latitude rule (ELA 1600 m@65N -> 1000 m@80N; dry-snow line 2500 m@66N -> 2000 m@78N);
    'wet_snow' where the rule says percolation and MEaSUREs melt days > 30 (2010-2012 mean). Melt = 0 is
    treated as no-data (25 km grid misses peripheral ice). AA: shelf (dist 0 / h < 100 m), coastal < 1500 m,
    interior; '_melt' suffix where RACMO snowmelt months imply > 30 melt d/yr."""
    out = []
    for _, r in df.iterrows():
        h = r.h_cov if np.isfinite(r.h_cov) else r.elev_m; lat = abs(r.lat); md = r.get("melt_days", np.nan)
        if r.hemi == "gl":
            ela = np.interp(lat, [65, 80], [1600, 1000]); dsl = np.interp(lat, [66, 78], [2500, 2000])
            f = "ablation" if h < ela else "percolation" if h < dsl else "dry_snow"
            if f == "percolation" and np.isfinite(md) and md > 30: f = "wet_snow"
        else:
            f = "shelf" if (r.dist_km == 0 or h < 100) else "coastal" if h < 1500 else "interior"
            if f != "shelf" and np.isfinite(md) and md > 30: f += "_melt"
        out.append(f)
    return out


def facies_fallback(df):
    out = []
    for _, r in df.iterrows():
        h, lat = r.h_cov if np.isfinite(r.h_cov) else r.elev_m, abs(r.lat)
        if r.hemi == "gl":
            ela = 1600 - 40 * (lat - 65); dsl = 2500 - 42 * (lat - 66)
            out.append("ablation" if h < ela else "percolation" if h < dsl else "dry_snow")
        else:
            out.append("shelf" if (r.dist_km == 0 or h < 100) else "coastal" if h < 1500 else "interior")
    return out


def stratum_name(r):
    if r.stratum == "none" or str(r.stratum).startswith("tr_"):
        return "GL interior transect" if str(r.stratum).startswith("tr_") else "ground truth"
    reg, hb = str(r.stratum).split("_")
    return f"{STRATUM_NAME.get(f'{r.hemi}_{reg}', r.stratum)} {HB.get(hb, hb)}"


def site_medians(df):
    g = df.groupby("site")
    m = g.agg(hemi=("hemi", "first"), lat=("lat", "first"), lon=("lon", "first"), x=("x", "first"), y=("y", "first"),
              stratum=("stratum", "first"), stratum_name=("stratum_name", "first"), kind=("kind", "first"), facies=("facies_proxy", "first"),
              h=("elev_m", "median"), slope=("slope_100m", "median"), dist_km=("dist_km", "first"), n_years=("year", "nunique"), n_visits=("year", "size"),
              e_sigma_cm=("e_sigma_cm", "median"), e_l=("e_l", "median"), sigma_bl30_cm=("sigma_bl30_cm", "median"), nu=("m_nu", "median"), H=("pl_H", "median"),
              adequate=("adequate", "mean"), adequate_bragg=("adequate_bragg_only", "mean"), white=("e_white", "mean"),
              mis15=("mis15", "median"), mis10=("mis10", "median"), mis15_m=("mis15_m", "median"), mis10_m=("mis10_m", "median"),
              exp_best=("best3", lambda x: (x == "exponential").mean()), pl_best=("best3", lambda x: (x == "powerlaw").mean()),
              g_best=("best3", lambda x: (x == "gaussian").mean()), l_capped=("l_capped", "mean"),
              S195=("S_195MHz_best_dB", "median"), S60=("S_60MHz_best_dB", "median"), S300=("S_300MHz_best_dB", "median"),
              rms_2_4=("rms_2-4m", "median"), rms_4_8=("rms_4-8m", "median"), rms_16_32=("rms_16-32m", "median"),
              aniso_4_8=("aniso_4-8m", "median"), aniso_8_16=("aniso_8-16m", "median"))
    for c in ("melt_days", "smb_mmwe", "wind10_ms"):
        if c in df: m[c] = g[c].first()
    return m.reset_index()


def table_by(df, key, label):
    rows = []
    for k, s in df.groupby(key):
        rows.append({label: k, "site-years": len(s), "sites": s.site.nunique(),
                     "best3 G/E/PL %": f"{100 * (s.best3 == 'gaussian').mean():.0f}/{100 * (s.best3 == 'exponential').mean():.0f}/{100 * (s.best3 == 'powerlaw').mean():.0f}",
                     "Matern best %": f"{100 * (s.best == 'matern').mean():.0f}",
                     "nu p5/50/95": f"{q(s.m_nu, 5):.2f}/{q(s.m_nu, 50):.2f}/{q(s.m_nu, 95):.2f}",
                     "H p5/50/95": f"{q(s.pl_H, 5):.2f}/{q(s.pl_H, 50):.2f}/{q(s.pl_H, 95):.2f}",
                     "exp sigma cm p5/50/95": f"{q(s.e_sigma_cm, 5):.1f}/{q(s.e_sigma_cm, 50):.1f}/{q(s.e_sigma_cm, 95):.1f}",
                     "sigma_bl30 cm p50": f"{q(s.sigma_bl30_cm, 50):.1f}",
                     "exp l m p5/50/95": f"{q(s.e_l, 5):.1f}/{q(s.e_l, 50):.1f}/{q(s.e_l, 95):.1f}",
                     "l capped %": f"{100 * s.l_capped.mean():.0f}",
                     "exp white %": f"{100 * s.e_white.mean():.0f}",
                     "adequate %": f"{100 * s.adequate.mean():.0f}", "Bragg-only adequate %": f"{100 * s.adequate_bragg_only.mean():.0f}",
                     "misfit 5 m dB p5/50/95": f"{q(s['bragg_60MHz_vs_best'], 5):+.1f}/{q(s['bragg_60MHz_vs_best'], 50):+.1f}/{q(s['bragg_60MHz_vs_best'], 95):+.1f}",
                     "misfit 1.5 m dB p5/50/95": f"{q(s.mis15, 5):+.1f}/{q(s.mis15, 50):+.1f}/{q(s.mis15, 95):+.1f}",
                     "misfit 1.0 m dB p5/50/95": f"{q(s.mis10, 5):+.1f}/{q(s.mis10, 50):+.1f}/{q(s.mis10, 95):+.1f}",
                     "misfit 0.75 m dB p50": f"{q(s['bragg_400MHz_vs_best'], 50):+.1f}",
                     "misfit 1.5 m vs Matern p5/50/95": f"{q(s.mis15_m, 5):+.1f}/{q(s.mis15_m, 50):+.1f}/{q(s.mis15_m, 95):+.1f}",
                     "misfit 1.0 m vs Matern p50": f"{q(s.mis10_m, 50):+.1f}",
                     "S(1.5 m) best dB p5/50/95": f"{q(s.S_195MHz_best_dB, 5):.0f}/{q(s.S_195MHz_best_dB, 50):.0f}/{q(s.S_195MHz_best_dB, 95):.0f}",
                     "D misfit 1/2/5/10/20 m dB p50": "/".join(f"{q(s[f'D_{r}m_exp_minus_meas_dB'], 50):+.1f}" for r in (1, 2, 5, 10, 20))})
    return pd.DataFrame(rows)


def year_partition(df):
    """Two-way additive site + year model at sites with >= 3 years; variance components."""
    out = {}
    for hemi in ("gl", "aa"):
        d = df[(df.hemi == hemi)]
        d = d[d.site.isin(d.groupby("site").year.nunique().pipe(lambda s: s[s >= 3]).index)]
        for var in ("log_sigma", "log_l", "S_195MHz_best_dB", "S_60MHz_best_dB", "m_nu", "mis15", "rms_4-8m", "rms_16-32m"):
            v = d[["site", "year", var]].dropna()
            if var.startswith("rms"): v[var] = 10 * np.log10(v[var].clip(1e-4))
            if var in ("log_sigma", "log_l"): v[var] = 20 * v[var]     # dB of amplitude / of l
            if len(v) < 30: continue
            y = v[var].values; site = v.site.values; yr = v.year.values
            a = pd.Series(0.0, index=np.unique(site)); b = pd.Series(0.0, index=np.unique(yr)); mu = y.mean()
            for _ in range(30):
                a = pd.Series(y - mu - b[yr].values).groupby(site).mean()
                b = pd.Series(y - mu - a[site].values).groupby(yr).mean(); b -= b.mean()
            res = y - mu - a[site].values - b[yr].values
            out[f"{hemi}_{var}"] = dict(n=len(v), n_sites=len(a), total_sd=float(y.std()), site_sd=float(a.std()), year_sd=float(b.std()), resid_sd=float(res.std()),
                                       year_effects={int(k): round(float(x), 2) for k, x in b.items()})
    return out


def variogram(m, var, hemi, edges_km=(5, 10, 20, 40, 80, 160, 320, 640, 1280)):
    d = m[(m.hemi == hemi)].dropna(subset=[var])
    if len(d) < 20: return None
    x, y, v = d.x.values, d.y.values, d[var].values
    i, j = np.triu_indices(len(d), 1)
    dist = np.hypot(x[i] - x[j], y[i] - y[j]) / 1e3; g = 0.5 * (v[i] - v[j]) ** 2
    e = np.array(edges_km); b = np.digitize(dist, e)
    return dict(lag_km=[float(np.sqrt(e[k - 1] * e[k])) for k in range(1, len(e))],
                gamma=[float(g[b == k].mean()) if (b == k).sum() > 20 else np.nan for k in range(1, len(e))],
                n=[int((b == k).sum()) for k in range(1, len(e))], var=float(v.var()))


def maps(m, df):
    for hemi in ("gl", "aa"):
        d = m[m.hemi == hemi]
        if d.empty: continue
        mk = np.load(ATM2 / f"mask_{hemi}.npz"); gt = mk["transform"]; ice = mk["ice"]
        ext = [gt[0], gt[0] + gt[1] * ice.shape[1], gt[3] + gt[5] * ice.shape[0], gt[3]]
        panels = [("e_sigma_cm", "exp sigma (cm, band < 30 m)", dict(norm=matplotlib.colors.LogNorm(2, 50), cmap="viridis")),
                  ("e_l", "exp l (m)", dict(norm=matplotlib.colors.LogNorm(1, 300), cmap="magma")),
                  ("nu", "Matern nu (0.5 = exponential)", dict(vmin=0.1, vmax=1.2, cmap="coolwarm")),
                  ("adequate", "exponential adequacy (fraction of years)", dict(vmin=0, vmax=1, cmap="RdYlGn")),
                  ("mis15", "exp misfit at 1.5 m (dB, exp - best)", dict(vmin=-6, vmax=6, cmap="RdBu_r")),
                  ("mis10", "exp misfit at 1.0 m (dB)", dict(vmin=-6, vmax=6, cmap="RdBu_r")),
                  ("S195", "S(1.5 m) best family (dB re m^4)", dict(vmin=-70, vmax=-45, cmap="plasma")),
                  ("H", "power-law H", dict(vmin=0, vmax=0.8, cmap="cividis"))]
        fig, axs = plt.subplots(2, 4, figsize=(22, 11 if hemi == "gl" else 10))
        for ax, (col, ttl, kw) in zip(axs.ravel(), panels):
            ax.imshow(ice, extent=ext, cmap="Greys", alpha=0.25, origin="upper")
            sc = ax.scatter(d.x, d.y, c=d[col], s=14, **kw); plt.colorbar(sc, ax=ax, shrink=0.7); ax.set_title(ttl); ax.set_aspect("equal")
            if hemi == "gl": ax.set_xlim(-700e3, 900e3); ax.set_ylim(-3400e3, -600e3)
            else:
                ax.set_xlim(d.x.min() - 200e3, d.x.max() + 200e3); ax.set_ylim(d.y.min() - 200e3, d.y.max() + 200e3)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Tier 2 ATM1B site medians, {hemi.upper()} ({len(d)} sites)"); fig.tight_layout()
        fig.savefig(OUT / f"map_{hemi}.png", dpi=110); plt.close(fig)


def fig_relations(m):
    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    for row, hemi in enumerate(("gl", "aa")):
        d = m[m.hemi == hemi]
        for ax, (xv, xl) in zip(axs[row], (("h", "elevation m"), ("dist_km", "distance to margin km"), ("slope", "100 m slope"), ("e_sigma_cm", "exp sigma cm"))):
            xx = d[xv].clip(1e-4) if xv in ("dist_km", "slope", "e_sigma_cm") else d[xv]
            sc = ax.scatter(xx, d.mis15, c=d.nu, s=12, cmap="coolwarm", vmin=0.1, vmax=1.2)
            if xv in ("dist_km", "slope", "e_sigma_cm"): ax.set_xscale("log")
            ax.axhline(0, color="k", lw=0.5); ax.axhspan(-3, 3, color="g", alpha=0.08); ax.set_xlabel(xl); ax.set_ylabel("exp misfit at 1.5 m (dB)"); ax.set_ylim(-15, 15)
            ax.set_title(f"{hemi.upper()} (colour = Matern nu)")
    plt.colorbar(sc, ax=axs, shrink=0.5); fig.savefig(OUT / "fig_misfit_vs_covariates.png", dpi=110); plt.close(fig)


def grouping(m):
    from sklearn.mixture import GaussianMixture
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score
    out = {}
    feats = ["log_sigma", "log_l", "nu", "mis15"]
    d = m.assign(log_sigma=np.log10(m.e_sigma_cm), log_l=np.log10(m.e_l)).dropna(subset=feats + ["h", "dist_km", "slope"])
    d = d[np.isfinite(d[feats]).all(1)]
    X = d[feats].values; X = (X - X.mean(0)) / X.std(0)
    bics = {k: float(GaussianMixture(k, n_init=5, random_state=0).fit(X).bic(X)) for k in range(1, 7)}
    kbest = min(bics, key=bics.get)
    drop = {k: bics[1] - v for k, v in bics.items()}; k10 = next((k for k in range(2, 7) if (drop[k] - drop[k - 1]) < 0.1 * max(drop[6], 1)), 6) - 1
    k10 = max(k10, 1)
    gm = GaussianMixture(k10, n_init=5, random_state=0).fit(X); d["cluster"] = gm.predict(X)
    prof = d.groupby("cluster").agg(n=("site", "size"), gl=("hemi", lambda x: (x == "gl").mean()), sigma_cm=("e_sigma_cm", "median"), l=("e_l", "median"), nu=("nu", "median"),
                                    mis15=("mis15", "median"), adequate=("adequate", "mean"), h=("h", "median"), dist_km=("dist_km", "median"), slope=("slope", "median"),
                                    facies=("facies", lambda x: x.value_counts().index[0]), strata=("stratum_name", lambda x: "; ".join(f"{k}:{v}" for k, v in x.value_counts().head(3).items())))
    out["gmm_bic"] = bics; out["gmm_k_bicmin"] = kbest; out["gmm_k_used"] = int(k10); out["gmm_profile"] = prof.round(3).to_dict("index")
    # covariate regression: mis15, nu, log sigma on covariates (GBT + permutation importance, 5-fold R2)
    cov = ["h", "dist_km", "slope", "lat"] + [c for c in ("melt_days", "smb_mmwe", "wind10_ms") if c in d and d[c].notna().mean() > 0.5]
    d["lat"] = d.lat.abs()
    reg = {}
    for tgt in ("mis15", "nu", "log_sigma", "log_l", "adequate"):
        for hemi in ("gl", "aa", "all"):
            dd = d if hemi == "all" else d[d.hemi == hemi]
            dd = dd.dropna(subset=cov + [tgt])
            if len(dd) < 40: continue
            Xc, yv = dd[cov].values, dd[tgt].values
            mdl = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0)
            r2 = float(cross_val_score(mdl, Xc, yv, cv=5, scoring="r2").mean())
            mdl.fit(Xc, yv); pi = permutation_importance(mdl, Xc, yv, n_repeats=10, random_state=0)
            reg[f"{tgt}_{hemi}"] = dict(n=len(dd), cv_r2=r2, importance={c: float(v) for c, v in zip(cov, pi.importances_mean)},
                                        spearman={c: float(dd[[c, tgt]].corr("spearman").iloc[0, 1]) for c in cov})
    out["gbt"] = reg
    # facies-only: does facies reproduce clusters / adequacy?
    out["adequacy_by_facies"] = d.groupby(["hemi", "facies"]).agg(n=("site", "size"), adequate=("adequate", "mean"), mis15=("mis15", "median"), nu=("nu", "median"),
                                                                   sigma_cm=("e_sigma_cm", "median"), l=("e_l", "median")).round(3).reset_index().to_dict("records")
    ct = pd.crosstab(d.facies, d.cluster, normalize="index").round(2)
    out["cluster_by_facies"] = ct.to_dict("index")
    # gradient test: adequacy / nu vs elevation bins within GL
    g = d[d.hemi == "gl"].assign(hb=pd.cut(d.h, [0, 500, 1000, 1500, 2000, 2500, 3500]))
    out["gl_elevation_gradient"] = g.groupby("hb", observed=True).agg(n=("site", "size"), adequate=("adequate", "mean"), nu=("nu", "median"), mis15=("mis15", "median"),
                                                                       sigma_cm=("e_sigma_cm", "median"), l=("e_l", "median"), lcap=("l_capped", "mean")).round(3).reset_index().astype({"hb": str}).to_dict("records")
    g = d[d.hemi == "aa"].assign(hb=pd.cut(d.h, [-10, 100, 500, 1000, 1500, 2500, 4000]))
    out["aa_elevation_gradient"] = g.groupby("hb", observed=True).agg(n=("site", "size"), adequate=("adequate", "mean"), nu=("nu", "median"), mis15=("mis15", "median"),
                                                                       sigma_cm=("e_sigma_cm", "median"), l=("e_l", "median"), lcap=("l_capped", "mean")).round(3).reset_index().astype({"hb": str}).to_dict("records")
    # cluster map
    for hemi in ("gl", "aa"):
        dd = d[d.hemi == hemi]
        if dd.empty: continue
        fig, ax = plt.subplots(figsize=(7, 8))
        sc = ax.scatter(dd.x, dd.y, c=dd.cluster, cmap="tab10", s=14, vmin=0, vmax=9); ax.set_aspect("equal"); ax.set_title(f"GMM clusters (log sigma, log l, nu, misfit 1.5 m), {hemi.upper()}, k={k10}")
        plt.colorbar(sc, ax=ax, shrink=0.6); fig.savefig(OUT / f"clusters_{hemi}.png", dpi=110); plt.close(fig)
    d[["site", "cluster"]].to_csv(OUT / "site_clusters.csv", index=False)
    return out


def recommendation(df, m):
    """Per-stratum exponential (sigma, l) with uncertainty, adequacy and needed nu."""
    rows = []
    for key, s in df.groupby(["hemi", "stratum_name"]):
        ms = m[(m.hemi == key[0]) & (m.stratum_name == key[1])]
        rows.append(dict(hemi=key[0], stratum=key[1], sites=len(ms), site_years=len(s),
                         sigma_cm_med=round(q(s.e_sigma_cm, 50), 1), sigma_cm_p5=round(q(s.e_sigma_cm, 5), 1), sigma_cm_p95=round(q(s.e_sigma_cm, 95), 1),
                         sigma_bl30_cm_med=round(q(s.sigma_bl30_cm, 50), 1),
                         l_m_med=round(q(s.e_l, 50), 1), l_m_p5=round(q(s.e_l, 5), 1), l_m_p95=round(q(s.e_l, 95), 1), l_capped_frac=round(s.l_capped.mean(), 2),
                         S195_dB_med=round(q(s.S_195MHz_best_dB, 50), 1), S195_exp_dB_med=round(q(s.S_195MHz_exponential_dB, 50), 1),
                         S60_dB_med=round(q(s.S_60MHz_best_dB, 50), 1),
                         adequate_frac=round(s.adequate.mean(), 2), bragg_adequate_frac=round(s.adequate_bragg_only.mean(), 2),
                         mis15_med=round(q(s.mis15, 50), 1), mis15_p5=round(q(s.mis15, 5), 1), mis15_p95=round(q(s.mis15, 95), 1),
                         mis10_med=round(q(s.mis10, 50), 1), nu_med=round(q(s.m_nu, 50), 2), nu_p5=round(q(s.m_nu, 5), 2), nu_p95=round(q(s.m_nu, 95), 2),
                         H_med=round(q(s.pl_H, 50), 2),
                         verdict=("use exponential" if s.adequate_bragg_only.mean() >= 0.7 else "exponential marginal (Matern nu needed)" if s.adequate_bragg_only.mean() >= 0.4 else "do not use exponential")))
    return pd.DataFrame(rows)


def main():
    df = load(); m = site_medians(df); m.to_csv(OUT / "site_medians.csv", index=False)
    tabs = {}
    tabs["by_stratum"] = table_by(df, ["hemi", "stratum_name"], "hemi/stratum")
    tabs["by_facies"] = table_by(df, ["hemi", "facies_proxy"], "hemi/facies")
    tabs["by_year"] = table_by(df, ["hemi", "year"], "hemi/year")
    tabs["by_class"] = table_by(df, ["hemi", "primary_class"], "hemi/pair class")
    tabs["by_kind"] = table_by(df, ["hemi", "kind"], "hemi/kind")
    A = dict(n_rows=len(df), n_sites=int(df.site.nunique()), year_partition=year_partition(df))
    A["variograms"] = {f"{h}_{v}": variogram(m.assign(log_sigma=np.log10(m.e_sigma_cm), log_l=np.log10(m.e_l), S195=m.S195), v, h)
                       for h in ("gl", "aa") for v in ("log_sigma", "log_l", "S195", "mis15", "nu")}
    try:
        A["grouping"] = grouping(m)
    except Exception as e:  # noqa: BLE001
        A["grouping_error"] = str(e)
    rec = recommendation(df, m); rec.to_csv(OUT / "recommendation_by_stratum.csv", index=False)
    maps(m, df); fig_relations(m)
    with open(OUT / "tables.md", "w") as f:
        for k, t in tabs.items():
            f.write(f"## {k}\n\n{t.to_markdown(index=False)}\n\n")
        f.write("## recommendation\n\n" + rec.to_markdown(index=False) + "\n")
    (OUT / "analysis.json").write_text(json.dumps(A, indent=1, default=float))
    print(tabs["by_stratum"].to_string()); print(rec.to_string())


if __name__ == "__main__":
    main()
