"""Tier 2 batch driver: per site (in phase order) choose visits, pull ILATM1B, run the
site roughness analysis, checkpoint one parquet row per site-visit.

  uv run claude_notes/atm_regional/tier2/batch.py --shard 0 --nshard 3 [--budget-gb 60] [--sites a b]
Rows: outputs/atm_regional/tier2/rows/<site>__<date>.parquet (scalars flat + json blobs).
Status: outputs/atm_regional/tier2/status_<shard>.json; pull log pull_log.csv (append)."""
from __future__ import annotations
import argparse, json, time, traceback
import numpy as np, pandas as pd
from common import OUT, CACHE, sites
import pull, roughness as rg

ROWS = OUT / "rows"; ROWS.mkdir(exist_ok=True)
YEAR_PREF = [2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009]
MIN_PLATELETS = 40


def choose_visits(site_row, V):
    v = V[(V.site == site_row.site) & (V.n >= MIN_PLATELETS)].copy()
    if v.empty: return v
    v = v.sort_values("n", ascending=False).drop_duplicates("date")     # biggest pass per date
    v["pref"] = v.year.map({y: i for i, y in enumerate(YEAR_PREF)})
    kind, phase = site_row.kind, int(site_row.phase)
    if kind in ("study_line", "firn_core", "egig_line_approx"):
        return v.sort_values("pref")
    v = v.sort_values(["pref", "n"], ascending=[True, False]).drop_duplicates("year")   # one date per year
    if kind == "repeat_years":
        v2 = v[v.year >= 2013]; v1 = v[v.year < 2013].head(2)
        return pd.concat([v2, v1])
    return v.head(1 if phase == 2 else 2)   # phase 2: one visit (budget); year-to-year comes from phase 1


def flatten(site_row, visit, res, mb, files):
    r = dict(site=site_row.site, hemi=site_row.hemi, lat=site_row.lat, lon=site_row.lon, x=site_row.x, y=site_row.y,
             regime=site_row.regime, stratum=site_row.stratum, kind=site_row.kind, phase=int(site_row.phase),
             h_cov=site_row.h, slope_cov=site_row.slope, dist_km=site_row.dist_km, r_med_cm_atm2=site_row.r_med_cm,
             date=str(visit.date)[:10], year=int(visit.year), t0=float(visit.t0), t1=float(visit.t1), n_platelets=int(visit.n),
             mb=mb, n_files=len(files), files=";".join(f.name for f in files), status=res.get("status"))
    if res.get("status") != "ok":
        r["n_used"] = res.get("n_used"); return r
    for k in ("n_used", "n_blocks", "primary_class", "swath_m", "density_per_m2", "elev_m", "slope_100m", "rms_resid_m", "mad_m",
              "dropped_qc", "dropped_blunder", "heading_deg", "sigma_bl30_m", "sigma_bl30_expmodel_m", "adequate", "adequate_bragg_only"):
        r[k] = res.get(k)
    for k, v in res["noise"].items(): r[f"noise_{k}"] = v
    fa = res["fits"]
    r["best"] = fa.get("best"); r["best3"] = fa.get("best3")
    for fam, tag in (("gaussian", "g"), ("exponential", "e"), ("powerlaw", "pl"), ("matern", "m")):
        f = fa.get(fam, {})
        for k in ("sigma", "l", "nu", "nu_err", "l_rel_err", "sigma_rel_err", "H", "H_err", "beta", "A", "c", "n0", "bic", "runs_p", "rms_log_resid", "sigma_fixn0", "l_fixn0"):
            if k in f: r[f"{tag}_{k}"] = f[k]
        r[f"{tag}_dbic"] = fa.get("dbic", {}).get(fam)
    for sec, key in (("along", "fits_along"), ("cross", "fits_cross")):
        f = res[key]
        r[f"{sec}_best"] = f.get("best")
        for fam, tag in (("exponential", "e"), ("powerlaw", "pl")):
            for k in ("sigma", "l", "H"):
                if k in f.get(fam, {}): r[f"{sec}_{tag}_{k}"] = f[fam][k]
    for k, v in res["octave_rms"].items(): r[f"rms_{k}"] = v
    for k, v in res["octave_rms_along"].items(): r[f"rms_along_{k}"] = v
    for k, v in res["octave_rms_cross"].items(): r[f"rms_cross_{k}"] = v
    for k, v in res["octave_rms"].items():
        a, c = res["octave_rms_along"].get(k), res["octave_rms_cross"].get(k)
        r[f"aniso_{k}"] = float(a / c) if (a is not None and c is not None and np.isfinite(a) and np.isfinite(c) and c > 0) else np.nan
    r.update(res["bragg"]); r.update(res["misfit"])
    r["D_json"] = json.dumps(dict(lag=res["D_lag"], D=res["D_all"], N=res["N_all"], D_same=res["D_same"], N_same=res["N_same"],
                                  D_xscan=res["D_xscan"], N_xscan=res["N_xscan"]), default=float)
    r["blocks_json"] = json.dumps(res["blocks"], default=float)
    return r


def run_site(site_row, V, budget, log):
    chosen = choose_visits(site_row, V)
    if chosen.empty:
        log.write(f"{site_row.site},,,no_visits,0,0\n"); return 0.0
    mb_site = 0.0
    for _, vis in chosen.iterrows():
        tag = f"{site_row.site}__{str(vis.date)[:10]}"
        rp = ROWS / f"{tag}.parquet"
        if rp.exists(): continue
        t0 = time.time()
        try:
            gr = pull.search_visit(site_row.lat, site_row.lon, vis.date, vis.t0, vis.t1)
            gr1b = [g for g in gr if g[2] == "ILATM1B"]
            need_mb = sum(g[1] for g in gr1b)
            if not gr1b:
                log.write(f"{site_row.site},{vis.date},,no_granules,0,0\n"); log.flush()
                pd.DataFrame([dict(site=site_row.site, date=str(vis.date)[:10], year=int(vis.year), status="no_granules")]).to_parquet(rp); continue
            if budget["used_mb"] + need_mb > budget["hard_mb"]:
                log.write(f"{site_row.site},{vis.date},,over_budget,{need_mb:.0f},0\n"); log.flush(); continue
            files, mb_new, ok = pull.download(gr1b, site_row.site, vis.date)
            budget["used_mb"] += mb_new; mb_site += mb_new
            if not files:
                log.write(f"{site_row.site},{vis.date},,download_failed,{need_mb:.0f},0\n"); log.flush()
                pd.DataFrame([dict(site=site_row.site, date=str(vis.date)[:10], year=int(vis.year), status="download_failed")]).to_parquet(rp); continue
            P = rg.load_site(files, site_row.lat, site_row.lon, site_row.hemi)
            res = rg.analyse_site(P) if P is not None else dict(status="no_points_in_site")
            row = flatten(site_row, vis, res, mb_new, files)
            row["elapsed_s"] = time.time() - t0
            pd.DataFrame([row]).to_parquet(rp)
            f = res.get("fits", {})
            e = f.get("exponential", {}); pl = f.get("powerlaw", {}); mt = f.get("matern", {})
            print(f"  {tag} {res.get('status')} n={res.get('n_used')} cls={res.get('primary_class')} best={f.get('best')} "
                  f"E(s={e.get('sigma', np.nan):.3f} l={e.get('l', np.nan):.1f} p={e.get('runs_p', np.nan):.2f}) PL(H={pl.get('H', np.nan):.2f}) M(nu={mt.get('nu', np.nan):.2f}) "
                  f"mis1.5={res.get('misfit', {}).get('bragg_195MHz_vs_best', np.nan):+.1f}dB adeq={res.get('adequate')} "
                  f"{mb_new:.0f}MB {time.time() - t0:.0f}s", flush=True)
            log.write(f"{site_row.site},{vis.date},{';'.join(g[0] for g in gr1b)},{res.get('status')},{need_mb:.0f},{mb_new:.0f}\n"); log.flush()
        except Exception as e:  # noqa: BLE001
            print(f"  !! {tag}: {e}\n{traceback.format_exc()[-600:]}", flush=True)
            log.write(f"{site_row.site},{vis.date},,error:{str(e)[:60].replace(',', ' ')},0,0\n"); log.flush()
    return mb_site


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--budget-gb", type=float, default=60.0); ap.add_argument("--sites", nargs="*"); ap.add_argument("--phases", nargs="*", type=int)
    a = ap.parse_args()
    S = sites(); V = pd.read_parquet(OUT / "visits.parquet")
    if a.sites: S = S[S.site.isin(a.sites)]
    if a.phases: S = S[S.phase.isin(a.phases)]
    S = S.iloc[a.shard::a.nshard]
    budget = dict(used_mb=used_mb_total(), hard_mb=a.budget_gb * 1e3 * 1.15)
    log = open(OUT / f"pull_log_{a.shard}.csv", "a")
    st = OUT / f"status_{a.shard}.json"
    for i, (_, row) in enumerate(S.iterrows()):
        print(f"== [{i + 1}/{len(S)}] {row.site} phase {row.phase} {row.kind} {row.stratum}  used {budget['used_mb'] / 1e3:.1f} GB", flush=True)
        run_site(row, V, budget, log)
        budget["used_mb"] = used_mb_total() if i % 5 == 0 else budget["used_mb"]
        st.write_text(json.dumps(dict(i=i + 1, n=len(S), site=row.site, phase=int(row.phase), used_gb=budget["used_mb"] / 1e3, time=time.ctime())))
        if (i + 1) % 50 == 0:
            try:
                import collect; collect.main()
            except Exception as e:  # noqa: BLE001
                print("collect failed", e)
    print("SHARD DONE", flush=True)


def used_mb_total():
    return sum(p.stat().st_size for p in CACHE.rglob("ILATM1B_*")) / 1e6


if __name__ == "__main__":
    main()
