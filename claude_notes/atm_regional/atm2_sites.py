"""Draft Tier 2 site list: stratified random sample of 10 km cells over regime candidates
(>= N_PER_REGIME each, repeat-year cells preferred), all cells with >= REPEAT_MIN years,
the 5 study lines, and firn-core / radar ground-truth sites (nearest cell within 15 km).
Run after atm2_analyze.py. Writes outputs/atm_regional/tier2_sites_draft.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from pyproj import Transformer
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "atm_regional"
RNG = np.random.default_rng(1)
N_TOTAL, N_PER_STRATUM, REPEAT_MIN, REPEAT_CAP = 400, 20, {"gl": 9, "aa": 5}, 100
ELEV_BANDS = [0, 500, 1500, 2500, 5000]
CRS = {"gl": "EPSG:3413", "aa": "EPSG:3031"}

FIXED = [  # name, hemi, lat, lon, kind
    ("study_westcoast_start", "gl", 71.399, -50.255, "study_line"), ("study_westcoast_mid", "gl", 70.966, -49.894, "study_line"),
    ("study_westcoast_end", "gl", 70.532, -49.561, "study_line"),
    ("study_geikie_start", "gl", 70.888, -44.78, "study_line"), ("study_geikie_mid", "gl", 70.495, -46.28, "study_line"),
    ("study_geikie_end", "gl", 69.875, -47.043, "study_line"),
    ("study_getz_start", "aa", -74.587, -117.56, "study_line"), ("study_getz_mid", "aa", -74.691, -120.064, "study_line"),
    ("study_getz_end", "aa", -74.765, -122.598, "study_line"),
    ("study_david_start", "aa", -75.134, 157.569, "study_line"), ("study_david_mid", "aa", -75.285, 160.127, "study_line"),
    ("study_david_end", "aa", -75.384, 162.749, "study_line"),
    ("core_B26", "gl", 77.2533, -49.2167, "firn_core"), ("core_CampCentury", "gl", 77.1714, -61.0778, "firn_core"),
    ("core_DYE2", "gl", 66.48, -46.28, "firn_core"), ("core_Summit", "gl", 72.58, -38.46, "firn_core"),
    ("core_B16", "gl", 73.9402, -37.6299, "firn_core"), ("core_B17", "gl", 75.2504, -37.6248, "firn_core"),
    ("core_B18", "gl", 76.617, -36.4033, "firn_core"), ("core_B21", "gl", 80.0, -41.1374, "firn_core"), ("core_B29", "gl", 76.0039, -43.492, "firn_core"),
    ("core_B19", "gl", 78.0, -36.23, "firn_core"), ("core_SEDome", "gl", 67.19, -36.47, "firn_core"), ("core_FA13A", "gl", 66.1812, -39.0435, "firn_core"),
] + [(f"egig_T{i:02d}_approx", "gl", 69.65 + 1.7 * f, -49.5 + 16.0 * f, "egig_line_approx") for i, f in enumerate(np.linspace(0, 1, 6))]
# Antarctic SUMup cores with coordinates (outputs/sumup/tier1_cores_60mhz.csv)
_su = pd.read_csv(ROOT / "outputs" / "sumup" / "tier1_cores_60mhz.csv")
for _, r in _su[_su.region == "antarctica"].iterrows():
    nm = str(r["name"]) if isinstance(r["name"], str) else str(r["label"]).replace(" ", "")
    FIXED.append((f"core_{nm.replace(' ', '').replace('(', '').replace(')', '')[:24]}", "aa", r.lat, r.lon, "firn_core"))


def main():
    rows = []
    for hemi in ("gl", "aa"):
        p = OUT / f"grid_{hemi}_clusters.csv"
        if not p.exists(): continue
        g = pd.read_csv(p); g["hemi"] = hemi
        g["repeat"] = g.n_years >= 2
        # strata = GMM regime x elevation band (so the interior is covered even where the GMM lumps it);
        # quota per stratum proportional to sqrt(n) with a floor of N_PER_STRATUM
        g["stratum"] = g.regime.astype(str) + "_h" + pd.cut(g.h, ELEV_BANDS, labels=False, include_lowest=True).fillna(0).astype(int).astype(str)
        cnt = g.stratum.value_counts(); cnt = cnt[cnt >= 5]; w = np.sqrt(cnt); quota = np.maximum(N_PER_STRATUM, np.round(w / w.sum() * (N_TOTAL // 2))).astype(int)
        for st, q in quota.items():
            s = g[g.stratum == st]; reg = int(s.regime.iloc[0]); prob = np.where(s.repeat, 3.0, 1.0) * np.where(s.above, 2.0, 1.0); prob /= prob.sum()
            pick = s.iloc[RNG.choice(len(s), min(q, len(s)), replace=False, p=prob)]
            for _, r in pick.iterrows():
                rows.append(dict(site=f"{hemi}_r{reg}_{int(r.cell)}", hemi=hemi, lat=r.lat, lon=r.lon, x=r.xm, y=r.ym, regime=int(reg),
                                 stratum=st, kind="stratified", n_years=int(r.n_years), years=r.years, above_floor=bool(r.above),
                                 r_med_cm=r.r_med, h=r.h, slope=r.slope, dist_km=r.dist_km,
                                 priority=1 if (r.above and r.repeat) else 2))
        # repeat-year cells: the most-repeated cells, capped, spread over strata (round-robin by n_years)
        rep = g[g.n_years >= REPEAT_MIN[hemi]].sort_values("n_years", ascending=False)
        rep = rep.assign(rank=rep.groupby("stratum").cumcount()).sort_values(["rank", "n_years"], ascending=[True, False]).head(REPEAT_CAP)
        for _, r in rep.iterrows():
            rows.append(dict(site=f"{hemi}_rep_{int(r.cell)}", hemi=hemi, lat=r.lat, lon=r.lon, x=r.xm, y=r.ym, regime=int(r.regime),
                             stratum=r.stratum, kind="repeat_years", n_years=int(r.n_years), years=r.years, above_floor=bool(r.above),
                             r_med_cm=r.r_med, h=r.h, slope=r.slope, dist_km=r.dist_km, priority=1))
        tr = Transformer.from_crs("EPSG:4326", CRS[hemi], always_xy=True)
        for name, hh, lat, lon, kind in FIXED:
            if hh != hemi: continue
            x, y = tr.transform(lon, lat); d = np.hypot(g.xm - x, g.ym - y) / 1e3; i = int(np.argmin(d))
            r = g.iloc[i]; hit = d[i] <= 15
            rows.append(dict(site=name, hemi=hemi, lat=lat, lon=lon, x=x, y=y, regime=int(r.regime) if hit else -1, stratum=r.stratum if hit else "none", kind=kind,
                             n_years=int(r.n_years) if hit else 0, years=r.years if hit else "none within 15 km",
                             above_floor=bool(r.above) if hit else None, r_med_cm=r.r_med if hit else np.nan, h=r.h if hit else np.nan,
                             slope=r.slope if hit else np.nan, dist_km=r.dist_km if hit else np.nan, priority=0,
                             nearest_cell_km=round(float(d[i]), 1)))
    df = pd.DataFrame(rows).drop_duplicates("site")
    df.to_csv(OUT / "tier2_sites_draft.csv", index=False)
    print(df.groupby(["hemi", "kind"]).size()); print("total", len(df)); print(df[df.kind=="stratified"].groupby(["hemi", "stratum"]).size()); print("repeat n_years:", df[df.kind=="repeat_years"].groupby("hemi").n_years.describe()[["count","min","50%","max"]])
    print(df[df.kind.isin(["study_line", "firn_core", "egig_line_approx"])][["site", "years", "nearest_cell_km", "r_med_cm", "above_floor"]].to_string())


if __name__ == "__main__":
    main()
