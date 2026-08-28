"""Final Tier 2 site list = draft (709) + Greenland interior transect thinning (25 km along
Summit-Camp Century, EGIG, and a Summit-NE line), ordered in processing phases:
phase 1 = repeat + study/ground truth; phase 2 = stratified; phase 3 = GL interior fallback.
Writes outputs/atm_regional/tier2/sites.csv"""
import numpy as np, pandas as pd
from pyproj import Transformer
from common import ROOT, OUT, CRS

TRANSECTS = {  # lat, lon waypoints
    "summit_campcentury": [(72.58, -38.46), (77.17, -61.08)],
    "egig": [(69.65, -49.5), (71.35, -33.5)],
    "summit_ne": [(72.58, -38.46), (75.5, -33.0), (79.0, -25.0)],
}


def main():
    d = pd.read_csv(ROOT / "outputs" / "atm_regional" / "tier2_sites_draft.csv")
    d = d[d.years != "none within 15 km"].copy()   # no ATM: skip (logged in note)
    d["phase"] = np.where(d.kind != "stratified", 1, 2)
    d.loc[(d.kind == "stratified") & (d.hemi == "gl") & (d.regime == 0), "phase"] = 3
    g = pd.read_csv(ROOT / "outputs" / "atm_regional" / "grid_gl.csv")
    tr = Transformer.from_crs("EPSG:4326", CRS["gl"], always_xy=True)
    used = set(d.x.round(0).astype(int).astype(str) + "_" + d.y.round(0).astype(int).astype(str))
    rows = []
    for name, wp in TRANSECTS.items():
        xy = np.array([tr.transform(lon, lat) for lat, lon in wp])
        seg = np.diff(xy, axis=0); L = np.hypot(*seg.T); s_end = np.r_[0, np.cumsum(L)]
        for s in np.arange(0, s_end[-1], 25e3):
            i = np.searchsorted(s_end, s, side="right") - 1; i = min(i, len(seg) - 1)
            p = xy[i] + seg[i] * (s - s_end[i]) / L[i]
            dist = np.hypot(g.xm - p[0], g.ym - p[1]) / 1e3
            cand = g[dist <= 8]
            if cand.empty: continue
            r = cand.sort_values("n_years", ascending=False).iloc[0]
            key = f"{int(round(r.xm))}_{int(round(r.ym))}"
            if key in used: continue
            used.add(key)
            rows.append(dict(site=f"gl_tr_{name[:6]}_{int(s / 1e3):04d}", hemi="gl", lat=r.lat, lon=r.lon, x=r.xm, y=r.ym,
                             regime=int(r.regime) if "regime" in r else -1, stratum=f"tr_{name}", kind="transect",
                             n_years=int(r.n_years), years=r.years, above_floor=bool(r.at_floor < 0.5), r_med_cm=r.r_med,
                             h=r.h, slope=r.slope, dist_km=r.dist_km, priority=2, phase=3))
    t = pd.DataFrame(rows)
    df = pd.concat([d, t], ignore_index=True).sort_values(["phase", "priority"], kind="stable").reset_index(drop=True)
    df.to_csv(OUT / "sites.csv", index=False)
    print(df.groupby(["phase", "hemi", "kind"]).size()); print("transect sites", len(t), t.groupby("stratum").size().to_dict())


if __name__ == "__main__":
    main()
