"""ATM flight visits per site from the ILATM2 platelet archive: for every site, every
(date, pass) with centre platelets within SITE_RADIUS_M -> UTC span, n platelets.
Writes outputs/atm_regional/tier2/visits.parquet"""
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from common import ATM2, OUT, CRS, SITE_RADIUS_M, sites


def main():
    S = sites()
    out = []
    for hemi in ("gl", "aa"):
        s = S[S.hemi == hemi]
        tree = cKDTree(np.c_[s.x, s.y])
        tr = Transformer.from_crs("EPSG:4326", CRS[hemi], always_xy=True)
        for p in sorted(ATM2.glob(f"platelets_*_{hemi}.parquet")):
            df = pd.read_parquet(p, columns=["lat", "lon", "utc_s", "date", "track", "h", "granule"])
            df = df[df.track == 0]
            x, y = tr.transform(df.lon.values, df.lat.values)
            hits = tree.query_ball_point(np.c_[x, y], SITE_RADIUS_M)
            idx = np.repeat(np.arange(len(df)), [len(h) for h in hits]); sid = np.concatenate([np.array(h, int) for h in hits]) if len(idx) else np.array([], int)
            if len(idx) == 0: continue
            m = pd.DataFrame(dict(site=s.site.values[sid], date=df.date.values[idx], utc=df.utc_s.values[idx], h=df.h.values[idx], granule=df.granule.values[idx]))
            m = m.sort_values(["site", "date", "utc"])
            # split passes: gap > 600 s within a date
            gap = (m.groupby(["site", "date"]).utc.diff().fillna(0) > 600).astype(int)
            m["pass_"] = gap.groupby([m.site, m.date]).cumsum()
            v = m.groupby(["site", "date", "pass_"]).agg(t0=("utc", "min"), t1=("utc", "max"), n=("utc", "size"), h=("h", "median"),
                                                          granules=("granule", lambda g: ";".join(sorted(set(g))))).reset_index()
            out.append(v); print(p.name, len(v), flush=True)
    V = pd.concat(out, ignore_index=True)
    V["year"] = pd.to_datetime(V.date).dt.year
    V.to_parquet(OUT / "visits.parquet", index=False)
    print(len(V), "visits;", V.groupby("site").size().describe())


if __name__ == "__main__":
    main()
