import pandas as pd, numpy as np, glob, os
from pyproj import Transformer
C = 299792458.0
tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

rows = []
tracks = {}
for f in sorted(glob.glob("lyr/*.parquet")):
    key = os.path.basename(f)[:-8]
    col, sp = key.split("__")
    d = pd.read_parquet(f)
    srf = d[d.lyr_id == 1].sort_values("gps_time")
    bot = d[d.lyr_id == 2].sort_values("gps_time")
    if len(srf) == 0: continue
    x, y = tr.transform(srf.lon.values, srf.lat.values)
    agl = C * srf.twtt.values / 2.0
    # bottom pick presence on the surface gps grid
    bt = np.full(len(srf), np.nan)
    if len(bot):
        gi = np.searchsorted(bot.gps_time.values, srf.gps_time.values).clip(1, len(bot)-1)
        near = np.abs(bot.gps_time.values[gi] - srf.gps_time.values) < 0.05
        bt[near] = bot.twtt.values[gi][near]
    tracks[key] = dict(col=col, sp=sp, x=x, y=y, t=srf.gps_time.values,
                       lat=srf.lat.values, lon=srf.lon.values,
                       elev=srf.elev.values, agl=agl, s_twtt=srf.twtt.values, b_twtt=bt)
    rows.append(dict(key=key, collection=col, segment=sp, n=len(srf),
                     agl_med=np.nanmedian(agl), agl_p10=np.nanpercentile(agl,10),
                     agl_p90=np.nanpercentile(agl,90),
                     bot_frac=np.isfinite(bt).mean()))
np.save("tracks.npy", tracks, allow_pickle=True)
s = pd.DataFrame(rows).sort_values("agl_med")
s.to_csv("seg_summary.csv", index=False)
pd.set_option("display.width", 200)
print(s.to_string(index=False))
