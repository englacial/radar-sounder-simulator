"""QC, ice-sheet masking, per-flight noise floor and noise-corrected roughness for
all ILATM2 platelets -> outputs/cache/atm2/platelets_qc.parquet and
outputs/atm_regional/flights.csv (per-day floors).

Thresholds: n_used >= N_MIN (100), n_removed <= 10 % of n_used, 0 < rms < 5 m,
|slope| < 0.5, on-ice (Natural Earth mask). Centre platelet: |dist_m| < 30 m
(track 0 nadir block); the middle cross-track segment (same footprint) is dropped. Floor per flight-day = p3 of centre
platelet rms over on-ice platelets (>= 200 needed, else campaign median floor).
r = sqrt(max(rms^2 - floor^2, 0)); at_floor = rms < floor * 10^(3/20).
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from pyproj import Transformer
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "cache" / "atm2"
OUT = ROOT / "outputs" / "atm_regional"; OUT.mkdir(exist_ok=True, parents=True)
N_MIN, EDIT_FRAC, FLOOR_PCT, FLOOR_DB = 100, 0.10, 3.0, 3.0
CRS = {"gl": "EPSG:3413", "aa": "EPSG:3031"}


def mask_lookup(hemi, x, y):
    m = np.load(CACHE / f"mask_{hemi}.npz")
    gt = m["transform"]; x0, dx, _, y0, _, dy = gt
    col = np.floor((x - x0) / dx).astype(int); row = np.floor((y - y0) / dy).astype(int)
    H, W = m["ice"].shape
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    r, c = np.clip(row, 0, H - 1), np.clip(col, 0, W - 1)
    ice = m["ice"][r, c] & ok; shelf = m["shelf"][r, c] & ok; dist = m["dist_km"][r, c]
    return ice, shelf, dist


def main():
    parts, flights = [], []
    for p in sorted(CACHE.glob("platelets_*_??.parquet")):
        yr, hemi = p.stem.split("_")[1:3]
        df = pd.read_parquet(p)
        n0 = len(df)
        df = df[(df.n_used >= N_MIN) & (df.n_removed <= EDIT_FRAC * df.n_used) & (df.rms_cm > 0) & (df.rms_cm < 500)
                & (df.slope_sn.abs() < 0.5) & (df.slope_we.abs() < 0.5)]
        n1 = len(df)
        tr = Transformer.from_crs("EPSG:4326", CRS[hemi], always_xy=True)
        x, y = tr.transform(df.lon.values, df.lat.values)
        ice, shelf, dist = mask_lookup(hemi, x, y)
        df = df.assign(x=x.astype(np.float32), y=y.astype(np.float32), shelf=shelf, dist_km=dist, hemi=hemi, year=int(yr))
        df = df[ice]
        n2 = len(df)
        # centre = track 0 (80 m nadir block); off-centre = segments with |dist| >= 30 m;
        # the middle segment (duplicates the nadir block) is dropped
        df = df[(df.track == 0) | (df.dist_m.abs() >= 30)]
        df["centre"] = df.track == 0
        df["slope"] = np.hypot(df.slope_sn, df.slope_we).astype(np.float32)
        # per-day floor from centre platelets
        fl = []
        for d, g in df[df.centre].groupby("date"):
            fl.append({"date": d, "hemi": hemi, "year": int(yr), "n_centre": len(g),
                       "floor_cm": float(np.percentile(g.rms_cm, FLOOR_PCT)) if len(g) >= 200 else np.nan,
                       "p50_cm": float(g.rms_cm.median()), "n_used_med": float(g.n_used.median()),
                       "h_med": float(g.h.median()), "n_all": int((df.date == d).sum())})
        fl = pd.DataFrame(fl)
        camp_floor = np.nanmedian(fl.floor_cm) if fl.floor_cm.notna().any() else np.nan
        fl["floor_cm"] = fl.floor_cm.fillna(camp_floor); fl["floor_src"] = np.where(fl.n_centre >= 200, "day", "campaign")
        df = df.merge(fl[["date", "floor_cm"]], on="date", how="left")
        df["r_cm"] = np.sqrt(np.maximum(df.rms_cm**2 - df.floor_cm**2, 0)).astype(np.float32)
        df["at_floor"] = df.rms_cm < df.floor_cm * 10 ** (FLOOR_DB / 20)
        print(f"{yr} {hemi}: {n0} -> qc {n1} ({n1/n0:.2f}) -> on-ice {n2} ({n2/n0:.2f}); days {df.date.nunique()}; "
              f"floor med {camp_floor:.1f} cm; at_floor {df.at_floor.mean():.2f}; centre frac {df.centre.mean():.2f}", flush=True)
        parts.append(df); flights.append(fl)
    df = pd.concat(parts, ignore_index=True)
    fl = pd.concat(flights, ignore_index=True)
    df.to_parquet(CACHE / "platelets_qc.parquet", index=False)
    fl.to_csv(OUT / "flights.csv", index=False)
    print("total", len(df), "days", len(fl))


if __name__ == "__main__":
    main()
