"""Pull ILATM2 (Icessn platelets) per campaign (year x hemisphere), parse to one
parquet per campaign, delete the raw CSVs (archive is ~10 GB; parquet ~1 GB).

  uv run claude_notes/atm_regional/atm2_pull.py [--years 2009 2019] [--hemi gl aa] [--keep-raw]
Outputs: outputs/cache/atm2/platelets_<year>_<hemi>.parquet, granules_<year>_<hemi>.json
All ILATM2 files (2009-2019) share one CSV layout ('#' comments, 11 columns):
  UTC s-of-day, lat, lon(0-360), h WGS84 (m), S-N slope, W-E slope, RMS_Fit (cm),
  n_used, n_removed, distance right of aircraft (m), track id (0 = 80 m nadir block,
  1..3 (or 1..5) = cross-track segments).
"""
from __future__ import annotations
import argparse, json, shutil, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "cache" / "atm2"
BBOX = {"gl": (-75, 58, -10, 84), "aa": (-180, -90, 180, -60)}
COLS = ["utc_s", "lat", "lon", "h", "slope_sn", "slope_we", "rms_cm", "n_used", "n_removed", "dist_m", "track"]


def parse_file(p: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(p, comment="#", header=None, names=COLS, skipinitialspace=True, dtype=np.float64)
    except Exception as e:
        print(f"  !! parse fail {p.name}: {e}"); return None
    if df.empty: return None
    name = p.name.split("_")
    ds, ts = name[1], name[2]
    df["date"] = pd.Timestamp(f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}")
    df["lon"] = ((df.lon + 180) % 360) - 180
    df["granule"] = p.name
    for c in ("lat", "lon"): df[c] = df[c].astype(np.float64)
    for c in ("h", "slope_sn", "slope_we", "rms_cm", "dist_m", "utc_s"): df[c] = df[c].astype(np.float32)
    for c in ("n_used", "n_removed", "track"): df[c] = df[c].astype(np.int32)
    return df


def pull_campaign(yr, hemi, keep_raw, threads=8):
    import earthaccess
    out = CACHE / f"platelets_{yr}_{hemi}.parquet"
    if out.exists():
        print(f"{yr} {hemi}: exists, skip"); return
    t0 = time.time()
    res = earthaccess.search_data(short_name="ILATM2", bounding_box=BBOX[hemi], temporal=(f"{yr}-01-01", f"{yr}-12-31"))
    mb = sum(float(g.size()) for g in res)
    print(f"{yr} {hemi}: {len(res)} granules, {mb:.0f} MB", flush=True)
    if not res: return
    raw = CACHE / "raw" / f"{yr}_{hemi}"; raw.mkdir(parents=True, exist_ok=True)
    todo = [g for g in res if not (raw / g["umm"]["GranuleUR"]).exists()]
    for attempt in range(3):
        if not todo: break
        try:
            earthaccess.download(todo, str(raw), threads=threads)
        except Exception as e:
            print(f"  download error (attempt {attempt}): {e}")
        todo = [g for g in res if not (raw / g["umm"]["GranuleUR"]).exists()]
    t1 = time.time()
    files = sorted(raw.glob("ILATM2_*"))
    dfs = [d for d in (parse_file(p) for p in files) if d is not None]
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(out, index=False)
    log = {"year": yr, "hemi": hemi, "n_granules_search": len(res), "n_files": len(files), "n_missing": len(todo),
           "mb": mb, "n_platelets": int(len(df)), "n_days": int(df.date.nunique()),
           "dates": sorted(str(d.date()) for d in df.date.unique()),
           "download_s": round(t1 - t0), "parse_s": round(time.time() - t1),
           "granules": [{"id": g["umm"]["GranuleUR"], "mb": float(g.size())} for g in res]}
    (CACHE / f"granules_{yr}_{hemi}.json").write_text(json.dumps(log, indent=1))
    print(f"  -> {len(df)} platelets, {df.date.nunique()} days, dl {t1-t0:.0f}s parse {time.time()-t1:.0f}s", flush=True)
    if not keep_raw:
        shutil.rmtree(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int, default=list(range(2009, 2020)))
    ap.add_argument("--hemi", nargs="*", default=["gl", "aa"])
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    import earthaccess
    earthaccess.login(strategy="netrc")
    for yr in a.years:
        for h in a.hemi:
            pull_campaign(yr, h, a.keep_raw)


if __name__ == "__main__":
    main()
