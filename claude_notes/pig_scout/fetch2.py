import pandas as pd, os, sys, json
from concurrent.futures import ThreadPoolExecutor
import xopr.ops_api as o

segs = pd.read_csv("pig_segments.csv")
todo = [(r.collection, r.segment_path) for _, r in segs.iterrows()
        if not os.path.exists(f"lyr/{r.collection}__{r.segment_path}.parquet")]
print(len(todo), "todo", flush=True)

def one(t):
    col, sp = t
    out = f"lyr/{col}__{sp}.parquet"
    try:
        res = o.get_layer_points(segment_name=sp, season_name=col, raise_errors=False)
        if isinstance(res, str): res = json.loads(res)
        d = res.get('data') if isinstance(res, dict) else None
        if isinstance(d, str): d = json.loads(d)
        if not d: return f"EMPTY {col} {sp} {str(res)[:120]}"
        df = pd.DataFrame({k: d[k] for k in ['lyr_id','gps_time','twtt','quality','lon','lat','elev']})
        df.to_parquet(out)
        return f"OK {col} {sp} {len(df)}"
    except Exception as e:
        return f"FAIL {col} {sp} {repr(e)[:160]}"

with ThreadPoolExecutor(max_workers=6) as ex:
    for m in ex.map(one, todo):
        print(m, flush=True)
