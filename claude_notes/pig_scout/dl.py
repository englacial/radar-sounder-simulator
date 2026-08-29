import json, sys, traceback
from concurrent.futures import ThreadPoolExecutor
from soundersim import opr
C=json.load(open("cands_final.json"))
jobs=set()
for r in C.values():
    for k,v in r['passes'].items():
        col=k.split("__")[0]
        for f in v['frames']: jobs.add((col,f))
jobs=sorted(jobs); print(len(jobs),"frames",flush=True)
def one(j):
    col,f=j
    try:
        d=opr.load_frame(col,f); return f"OK {col} {f} {d.Data.shape}"
    except Exception as e: return f"FAIL {col} {f} {repr(e)[:150]}"
with ThreadPoolExecutor(max_workers=4) as ex:
    for m in ex.map(one,jobs): print(m,flush=True)
