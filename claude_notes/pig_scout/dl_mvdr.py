import pickle
from concurrent.futures import ThreadPoolExecutor
from soundersim import opr
A=pickle.load(open("assembled.pkl","rb"))
jobs=set()
for cid,rec in A.items():
    for k,p in rec['passes'].items():
        if p['agl']<2000: continue
        for f in p['frames']: jobs.add((k.split("__")[0],f))
jobs=sorted(jobs); print(len(jobs),"mvdr frames",flush=True)
def one(j):
    col,f=j
    try:
        d=opr.load_frame(col,f,data_product="CSARP_mvdr"); return f"OK {col} {f} {d.Data.shape}"
    except Exception as e: return f"FAIL {col} {f} {repr(e)[:160]}"
with ThreadPoolExecutor(max_workers=3) as ex:
    for m in ex.map(one,jobs): print(m,flush=True)
