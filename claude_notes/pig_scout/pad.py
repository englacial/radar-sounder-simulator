import json, pandas as pd
F=pd.read_csv("frame_times.csv")
C=json.load(open("cands_final.json"))
for cid,r in C.items():
    for k,v in r['passes'].items():
        col,sp=k.split("__")
        avail=sorted(F[(F.collection==col)&(F.sp==sp)].frame.tolist())
        nums=[int(f.split('_')[-1]) for f in v['frames']]
        lo,hi=min(nums)-1,max(nums)+1
        new=[n for n in avail if lo<=n<=hi]
        v['frames']=[f"{sp}_{n:03d}" for n in new]
json.dump(C,open("cands_final.json","w"),indent=1)
print(sum(len(v['frames']) for r in C.values() for v in r['passes'].values()))
