import numpy as np, pandas as pd
from scipy.spatial import cKDTree
T=np.load("tracks.npy",allow_pickle=True).item()
P=np.load("prof.npy",allow_pickle=True).item()
def thin(k,step):
    d=T[k];x,y=d['x'],d['y']
    s=np.concatenate([[0],np.cumsum(np.hypot(np.diff(x),np.diff(y)))])
    i=np.unique(np.searchsorted(s,np.arange(0,s[-1],step))); return i[i<len(x)]
trees={}
for k in T:
    i=thin(k,100.); trees[k]=(cKDTree(np.c_[T[k]['x'][i],T[k]['y'][i]]),i)
for w,p in P.items():
    Q=np.c_[p['x'],p['y']]
    print(f"=== window {w}  ({p['high'].split('__')[1]}, {p['s'][-1]:.0f} km, lat {p['lat'][0]:.2f}->{p['lat'][-1]:.2f} lon {p['lon'][0]:.2f}->{p['lon'][-1]:.2f})")
    for k,(tree,i) in trees.items():
        dd,_=tree.query(Q); c=(dd<600).mean()
        if c<0.75: continue
        agl=np.nanmedian(T[k]['agl'])
        # bed frac of that pass over the window footprint
        j=tree.query(Q)[1]
        idx=i[j]
        bf=np.isfinite(T[k]['b_twtt'][idx]).mean()
        print(f"    {k:36s} cov={c:.2f} d_med={np.median(dd[dd<600]):6.1f} AGL={agl:8.0f} bedfrac={bf:.2f}")
