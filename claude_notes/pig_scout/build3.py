import numpy as np, pandas as pd, json
from scipy.spatial import cKDTree
T=np.load("tracks.npy",allow_pickle=True).item()
P=np.load("prof.npy",allow_pickle=True).item()
F=pd.read_csv("frame_times.csv"); F['t0']=F.t0*1000.0
CAND={"A":0,"B":3,"C":2,"D":6,"E":10}
PASSES=["2009_Antarctica_DC8__20091020_06","2009_Antarctica_DC8__20091020_03",
        "2012_Antarctica_DC8__20121023_04","2014_Antarctica_DC8__20141029_05",
        "2016_Antarctica_DC8__20161104_05","2018_Antarctica_DC8__20181107_01"]
def thin(k,step):
    d=T[k];x,y=d['x'],d['y']
    s=np.concatenate([[0],np.cumsum(np.hypot(np.diff(x),np.diff(y)))])
    i=np.unique(np.searchsorted(s,np.arange(0,s[-1],step))); return i[i<len(x)]
trees={}
for k in PASSES:
    i=thin(k,100.); trees[k]=(cKDTree(np.c_[T[k]['x'][i],T[k]['y'][i]]),i)
def contiguous(ii, gap=300):
    m=len(ii)//2; a=b=m
    while a>0 and abs(int(ii[a])-int(ii[a-1]))<gap: a-=1
    while b<len(ii)-1 and abs(int(ii[b+1])-int(ii[b]))<gap: b+=1
    return a,b+1
out={}
for cid,w in CAND.items():
    p=P[w]; Q=np.c_[p['x'],p['y']]
    rec=dict(window=w, high_ref=p['high'], L_km=float(p['s'][-1]),
             lat0=float(p['lat'][0]),lon0=float(p['lon'][0]),
             lat1=float(p['lat'][-1]),lon1=float(p['lon'][-1]), passes={})
    for k in PASSES:
        tree,idx=trees[k]; dd,jj=tree.query(Q)
        if (dd<600).mean()<0.85: continue
        ii=idx[jj]; tt=T[k]['t'][ii]
        tm=np.median(tt); keep=np.abs(tt-tm)<1500.
        if keep.mean()<0.9: print("  !! partial", cid, k, round(float(keep.mean()),2))
        a,b=int(np.argmax(keep)), len(keep)-int(np.argmax(keep[::-1]))
        cov=float((dd[keep]<600).mean()); ta,tb=tt[keep].min(),tt[keep].max()
        col,sp=k.split("__")
        sub=F[(F.collection==col)&(F.sp==sp)].sort_values('frame').reset_index(drop=True)
        te=np.r_[sub.t0.values[1:], sub.t0.values[-1]+1e4]
        sel=sub[(te>ta-2)&(sub.t0.values<tb+2)]
        rec["passes"][k]=dict(agl=float(np.nanmedian(T[k]['agl'])), cov=cov,
            d_med=float(np.median(dd[keep][dd[keep]<600])), frac_win=float(keep.mean()),
            reversed=bool(ii[keep][-1]<ii[keep][0]), frames=sel.fid.tolist())
    out[cid]=rec
    print(f"{cid} w{w} {rec['L_km']:.0f}km ({rec['lat0']:.2f},{rec['lon0']:.2f})->({rec['lat1']:.2f},{rec['lon1']:.2f})")
    for k,v in rec['passes'].items():
        print(f"   {k.split('__')[1]:14s} AGL{v['agl']:7.0f} cov{v['cov']:.2f} d{v['d_med']:5.1f} rev={int(v['reversed'])} frames {v['frames'][0]}..{v['frames'][-1]} ({len(v['frames'])})")
json.dump(out,open("cands_final.json","w"),indent=1)
print("total frames:", sum(len(v['frames']) for r in out.values() for v in r['passes'].values()))
