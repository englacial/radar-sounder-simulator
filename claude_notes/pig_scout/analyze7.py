import numpy as np, pandas as pd
from scipy.spatial import cKDTree
C=299792458.0
T=np.load("tracks.npy",allow_pickle=True).item()
R=pd.read_csv("runs.csv")
low=[k for k in T if np.nanmedian(T[k]['agl'])<2000]
def thin(k,step):
    d=T[k];x,y=d['x'],d['y']
    s=np.concatenate([[0],np.cumsum(np.hypot(np.diff(x),np.diff(y)))])
    i=np.unique(np.searchsorted(s,np.arange(0,s[-1],step))); return i[i<len(x)],s
lowT={}
for k in low:
    i,_=thin(k,100.); lowT[k]=(cKDTree(np.c_[T[k]['x'][i],T[k]['y'][i]]), T[k], i)
prof={}
for idx,r in R.iterrows():
    hk=r.high; hi,hs=thin(hk,50.); d=T[hk]
    j,k2=int(r.j),int(r.k)
    x,y=d['x'][hi][j:k2],d['y'][hi][j:k2]
    P=np.c_[x,y]
    covs=[]
    for lk,(tree,td,li) in lowT.items():
        if lk.split('__')[1][:8]==hk.split('__')[1][:8]: continue
        dd,ii=tree.query(P)
        c=(dd<500).mean()
        if c>0.7:
            covs.append((lk, round(float(c),2), round(float(np.median(dd[dd<500])),1),
                         round(float(np.nanmedian(td['agl'])),0)))
    covs.sort(key=lambda t:-t[1])
    st,bt,el=d['s_twtt'][hi][j:k2],d['b_twtt'][hi][j:k2],d['elev'][hi][j:k2]
    th=C*(bt-st)/2/np.sqrt(3.15); srf=el-C*st/2; be=srf-th
    s_km=(hs[hi][j:k2]-hs[hi][j])/1e3
    prof[idx]=dict(s=s_km,bed=be,srf=srf,x=x,y=y,lat=d['lat'][hi][j:k2],lon=d['lon'][hi][j:k2],
                   agl=d['agl'][hi][j:k2],thick=th, high=hk, j=j,k=k2)
    print(f"--- {idx}: {hk} j={j} L={r.L_km:.0f}km bedfrac={r.bed_frac:.2f} relief={r.relief:.0f}m "
          f"({r.lat0:.2f},{r.lon0:.2f})->({r.lat1:.2f},{r.lon1:.2f})")
    for c in covs: print("      low:",c)
np.save("prof.npy",prof,allow_pickle=True)
