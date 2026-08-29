import numpy as np, pandas as pd, json
from scipy.spatial import cKDTree
from pyproj import Transformer
C=299792458.0
T=np.load("tracks.npy",allow_pickle=True).item()
LON0,LON1,LAT0,LAT1=-105.0,-94.0,-77.2,-74.4
def thin(k,step):
    d=T[k];x,y=d['x'],d['y']
    s=np.concatenate([[0],np.cumsum(np.hypot(np.diff(x),np.diff(y)))])
    i=np.unique(np.searchsorted(s,np.arange(0,s[-1],step))); return i[i<len(x)],s
high=[k for k in T if np.nanpercentile(T[k]['agl'],90)>8000]
low=[k for k in T if np.nanmedian(T[k]['agl'])<2000]
lowT={}
for k in low:
    i,_=thin(k,100.); lowT[k]=(cKDTree(np.c_[T[k]['x'][i],T[k]['y'][i]]),i)
def runs(m):
    a=np.r_[False,m,False].astype(int);d=np.diff(a)
    return list(zip(np.where(d==1)[0],np.where(d==-1)[0]))
rows=[]
for hk in high:
    hi,hs=thin(hk,50.); d=T[hk]
    x,y,s=d['x'][hi],d['y'][hi],hs[hi]
    lat,lon,agl=d['lat'][hi],d['lon'][hi],d['agl'][hi]
    stw,btw,elev=d['s_twtt'][hi],d['b_twtt'][hi],d['elev'][hi]
    inbox=(lon>LON0)&(lon<LON1)&(lat>LAT0)&(lat<LAT1)
    if not inbox.any(): continue
    P=np.c_[x,y]
    best=np.full(len(P),np.inf); who=np.empty(len(P),object)
    for lk,(tree,li) in lowT.items():
        if lk.split('__')[1][:8]==hk.split('__')[1][:8]: continue
        dd,_=tree.query(P); m=dd<best; best[m]=dd[m]; who[m]=lk
    ok=inbox&(best<800)
    for a,b in runs(ok):
        if s[b-1]-s[a]<45e3: continue
        # best straight sub-window 50-100km
        cand=None
        for Wkm in (100,90,80,70,60,50):
            Wm=Wkm*1e3; j=a
            while j<b:
                k2=np.searchsorted(s[a:b],s[j]+Wm)+a
                if k2>=b: break
                xs,ys=x[j:k2],y[j:k2]
                v=np.array([xs[-1]-xs[0],ys[-1]-ys[0]]);v/=np.hypot(*v);n=np.array([-v[1],v[0]])
                dev=np.abs((xs-xs[0])*n[0]+(ys-ys[0])*n[1]).max()
                bt,st=btw[j:k2],stw[j:k2]; bf=np.isfinite(bt).mean()
                if dev<300 and bf>0.5:
                    th=C*(bt-st)/2/np.sqrt(3.15); be=elev[j:k2]-C*st/2-th
                    rel=np.nanpercentile(be,95)-np.nanpercentile(be,5)
                    sc=Wkm/100+bf+rel/1000-dev/1000
                    if cand is None or sc>cand['score']:
                        cand=dict(high=hk,low=pd.Series(who[j:k2]).mode()[0],
                                  n_low=len(set(who[j:k2])), j=j,k=k2,W=Wkm,
                                  L_km=(s[k2-1]-s[j])/1e3,dev=dev,bed_frac=bf,
                                  agl_h=np.nanmedian(agl[j:k2]),d_med=np.median(best[j:k2]),
                                  d_max=best[j:k2].max(),thick=np.nanmedian(th),relief=rel,
                                  lat0=lat[j],lon0=lon[j],lat1=lat[k2-1],lon1=lon[k2-1],
                                  mx=(xs[0]+xs[-1])/2,my=(ys[0]+ys[-1])/2,score=sc)
                j=np.searchsorted(s[a:b],s[j]+2.5e3)+a
            if cand is not None: break
        if cand: rows.append(cand)
df=pd.DataFrame(rows).sort_values("score",ascending=False)
pd.set_option("display.width",320);pd.set_option("display.max_columns",40)
print(len(df))
print(df.drop(columns=['mx','my']).to_string(index=False,float_format=lambda v:f"{v:.1f}"))
df.to_csv("runs.csv",index=False)
