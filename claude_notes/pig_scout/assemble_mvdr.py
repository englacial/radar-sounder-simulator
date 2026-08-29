"""Concatenate frames per pass; keep the single time-contiguous transit of the line."""
import numpy as np, json, pickle
from pyproj import Transformer
from soundersim import opr
tr=Transformer.from_crs("EPSG:4326","EPSG:3031",always_xy=True)
C=json.load(open("cands_final.json"))
P=np.load("prof.npy",allow_pickle=True).item()
out={}
for cid,r in C.items():
    p=P[r['window']]; ax,ay=p['x'],p['y']
    sref=np.r_[0,np.cumsum(np.hypot(np.diff(ax),np.diff(ay)))]; L=sref[-1]
    passes={}
    for k,v in r['passes'].items():
        if v['agl']<2000: continue
        col=k.split("__")[0]
        D=[];lat=[];lon=[];elev=[];srf=[];tw=None;fids=[];t=[]
        for f in sorted(v['frames']):
            ds=opr.load_frame(col,f,data_product="CSARP_mvdr")
            D.append(np.asarray(ds.Data.values)); lat.append(ds.Latitude.values)
            lon.append(ds.Longitude.values); elev.append(ds.Elevation.values)
            srf.append(ds.Surface.values)
            t.append(ds.slow_time.values.astype('datetime64[ns]').astype(np.int64)/1e9)
            fids += [f]*ds.sizes['slow_time']
            if tw is None: tw=ds.twtt.values
            elif not np.allclose(ds.twtt.values,tw): raise SystemExit(f"twtt grid differs {f}")
        D=np.concatenate(D,0); lat=np.concatenate(lat); lon=np.concatenate(lon)
        elev=np.concatenate(elev); srf=np.concatenate(srf); t=np.concatenate(t)
        fids=np.array(fids)
        o=np.argsort(t)                      # time order
        D,lat,lon,elev,srf,t,fids=[a[o] for a in (D,lat,lon,elev,srf,t,fids)]
        x,y=tr.transform(lon,lat)
        d2=(x[:,None]-ax[None,:])**2+(y[:,None]-ay[None,:])**2
        j=d2.argmin(1); perp=np.sqrt(d2[np.arange(len(x)),j]); s=sref[j]
        good=(perp<1500)&(s>0)&(s<L)
        # longest contiguous run in time
        m=np.r_[False,good,False].astype(int); d=np.diff(m)
        runs=list(zip(np.where(d==1)[0],np.where(d==-1)[0]))
        a,b=max(runs,key=lambda ab:np.ptp(sref[j[ab[0]:ab[1]]]))
        idx=np.arange(a,b)
        rev=s[idx][-1]<s[idx][0]
        if rev: idx=idx[::-1]
        passes[k]=dict(agl=v['agl'],twtt=tw,D=D[idx],s=s[idx],perp=perp[idx],
                       elev=elev[idx],srf=srf[idx],fids=fids[idx],lat=lat[idx],lon=lon[idx],
                       frames=sorted(set(fids[idx].tolist())),d_med=float(np.median(perp[idx])),
                       reversed=bool(rev),dt=float(tw[1]-tw[0]))
        print(cid,k,len(idx),"tr, s",round(float(s[idx].min())/1e3,1),"-",round(float(s[idx].max())/1e3,1),
              "km, perp med/p95",round(float(np.median(perp[idx]))),"/",round(float(np.percentile(perp[idx],95))),
              "rev",int(rev),"frames",passes[k]['frames'],flush=True)
    out[cid]=dict(meta=r,seg=dict(x=ax,y=ay,s=sref),passes=passes,prof=p)
pickle.dump(out,open("assembled_mvdr.pkl","wb"))
