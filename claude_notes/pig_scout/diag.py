import numpy as np, pickle, pandas as pd, json
from scipy.spatial import cKDTree
from rasterio.transform import Affine
C=299792458.0
A=pickle.load(open("assembled.pkl","rb"))
bmx=np.load("bm_extra.npz"); mask=bmx['mask']
tfl,crs,mc=json.load(open("bedtf.json")); tf=Affine(*tfl)
G=pd.read_csv("/home/thomasteisberg/Documents/coherent-radar-simulator/outputs/atm_regional/grid_aa.csv")
gt=cKDTree(np.c_[G.xc,G.yc])
rows=[]
for cid,rec in A.items():
    p0=rec['prof']
    # bed twtt rel surface, from the OPS pick on the reference high track
    sbed=p0['s']; tbed=2*p0['thick']*np.sqrt(3.15)/C   # s in km, twtt below surface
    d,j=gt.query(np.c_[p0['x'],p0['y']],distance_upper_bound=12e3)
    r=G.r_med.values[j[np.isfinite(d)]]
    mk_i=((tf.f-p0['y'])/500).astype(int); mk_j=((p0['x']-tf.c)/500).astype(int)
    mk=mask[np.clip(mk_i,0,mask.shape[0]-1),np.clip(mk_j,0,mask.shape[1]-1)]
    for k,p in sorted(rec['passes'].items(), key=lambda kv:-kv[1]['agl']):
        tb=np.interp(p['s']/1e3, sbed, tbed, left=np.nan, right=np.nan)
        rel=(p['twtt'][None,:]-p['srf'][:,None])
        dt=p['dt']
        i0=((tb-p['srf']*0)-rel[:,0])/dt
        n=p['D'].shape[1]; snr=np.full(len(p['s']),np.nan)
        w=int(round(0.5e-6/dt))+1; wn=int(round(3e-6/dt))
        for a in range(len(p['s'])):
            if not np.isfinite(i0[a]): continue
            c0=int(round(i0[a]))
            if c0-w<0 or c0+wn+50>=n: continue
            pk=np.nanmax(p['D'][a,c0-w:c0+w+1])
            nf=np.nanmedian(p['D'][a,c0+wn:c0+wn+200]) if c0+wn+200<n else np.nanmedian(p['D'][a,-200:])
            if nf>0 and pk>0: snr[a]=10*np.log10(pk/nf)
        ok=np.isfinite(snr)
        rows.append(dict(cand=cid,pass_=k.split('__')[1],season=k.split('__')[0][:4],
            agl=round(p['agl']),
            snr_med=round(float(np.nanmedian(snr)),1),
            f_snr5=round(float((snr[ok]>5).mean()),2) if ok.any() else np.nan,
            f_snr10=round(float((snr[ok]>10).mean()),2) if ok.any() else np.nan,
            f_eval=round(float(ok.mean()),2),
            ops_pick=round(float(np.isfinite(p0['thick']).mean()),2),
            thick_med=round(float(np.nanmedian(p0['thick']))),
            relief=round(float(np.nanpercentile(p0['bed'],95)-np.nanpercentile(p0['bed'],5))),
            atm_r_med=round(float(np.nanmedian(r)),1), atm_r_p90=round(float(np.nanpercentile(r,90)),1),
            grounded=round(float((mk==2).mean()),2), floating=round(float((mk==3).mean()),2)))
df=pd.DataFrame(rows); pd.set_option("display.width",250)
print(df.to_string(index=False))
df.to_csv("diag.csv",index=False)
