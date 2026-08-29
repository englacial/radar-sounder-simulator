import numpy as np, pickle, json, pandas as pd
from scipy.spatial import cKDTree
from rasterio.transform import Affine
A=pickle.load(open("assembled.pkl","rb")); m=np.load("bm_extra.npz")['mask']
tfl,crs,mc=json.load(open("bedtf.json")); tf=Affine(*tfl)
G=pd.read_csv("/home/thomasteisberg/Documents/coherent-radar-simulator/outputs/atm_regional/grid_aa.csv")
gt=cKDTree(np.c_[G.xc,G.yc])
T2=pd.read_csv("/home/thomasteisberg/Documents/coherent-radar-simulator/outputs/atm_regional/tier2/site_medians.csv")
T2=T2[T2.hemi=='aa']; st=cKDTree(np.c_[T2.x,T2.y])
rows=[];prof={}
for cid,rec in A.items():
    p=rec['prof']
    i=((tf.f-p['y'])/500).astype(int); j=((p['x']-tf.c)/500).astype(int)
    mk=m[np.clip(i,0,m.shape[0]-1),np.clip(j,0,m.shape[1]-1)]
    d,k=gt.query(np.c_[p['x'],p['y']],distance_upper_bound=12e3)
    r=np.where(np.isfinite(d),G.r_med.values[np.clip(k,0,len(G)-1)],np.nan)
    prof[cid]=dict(s=p['s'],r=r,mask=mk)
    g=mk==2
    d2,k2=st.query(np.c_[p['x'][::10],p['y'][::10]],distance_upper_bound=20e3)
    sid=sorted(set(T2.site.values[k2[np.isfinite(d2)]]))
    sub=T2[T2.site.isin(sid)]
    rows.append(dict(cand=cid, gl_s_km=round(float(p['s'][np.flatnonzero(mk[1:]!=mk[:-1])[0]+1]),1) if (mk[1:]!=mk[:-1]).any() else np.nan,
        grounded_frac=round(float(g.mean()),2),
        r_all_p50=round(float(np.nanmedian(r)),1), r_all_p90=round(float(np.nanpercentile(r,90)),1),
        r_gnd_p50=round(float(np.nanmedian(r[g])),1), r_gnd_p90=round(float(np.nanpercentile(r[g],90)),1),
        t2_sites=",".join(sid),
        t2_adeq=f"{int(sub.adequate.sum())}/{len(sub)}" if len(sub) else "-",
        t2_l=round(float(sub.e_l.median()),1) if len(sub) else np.nan,
        t2_sigma=round(float(sub.e_sigma_cm.median()),1) if len(sub) else np.nan))
np.save("rprof.npy",prof,allow_pickle=True)
df=pd.DataFrame(rows); pd.set_option("display.width",250)
print(df.to_string(index=False)); df.to_csv("rough2.csv",index=False)
