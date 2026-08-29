import numpy as np, pandas as pd
from scipy.spatial import cKDTree
P=np.load("prof.npy",allow_pickle=True).item()
G=pd.read_csv("/home/thomasteisberg/Documents/coherent-radar-simulator/outputs/atm_regional/grid_aa.csv")
T2=pd.read_csv("/home/thomasteisberg/Documents/coherent-radar-simulator/outputs/atm_regional/tier2/site_medians.csv")
T2=T2[T2.hemi=='aa']
gt=cKDTree(np.c_[G.xc,G.yc]); st=cKDTree(np.c_[T2.x,T2.y])
rows=[]
for i,p in P.items():
    d,j=gt.query(np.c_[p['x'],p['y']], distance_upper_bound=12e3)
    ok=np.isfinite(d)
    r=G.r_med.values[j[ok]] if ok.any() else np.array([np.nan])
    # tier2 sites within 25 km
    d2,j2=st.query(np.c_[p['x'][::20],p['y'][::20]], distance_upper_bound=25e3)
    sid=sorted(set(T2.site.values[j2[np.isfinite(d2)]]))
    sub=T2[T2.site.isin(sid)]
    rows.append(dict(win=i, atm_cells=int(ok.sum()), r_med_p50=np.nanmedian(r),
                     r_med_p90=np.nanpercentile(r,90) if ok.any() else np.nan,
                     r_med_max=np.nanmax(r) if ok.any() else np.nan,
                     n_t2=len(sub), t2_adeq=sub.adequate.mean() if len(sub) else np.nan,
                     t2_adeq_bragg=sub.adequate_bragg.mean() if len(sub) else np.nan,
                     t2_sigma_cm=sub.e_sigma_cm.median() if len(sub) else np.nan,
                     t2_l_m=sub.e_l.median() if len(sub) else np.nan,
                     t2_mis15=sub.mis15.abs().max() if len(sub) else np.nan,
                     sites=",".join(sid)))
df=pd.DataFrame(rows)
pd.set_option("display.width",300);pd.set_option("display.max_columns",30)
print(df.to_string(index=False,float_format=lambda v:f"{v:.2f}"))
df.to_csv("rough_by_window.csv",index=False)
