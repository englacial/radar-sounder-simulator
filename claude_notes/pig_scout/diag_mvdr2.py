"""Bed visibility, standard vs mvdr: peak-over-clutter (window ABOVE the bed)
and peak-over-noise (window BELOW), split by ice thickness."""
import numpy as np, pickle, pandas as pd
C=299792458.0
S=pickle.load(open("assembled.pkl","rb")); M=pickle.load(open("assembled_mvdr.pkl","rb"))
def metrics(p,p0):
    tb=np.interp(p['s']/1e3,p0['s'],2*p0['thick']*np.sqrt(3.15)/C,left=np.nan,right=np.nan)
    thk=np.interp(p['s']/1e3,p0['s'],p0['thick'],left=np.nan,right=np.nan)
    rel0=p['twtt'][0]-p['srf']; dt=p['dt']; n=p['D'].shape[1]
    i0=(tb-rel0)/dt
    w=int(round(0.5e-6/dt))+1
    a1,a2=int(round(5e-6/dt)),int(round(1.5e-6/dt))   # clutter window above bed
    b1,b2=int(round(3e-6/dt)),int(round(6e-6/dt))     # noise window below bed
    ca=np.full(len(p['s']),np.nan); cb=np.full(len(p['s']),np.nan)
    for k in range(len(p['s'])):
        if not np.isfinite(i0[k]): continue
        c0=int(round(i0[k]))
        if c0-w<0 or c0-a1<0: continue
        pk=np.nanmax(p['D'][k,c0-w:c0+w+1])
        up=np.nanmedian(p['D'][k,c0-a1:c0-a2])
        if pk>0 and up>0: ca[k]=10*np.log10(pk/up)
        if c0+b2<n:
            dn=np.nanmedian(p['D'][k,c0+b1:c0+b2])
            if pk>0 and dn>0: cb[k]=10*np.log10(pk/dn)
    return ca,cb,thk,p['s']/1e3
rows=[]
for cid in S:
    p0=S[cid]['prof']
    for k in M[cid]['passes']:
        ca_s,cb_s,thk_s,s_s=metrics(S[cid]['passes'][k],p0)
        ca_m,cb_m,thk_m,s_m=metrics(M[cid]['passes'][k],p0)
        grid=np.arange(0,p0['s'][-1],0.05)          # common 50 m axis
        rs=lambda x,y: np.interp(grid,x,y,left=np.nan,right=np.nan)
        ca_s,cb_s=rs(s_s,ca_s),rs(s_s,cb_s); ca_m,cb_m=rs(s_m,ca_m),rs(s_m,cb_m)
        thk=np.interp(grid,p0['s'],p0['thick'],left=np.nan,right=np.nan)
        for lbl,sel in (("all",np.isfinite(thk)),
                        ("ice<1000 m",thk<1000),("ice>1000 m",thk>=1000)):
            if sel.sum()<100: continue
            f=lambda a: np.nanmedian(a[sel])
            g=lambda a: np.nanmean(a[sel][np.isfinite(a[sel])]>6)
            rows.append(dict(cand=cid,seg=k.split('__')[1],agl=round(S[cid]['passes'][k]['agl']),
                part=lbl, n_km=round(float(sel.sum()*0.05)),
                cl_std=round(float(f(ca_s)),1), cl_mvdr=round(float(f(ca_m)),1),
                d_clut=round(float(f(ca_m)-f(ca_s)),1),
                vis_std=round(float(g(ca_s)),2), vis_mvdr=round(float(g(ca_m)),2),
                nf_std=round(float(f(cb_s)),1), nf_mvdr=round(float(f(cb_m)),1)))
df=pd.DataFrame(rows)
pd.set_option("display.width",240)
print(df.to_string(index=False))
df.to_csv("diag_mvdr2.csv",index=False)
