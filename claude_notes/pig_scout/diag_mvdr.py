import numpy as np, pickle, pandas as pd
C=299792458.0
S=pickle.load(open("assembled.pkl","rb"))
M=pickle.load(open("assembled_mvdr.pkl","rb"))
def snr(p,p0):
    tb=np.interp(p['s']/1e3,p0['s'],2*p0['thick']*np.sqrt(3.15)/C,left=np.nan,right=np.nan)
    rel0=p['twtt'][0]-p['srf']; dt=p['dt']; n=p['D'].shape[1]
    i0=(tb-rel0)/dt
    w=int(round(0.5e-6/dt))+1; wn=int(round(3e-6/dt))
    out=np.full(len(p['s']),np.nan)
    for a in range(len(p['s'])):
        if not np.isfinite(i0[a]): continue
        c0=int(round(i0[a]))
        if c0-w<0 or c0+wn+50>=n: continue
        pk=np.nanmax(p['D'][a,c0-w:c0+w+1])
        nf=np.nanmedian(p['D'][a,c0+wn:min(c0+wn+200,n)])
        if nf>0 and pk>0: out[a]=10*np.log10(pk/nf)
    return out
rows=[]
for cid in S:
    p0=S[cid]['prof']
    for k in M[cid]['passes']:
        a=snr(S[cid]['passes'][k],p0); b=snr(M[cid]['passes'][k],p0)
        oa,ob=np.isfinite(a),np.isfinite(b)
        rows.append(dict(cand=cid,season=k.split('__')[0][:4],seg=k.split('__')[1],
            agl=round(S[cid]['passes'][k]['agl']),
            std_med=round(float(np.nanmedian(a)),1), mvdr_med=round(float(np.nanmedian(b)),1),
            d_med=round(float(np.nanmedian(b)-np.nanmedian(a)),1),
            std_f10=round(float((a[oa]>10).mean()),2), mvdr_f10=round(float((b[ob]>10).mean()),2),
            std_f5=round(float((a[oa]>5).mean()),2), mvdr_f5=round(float((b[ob]>5).mean()),2)))
df=pd.DataFrame(rows).sort_values(['cand','agl'])
pd.set_option("display.width",220); print(df.to_string(index=False))
df.to_csv("diag_mvdr.csv",index=False)
