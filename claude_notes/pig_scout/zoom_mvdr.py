import numpy as np, pickle, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
C=299792458.0
S=pickle.load(open("assembled.pkl","rb")); M=pickle.load(open("assembled_mvdr.pkl","rb"))
JOBS=[("A","2012_Antarctica_DC8__20121023_04",64,100,14,28),
      ("A","2009_Antarctica_DC8__20091020_06",64,100,14,28),
      ("A","2012_Antarctica_DC8__20121023_04",0,36,0,20)]
fig,axs=plt.subplots(len(JOBS)*2,1,figsize=(15,2.5*len(JOBS)*2))
r=0
for cid,k,s0,s1,y0,y1 in JOBS:
    for lbl,src in (("CSARP_standard",S),("CSARP_mvdr",M)):
        p=src[cid]['passes'][k]; p0=S[cid]['prof']; ax=axs[r]; r+=1
        rel=(p['twtt'][None,:]-p['srf'][:,None])*1e6; dt=p['dt']*1e6
        g=np.arange(y0,y1,dt/2)
        j=((g[None,:]-rel[:,0][:,None])/dt).round().astype(int)
        ok=(j>=0)&(j<p['D'].shape[1]); rows=np.repeat(np.arange(len(p['s']))[:,None],len(g),1)
        out=np.full((len(p['s']),len(g)),np.nan,np.float32); out[ok]=p['D'][rows[ok],j[ok]]
        dB=10*np.log10(np.maximum(out,1e-30))
        m=(p['s']/1e3>=s0)&(p['s']/1e3<=s1); sub=dB[m]
        vmax=np.nanpercentile(sub,99.8); vmin=np.nanpercentile(sub,35)
        ax.imshow(sub.T,aspect='auto',origin='upper',cmap='gray_r',
                  extent=[s0,s1,g[-1],g[0]],vmin=vmin,vmax=vmax,interpolation='nearest')
        ax.plot(p0['s'],2*p0['thick']*np.sqrt(3.15)/C*1e6,lw=0.8,color='C3',alpha=.8)
        ax.set_xlim(s0,s1); ax.set_ylabel("t−t_surf (µs)",fontsize=8); ax.tick_params(labelsize=8)
        mv=lbl.endswith('mvdr')
        ax.set_title(f"{cid}: {k.split('__')[1]}  AGL {p['agl']:.0f} m  s={s0}–{s1} km  |  {lbl}",
                     fontsize=9.5,loc='left',color=('C0' if mv else 'k'),
                     fontweight=('bold' if mv else 'normal'))
fig.tight_layout(); fig.savefig("mvdr_zoom.png",dpi=115); print("ok")
