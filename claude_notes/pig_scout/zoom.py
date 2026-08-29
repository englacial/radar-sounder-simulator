import numpy as np, pickle, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
C=299792458.0
A=pickle.load(open("assembled.pkl","rb"))
JOBS=[("D","2012_Antarctica_DC8__20121023_04",55,100,-1,10),
      ("D","2016_Antarctica_DC8__20161104_05",55,100,-1,10),
      ("A","2012_Antarctica_DC8__20121023_04",0,45,-1,18),
      ("A","2016_Antarctica_DC8__20161104_05",0,45,-1,18),
      ("C","2012_Antarctica_DC8__20121023_04",45,100,-1,22),
      ("B","2012_Antarctica_DC8__20121023_04",25,60,-1,25)]
fig,axs=plt.subplots(len(JOBS),1,figsize=(15,3.0*len(JOBS)))
for ax,(cid,k,s0,s1,y0,y1) in zip(axs,JOBS):
    rec=A[cid]; p=rec['passes'][k]; p0=rec['prof']
    rel=(p['twtt'][None,:]-p['srf'][:,None])*1e6; dt=p['dt']*1e6
    grid=np.arange(y0,y1,dt/2)
    j=((grid[None,:]-rel[:,0][:,None])/dt).round().astype(int)
    ok=(j>=0)&(j<p['D'].shape[1]); rows=np.repeat(np.arange(len(p['s']))[:,None],len(grid),1)
    out=np.full((len(p['s']),len(grid)),np.nan,np.float32); out[ok]=p['D'][rows[ok],j[ok]]
    dB=10*np.log10(np.maximum(out,1e-30))
    m=(p['s']/1e3>=s0)&(p['s']/1e3<=s1)
    sub=dB[m]; vmax=np.nanpercentile(sub,99.9); vmin=np.nanpercentile(sub,40)
    ax.imshow(sub.T,aspect='auto',origin='upper',cmap='gray_r',
              extent=[s0,s1,grid[-1],grid[0]],vmin=vmin,vmax=vmax,interpolation='nearest')
    ax.plot(p0['s'],2*p0['thick']*np.sqrt(3.15)/C*1e6,lw=0.6,color='C3',alpha=.7)
    ax.set_xlim(s0,s1); ax.set_ylabel("t−t_surf (µs)",fontsize=8)
    ax.set_title(f"{cid}: {k.split('__')[0]} {k.split('__')[1]}  AGL {p['agl']:.0f} m   s={s0}-{s1} km",fontsize=9,loc='left')
fig.tight_layout(); fig.savefig("zooms.png",dpi=105); print("ok")
