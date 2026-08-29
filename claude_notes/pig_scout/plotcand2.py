import numpy as np, pickle, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from rasterio.transform import Affine
C=299792458.0
A=pickle.load(open("assembled.pkl","rb"))
bed=np.load("bed.npy"); tfl,crs,mc=json.load(open("bedtf.json")); tf=Affine(*tfl)
H,W=bed.shape; ext=[tf.c,tf.c+W*tf.a,tf.f+H*tf.e,tf.f]
TITLE=json.load(open("titles.json")) if __import__("os").path.exists("titles.json") else {}
for cid,rec in A.items():
    ps=rec['passes']; keys=sorted(ps,key=lambda k:-ps[k]['agl']); n=len(keys)
    p0=rec['prof']
    th=p0['thick'][np.isfinite(p0['thick'])]
    tmax=2*np.percentile(th,99.5)*np.sqrt(3.15)/C*1e6*1.18+2.0
    fig=plt.figure(figsize=(15,2.4+2.5*n))
    gs=GridSpec(n+1,2,figure=fig,width_ratios=[1,4.3],height_ratios=[1.6]+[1]*n,hspace=0.36,wspace=0.13)
    axm=fig.add_subplot(gs[0,0])
    axm.imshow(bed,extent=ext,origin="upper",cmap="terrain",vmin=-1800,vmax=1200)
    for c2,r2 in A.items(): axm.plot(r2['seg']['x'],r2['seg']['y'],lw=1.0,color='0.3')
    axm.plot(rec['seg']['x'],rec['seg']['y'],lw=2.8,color='red')
    axm.plot(rec['seg']['x'][0],rec['seg']['y'][0],'o',ms=6,color='red')
    axm.set_xlim(-1.70e6,-1.50e6); axm.set_ylim(-3.6e5,-1.0e5); axm.set_aspect('equal')
    axm.set_xticks([]); axm.set_yticks([])
    axm.set_title("this line (red, dot = km 0)\nother candidates (grey)",fontsize=8)
    axp=fig.add_subplot(gs[0,1])
    axp.plot(p0['s'],p0['srf'],lw=1.1,color='C0',label='ice surface (radar pick)')
    axp.plot(p0['s'],p0['bed'],lw=1.1,color='C3',label='bed (OPS "bottom" pick)')
    axp.set_ylabel("elevation (m, WGS84)",fontsize=8); axp.legend(fontsize=7,ncol=2,loc='lower right')
    axp.grid(alpha=.3); axp.set_xlim(0,p0['s'][-1]); axp.tick_params(labelsize=8)
    axp.set_title(f"Candidate {cid} — {p0['s'][-1]:.0f} km,  "
                  f"({p0['lat'][0]:.3f}, {p0['lon'][0]:.3f}) → ({p0['lat'][-1]:.3f}, {p0['lon'][-1]:.3f})"
                  + (("\n"+TITLE[cid]) if cid in TITLE else ""),fontsize=11)
    ybed=2*p0['thick']*np.sqrt(3.15)/C*1e6
    for i,k in enumerate(keys):
        p=ps[k]; ax=fig.add_subplot(gs[i+1,:])
        rel=(p['twtt'][None,:]-p['srf'][:,None])*1e6; dt=p['dt']*1e6
        grid=np.arange(-2.0,tmax,dt/2)
        j=((grid[None,:]-rel[:,0][:,None])/dt).round().astype(int)
        ok=(j>=0)&(j<p['D'].shape[1])
        rows=np.repeat(np.arange(len(p['s']))[:,None],len(grid),1)
        out=np.full((len(p['s']),len(grid)),np.nan,np.float32); out[ok]=p['D'][rows[ok],j[ok]]
        dB=10*np.log10(np.maximum(out,1e-30))
        vmax=np.nanpercentile(dB,99.93); vmin=np.nanpercentile(dB,45)
        ax.imshow(dB.T,aspect='auto',origin='upper',cmap='gray_r',
                  extent=[p['s'][0]/1e3,p['s'][-1]/1e3,grid[-1],grid[0]],
                  vmin=vmin,vmax=vmax,interpolation='nearest')
        fids=p['fids']; ch=np.flatnonzero(fids[1:]!=fids[:-1]); edges=np.r_[0,ch+1,len(fids)]
        for a,b in zip(edges[:-1],edges[1:]):
            if b-a<200: continue
            ax.axvline(p['s'][a]/1e3,color='C0',lw=0.9,ls=':')
            ax.text((p['s'][a]+p['s'][b-1])/2e3,grid[0]+0.085*(grid[-1]-grid[0]),
                    fids[a],fontsize=8.5,ha='center',color='C0',weight='bold')
        ax.plot(p0['s'],ybed,lw=0.7,color='C3',alpha=.8)
        ax.set_ylabel("t − t_surface (µs)",fontsize=8); ax.tick_params(labelsize=8)
        ax.set_xlim(0,p0['s'][-1])
        ax.set_title(f"{k.split('__')[0]}  {k.split('__')[1]}   |   AGL {p['agl']:.0f} m   |   "
                     f"cross-track offset from the line: median {p['d_med']:.0f} m, p95 "
                     f"{np.percentile(p['perp'],95):.0f} m   |   CSARP_standard",
                     fontsize=9,loc='left')
        if i==n-1: ax.set_xlabel("distance along line (km)")
    fig.savefig(f"cand_{cid}.png",dpi=100,bbox_inches='tight'); plt.close(fig); print("wrote",cid,flush=True)
