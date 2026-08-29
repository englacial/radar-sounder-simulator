import numpy as np, pickle, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
C=299792458.0
S=pickle.load(open("assembled.pkl","rb")); M=pickle.load(open("assembled_mvdr.pkl","rb"))

def img(p,tmax):
    rel=(p['twtt'][None,:]-p['srf'][:,None])*1e6; dt=p['dt']*1e6
    g=np.arange(-2.0,tmax,dt/2)
    j=((g[None,:]-rel[:,0][:,None])/dt).round().astype(int)
    ok=(j>=0)&(j<p['D'].shape[1]); rows=np.repeat(np.arange(len(p['s']))[:,None],len(g),1)
    out=np.full((len(p['s']),len(g)),np.nan,np.float32); out[ok]=p['D'][rows[ok],j[ok]]
    return 10*np.log10(np.maximum(out,1e-30)), g

def contrast(p,p0):
    """bed peak (±0.5 µs) over the median power 1.5-5 µs above it, in dB."""
    tb=np.interp(p['s']/1e3,p0['s'],2*p0['thick']*np.sqrt(3.15)/C,left=np.nan,right=np.nan)
    rel0=p['twtt'][0]-p['srf']; dt=p['dt']
    i0=(tb-rel0)/dt; w=int(round(0.5e-6/dt))+1
    a1,a2=int(round(5e-6/dt)),int(round(1.5e-6/dt))
    c=np.full(len(p['s']),np.nan)
    for k in range(len(p['s'])):
        if not np.isfinite(i0[k]): continue
        k0=int(round(i0[k]))
        if k0-w<0 or k0-a1<0 or k0+w+1>p['D'].shape[1]: continue
        pk=np.nanmax(p['D'][k,k0-w:k0+w+1]); up=np.nanmedian(p['D'][k,k0-a1:k0-a2])
        if pk>0 and up>0: c[k]=10*np.log10(pk/up)
    return p['s']/1e3, c

def smooth(s,c,win=1.0):
    g=np.arange(0,s[-1],0.05); v=np.interp(g,s,c,left=np.nan,right=np.nan)
    n=max(3,int(win/0.05)); k=np.ones(n)/n
    ok=np.isfinite(v); vv=np.where(ok,v,0.0)
    num=np.convolve(vv,k,'same'); den=np.convolve(ok.astype(float),k,'same')
    out=np.where(den>0.3,num/np.maximum(den,1e-9),np.nan)
    return g,out

for cid in S:
    p0=S[cid]['prof']; th=p0['thick'][np.isfinite(p0['thick'])]
    tmax=2*np.percentile(th,99.5)*np.sqrt(3.15)/C*1e6*1.18+2.0
    keys=sorted(M[cid]['passes'],key=lambda k:-M[cid]['passes'][k]['agl'])
    n=2*len(keys)
    fig=plt.figure(figsize=(15,2.0+2.3*n+2.0))
    gs=GridSpec(n+2,1,figure=fig,height_ratios=[1.2]+[1]*n+[1.15],hspace=0.44)
    axp=fig.add_subplot(gs[0])
    axp.plot(p0['s'],p0['srf'],lw=1.1,color='C0',label='ice surface (radar pick)')
    axp.plot(p0['s'],p0['bed'],lw=1.1,color='C3',label='bed (OPS "bottom" pick)')
    axp.set_xlim(0,p0['s'][-1]); axp.grid(alpha=.3); axp.legend(fontsize=7,ncol=2,loc='lower right')
    axp.set_ylabel("elev (m, WGS84)",fontsize=8); axp.tick_params(labelsize=8)
    axp.set_title(f"Candidate {cid} — high-altitude passes: CSARP_standard vs CSARP_mvdr",fontsize=12.5)
    ybed=2*p0['thick']*np.sqrt(3.15)/C*1e6
    r=1
    for k in keys:
        for lbl,src in (("CSARP_standard",S),("CSARP_mvdr",M)):
            p=src[cid]['passes'][k]; ax=fig.add_subplot(gs[r]); r+=1
            dB,g=img(p,tmax)
            vmax=np.nanpercentile(dB,99.93); vmin=np.nanpercentile(dB,45)
            ax.imshow(dB.T,aspect='auto',origin='upper',cmap='gray_r',
                      extent=[p['s'][0]/1e3,p['s'][-1]/1e3,g[-1],g[0]],
                      vmin=vmin,vmax=vmax,interpolation='nearest')
            fids=p['fids']; ch=np.flatnonzero(fids[1:]!=fids[:-1]); ed=np.r_[0,ch+1,len(fids)]
            for a,b in zip(ed[:-1],ed[1:]):
                if b-a<200: continue
                ax.axvline(p['s'][a]/1e3,color='C0',lw=0.9,ls=':')
                ax.text((p['s'][a]+p['s'][b-1])/2e3,g[0]+0.085*(g[-1]-g[0]),fids[a],
                        fontsize=8.5,ha='center',color='C0',weight='bold')
            ax.plot(p0['s'],ybed,lw=0.7,color='C3',alpha=.8)
            ax.set_xlim(0,p0['s'][-1]); ax.set_ylabel("t − t_surf (µs)",fontsize=8); ax.tick_params(labelsize=8)
            mv=lbl.endswith('mvdr')
            ax.set_title(f"{k.split('__')[0]}  {k.split('__')[1]}   |   AGL {p['agl']:.0f} m   |   {lbl}",
                         fontsize=9.5,loc='left',color=('C0' if mv else 'k'),
                         fontweight=('bold' if mv else 'normal'))
    axc=fig.add_subplot(gs[-1])
    cols={keys[0]:'#1a5e8a'}
    if len(keys)>1: cols[keys[1]]='#8a3b1a'
    for k in keys:
        for lbl,src,ls in (("standard",S,'--'),("mvdr",M,'-')):
            s,c=contrast(src[cid]['passes'][k],p0); gg,cc=smooth(s,c)
            axc.plot(gg,cc,ls,lw=1.5 if ls=='-' else 1.1,color=cols[k],alpha=1 if ls=='-' else .65,
                     label=f"{k.split('__')[1]} · {lbl}")
    axc.axhline(0,color='0.5',lw=.8); axc.axhline(6,color='C2',lw=.8,ls=':')
    axc.set_xlim(0,p0['s'][-1]); axc.grid(alpha=.3); axc.legend(fontsize=7.5,ncol=2)
    axc.set_ylabel("bed / clutter (dB)",fontsize=8); axc.tick_params(labelsize=8)
    axc.set_xlabel("distance along line (km)")
    axc.set_title("bed peak over the median power 1.5–5 µs above it, 1 km running mean "
                  "(dotted = 6 dB); gaps = no OPS bed pick",fontsize=8.5,loc='left',color='0.35')
    fig.savefig(f"mvdr_{cid}.png",dpi=125,bbox_inches='tight'); plt.close(fig); print("wrote",cid,flush=True)
