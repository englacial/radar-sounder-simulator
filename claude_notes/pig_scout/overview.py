import numpy as np, pickle, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.transform import Affine
from pyproj import Transformer
tr=Transformer.from_crs("EPSG:4326","EPSG:3031",always_xy=True)
A=pickle.load(open("assembled.pkl","rb"))
bed=np.load("bed.npy"); bmx=np.load("bm_extra.npz"); mask=bmx['mask']; surf=bmx['surface']
tfl,crs,mc=json.load(open("bedtf.json")); tf=Affine(*tfl)
H,W=bed.shape; ext=[tf.c,tf.c+W*tf.a,tf.f+H*tf.e,tf.f]
X=tf.c+(np.arange(W)+.5)*500; Y=tf.f-(np.arange(H)+.5)*500
fig,axs=plt.subplots(1,2,figsize=(19,9.2))
for ax,zoom in zip(axs,[False,True]):
    im=ax.imshow(bed,extent=ext,origin="upper",cmap="terrain",vmin=-1800,vmax=1200)
    ax.contour(X,Y,(mask==3).astype(float),[0.5],colors='k',linewidths=1.0)   # floating edge
    ax.contour(X,Y,(mask==0).astype(float),[0.5],colors='navy',linewidths=1.2) # ocean
    cs=ax.contour(X,Y,surf,np.arange(0,2600,200),colors='0.35',linewidths=0.45)
    cols=dict(A='red',B='darkorange',C='magenta',D='blue',E='lime')
    for cid,rec in A.items():
        sx,sy=rec['seg']['x'],rec['seg']['y']
        ax.plot(sx,sy,lw=3.0,color=cols[cid],solid_capstyle='butt')
        ax.plot(sx[0],sy[0],'o',ms=7,color=cols[cid],mec='k')
        ax.annotate(cid,(sx[0],sy[0]),xytext=(6,6),textcoords='offset points',
                    fontsize=15,weight='bold',color=cols[cid],
                    path_effects=None)
    for lon,lat,nm in [(-101.6,-75.05,'Pine Island Ice Shelf'),(-100.3,-75.15,'PIG grounding zone'),
                       (-98.5,-75.5,'upper PIG trunk'),(-106.0,-75.2,'Thwaites'),
                       (-96.0,-76.4,'PIG catchment (interior)')]:
        x,y=tr.transform(lon,lat); ax.plot(x,y,'k+',ms=9)
        ax.annotate(nm,(x,y),xytext=(5,4),textcoords='offset points',fontsize=9)
    ax.set_aspect('equal')
    if zoom: ax.set_xlim(-1.66e6,-1.53e6); ax.set_ylim(-3.4e5,-1.3e5); ax.set_title("zoom on the candidate corridor")
    else:
        ax.set_xlim(ext[0],ext[1]); ax.set_ylim(ext[2],ext[3])
        ax.set_title("Pine Island sector — BedMachine v3 bed (colour), surface contours 200 m,\n"
                     "black = grounding line, blue = calving front")
        fig.colorbar(im,ax=ax,shrink=0.6,label="bed elevation (m, WGS84 ellipsoid)")
    ax.set_xlabel("EPSG:3031 x (m)"); ax.set_ylabel("y (m)")
fig.savefig("overview.png",dpi=95,bbox_inches='tight'); print("ok")
