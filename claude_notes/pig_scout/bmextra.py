import numpy as np, xarray as xr, earthaccess, json
from soundersim.opr import BEDMACHINE, _padded_proj_bounds
prod=BEDMACHINE["antarctic"]
pb=_padded_proj_bounds((-106,-77.5,-93,-74.0), prod["crs"], 20e3)
earthaccess.login(strategy="netrc")
fs=earthaccess.get_fsspec_https_session()
with fs.open(prod["url"], block_size=4*2**20, cache_type="blockcache") as f, \
     xr.open_dataset(f, engine="h5netcdf") as src:
    x,y=src["x"].values, src["y"].values; step=500.
    x0,y0,x1,y1=pb
    ci=np.where((x>=x0-step)&(x<=x1+step))[0]; ri=np.where((y>=y0-step)&(y<=y1+step))[0]
    rs,cs=slice(ri[0],ri[-1]+1), slice(ci[0],ci[-1]+1)
    surf=src["surface"][rs,cs].values.astype(np.float32)
    mask=src["mask"][rs,cs].values.astype(np.int16)
    thk=src["thickness"][rs,cs].values.astype(np.float32)
np.savez("bm_extra.npz", surface=surf, mask=mask, thickness=thk)
print(surf.shape, np.unique(mask))
