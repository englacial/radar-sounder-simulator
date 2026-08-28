"""1 km ice masks + distance-to-margin rasters from Natural Earth 10m polygons.
Greenland (EPSG:3413): ice = glaciated_areas (ice sheet + peripheral ice); dist = to nearest
non-ice cell (coast or ice margin). Antarctica (EPSG:3031): grounded = land, shelf =
ice-shelf polys; dist = grounded ice to nearest non-grounded cell (grounding line / coast).
Writes outputs/cache/atm2/mask_<hemi>.npz (ice, shelf, dist_km, transform).
"""
import numpy as np, geopandas as gpd
from pathlib import Path
from rasterio import features
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
import cartopy.io.shapereader as shpreader

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "cache" / "atm2"
RES = 1000.0
GRIDS = {"gl": ("EPSG:3413", (-1000e3, -3500e3, 1000e3, -500e3)),      # x0,y0,x1,y1
         "aa": ("EPSG:3031", (-2900e3, -2600e3, 2900e3, 2600e3))}

def ne(name):
    return gpd.read_file(shpreader.natural_earth(resolution="10m", category="physical", name=name))

def rasterize(gdf, crs, bounds):
    x0, y0, x1, y1 = bounds
    W, H = int((x1 - x0) / RES), int((y1 - y0) / RES)
    tr = from_origin(x0, y1, RES, RES)
    g = gdf.to_crs(crs)
    g = g[g.is_valid | g.buffer(0).is_valid]
    geoms = [gm.buffer(0) for gm in g.geometry if gm is not None and not gm.is_empty]
    arr = features.rasterize(((gm, 1) for gm in geoms), out_shape=(H, W), transform=tr, fill=0, dtype="uint8")
    return arr.astype(bool), tr

for hemi, (crs, bounds) in GRIDS.items():
    if hemi == "gl":
        glac = ne("glaciated_areas"); glac = glac.cx[-75:-10, 58:84]
        ice, tr = rasterize(glac, crs, bounds); shelf = np.zeros_like(ice)
        grounded = ice
    else:
        land = ne("land"); land = land.cx[-180:180, -90:-60]
        shelfp = ne("antarctic_ice_shelves_polys")
        grounded, tr = rasterize(land, crs, bounds)
        shelf, _ = rasterize(shelfp, crs, bounds)
        shelf &= ~grounded
        ice = grounded | shelf
    dist = distance_transform_edt(grounded) * RES / 1e3   # km to nearest non-grounded cell
    np.savez_compressed(CACHE / f"mask_{hemi}.npz", ice=ice, shelf=shelf, grounded=grounded, dist_km=dist.astype(np.float32),
                        transform=np.array(tr.to_gdal()), crs=crs)
    print(hemi, ice.shape, "ice km2", ice.sum(), "shelf", shelf.sum(), "max dist", dist.max())
