"""Real OPR/CReSIS frames + PGC DEM windows as soundersim scenes.

Frames come from Open Polar Radar via xopr (CSARP_standard: Data with dims
slow_time x twtt, twtt in seconds since transmit). Surface DEMs are the PGC
REMA v2.0 (Antarctic) / ArcticDEM v4.1 (Arctic) 32 m mosaics, discovered via
the PGC STAC API and window-read as COGs over HTTP.

Vertical datum: per PGC's product documentation, both mosaic products are
"referenced to the WGS84 ellipsoid" (https://www.pgc.umn.edu/data/rema/,
https://www.pgc.umn.edu/data/arcticdem/), matching CReSIS Elevation (m above
WGS84 ellipsoid). No geoid offset is applied.

Everything network-touching is cached under outputs/cache/ so reruns are
offline: frames as NetCDF, DEM windows as GeoTIFF + JSON sidecar.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from pyproj import Transformer
from scipy import ndimage

from .synthetic import SyntheticScene

PGC_STAC = "https://stac.pgc.umn.edu/api/v1/"
DEM_PRODUCTS = {
    "antarctic": ("rema-mosaics-v2.0-32m", "EPSG:3031"),
    "arctic": ("arcticdem-mosaics-v4.1-32m", "EPSG:3413"),
}
# BedMachine bed topography (NSIDC Earthdata Cloud, authenticated via ~/.netrc).
# Both products reference bed/surface to the EIGEN-6C4 GEOID (verified in the
# file metadata: variable ``geoid`` is "EIGEN-6C4 Geoid - WGS84 Ellipsoid
# difference", attr geoid="eigen-6c4 (Forste et al 2014)"); ellipsoidal height
# = bed + geoid. Native CRSs match the PGC DEM CRSs above.
_NSIDC = "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected"
BEDMACHINE = {
    "arctic": {
        "url": f"{_NSIDC}/ICEBRIDGE/IDBMG4/5/1993/01/01/BedMachineGreenland-v5.nc",
        "product": "BedMachine Greenland v5 (IDBMG4)", "crs": "EPSG:3413",
        "posting": 150.0,
    },
    "antarctic": {
        "url": f"{_NSIDC}/MEASURES/NSIDC-0756/3/1970/01/01/BedMachineAntarctica-v3.nc",
        "product": "BedMachine Antarctica v3 (NSIDC-0756)", "crs": "EPSG:3031",
        "posting": 500.0,
    },
}
DATUM_NOTE = ("heights in m above the WGS84 ellipsoid per PGC documentation; "
              "matches CReSIS Elevation, no geoid offset applied")
CACHE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "cache"
NODATA = -9999.0

FRAME_VARS = ["Data", "Latitude", "Longitude", "Elevation", "Surface",
              "Roll", "Pitch", "Heading"]


def load_frame(season, frame_id, data_product="CSARP_standard", cache_dir=None):
    """Load an OPR frame (e.g. season "2017_Antarctica_P3", frame_id
    "20171121_03_005") via xopr, cached as NetCDF under outputs/cache/."""
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache = cache_dir / f"frame_{season}_{frame_id}_{data_product}.nc"
    if cache.exists():
        return xr.open_dataset(cache, engine="h5netcdf")

    import xopr  # import on cache miss only (syncs its catalog on connect)

    date, seg, frame_num = frame_id.split("_")
    conn = xopr.OPRConnection(cache_dir=str(cache_dir / "xopr"))
    items = conn.query_frames(collections=[season],
                              segment_paths=[f"{date}_{seg}"],
                              properties={"opr:frame": int(frame_num)})
    if items is None or len(items) == 0:
        raise LookupError(f"frame {frame_id} not found in {season}")
    raw = conn.load_frame(items.iloc[0], data_product=data_product)

    ds = raw[[v for v in FRAME_VARS if v in raw]].transpose("slow_time", "twtt")
    ds = ds.reset_coords(drop=True)
    ds.attrs = {k: v for k, v in raw.attrs.items()
                if isinstance(v, (str, int, float, np.integer, np.floating))}
    ds.attrs.update(season=season, frame_id=frame_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(cache, engine="h5netcdf")
    return ds


def _padded_proj_bounds(bounds, crs, pad_m):
    """Densified lon/lat bbox -> padded (xmin, ymin, xmax, ymax) in ``crs``."""
    lon0, lat0, lon1, lat1 = bounds
    t = np.linspace(0.0, 1.0, 64)
    lons = np.concatenate([lon0 + (lon1 - lon0) * t, np.full_like(t, lon1),
                           lon1 + (lon0 - lon1) * t, np.full_like(t, lon0)])
    lats = np.concatenate([np.full_like(t, lat0), lat0 + (lat1 - lat0) * t,
                           np.full_like(t, lat1), lat1 + (lat0 - lat1) * t])
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lons, lats)
    return (x.min() - pad_m, y.min() - pad_m, x.max() + pad_m, y.max() + pad_m)


def fetch_dem_window(bounds, region, pad_m=0.0, cache_dir=None):
    """Windowed 32 m mosaic DEM covering a lon/lat bbox padded by pad_m meters.

    bounds: (lon_min, lat_min, lon_max, lat_max), WGS84. region: "antarctic"
    (REMA v2.0) or "arctic" (ArcticDEM v4.1). Returns (dem, transform, crs)
    with nodata as NaN; cached as GeoTIFF + JSON sidecar under outputs/cache/.
    """
    collection, crs = DEM_PRODUCTS[region]
    proj_bounds = _padded_proj_bounds(bounds, crs, pad_m)
    key = hashlib.sha256(json.dumps(
        [collection, [round(b, 1) for b in proj_bounds]]).encode()).hexdigest()[:12]
    cache_dir = Path(cache_dir or CACHE_DIR)
    tif = cache_dir / f"dem_{collection}_{key}.tif"

    if not tif.exists():
        # STAC search polygon: the padded projected bbox, back in lon/lat.
        inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        x0, y0, x1, y1 = proj_bounds
        t = np.linspace(0.0, 1.0, 64)
        xs = np.concatenate([x0 + (x1 - x0) * t, np.full_like(t, x1),
                             x1 + (x0 - x1) * t, np.full_like(t, x0)])
        ys = np.concatenate([np.full_like(t, y0), y0 + (y1 - y0) * t,
                             np.full_like(t, y1), y1 + (y0 - y1) * t])
        lons, lats = inv.transform(xs, ys)
        poly = {"type": "Polygon",
                "coordinates": [list(zip(lons.tolist(), lats.tolist()))]}

        from pystac_client import Client
        from rasterio.merge import merge

        search = Client.open(PGC_STAC).search(collections=[collection],
                                              intersects=poly)
        stac_items = list(search.items())
        if not stac_items:
            raise LookupError(f"no {collection} tiles intersect {bounds}")
        srcs = [rasterio.open(it.assets["dem"].href) for it in stac_items]
        arr, transform = merge(srcs, bounds=proj_bounds, res=32.0, nodata=NODATA)
        for s in srcs:
            s.close()
        dem = arr[0].astype(np.float32)

        cache_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(tif, "w", driver="GTiff", height=dem.shape[0],
                           width=dem.shape[1], count=1, dtype="float32",
                           crs=crs, transform=transform, nodata=NODATA) as dst:
            dst.write(dem, 1)
        tif.with_suffix(".json").write_text(json.dumps({
            "collection": collection, "stac_api": PGC_STAC, "crs": crs,
            "bounds_lonlat": list(bounds), "pad_m": pad_m,
            "proj_bounds": list(proj_bounds), "vertical_datum": DATUM_NOTE,
            "items": [it.id for it in stac_items],
        }, indent=1) + "\n")

    with rasterio.open(tif) as src:
        dem = src.read(1)
        transform, crs = src.transform, str(src.crs)
    dem = np.where(dem == NODATA, np.nan, dem).astype(np.float32)
    return dem, transform, crs


def fill_nodata_nearest(dem):
    """Fill NaN holes with the nearest valid value; returns (filled, fraction)."""
    bad = ~np.isfinite(dem)
    frac = float(bad.mean())
    if frac:
        idx = ndimage.distance_transform_edt(bad, return_distances=False,
                                             return_indices=True)
        dem = dem[tuple(idx)]
    return dem, frac


def frame_scene(frame, *, n_traces=150, ct_dist=4000.0, region=None,
                margin_m=500.0, cache_dir=None):
    """Build a SyntheticScene container from a real frame + real DEM window.

    Subsamples ~n_traces evenly spaced traces; the DEM window covers the
    subsampled track bounds padded by ct_dist + margin_m. Returns
    (scene, info) where info records the trace indices and DEM fill fraction.
    """
    idx = np.unique(np.round(np.linspace(
        0, frame.sizes["slow_time"] - 1, n_traces)).astype(int))
    lat = np.asarray(frame.Latitude.values[idx], np.float64)
    lon = np.asarray(frame.Longitude.values[idx], np.float64)
    lon = np.where(lon > 180.0, lon - 360.0, lon)  # unwrap to [-180, 180]
    elev = np.asarray(frame.Elevation.values[idx], np.float64)
    nav_llh = np.column_stack([lat, lon, elev])

    if region is None:
        region = "antarctic" if lat.mean() < 0 else "arctic"
    bounds = (lon.min(), lat.min(), lon.max(), lat.max())
    dem, transform, crs = fetch_dem_window(bounds, region,
                                           pad_m=ct_dist + margin_m,
                                           cache_dir=cache_dir)
    dem, fill_frac = fill_nodata_nearest(dem)

    name = f"opr_{frame.attrs.get('frame_id', 'frame')}"
    params = {"season": frame.attrs.get("season"),
              "frame_id": frame.attrs.get("frame_id"),
              "dem_product": DEM_PRODUCTS[region][0], "region": region,
              "posting": 32.0, "ct_dist": ct_dist,
              "n_traces": len(idx), "nodata_fill_fraction": fill_frac,
              "vertical_datum": DATUM_NOTE}
    # Per-trace attitude for antenna roll_source="nav" (CReSIS Roll: radians,
    # positive = right wing down); NaNs are treated as 0 downstream.
    roll = (np.asarray(frame.Roll.values[idx], np.float64)
            if "Roll" in frame else None)
    scene = SyntheticScene(name, dem, transform, crs, nav_llh, params,
                           nav_roll=roll)
    info = {"trace_idx": idx, "region": region, "fill_fraction": fill_frac}
    return scene, info


def fetch_bedmachine_window(bounds, region, pad_m=0.0, cache_dir=None):
    """Windowed BedMachine bed elevation covering a lon/lat bbox + pad_m meters.

    bounds/region as in ``fetch_dem_window``. Returns (bed, transform, crs,
    meta) at the product's NATIVE posting (150 m Greenland / 500 m Antarctica),
    with bed converted to meters above the WGS84 ELLIPSOID (bed + geoid, both
    read from the product; see BEDMACHINE datum note). ``meta`` records the
    product/version and the BedMachine mask composition of the window. Cached
    as GeoTIFF + JSON sidecar under outputs/cache/; reruns are offline.
    """
    prod = BEDMACHINE[region]
    crs = prod["crs"]
    proj_bounds = _padded_proj_bounds(bounds, crs, pad_m)
    key = hashlib.sha256(json.dumps(
        [prod["url"], [round(b, 1) for b in proj_bounds]]).encode()).hexdigest()[:12]
    cache_dir = Path(cache_dir or CACHE_DIR)
    tif = cache_dir / f"bedmachine_{region}_{key}.tif"

    if not tif.exists():
        import earthaccess

        earthaccess.login(strategy="netrc")
        fs = earthaccess.get_fsspec_https_session()
        with fs.open(prod["url"], block_size=4 * 2**20,
                     cache_type="blockcache") as f, \
                xr.open_dataset(f, engine="h5netcdf") as src:
            geoid_attrs = dict(src["geoid"].attrs)
            version = str(src.attrs.get("version", ""))
            x, y = src["x"].values, src["y"].values  # x asc, y desc
            step = float(prod["posting"])
            x0, y0, x1, y1 = proj_bounds
            ci = np.where((x >= x0 - step) & (x <= x1 + step))[0]
            ri = np.where((y >= y0 - step) & (y <= y1 + step))[0]
            if not len(ci) or not len(ri):
                raise LookupError(f"{prod['product']} does not cover {bounds}")
            rs, cs = slice(ri[0], ri[-1] + 1), slice(ci[0], ci[-1] + 1)
            bed = src["bed"][rs, cs].values.astype(np.float32)
            geoid = src["geoid"][rs, cs].values.astype(np.float32)
            mask = src["mask"][rs, cs].values
            bed = np.where(bed == NODATA, np.nan, bed) + geoid  # -> ellipsoidal
            transform = rasterio.transform.from_origin(
                x[ci[0]] - step / 2, y[ri[0]] + step / 2, step, step)

        counts = {str(k): int(v) for k, v in
                  zip(*np.unique(mask, return_counts=True))}
        cache_dir.mkdir(parents=True, exist_ok=True)
        out = np.where(np.isfinite(bed), bed, NODATA).astype(np.float32)
        with rasterio.open(tif, "w", driver="GTiff", height=bed.shape[0],
                           width=bed.shape[1], count=1, dtype="float32",
                           crs=crs, transform=transform, nodata=NODATA) as dst:
            dst.write(out, 1)
        tif.with_suffix(".json").write_text(json.dumps({
            "product": prod["product"], "version": version, "url": prod["url"],
            "crs": crs, "posting_m": step, "bounds_lonlat": list(bounds),
            "pad_m": pad_m, "proj_bounds": list(proj_bounds),
            "vertical_datum": ("bed + geoid: meters above the WGS84 ellipsoid; "
                               "product bed is geoid-referenced, geoid var: "
                               + json.dumps(geoid_attrs, default=str)),
            "mask_counts": counts,
            "mask_legend": "0 ocean, 1 ice-free land, 2 grounded ice, "
                           "3 floating ice, 4 other",
        }, indent=1) + "\n")

    with rasterio.open(tif) as src:
        bed = src.read(1)
        transform, crs = src.transform, str(src.crs)
    bed = np.where(bed == NODATA, np.nan, bed).astype(np.float32)
    meta = json.loads(tif.with_suffix(".json").read_text())
    return bed, transform, crs, meta


def resample_to_grid(src, src_transform, src_crs, shape, transform, crs):
    """Bilinearly resample a raster onto a target grid (rasterio.warp).

    Used to put the native 150/500 m BedMachine bed on a frame's 32 m surface
    DEM grid -- heavy oversampling, honest about the bed's true resolution.
    """
    from rasterio.warp import Resampling, reproject

    dst = np.full(shape, np.nan, np.float32)
    reproject(np.asarray(src, np.float32), dst, src_transform=src_transform,
              src_crs=src_crs, dst_transform=transform, dst_crs=crs,
              src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    return dst


def load_bottom_pick(frame, cache_dir=None):
    """The frame's measured Bottom (bed) pick twtt on the frame slow_time grid.

    Loads OPR layer picks via xopr (CSARP_layer files, OPS db fallback),
    selects the "bottom" layer, and interpolates its twtt onto the frame's
    slow_time (nearest-neighbor within 2 s, NaN where no pick). Cached as
    NetCDF under outputs/cache/. Returns a float64 array (n_slow_time,).
    """
    season = frame.attrs.get("season")
    frame_id = frame.attrs.get("frame_id")
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache = cache_dir / f"layers_{season}_{frame_id}_bottom.nc"
    if not cache.exists():
        import xopr

        conn = xopr.OPRConnection(cache_dir=str(cache_dir / "xopr"))
        # NetCDF round-trips attrs as numpy ints, which xopr's duckdb STAC
        # search rejects -- coerce to Python scalars.
        frame = frame.copy()
        frame.attrs = {k: (int(v) if isinstance(v, np.integer) else v)
                       for k, v in frame.attrs.items()}
        layers = conn.get_layers(frame, errors="error")
        bot_keys = [k for k in layers if k.lower().split(":")[-1] == "bottom"]
        if not bot_keys:
            raise LookupError(
                f"no bottom layer for {frame_id}; layers: {list(layers)}")
        lay = layers[bot_keys[0]]
        ds = xr.Dataset({"twtt": ("slow_time", np.asarray(
            lay["twtt"].values, np.float64))},
            coords={"slow_time": lay["slow_time"].values})
        ds.attrs.update(season=season, frame_id=frame_id, layer=bot_keys[0])
        cache_dir.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(cache, engine="h5netcdf")

    lay = xr.open_dataset(cache, engine="h5netcdf")
    t_lay = lay.slow_time.values.astype("datetime64[ns]").astype(np.int64)
    t_frm = frame.slow_time.values.astype("datetime64[ns]").astype(np.int64)
    tw = np.asarray(lay.twtt.values, np.float64)
    ok = np.isfinite(tw) & (tw > 0)
    out = np.full(len(t_frm), np.nan)
    if ok.any():
        tl = t_lay[ok]
        j = np.searchsorted(tl, t_frm).clip(1, len(tl) - 1)
        j -= (t_frm - tl[j - 1]) < (tl[j] - t_frm)
        near = np.abs(tl[j] - t_frm) < int(2e9)  # within 2 s
        out[near] = tw[ok][j[near]]
    return out
