"""10 km grids of noise-corrected platelet roughness r, slope, covariates per ice sheet.
Reads outputs/cache/atm2/platelets_qc.parquet; writes outputs/atm_regional/grid_<hemi>.nc,
grid_<hemi>_r_med.tif and map PNGs. Cells need >= N_CELL centre platelets.
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import rasterio
from rasterio.transform import from_origin
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "cache" / "atm2"
OUT = ROOT / "outputs" / "atm_regional"
RES, N_CELL = 10e3, 30
CRS = {"gl": "EPSG:3413", "aa": "EPSG:3031"}
EXT = {"gl": (-1000e3, -3500e3, 1000e3, -500e3), "aa": (-2900e3, -2600e3, 2900e3, 2600e3)}


def grid(df, hemi):
    x0, y0, x1, y1 = EXT[hemi]
    nx, ny = int((x1 - x0) / RES), int((y1 - y0) / RES)
    ci = np.floor((df.x - x0) / RES).astype(int); ri = np.floor((df.y - y0) / RES).astype(int)
    ok = (ci >= 0) & (ci < nx) & (ri >= 0) & (ri < ny)
    df = df[ok].assign(cell=(ri[ok] * nx + ci[ok]).values)
    c = df[df.centre]; o = df[~df.centre]
    g = c.groupby("cell")
    lr = np.log10(np.maximum(c.r_cm, 0.3))
    agg = pd.DataFrame({
        "n": g.size(), "n_days": g.date.nunique(), "n_years": g.year.nunique(),
        "r_med": g.r_cm.median(), "r_p10": g.r_cm.quantile(0.10), "r_p90": g.r_cm.quantile(0.90),
        "rms_med": g.rms_cm.median(), "floor_med": g.floor_cm.median(),
        "logr_mean": lr.groupby(c.cell).mean(), "logr_std": lr.groupby(c.cell).std(),
        "at_floor": g.at_floor.mean(), "h": g.h.median(), "slope": g.slope.median(),
        "slope_p90": g.slope.quantile(0.90), "dist_km": g.dist_km.median(), "shelf": g.shelf.mean(),
        "lat": g.lat.median(), "lon": g.lon.median(), "xm": g.x.median(), "ym": g.y.median(), "month": g.date.agg(lambda s: int(s.dt.month.mode()[0])),
        "doy": g.date.agg(lambda s: float(s.dt.dayofyear.median())),
        "years": g.year.agg(lambda s: ",".join(str(v) for v in sorted(s.unique()))),
    })
    off = o.groupby("cell").r_cm.median().rename("r_off")
    agg = agg.join(off); agg["aniso_off_centre"] = agg.r_off / agg.r_med
    agg = agg[agg.n >= N_CELL]
    agg["row"] = agg.index // nx; agg["col"] = agg.index % nx
    agg["xc"] = x0 + (agg.col + 0.5) * RES; agg["yc"] = y0 + (agg.row + 0.5) * RES
    return agg, (nx, ny)


def to_xr(agg, hemi, shape):
    nx, ny = shape; x0, y0, x1, y1 = EXT[hemi]
    xs = x0 + (np.arange(nx) + 0.5) * RES; ys = y0 + (np.arange(ny) + 0.5) * RES
    ds = xr.Dataset(coords={"y": ys, "x": xs})
    for v in [c for c in agg.columns if c not in ("years", "row", "col", "xc", "yc")]:
        a = np.full((ny, nx), np.nan, np.float32); a[agg.row, agg.col] = agg[v].values.astype(np.float32); ds[v] = (("y", "x"), a)
    ds.attrs.update(crs=CRS[hemi], res_m=RES, note="ILATM2 centre (track 0) platelets, noise-corrected r_cm per flight-day floor")
    return ds


def maps(ds, agg, hemi):
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    proj = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70) if hemi == "gl" else ccrs.SouthPolarStereo(true_scale_latitude=-71)
    panels = [("r_med", "noise-corrected roughness r, median (cm)", LogNorm(1, 100), "viridis"),
              ("r_p90", "r p90 (cm)", LogNorm(1, 100), "viridis"),
              ("at_floor", "fraction of platelets within 3 dB of floor", None, "magma"),
              ("slope", "platelet (~80 m) slope, median (m/m)", LogNorm(3e-4, 0.1), "cividis"),
              ("n_years", "years with coverage", None, "plasma"),
              ("aniso_off_centre", "off-centre / centre r ratio", None, "coolwarm")]
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={"projection": proj})
    for ax, (v, title, norm, cmap) in zip(axs.ravel(), panels):
        ax.set_extent(EXT[hemi][0:4:2] + EXT[hemi][1:4:2], crs=proj) if False else None
        ax.set_extent([EXT[hemi][0], EXT[hemi][2], EXT[hemi][1], EXT[hemi][3]], crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.4, color="0.4")
        kw = dict(norm=norm) if norm else (dict(vmin=0.5, vmax=1.5) if v == "aniso_off_centre" else {})
        sc = ax.scatter(agg.xc, agg.yc, c=agg[v], s=3, cmap=cmap, transform=proj, **kw)
        plt.colorbar(sc, ax=ax, shrink=0.6); ax.set_title(title)
    fig.suptitle(f"ILATM2 10 km screen, {'Greenland' if hemi == 'gl' else 'Antarctica'}: {len(agg)} cells")
    fig.tight_layout(); fig.savefig(OUT / f"map_{hemi}.png", dpi=130); plt.close(fig)


def main():
    df = pd.read_parquet(CACHE / "platelets_qc.parquet")
    for hemi in ("gl", "aa"):
        d = df[df.hemi == hemi]
        if d.empty: continue
        agg, shape = grid(d, hemi)
        agg.to_csv(OUT / f"grid_{hemi}.csv", index_label="cell")
        ds = to_xr(agg, hemi, shape); ds.to_netcdf(OUT / f"grid_{hemi}.nc", engine="h5netcdf")
        x0, y0, x1, y1 = EXT[hemi]
        with rasterio.open(OUT / f"grid_{hemi}_r_med.tif", "w", driver="GTiff", height=shape[1], width=shape[0], count=1,
                           dtype="float32", crs=CRS[hemi], transform=from_origin(x0, y1, RES, RES), nodata=np.nan) as dst:
            dst.write(ds.r_med.values[::-1], 1)
        maps(ds, agg, hemi)
        print(hemi, "cells", len(agg), "r_med median", round(agg.r_med.median(), 2), "at_floor mean", round(agg.at_floor.mean(), 2))


if __name__ == "__main__":
    main()
