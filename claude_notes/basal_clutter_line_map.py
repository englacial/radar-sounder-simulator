"""Minimal location map for the basal-clutter study line (EPSG:3031).

Anchor track 20161105_05_005-007 (2016_Antarctica_DC8): full 148.45 km muted,
0-100 km highlighted, grounding-line crossing at s = 69.7 km ticked. Coastline
context from the BedMachine Antarctica v3 mask (ocean shaded, GL drawn).

    uv run python claude_notes/basal_clutter_line_map.py
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from soundersim.opr import BEDMACHINE, CACHE_DIR, load_frame  # noqa: E402

SEASON = "2016_Antarctica_DC8"
FRAMES = ("20161105_05_005", "20161105_05_006", "20161105_05_007")
S_HI_KM = 100.0          # highlighted (radargram) section
GL_KM = 69.7             # grounding-line crossing on the anchor axis
PAD_M = 75_000.0
OUT = ROOT / "outputs/basal_clutter/full_line/line_map.png"
MIRROR = ROOT / "outputs/verification/basal_clutter_full_line/line_map.png"


def track():
    """Anchor track in EPSG:3031 with the run_basal_clutter along-track axis
    (s = 0 at trace 0 of frame _005)."""
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    xs, ys = [], []
    for fid in FRAMES:
        fr = load_frame(SEASON, fid)
        lat = np.asarray(fr.Latitude.values, np.float64)
        lon = np.asarray(fr.Longitude.values, np.float64)
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        x, y = tr.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    x, y = np.concatenate(xs), np.concatenate(ys)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return x, y, s


def bedmachine_mask(proj_bounds, cache_dir=None):
    """BedMachine Antarctica v3 mask on its native 500 m grid over an
    EPSG:3031 bbox. Same NSIDC/earthaccess path as opr.fetch_bedmachine_window
    (which keeps bed+geoid only); cached as its own GeoTIFF."""
    prod = BEDMACHINE["antarctic"]
    cache_dir = Path(cache_dir or CACHE_DIR)
    key = hashlib.sha256(json.dumps(
        [prod["url"], "mask",
         [round(b, 1) for b in proj_bounds]]).encode()).hexdigest()[:12]
    tif = cache_dir / f"bedmachine_mask_antarctic_{key}.tif"
    if not tif.exists():
        import earthaccess

        earthaccess.login(strategy="netrc")
        fs = earthaccess.get_fsspec_https_session()
        with fs.open(prod["url"], block_size=4 * 2**20,
                     cache_type="blockcache") as f, \
                xr.open_dataset(f, engine="h5netcdf") as src:
            x, y = src["x"].values, src["y"].values
            step = float(prod["posting"])
            x0, y0, x1, y1 = proj_bounds
            ci = np.where((x >= x0 - step) & (x <= x1 + step))[0]
            ri = np.where((y >= y0 - step) & (y <= y1 + step))[0]
            rs, cs = slice(ri[0], ri[-1] + 1), slice(ci[0], ci[-1] + 1)
            mask = src["mask"][rs, cs].values.astype(np.uint8)
            transform = rasterio.transform.from_origin(
                x[ci[0]] - step / 2, y[ri[0]] + step / 2, step, step)
            version = str(src.attrs.get("version", ""))
        cache_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(tif, "w", driver="GTiff", height=mask.shape[0],
                           width=mask.shape[1], count=1, dtype="uint8",
                           crs=prod["crs"], transform=transform) as dst:
            dst.write(mask, 1)
        tif.with_suffix(".json").write_text(json.dumps({
            "product": prod["product"], "version": version, "url": prod["url"],
            "crs": prod["crs"], "posting_m": prod["posting"],
            "proj_bounds": list(proj_bounds), "band": "mask",
            "legend": "0 ocean, 1 ice-free land, 2 grounded ice, "
                      "3 floating ice, 4 lake Vostok",
        }, indent=1) + "\n")
    with rasterio.open(tif) as src:
        return src.read(1), src.transform, tif


def scale_bar(ax, length_m, x0, y0, label):
    ax.plot([x0, x0 + length_m], [y0, y0], color="#111111", lw=1.6,
            solid_capstyle="butt", zorder=6)
    ax.text(x0 + length_m / 2, y0 + 1_800, label, ha="center", va="bottom",
            fontsize=7.5, color="#111111", zorder=6)


def main():
    x, y, s = track()
    print(f"track: {len(s)} traces, {s[-1] / 1e3:.2f} km")

    ext = (x.min() - PAD_M, y.min() - PAD_M, x.max() + PAD_M, y.max() + PAD_M)
    mask, tf, tif = bedmachine_mask(ext)
    print(f"mask {mask.shape} from {tif.name}, "
          f"classes {sorted(np.unique(mask).tolist())}")

    h, w = mask.shape
    x0, y1 = tf * (0, 0)
    x1, y0 = tf * (w, h)
    imext = (x0, x1, y0, y1)

    # 0 ocean, 1 ice-free land, 2 grounded ice, 3 floating ice
    cmap = ListedColormap(["#c3d8ee", "#e3e0da", "#fdfdfe", "#e7edf4"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=220)
    ax.imshow(mask, extent=imext, origin="upper", cmap=cmap, norm=norm,
              interpolation="nearest", zorder=0)
    # grounding line: grounded ice / ice-free land vs floating ice / ocean
    gx = np.linspace(x0 + tf.a / 2, x1 - tf.a / 2, w)
    gy = np.linspace(y1 + tf.e / 2, y0 - tf.e / 2, h)
    ax.contour(gx, gy, ((mask == 1) | (mask == 2)).astype(float), [0.5],
               colors="#6b7280", linewidths=0.9, zorder=1)

    ax.plot(x, y, color="#9aa3ad", lw=0.9, zorder=2,
            label=f"full line ({s[-1] / 1e3:.1f} km)")
    hi = s <= S_HI_KM * 1e3
    ax.plot(x[hi], y[hi], color="#d1462f", lw=2.2, solid_capstyle="round",
            zorder=3, label=f"0-{S_HI_KM:.0f} km (radargrams)")

    ax.plot([], [], color="#6b7280", lw=0.9, label="grounding line")
    i0 = int(np.argmin(np.abs(s - GL_KM * 1e3)))
    ax.plot(x[i0], y[i0], marker="o", ms=4.5, mfc="white", mec="#111111",
            mew=1.1, zorder=5, ls="none",
            label=f"GL crossing (s = {GL_KM:.1f} km)")
    ax.plot(x[0], y[0], marker="o", ms=3.2, color="#111111", zorder=5,
            ls="none")
    ax.annotate("s = 0", (x[0], y[0]), textcoords="offset points",
                xytext=(6, 6), fontsize=7.5, color="#111111")

    ax.set_xlim(ext[0], ext[2])
    ax.set_ylim(ext[1], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#c8ccd2")
        sp.set_linewidth(0.8)

    span = ext[2] - ext[0]
    scale_bar(ax, 50_000.0, ext[0] + 0.06 * span,
              ext[1] + 0.055 * (ext[3] - ext[1]), "50 km")
    ax.legend(loc="upper right", frameon=False, fontsize=7.5,
              handlelength=1.6, borderpad=0.2, labelspacing=0.45)
    fig.tight_layout(pad=0.4)

    for p in (OUT, MIRROR):
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        print(f"wrote {p}")
    print("extent (EPSG:3031 m): "
          + ", ".join(f"{v:.0f}" for v in ext))


if __name__ == "__main__":
    main()
