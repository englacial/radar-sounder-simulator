"""Measured vs bed-construction comparison radargrams, getz pilot.

Rows: passes. Cols: measured | BedMachine | DEMOGORGN | picked (baseline) |
picked + cross-track decorr. One scalar normalization per panel (median
surface peak), time below surface, shared color scale.

    uv run python claude_notes/bed_comparison_fig.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "tools")
import run_basal_clutter as rbc  # noqa: E402

rbc.activate_line("antarctica_getz")

PASSES = ["real_low", "real_10km"]
CASES = [("spec split, no rough", "pilot_dgnspec"),
         ("spec + sigma 0.05 m", "pilot_dgns05"),
         ("spec + sigma 0.10 m", "pilot_dgns10"),
         ("spec + sigma 0.22 m", "pilot_dgnboth")]
Y_US = (-1.0, 13.5)


def spk_norm(P, twtt, t_surf):
    pk = np.empty(P.shape[0])
    for i in range(P.shape[0]):
        m = np.abs(twtt - t_surf[i]) < 0.4e-6
        pk[i] = P[i, m].max() if m.any() else np.nan
    return float(np.nanmedian(pk))


def panel(ax, P, twtt, t_surf, grid, title=None):
    norm = spk_norm(P, twtt, t_surf)
    rel = (twtt[None, :] - t_surf[:, None]) * 1e6
    img = np.full((P.shape[0], grid.size), np.nan)
    for i in range(P.shape[0]):
        img[i] = np.interp(grid, rel[i], P[i], left=np.nan, right=np.nan)
    db = 10 * np.log10(np.maximum(img, 1e-300) / norm)
    im = ax.imshow(db.T, aspect="auto", origin="upper", cmap="viridis",
                   vmin=-100, vmax=0,
                   extent=[0, db.shape[0], grid[-1], grid[0]])
    if title:
        ax.set_title(title, fontsize=11)
    return im


def measured(key):
    import xarray as xr
    parts = rbc.PASSES[key]["pilot"]
    season = rbc.PASSES[key].get("season", rbc.SEASON)
    fs = []
    for fid, (a, b) in parts:
        f = rbc.load_frame(season, fid)
        a = a or 0
        b = f.sizes["slow_time"] if b is None else b
        fs.append(f.isel(slow_time=slice(a, b)))
    f = fs[0] if len(fs) == 1 else xr.concat(fs, dim="slow_time",
                                             combine_attrs="override")
    if rbc.PASSES[key]["rev"]:      # onto the common line orientation,
        f = f.isel(slow_time=slice(None, None, -1))   # like the sim chain
    return (np.asarray(f.Data.values, np.float64),
            np.asarray(f.twtt.values, np.float64),
            np.asarray(f.Surface.values, np.float64))


fig, axes = plt.subplots(len(PASSES), len(CASES) + 1,
                         figsize=(4.0 * (len(CASES) + 1), 4.8 * len(PASSES)),
                         sharey=True, constrained_layout=True)
for r, key in enumerate(PASSES):
    Pm, twm, sm = measured(key)
    dt_us = np.median(np.diff(twm)) * 1e6
    grid = np.arange(Y_US[0], Y_US[1], dt_us)
    panel(axes[r, 0], Pm, twm, sm,
          grid, title=("MEASURED" if r == 0 else None))
    axes[r, 0].set_ylabel(f"{key}\ntime below surface (us)")
    for c, (title, case) in enumerate(CASES, start=1):
        hits = glob.glob(f"outputs/antarctica_getz/{case}/proc_cache/"
                         f"{key}_pilot_*_proc.npz")
        if not hits:
            axes[r, c].text(0.5, 0.5, "missing", ha="center",
                            transform=axes[r, c].transAxes)
            continue
        z = np.load(hits[0])
        P = np.abs(z["Fs"] + z["Fb"]) ** 2
        im = panel(axes[r, c], P, z["twtt"], z["nadir"][:, 0], grid,
                   title=(title if r == 0 else None))
for ax in axes[-1]:
    ax.set_xlabel("trace")
fig.colorbar(im, ax=axes, shrink=0.6, label="dB rel median surface peak")
fig.suptitle("getz pilot: bed-construction comparison, matched processing")
out = Path("claude_notes/bed_comparison_radargrams.png")
fig.savefig(out, dpi=110)
print("wrote", out)
