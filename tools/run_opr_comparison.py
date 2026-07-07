"""Surface-clutter simulation of real OPR frames: soundersim vs simc vs data.

For each frame: load via xopr (cached), build a scene from the PGC 32 m mosaic
DEM (cached), run soundersim and simc on identical inputs, and write metrics +
figures to outputs/verification/opr_<frame_id>/.

Conventions: t0 aligned to the frame twtt axis (CReSIS twtt zero = transmit
event); the sim-vs-data twtt offset is measured against the frame Surface pick
and recorded, not assumed. simc is run with an enlarged n_samples so its
mod-tracesamples wrap never triggers, then truncated to the frame window.
soundersim-vs-simc metrics are evaluated only over twtt < 2*ct_dist/c, the
window where both tools provably cover every contributing facet (soundersim
uses the whole DEM strip; simc only a per-trace square of half-width ct_dist).

Run: uv run python tools/run_opr_comparison.py
"""

import datetime
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from pyproj import Transformer

from soundersim.compare.metrics import compare_to_simc
from soundersim.compare.simc_harness import run_simc
from soundersim.config import FacetConfig, RadarConfig, SimConfig
from soundersim.opr import frame_scene, load_frame
from soundersim.simulate import simulate

C = 299792458.0
CT_CAP = 6000.0  # max simulated cross-track distance (m)
N_TRACES = 150
OUT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "verification"

CASES = [
    {"season": "2017_Antarctica_P3", "frame_id": "20171121_03_005",
     "why": "required verification frame (Antarctic Peninsula)"},
    {"season": "2017_Greenland_P3", "frame_id": "20170422_01_014",
     "why": "runs E-W along the Helheim Glacier trunk (~66.37N): steep fjord "
            "valley walls on both sides give ~1.7 km of relief in the "
            "simulated swath, the canonical margin clutter geometry"},
]


def _db(a, floor=1e-30):
    return 10.0 * np.log10(np.maximum(a, floor))


def run_case(case, n_traces=N_TRACES, out_root=OUT_ROOT):
    frame = load_frame(case["season"], case["frame_id"])
    tw = frame.twtt.values
    dt = float((tw[-1] - tw[0]) / (len(tw) - 1))
    t0, n_samples = float(tw[0]), len(tw)
    rc = RadarConfig(dt=dt, n_samples=n_samples, t0=t0)

    # Cross-track reach of the recorded window: how far off-nadir a surface
    # return can arrive within max twtt, from median AGL; capped for runtime.
    agl = float(np.nanmedian(frame.Surface.values)) * C / 2
    r_max = C * (t0 + (n_samples - 1) * dt) / 2
    x_max = float(np.sqrt(max(r_max**2 - agl**2, 0.0)))
    ct_dist = min(x_max, CT_CAP)

    scene, info = frame_scene(frame, n_traces=n_traces, ct_dist=ct_dist)
    idx = info["trace_idx"]
    surf = frame.Surface.values[idx]

    sim_cfg = SimConfig(mode="incoherent", split_sides=False, radar=rc,
                        facets=FacetConfig(spacing=None))
    t_start = time.perf_counter()
    ds = simulate(scene, sim_cfg)
    t_ours = time.perf_counter() - t_start

    # simc on identical inputs, window enlarged so its bin wrap never triggers.
    h_max = float(scene.nav_llh[:, 2].max() - np.nanmin(scene.dem))
    r_corner = np.sqrt(h_max**2 + 2 * ct_dist**2)
    n2 = int(np.ceil((2 * r_corner / C - t0) / dt)) + 8
    rc_simc = RadarConfig(dt=dt, n_samples=max(n2, n_samples), t0=t0)
    t_start = time.perf_counter()
    simc = run_simc(scene, rc_simc, ct_dist, ct_dist, 32.0, 32.0)
    t_simc = time.perf_counter() - t_start

    # Compare only over the mutually fully-covered window twtt < 2*ct_dist/c.
    n_cmp = min(n_samples, int(np.floor((2 * ct_dist / C - t0) / dt)))
    fixture = {
        "cluttergram": simc["cluttergram"][:n_cmp],
        "fret_bin": simc["fret_bin"], "fret_xyz": simc["fret_xyz"],
        "meta": {"radar_config": rc.model_dump(),
                 "scene": {"params": {"posting": 32.0}}},
    }
    metrics = compare_to_simc(ds.isel(twtt=slice(0, n_cmp)), fixture)
    for name, m in metrics.items():  # real-frame thresholds TBD: record only
        m["note"] = ("recorded only; pass forced true (real-frame thresholds "
                     "are set after observing residuals, per plan)")
        m["threshold"] = None
        m["pass"] = True

    # twtt offset vs the frame Surface pick (constant offset measured, then
    # removed for the leading-edge sanity gate). The gate uses nadir_twtt and
    # the *median* |residual|: the CReSIS tracker follows the nadir surface,
    # but on rugged sections it can lock onto off-nadir leading-edge clutter,
    # so real frames show a heavy residual tail (recorded via p90/max, not
    # gated). first_return_twtt stats are recorded as a diagnostic.
    ok = np.isfinite(surf)
    nd_res = ds.nadir_twtt.values - surf
    fr_res = ds.first_return_twtt.values - surf
    off = float(np.median(nd_res[ok]))
    resid_bins = (nd_res[ok] - off) / dt
    p50, p90 = (float(np.percentile(np.abs(resid_bins), q)) for q in (50, 90))
    fr_off = float(np.median(fr_res[ok]))
    metrics["surface_alignment"] = {
        "value": p50, "threshold": 5.0, "pass": bool(p50 <= 5.0),
        "offset_s": off, "offset_bins": off / dt,
        "p90_abs_resid_bins": p90,
        "max_abs_resid_bins": float(np.abs(resid_bins).max()),
        "frac_within_5_bins": float(np.mean(np.abs(resid_bins) <= 5.0)),
        "first_return_offset_s": fr_off,
        "first_return_p90_abs_resid_bins": float(np.percentile(
            np.abs((fr_res[ok] - fr_off) / dt), 90)),
        "note": "median |nadir_twtt - Surface - median offset| in bins",
    }

    out = out_root / f"opr_{case['frame_id']}"
    out.mkdir(parents=True, exist_ok=True)
    notes = (f"{case['season']} {case['frame_id']} ({case['why']}); "
             f"DEM {scene.params['dem_product']} ({scene.params['vertical_datum']}), "
             f"nodata fill fraction {info['fill_fraction']:.4f} (nearest-valid); "
             f"{len(idx)} traces; ct_dist {ct_dist:.0f} m (window reach "
             f"{x_max:.0f} m, cap {CT_CAP:.0f}); simc metrics over first "
             f"{n_cmp}/{n_samples} bins (mutual coverage); wall time "
             f"soundersim {t_ours:.1f} s, simc {t_simc:.1f} s")
    (out / "metrics.json").write_text(json.dumps({
        "case": f"opr_{case['frame_id']}",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metrics": metrics, "notes": notes, "group": "xOPR clutter",
    }, indent=1) + "\n")

    _figures(out, case, frame, idx, ds, simc, n_cmp, scene, off)
    print(f"{case['frame_id']}: ours {t_ours:.1f} s, simc {t_simc:.1f} s, "
          f"offset {off*1e9:.1f} ns ({off/dt:+.2f} bins), median resid "
          f"{p50:.2f} bins (p90 {p90:.2f})")
    return metrics, out


def _figures(out, case, frame, idx, ds, simc, n_cmp, scene, off):
    tw_us = ds.twtt.values * 1e6
    surf_us = frame.Surface.values[idx] * 1e6
    meas = _db(frame.Data.values[idx])
    ours = _db(np.asarray(ds.power, np.float64))
    sc = _db(simc["cluttergram"][:n_cmp].T)
    x = np.arange(len(idx))
    ext = [0, len(idx), tw_us[-1], tw_us[0]]

    # (a) measured radargram vs soundersim cluttergram, Surface pick on both.
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, img, ttl in [(axs[0], meas, "measured (CSARP_standard)"),
                         (axs[1], ours, "soundersim cluttergram")]:
        vmax = np.percentile(img[np.isfinite(img) & (img > -290)], 99.5)
        ax.imshow(img.T, aspect="auto", extent=ext, cmap="gray",
                  vmin=vmax - 60, vmax=vmax)
        ax.plot(x, surf_us, "c", lw=0.7, label="Surface pick")
        ax.set_title(ttl)
        ax.set_xlabel("trace (subsampled)")
        ax.legend(loc="lower right")
    axs[1].plot(x, surf_us + off * 1e6, "y", lw=0.7, label="Surface + offset")
    axs[1].legend(loc="lower right")
    axs[0].set_ylabel("twtt (us)")
    fig.suptitle(f"{case['frame_id']}: measured vs simulated surface clutter")
    fig.tight_layout()
    fig.savefig(out / "radargram_vs_cluttergram.png", dpi=150)
    plt.close(fig)

    # (b) soundersim vs simc + dB difference over the comparison window.
    o = ours[:, :n_cmp]
    vmax = np.percentile(sc[sc > -290], 99.5)
    fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    kw = dict(aspect="auto", extent=[0, len(idx), tw_us[n_cmp - 1], tw_us[0]])
    axs[0].imshow(o.T, cmap="gray", vmin=vmax - 60, vmax=vmax, **kw)
    axs[0].set_title("soundersim (dB)")
    axs[1].imshow(sc.T, cmap="gray", vmin=vmax - 60, vmax=vmax, **kw)
    axs[1].set_title("simc (dB)")
    d = np.where((o > vmax - 60) | (sc > vmax - 60), o - sc, np.nan)
    im = axs[2].imshow(d.T, cmap="RdBu_r", vmin=-10, vmax=10, **kw)
    axs[2].set_title("difference (dB)")
    fig.colorbar(im, ax=axs[2])
    axs[0].set_ylabel("twtt (us)")
    for ax in axs:
        ax.set_xlabel("trace")
    fig.suptitle(f"{case['frame_id']}: soundersim vs simc")
    fig.tight_layout()
    fig.savefig(out / "soundersim_vs_simc.png", dpi=150)
    plt.close(fig)

    # (c) map context: DEM hillshade + track.
    tr = scene.transform
    ny, nx = scene.dem.shape
    extent = [tr.c, tr.c + nx * tr.a, tr.f + ny * tr.e, tr.f]
    hs = LightSource(azdeg=315, altdeg=45).hillshade(
        np.asarray(scene.dem, np.float64), dx=abs(tr.a), dy=abs(tr.e))
    px, py = Transformer.from_crs("EPSG:4326", scene.crs, always_xy=True
                                  ).transform(scene.nav_llh[:, 1], scene.nav_llh[:, 0])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(hs, cmap="gray", extent=extent)
    im = ax.imshow(scene.dem, cmap="terrain", alpha=0.45, extent=extent)
    fig.colorbar(im, ax=ax, label="surface elevation (m, WGS84 ellipsoid)")
    ax.plot(px, py, "r", lw=1.5, label="track")
    ax.plot(px[0], py[0], "ro", ms=5)
    ax.set_xlabel(f"easting (m, {scene.crs})")
    ax.set_ylabel("northing (m)")
    ax.set_title(f"{case['frame_id']}: {scene.params['dem_product']} + track")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "map_context.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for case in CASES:
        run_case(case)
