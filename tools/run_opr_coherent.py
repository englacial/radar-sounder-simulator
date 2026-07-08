"""Coherent (complex-field) xOPR clutter cases (M13, simplified per review).

For each cached OPR frame we subdivide the DEM to meet a Fresnel-zone facet-size
criterion and run the coherent kernel. A case dir
``outputs/verification/opr_<frame_id>_coherent/`` is written (group
"xOPR clutter"); the stage-1 incoherent-vs-simc case is left untouched.

Deliverable per frame:
  * a figure row on shared twtt/slow_time axes -- measured radargram (dB) |
    coherent |field|^2 (dB), with the frame Surface pick overlaid on both;
  * a speckle panel (detrended surface-return intensity vs the exponential);
  * metrics.json: speckle contrast, the subdivision choice + estimated per-facet
    nadir LPA error, wall time, and one loose sanity gate (smoothed-coherent
    surface leading edge tracking the frame's Surface pick after removing a
    constant offset, as in the stage-1 incoherent case).

Subdivision: the ideal L <= 0.1*sqrt(lambda*r_min) (~2.3 m here) is intractable
(~200x the native facet count). We degrade to beta=0.5 in check_facet_size, i.e.
L <= 0.5*sqrt(lambda*r_min) (half a Fresnel-zone radius, ~11-12 m), and RECORD
the estimated per-facet nadir LPA envelope error at that size. Cross-track reach
and trace count are trimmed for runtime; both recorded in the notes.

Physics honesty (per plan): at DEM-derived facet scales the coherent output is
statistically meaningful (speckle, envelope) but NOT deterministically phase-
accurate -- a 32 m DEM cannot supply lambda-scale surface phase, which is why
simc's authors stayed incoherent (cf. Gerekos 2023 analytic sub-facet roughness,
out of scope here).

Run: uv run python tools/run_opr_coherent.py
"""

import datetime
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

from soundersim.compare.brute_force import _contributions, flat_rectangle_samples
from soundersim.config import FacetConfig, RadarConfig, SimConfig
from soundersim.kernels.coherent import lpa_contributions
from soundersim.opr import frame_scene, load_frame
from soundersim.physics import fresnel_normal
from soundersim.simulate import simulate

# Reuse the frame list, dB helper and output root from the stage-1 tool (DRY).
from run_opr_comparison import CASES, OUT_ROOT, _db  # noqa: E402

C = 299792458.0
F0 = 195e6                 # MCoRDS band center; sets lambda ~ 1.54 m
BETA = 0.5                 # check_facet_size beta -> L <= 0.5*sqrt(lam*r_min)
CT_DIST_COH = 3000.0       # trimmed cross-track reach for the coherent run (m)
N_TRACES_COH = 100         # trimmed trace count for runtime


def _lpa_nadir_error(L, r, k, gamma):
    """Envelope-normalized |LPA - brute force| for one flat L x L facet viewed
    at nadir range r -- the worst-case LPA validity point (near-nadir, large L).
    Envelope = (k/2pi)|gamma| A / r^2 (the sinc=1 amplitude)."""
    lam = 2.0 * np.pi / k
    p = np.array([0.0, 0.0, r])
    pts, nrm, dA = flat_rectangle_samples(L, L, lam / 12.0)
    bf = _contributions(p, pts, nrm, dA, k, gamma)[0].sum()
    lp = lpa_contributions(
        p, np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), np.array([L * L]),
        np.array([[L, 0.0, 0.0]]), np.array([[0.0, L, 0.0]]), k, gamma, xp=np)[0][0]
    env = (k / (2.0 * np.pi)) * abs(gamma) * L * L / r ** 2
    return float(abs(lp - bf) / env)


def _leading_edge(power, drop_db=15.0):
    """Per-trace first fast-time bin exceeding (per-trace peak) - drop_db."""
    thr = power.max(axis=1, keepdims=True) * 10.0 ** (-drop_db / 10.0)
    over = power > thr
    has = over.any(axis=1)
    idx = np.where(has, over.argmax(axis=1), -1)
    return idx, has


def run_coherent_case(case, n_traces=N_TRACES_COH, ct_dist=CT_DIST_COH,
                      out_root=OUT_ROOT, spacing=None):
    frame = load_frame(case["season"], case["frame_id"])
    tw = frame.twtt.values
    dt = float((tw[-1] - tw[0]) / (len(tw) - 1))
    t0, n_samples = float(tw[0]), len(tw)
    rc = RadarConfig(dt=dt, n_samples=n_samples, t0=t0, f0=F0)
    lam = rc.wavelength
    k = 2.0 * np.pi / lam
    gamma = fresnel_normal(1.0, 3.17)

    # r_min: minimum platform->surface nadir range from the frame Surface pick
    # (twtt * c / 2). The closest facet is at nadir, so this sets the tightest
    # Fresnel-zone limit; spacing = beta * sqrt(lam * r_min).
    r_min = float(np.nanmin(frame.Surface.values)) * C / 2.0
    # Default subdivision from the Fresnel criterion; ``spacing`` override lets
    # the integration test use a coarse (fast) grid without touching the physics
    # path or the sanity gate.
    if spacing is None:
        spacing = BETA * np.sqrt(lam * r_min)
    lpa_err = _lpa_nadir_error(spacing, r_min, k, gamma)

    scene, info = frame_scene(frame, n_traces=n_traces, ct_dist=ct_dist)
    idx = info["trace_idx"]
    n_native = (scene.dem.shape[0] - 1) * (scene.dem.shape[1] - 1)

    cfg = SimConfig(mode="coherent", split_sides=False, radar=rc,
                    facets=FacetConfig(spacing=spacing))
    t = time.perf_counter()
    ds_coh = simulate(scene, cfg)
    t_coh = time.perf_counter() - t
    # Subdivided facet count from the build_facets refinement geometry (cheap;
    # avoids materializing the ~1e7-facet grid a second time).
    ny, nx = scene.dem.shape
    f = 32.0 / spacing
    nrv = max(2, int(round((ny - 1) * f)) + 1)
    ncv = max(2, int(round((nx - 1) * f)) + 1)
    n_facets = (nrv - 1) * (ncv - 1)

    coh = np.asarray(ds_coh.power, np.float64)   # |field|^2 (T, n_samples)

    # Facet-scale fast-time smoothing (leading-edge extraction): boxcar of
    # width ~ subdivided facet size / range-bin.
    range_bin = C * dt / 2.0
    w = max(1, int(round(spacing / range_bin)))
    sm_coh = uniform_filter1d(coh, w, axis=1, mode="nearest")

    # Clutter-active region: bins within 30 dB of the coherent peak.
    active = sm_coh > sm_coh.max() * 1e-3
    n_show = min(n_samples, int(np.where(active.any(0))[0].max()) + 50) \
        if active.any() else n_samples

    # Leading edge (used by the speckle band and the sanity gate below).
    le_c, hc = _leading_edge(sm_coh)

    # Speckle contrast on the SURFACE RETURN, sliding window along slow time
    # (per plan). Traces are aligned on the smoothed-incoherent leading edge and
    # a surface-following band (leading edge +3 .. +3+60 raw bins) extracted;
    # within it the envelope varies slowly from trace to trace while the
    # platform moves many wavelengths, so trace-to-trace fluctuation is speckle.
    # Per fast-time row: std/mean over an 11-trace sliding window; the reported
    # contrast is the median over all (row, window) cells. Fully developed
    # (exponential) speckle -> ~1; a deterministic specular component (Rician)
    # pulls it below 1, sparse few-facet bins push it above. (Detrending by a
    # fast-time boxcar instead would just measure the specular spike/boxcar
    # ratio -- the surface apex is near-deterministic, not speckle.)
    band0, win = 3, 11
    keep = np.where(hc)[0]
    bandw = int(min(60, (n_samples - (le_c[keep] + band0)).min()))
    aligned = np.stack([coh[t, le_c[t] + band0: le_c[t] + band0 + bandw]
                        for t in keep])                        # (T', bandw)
    mu = uniform_filter1d(aligned, win, axis=0, mode="nearest")
    mu2 = uniform_filter1d(aligned ** 2, win, axis=0, mode="nearest")
    var = np.maximum(mu2 - mu ** 2, 0.0)
    h = win // 2
    valid = mu[h:-h] > 0
    cmap_ = np.sqrt(var[h:-h])[valid] / mu[h:-h][valid]
    speckle_contrast = float(np.median(cmap_))
    resid = (aligned[h:-h] / np.maximum(mu[h:-h], 1e-300))[valid]

    # Sanity gate: smoothed-coherent surface leading edge vs the frame's
    # measured Surface pick, after removing the constant offset (as in the
    # stage-1 incoherent case; the offset absorbs system delay / DEM epoch).
    surf_bin = (frame.Surface.values[idx] - t0) / dt
    both = hc & np.isfinite(surf_bin)
    le_resid = le_c[both] - surf_bin[both]
    le_offset = float(np.median(le_resid))
    dle = np.abs(le_resid - le_offset)
    # Median, not p90: on rugged terrain the tracker locks onto off-nadir
    # clutter in places (heavy tail), same as the stage-1 incoherent gate.
    le_med = float(np.median(dle))
    le_p90 = float(np.percentile(dle, 90))
    le_max = float(dle.max())
    gate_thr = 5.0
    gate_pass = bool(le_med <= gate_thr)

    rec = ("recorded only; pass forced true (real-frame thresholds are set "
           "after observing residuals, per plan)")
    metrics = {
        "speckle_contrast": {
            "value": speckle_contrast, "threshold": None, "pass": True,
            "op": "~1", "band_bins": bandw, "window_traces": win,
            "note": "median std/mean of coherent surface-return intensity over "
            f"{win}-trace sliding slow-time windows, per fast-time row of a "
            f"surface-following band (leading edge +{band0}..+{band0+bandw} "
            "bins); ~1 for fully developed (exponential) speckle, <1 where a "
            "deterministic specular component remains (Rician), >1 in sparse "
            "few-facet bins. " + rec},
        "lpa_nadir_error": {
            "value": lpa_err, "threshold": None, "pass": True, "op": "record",
            "facet_size_m": float(spacing), "r_min_m": r_min,
            "note": "envelope-normalized single-facet LPA error at nadir (worst "
            "case); off-nadir clutter facets are sinc-suppressed and far more "
            "accurate. " + rec},
        "surface_leading_edge": {
            "value": le_med, "threshold": gate_thr, "pass": gate_pass, "op": "<=",
            "p90_bins": le_p90, "max_abs_bins": le_max, "offset_bins": le_offset,
            "note": "median |smoothed-coherent leading edge - frame Surface "
            "pick| in raw bins after removing the constant offset (recorded; "
            "absorbs system delay / DEM epoch). Median gates and p90/max are "
            "recorded because the tracker locks onto off-nadir clutter on "
            "rugged sections (heavy tail), as in the stage-1 incoherent case"},
    }

    out = out_root / f"opr_{case['frame_id']}_coherent"
    out.mkdir(parents=True, exist_ok=True)
    notes = (
        f"{case['season']} {case['frame_id']} ({case['why']}); coherent "
        f"cluttergram. f0 {F0/1e6:.0f} MHz (lambda {lam:.2f} m), media air/ice "
        f"(surface interface only, gamma {gamma:.3f}). Subdivision: spacing "
        f"{spacing:.1f} m = beta {BETA} * sqrt(lambda*r_min) with r_min "
        f"{r_min:.0f} m (check_facet_size beta=0.5); ideal "
        f"0.1*sqrt(lambda*r_min) ~{0.1*np.sqrt(lam*r_min):.1f} m was "
        f"intractable. Native 32 m grid {n_native} cells -> subdivided "
        f"{n_facets} facets ({n_facets/max(n_native,1):.1f}x). Estimated "
        f"per-facet nadir LPA envelope error {lpa_err*100:.0f}% (worst case; "
        f"off-nadir far smaller). Runtime trims: ct_dist {ct_dist:.0f} m, "
        f"{len(idx)} traces. Leading-edge extraction uses a {w}-bin facet-scale "
        f"boxcar (facet {spacing:.1f} m / range-bin {range_bin:.1f} m). Speckle "
        f"contrast: sliding {win}-trace slow-time window over a surface-"
        f"following band ({bandw} bins after the leading edge). "
        f"Wall time coherent {t_coh:.1f} s. HONESTY: (1) at 32 m DEM posting "
        f"the surface phase is not lambda-accurate, so this coherent product is "
        f"meaningful for speckle/envelope statistics but not deterministic "
        f"phase; (2) at these (necessarily large) facets the coherent LPA "
        f"carries facet directivity (sinc^2): smooth DEM facets act as flat "
        f"mirrors, so the coherent return is specular-dominated (bright surface "
        f"leading edge, little off-nadir diffuse clutter), whereas the measured "
        f"frame shows broad diffuse clutter fed by sub-lambda roughness the "
        f"32 m DEM cannot represent (cf. Gerekos 2023).")
    (out / "metrics.json").write_text(json.dumps({
        "case": f"opr_{case['frame_id']}_coherent",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metrics": metrics, "notes": notes, "group": "xOPR clutter",
    }, indent=1) + "\n")

    _figures(out, case, frame, idx, ds_coh, resid, n_show, le_offset * dt,
             speckle_contrast)
    print(f"{case['frame_id']}_coherent: coh {t_coh:.1f}s | "
          f"spacing {spacing:.1f}m {n_facets} facets LPA~{lpa_err*100:.0f}% | "
          f"speckle {speckle_contrast:.2f} "
          f"| leadedge med {le_med:.1f} p90 {le_p90:.1f} bins (pass {gate_pass})")
    return metrics, out


def _figures(out, case, frame, idx, ds_coh, resid, n_show, le_offset_s,
             speckle_contrast):
    tw_us = ds_coh.twtt.values * 1e6
    surf_us = frame.Surface.values[idx] * 1e6
    meas = _db(frame.Data.values[idx])
    coh_db = _db(np.asarray(ds_coh.power, np.float64))
    x = np.arange(len(idx))
    ext = [0, len(idx), tw_us[n_show - 1], tw_us[0]]
    sl = slice(0, n_show)

    # (a) the deliverable row: measured | coherent, Surface pick on both.
    fig, axs = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    coh_fin = coh_db[:, sl][np.isfinite(coh_db[:, sl]) & (coh_db[:, sl] > -290)]
    vmax = np.percentile(coh_fin, 99.5)
    kw = dict(aspect="auto", extent=ext, cmap="gray", vmin=vmax - 60, vmax=vmax)
    meas_fin = meas[:, sl][np.isfinite(meas[:, sl]) & (meas[:, sl] > -290)]
    mv = np.percentile(meas_fin, 99.5)
    axs[0].imshow(meas[:, sl].T, **{**kw, "vmin": mv - 60, "vmax": mv})
    axs[0].plot(x, surf_us, "c", lw=0.7, label="Surface pick")
    axs[0].set_title("measured (CSARP_standard, dB)")
    axs[0].legend(loc="lower right", fontsize=8)
    axs[1].imshow(coh_db[:, sl].T, **kw)
    axs[1].plot(x, surf_us + le_offset_s * 1e6, "y", lw=0.7,
                label="Surface pick + offset")
    axs[1].set_title("coherent |field|^2 (dB)")
    axs[1].legend(loc="lower right", fontsize=8)
    axs[0].set_ylabel("twtt (us)")
    for ax in axs:
        ax.set_xlabel("trace (subsampled)")
    fig.suptitle(f"{case['frame_id']}: measured vs simulated coherent "
                 f"surface clutter")
    fig.tight_layout()
    fig.savefig(out / "radargram_vs_coherent.png", dpi=130)
    plt.close(fig)

    # (b) speckle panel: detrended surface-return intensity histogram vs the
    # unit-mean exponential (fully developed speckle).
    fig, ax = plt.subplots(figsize=(7, 5))
    r = resid[resid < np.percentile(resid, 99.5)]
    ax.hist(r, bins=60, density=True, alpha=0.6, label="coherent (detrended)")
    xs = np.linspace(0, r.max(), 200)
    ax.plot(xs, np.exp(-xs), "r", lw=1.8, label="exponential (contrast 1)")
    ax.set_xlabel("intensity / local mean")
    ax.set_ylabel("pdf")
    ax.set_title(f"{case['frame_id']}: surface-return speckle\n"
                 f"contrast std/mean = {speckle_contrast:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "speckle.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    for case in CASES:
        run_coherent_case(case)
