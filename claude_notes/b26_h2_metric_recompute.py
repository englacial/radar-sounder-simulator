"""H2 (free experiment): re-baseline the B26 sim-vs-measured firn gap with a
symmetric, null-robust band estimator.

The report's band level is the MEDIAN of a 5 m-smoothed dB depth profile at the
single closest-approach trace. For a sparse coherent echo train with deep
interference nulls that sits well below the mean power; the measured profile is
smooth, so the asymmetry inflates the apparent gap. Here we recompute, from the
cached fields, the incoherent MEAN power over the band AND over all traces
(per-trace normalized to that trace's own 5 m-smoothed surface peak, identical
depth axis and smoothing to the tool), for every cached run and both measured
products, and compare each equal-placement run against its own 1-D transfer-
matrix expectation (b26_contrast_calc.py method).

Run: uv run python claude_notes/b26_h2_metric_recompute.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_firn_investigation as rfi  # noqa: E402
from soundersim.opr import load_frame  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from run_b26_comparison import (ALONG_M, FRAME_ID, OUT_DEFAULT,  # noqa: E402
                                SEASON, sub_frame)

C = 299792458.0
BANDS = ((20.0, 70.0), (80.0, 120.0))
SMOOTH_M = 5.0
RUNS = OUT_DEFAULT / "runs"
FIRN_KEYS = ["firn_N10", "firn_N20", "firn_N40", "firn_N80",
             "firn_N40_s0", "firn_N40_s1", "firn_N40_s2",
             "firn_N40_rough_mcords", "firn_N40_rough_ar"]


# ---------------------------------------------------------------- profiles
def norm_profiles(power, twtt, t_surf, dt):
    """(depth[T,n], ratio[T,n]): per-trace 5 m-boxcar-smoothed power divided by
    that trace's own smoothed surface peak, on the tool's depth axis.
    Identical maths to run_b26_comparison.profile_vs_depth, vectorized."""
    bin_depth = C * dt / (2.0 * np.sqrt(rfi.EPS_MEAN))
    w = max(int(round(SMOOTH_M / bin_depth)) | 1, 3)
    k = np.ones(w) / w
    ps = np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 1, power)
    depth = np.empty_like(ps)
    ratio = np.empty_like(ps)
    for t in range(ps.shape[0]):
        i0 = int(np.clip(np.searchsorted(twtt, t_surf[t]) - int(0.3e-6 / dt),
                         0, len(twtt) - 2))
        i1 = int(np.clip(np.searchsorted(twtt, t_surf[t] + 1.0e-6),
                         i0 + 1, len(twtt)))
        pk = max(ps[t, i0:i1].max(), 1e-300)
        ratio[t] = ps[t] / pk
        depth[t] = (twtt - t_surf[t]) * C / (2.0 * np.sqrt(rfi.EPS_MEAN))
    return depth, ratio


def surface_peak_twtt(power, twtt, t_guess, dt):
    """Per-trace surface peak time within +-0.8 us of a guess (tool's helper)."""
    n = len(twtt)
    out = np.full(power.shape[0], np.nan)
    for t in range(power.shape[0]):
        if not np.isfinite(t_guess[t]):
            continue
        a = int(np.clip((t_guess[t] - 0.8e-6 - twtt[0]) / dt, 0, n - 2))
        b = int(np.clip((t_guess[t] + 0.8e-6 - twtt[0]) / dt, a + 1, n))
        out[t] = twtt[a + int(np.argmax(power[t, a:b]))]
    return out


def estimators(depth, ratio, j0, band):
    """The four extraction variants, dB rel own surface peak."""
    def db(x):
        return 10.0 * np.log10(max(float(x), 1e-30))
    m = (depth >= band[0]) & (depth < band[1])
    r0 = ratio[j0][m[j0]]
    good = np.isfinite(ratio) & m
    return {
        "median_dB_j0": float(np.median(10 * np.log10(
            np.maximum(r0, 1e-12)))),                      # the report metric
        "mean_pow_j0": db(r0.mean()),
        "median_dB_all": float(np.median(10 * np.log10(
            np.maximum(ratio[good], 1e-12)))),
        "mean_pow_all": db(ratio[good].mean()),            # the fair metric
    }


# ------------------------------------------------------- 1-D TMM expectation
FIX = ROOT / "tests" / "fixtures" / "firn"
F0, B = 195e6, 30e6
LAM = C / F0


def _load_core():
    lines = (FIX / "ngt37C95.2_density.tab").read_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Depth ice/snow"))
    d = np.loadtxt(FIX / "ngt37C95.2_density.tab", delimiter="\t",
                   skiprows=hdr + 1)
    return d[:, 0], d[:, 1]


Z, RHO = _load_core()
DZ = float(np.median(np.diff(Z)))


def _smooth(x, meters):
    k = int(round(meters / DZ)) | 1
    box = np.ones(k)
    return np.convolve(x, box, "same") / np.convolve(np.ones_like(x), box, "same")


EPS_FULL = (1.0 + 0.845 * RHO / 1000.0) ** 2
EPS_S = (1.0 + 0.845 * _smooth(RHO, 0.1) / 1000.0) ** 2
N_FULL, N_S = np.sqrt(EPS_FULL), np.sqrt(EPS_S)
EPS_MEAN_1D = float(EPS_FULL.mean())
RES_FIRN = 1.44 * C / (2 * B) / np.sqrt(EPS_MEAN_1D)
_i02 = int(np.argmin(np.abs(Z - 0.2)))
GAMMA_SURF = ((1 - N_S[_i02]) / (1 + N_S[_i02])) ** 2


def tmm_r(n_stack, d, lam):
    kx = 2 * np.pi / lam * n_stack
    phi = kx[1:-1] * d
    M = np.eye(2, dtype=complex)
    for m in range(len(d)):
        ratio = kx[m + 1] / kx[m]
        D = 0.5 * np.array([[1 + ratio, 1 - ratio], [1 - ratio, 1 + ratio]],
                           dtype=complex)
        M = M @ D @ np.diag([np.exp(-1j * phi[m]), np.exp(1j * phi[m])])
    ratio = kx[-1] / kx[-2]
    M = M @ (0.5 * np.array([[1 + ratio, 1 - ratio], [1 - ratio, 1 + ratio]],
                            dtype=complex))
    return M[1, 0] / M[0, 0]


def expect_1d(depths, band):
    """b26_contrast_calc's N-point-sampled expectation: sum of Fresnel gammas
    of the point-sampled interfaces inside the band, spread over the band at
    one in-firn range cell each, normalized to the air->firn surface power."""
    nn = np.array([N_S[np.argmin(np.abs(Z - d))] for d in depths]
                  + [N_S[np.argmin(np.abs(Z - (depths[-1] + 1.0)))]])
    gam = ((nn[:-1] - nn[1:]) / (nn[:-1] + nn[1:])) ** 2
    iface_z = np.concatenate([depths[1:], [depths[-1] + 0.01]])
    m = (iface_z >= band[0]) & (iface_z <= band[1])
    if not m.any():
        return float("nan")
    per_cell = gam[m].sum() * RES_FIRN / (band[1] - band[0])
    return 10 * np.log10(per_cell / GAMMA_SURF)


def expect_1d_fullres(band):
    edges = np.arange(Z[0], Z.max(), RES_FIRN)
    idx = np.searchsorted(Z, edges)
    zc, p = [], []
    for i in range(len(edges)):
        s = idx[i]
        e = idx[i + 1] if i + 1 < len(edges) else len(Z)
        if e - s < 1:
            continue
        stack = np.concatenate(([N_FULL[s - 1] if s else 1.0], N_FULL[s:e],
                                [N_FULL[min(e, len(N_FULL) - 1)]]))
        zc.append(edges[i] + RES_FIRN / 2)
        p.append(abs(tmm_r(stack, np.full(e - s, DZ), LAM)) ** 2)
    zc, p = np.array(zc), np.array(p)
    m = (zc >= band[0]) & (zc <= band[1])
    return 10 * np.log10(p[m].mean() / GAMMA_SURF)


# ------------------------------------------------------------------- main
def main():
    cfg = json.loads((OUT_DEFAULT / "run_config.json").read_text())
    n_traces, along = cfg["n_traces"], cfg["along_m"]
    j0_sim = cfg["closest_trace"]["sim_index"]

    frame = load_frame(SEASON, FRAME_ID)
    fsub, sinfo = sub_frame(frame, along)
    idx = np.unique(np.round(np.linspace(
        0, fsub.sizes["slow_time"] - 1, n_traces)).astype(int))
    surf_pick = np.asarray(fsub.Surface.values, np.float64)[idx]

    # ---- simulated ----
    wide = dict(np.load(RUNS / "wide_surface_bed.npz"))
    tw = wide["twtt"]
    dt = float(tw[1] - tw[0])
    E2 = wide["field"].sum(-1)
    totals = {"surface+bed": E2}
    for k in FIRN_KEYS:
        p = RUNS / f"{k}.npz"
        if p.exists():
            totals[k] = E2 + np.load(p)["field"][..., 1:].sum(-1)

    rows = {}
    for name, E in totals.items():
        pw = np.abs(E) ** 2
        t_s = surface_peak_twtt(pw, tw, surf_pick, dt)
        d, r = norm_profiles(pw, tw, t_s, dt)
        rows[name] = {f"{lo:.0f}-{hi:.0f}m": estimators(d, r, j0_sim, (lo, hi))
                      for lo, hi in BANDS}

    # ---- measured (both products), same estimators over all sub-frame traces
    for tag, prod in (("measured", "CSARP_standard"),
                      ("measured_qlook", "CSARP_qlook")):
        fr = frame if prod == "CSARP_standard" else load_frame(
            SEASON, FRAME_ID, data_product=prod)
        sub, info = sub_frame(fr, along)
        twm = np.asarray(fr.twtt.values, np.float64)
        dtm = float(twm[1] - twm[0])
        pw = np.asarray(sub.Data.values, np.float64)
        pw = np.nan_to_num(pw, nan=0.0)
        t_s = surface_peak_twtt(pw, twm, np.asarray(sub.Surface.values,
                                                    np.float64), dtm)
        keep = np.isfinite(t_s)
        d, r = norm_profiles(pw[keep], twm, t_s[keep], dtm)
        j0 = int(np.searchsorted(np.flatnonzero(keep), info["i0_local"]))
        j0 = min(j0, d.shape[0] - 1)
        rows[tag] = {f"{lo:.0f}-{hi:.0f}m": estimators(d, r, j0, (lo, hi))
                     for lo, hi in BANDS}
        rows[tag]["n_traces"] = int(keep.sum())

    # ---- 1-D expectations for the equal-placement stacks ----
    exp1d = {}
    for n in (10, 20, 40, 80):
        dpt = rfi.equal_depths(n)
        exp1d[f"firn_N{n}"] = {f"{lo:.0f}-{hi:.0f}m": expect_1d(dpt, (lo, hi))
                               for lo, hi in BANDS}
    for n, s in ((40, 0), (40, 1), (40, 2)):
        dpt = rfi.random_depths(n, s)
        exp1d[f"firn_N{n}_s{s}"] = {f"{lo:.0f}-{hi:.0f}m":
                                    expect_1d(dpt, (lo, hi))
                                    for lo, hi in BANDS}
    for k in ("firn_N40_rough_mcords", "firn_N40_rough_ar"):
        exp1d[k] = exp1d["firn_N40"]
    full = {f"{lo:.0f}-{hi:.0f}m": expect_1d_fullres((lo, hi))
            for lo, hi in BANDS}

    # ---- report ----
    ref = json.loads((OUT_DEFAULT / "run_config.json").read_text())[
        "band_levels_db_rel_surface"]
    for lo, hi in BANDS:
        b = f"{lo:.0f}-{hi:.0f}m"
        print(f"\n=== band {b} (dB rel own surface peak) ===")
        print(f"{'run':24s} {'med@j0':>8s} {'(report)':>9s} {'meanP@j0':>9s} "
              f"{'medAll':>8s} {'meanPall':>9s} {'1-D exp':>8s} {'resid':>7s}")
        for k, v in rows.items():
            e = v[b]
            r_ref = ref.get(k, {}).get(b, float("nan"))
            ex = exp1d.get(k, {}).get(b, float("nan"))
            res = e["mean_pow_all"] - ex
            print(f"{k:24s} {e['median_dB_j0']:8.2f} {r_ref:9.2f} "
                  f"{e['mean_pow_j0']:9.2f} {e['median_dB_all']:8.2f} "
                  f"{e['mean_pow_all']:9.2f} {ex:8.2f} {res:7.2f}")
        print(f"{'1-D full-res TMM':24s} {'':8s} {'':9s} {'':9s} {'':8s} "
              f"{'':9s} {full[b]:8.2f}")
        # headline gap
        for mk in ("measured", "measured_qlook"):
            for est in ("median_dB_j0", "mean_pow_all"):
                g = rows[mk][b][est] - rows["firn_N40"][b][est]
                print(f"  gap {mk} - firn_N40 [{est}] = {g:+.2f} dB")

    out = {"bands": {f"{lo:.0f}-{hi:.0f}m": {
        "rows": {k: v[f"{lo:.0f}-{hi:.0f}m"] for k, v in rows.items()},
        "expect_1d": {k: v[f"{lo:.0f}-{hi:.0f}m"] for k, v in exp1d.items()},
        "expect_1d_fullres": full[f"{lo:.0f}-{hi:.0f}m"]}
        for lo, hi in BANDS},
        "n_sim_traces": int(n_traces),
        "n_measured_traces": {k: rows[k]["n_traces"]
                              for k in ("measured", "measured_qlook")}}
    p = ROOT / "claude_notes" / "b26_h2_metric_recompute.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
