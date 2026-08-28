"""Along-track surface-roughness statistics from ICESat-2 ATL06 / ATL03.

Given an ATL06 (20 m land-ice segments) and/or ATL03 (photons) granule and a
lon/lat box, extract the beam profiles crossing the box, detrend at a chosen
scale, and compute
  - RMS height in along-track scale bands,
  - the 1-D along-track ACF with Gaussian / exponential fits (sigma, l),
  - the Welch PSD with Gaussian / exponential / power-law fits,
  - (ATL06) the per-segment sub-40 m spread `h_robust_sprd` and `h_li_sigma`,
  - (ATL03) the within-shot photon spread as a sub-footprint RMS proxy.
With --synthetic, a Gaussian-ACF surface at 0.7 m posting is generated, then
observed through an 11 m Gaussian footprint and 20 m ATL06 posting, to show
what survives.  Isotropy is assumed throughout: a 1-D transect through an
isotropic 2-D Gaussian (or exponential) surface has the same ACF shape and l.

Usage
  uv run claude_notes/icesat2_roughness/roughness_from_atl.py --synthetic
  uv run claude_notes/icesat2_roughness/roughness_from_atl.py \
      --atl06 outputs/icesat2/ATL06_....h5 --bbox -49.77 70.50 -49.41 70.65
  ... --atl03 ATL03_....h5 --bbox ...   (photon path, strong beams)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, welch

R_EARTH = 6371e3
FOOTPRINT_FWHM_M = 11.0     # Magruder et al. 2021: ~10.6-12 m; design 17 m
FOOTPRINT_SIG_M = FOOTPRINT_FWHM_M / 2.355


# ----------------------------------------------------------------- readers
def _along_track(lat, lon):
    """Cumulative great-circle distance (m) along a point sequence."""
    la, lo = np.radians(lat), np.radians(lon)
    d = R_EARTH * np.hypot(np.diff(la), np.diff(lo) * np.cos(la[:-1]))
    return np.concatenate([[0.0], np.cumsum(d)])


def _in_box(lat, lon, bbox):
    if bbox is None:
        return np.ones_like(lat, bool)
    x0, y0, x1, y1 = bbox
    return (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)


def read_atl06(path, bbox, beams=None):
    """-> {beam: dict(s, h, h_sigma, sprd, dhdx, lat, lon)}; quality-filtered."""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for gt in beams or [g for g in f if g.startswith("gt")]:
            if f"{gt}/land_ice_segments" not in f:
                continue
            g = f[f"{gt}/land_ice_segments"]
            lat, lon, h = g["latitude"][:], g["longitude"][:], g["h_li"][:]
            q = g["atl06_quality_summary"][:] == 0
            m = _in_box(lat, lon, bbox) & q & np.isfinite(h) & (np.abs(h) < 1e4)
            if m.sum() < 50:
                continue
            fs = g["fit_statistics"]
            out[gt] = dict(
                s=_along_track(lat[m], lon[m]), h=h[m].astype(float),
                h_sigma=g["h_li_sigma"][:][m], sprd=fs["h_robust_sprd"][:][m],
                dhdx=fs["dh_fit_dx"][:][m], lat=lat[m], lon=lon[m],
                strong=bool(f[gt].attrs.get("atlas_beam_type", b"strong") == b"strong"),
                dx=20.0)
    return out


def read_atl03(path, bbox, beams=None, bin_m=1.0, conf_min=3):
    """Photon path: land-ice signal photons -> per-shot (0.7 m) statistics.

    Returns both the shot-median profile (posted on bin_m) and the pooled
    within-shot spread, the sub-footprint RMS proxy (instrument noise
    included -- calibrate on a flat target before quoting).
    """
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for gt in beams or [g for g in f if g.startswith("gt")]:
            if f"{gt}/heights" not in f:
                continue
            g = f[f"{gt}/heights"]
            lat, lon = g["lat_ph"][:], g["lon_ph"][:]
            m = _in_box(lat, lon, bbox)
            if m.sum() < 1000:
                continue
            h = g["h_ph"][:][m]
            conf = g["signal_conf_ph"][:, 3][m]          # column 3 = land ice
            seg_dist = f[f"{gt}/geolocation/segment_dist_x"][:]
            ph_idx = f[f"{gt}/geolocation/ph_index_beg"][:] - 1
            ph_cnt = f[f"{gt}/geolocation/segment_ph_cnt"][:]
            seg_of_ph = np.repeat(np.arange(len(ph_cnt)), ph_cnt)[:len(g["h_ph"])]
            x = (seg_dist[seg_of_ph] + g["dist_ph_along"][:])[m]
            shot = g["pce_mframe_cnt"][:][m].astype(np.int64) * 1000 + g["ph_id_pulse"][:][m]
            keep = conf >= conf_min
            x, h, shot = x[keep], h[keep], shot[keep]
            if len(h) < 1000:
                continue
            # within-shot spread (sub-footprint sigma proxy)
            order = np.argsort(shot, kind="stable")
            shot, x, h = shot[order], x[order], h[order]
            _, first, cnt = np.unique(shot, return_index=True, return_counts=True)
            multi = cnt >= 3
            resid = []
            for i0, n in zip(first[multi], cnt[multi]):
                resid.append(h[i0:i0 + n] - np.median(h[i0:i0 + n]))
            resid = np.concatenate(resid) if resid else np.array([])
            # shot/bin-median profile
            xb = np.floor((x - x.min()) / bin_m).astype(int)
            nb = xb.max() + 1
            sums = np.bincount(xb, h, nb); cnts = np.bincount(xb, None, nb)
            ok = cnts > 0
            out[gt] = dict(s=(np.arange(nb)[ok] + 0.5) * bin_m, h=sums[ok] / cnts[ok],
                           dx=bin_m, ph_per_shot=float(cnt.mean()),
                           within_shot_resid=resid,
                           strong=bool(f[gt].attrs.get("atlas_beam_type", b"strong") == b"strong"))
    return out


# ------------------------------------------------------------ statistics
def uniform_chunks(s, h, dx, max_gap_factor=3.0, min_len_m=500.0):
    """Split at gaps, resample each contiguous run onto a uniform dx grid."""
    brk = np.where(np.diff(s) > max_gap_factor * dx)[0] + 1
    for a, b in zip(np.r_[0, brk], np.r_[brk, len(s)]):
        if s[b - 1] - s[a] < min_len_m:
            continue
        su = np.arange(s[a], s[b - 1], dx)
        yield su, np.interp(su, s[a:b], h[a:b])


def detrend(h, dx, scale_m):
    """Remove structure longer than scale_m (Savitzky-Golay, poly 2)."""
    w = int(scale_m / dx) | 1
    if w < 5 or w >= len(h):
        return h - np.polyval(np.polyfit(np.arange(len(h)), h, 2), np.arange(len(h)))
    return h - savgol_filter(h, w, 2)


def band_rms(h, dx, edges):
    """RMS height per along-track scale band (difference of SG low-passes)."""
    out = {}
    lp = {}
    for e in edges:
        w = int(e / dx) | 1
        lp[e] = savgol_filter(h, w, 2) if 5 <= w < len(h) else h * 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = lp[lo] - lp[hi]
        out[f"{lo:g}-{hi:g}m"] = float(np.sqrt(np.mean(band ** 2)))
    return out


def acf(h, dx, max_lag_m):
    n = int(max_lag_m / dx)
    h = h - h.mean()
    c = np.array([np.mean(h[:len(h) - k] * h[k:]) for k in range(n)])
    return np.arange(n) * dx, c


def acf_gauss(r, s2, l): return s2 * np.exp(-(r / l) ** 2)
def acf_exp(r, s2, l): return s2 * np.exp(-np.abs(r) / l)


def fit_acf(r, c, dx, fp_sig=0.0):
    """Fit Gaussian and exponential ACFs; lag 0 excluded (white-noise spike).

    Gaussian fit is footprint-aware: a Gaussian-ACF surface seen through a
    Gaussian footprint of std fp_sig has ACF sigma^2 (l/l_eff) exp(-(r/l_eff)^2),
    l_eff^2 = l^2 + 4 fp_sig^2 -- so for l << 2 fp_sig only sigma^2 * l is
    constrained; the returned l uncertainty says so.
    """
    m = (r > 0) & np.isfinite(c)
    res = {}
    c1 = max(float(c[1]), 1e-8) if len(c) > 1 else 1e-8
    def g_fp(r_, s2, l):
        le = np.sqrt(l ** 2 + 4 * fp_sig ** 2)
        return s2 * l / le * np.exp(-(r_ / le) ** 2)
    for name, fn in (("gaussian", g_fp), ("exponential", acf_exp)):
        try:
            p, cov = curve_fit(fn, r[m], c[m], p0=(c1, max(3 * dx, 2 * fp_sig)),
                               bounds=(1e-12, np.inf), maxfev=20000)
            err = np.sqrt(np.diag(cov))
            rss = float(np.sum((c[m] - fn(r[m], *p)) ** 2))
            res[name] = dict(constrained=bool(err[1] < p[1]), sigma_m=float(np.sqrt(p[0])),
                             l_m=float(p[1]), l_err_m=float(err[1]), rss=rss)
        except Exception as e:  # noqa: BLE001
            res[name] = dict(error=str(e))
    return res


def psd(h, dx, nperseg=None):
    nperseg = nperseg or min(len(h), 1024)
    k, p = welch(h - h.mean(), fs=1 / dx, nperseg=nperseg, detrend="linear")
    return 2 * np.pi * k[1:], p[1:] / (2 * np.pi)   # angular wavenumber, W(k)


def transfer(k, fp_sig=0.0, seg_len=0.0):
    """Power transfer of the observation: Gaussian footprint x boxcar segment."""
    t = np.exp(-(k * fp_sig) ** 2)
    if seg_len > 0:
        t = t * np.sinc(k * seg_len / (2 * np.pi)) ** 2
    return t


def fit_psd(k, p, dx, fp_sig=0.0, seg_len=0.0):
    """Gaussian / exponential / power-law PSD fits (log space), each as
    one-sided  2 W(k) T(k) + N0  with the footprint/segment transfer T and a
    white noise floor N0.  Two-sided 1-D spectra of an isotropic surface:
      Gaussian     W(k) = sigma^2 l/(2 sqrt(pi)) exp(-(k l)^2/4)
      exponential  W(k) = sigma^2 l/pi / (1 + (k l)^2)
      power law    W(k) = A k^-beta   (beta = 2H+1, H = Hurst exponent)
    """
    m = (k > 0) & (p > 0) & (k < np.pi / dx * 0.8)
    # log-spaced bins so the high-k (noise-floor) decade does not dominate
    edges = np.geomspace(k[m].min(), k[m].max(), 41)
    idx = np.clip(np.digitize(k[m], edges) - 1, 0, 39)
    kk = np.array([np.exp(np.mean(np.log(k[m][idx == i]))) for i in range(40) if np.any(idx == i)])
    pb = np.array([np.mean(p[m][idx == i]) for i in range(40) if np.any(idx == i)])
    lk, lp = np.log(kk), np.log(pb)
    p = pb; m = np.ones(len(pb), bool)
    T = transfer(kk, fp_sig, seg_len)
    n0_guess = max(float(np.median(p[m][-max(3, m.sum() // 10):])), 1e-12)
    plateau = max(float(np.median(p[m][:max(3, m.sum() // 20)])) - n0_guess, 1e-12)
    l0 = max(10 * dx, 2 * fp_sig)
    res = {}
    # log-parametrised so the optimiser is scale-free; q = (ln s2, ln l, ln n0)
    def g(lk_, a, b, c): s2, l, n0 = np.exp((a, b, c)); return np.log(s2 * l / np.sqrt(np.pi) * np.exp(-(kk * l) ** 2 / 4) * T + n0)
    def e(lk_, a, b, c): s2, l, n0 = np.exp((a, b, c)); return np.log(2 * s2 * l / np.pi / (1 + (kk * l) ** 2) * T + n0)
    def pl(lk_, a, beta, c): A, n0 = np.exp((a, c)); return np.log(2 * A * kk ** -beta * T + n0)
    s2l = plateau * np.sqrt(np.pi)
    for name, fn, p0 in (("gaussian", g, (np.log(s2l / l0), np.log(l0), np.log(n0_guess))),
                         ("exponential", e, (np.log(s2l / l0), np.log(l0), np.log(n0_guess))),
                         ("powerlaw", pl, (np.log(plateau * kk[0] ** 2 / 2), 2.0, np.log(n0_guess)))):
        try:
            pp, cov = curve_fit(fn, lk, lp, p0=p0, maxfev=50000)
            err = np.sqrt(np.abs(np.diag(cov)))
            rss = float(np.sum((lp - fn(lk, *pp)) ** 2))
            n0 = float(np.exp(pp[2]))
            ok = bool(np.all(np.isfinite(err)) and err[1] < 1.0)
            if name == "powerlaw":
                res[name] = dict(constrained=ok, A=float(np.exp(pp[0])), beta=float(pp[1]), beta_err=float(err[1]),
                                 hurst_H=float((pp[1] - 1) / 2), noise_floor_m=float(np.sqrt(n0 * np.pi / dx)), rss_log=rss)
            else:
                s2, l = float(np.exp(pp[0])), float(np.exp(pp[1]))
                res[name] = dict(constrained=ok, sigma_m=float(np.sqrt(s2)), l_m=l, l_rel_err=float(err[1]),
                                 sigma2_l=s2 * l, sigma2_l_rel_err=float(np.sqrt(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1])),
                                 noise_floor_m=float(np.sqrt(n0 * np.pi / dx)), rss_log=rss)
        except Exception as ex:  # noqa: BLE001
            res[name] = dict(error=str(ex))
    return res


def analyse_profile(s, h, dx, detrend_m, max_lag_m, band_edges, fp_sig=0.0, seg_len=0.0):
    band_edges = [e for e in band_edges if e >= 3 * dx] or [3 * dx, 5 * dx]
    chunks = list(uniform_chunks(s, h, dx))
    if not chunks:
        return None
    hd = [detrend(hc, dx, max(detrend_m, 25 * dx)) for _, hc in chunks]
    ac = [acf(x, dx, max_lag_m) for x in hd]
    r = ac[0][0]
    c = np.average([a[1] for a in ac], axis=0, weights=[len(x) for x in hd])
    hcat = np.concatenate(hd)
    k, p = psd(hcat, dx)
    keep = k > 2 * np.pi / max(detrend_m, 25 * dx)     # below the detrend scale only
    k, p = k[keep], p[keep]
    return dict(n_points=int(len(hcat)), length_km=float(sum(len(x) for x in hd) * dx / 1e3),
                rms_total_m=float(hcat.std()), band_rms_m=band_rms(hcat, dx, band_edges),
                acf_fit=fit_acf(r, c, dx, fp_sig), psd_fit=fit_psd(k, p, dx, fp_sig, seg_len),
                _acf=(r, c), _psd=(k, p))


# ------------------------------------------------------------- synthetic
def synth_gaussian_surface(n, dx, sigma, l, rng, beta=None):
    """1-D transect of an isotropic Gaussian-ACF (or power-law) surface."""
    k = 2 * np.pi * np.fft.rfftfreq(n, dx)
    if beta is None:
        W = sigma ** 2 * l / (2 * np.sqrt(np.pi)) * np.exp(-(k * l) ** 2 / 4)
    else:
        W = np.where(k > 0, k ** -beta, 0.0)
    w = np.fft.rfft(rng.standard_normal(n))        # white, PSD = dx/(2 pi)
    h = np.fft.irfft(w * np.sqrt(W * 2 * np.pi / dx), n)
    if beta is not None:
        h *= sigma / h.std()
    return h


def footprint_smooth(h, dx, sig_m=FOOTPRINT_SIG_M):
    k = 2 * np.pi * np.fft.rfftfreq(len(h), dx)
    return np.fft.irfft(np.fft.rfft(h) * np.exp(-(k * sig_m) ** 2 / 2), len(h))


def run_synthetic(args):
    rng = np.random.default_rng(0)
    dx, sigma, l = 0.7, args.sigma, args.l
    n = int(args.length_km * 1e3 / dx)
    s = np.arange(n) * dx
    true = synth_gaussian_surface(n, dx, sigma, l, rng)
    true += synth_gaussian_surface(n, dx, 2.0, 3000.0, rng)  # long-wave undulation (detrended)
    fp = footprint_smooth(true, dx)
    seg = np.convolve(fp, np.ones(57) / 57, "same")     # 40 m linear-fit window
    cases = {  # name: (s, h, dx, fp_sig, seg_len)
        "truth_0.7m_nofootprint": (s, true, dx, 0.0, 0.0),
        "atl03_like_0.7m_11m_footprint_+0.10m_noise":
            (s, fp + 0.10 * rng.standard_normal(n), dx, FOOTPRINT_SIG_M, 0.0),
        "atl06_like_20m_11m_footprint_40m_fit_+0.03m_noise":
            (s[::28], seg[::28] + 0.03 * rng.standard_normal(len(s[::28])), 20.0, FOOTPRINT_SIG_M, 40.0),
    }
    print(f"synthetic Gaussian-ACF surface: sigma={sigma} m, l={l} m, {args.length_km} km")
    report = {}
    for name, (ss, hh, d, fs, sl) in cases.items():
        res = analyse_profile(ss, hh, d, args.detrend_m, args.max_lag_m, args.bands, fs, sl)
        report[name] = {k: v for k, v in res.items() if not k.startswith("_")}
        _print(name, res)
    # photon-spread proxy: sub-footprint sigma from within-shot variance
    n_ph = 8
    win = footprint_smooth(true, dx)
    sub = true - win
    ph = sub[:, None] + 0.10 * rng.standard_normal((n, n_ph))
    var_ph = np.sum((ph - ph.mean(axis=1, keepdims=True)) ** 2) / (n * (n_ph - 1))
    print(f"\nwithin-shot photon spread: sqrt(var - noise^2) = "
          f"{np.sqrt(max(var_ph - 0.10**2, 0)):.3f} m  (true sub-footprint rms {sub.std():.3f} m; "
          f"needs instrument noise known to <~1 cm)")
    return report


def _print(name, res):
    if res is None:
        print(f"{name}: no usable chunks"); return
    print(f"\n[{name}]  n={res['n_points']}  {res['length_km']:.1f} km  rms_total={res['rms_total_m']:.3f} m")
    print("  band rms:", {k: round(v, 3) for k, v in res["band_rms_m"].items()})
    for kind in ("acf_fit", "psd_fit"):
        print(f"  {kind}:")
        for m, v in res[kind].items():
            print(f"    {m:12s} {json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})}")


def _plot(results, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, res in results.items():
        if res is None:
            continue
        r, c = res["_acf"]; k, p = res["_psd"]
        ax[0].plot(r, c / c[0], label=name)
        ax[1].loglog(2 * np.pi / k, p, label=name)
    ax[0].set(xlabel="lag (m)", ylabel="normalised ACF", xscale="log")
    ax[1].set(xlabel="wavelength (m)", ylabel="W(k) (m^3)")
    ax[1].axvspan(0.5, 5, color="0.85", label="radar Bragg band 60-400 MHz")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atl06"); ap.add_argument("--atl03")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LONMIN", "LATMIN", "LONMAX", "LATMAX"))
    ap.add_argument("--beams", nargs="*")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sigma", type=float, default=0.05); ap.add_argument("--l", type=float, default=3.0)
    ap.add_argument("--length-km", type=float, default=20.0)
    ap.add_argument("--detrend-m", type=float, default=1000.0)
    ap.add_argument("--max-lag-m", type=float, default=200.0)
    ap.add_argument("--bands", nargs="*", type=float, default=[1.4, 5, 20, 60, 100])
    ap.add_argument("--out", default="outputs/icesat2")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.synthetic:
        rep = run_synthetic(args)
        (out / "synthetic_report.json").write_text(json.dumps(rep, indent=1))
        return
    if args.atl06:
        for gt, d in read_atl06(args.atl06, args.bbox, args.beams).items():
            edges = [e for e in args.bands if e >= 3 * d["dx"]] or [60, 100]
            res = analyse_profile(d["s"], d["h"], d["dx"], args.detrend_m, args.max_lag_m, [40] + edges,
                                  FOOTPRINT_SIG_M, 40.0)
            if res:
                res["h_robust_sprd_med_m"] = float(np.median(d["sprd"]))
                res["h_li_sigma_med_m"] = float(np.median(d["h_sigma"]))
                res["dh_fit_dx_rms"] = float(np.sqrt(np.mean(d["dhdx"] ** 2)))
                res["strong_beam"] = d["strong"]
                print(f"\n== ATL06 {gt} strong={d['strong']} h_robust_sprd med {res['h_robust_sprd_med_m']:.3f} m "
                      f"h_li_sigma med {res['h_li_sigma_med_m']:.3f} m  slope rms {res['dh_fit_dx_rms']:.4f}")
            _print(f"ATL06 {gt}", res); results[f"ATL06 {gt}"] = res
    if args.atl03:
        for gt, d in read_atl03(args.atl03, args.bbox, args.beams).items():
            res = analyse_profile(d["s"], d["h"], d["dx"], args.detrend_m, args.max_lag_m, args.bands,
                                  FOOTPRINT_SIG_M, 0.0)
            if res:
                rs = d["within_shot_resid"]
                res["within_shot_rms_m"] = float(rs.std()) if len(rs) else None
                res["ph_per_shot"] = d["ph_per_shot"]
                print(f"\n== ATL03 {gt} strong={d['strong']} photons/shot {d['ph_per_shot']:.1f} "
                      f"within-shot rms {res['within_shot_rms_m']:.3f} m (instrument noise included)")
            _print(f"ATL03 {gt}", res); results[f"ATL03 {gt}"] = res
    if results:
        tag = Path(args.atl06 or args.atl03).stem
        (out / f"{tag}_roughness.json").write_text(json.dumps(
            {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items() if v}, indent=1))
        _plot(results, out / f"{tag}_roughness.png")


if __name__ == "__main__":
    main()
