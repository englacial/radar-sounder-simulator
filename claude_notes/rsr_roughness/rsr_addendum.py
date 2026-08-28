"""Addendum: is the surface-peak spread speckle or along-track mean variation?
Detrend log power with a running median (window w traces), look at residual CV
and its implied incoherent fraction for a nominal N-look product; roll/pitch
correlation; ATM SPM prediction per block."""
import json, numpy as np
from pathlib import Path
from scipy import ndimage, stats
import sys; sys.path.insert(0, str(Path(__file__).parent))
from rsr_prototype import surface_peaks_standard, surface_peaks_qlook, fit_nlook, sigma_from_ratio, K, OUT

std = surface_peaks_standard(); P = std["P"]; s = std["s"]
lp = 10*np.log10(P)
for w in (5, 11, 21, 41, 81):
    trend = ndimage.median_filter(lp, w, mode="nearest")
    res = 10**((lp-trend)/10)
    cv = res.std()/res.mean()
    # N-look Rice: CV^2 = f(2-f)/N with f = Pn/(Pc+Pn)  -> f = 1 - sqrt(1 - N CV^2)
    out = {f"N{N}": (1-np.sqrt(max(1-N*cv**2, 0))) for N in (6, 11)}
    print(f"detrend {w:3d} traces ({w*14.9:5.0f} m): residual CV {cv:.3f}, std(trend) {trend.std():.2f} dB, "
          f"incoherent fraction if N=6: {out['N6']:.2f}, N=11: {out['N11']:.2f}")
# spatial spectrum of the trend: how much of the log-power variance is at > 150 m scales
print("total std log-power dB", lp.std())
print("corr(|roll|, lp)", np.corrcoef(np.abs(std["roll"]), lp)[0,1], " corr(|pitch|, lp)", np.corrcoef(np.abs(std["pitch"]), lp)[0,1],
      " roll deg p5/50/95", np.percentile(np.degrees(std["roll"]), [5,50,95]))
# autocorrelation of log power vs lag in traces
x = lp-lp.mean(); print("acf lags 1..10:", [round(float(np.corrcoef(x[:-l], x[l:])[0,1]),2) for l in range(1,11)])
# expected acf for pure 11-line boxcar averaging decimated by 6: overlap (11-6)/11 at lag 1, 0 beyond
print("pure-speckle acf from 11-line/6-decim overlap: lag1", 5/11, "lag2", 0)
# qlook 10-look: residual CV after detrending over 5 traces (870 m)
ql = surface_peaks_qlook(); lq = 10*np.log10(ql["P"])
for w in (3,5,9):
    tr = ndimage.median_filter(lq, w, mode="nearest"); r = 10**((lq-tr)/10); cv=r.std()/r.mean()
    print(f"qlook detrend {w} ({w*174:.0f} m): residual CV {cv:.3f}, incoherent fraction if N=10: {1-np.sqrt(max(1-10*cv**2,0)):.2f}; pure-speckle-all-incoherent CV would be {1/np.sqrt(10):.3f}")
# ATM SPM prediction summary
d = json.load(open(OUT/"rsr_prototype_20170510_03_013.json"))
a = d["atm"]
print("ATM blocks n", len(a), "s range", a[0]["s0"], a[-1]["s1"])
for key in ("sigma_5_64_cm","sigma_5_32_cm","sigma_1_64_cm","sigma_pl_band_cm","sigma_gauss_band_cm","ratio_pred_dB","g_sigma_cm"):
    v = np.array([r[key] for r in a], float); v=v[np.isfinite(v)]
    print(f"  {key}: median {np.median(v):.2f} p5 {np.percentile(v,5):.2f} p95 {np.percentile(v,95):.2f}")
sel=[r for r in a if 38<=r["s0"]<55]
print("blocks near pilot:", [(r["s0"], round(r["sigma_5_64_cm"],1), round(r["sigma_pl_band_cm"],1), round(r["ratio_pred_dB"],1)) for r in sel])
print("2k*sigma for sigma 8, 13, 18 cm:", [round(2*K*x,2) for x in (0.08,0.13,0.18)])
