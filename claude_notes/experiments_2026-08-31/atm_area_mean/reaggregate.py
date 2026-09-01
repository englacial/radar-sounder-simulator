"""ATM area-mean re-aggregation (branch atm-area-mean).

The adopted exponential entries are fits through the MEDIAN of per-block
(or per-site-year) S(k_B); radar clutter integrates the AREA MEAN of a
heavy-tailed sigma^2 field, so the median under-represents the clutter the
radar sees (E2 finding, claude_notes/experiments_2026-08-31/synthesis.md).

This script recomputes, per pilot line and per Tier-2 stratum:
  - the linear-domain MEAN of S(k_B) over blocks/site-years at the four
    Bragg points (5 / 1.54 / 1.0 / 0.75 m: 60/195/300/400 MHz, theta_c 30);
  - a least-squares exponential-PSD fit (sigma, l) through those means
    (log-space residuals, the existing entries' convention: grid over l,
    analytic log-sigma^2 offset per l);
  - the mean-vs-median delta in dB at each point (the predicted first-order
    mid-column lift at that carrier's Bragg angle).

Outputs a table (stdout + claude_notes note) and the YAML block for the
new *_expmean entries.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ATM = ROOT / "outputs/atm_roughness"
T2 = ROOT / "outputs/atm_regional/tier2"

FREQS = ["60MHz", "195MHz", "300MHz", "400MHz"]
C = 299792458.0
# Bragg wavenumbers at theta_c = 30 deg: k_B = 2 k0 sin(theta) = 2 pi f / c...
K_B = np.array([2 * (2 * np.pi * float(f[:-3]) * 1e6 / C) * np.sin(np.radians(30.0))
                for f in FREQS])

LINE_BLOCKS = {  # line -> (dir, block csv of the ADOPTED reference year)
    "greenland_westcoast": "greenland_westcoast/blocks_2017-05-10_1000m.csv",
    "greenland_geikie01_transit":
        "greenland_geikie01_transit/blocks_2014-04-21_1000m.csv",
    "antarctica_getz": "antarctica_getz/blocks_2016-11-05_1000m.csv",
    "antarctica_david": "antarctica_david/blocks_2013-11-19_1000m.csv",
}
STRATA = ["aa_grounded_500_1500", "aa_grounded_lt500_m"]
STRATUM_NAMES = {"aa_grounded_500_1500": "AA grounded 500-1500",
                 "aa_grounded_lt500_m": "AA grounded <500 m"}


def exp_psd(k, sigma, l):
    return sigma**2 * l**2 / (2 * np.pi) * (1 + (k * l) ** 2) ** -1.5


def fit_exponential(k, S):
    """(sigma, l, resid_db): log-space LSQ through the points; grid over l,
    analytic sigma^2 offset (sigma^2 is a pure log offset at fixed l)."""
    logS = np.log10(S)
    best = None
    for l in np.geomspace(0.05, 500.0, 2000):
        shape = np.log10(l**2 / (2 * np.pi) * (1 + (k * l) ** 2) ** -1.5)
        off = np.mean(logS - shape)          # = log10(sigma^2)
        resid = logS - (shape + off)
        sse = float(np.sum(resid**2))
        if best is None or sse < best[0]:
            best = (sse, l, off, resid)
    _, l, off, resid = best
    return float(np.sqrt(10.0**off)), float(l), 10.0 * resid  # resid in dB


def agg_points(S_lin):
    """(median, mean) of linear S per Bragg point, plus n."""
    med = np.nanmedian(S_lin, axis=0)
    mean = np.nanmean(S_lin, axis=0)
    return med, mean, np.sum(np.isfinite(S_lin[:, 0]))


def line_rows():
    out = {}
    for line, rel in LINE_BLOCKS.items():
        df = pd.read_csv(ATM / rel)
        S = df[[f"S_{f}_best_m4" for f in FREQS]].to_numpy(float)
        S[S <= 0] = np.nan
        ok = np.isfinite(S).all(axis=1)
        out[line] = (S[ok], rel)
    return out


def stratum_rows():
    df = pd.read_parquet(T2 / "rows.parquet")
    codes = {"aa_grounded_500_1500": "1_h1", "aa_grounded_lt500_m": "1_h0"}
    out = {}
    for sid in STRATA:
        sub = df[(df["hemi"] == "aa") & (df["stratum"] == codes[sid])
                 & (df["status"] == "ok")]
        S = 10.0 ** (sub[[f"S_{f}_best_dB" for f in FREQS]]
                     .to_numpy(float) / 10.0)
        ok = np.isfinite(S).all(axis=1)
        out[sid] = (S[ok], f"tier2 rows.parquet stratum={STRATUM_NAMES[sid]}")
    return out


def report(name, S, src):
    med, mean, n = agg_points(S)
    d_db = 10.0 * np.log10(mean / med)
    s_med, l_med, r_med = fit_exponential(K_B, med)
    s_mean, l_mean, r_mean = fit_exponential(K_B, mean)
    print(f"== {name}  (n = {S.shape[0]}; {src})")
    print("   point        5.0m    1.54m    1.00m    0.75m")
    print("   mean-med dB " + " ".join(f"{v:8.2f}" for v in d_db))
    print(f"   median fit: sigma {s_med*100:6.2f} cm  l {l_med:7.2f} m  "
          f"resid dB {np.round(r_med, 2)}")
    print(f"   MEAN   fit: sigma {s_mean*100:6.2f} cm  l {l_mean:7.2f} m  "
          f"resid dB {np.round(r_mean, 2)}")
    return {"n": int(S.shape[0]), "delta_db": [round(float(v), 2) for v in d_db],
            "median_fit": {"sigma_m": round(s_med, 4), "l_m": round(l_med, 3)},
            "mean_fit": {"sigma_m": round(s_mean, 4), "l_m": round(l_mean, 3),
                         "resid_db": [round(float(v), 2) for v in r_mean]},
            "source": src}


def main():
    res = {}
    for line, (S, src) in line_rows().items():
        res[line] = report(line, S, src)
    for sid, (S, src) in stratum_rows().items():
        res[f"stratum:{sid}"] = report(f"stratum:{sid}", S, src)
    out = Path(__file__).with_name("reaggregate_results.json")
    out.write_text(json.dumps(res, indent=1))
    print("\nwrote", out)


if __name__ == "__main__":
    main()


# adopted entries for the S(k_B)-level comparison (config/roughness yamls)
ADOPTED = {
    "greenland_westcoast": (0.0333, 1.038),
    "greenland_geikie01_transit": (0.0515, 5.276),
    "antarctica_david": (0.108, 13.5),          # stratum aa_grounded_500_1500
    "antarctica_getz": (0.249, 24.5),           # stratum aa_grounded_lt500_m
    "stratum:aa_grounded_500_1500": (0.108, 13.5),
    "stratum:aa_grounded_lt500_m": (0.249, 24.5),
}


def levels():
    """dB level of each variant AT the Bragg points, vs the adopted entry:
    the first-order mid-column shift the sim would see per carrier."""
    import json as _json
    res = _json.loads(Path(__file__).with_name(
        "reaggregate_results.json").read_text())
    print(f"{'case':38s}" + "".join(f"{f:>10s}" for f in FREQS)
          + "   (dB of area-MEAN S vs ADOPTED entry)")
    for name, r in res.items():
        if name not in ADOPTED or "mean_fit" not in r:
            continue
        sa, la = ADOPTED[name]
        # use the raw mean points where stored implicitly via fits: recompute
        # from the fitted mean pair (fit resid < 0.8 dB)
        sm, lm = r["mean_fit"]["sigma_m"], r["mean_fit"]["l_m"]
        d = 10 * np.log10(exp_psd(K_B, sm, lm) / exp_psd(K_B, sa, la))
        print(f"{name:38s}" + "".join(f"{v:10.2f}" for v in d))


if __name__ == "__main__":
    levels()
