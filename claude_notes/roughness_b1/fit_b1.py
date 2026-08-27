"""Build config/roughness/atm_b1.yaml from the ATM roughness results and
tabulate the effective Gaussian pairs, the matching-rule comparison and the
residual vs the measured spectrum.

    uv run python claude_notes/roughness_b1/fit_b1.py

Inputs: claude_notes/roughness_b1/atm_inputs/ (copied from the ATM study,
outputs/atm_roughness). Outputs: config/roughness/atm_b1.yaml,
outputs/roughness_b1/{b1_table.md,residuals.png,rules.md}.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import surface_roughness_b1 as b1  # noqa: E402

IN = ROOT / "claude_notes/roughness_b1/atm_inputs"
OUT = ROOT / "outputs/roughness_b1"
OUT.mkdir(parents=True, exist_ok=True)
FIX = (0.049474, 2.982179)
BRAGG_LAM = {"60MHz": 5.0, "195MHz": 1.5, "300MHz": 1.0, "400MHz": 0.75}
FREQS = [60e6, 150e6, 195e6, 225e6, 300e6, 400e6]
THETAS = [20.0, 30.0, 40.0]


def median_powerlaw(summ):
    """Power law through the line-median per-block S(k_B) at the 4 Bragg wavelengths."""
    k = np.array([2 * np.pi / BRAGG_LAM[n] for n in BRAGG_LAM])
    s = np.array([summ["bragg"][n]["best_db_q"][1] for n in BRAGG_LAM])
    slope, icpt = np.polyfit(np.log10(k), s, 1)
    return {"family": "powerlaw", "A_m4": float(10 ** (icpt / 10)), "beta": float(-slope / 10)}


def csv_block_median(path, s0, s1):
    rows = [r for r in csv.DictReader(open(path)) if s0 <= float(r["s0_km"]) < s1]
    out = {}
    for n in BRAGG_LAM:
        v = np.array([float(r[f"S_{n}_best_m4"]) for r in rows if r[f"S_{n}_best_m4"] not in ("", "nan")])
        out[n] = float(10 * np.log10(np.median(v)))
    return out, len(rows)


def db(x):
    return 10 * np.log10(x)


def rules(S, k_b, lam_band=(1.0, 5.0)):
    """(sigma, l) under the tangent, variance(0.5-30 m) and two-k rules + RMS dB residual over lam_band."""
    lam = np.geomspace(lam_band[0], lam_band[1], 40)
    kk = 2 * np.pi / lam
    out = {}

    # second residual band: the Bragg wavelengths of theta_c = 20..40 deg
    lam_th = np.geomspace(2 * np.pi / k_b * np.sin(np.radians(30)) / np.sin(np.radians(40)),
                          2 * np.pi / k_b * np.sin(np.radians(30)) / np.sin(np.radians(20)), 40)
    kth = 2 * np.pi / lam_th

    def resid(sig, l):
        r15 = np.sqrt(np.mean((db(b1.gaussian_psd(kk, sig, l)) - db(S(kk))) ** 2))
        rth = np.sqrt(np.mean((db(b1.gaussian_psd(kth, sig, l)) - db(S(kth))) ** 2))
        return float(r15), float(rth)
    sig, l = b1.tangent_pair(S, k_b)
    out["tangent"] = (sig, l, *resid(sig, l))
    # band variance 0.5-30 m
    kg = np.geomspace(2 * np.pi / 30, 2 * np.pi / 0.5, 4000)
    var = float(np.trapezoid(S(kg) * 2 * np.pi * kg, kg))
    target = np.log(S(k_b))

    def f(l):
        return np.log(var * l ** 2 / (4 * np.pi)) - k_b ** 2 * l ** 2 / 4 - target
    ls = np.geomspace(1e-3, 1e3, 20000)
    fv = f(ls)
    roots = ls[:-1][np.sign(fv[:-1]) != np.sign(fv[1:])]
    if len(roots):
        best = min(((resid(np.sqrt(var), r), r) for r in roots))
        out["variance"] = (float(np.sqrt(var)), float(best[1]), *best[0])
    else:
        out["variance"] = (float(np.sqrt(var)), float("nan"), float("nan"), float("nan"))
    # two wavenumbers: k_b and the 60 MHz Bragg (5 m), or 1.5 m when k_b is itself at 5 m
    k2 = 2 * np.pi / (5.0 if abs(2 * np.pi / k_b - 5.0) > 0.1 else 1.5)
    l2 = 4 * (np.log(S(k2)) - np.log(S(k_b))) / (k_b ** 2 - k2 ** 2)
    if l2 > 0:
        l = np.sqrt(l2)
        sig = np.sqrt(4 * np.pi * S(k_b) * np.exp(k_b ** 2 * l ** 2 / 4) / l ** 2)
        out["two_k"] = (float(sig), float(l), *resid(sig, l))
    else:
        out["two_k"] = (float("nan"),) * 4
    return out


def main():
    wc = {y: json.load(open(IN / f"summary_{d}.json")) for y, d in
          [("2016", "2016-05-11"), ("2017", "2017-05-10"), ("2019", "2019-05-14")]}
    gk = json.load(open(IN / "summary_2014-04-21.json"))
    gz = json.load(open(IN / "summary_2016-11-05.json"))
    spectra = {}
    for y, s in wc.items():
        spectra[f"westcoast_{y}"] = {**median_powerlaw(s),
                                     "provenance": f"ILATM1B {s['date']}, line-median per-block best-family S(k_B) at 5/1.5/1.0/0.75 m, power law through the medians; pooled fit H={s['pooled_fit']['powerlaw']['H']:.2f}"}
    e = gk["q1"]
    spectra["geikie_2014"] = {"family": "exponential", "sigma_m": float(e["exp_sigma_q"][1]), "l_m": float(e["exp_l_q"][1]),
                              "provenance": "ILATM1B 2014-04-21, exponential ACF best in 83 % of 1 km blocks (white residuals 79 %); block-median sigma, l"}
    spectra["getz_2016"] = {**median_powerlaw(gz),
                            "provenance": f"ILATM1B 2016-11-05 (low pass), LINE-median per-block S(k_B); pooled fit is 3-5 dB higher (H={gz['pooled_fit']['powerlaw']['H']:.2f}); 3 regimes along s, see note"}
    pil, npil = csv_block_median(IN / "blocks_2016-11-05_1000m.csv", 30.0, 40.0)
    spectra["getz_2016_pilot_window"] = {"family": "powerlaw", "provenance": f"blocks s 30-40 km only ({npil} blocks; straddles the 32 km regime boundary) -- sensitivity, not used by default"}
    k = np.array([2 * np.pi / BRAGG_LAM[n] for n in BRAGG_LAM]); s = np.array([pil[n] for n in BRAGG_LAM])
    sl, ic = np.polyfit(np.log10(k), s, 1)
    spectra["getz_2016_pilot_window"].update({"A_m4": float(10 ** (ic / 10)), "beta": float(-sl / 10)})
    for v in spectra.values():
        for kk in ("A_m4", "beta", "sigma_m", "l_m"):
            if kk in v:
                v[kk] = float(f"{v[kk]:.5g}")
    table = {
        "schema_version": 1,
        "description": "Path B1 effective-Gaussian surface roughness: measured 2-D surface PSDs per line (OIB ATM L1B, claude_notes/atm_roughness), matched by tools/surface_roughness_b1.py at the Bragg wavenumber of the clutter angle. PSD convention: int S d2k = sigma^2; Gaussian S = sigma^2 l^2/(4 pi) exp(-k^2 l^2/4); power law S = A k^-beta; exponential S = sigma^2 l^2/(2 pi)(1+k^2 l^2)^-1.5.",
        "theta_c_deg": 30.0,
        "rule": "tangent",
        "spectra": spectra,
        "lines": {
            "greenland_westcoast": {"default": "westcoast_2017",
                                    "passes": {"p3_2016": "westcoast_2016", "p3_2017": "westcoast_2017", "p3_2019": "westcoast_2019"}},
            "greenland_geikie01_transit": {"default": "geikie_2014"},
            "antarctica_getz": {"default": "getz_2016"},
            "antarctica_david": {"default": "getz_2016", "note": "no ATM on the line; Getz transferred (same season/ice type argument, plan caveat)"},
        },
    }
    cfg = ROOT / "config/roughness/atm_b1.yaml"
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(yaml.safe_dump(table, sort_keys=False, width=100))
    print("wrote", cfg)

    # ---- tables
    lines = ["# B1 effective Gaussian pairs (tangent rule at theta_c)\n",
             "| spectrum | f (MHz) | Lambda_B (m) | S_meas(k_B) dB | theta 20: sigma cm / l m | theta 30: sigma cm / l m | theta 40: sigma cm / l m |", "|---|---|---|---|---|---|---|"]
    for sid, sp in spectra.items():
        S = b1.spectrum(sp)
        for f in FREQS:
            cells = []
            for th in THETAS:
                kb = b1.bragg_k(f, th)
                sg, l = b1.tangent_pair(S, kb)
                cells.append(f"{sg * 100:.2f} / {l:.3f}")
            kb30 = b1.bragg_k(f, 30.0)
            lines.append(f"| {sid} | {f / 1e6:.0f} | {2 * np.pi / kb30:.2f} | {db(S(kb30)):.1f} | " + " | ".join(cells) + " |")
    lines += ["", "# Matching-rule comparison at theta_c = 30: RMS dB residual of S_G vs S_meas over Lambda 1-5 m / over the theta_c = 20-40 deg Bragg band\n",
              "| spectrum | f (MHz) | tangent: sigma cm / l m / rms(1-5 m) / rms(20-40 deg) | variance(0.5-30 m): sigma / l / rms / rms | two-k (k_B & 5 m): sigma / l / rms / rms |", "|---|---|---|---|---|"]
    for sid, sp in spectra.items():
        S = b1.spectrum(sp)
        for f in FREQS:
            r = rules(S, b1.bragg_k(f, 30.0))
            lines.append(f"| {sid} | {f / 1e6:.0f} | " + " | ".join(
                f"{r[n][0] * 100:.2f} / {r[n][1]:.3f} / {r[n][2]:.1f} / {r[n][3]:.1f}" for n in ("tangent", "variance", "two_k")) + " |")
    # residual table over 0.75-30 m at theta 30, main spectra
    lam = np.array([0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0])
    lines += ["", "# Residual 10log10(S_G/S_meas) dB at theta_c = 30 over Lambda = 0.75-30 m\n",
              "| spectrum | f (MHz) | " + " | ".join(f"{v:g} m" for v in lam) + " |", "|---|---|" + "---|" * len(lam)]
    for sid, sp in spectra.items():
        S = b1.spectrum(sp)
        for f in FREQS:
            sg, l = b1.tangent_pair(S, b1.bragg_k(f, 30.0))
            res = db(b1.gaussian_psd(2 * np.pi / lam, sg, l)) - db(S(2 * np.pi / lam))
            lines.append(f"| {sid} | {f / 1e6:.0f} | " + " | ".join(f"{v:+.1f}" for v in res) + " |")
        res = db(b1.gaussian_psd(2 * np.pi / lam, *FIX)) - db(S(2 * np.pi / lam))
        lines.append(f"| {sid} | fixture | " + " | ".join(f"{v:+.0f}" for v in res) + " |")
    # band RMS of the measured spectra: sub-facet band (0.5-30 m) vs the wider 0.5-100 m
    lines += ["", "# Measured band RMS height (cm): the sub-facet band 0.5-30 m (used by the variance rule) vs 0.5-100 m (includes DEM-carried facet tilt; NOT used)\n",
              "| spectrum | sigma 0.5-30 m | sigma 0.5-100 m | ratio dB |", "|---|---|---|---|"]
    for sid, sp in spectra.items():
        S = b1.spectrum(sp)
        v = {}
        for lo in (30.0, 100.0):
            kg = np.geomspace(2 * np.pi / lo, 2 * np.pi / 0.5, 4000)
            v[lo] = float(np.trapezoid(S(kg) * 2 * np.pi * kg, kg))
        lines.append(f"| {sid} | {100 * np.sqrt(v[30.0]):.2f} | {100 * np.sqrt(v[100.0]):.2f} | {db(v[100.0] / v[30.0]):+.1f} |")
    (OUT / "b1_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- residual plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main_ids = ["westcoast_2017", "westcoast_2016", "westcoast_2019", "geikie_2014", "getz_2016"]
    fig, axs = plt.subplots(2, len(main_ids), figsize=(4 * len(main_ids), 7), sharex=True)
    lam = np.geomspace(0.5, 40, 300); kk = 2 * np.pi / lam
    for j, sid in enumerate(main_ids):
        S = b1.spectrum(spectra[sid])
        axs[0, j].plot(lam, db(S(kk)), "k", lw=2, label="measured (ATM)")
        axs[0, j].plot(lam, db(b1.gaussian_psd(kk, *FIX)), "k--", label="fixture 4.9 cm / 2.98 m")
        axs[1, j].axhline(0, color="k", lw=0.8)
        for f, c in zip(FREQS, plt.cm.viridis(np.linspace(0, 0.9, len(FREQS)))):
            sg, l = b1.tangent_pair(S, b1.bragg_k(f, 30.0))
            g = db(b1.gaussian_psd(kk, sg, l))
            axs[0, j].plot(lam, g, color=c, label=f"B1 {f / 1e6:.0f} MHz ({sg * 100:.1f} cm, {l:.2f} m)")
            axs[1, j].plot(lam, g - db(S(kk)), color=c)
            axs[1, j].axvline(2 * np.pi / b1.bragg_k(f, 30.0), color=c, ls=":", lw=0.8)
        axs[0, j].set(xscale="log", ylim=(-100, -20), title=sid)
        axs[1, j].set(xscale="log", ylim=(-30, 10), xlabel="surface wavelength (m)")
        axs[0, j].legend(fontsize=6)
    axs[0, 0].set_ylabel("2-D PSD S(k), dB re m^4"); axs[1, 0].set_ylabel("B1 - measured, dB")
    fig.suptitle("B1 effective Gaussian vs measured surface PSD (theta_c = 30 deg; dotted = Bragg wavelength)")
    fig.tight_layout(); fig.savefig(OUT / "residuals.png", dpi=110)
    print("wrote", OUT / "residuals.png")


if __name__ == "__main__":
    main()
