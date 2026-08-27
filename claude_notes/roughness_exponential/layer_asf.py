"""Analytic angular scattering function of the C&S 2020 Fig. 11 firn layers
(validation c, Doppler-spectrum shape) -- Gaussian vs exponential ACF.

C&S compare Doppler spectra (normalized to peak, in-firn illumination angle
within +-7 deg) against their S-IEM model: peak = the specular disc
sigma0_c = 2 pi^2 (h + z/n_f) |Gamma|^2 / lambda_f exp(-(2 k_f sigma)^2)
(their Eq. 7, RCS per unit Fresnel-disc area) and the off-nadir decay =
sigma0_s(theta) (Eq. 5 with the exponential W^n of Eq. 6). The simulator's
per-facet incoherent law (area-only D_Phi) is the scalar-PO Kirchhoff form
sigma0 = (k_f^2/pi) cos^2(theta) |Gamma|^2 e^{-x} sum_m x^m/m! W_m(2 k_f
sin theta), x = (2 k_f sigma cos theta)^2 -- the Kirchhoff term of C&S Eq. 5
to cos^4(theta) (0.13 dB at 7 deg). This script tabulates
sigma0_s(theta)/sigma0_c for both ACFs at four depths of the MCoRDS3
inversion profile (195 MHz, h = 470 m, Kovacs n_f from the B26 core), i.e.
the Doppler-spectrum shape each ACF predicts, next to the C&S observation
(>= 25 dB below peak within 7 deg, MCoRDS3).

Writes outputs/roughness_exponential/{layer_asf.json, fig_layer_asf.png}.
"""
import json
import sys
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import run_b26_comparison as b26  # noqa: E402
import run_firn_investigation as rfi  # noqa: E402
from soundersim import roughness as rg  # noqa: E402

OUT = ROOT / "outputs" / "roughness_exponential"
OUT.mkdir(parents=True, exist_ok=True)
C = 299792458.0
F0, H = 195e6, 470.0
DEPTHS = (10.0, 30.0, 55.0, 80.0)
TH = np.linspace(0.0, 7.0, 71)          # in-firn illumination angle, deg


def sigma0_inc(acf, sig, l, kf, th):
    nt = rg.n_terms_for((2 * kf * sig) ** 2)
    with jax.enable_x64():
        dp = np.array(rg.d_phi(sig, l, 2 * kf * np.cos(th), 2 * kf * np.sin(th),
                               0 * th, 1.0, 1.0, n_terms=nt, area_only=True,
                               acf=acf))
    return (kf ** 2 / np.pi) * np.cos(th) ** 2 * dp     # |Gamma|^2 = 1


def main():
    th = np.deg2rad(TH)
    out = {"f0_hz": F0, "h_m": H, "theta_deg": TH.tolist(), "layers": {}}
    fig, axes = plt.subplots(1, len(DEPTHS), figsize=(15, 4), sharey=True)
    for ax, z in zip(axes, DEPTHS):
        sig, l = (float(v[0]) for v in b26.layer_roughness(np.array([z]), "mcords"))
        nf = float(np.sqrt(rfi.point_eps(z)))
        kf = 2 * np.pi * F0 / C * nf
        lam_f = C / F0 / nf
        s0c = 2 * np.pi ** 2 * (H + z / nf) / lam_f * np.exp(-(2 * kf * sig) ** 2)
        rec = {"sigma_m": sig, "l_m": l, "n_f": nf, "k_f_sigma": kf * sig,
               "sigma0_c_db": 10 * np.log10(s0c), "rel_db": {}}
        for acf in ("gaussian", "exponential"):
            s0 = sigma0_inc(acf, sig, l, kf, th)
            rel = 10 * np.log10(s0 / s0c)
            rec["rel_db"][acf] = {f"{a:g}deg": float(np.interp(a, TH, rel))
                                  for a in (0, 1, 2, 4, 7)}
            rec["rel_db"][acf]["curve"] = np.round(rel, 2).tolist()
            ax.plot(TH, rel, label=acf)
            ax.plot(-TH, rel, color=ax.lines[-1].get_color())
        ax.axhline(-25, color="k", ls=":", lw=0.8, label="C&S: <= -25 dB within 7 deg")
        ax.set_title(f"z = {z:.0f} m: sigma {100*sig:.1f} cm, L {l:.2f} m, k_f sigma {kf*sig:.2f}")
        ax.set_xlabel("in-firn illumination angle (deg)")
        ax.grid(alpha=0.3)
        ax.set_ylim(-80, 5)
        out["layers"][f"{z:g}m"] = rec
    axes[0].set_ylabel("sigma0_s(theta) / sigma0_c  (dB rel specular peak)")
    axes[0].legend(fontsize=8)
    fig.suptitle("C&S 2020 Fig. 11 (MCoRDS3) layer roughness: incoherent angular scattering "
                 "function rel. the specular disc (their Eq. 7), 195 MHz, h = 470 m")
    fig.tight_layout()
    fig.savefig(OUT / "fig_layer_asf.png", dpi=130)
    (OUT / "layer_asf.json").write_text(json.dumps(out, indent=1) + "\n")
    for z, rec in out["layers"].items():
        print(f"{z}: sigma {100*rec['sigma_m']:.1f} cm L {rec['l_m']:.2f} m k_f sigma {rec['k_f_sigma']:.2f}")
        for acf, v in rec["rel_db"].items():
            print(f"   {acf:12s}", "  ".join(f"{a}: {v[a]:+6.1f}" for a in ("0deg", "1deg", "2deg", "4deg", "7deg")))


if __name__ == "__main__":
    main()
