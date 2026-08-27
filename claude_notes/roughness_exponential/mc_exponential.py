"""Exponential-ACF validation (a), (b) and the series checks
(claude_notes/roughness_exponential_2026-08-27.md).

(a) facet-in-isolation Monte Carlo: 4x7 lambda facet at lambda/40, correlated
    surfaces by spectral filtering with the 2-D exponential-ACF spectrum
    (1 + k^2 l^2)^-3/2 (and the Gaussian one for the control), ensemble
    <|Phi|^2> vs |<Phi>|^2 + D_Phi(area_only, acf) over sigma/lambda x
    l/lambda x {nadir, 30 deg, 50 deg}; dB error maps.
(b) Haynes 2018 nadir Fresnel disc: coherent power ACF-independent, the
    incoherent share per ACF, total vs the Haynes closed form.
(c) W_m tail: Gaussian vs exponential at the same (sigma, l) at the Bragg
    points 5 / 1.5 / 1.0 / 0.75 m, and the n_terms_for convergence table.

Writes outputs/roughness_exponential/{mc_metrics.json, fig_mc_*.png,
fig_wm_tail.png}.  Run: uv run python claude_notes/roughness_exponential/mc_exponential.py
"""
import json
import math
import sys
import time
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from soundersim import roughness as rg  # noqa: E402
from soundersim.compare import gerekos as gk  # noqa: E402
from soundersim.compare import haynes  # noqa: E402

OUT = ROOT / "outputs" / "roughness_exponential"
OUT.mkdir(parents=True, exist_ok=True)
LAM = 1.0
K = 2 * np.pi / LAM
# facet: the paper's 4x7 lambda (edge remainder O(1) vs the area term) and
# a 16x28 lambda control where the edge/area ratio is 12 dB smaller
FACETS = {"4x7": (4.0, 7.0), "16x28": (16.0, 28.0)}
LX, LY = 4.0, 7.0
DX = LAM / 40
SIGMAS = np.array([0.02, 0.05, 0.10, 0.20])
CORR = np.array([0.5, 1.0, 2.0, 4.0])
GEOMS = [("nadir", 0.0), ("oblique 30", 30.0), ("wide 50", 50.0)]
N_REAL = int(sys.argv[1]) if len(sys.argv) > 1 else 300
C = 299792458.0


def spectral_surface(n, dx, l, acf, rng):
    """(n, n) unit-variance periodic surface with the ACF's 2-D spectrum
    (exact on-grid variance by construction; the sub-Nyquist part of the
    exponential spectrum, (1 + (k_N l)^2)^-1/2 <= 1.6 % here, is dropped)."""
    kx = 2 * np.pi * np.fft.fftfreq(n, dx)
    k2 = kx[:, None] ** 2 + kx[None, :] ** 2
    h = (np.exp(-k2 * l * l / 8.0) if acf == "gaussian"
         else (1.0 + k2 * l * l) ** -0.75)          # sqrt(S)
    z = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * h))
    return z / np.sqrt(np.mean(h * h))


def analytic(acf, sigma, l, a0, b0, kk, area_only=True):
    coh = (gk.smooth_phase(a0, b0, LX, LY) * np.exp(-0.5 * (sigma * kk) ** 2)) ** 2
    nt = rg.n_terms_for((sigma * kk) ** 2)
    with jax.enable_x64():
        dp = float(rg.d_phi(sigma, l, kk, a0, b0, LX, LY, n_terms=nt,
                            area_only=area_only, acf=acf))
    return coh, dp


def mc_sweep(acf):
    global LX, LY
    nx, ny = int(round(LX / DX)), int(round(LY / DX))
    xs = (np.arange(nx) - (nx - 1) / 2.0) * DX
    ys = (np.arange(ny) - (ny - 1) / 2.0) * DX
    coeffs = [gk.facet_coeffs(np.radians(th), 0.0, K) for _, th in GEOMS]
    bases = [np.exp(1j * (a0 * xs[:, None] + b0 * ys[None, :]))
             for a0, b0, _ in coeffs]
    mc = np.zeros((len(GEOMS), len(CORR), len(SIGMAS)))
    coh = np.zeros_like(mc)
    inc = np.zeros_like(mc)
    inc_full = np.full_like(mc, np.nan)   # Gaussian exact finite-facet series
    rng = np.random.default_rng(20260827)
    for li, l in enumerate(CORR):
        n = int(2 ** np.ceil(np.log2(max(nx, ny, 12 * l / DX))))
        acc = np.zeros((len(GEOMS), len(SIGMAS)))
        for _ in range(N_REAL):
            u = spectral_surface(n, DX, l, acf, rng)[:nx, :ny]
            for gi, (a0, b0, kk) in enumerate(coeffs):
                pert = np.exp((-1j * kk) * SIGMAS[:, None, None] * u[None])
                phi = (bases[gi][None] * pert).sum(axis=(1, 2)) * DX * DX
                acc[gi] += np.abs(phi) ** 2
        mc[:, li] = acc / N_REAL
        for gi, (a0, b0, kk) in enumerate(coeffs):
            for si, s in enumerate(SIGMAS):
                coh[gi, li, si], inc[gi, li, si] = analytic(acf, s, l, a0, b0, kk)
                if acf == "gaussian":
                    inc_full[gi, li, si] = float(gk.d_phi_ref(s, l, kk, a0, b0, LX, LY))
        print(f"  {acf} l={l}: n={n} done", flush=True)
    return mc, coh, inc, inc_full


def fig_mc(path, res, ftag):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    for row, acf in enumerate(("exponential", "gaussian")):
        mc, coh, inc, incf = res[acf]
        for gi, (name, _) in enumerate(GEOMS):
            ax = axes[row, gi]
            for li, l in enumerate(CORR):
                err = 10 * np.log10(mc[gi, li] / (coh[gi, li] + inc[gi, li]))
                ax.plot(SIGMAS, err, "o-", label=f"l = {l:g} lam")
                if acf == "gaussian":
                    errf = 10 * np.log10(mc[gi, li] / (coh[gi, li] + incf[gi, li]))
                    ax.plot(SIGMAS, errf, "x:", color=ax.lines[-1].get_color())
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{acf} surfaces, {name}" + ("  (x: exact Gaussian series)"
                                                      if acf == "gaussian" else ""))
            ax.set_xlabel("sigma / lambda")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
        axes[row, 0].set_ylabel("MC <|Phi|^2> / (|<Phi>|^2 + D_Phi area-only)  [dB]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Facet {ftag} lambda, lambda/40 sampling, {N_REAL} realizations per point")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def haynes_disc():
    """Nadir rough Fresnel disc: coherent + incoherent share per ACF."""
    h, d, l, gamma = 8000.0, 4.0, 2.0, -0.281
    rf = haynes.fresnel_radius(LAM, h)
    n = int(np.ceil(2 * rf / d))
    ax = (np.arange(n) - (n - 1) / 2.0) * d
    X, Y = np.meshgrid(ax, ax)
    keep = np.hypot(X, Y).ravel() <= rf
    cx, cy = X.ravel()[keep], Y.ravel()[keep]
    r = np.sqrt(cx * cx + cy * cy + h * h)
    cos = h / r
    A0, B0, KK = -2 * K * cx / r, -2 * K * cy / r, 2 * K * cos
    f = 1j * (K / (2 * np.pi)) * gamma * cos * np.exp(-2j * K * r) / (r * r)
    rows = []
    for s in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25):
        phi_c = gk.smooth_phase(A0, B0, d, d) * np.exp(-0.5 * (s * KK) ** 2)
        coh = abs((f * phi_c).sum()) ** 2
        nt = rg.n_terms_for((2 * K * s) ** 2)
        inc = {}
        with jax.enable_x64():
            for acf in ("gaussian", "exponential"):
                dp = np.array(rg.d_phi(s, l, KK, A0, B0, d, d, n_terms=nt,
                                       area_only=True, acf=acf))
                inc[acf] = float((np.abs(f) ** 2 * dp).sum())
        inc["gaussian_full"] = gk.rough_disk_power(h, rf, d, K, gamma, s, l)[1]
        ref = haynes.mean_power(h, s, l, LAM, gamma)
        rows.append({"sigma_lam": s, "k_sigma": K * s, "coh": coh,
                     "coh_ref": gamma ** 2 / h ** 2 * np.exp(-(2 * K * s) ** 2),
                     "inc": inc, "haynes_total": ref,
                     "total_db_vs_haynes": {a: 10 * np.log10((coh + v) / ref)
                                            for a, v in inc.items()},
                     "inc_share_db": {a: 10 * np.log10(v / (coh + v))
                                      for a, v in inc.items()}})
    return rows


def wm_tail():
    """Incoherent series per unit area (sum_m w_m W_m, dB re m^2) at the four
    Bragg points, geikie (sigma, l) and a firn-layer (sigma, l), Gaussian vs
    exponential; plus the n_terms_for convergence table."""
    out = {}
    for tag, (sig, l) in {"geikie_0.0515_5.276": (0.0515, 5.276),
                          "cs_fig11_layer_0.04_3.0": (0.04, 3.0)}.items():
        rows = []
        for lam_b, f_mhz in ((5.0, 60), (1.5, 195), (1.0, 300), (0.75, 400)):
            th = np.arcsin(lam_b * f_mhz * 1e6 / C / 2)  # 2 k sin(th) = 2 pi / lam_b
            k = 2 * np.pi * f_mhz * 1e6 / C
            kb = 2 * np.pi / lam_b
            nt = rg.n_terms_for((2 * k * sig) ** 2)
            vals = {}
            with jax.enable_x64():
                for acf in ("gaussian", "exponential"):
                    dp = float(rg.d_phi(sig, l, 2 * k * np.cos(th), kb, 0.0, 1.0, 1.0,
                                        n_terms=nt, area_only=True, acf=acf))
                    vals[acf] = 10 * np.log10(dp)
            x = (2 * k * sig * np.cos(th)) ** 2
            m1 = {"gaussian": x * np.exp(-x) * np.pi * l * l * np.exp(-(kb * l) ** 2 / 4),
                  "exponential": x * np.exp(-x) * 2 * np.pi * l * l * (1 + (kb * l) ** 2) ** -1.5}
            rows.append({"bragg_m": lam_b, "f_mhz": f_mhz, "theta_deg": float(np.degrees(th)),
                         "kb_l": kb * l, "x_sigma2K2": x, "n_terms": nt,
                         "dphi_per_area_db": vals,
                         "m1_share_db": {a: 10 * np.log10(m1[a]) - vals[a] for a in vals},
                         "exp_minus_gauss_db": vals["exponential"] - vals["gaussian"]})
        out[tag] = rows
    # convergence: n_terms_for vs a 1000-term float64 sum
    conv = []
    for f_mhz in (60, 195, 300, 400):
        k = 2 * np.pi * f_mhz * 1e6 / C
        for sig in (0.01, 0.0515, 0.10, 0.20):
            for l in (0.5, 1.0, 5.0):
                th = np.deg2rad(np.array([0.0, 10.0, 30.0, 50.0, 70.0, 89.9]))
                nt = rg.n_terms_for((2 * k * sig) ** 2)
                with jax.enable_x64():
                    e = np.array(rg.d_phi(sig, l, 2 * k * np.cos(th), 2 * k * np.sin(th),
                                          0 * th, 1.0, 1.0, n_terms=nt, area_only=True,
                                          acf="exponential"))
                x = (2 * k * sig * np.cos(th)) ** 2
                kb = 2 * k * np.sin(th)
                ref = np.zeros_like(th)
                for m in range(1, 1000):
                    ref += (np.exp(m * np.log(x) - math.lgamma(m + 1) - x)
                            * 2 * np.pi * (l / m) ** 2 * (1 + (kb * l / m) ** 2) ** -1.5)
                conv.append({"f_mhz": f_mhz, "sigma_m": sig, "l_m": l, "n_terms": nt,
                             "x_max": float(x.max()), "kb_l_max": float((kb * l).max()),
                             "max_abs_err_db": float(np.max(np.abs(10 * np.log10(e / ref))))})
    return out, conv


def fig_wm(path, wm):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (tag, rows) in zip(axes, wm.items()):
        lb = [r["bragg_m"] for r in rows]
        for acf in ("gaussian", "exponential"):
            ax.plot(lb, [r["dphi_per_area_db"][acf] for r in rows], "o-", label=acf)
        ax.set_xscale("log")
        ax.set_xlabel("Bragg wavelength (m)  [5 m @60, 1.5 @195, 1 @300, 0.75 @400 MHz, theta 30 deg]")
        ax.set_ylabel("D_Phi / area  (dB re m^2)")
        ax.set_title(tag)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    global LX, LY
    t0 = time.perf_counter()
    metrics = {"n_real": N_REAL, "dx_lam": DX, "sigmas_lam": SIGMAS.tolist(),
               "corr_lam": CORR.tolist(), "geoms": GEOMS, "mc": {}}
    for ftag, (LX, LY) in FACETS.items():
        res = {}
        for acf in ("exponential", "gaussian"):
            res[acf] = mc_sweep(acf)
        fig_mc(OUT / f"fig_mc_error_map_{ftag}.png", res, ftag)
        metrics["mc"][ftag] = {}
        for acf, (mc, coh, inc, incf) in res.items():
            err = 10 * np.log10(mc / (coh + inc))
            metrics["mc"][ftag][acf] = {
                "err_db_area_only[geom][l][sigma]": np.round(err, 3).tolist(),
                "max_abs_err_db": float(np.abs(err).max()),
                "max_abs_err_db_l_le_2lam": float(np.abs(err[:, :3]).max()),
                "coh_share_db[geom][l][sigma]": np.round(10 * np.log10(coh / (coh + inc)), 2).tolist(),
            }
            if acf == "gaussian":
                errf = 10 * np.log10(mc / (coh + incf))
                metrics["mc"][ftag][acf]["err_db_exact_series[geom][l][sigma]"] = np.round(errf, 3).tolist()
                metrics["mc"][ftag][acf]["max_abs_err_db_exact_series"] = float(np.abs(errf).max())
        print(f"MC {ftag} wall", round(time.perf_counter() - t0, 1), "s", flush=True)
    metrics["haynes_disc"] = haynes_disc()
    wm, conv = wm_tail()
    metrics["wm_tail"] = wm
    metrics["convergence"] = conv
    metrics["convergence_max_abs_err_db"] = max(c["max_abs_err_db"] for c in conv)
    fig_wm(OUT / "fig_wm_tail.png", wm)
    (OUT / "mc_metrics.json").write_text(json.dumps(metrics, indent=1, default=float) + "\n")
    # console summary
    for ftag in FACETS:
        for acf in ("exponential", "gaussian"):
            m = metrics["mc"][ftag][acf]
            print(f"\n{ftag} facet, {acf} surfaces: max |err| {m['max_abs_err_db']:.2f} dB "
                  f"(l <= 2 lam: {m['max_abs_err_db_l_le_2lam']:.2f})")
            for gi, (name, _) in enumerate(GEOMS):
                print(f"  {name:10s}", " | ".join(
                    f"l={l:g}: " + " ".join(f"{v:+5.2f}" for v in m['err_db_area_only[geom][l][sigma]'][gi][li])
                    for li, l in enumerate(CORR)))
            if acf == "gaussian":
                print("  exact-series max |err|", round(m["max_abs_err_db_exact_series"], 2), "dB")
    print("\nHaynes disc (total dB vs Haynes; incoherent share dB):")
    for r in metrics["haynes_disc"]:
        print(f"  sigma {r['sigma_lam']:.2f} lam  coh vs ref {10*np.log10(r['coh']/r['coh_ref']) if r['coh_ref'] > 0 else 0:+.3f} dB  "
              + "  ".join(f"{a}: {r['total_db_vs_haynes'][a]:+.2f} / {r['inc_share_db'][a]:+.1f}"
                          for a in r["inc"]))
    print("\nW_m tail (D_Phi/area dB, gauss / exp / diff):")
    for tag, rows in wm.items():
        for r in rows:
            print(f"  {tag} {r['bragg_m']:.2f} m: {r['dphi_per_area_db']['gaussian']:.1f} / "
                  f"{r['dphi_per_area_db']['exponential']:.1f} / {r['exp_minus_gauss_db']:+.1f}  "
                  f"(m=1 share exp {r['m1_share_db']['exponential']:+.2f} dB, n_terms {r['n_terms']})")
    print("convergence max |err|", metrics["convergence_max_abs_err_db"], "dB")
    print("total wall", round(time.perf_counter() - t0, 1), "s")


if __name__ == "__main__":
    main()
