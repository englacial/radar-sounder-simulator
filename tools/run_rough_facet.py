"""Build the ``rough_facet`` verification report case (Gerekos et al. 2023).

Heavier companions to tests/test_roughness*.py, covering the full
verification plan of claude_notes/roughness_implementation_plan.md:

(a) facet-in-isolation Monte Carlo (paper Section 4.1 / Fig 4 setup): a
    4x7 lambda facet at lambda/40 sampling, 150 correlated-Gaussian-surface
    realizations per point, sweeping sigma/lambda at three correlation
    lengths and three viewing geometries (nadir, 20 deg oblique in-plane,
    35 deg off-principal-axis), ensemble <|Phi|^2> vs the analytic
    |<Phi>|^2 + D_Phi;
(c) Haynes 2018 rough-Fresnel-zone disk at nadir: the deterministic
    rough-facet ensemble power of a Fresnel-zone facet disk vs the
    compare/haynes.py closed forms (total and coherent-only);
(d) series convergence: the fixed 10-term series vs mpmath (50 digits) at
    sigma <= lambda/20, plus the n_terms_for-sized series, and the Weideman
    Faddeeva approximation vs scipy.special.wofz.

Writes outputs/verification/rough_facet/{metrics.json, fig_*.png}.
Run: uv run python tools/run_rough_facet.py   (~1 minute)
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mpmath as mp  # noqa: E402
from scipy.special import wofz  # noqa: E402

from soundersim import roughness as rg  # noqa: E402
from soundersim.compare import gerekos as gk  # noqa: E402
from soundersim.compare import haynes  # noqa: E402
from soundersim.compare.haynes import gaussian_surface  # noqa: E402
from soundersim.compare.plots import write_metrics  # noqa: E402

OUTDIR = ROOT / "outputs" / "verification" / "rough_facet"
GROUP = "Radar equation comparison"

LAM = 1.0
K_W = 2.0 * np.pi / LAM
LX, LY = 4.0 * LAM, 7.0 * LAM         # paper Section 4.1 facet
DX = LAM / 40.0                        # paper's sampling
N_REAL = 150
SIGMAS = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
CORR_LS = (0.5 * LAM, 1.0 * LAM, 2.0 * LAM)
GEOMS = [("nadir", 0.0, 0.0), ("20 deg oblique", 20.0, 0.0),
         ("35/55 deg off-axis", 35.0, 55.0)]
# fixed categorical order for the l series (colorblind-safe blue/orange/teal)
L_COLORS = ("#4053d3", "#ddb310", "#00b25d")


def mc_sweep():
    """<|Phi|^2> (MC) and analytic |<Phi>|^2 + D_Phi over the full sweep.

    Surfaces are generated once per (l, realization) and re-used across every
    sigma and geometry (the sweep is a pure rescaling z = sigma * u), exactly
    like the Haynes coherence-loss ensembles. Returns arrays indexed
    [geom, l, sigma].
    """
    nx, ny = int(round(LX / DX)), int(round(LY / DX))
    xs = (np.arange(nx) - (nx - 1) / 2.0) * DX
    ys = (np.arange(ny) - (ny - 1) / 2.0) * DX
    coeffs = [gk.facet_coeffs(np.radians(th), np.radians(ph), K_W)
              for _, th, ph in GEOMS]
    bases = [np.exp(1j * (a0 * xs[:, None] + b0 * ys[None, :]))
             for a0, b0, _ in coeffs]
    mc = np.zeros((len(GEOMS), len(CORR_LS), len(SIGMAS)))
    ana = np.zeros_like(mc)
    rng = np.random.default_rng(20230715)
    for li, l in enumerate(CORR_LS):
        n_surf = max(nx, ny, int(np.ceil(10.0 * l / DX)))
        acc = np.zeros((len(GEOMS), len(SIGMAS)))
        for _ in range(N_REAL):
            u = gaussian_surface(n_surf, DX, l, rng)[:nx, :ny]
            for gi, (a0, b0, kk) in enumerate(coeffs):
                pert = np.exp((-1j * kk) * SIGMAS[:, None, None] * u[None])
                phi = (bases[gi][None] * pert).sum(axis=(1, 2)) * DX * DX
                acc[gi] += np.abs(phi) ** 2
        mc[:, li] = acc / N_REAL
        for gi, (a0, b0, kk) in enumerate(coeffs):
            coh = (gk.smooth_phase(a0, b0, LX, LY)
                   * np.exp(-0.5 * (SIGMAS * kk) ** 2)) ** 2
            dp = np.array([float(gk.d_phi_ref(s, l, kk, a0, b0, LX, LY))
                           for s in SIGMAS])
            ana[gi, li] = coh + dp
    return mc, ana


def haynes_scan():
    """Rough Fresnel-zone facet disk at nadir vs the Haynes closed forms.

    h = 8000 lam so 4-lam facets satisfy the LPA envelope
    (0.1*sqrt(lam*h) = 8.9 lam); disk radius = the flat-surface Fresnel
    radius, l = 2 lam <= facet size. Returns (sigmas, total_sim, total_ref,
    coh_sim, coh_ref) in the soundersim |field|^2 normalization.
    """
    h, d, l, gamma = 8000.0 * LAM, 4.0 * LAM, 2.0 * LAM, -0.281
    rf = haynes.fresnel_radius(LAM, h)
    sigmas = np.array([0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25])
    tot_s, tot_r, coh_s, coh_r = [], [], [], []
    for s in sigmas:
        coh, inc = gk.rough_disk_power(h, rf, d, K_W, gamma, s, l)
        tot_s.append(coh + inc)
        coh_s.append(coh)
        tot_r.append(haynes.mean_power(h, s, l, LAM, gamma))
        coh_r.append(gamma ** 2 / h ** 2 * np.exp(-(2.0 * K_W * s) ** 2))
    return sigmas, map(np.array, (tot_s, tot_r, coh_s, coh_r))


def convergence():
    """(d): 10-term and n_terms_for-sized series vs mpmath at
    sigma <= lam/20, plus the Faddeeva approximation vs wofz."""
    mp.mp.dps = 50

    def f_mp(m, a0, edge, l):
        am = (mp.mpf(a0) * l ** 2 + 1j * 2 * edge * m) / (2 * l * mp.sqrt(m))
        x = mp.re(am)
        return (1 - mp.e ** (-(mp.mpf(edge) ** 2 * m) / l ** 2)
                * mp.cos(edge * a0) + mp.sqrt(mp.pi) * mp.e ** (-x ** 2)
                * (mp.re(am * mp.erfi(am)) - x * mp.erfi(x)))

    def d_phi_mp(sig, l, kk, a0, b0, n=60):
        x = mp.mpf(sig) ** 2 * mp.mpf(kk) ** 2
        tot = sum(x ** m / mp.factorial(m) * (mp.mpf(l) ** 4 / m ** 2)
                  * f_mp(m, a0, LX, mp.mpf(l)) * f_mp(m, b0, LY, mp.mpf(l))
                  for m in range(1, n + 1))
        return float(mp.e ** (-x) * tot)

    worst10 = worst_full = 0.0
    with jax.enable_x64():
        for l in CORR_LS:
            for sig in (LAM / 40.0, LAM / 20.0):
                for _, th, ph in GEOMS:
                    a0, b0, kk = gk.facet_coeffs(np.radians(th),
                                                 np.radians(ph), K_W)
                    ref = d_phi_mp(sig, l, kk, a0, b0)
                    ten = float(rg.d_phi(sig, l, kk, a0, b0, LX, LY,
                                         n_terms=10))
                    nt = rg.n_terms_for((sig * kk) ** 2)
                    ful = float(rg.d_phi(sig, l, kk, a0, b0, LX, LY,
                                         n_terms=nt))
                    worst10 = max(worst10, abs(ten - ref) / ref)
                    worst_full = max(worst_full, abs(ful - ref) / ref)
        rng = np.random.default_rng(3)
        z = rng.uniform(-60, 60, 3000) + 1j * 10 ** rng.uniform(-8, 2, 3000)
        z[:300] = np.real(z[:300])
        wrel = float(np.max(np.abs(np.asarray(rg.faddeeva(z)) - wofz(z))
                            / np.abs(wofz(z))))
    return worst10, worst_full, wrel


def fig_mc(path, mc, ana):
    fig, axes = plt.subplots(1, len(GEOMS), figsize=(13, 4.2), sharey=True,
                             constrained_layout=True)
    ref = (LX * LY) ** 2
    for gi, ax in enumerate(axes):
        for li, l in enumerate(CORR_LS):
            c = L_COLORS[li]
            ax.plot(SIGMAS, 10 * np.log10(ana[gi, li] / ref), "-", color=c,
                    lw=2, label=f"analytic, l = {l:g} lam")
            ax.plot(SIGMAS, 10 * np.log10(mc[gi, li] / ref), "o", color=c,
                    ms=5, mfc="white", label=f"MC (N={N_REAL}), l = {l:g}")
        ax.set_title(f"{GEOMS[gi][0]}")
        ax.set_xlabel("sigma / lambda")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("<|Phi|^2> / (Lx Ly)^2  [dB]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Rough 4x7 lambda facet: ensemble MC vs Eq 20+21 analytic")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_haynes(path, sigmas, tot_s, tot_r, coh_s, coh_r):
    fig, ax = plt.subplots(figsize=(6.5, 4.4), constrained_layout=True)
    ref = tot_r[0]
    ax.plot(sigmas, 10 * np.log10(tot_r / ref), "-", color="#4053d3", lw=2,
            label="Haynes Eq 34-36 total")
    ax.plot(sigmas, 10 * np.log10(tot_s / ref), "o", color="#4053d3", ms=6,
            mfc="white", label="rough-facet disk total")
    ax.plot(sigmas, 10 * np.log10(coh_r / ref), "--", color="#ddb310", lw=2,
            label="exp(-(2 k sigma)^2) coherent")
    ax.plot(sigmas, 10 * np.log10(coh_s / ref), "s", color="#ddb310", ms=6,
            mfc="white", label="rough-facet coherent part")
    ax.set_xlabel("sigma_h / lambda")
    ax.set_ylabel("power rel. smooth disk [dB]")
    ax.set_title("Fresnel-zone disk at nadir: rough facets vs Haynes 2018")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("MC sweep (a)...")
    mc, ana = mc_sweep()
    res_db = 10.0 * np.log10(mc / ana)
    gated = res_db[:, :, SIGMAS <= 0.30]
    mc_max = float(np.abs(gated).max())
    mc_max_all = float(np.abs(res_db).max())

    print("Haynes scan (c)...")
    sigmas_h, (tot_s, tot_r, coh_s, coh_r) = haynes_scan()
    tot_db = 10.0 * np.log10(tot_s / tot_r)
    coh_db = 10.0 * np.log10(coh_s / coh_r)
    hay_tot_max = float(np.abs(tot_db).max())
    hay_coh_max = float(np.abs(coh_db[sigmas_h <= 0.15]).max())

    print("convergence (d)...")
    worst10, worst_full, wrel = convergence()

    metrics = {
        "mc_total_residual_db_max": {
            "value": mc_max, "threshold": 1.0, "op": "<=",
            "pass": mc_max <= 1.0,
            "region": "sigma <= 0.3 lam, all l and geometries"},
        "mc_total_residual_db_max_incl_04": {
            "value": mc_max_all, "threshold": 1.5, "op": "<=",
            "pass": mc_max_all <= 1.5, "region": "including sigma = 0.4 lam"},
        "haynes_total_residual_db_max": {
            "value": hay_tot_max, "threshold": 1.0, "op": "<=",
            "pass": hay_tot_max <= 1.0, "region": "sigma_h <= 0.25 lam"},
        "haynes_coherent_residual_db_max": {
            "value": hay_coh_max, "threshold": 0.5, "op": "<=",
            "pass": hay_coh_max <= 0.5, "region": "sigma_h <= 0.15 lam"},
        "series_10term_relerr_max": {
            "value": worst10, "threshold": 1e-9, "op": "<=",
            "pass": worst10 <= 1e-9, "region": "sigma <= lam/20 vs mpmath"},
        "series_full_relerr_max": {
            "value": worst_full, "threshold": 1e-8, "op": "<=",
            "pass": worst_full <= 1e-8, "region": "n_terms_for vs mpmath"},
        "faddeeva_relerr_max": {
            "value": wrel, "threshold": 1e-11, "op": "<=",
            "pass": wrel <= 1e-11},
    }
    write_metrics(
        OUTDIR / "metrics.json", "rough_facet", metrics, group=GROUP,
        notes=f"Gerekos et al. 2023 rough rectangular facet: (a) 4x7 lam "
              f"facet at lam/40 sampling, {N_REAL} correlated-surface "
              f"realizations per point, sigma/lam in {SIGMAS.tolist()}, "
              f"l/lam in {[float(l) for l in CORR_LS]}, nadir/oblique/"
              "off-axis monostatic geometries: ensemble <|Phi|^2> vs the "
              "Eq 20+21 analytic (coherent + D_Phi series). (c) Fresnel-zone "
              "facet disk (4-lam facets, l = 2 lam, h = 8000 lam) vs the "
              "Haynes 2018 Eq 34-36 closed forms, total and coherent-only. "
              "(d) fixed-term series vs mpmath (50 digits) and Weideman "
              "Faddeeva vs scipy wofz. sigma = 0.4 lam recorded separately "
              "(Kirchhoff/LPA validity edge, worst near sigma ~ lam/4 with "
              "large l per the paper).")
    fig_mc(OUTDIR / "fig_mc_sweep.png", mc, ana)
    fig_haynes(OUTDIR / "fig_haynes_disk.png", sigmas_h, tot_s, tot_r,
               coh_s, coh_r)
    for k, v in metrics.items():
        print(f"  {k}: {v['value']:.3g} ({'PASS' if v['pass'] else 'FAIL'})")
    return 0 if all(v["pass"] for v in metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
