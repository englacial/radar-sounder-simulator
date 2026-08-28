"""Simplified model: surface clutter at the bed delay vs frequency.

clutter/bed ratio  C(f) ∝ k^4 cos^4(th) S(2k sin th) * AF2(th) / B(f)

- th: off-nadir angle whose delay equals the bed's, cos th = h / (h + n d)
  (the bed's extra two-way delay is the OPTICAL path n d)
- S(k): surface height PSD (2-D, m^4): gaussian / exponential / powerlaw,
  all normalised to the same value at 5 m wavelength (the ATM finding: the
  amplitude at 5 m is common, the form differs)
- k^4 cos^4 S(2k sin th): first-order (SPM / Kirchhoff m=1) diffuse law
- AF2: two-way cross-track array factor at th, normalised to nadir;
  'span': N elements over a fixed physical span W (Hann taper);
  'lam':  N elements at lambda/2 (beam width fixed in degrees)
- B(f): bed factor; 1 (specular, f-independent) or coherent-loss
  exp(-(2 k_ice sigma_b)^2) for a rough bed
Everything else (spreading, attenuation, area per range bin) is
frequency-independent and cancels in the ratio.
    uv run python claude_notes/freq_regimes/freq_regimes.py
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

C = 299792458.0; N_ICE = 1.78
OUT = Path(__file__).resolve().parents[2] / "outputs" / "freq_regimes"; OUT.mkdir(parents=True, exist_ok=True)


def psd(kind, k, l=3.0, H=0.4, k5=2 * np.pi / 5.0):
    """2-D PSD shape, normalised to 1 at wavelength 5 m."""
    if kind == "gaussian":
        s = np.exp(-(k * l) ** 2 / 4)
    elif kind == "exponential":
        s = (1 + (k * l) ** 2) ** -1.5
    else:  # power law, beta = 2H + 2
        s = k ** -(2 * H + 2)
    return s / (psd_raw(kind, k5, l, H))


def psd_raw(kind, k, l, H):
    if kind == "gaussian":
        return np.exp(-(k * l) ** 2 / 4)
    if kind == "exponential":
        return (1 + (k * l) ** 2) ** -1.5
    return k ** -(2 * H + 2)


def af2(th, lam, mode, W=10.0, N=16):
    """Two-way array factor at angle th (rad), Hann-tapered N elements,
    averaged over +-2 deg so sidelobe nulls do not dominate the picture."""
    w = np.hanning(N + 2)[1:-1]
    m = np.arange(N) - (N - 1) / 2
    d = W / (N - 1) if mode == "span" else lam / 2
    tt = th + np.radians(np.linspace(-2, 2, 41))
    ph = 2 * np.pi * d / lam * np.sin(tt)[:, None] * m
    a = np.abs(np.sum(w * np.exp(1j * ph), axis=1)) / np.sum(w)
    return float(np.mean(a ** 4))      # tx and rx taper, power


def clutter_angle(h, d):
    return np.arccos(h / (h + N_ICE * d))


def ratio_db(f, kind, th, ant, l=3.0, H=0.4, sigma_bed=0.0):
    k = 2 * np.pi * f / C
    lam = C / f
    kb = 2 * k * np.sin(th)
    s = (k ** 4) * np.cos(th) ** 4 * psd(kind, kb, l, H)
    a = np.array([af2(th, L, ant) for L in lam])
    # coherent bed loses exp(-(2 k_ice sigma)^2); a diffuse floor 10 dB down remains
    bed = (0.9 * np.exp(-(2 * k * N_ICE * sigma_bed) ** 2) + 0.1) if sigma_bed else 1.0
    return 10 * np.log10(s * a / bed)


def main():
    f = np.geomspace(20e6, 500e6, 200)
    geoms = [("airborne 500 m AGL, 1 km ice", 500.0, 1000.0),
             ("HAPS 14 km alt (12.5 km AGL), 1 km ice", 12500.0, 1000.0),
             ("HAPS 14 km alt (12.5 km AGL), 3 km ice", 12500.0, 3000.0),
             ("HAPS 20 km alt (18.5 km AGL), 1 km ice", 18500.0, 1000.0),
             ("HAPS 20 km alt (18.5 km AGL), 3 km ice", 18500.0, 3000.0),
             ("orbital 600 km, 2 km ice", 600e3, 2000.0)]
    kinds = [("gaussian l=3 m", "gaussian", dict(l=3.0)),
             ("exponential l=5 m", "exponential", dict(l=5.0)),
             ("power law H=0.4", "powerlaw", dict(H=0.4))]
    fig, axes = plt.subplots(2, 6, figsize=(26, 8), sharex=True)
    for j, (gname, h, d) in enumerate(geoms):
        th = clutter_angle(h, d)
        for i, ant in enumerate(("lam", "span")):
            ax = axes[i, j]
            for kname, kind, kw in kinds:
                r = ratio_db(f, kind, th, ant, **kw)
                ax.plot(f / 1e6, r - r[np.argmin(np.abs(f - 60e6))], label=kname)
            r = ratio_db(f, "powerlaw", th, ant, H=0.4, sigma_bed=0.10)
            ax.plot(f / 1e6, r - r[np.argmin(np.abs(f - 60e6))], "k--", lw=1,
                    label="power law + rough bed (0.10 m)")
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_xscale("log"); ax.set_ylim(-60, 40); ax.grid(alpha=0.3)
            ax.set_title(f"{gname}\nclutter angle {np.degrees(th):.0f} deg, "
                         + ("array: N el at lambda/2 (fixed beam)" if ant == "lam"
                            else "array: 16 el on fixed 10 m span"), fontsize=9)
            if j == 0:
                ax.set_ylabel("surface clutter at bed delay / bed\n(dB rel. 60 MHz)")
            if i == 1:
                ax.set_xlabel("frequency (MHz)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Simplified model: which way frequency moves bed-delay surface clutter "
                 "(first-order diffuse law x two-way array factor; curves normalised at 60 MHz)")
    fig.tight_layout(); fig.savefig(OUT / "freq_regimes.png", dpi=130)

    # crossover map: for a Gaussian surface, the frequency above which higher f helps
    fig2, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    th = np.radians(np.linspace(2, 80, 200))
    for l in (1.0, 3.0, 5.0, 10.0):
        ax[0].plot(np.degrees(th), C * np.sqrt(2) / (2 * np.pi * l * np.sin(th)) / 1e6, label=f"l = {l:g} m")
    ax[0].set_yscale("log"); ax[0].set_ylim(10, 2000); ax[0].grid(alpha=0.3)
    ax[0].set_xlabel("clutter angle (deg)"); ax[0].set_ylabel("crossover frequency (MHz)")
    ax[0].set_title("Gaussian ACF: above this f, higher f -> LESS clutter\n(k l sin th = sqrt 2)")
    for th_c, name in ((6, "orbital"), (24, "HAPS 20 km / 1 km ice"), (29, "HAPS 14 km / 1 km ice"), (39, "HAPS 20 km / 3 km ice"), (46, "HAPS 14 km / 3 km ice"), (77, "airborne")):
        ax[0].axvline(th_c, color="0.6", ls=":"); ax[0].text(th_c + 0.5, 1500, name, fontsize=8, rotation=90, va="top")
    ax[0].legend(fontsize=8)
    for W in (5.0, 10.0, 20.0, 40.0):
        ax[1].plot(np.degrees(th), C / (W * np.sin(th)) / 1e6, label=f"span {W:g} m")
    ax[1].set_yscale("log"); ax[1].set_ylim(10, 2000); ax[1].grid(alpha=0.3)
    ax[1].set_xlabel("clutter angle (deg)"); ax[1].set_ylabel("frequency (MHz)")
    ax[1].set_title("Fixed-span array: above this f the clutter angle is outside\nthe main lobe (lambda / W = sin th) and the array starts to help")
    for th_c in (6, 24, 29, 39, 46, 77):
        ax[1].axvline(th_c, color="0.6", ls=":")
    ax[1].legend(fontsize=8)
    fig2.tight_layout(); fig2.savefig(OUT / "freq_crossovers.png", dpi=130)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
