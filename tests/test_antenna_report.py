"""M22 report case ``antenna_patterns`` (group "Radar equation comparison"):
cross-track clutter suppression from antenna patterns.

Flat + hill scenes (incoherent mode: speckle-free cluttergrams make the
suppression readable; the kernels share one gain convention, g**4 on power)
under three antennas at MCoRDS-like geometry (195 MHz-class, 1000 m AGL):

- isotropic (today's default; the stage-2 Helheim comparison showed the
  simulated cluttergram carries MORE off-nadir clutter than the measured
  frame -- exactly what a real antenna pattern suppresses),
- half-wave dipole, axis ALONG-track (a single MCoRDS-like element: nearly
  no gain variation in the cross-track plane, so cross-track clutter is
  barely touched -- the honest single-element baseline),
- 5-element uniform cross-track array, spacing 0.5 lam, boresight nadir
  (array factor sin(5x)/(5 sin x): first null at sin(theta_ct) = 0.4).

Off-nadir suppression metric: total power in the 25-55 deg apparent-
incidence band (twtt mapped to theta by cos(theta) = t_nadir/t), isotropic
vs pattern, in dB. The flat-scene 30-deg bin is additionally checked against
the analytic ring average of g**4 over azimuth (the per-bin geometry factors
cancel in the iso-normalized ratio). Gate values are set from the first run
(repo convention); measured values inline next to each metric.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from soundersim import simulate
from soundersim import synthetic as syn
from soundersim.compare import plots
from soundersim.config import AntennaConfig, FacetConfig, RadarConfig, SimConfig
from soundersim.physics import C

OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "verification"
GROUP = "Radar equation comparison"

H = 1000.0  # AGL
DT, NSAMP = 1e-8, 1250

ANTENNAS = {
    "isotropic": AntennaConfig(),
    "dipole": AntennaConfig(kind="dipole", axis="along_track"),
    "array_5x": AntennaConfig(kind="array", n_elements=5, spacing_lam=0.5),
}


def _cfg(ant):
    return SimConfig(mode="incoherent",
                     radar=RadarConfig(dt=DT, n_samples=NSAMP, t0=0.0,
                                       antenna=ant),
                     facets=FacetConfig())


def _theta(twtt, t_nadir):
    """Apparent incidence angle (rad) of a flat-surface ring at delay twtt."""
    with np.errstate(invalid="ignore"):
        return np.arccos(np.clip(t_nadir / np.maximum(twtt, 1e-30), 0.0, 1.0))


def _ring_avg_g4(ant, theta, n_az=4096):
    """Analytic azimuth average of g**4 on the theta ring (flat surface)."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)
    st = np.sin(theta)
    if ant.kind == "isotropic":
        return 1.0
    if ant.kind == "dipole":  # axis along-track (+x): cos(psi)=sin(t)cos(phi)
        ca = st * np.cos(phi)
        g = np.cos(np.pi / 2.0 * ca) / np.sqrt(np.maximum(1 - ca * ca, 1e-12))
    else:  # cross-track array: u = sin(t)sin(phi)
        x = np.pi * ant.spacing_lam * st * np.sin(phi)
        n = ant.n_elements
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.abs(np.sin(n * x) / (n * np.sin(x)))
        g = np.where(np.abs(np.sin(x)) < 1e-12, 1.0, g)
    return float((g ** 4).mean())


@pytest.mark.integration
def test_antenna_patterns_report():
    scenes = {"flat": syn.flat_scene(elevation=500.0, altitude=H),
              "hill": syn.hill_scene(elevation=500.0, altitude=H)}
    runs = {(sn, an): simulate(sc, _cfg(ant))
            for sn, sc in scenes.items() for an, ant in ANTENNAS.items()}

    twtt = runs[("flat", "isotropic")].twtt.values
    t_nadir = 2.0 * H / C
    theta = _theta(twtt, t_nadir)
    band = (theta >= np.deg2rad(25.0)) & (theta <= np.deg2rad(55.0))
    mid = 10  # mid trace of 20

    # ---- off-nadir band suppression (iso / pattern, dB), per scene
    supp = {}
    for sn in scenes:
        p_iso = runs[(sn, "isotropic")].power.values[mid]
        for an in ("dipole", "array_5x"):
            p = runs[(sn, an)].power.values[mid]
            supp[(sn, an)] = float(10.0 * np.log10(
                p_iso[band].sum() / p[band].sum()))

    # ---- flat-scene ~30-deg band vs analytic ring average of g**4.
    # A single range bin holds only a handful of 50 m facets (the bin's
    # horizontal annulus is ~3 m wide), so the azimuth-steep array pattern
    # needs a band average: 27-33 deg, iso-power-weighted analytic prediction.
    band30 = (theta >= np.deg2rad(27.0)) & (theta <= np.deg2rad(33.0))
    p_iso_flat = runs[("flat", "isotropic")].power.values[mid]
    ring_err = {}
    for an in ("dipole", "array_5x"):
        meas = (runs[("flat", an)].power.values[mid, band30].sum()
                / p_iso_flat[band30].sum())
        g4 = np.array([_ring_avg_g4(ANTENNAS[an], t)
                       for t in theta[band30]])
        pred = float((p_iso_flat[band30] * g4).sum()
                     / p_iso_flat[band30].sum())
        ring_err[an] = float(abs(10.0 * np.log10(meas / pred)))

    # ---- hill-echo band: strongest clutter above the flat background
    # (hill at 1500 m cross-track: echo near r = 1700 m -> twtt ~ 11.3 us)
    hill_band = (twtt >= 10.8e-6) & (twtt <= 12.0e-6)
    hill_supp = {}
    for an in ("dipole", "array_5x"):
        hill_supp[an] = float(10.0 * np.log10(
            runs[("hill", "isotropic")].power.values[mid, hill_band].sum()
            / runs[("hill", an)].power.values[mid, hill_band].sum()))

    # ---- metrics (gates set from the first run; measured values inline)
    metrics = {
        "flat_array_offnadir_suppression_db": {
            "value": supp[("flat", "array_5x")], "threshold": 6.0,
            "op": ">=", "pass": supp[("flat", "array_5x")] >= 6.0,
            "region": "25-55 deg apparent incidence, mid trace",
            "comment": "5-element cross-track array vs isotropic "
                       "(measured 8.3 dB; band average -- suppression is "
                       "much deeper near the array nulls, see figure)"},
        "flat_dipole_offnadir_suppression_db": {
            "value": supp[("flat", "dipole")], "threshold": 3.5, "op": "<=",
            "pass": supp[("flat", "dipole")] <= 3.5,
            "comment": "along-track dipole barely touches cross-track "
                       "clutter (measured 2.3 dB: only the ring's "
                       "along-track azimuths are attenuated) -- the "
                       "suppression story needs the array, not a single "
                       "element"},
        "hill_array_suppression_db": {
            "value": hill_supp["array_5x"], "threshold": 7.0, "op": ">=",
            "pass": hill_supp["array_5x"] >= 7.0,
            "region": "hill-echo band 10.8-12.0 us (hill 1500 m cross-track "
                      "~ 56 deg off nadir)",
            "comment": "measured 9.4 dB"},
        "hill_dipole_suppression_db": {
            "value": hill_supp["dipole"], "threshold": 5.0, "op": "<=",
            "pass": hill_supp["dipole"] <= 5.0,
            "comment": "measured 3.8 dB (the band also contains the flat "
                       "ring's along-track azimuths, which the dipole does "
                       "suppress; the cross-track hill itself sits at the "
                       "dipole's broadside g = 1)"},
        "array_ring_avg_error_30deg_db": {
            "value": ring_err["array_5x"], "threshold": 1.0, "op": "<=",
            "pass": ring_err["array_5x"] <= 1.0,
            "comment": "flat-scene 27-33 deg band iso-normalized ratio vs "
                       "analytic azimuth average of g**4 (measured 0.3 dB)"},
        "dipole_ring_avg_error_30deg_db": {
            "value": ring_err["dipole"], "threshold": 1.0, "op": "<=",
            "pass": ring_err["dipole"] <= 1.0,
            "comment": "measured 0.03 dB"},
    }

    outdir = OUTDIR / "antenna_patterns"
    plots.write_metrics(
        outdir / "metrics.json", "antenna_patterns", metrics, group=GROUP,
        notes="Antenna patterns (M22): flat + hill incoherent cluttergrams "
              "under isotropic / along-track half-wave dipole / 5-element "
              "cross-track uniform array (0.5 lam spacing, boresight nadir), "
              "1000 m AGL. Two-way gain convention: one-way FIELD gain g, "
              "fields weighted g^2, power g^4 (antenna.py). The array "
              f"suppresses the 25-55 deg off-nadir band by "
              f"{supp[('flat', 'array_5x')]:.1f} dB and the cross-track "
              f"hill echo by {hill_supp['array_5x']:.1f} dB, while the "
              "along-track dipole leaves cross-track clutter nearly "
              f"untouched ({supp[('flat', 'dipole')]:.1f} dB) -- the "
              "physics behind the stage-2 Helheim finding that the measured "
              "frame shows less off-nadir clutter than the isotropic "
              "simulation. Per-bin ratios match the analytic azimuth "
              "average of g^4 to "
              f"{max(ring_err.values()):.2f} dB at 30 deg.")

    _figure(outdir / "antenna_patterns.png", runs, scenes, twtt, theta, mid)

    for name, e in metrics.items():
        assert e["pass"], (name, e["value"])


AGG = 32  # fast-time aggregation to facet scale (~48 m range: 50 m facets)


def _figure(path, runs, scenes, twtt, theta, mid):
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 7.2),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    tus = twtt[::AGG][: len(twtt) // AGG] * 1e6  # aggregated bin starts
    for i, sn in enumerate(scenes):
        agg_runs = {an: plots.aggregate(runs[(sn, an)].power.values, AGG)
                    for an in ANTENNAS}
        ref = agg_runs["isotropic"].max()
        for j, an in enumerate(ANTENNAS):
            ax = axes[i, j]
            p = agg_runs[an]
            db = 10.0 * np.log10(np.maximum(p, ref * 1e-12) / ref)
            im = ax.imshow(db, aspect="auto", vmin=-60, vmax=0,
                           extent=[tus[0], tus[-1], p.shape[0] - 0.5, -0.5],
                           cmap="viridis")
            ax.set_title(f"{sn} / {an}", fontsize=10)
            ax.set_xlim(6.2, 12.4)
            if j == 0:
                ax.set_ylabel("trace")
            ax.set_xlabel("twtt (us)")
        fig.colorbar(im, ax=axes[i, 2], label="dB rel iso peak", pad=0.02)

        ax = axes[i, 3]
        for an, color in zip(ANTENNAS, ("0.3", "C1", "C0")):
            db = 10.0 * np.log10(
                np.maximum(agg_runs[an][mid], ref * 1e-12) / ref)
            ax.plot(tus, db, color=color, lw=1.2, label=an)
        for th_deg in (25, 40, 55):
            t_mark = 2.0 * H / np.cos(np.deg2rad(th_deg)) / C * 1e6
            ax.axvline(t_mark, color="0.8", lw=0.8, zorder=0)
            ax.text(t_mark, 1.5, f"{th_deg}\N{DEGREE SIGN}", fontsize=7,
                    ha="center", color="0.5")
        if sn == "hill":
            ax.axvspan(10.8, 12.0, color="C3", alpha=0.07)
            ax.text(11.4, -68, "hill band", fontsize=7, ha="center",
                    color="C3")
        ax.set_xlim(6.2, 12.4)
        ax.set_ylim(-72, 4)
        ax.set_xlabel("twtt (us)")
        ax.set_ylabel("power (dB rel iso peak)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"{sn}: mid-trace profiles ({AGG}-bin aggregated)",
                     fontsize=10)
    fig.suptitle("Antenna-pattern clutter suppression (incoherent, 1000 m "
                 "AGL): isotropic vs along-track dipole vs 5-element "
                 "cross-track array")
    fig.savefig(path, dpi=90)
    plt.close(fig)
