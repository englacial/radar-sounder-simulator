"""M19 firn power plateau: 3-D analog of Culberg & Schroeder (2020) Fig. 9.

Report case (group "Firn clutter (Culberg & Schroeder 2020)"): a flat MCoRDS3-
like scene (f0 = 195 MHz, 500 m AGL) over a firn stack built from the B26
ice-core density profile (tests/fixtures/firn/, provenance in its README):
20 offset-interface firn layers at uniform 5 m spacing over 5-100 m depth,
each medium taking the MEAN density of its slab through the Kovacs et al.
(1993) relation eps = (1 + 0.845*rho_gcc)^2 (the paper's Eq. 4); attenuation
neglected (paper: < 3 dB two-way through the firn column).

Decimation notes (measured during development, claude_notes/m19_dev_run.py):

- UNIFORM node spacing is essential: with depth-graded spacing (2.5 m near
  surface, ~14 m deep) the deep interface contrasts scale with slab thickness
  -- the smooth compaction trend across a thick slab produces as large a
  density step as near-surface variability, and the plateau/rolloff structure
  vanishes (measured rolloff 0.5 dB). With uniform 5 m slabs both the
  densification-rate trend step and the layer variability decay with depth
  and the Fig. 9 structure appears (rolloff 15.6 dB).
- Slab-MEAN density sampling gives a smooth, deterministic contrast profile
  (interface gamma -32.7 dB at 5 m to -55.9 dB at 100 m). Midpoint POINT
  sampling (variability-preserving) was also measured: same band-mean trend,
  +-10 dB layer-to-layer scatter resembling Fig. 9's spikiness -- slab-mean
  is used for gate robustness.
- O(tens) of layers is a physics check of the multilayer machinery, not a
  stratigraphy capability claim (plan M19 scoping caution): the paper's model
  has mm-scale layers; within-range-bin thin-film interference is NOT
  represented here.

Per-layer observable: the window-total complex field per layer (the
test_haynes_coherent.py convention -- per-bin coherent values on smooth
surfaces are cancellation-dominated), spreading-compensated by the nadir
image-method range (h + sum dz_i/n_i)^2 and normalized to the surface layer
(Fig. 9's "normalized reflection coefficient" convention). Individual weak
deep layers (gamma < ~-50 dB) carry up to ~6 dB of aperture/facet-
quantization residual (trace-dependent rim-ringing interference measured
against the flat-layer gamma^2 closed form), so all gates use DEPTH-BAND
means (band-mean residual measured <= 1.6 dB).

The incoherent kernel (stage-1/simc convention) carries NO interface
reflectivity -- its per-layer power is pure transmission/spreading geometry --
so its depth profile is structureless by construction (measured: 2.6 dB total
decay over 100 m, band rolloff 1.6 dB). That IS the coherent-vs-incoherent
story: the plateau + rolloff requires the coherent specular physics
(reflection coefficients from the density profile); an incoherent facet
clutter sum cannot produce it. The incoherent run uses 10 uniform 10-m nodes
(its smooth geometric profile needs no fine sampling), halving its compile
cost.

Compile-cost note (kernels/multilayer.py warning, measured): simulate()'s
multilayer path compiles one XLA graph per target interface, target j
unrolling j crossings x ~35 Newton iterations -- total compile scales
~0.18*N^2 s (N=10: 16 s, N=20: 69 s, N=30: 167 s measured cold). N=20
coherent + N=10 incoherent keeps this case ~2 min; more layers needs kernel
work (scanned solve / persistent compile cache), recorded as the scaling
limit.

Gates (thresholds set from the first run, per repo convention; measured
values inline):

- plateau: per-layer power over 0 < z <= 40 m within 12 dB of its maximum
  (measured span 10.5 dB) -- "elevated, slowly decaying over the upper tens
  of meters".
- rolloff: plateau band (<=40 m) mean minus deep band (>=70 m) mean >= 8 dB
  (measured 15.6 dB), with band means monotonically decreasing by >= 2 dB
  per band (measured steps 5.9 and 9.6 dB).
- coherent-vs-incoherent: coherent band rolloff exceeds the incoherent one
  by >= 8 dB (measured excess 14.0 dB).
- physics closure: band-mean |sim - decimated-gamma^2 prediction| <= 3 dB
  (measured max 1.6 dB); no dropped power (flat scene, window covers the
  corner returns).

Context lines on the figure (never gated): the digitized Fig. 9 curves from
tests/fixtures/firn/ -- the paper's 1-D transfer-matrix simulated normalized
reflection coefficient for MCoRDS3 (fig09b, same band as this case) and AR
(fig09a, 750 MHz), which track the paper's empirical profiles.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import soundersim
from soundersim.compare import plots
from soundersim.config import (DemInterface, FacetConfig, Medium,
                               OffsetInterface, RadarConfig, SimConfig)
from soundersim.physics import C, fresnel_normal
from soundersim import synthetic as syn

FIXDIR = Path(__file__).resolve().parent / "fixtures" / "firn"
OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "verification"
GROUP = "Firn clutter (Culberg & Schroeder 2020)"

H, ELEV, EXTENT = 500.0, 500.0, 600.0   # platform AGL, surface elev, scene (m)
F0, DT, NSAMP = 195e6, 5e-9, 512        # MCoRDS3-like carrier; 5 ns bins
T0 = 2.0 * (H - 10.0) / C
DEPTHS_C = np.arange(5.0, 100.1, 5.0)   # 20 coherent layers
DEPTHS_I = np.arange(10.0, 100.1, 10.0)  # 10 incoherent nodes (see docstring)
PLAT_MAX, DEEP_MIN = 40.0, 70.0         # depth bands (m)


def load_b26(smooth_m=0.1):
    """B26 density profile (depth m, density kg/m^3), lightly smoothed.

    PANGAEA tab format: comment block, then a 'Depth ice/snow [m]' header line,
    then tab-separated data at 1 mm sampling. A ~0.1 m boxcar suppresses
    densitometer noise while keeping the seasonal-scale variability.
    """
    path = FIXDIR / "ngt37C95.2_density.tab"
    lines = path.read_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Depth ice/snow"))
    data = np.loadtxt(path, delimiter="\t", skiprows=hdr + 1)
    z, rho = data[:, 0], data[:, 1]
    k = int(round(smooth_m / np.median(np.diff(z)))) | 1
    return z, np.convolve(rho, np.ones(k) / k, mode="same")


def eps_kovacs(rho_kgm3):
    """Kovacs et al. (1993) density->permittivity, C&S 2020 Eq. (4):
    eps' = (1 + 0.845 * rho)^2 with rho in g/cm^3 (imaginary part neglected)."""
    return (1.0 + 0.845 * np.asarray(rho_kgm3) / 1000.0) ** 2


def load_digitized(name):
    """Digitized Fig. 9 curve (depth m, dB); '#'-comment + header-line CSV."""
    rows = []
    for line in (FIXDIR / name).read_text().splitlines():
        parts = line.split(",")
        if len(parts) == 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue  # header line
    return np.array(rows)


def firn_stack(depths, z, rho):
    """(media, interfaces, eps array) for a slab-mean decimated firn stack."""
    edges = np.concatenate([[0.0], depths, [depths[-1] + 10.0]])
    rho_slab = [rho[(z >= a) & (z < b)].mean() if ((z >= a) & (z < b)).any()
                else float(np.interp(0.5 * (a + b), z, rho))
                for a, b in zip(edges[:-1], edges[1:])]
    eps = np.concatenate([[1.0], eps_kovacs(np.array(rho_slab))])
    media = [Medium(name="air", eps_r=1.0)] + [
        Medium(name=f"firn_{i}", eps_r=float(e)) for i, e in enumerate(eps[1:])]
    ifaces = [DemInterface(name="surface")] + [
        OffsetInterface(name=f"L{d:g}", reference="surface", offset=-float(d))
        for d in depths]
    return media, ifaces, eps


def run(mode, depths, z, rho):
    media, ifaces, eps = firn_stack(depths, z, rho)
    scene = syn.flat_scene(elevation=ELEV, altitude=H, extent=EXTENT,
                           posting=4.0, n_traces=3)
    cfg = SimConfig(mode=mode, radar=RadarConfig(dt=DT, n_samples=NSAMP, t0=T0,
                                                 f0=F0),
                    facets=FacetConfig(spacing=4.0), media=media,
                    interfaces=ifaces)
    return soundersim.simulate(scene, cfg), eps


def normalized_profile(ds, depths, eps):
    """Per-layer power (dB rel. surface), nadir-spreading-compensated.

    Coherent: |window-total field|^2 (see docstring); incoherent: window power
    sum. Compensation (r_eff/h)^2 with r_eff = h + sum(dz_i/n_i), the
    image-method nadir effective range through the stack.
    """
    if "field" in ds:
        p = (np.abs(ds.field.sum("twtt").values) ** 2).mean(axis=0)
    else:
        p = ds.power.sum("twtt").values.mean(axis=0)
    dz = np.diff(np.concatenate([[0.0], depths]))
    r_eff = H + np.concatenate([[0.0],
                                np.cumsum(dz / np.sqrt(eps[1:1 + len(dz)]))])
    p = p * (r_eff / H) ** 2
    return 10.0 * np.log10(p / p[0])


def combined_depth_trace(ds, box_bins=7):
    """Layer-combined trace power vs depth (dB rel. its peak).

    Coherent: fields summed over layers, boxcar-convolved over ~35 ns
    (emulating an MCoRDS3-like ~3.7 m in-firn range resolution with a
    rectangular pulse), then |.|^2, trace-averaged. Incoherent: additive
    power, boxcar power sum. twtt -> depth via the per-layer nadir_twtt.
    """
    box = np.ones(box_bins)
    if "field" in ds:
        tot = ds.field.sum("layer").values
        p = np.array([np.abs(np.convolve(t, box, mode="same")) ** 2
                      for t in tot]).mean(axis=0)
    else:
        tot = ds.power.sum("layer").values
        p = np.array([np.convolve(t, box, mode="same")
                      for t in tot]).mean(axis=0)
    nt = ds.nadir_twtt.values.mean(axis=0)
    node_depth = np.concatenate([[0.0],
                                 [float(str(n)[1:]) for n in
                                  ds.layer.values[1:]]])
    depth = np.interp(ds.twtt.values, nt, node_depth)
    return depth, 10.0 * np.log10(np.maximum(p / p.max(), 1e-12))


def band_means(depths, prof_db):
    """(plateau <=40 m, mid 40-70 m, deep >=70 m) means over the layer values."""
    d, r = np.asarray(depths), np.asarray(prof_db)
    return (float(r[(d > 0) & (d <= PLAT_MAX)].mean()),
            float(r[(d > PLAT_MAX) & (d < DEEP_MIN)].mean()),
            float(r[d >= DEEP_MIN].mean()))


@pytest.mark.integration
def test_firn_plateau():
    z, rho = load_b26()
    ds_c, eps_c = run("coherent", DEPTHS_C, z, rho)
    ds_i, eps_i = run("incoherent", DEPTHS_I, z, rho)

    zc = np.concatenate([[0.0], DEPTHS_C])
    zi = np.concatenate([[0.0], DEPTHS_I])
    r_c = normalized_profile(ds_c, DEPTHS_C, eps_c)
    r_i = normalized_profile(ds_i, DEPTHS_I, eps_i)

    # decimated-gamma^2 prediction (flat-layer image method, dB rel. surface)
    gam = fresnel_normal(eps_c[:-1], eps_c[1:])
    pred = 20.0 * np.log10(np.abs(gam) / np.abs(gam[0]))

    # ---- measurements
    lay = zc > 0
    plat = lay & (zc <= PLAT_MAX)
    span = float(r_c[plat].max() - r_c[plat].min())
    b_c = band_means(zc, r_c)
    b_i = band_means(zi, r_i)
    rolloff_c = b_c[0] - b_c[2]
    rolloff_i = b_i[0] - b_i[2]
    steps = (b_c[0] - b_c[1], b_c[1] - b_c[2])
    res = r_c - pred
    res_bands = np.abs(band_means(zc, res))  # surface (z=0) excluded by bands
    res_band_max = float(np.max(res_bands))
    tot_c = ds_c.power.sum("twtt").values.mean(axis=0)
    drop_frac = float((ds_c.dropped_power.values.mean(axis=0)
                       / np.maximum(tot_c, 1e-300)).max())

    metrics = {
        "plateau_span_db": {
            "value": span, "threshold": 12.0, "op": "<=",
            "pass": span <= 12.0,
            "region": "0 < z <= 40 m, dB below band max"},
        "rolloff_db": {
            "value": float(rolloff_c), "threshold": 8.0, "op": ">=",
            "pass": rolloff_c >= 8.0,
            "bands": "mean(<=40 m) - mean(>=70 m)"},
        "band_step_min_db": {
            "value": float(min(steps)), "threshold": 2.0, "op": ">=",
            "pass": min(steps) >= 2.0,
            "bands": "monotone band means <=40 / 40-70 / >=70 m"},
        "coh_minus_inc_rolloff_db": {
            "value": float(rolloff_c - rolloff_i), "threshold": 8.0,
            "op": ">=", "pass": (rolloff_c - rolloff_i) >= 8.0,
            "incoherent_rolloff_db": float(rolloff_i)},
        "gamma_pred_band_residual_db": {
            "value": res_band_max, "threshold": 3.0, "op": "<=",
            "pass": res_band_max <= 3.0,
            "per_layer_max_db": float(np.abs(res[lay]).max())},
        "dropped_power_frac_max": {
            "value": drop_frac, "threshold": 1e-6, "op": "<=",
            "pass": drop_frac <= 1e-6},
    }

    outdir = OUTDIR / "firn_plateau"
    plots.write_metrics(
        outdir / "metrics.json", "firn_plateau", metrics, group=GROUP,
        notes="3-D coherent facet simulation of the C&S 2020 Fig. 9 firn "
              "power plateau: B26 density profile (PANGAEA 57798) decimated "
              "to 20 uniform 5 m slab-mean layers (offset interfaces, Kovacs "
              "eps(rho)), MCoRDS3-like 195 MHz at 500 m AGL, 3 traces, 4 m "
              "facets on a 600 m flat scene. Per-layer window-total field "
              "power (dB rel. surface, nadir-spreading compensated) shows "
              f"the plateau (span {span:.1f} dB over the upper 40 m) and "
              f"monotonic rolloff ({rolloff_c:.1f} dB to 70-100 m), tracking "
              "the decimated-gamma^2 closed form to "
              f"{res_band_max:.1f} dB in band means. The incoherent kernel "
              "(no interface reflectivity by convention) shows no such "
              f"structure ({rolloff_i:.1f} dB band rolloff): the plateau is "
              "coherent/specular physics. Digitized Fig. 9 transfer-matrix "
              "curves are context only. NOTE: uniform node spacing is "
              "required (depth-graded slabs alias the compaction trend into "
              "deep contrasts and erase the rolloff); mm-scale within-bin "
              "thin-film interference is out of scope at O(tens) layers "
              "(plan M19 scoping caution).")

    _figure(outdir / "firn_depth_power.png", zc, r_c, pred, zi, r_i,
            ds_c, ds_i)

    assert span <= 12.0, f"plateau span {span:.2f} dB"
    assert rolloff_c >= 8.0, f"rolloff {rolloff_c:.2f} dB"
    assert min(steps) >= 2.0, f"band steps {steps}"
    assert rolloff_c - rolloff_i >= 8.0, (
        f"coherent-incoherent rolloff excess {rolloff_c - rolloff_i:.2f} dB")
    assert res_band_max <= 3.0, f"gamma-prediction band residual {res_bands}"
    assert drop_frac <= 1e-6


def _figure(path, zc, r_c, pred, zi, r_i, ds_c, ds_i):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    ax = axes[0]
    for name, color in (("fig09b_digitized.csv", "0.55"),
                        ("fig09a_digitized.csv", "0.8")):
        d = load_digitized(name)
        d = d[np.argsort(d[:, 0])]  # sort by depth: fig09a digitization order
        # self-intersects (59 depth reversals); the curve is a depth profile
        lbl = ("C&S20 Fig. 9b 1-D model (MCoRDS3)" if "09b" in name
               else "C&S20 Fig. 9a 1-D model (AR, 750 MHz)")
        ax.plot(d[:, 0], d[:, 1], "-", color=color, lw=1.0, label=lbl)
    ax.plot(zc, pred, "k--", lw=1.0, label="decimated gamma^2 (closed form)")
    ax.plot(zc, r_c, "o-", color="C0", ms=4, label="soundersim coherent")
    ax.plot(zi, r_i, "s-", color="C1", ms=4,
            label="soundersim incoherent (no reflectivity)")
    ax.axvspan(0, PLAT_MAX, color="C0", alpha=0.06)
    ax.axvspan(DEEP_MIN, 100, color="C3", alpha=0.06)
    ax.set_xlim(0, 105)
    ax.set_ylim(-60, 3)
    ax.set_xlabel("depth (m)")
    ax.set_ylabel("per-layer power (dB rel. surface)")
    ax.set_title("Per-layer depth-power profile (plateau / rolloff bands shaded)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    for ds, lbl, color in ((ds_c, "coherent (fields summed over layers)", "C0"),
                           (ds_i, "incoherent (power sum)", "C1")):
        depth, db = combined_depth_trace(ds)
        m = depth < 100.0  # beyond the deepest node the twtt map clamps
        ax.plot(depth[m], db[m], "-", color=color, lw=1.1, label=lbl)
    ax.set_xlim(0, 105)
    ax.set_ylim(-55, 2)
    ax.set_xlabel("depth mapped from twtt (m)")
    ax.set_ylabel("trace power (dB rel. peak)")
    ax.set_title("Combined trace, 35 ns boxcar (~3.7 m in-firn resolution)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Firn power plateau: B26 stack, 195 MHz, 500 m AGL "
                 "(Culberg & Schroeder 2020 Fig. 9 analog)")
    fig.savefig(path, dpi=90)
    plt.close(fig)
