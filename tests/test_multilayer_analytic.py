"""M17 multilayer analytic + referee verification (integration report cases).

Three report cases under "Radar equation comparison" validating the stage-3
multilayer machinery (refraction solve -> Fresnel/attenuation/spreading ->
in-medium phase; kernels/multilayer.py) end to end:

1. slab_absolute -- parameter-free absolute closed form for the flat-slab
   nadir bed return through simulate() (the haynes_constants pattern extended
   below the surface): window-integrated coherent bed-layer field vs the
   image-in-dielectric-halfspace solution (Peters et al. 2005, JGR 110)

       tau_down*tau_up * Gamma_bed * exp(-2j*k0*(h + n*d)) / (2*(h + d/n))

   across depth (50-1000 m), permittivity (1.5-3.17) and altitude
   (500-2000 m) sweeps, one point with nonzero attenuation (two-way FIELD
   factor 10**(-d_km*att/10), physics.py convention); bed nadir delay exact
   to the fast-time bin; and the surface layer of a multilayer run gated
   bit-compatible with the stage-2 single-interface path on the
   haynes_constants end-to-end configuration.
2. twomedia_field -- coherent multilayer kernel vs a sub-wavelength two-media
   brute-force field referee (exact per-sample Fermat crossing + direct
   summation in the M9 normalization, in-medium phase per leg;
   compare/brute_force_layered.py) on tiny kernel-level scenes: flat slab,
   gently rough bed under a flat surface, and flat bed under a gently rough
   surface (the two rough cases M16's smoke test did not cover). Gated on the
   window-integrated complex field (per the M16 finding, fine-dt per-bin
   splits are binning quantization) plus facet-scale profile agreement; the
   rough-surface local-plane chaining degradation is quantified with a
   same-facet f64 exact-Fermat referee (no tessellation confound) across a
   roughness-amplitude sweep, tying back to the M15 anchoring-error scaling.
3. bed_falloff -- the Haynes fall-off family extended below the surface:
   flat surface + flat bed, altitude sweep at fixed depth AND depth sweep at
   fixed altitude; the coherent nadir bed power follows the closed form's
   EFFECTIVE range, (h + d/n)^-2 (slope -2 in log r_eff), completing the
   stage-1/2 -4/-3/-2 family with the refraction-corrected range.

Scene sizing follows the measured haynes_coherent constraints: slab scenes are
u = 45 Fresnel units wide (extent 45*sqrt(lam*r_eff), ~2 % hard-square-rim
field ringing) with bed facets 0.09*sqrt(lam_ice*r_eff) (LPA error << 1 %;
r_eff = h + d/n is the nadir wavefront curvature radius of the refracted
two-media path, the quantity L_par reduces to at nadir). The ellipsoid-
curvature factor R/(R+h) used by the surface-return cases is O(3e-4) at these
altitudes and is omitted from the bed closed form (documented, inside gates).

The rough twomedia scenes are deliberately Rayleigh-smooth (A <= ~lam_ice/6)
so the window-integrated total stays coherent-dominated: rougher interfaces
turn the total into a cancellation residual where the kernel-vs-referee
tessellation difference dominates (the same reason haynes cases avoid per-bin
smooth-surface observables). Roughness beyond that regime is covered by the
same-facet chaining sweep, which has no tessellation confound by construction.

Thresholds were set from the first run (repo convention); measured then:

- slab_absolute: bed mag err <= 0.0062, phase err <= 0.64 deg, all delays
  bin-exact, attenuation-law ratio 1.0002 (3 dB one-way in 300 m at
  10 dB/km), surface layer bitwise equal to stage 2.
- twomedia_field: flat 1.0001 / +0.46 deg, rough bed 1.0006 / +0.66 deg,
  rough surface (A = 0.15 m) 1.0019 / +0.63 deg; profile max dB diff <= 1.02
  (agg = 2 bins, above -20 dB); degradation onset at A = 0.25 m:
  1.031 / +5.7 deg, chained opl error RMS 1.6 mm (A = 0.25) growing to
  1.34 m (A = 4) with log-log slope ~2.4 (M15 anchoring error A*k^2*delta^2/2
  with the effective anchor offset delta itself growing once the pass-1
  mean-plane crossing error exceeds a facet).
- bed_falloff: altitude-sweep slope -2.001, depth-sweep slope -2.007,
  abs level err <= 0.026.
"""

from pathlib import Path

import numpy as np
import pytest

import soundersim
from soundersim import synthetic as syn
from soundersim.compare import plots
from soundersim.compare.brute_force_layered import (
    fermat_crossing_batch,
    local_plane_opl,
    surface_facets,
    two_media_trace,
)
from soundersim.config import (DemInterface, FacetConfig, Medium, RadarConfig,
                               SimConfig)
from soundersim.kernels.multilayer import refracted_cluttergram
from soundersim.physics import C, fresnel_normal
from soundersim.refraction import snell_crossing

OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "verification"
GROUP = "Radar equation comparison"
F0 = 195e6
LAM = C / F0
K0 = 2.0 * np.pi / LAM
EPS_BED = 8.0


def _media(eps_ice, att=0.0, eps_bed=EPS_BED):
    return [Medium(name="air", eps_r=1.0),
            Medium(name="ice", eps_r=eps_ice, attenuation_db_per_km=att),
            Medium(name="bed", eps_r=eps_bed)]


def _slab_run(h, d, eps_ice, att=0.0, u=45.0, fac=0.09):
    """Coherent flat-slab simulate() run sized for the absolute comparison."""
    n = np.sqrt(eps_ice)
    r_eff = h + d / n
    ext = u * np.sqrt(LAM * r_eff)
    spacing = fac * np.sqrt((LAM / n) * r_eff)
    dt = 20e-9
    t0 = 2.0 * (h - 5.0) / C
    opl_max = np.sqrt(h * h + 2.0 * (ext / 2.0) ** 2) + n * d + 10.0
    n_samples = int(np.ceil((2.0 * opl_max / C - t0) / dt)) + 4
    scene = syn.slab_scene(surface=500.0, depth=d, extent=ext,
                           posting=ext / 64.0, n_traces=2, altitude=h)
    cfg = SimConfig(
        mode="coherent",
        radar=RadarConfig(dt=dt, n_samples=n_samples, t0=t0, f0=F0),
        facets=FacetConfig(spacing=spacing), media=_media(eps_ice, att),
        interfaces=[DemInterface(name="surface"), DemInterface(name="bed")])
    return soundersim.simulate(scene, cfg)


def _bed_vs_closed_form(ds, eps_ice, att=0.0):
    """Per-trace bed observables vs the image-in-dielectric closed form.

    h, d are taken from the dataset's own nadir geometry (per-layer
    nadir_twtt, float64 vertical path), so the comparison is parameter-free.
    Returns lists over traces: mag ratio, phase err (deg), delay-bin-exact
    flags, dropped fraction, measured |field|, r_eff, predicted power.
    """
    n = np.sqrt(eps_ice)
    tau2 = 1.0 - fresnel_normal(1.0, eps_ice) ** 2
    gam_b = fresnel_normal(eps_ice, EPS_BED)
    dt = float(ds.twtt[1] - ds.twtt[0])
    t0 = float(ds.twtt[0])
    out = {k: [] for k in ("mag", "phase", "bin_ok", "drop", "absf", "reff",
                           "pred")}
    for tr in range(ds.sizes["slow_time"]):
        opl = C * float(ds.nadir_twtt.sel(layer="bed")[tr]) / 2.0
        h = C * float(ds.nadir_twtt.sel(layer="surface")[tr]) / 2.0
        d = (opl - h) / n
        r_eff = h + d / n
        bed = np.asarray(ds.field.sel(layer="bed")[tr].values)
        f = complex(bed.sum())
        loss_db = d / 1000.0 * att  # one-way dB = two-way FIELD attenuation
        ref = (tau2 * gam_b * 10.0 ** (-loss_db / 10.0)
               * np.exp(-2j * K0 * opl) / (2.0 * r_eff))
        first = int(np.nonzero(np.abs(bed))[0][0])
        out["mag"].append(abs(f) / abs(ref))
        out["phase"].append(float(np.degrees(np.angle(f / ref))))
        out["bin_ok"].append(first == int(np.floor((2.0 * opl / C - t0) / dt)))
        pw = np.abs(bed) ** 2
        out["drop"].append(float(ds.dropped_power.sel(layer="bed")[tr])
                           / float(pw.sum()))
        out["absf"].append(abs(f))
        out["reff"].append(r_eff)
        out["pred"].append((tau2 * gam_b * 10.0 ** (-loss_db / 10.0)
                            / (2.0 * r_eff)) ** 2)
    return out


# ---------------------------------------------------------------- case 1


@pytest.mark.integration
def test_slab_absolute():
    """Flat-slab nadir bed return vs the absolute closed form (no fitted
    constants) over depth/permittivity/altitude sweeps + attenuation point;
    surface layer gated unchanged vs the stage-2 haynes_constants config."""
    sweep = [  # (h, d, eps_ice, att_db_per_km)
        (1000.0, 50.0, 3.17, 0.0),
        (1000.0, 300.0, 3.17, 0.0),
        (1000.0, 1000.0, 3.17, 0.0),
        (500.0, 300.0, 1.5, 0.0),
        (2000.0, 300.0, 2.2, 0.0),
        (1000.0, 300.0, 3.17, 10.0),
    ]
    labels, mags, phases = [], [], []
    bin_misses, drop_max = 0, 0.0
    absf = {}
    for h, d, eps, att in sweep:
        ds = _slab_run(h, d, eps, att=att)
        r = _bed_vs_closed_form(ds, eps, att=att)
        labels.append(f"h={h:.0f} d={d:.0f} eps={eps:g}"
                      + (f" att={att:g}" if att else ""))
        mags.append(float(np.mean(r["mag"])))
        phases.append(float(np.mean(r["phase"])))
        bin_misses += sum(not b for b in r["bin_ok"])
        drop_max = max(drop_max, max(r["drop"]))
        absf[(h, d, eps, att)] = float(np.mean(r["absf"]))
    mag_err = float(np.abs(np.array(mags) - 1.0).max())
    phase_err = float(np.abs(phases).max())

    # attenuation law: 10 dB/km one-way over ~300 m of ice -> two-way FIELD
    # factor 10**(-3 dB / 10) vs the identical unattenuated geometry
    att_ratio = (absf[(1000.0, 300.0, 3.17, 10.0)]
                 / absf[(1000.0, 300.0, 3.17, 0.0)]) / 10.0 ** (-0.3)

    # surface layer unchanged vs stage 2 on the haynes_constants e2e config
    lam_m, h_m = 2.0, 2000.0
    rc = RadarConfig(dt=20e-9, n_samples=300, t0=2.0 * (h_m - 4.0) / C,
                     f0=C / lam_m)
    slab = syn.slab_scene(surface=500.0, depth=300.0, extent=2800.0,
                          posting=25.0, n_traces=3, altitude=h_m)
    ds_multi = soundersim.simulate(slab, SimConfig(
        mode="coherent", radar=rc, facets=FacetConfig(spacing=6.25),
        media=_media(3.17),
        interfaces=[DemInterface(name="surface"), DemInterface(name="bed")]))
    flat = syn.flat_scene(altitude=h_m, n_traces=3, extent=2800.0,
                          posting=25.0)
    ds_single = soundersim.simulate(flat, SimConfig(
        mode="coherent", radar=rc, facets=FacetConfig(spacing=6.25)))
    surf = np.asarray(ds_multi.field.sel(layer="surface").values)
    single = np.asarray(ds_single.field.values)
    peak = float(np.abs(single).max())
    stage2_diff = float(np.abs(surf - single).max()) / peak
    r_curv = 6.3967e6  # mean ellipsoid curvature radius at 75 N
    ref = (fresnel_normal(1.0, 3.17) * (r_curv / (r_curv + h_m))
           * np.exp(-2j * (2.0 * np.pi / lam_m) * h_m) / (2.0 * h_m))
    tot = complex(surf[1].sum())
    surf_mag = float(abs(tot) / abs(ref))
    surf_phase = float(np.degrees(np.angle(tot / ref)))

    metrics = {
        "bed_mag_err_max": {"value": mag_err, "threshold": 0.03, "op": "<=",
                            "pass": mag_err <= 0.03},
        "bed_phase_err_max_deg": {"value": phase_err, "threshold": 3.0,
                                  "op": "<=", "pass": phase_err <= 3.0},
        "delay_bin_mismatches": {"value": bin_misses, "threshold": 0,
                                 "op": "<=", "pass": bin_misses == 0},
        "bed_dropped_frac_max": {"value": drop_max, "threshold": 1e-6,
                                 "op": "<=", "pass": drop_max <= 1e-6},
        "attenuation_law_ratio": {"value": float(att_ratio), "target": 1.0,
                                  "threshold": 0.01, "tolerance": "+-0.01",
                                  "pass": abs(att_ratio - 1.0) <= 0.01},
        "surface_stage2_max_reldiff": {"value": stage2_diff, "threshold": 1e-6,
                                       "op": "<=", "pass": stage2_diff <= 1e-6},
        "surface_e2e_mag_ratio": {"value": surf_mag, "target": 1.0,
                                  "threshold": 0.05, "tolerance": "+-0.05",
                                  "pass": abs(surf_mag - 1.0) <= 0.05},
        "surface_e2e_phase_err_deg": {"value": abs(surf_phase),
                                      "threshold": 5.0, "op": "<=",
                                      "pass": abs(surf_phase) <= 5.0},
    }
    plots.write_metrics(
        OUTDIR / "slab_absolute" / "metrics.json", "slab_absolute", metrics,
        group=GROUP,
        notes="Parameter-free absolute check of the multilayer coherent bed "
              "return vs the image-in-dielectric closed form tau_d*tau_u*"
              "Gamma_b*exp(-2jk0(h+nd))/(2(h+d/n)) (Peters et al. 2005): "
              f"sweep {labels}. Window-integrated bed-layer field, h/d from "
              "the run's own nadir geometry; scenes 45 Fresnel units wide "
              "(~2% hard-rim ringing bound), bed facets 0.09*sqrt(lam_ice*"
              "r_eff). Bed nadir delay lands in exactly the closed-form bin "
              "everywhere; the 10 dB/km point scales by the documented "
              "two-way field law 10^(-d_km*att/10); the surface layer of the "
              "haynes_constants config is bitwise stage-2 (max rel diff "
              f"{stage2_diff:.1e}) and matches its closed form.")
    plots.constants_panel(OUTDIR / "slab_absolute" / "slab_constants.png",
                          labels, mags, phases, mag_tol=0.03, phase_tol=3.0,
                          title="Flat-slab image-in-dielectric absolute checks")

    for name, m in metrics.items():
        assert m["pass"], f"{name}: {m}"


# ---------------------------------------------------------------- case 2

H2, D2 = 500.0, 60.0
T0_2, DT_2, NSAMP_2 = 3.9e-6, 1e-8, 80
EPS_ICE = 3.17
N_ICE = float(np.sqrt(EPS_ICE))
LAM_ICE = LAM / N_ICE
GAM_B = float(fresnel_normal(EPS_ICE, EPS_BED))
P2 = np.array([[0.0, 0.0, H2]])
UCT = np.array([[0.0, -1.0, 0.0]])


def _kernel_vs_referee(surf_fn, bed_fn, extent, fs, fb, rough_surface=False):
    """One twomedia sub-case: multilayer kernel vs the exact-crossing referee.

    Kernel-level planar scenes (surface_facets) so the referee shares the
    exact analytic interfaces. Returns comparison dict.
    """
    surf = surface_facets(extent, fs, surf_fn)
    bed = surface_facets(extent, fb, bed_fn, z0=-D2)
    kern, kdrop = refracted_cluttergram(
        P2, UCT, bed, [surf], [1.0, EPS_ICE], [0.0, 0.0], mode="coherent",
        t0=T0_2, dt=DT_2, n_samples=NSAMP_2, c=C, gamma=GAM_B, k0=K0)
    fine = surface_facets(extent, LAM_ICE / 8.0, bed_fn, z0=-D2)
    if rough_surface:
        r0 = snell_crossing(P2[0], fine.centers, np.zeros(3),
                            np.array([0.0, 0.0, 1.0]), 1.0, N_ICE, xp=np)
        x, _, _, _ = fermat_crossing_batch(
            P2[0], fine.centers, surf_fn, 1.0, N_ICE, x0=r0.x[:, :2],
            half0=6.0, chunk=32768)
        eps = 1e-4  # analytic normal at the crossing by central difference
        gx = (np.asarray(surf_fn(x[:, 0] + eps, x[:, 1]))
              - np.asarray(surf_fn(x[:, 0] - eps, x[:, 1]))) / (2 * eps)
        gy = (np.asarray(surf_fn(x[:, 0], x[:, 1] + eps))
              - np.asarray(surf_fn(x[:, 0], x[:, 1] - eps))) / (2 * eps)
        nrm = np.column_stack([-gx, -gy, np.ones(len(x))])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    else:
        r = snell_crossing(P2[0], fine.centers, np.zeros(3),
                           np.array([0.0, 0.0, 1.0]), 1.0, N_ICE, xp=np)
        assert bool(r.valid.all())
        x, nrm = r.x, np.array([0.0, 0.0, 1.0])
    ref, _ = two_media_trace(P2[0], fine, x, nrm, 1.0, EPS_ICE, GAM_B, K0,
                             T0_2, DT_2, NSAMP_2, C)
    k = kern[0]
    assert float(kdrop[0]) == 0.0
    ratio = complex(k.sum() / ref.sum())
    agg = 2
    n = (NSAMP_2 // agg) * agg
    pk = (np.abs(k[:n]) ** 2).reshape(-1, agg).sum(1)
    pr = (np.abs(ref[:n]) ** 2).reshape(-1, agg).sum(1)
    m = pr > pr.max() * 1e-2  # facet-scale bins above -20 dB of peak
    prof_db = float(np.abs(10.0 * np.log10(pk[m] / pr[m])).max())
    return {"mag": abs(ratio), "phase": float(np.degrees(np.angle(ratio))),
            "peak_diff": int(abs(int(np.abs(k).argmax())
                                 - int(np.abs(ref).argmax()))),
            "prof_db": prof_db, "n_prof_bins": int(m.sum()),
            "pk": pk, "pr": pr,
            "x_us": (T0_2 + np.arange(len(pk)) * agg * DT_2) * 1e6}


@pytest.mark.integration
def test_twomedia_field():
    """Coherent multilayer kernel vs the exact-crossing sub-wavelength
    referee on flat, rough-bed and rough-surface two-media scenes, plus the
    same-facet chaining-degradation sweep (M15 tie-in)."""
    flat = lambda x, y: 0.0 * x  # noqa: E731
    rough_bed = lambda x, y: 0.15 * np.sin(2 * np.pi * y / 30.0)  # noqa: E731
    rsurf = lambda a: (lambda x, y: a * np.sin(2 * np.pi * y / 40.0))
    cases = {
        "flat": _kernel_vs_referee(flat, flat, 80.0, 5.0, 2.5),
        "roughbed": _kernel_vs_referee(flat, rough_bed, 80.0, 5.0, 2.0),
        "roughsurf": _kernel_vs_referee(rsurf(0.15), flat, 60.0, 2.0, 2.5,
                                        rough_surface=True),
    }
    # degradation onset: the same rough-surface geometry at A = 0.25 m
    onset = _kernel_vs_referee(rsurf(0.25), flat, 60.0, 2.0, 2.5,
                               rough_surface=True)

    # same-facet chaining sweep: kernel's two-pass local-plane chain (f64
    # replica) vs exact Fermat on the true surface, SAME target facet centers
    # -- isolates the M15 anchoring error with no tessellation confound.
    amps = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    bed_flat = surface_facets(60.0, 2.5, flat, z0=-D2)
    rms, mx = [], []
    for a in amps:
        sf = rsurf(float(a))
        surf = surface_facets(60.0, 2.0, sf)
        _, opl_k, _ = local_plane_opl(P2[0], bed_flat.centers, surf, 1.0,
                                      N_ICE)
        r0 = snell_crossing(P2[0], bed_flat.centers, np.zeros(3),
                            np.array([0.0, 0.0, 1.0]), 1.0, N_ICE, xp=np)
        _, _, _, opl_e = fermat_crossing_batch(
            P2[0], bed_flat.centers, sf, 1.0, N_ICE, x0=r0.x[:, :2],
            half0=max(6.0, 4.0 * float(a)))
        derr = np.abs(opl_k - opl_e)
        rms.append(float(np.sqrt((derr ** 2).mean())))
        mx.append(float(derr.max()))
    slope = float(np.polyfit(np.log(amps), np.log(rms), 1)[0])

    metrics = {}
    tol = {"flat": (0.02, 2.0), "roughbed": (0.02, 2.0),
           "roughsurf": (0.02, 2.5)}
    for name, r in cases.items():
        mt, pt = tol[name]
        metrics[f"{name}_mag_ratio"] = {
            "value": r["mag"], "target": 1.0, "threshold": mt,
            "tolerance": f"+-{mt}", "pass": abs(r["mag"] - 1.0) <= mt}
        metrics[f"{name}_phase_err_deg"] = {
            "value": abs(r["phase"]), "threshold": pt, "op": "<=",
            "pass": abs(r["phase"]) <= pt}
        metrics[f"{name}_profile_maxdb"] = {
            "value": r["prof_db"], "threshold": 1.5, "op": "<=",
            "pass": r["prof_db"] <= 1.5, "agg_bins": 2,
            "gated_bins": r["n_prof_bins"]}
        metrics[f"{name}_peak_bin_diff"] = {
            "value": r["peak_diff"], "threshold": 0, "op": "<=",
            "pass": r["peak_diff"] == 0}
    metrics["onset_mag_ratio_A0.25"] = {
        "value": onset["mag"], "target": 1.0, "threshold": 0.10,
        "tolerance": "+-0.10", "pass": abs(onset["mag"] - 1.0) <= 0.10,
        "note": "degradation onset, loosely gated"}
    metrics["onset_phase_err_deg_A0.25"] = {
        "value": abs(onset["phase"]), "threshold": 15.0, "op": "<=",
        "pass": abs(onset["phase"]) <= 15.0}
    metrics["chaining_opl_rms_m_A0.25"] = {
        "value": rms[0], "threshold": 5e-3, "op": "<=",
        "pass": rms[0] <= 5e-3}
    metrics["chaining_scaling_exponent"] = {
        "value": slope, "target": 2.4, "threshold": 1.0, "tolerance": "+-1.0",
        "pass": abs(slope - 2.4) <= 1.0,
        "note": "RMS opl error vs A, log-log"}
    plots.write_metrics(
        OUTDIR / "twomedia_field" / "metrics.json", "twomedia_field", metrics,
        group=GROUP,
        notes="Multilayer kernel vs exact-crossing sub-wavelength (lam_ice/8) "
              "two-media brute-force referee, h=500 m over a 60 m slab, "
              "195 MHz: flat slab (80 m aperture), rough bed A=0.15 m "
              "Lambda=30 m under flat surface, flat bed under rough surface "
              "A=0.15 m Lambda=40 m (60 m aperture). Window-integrated field "
              "gates (per the M16 finding fine-dt per-bin splits are binning "
              "quantization) + facet-scale (agg=2) profile above -20 dB. "
              "Rough-surface degradation: at A=0.25 m the full-field error is "
              f"{onset['mag']:.3f}/{onset['phase']:+.1f} deg; the same-facet "
              "f64 chain-vs-Fermat sweep (no tessellation confound) has opl "
              f"error RMS {rms[0]:.2e} m at A=0.25 growing to {rms[-1]:.2e} m "
              f"at A=4 (log-log slope {slope:+.2f}; M15 anchoring error "
              "A*k^2*delta^2/2 with the effective anchor offset delta itself "
              "growing once the pass-1 mean-plane crossing error exceeds a "
              "facet).")
    plots.referee_profile_panels(
        OUTDIR / "twomedia_field" / "kernel_vs_referee.png",
        [{"title": t, "x": cases[n]["x_us"], "kernel": cases[n]["pk"],
          "referee": cases[n]["pr"]}
         for n, t in (("flat", "flat slab"),
                      ("roughbed", "rough bed (A=0.15 m)"),
                      ("roughsurf", "rough surface (A=0.15 m)"))])
    plots.chaining_error_panel(
        OUTDIR / "twomedia_field" / "chaining_error.png", amps, rms, mx,
        slope=slope)

    for name, m in metrics.items():
        assert m["pass"], f"{name}: {m}"


# ---------------------------------------------------------------- case 3


@pytest.mark.integration
def test_bed_falloff():
    """Coherent nadir bed power falls off as (h + d/n)^-2: altitude sweep at
    fixed depth and depth sweep at fixed altitude (the Haynes family extended
    below the surface with the refraction-corrected effective range)."""
    eps = 3.17
    sweeps = {
        "altitude": [(h, 300.0) for h in (500.0, 1000.0, 2000.0, 4000.0)],
        "depth": [(800.0, d) for d in (100.0, 500.0, 1000.0, 2000.0)],
    }
    slopes, level_err = {}, 0.0
    panels = {}
    for name, pts in sweeps.items():
        reff, meas, pred = [], [], []
        for h, d in pts:
            ds = _slab_run(h, d, eps)
            r = _bed_vs_closed_form(ds, eps)
            assert all(r["bin_ok"])
            reff.append(float(np.mean(r["reff"])))
            meas.append(float(np.mean(np.array(r["absf"]) ** 2)))
            pred.append(float(np.mean(r["pred"])))
        slopes[name] = float(np.polyfit(np.log(reff), np.log(meas), 1)[0])
        level_err = max(level_err, float(
            np.abs(np.array(meas) / np.array(pred) - 1.0).max()))
        panels[name] = (reff, meas, pred)

    metrics = {
        "altitude_sweep_slope": {
            "value": slopes["altitude"], "target": -2.0, "threshold": 0.05,
            "tolerance": "+-0.05",
            "pass": abs(slopes["altitude"] + 2.0) <= 0.05},
        "depth_sweep_slope": {
            "value": slopes["depth"], "target": -2.0, "threshold": 0.05,
            "tolerance": "+-0.05", "pass": abs(slopes["depth"] + 2.0) <= 0.05},
        "abs_level_max_err": {"value": level_err, "threshold": 0.05,
                              "op": "<=", "pass": level_err <= 0.05},
    }
    plots.write_metrics(
        OUTDIR / "bed_falloff" / "metrics.json", "bed_falloff", metrics,
        group=GROUP,
        notes="Flat surface + flat bed, coherent nadir bed |field|^2 vs the "
              "closed form's effective range h + d/n: altitude sweep "
              "500-4000 m at d=300 m and depth sweep 100-2000 m at h=800 m "
              "(eps_ice=3.17). Both fit slope -2 in log(r_eff) and the "
              "absolute level is the parameter-free closed form (extends the "
              "-4/-3/-2 Haynes family below the surface). Incoherent "
              "rough-bed r^-3 analog omitted (runtime; the incoherent "
              "multilayer path shares this geometry pipeline and is gated by "
              "the CI slab/attenuation/energy tests).")
    for name, (reff, meas, pred) in panels.items():
        plots.effective_range_falloff_panel(
            OUTDIR / "bed_falloff" / f"falloff_{name}.png", reff, meas, pred,
            slope=slopes[name],
            title=f"Bed fall-off, {name} sweep (slope {slopes[name]:+.3f})")

    for name, m in metrics.items():
        assert m["pass"], f"{name}: {m}"
