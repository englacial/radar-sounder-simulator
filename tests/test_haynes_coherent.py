"""Haynes et al. 2018 coherent benchmark suite (M12, integration).

Four report cases under "Radar equation comparison" validating the coherent
kernel and its M9 absolute normalization (field convention in
claude_notes/coherent_normalization.md) against the paper's closed forms:

1. haynes_r2_smooth -- flat-scene altitude sweep through simulate() in
   coherent mode: leading-edge |field|^2 falls off as r^-2 (completing the
   -4/-3/-2 triad of Haynes Table I), plus a parameter-free absolute-level
   check of the same window against the Fresnel-ring closed form.
2. haynes_constants -- absolute amplitude/phase checks: kernel tapered-plate
   field vs the image-method Gamma*exp(-2jkh)/(2h); first-Fresnel-zone disk =
   4x plate power (Eq. 16-17); end-to-end simulate() total field vs the same
   closed form.
3. haynes_coherence_loss -- Gaussian rough-surface ensembles over
   Fresnel-zone disks vs I_o*(1 - L) with the Eq. (35) coherence-loss series
   (Fig. 5 reproduction).
4. haynes_speckle -- rough-surface ensemble speckle statistics: amplitude
   Rayleigh / power exponential.

Design constraints carried from measured findings (see test_coherent_kernel /
test_cross_kernel docstrings):

- LPA facet-size envelope: benchmark facets are kept <= ~0.1*sqrt(lam*r_min)
  so the LPA amplitude error is ~1% (5% breakdown is at 0.23*sqrt(lam*r)).
- No per-range-bin coherent observables on smooth surfaces: a smooth-surface
  range bin is a cancellation-dominated Fresnel-ring integral whose value is
  both bin-width-sensitive (vanishing when the bin range-width is a multiple
  of lam/2) and facet-quantization-sensitive (the facet range extent
  L*sin(theta) smears the ring boundary); the deterministic cases therefore
  gate on total fields, and the ensemble case needs no range binning.

All randomness is seeded with fixed constants. Thresholds were set from the
first run (per repo convention); measured values are recorded inline.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.stats

from soundersim import simulate
from soundersim.compare import plots
from soundersim.compare.brute_force import _contributions
from soundersim.compare.haynes import (
    coherence_loss,
    fresnel_radius,
    gaussian_surface,
    mean_power,
    noise_floor_power,
)
from soundersim.config import FacetConfig, RadarConfig, SimConfig
from soundersim.kernels.coherent import coherent_cluttergram
from soundersim.physics import fresnel_normal
from soundersim.synthetic import flat_scene

OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "verification"
GROUP = "Radar equation comparison"
C = 299792458.0
GAMMA_SIM = float(fresnel_normal(1.0, 3.17))  # what simulate() derives (-0.2807)
GAMMA = -0.281                                # kernel-level tests (any value works)
R_CURV = 6.3967e6  # mean ellipsoid curvature radius at 75 N (sqrt(M*N))
UCT = np.array([[0.0, -1.0, 0.0]])


# ---------------------------------------------------------------- case 1


@pytest.mark.integration
def test_haynes_r2_smooth():
    """Coherent nadir return falls off as r^-2, absolute level vs image method.

    Flat scene through simulate() in coherent mode, lam = 2 m. The coherent
    "leading edge" of a smooth surface IS the total nadir Fresnel-zone return
    (Haynes Table I coherent rows): all Fresnel rings beyond the first cancel
    and the total field is the image-method Gamma*(R/(R+h))*exp(-2jkh)/(2h).
    The per-trace total field (sum of ds.field over the full twtt window,
    which covers the entire scene) is therefore the observable; its power
    completes the -4/-3/-2 triad next to the stage-1 incoherent sweep.

    (A per-range-bin leading window -- the stage-1 method -- is NOT usable
    coherently: a range bin on a smooth surface is a cancellation-dominated
    Fresnel-ring integral whose resultant depends on the sharpness of the bin
    boundary in range, and facet-center bin assignment smears that boundary
    by the facet range extent L*sin(theta) -- an altitude-dependent
    suppression measured at up to ~100x in bin 8 at 500 m. The binned-field
    granularity note in test_coherent_kernel.py is the same effect.)

    Scene sizing: extent = 45*sqrt(lam*h) so the square hard rim's Fresnel-
    integral ringing is a constant ~2% of the field at every altitude
    (u = extent/sqrt(lam*h); a DEM cannot carry the raised-cosine taper the
    kernel-level constants case uses); facets extent/512 = 0.088*sqrt(lam*h)
    (~0.7% LPA error, finding (1)).

    Measured on first run: slope -1.992, max |level ratio - 1| = 0.030.
    """
    lam = 2.0
    k = 2.0 * np.pi / lam
    altitudes = [500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    dt = 40e-9

    r_fit, lead, pred = [], [], []
    for h in altitudes:
        ext = 45.0 * np.sqrt(lam * h)
        scene = flat_scene(altitude=h, n_traces=3, extent=ext,
                           posting=ext / 128.0)
        r_far = np.sqrt(h * h + 2.0 * (ext / 2.0) ** 2)  # corner range (flat)
        t0 = 2.0 * (h - 5.0) / C
        n_samples = int(np.ceil(2.0 * (r_far - h + 70.0) / C / dt)) + 4
        rc = RadarConfig(dt=dt, n_samples=n_samples, t0=t0, f0=C / lam)
        cfg = SimConfig(mode="coherent", radar=rc,
                        facets=FacetConfig(spacing=ext / 512.0))
        ds = simulate(scene, cfg)
        total = np.asarray(ds.power.values, np.float64).sum()
        assert float(ds.dropped_power.values.max()) <= 1e-9 * total
        f_tot = np.asarray(ds.field.values).sum(axis=1)  # per-trace total field
        lead.append(float((np.abs(f_tot) ** 2).mean()))
        r_fit.append(float(ds.first_return_twtt.values.mean()) * C / 2.0)
        pred.append(abs(GAMMA_SIM * (R_CURV / (R_CURV + h)) / (2.0 * h)) ** 2)

    r_fit, lead, pred = np.array(r_fit), np.array(lead), np.array(pred)
    slope = float(np.polyfit(np.log(r_fit), np.log(lead), 1)[0])
    level_err = float(np.abs(lead / pred - 1.0).max())

    metrics = {
        "coherent_nadir_r2_slope": {"value": slope, "target": -2.0,
                                    "threshold": 0.1, "tolerance": "+-0.1",
                                    "pass": abs(slope + 2.0) <= 0.1},
        "abs_level_max_err": {"value": level_err, "threshold": 0.10,
                              "op": "<=", "pass": level_err <= 0.10},
    }
    plots.write_metrics(
        OUTDIR / "haynes_r2_smooth" / "metrics.json", "haynes_r2_smooth",
        metrics, group=GROUP,
        notes=f"Coherent flat-scene sweep, altitudes {altitudes} m, lam=2 m, "
              f"scenes 45*sqrt(lam*h) wide with 0.088*sqrt(lam*h) facets. "
              f"Total nadir coherent power fits r^-2 ({slope:+.3f}), "
              f"completing the -4/-3/-2 triad (stage-1 'haynes' case has -4/"
              f"-3); absolute level matches the parameter-free image-method "
              f"|Gamma*(R/(R+h))/(2h)|^2 to {level_err:.1%} (~2% field "
              f"hard-rim ringing bound, documented).")
    plots.coherent_r2_panel(OUTDIR / "haynes_r2_smooth" / "r2_nadir_return.png",
                            r_fit, lead, pred, slope=slope)

    assert abs(slope + 2.0) <= 0.1, f"slope {slope}"
    assert level_err <= 0.10, f"absolute level error {level_err}"


# ---------------------------------------------------------------- case 2


def _grid_facets(half_extent, d):
    """Square facet grid (spacing d, z=0, normals +z) centered on the origin."""
    n = int(np.ceil(2.0 * half_extent / d))
    ax = (np.arange(n) - (n - 1) / 2.0) * d
    X, Y = np.meshgrid(ax, ax)
    centers = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    m = len(centers)
    normals = np.tile([0.0, 0.0, 1.0], (m, 1))
    e1 = np.tile([d, 0.0, 0.0], (m, 1))
    e2 = np.tile([0.0, d, 0.0], (m, 1))
    return centers, normals, e1, e2


def _disk_kernel_field(h, radius, d, k, gamma, taper_start=None):
    """Total coherent-kernel field of a facet disk seen from (0, 0, h).

    Facets keep their center inside ``radius``; ``taper_start`` applies the
    raised-cosine area taper of brute_force.flat_disk_samples (suppresses the
    non-convergent hard-rim Fresnel ringing for the image-method comparison).
    Window covers the whole disk; asserts nothing is dropped.
    """
    centers, normals, e1, e2 = _grid_facets(radius, d)
    rho = np.hypot(centers[:, 0], centers[:, 1])
    keep = rho <= radius
    centers, normals, e1, e2, rho = (a[keep] for a in
                                     (centers, normals, e1, e2, rho))
    areas = np.full(len(centers), d * d)
    if taper_start is not None:
        edge = rho > taper_start
        areas[edge] *= 0.5 * (1.0 + np.cos(np.pi * (rho[edge] - taper_start)
                                           / (radius - taper_start)))
    r_max = np.sqrt(h * h + radius * radius)
    t0 = 2.0 * (h - 2.0) / C
    dt = 2.0 * 2.0 / C
    n_samples = int(np.ceil((r_max - h + 4.0) / 2.0)) + 3
    field, dropped = coherent_cluttergram(
        np.array([[0.0, 0.0, h]]), UCT, centers, normals, areas, e1, e2,
        k=k, gamma=gamma, t0=t0, dt=dt, n_samples=n_samples, c=C)
    assert dropped[0] == 0.0
    return complex(field[0].sum())


@pytest.mark.integration
def test_haynes_constants():
    """Absolute amplitude checks vs Haynes closed forms (no fitted constants).

    (a) Tapered plate through the coherent kernel (lam = 1, 2*lam facets,
        raised-cosine taper over Fresnel zones 6..22 -- the brute-force-
        validated approach; a hard-edged plate never converges) at h = 1000,
        2000, 4000 lam vs the image method Gamma*exp(-2jkh)/(2h).
    (b) Hard-edged first-Fresnel-zone disk (radius sqrt(lam*h/2), 1*lam
        facets) = 4x the infinite-plate power (Eq. 16-17).
    (c) End-to-end simulate(): flat scene 2800 m square (u = L/sqrt(lam*h) ~
        44 -> hard-edge square-rim ringing ~2% of the field), lam = 2 m,
        h = 2000 m, 6.25 m facets; total field of the nadir trace vs
        Gamma*(R/(R+h))*exp(-2jkh)/(2h). Hard-edge ringing and f32 phase are
        inside the 5%/5 deg gates; the square-plate Fresnel-integral rim
        contribution DOES converge (unlike a disk's), just slowly.

    Measured on first run: plate mag err <= 0.0010, phase err <= 0.30 deg;
    Fresnel-zone ratio 0.9979; e2e mag ratio 0.9881, phase err 2.03 deg.
    """
    lam = 1.0
    k = 2.0 * np.pi / lam

    # (a) tapered plate at several altitudes
    labels, mag_ratio, phase_deg = [], [], []
    for h in (1000.0, 2000.0, 4000.0):
        f = _disk_kernel_field(h, np.sqrt(22.0 * lam * h), 2.0 * lam, k, GAMMA,
                               taper_start=np.sqrt(6.0 * lam * h))
        ref = GAMMA * np.exp(-2j * k * h) / (2.0 * h)
        labels.append(f"plate h={h:.0f} lam")
        mag_ratio.append(abs(f) / abs(ref))
        phase_deg.append(float(np.degrees(np.angle(f / ref))))
    plate_mag_err = float(np.abs(np.array(mag_ratio) - 1.0).max())
    plate_phase_err = float(np.abs(phase_deg).max())

    # (b) first-Fresnel-zone disk -> 4x plate power
    h = 2000.0
    f_fz = _disk_kernel_field(h, fresnel_radius(lam, h), 1.0 * lam, k, GAMMA)
    plate_p = abs(GAMMA / (2.0 * h)) ** 2
    fz_ratio = float(abs(f_fz) ** 2 / (4.0 * plate_p))

    # (c) end-to-end simulate() total field over a large flat scene
    lam_m, h_m = 2.0, 2000.0
    k_m = 2.0 * np.pi / lam_m
    scene = flat_scene(altitude=h_m, n_traces=3, extent=2800.0, posting=25.0)
    rc = RadarConfig(dt=20e-9, n_samples=300, t0=2.0 * (h_m - 4.0) / C,
                     f0=C / lam_m)
    cfg = SimConfig(mode="coherent", radar=rc, facets=FacetConfig(spacing=6.25))
    ds = simulate(scene, cfg)
    assert float(ds.dropped_power.values.max()) == 0.0
    tot = complex(np.asarray(ds.field.values)[1].sum())  # middle (nadir) trace
    ref = complex(GAMMA_SIM * (R_CURV / (R_CURV + h_m))
                  * np.exp(-2j * k_m * h_m) / (2.0 * h_m))
    e2e_mag = float(abs(tot) / abs(ref))
    e2e_phase = float(np.degrees(np.angle(tot / ref)))
    labels.append("simulate() h=2000 m")
    mag_ratio.append(e2e_mag)
    phase_deg.append(e2e_phase)

    metrics = {
        "plate_mag_err_max": {"value": plate_mag_err, "threshold": 0.02,
                              "op": "<=", "pass": plate_mag_err <= 0.02},
        "plate_phase_err_max_deg": {"value": plate_phase_err, "threshold": 2.0,
                                    "op": "<=", "pass": plate_phase_err <= 2.0},
        "fresnel_zone_4x_ratio": {"value": fz_ratio, "target": 1.0,
                                  "threshold": 0.05, "tolerance": "+-0.05",
                                  "pass": abs(fz_ratio - 1.0) <= 0.05},
        "e2e_mag_ratio": {"value": float(e2e_mag), "target": 1.0,
                          "threshold": 0.05, "tolerance": "+-0.05",
                          "pass": abs(e2e_mag - 1.0) <= 0.05},
        "e2e_phase_err_deg": {"value": abs(e2e_phase), "threshold": 5.0,
                              "op": "<=", "pass": abs(e2e_phase) <= 5.0},
    }
    plots.write_metrics(
        OUTDIR / "haynes_constants" / "metrics.json", "haynes_constants",
        metrics, group=GROUP,
        notes="Absolute (parameter-free) amplitude checks of the M9 "
              "normalization: kernel tapered plate vs image method "
              "Gamma*exp(-2jkh)/(2h) at 3 altitudes; hard-edged first-Fresnel-"
              "zone disk = 4x plate power (Haynes Eq. 16-17); end-to-end "
              "simulate() 2800 m flat scene at 2000 m AGL (hard square rim -> "
              "~2% Fresnel ringing, documented; raised-cosine taper is not "
              "expressible through a DEM).")
    plots.constants_panel(OUTDIR / "haynes_constants" / "constants.png",
                          labels, mag_ratio, phase_deg, mag_tol=0.05,
                          phase_tol=5.0)

    assert plate_mag_err <= 0.02
    assert plate_phase_err <= 2.0
    assert abs(fz_ratio - 1.0) <= 0.05
    assert abs(e2e_mag - 1.0) <= 0.05
    assert abs(e2e_phase) <= 5.0


# ---------------------------------------------------------------- case 3

SIGMAS = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
H_LIST = (2000.0, 8000.0)
L_LIST = (2.0, 8.0)
N_REAL = 200
DX = 0.25  # lam/4 grid (Haynes used lam/2); noise floor 4x lower than Fig. 5


def _ensemble_mean_power(h, l, seed, lam=1.0, gamma=GAMMA):
    """<|field|^2> over N_REAL Gaussian-rough Fresnel-zone disks vs SIGMAS.

    Haynes's own numerical method (Eq. 29-31) with the brute-force integrand:
    fixed lam/4 grid out to the flat-surface Fresnel radius, correlated
    surface z = sigma * u with one unit-variance realization u shared across
    the sigma sweep (float64 throughout). Vertical normals / cell areas match
    the paper's phase-integral convention; the amplitude terms cos(theta)/r^2
    differ from 1/h^2 by O(lam/h) over the disk.
    """
    k = 2.0 * np.pi / lam
    rf = fresnel_radius(lam, h)
    n = int(np.ceil(2.0 * rf / DX))
    ax = (np.arange(n) - (n - 1) / 2.0) * DX
    X, Y = np.meshgrid(ax, ax)
    keep = (X ** 2 + Y ** 2).ravel() <= rf * rf
    rho2 = (X ** 2 + Y ** 2).ravel()[keep]
    pref = 1j * (k / (2.0 * np.pi)) * gamma * DX * DX
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(SIGMAS))
    for _ in range(N_REAL):
        u = gaussian_surface(n, DX, l, rng).ravel()[keep]
        z = SIGMAS[:, None] * u[None, :]          # (n_sigma, n_pts)
        dz = h - z
        r = np.sqrt(rho2[None, :] + dz * dz)
        f = (pref * (dz / r) * np.exp(-2j * k * r) / (r * r)).sum(axis=1)
        acc += np.abs(f) ** 2
    return acc / N_REAL


@pytest.mark.integration
def test_haynes_coherence_loss():
    """Rough-surface coherence loss vs Haynes Eq. (34)-(36) (Fig. 5).

    Ensembles of N=200 seeded realizations per (h, l): h in {2000, 8000} lam,
    Gaussian correlation lengths l in {2, 8} lam, sigma_h/lam in [0, 0.5].
    Measured ensemble <|field|^2> vs analytic (Gamma^2/h^2)(1 - L) with the
    Eq. (35) series PLUS the Eq. 115 discretization noise floor (the floor is
    up to ~20% of the analytic once 1-L is ~40 dB down; Haynes plots it for
    the same reason). Gates cover the coherent-dominated region
    (sigma <= 0.15 lam) and the sigma ~ lam/4 transition (0.20-0.30 lam,
    where the ensemble is diffuse-dominated: s.e. of the mean ~ 1/sqrt(200) =
    0.3 dB, 12 gated points); sigma = 0.4, 0.5 are recorded but not gated
    (Kirchhoff validity edge l >~ sigma_h, per the paper -- residuals there
    measured <= +1.1 dB, systematically positive like the paper's own Fig. 5
    dots near the floor). Also gates the sigma = 0 absolute level (= 4x plate
    = Gamma^2/h^2) and the 12.04 dB R^2 altitude drop 2000 -> 8000 lam.

    Thresholds set from the first run; measured: coherent-region residual
    0.25 dB, transition residual 1.02 dB (worst at h=8000, l=2, sigma=0.25),
    sigma0 err 0.00025, falloff 12.040 dB.
    """
    lam, gamma = 1.0, GAMMA
    curves, meas0 = {}, {}
    for hi, h in enumerate(H_LIST):
        for li, l in enumerate(L_LIST):
            meas = _ensemble_mean_power(h, l, seed=31_000 + 100 * hi + li)
            ana = np.array([mean_power(h, s, l, lam, gamma) for s in SIGMAS])
            curves[(h, l)] = (meas, ana)
        meas0[h] = curves[(h, L_LIST[0])][0][0]

    coh = SIGMAS <= 0.15
    tra = (SIGMAS >= 0.20) & (SIGMAS <= 0.30)
    # forward model = analytic + the Eq. 115 discretization noise floor (up to
    # ~20% of the analytic power once 1-L is 40 dB down at sigma >= 0.4)
    res_db = {(h, l): 10.0 * np.log10(m / (a + noise_floor_power(DX, 1.0, h,
                                                                 GAMMA)))
              for (h, l), (m, a) in curves.items()}
    coh_max = float(max(np.abs(r[coh]).max() for r in res_db.values()))
    tra_max = float(max(np.abs(r[tra]).max() for r in res_db.values()))
    sigma0_err = float(max(abs(curves[(h, l)][0][0] / (gamma ** 2 / h ** 2) - 1.0)
                           for h in H_LIST for l in L_LIST))
    falloff_db = float(10.0 * np.log10(meas0[H_LIST[0]] / meas0[H_LIST[1]]))
    falloff_ref = float(20.0 * np.log10(H_LIST[1] / H_LIST[0]))  # 12.04 dB (R^2)

    metrics = {
        "coherent_region_residual_db_max": {
            "value": coh_max, "threshold": 0.6, "op": "<=",
            "pass": coh_max <= 0.6, "region": "sigma_h <= 0.15 lam"},
        "transition_residual_db_max": {
            "value": tra_max, "threshold": 1.5, "op": "<=",
            "pass": tra_max <= 1.5, "region": "sigma_h 0.20-0.30 lam"},
        "sigma0_abs_err_max": {
            "value": sigma0_err, "threshold": 0.02, "op": "<=",
            "pass": sigma0_err <= 0.02},
        "r2_falloff_db_sigma0": {
            "value": falloff_db, "target": round(falloff_ref, 2),
            "threshold": 0.3, "tolerance": "+-0.3",
            "pass": abs(falloff_db - falloff_ref) <= 0.3},
    }
    plots.write_metrics(
        OUTDIR / "haynes_coherence_loss" / "metrics.json",
        "haynes_coherence_loss", metrics, group=GROUP,
        notes=f"Gaussian rough Fresnel-zone disks, N={N_REAL} seeded "
              f"realizations, lam/4 grid, h in {list(H_LIST)} lam, l in "
              f"{list(L_LIST)} lam, sigma_h/lam in {SIGMAS.tolist()}. "
              "Ensemble <|field|^2> vs Haynes Eq. (34)-(36) analytic "
              "I_o*(1-L) plus the Eq. 115 discretization noise floor; "
              "sigma_h = 0.4, 0.5 recorded but ungated (Kirchhoff validity "
              "l >~ sigma_h marginal there, per the paper).")
    panels = []
    for h in H_LIST:
        panels.append({
            "title": f"altitude = {h:.0f} lam",
            "curves": [(f"l = {l:.0f} lam", curves[(h, l)][0],
                        curves[(h, l)][1]) for l in L_LIST],
            "l0": np.array([gamma ** 2 / h ** 2
                            * (1.0 - coherence_loss(h, s, 0.0, lam))
                            for s in SIGMAS]),
            "floor": noise_floor_power(DX, lam, h, gamma),
        })
    plots.coherence_loss_panels(
        OUTDIR / "haynes_coherence_loss" / "coherence_loss.png", SIGMAS, panels)

    assert coh_max <= 0.6, f"coherent-region residual {coh_max} dB"
    assert tra_max <= 1.5, f"transition residual {tra_max} dB"
    assert sigma0_err <= 0.02
    assert abs(falloff_db - falloff_ref) <= 0.3


# ---------------------------------------------------------------- case 4


@pytest.mark.integration
def test_haynes_speckle():
    """Speckle statistics of a very rough surface: Rayleigh / exponential.

    1000 seeded realizations of a 20x20 lam patch (lam/2 samples, h = 150 lam)
    with iid vertical offsets sigma_h = 2 lam -- iid per the cross-kernel
    finding (correlated heights leave residual coherence), sigma_h >> lam so
    phases are uniform. Per-realization total complex field (single-look):
    amplitude Rayleigh, power exponential, no coherent mean component. The
    lam/2-multiple bin-width rule is not needed here (no range binning; the
    statistic is the full-patch field). Brute-force integrand (f64) is used
    per realization; kernel speckle is already tied to the incoherent kernel
    by the cross-kernel ensemble test.

    Measured on first run: power std/mean 0.9922, Rayleigh moment ratio
    0.7848 (pi/4 = 0.7854), KS stat 0.0218, coherent fraction 1.2e-4.
    """
    lam, h, sigma_h, n_real = 1.0, 150.0, 2.0, 1000
    k = 2.0 * np.pi / lam
    npts = 40
    ax = (np.arange(npts) - (npts - 1) / 2.0) * 0.5 * lam
    X, Y = np.meshgrid(ax, ax)
    base = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    m = len(base)
    rng = np.random.default_rng(7)
    pts = np.broadcast_to(base, (n_real, m, 3)).copy()
    pts[:, :, 2] = sigma_h * rng.standard_normal((n_real, m))
    contrib, _ = _contributions(np.array([0.0, 0.0, h]), pts,
                                np.array([0.0, 0.0, 1.0]),
                                np.float64(0.25), k, GAMMA)
    fields = contrib.sum(axis=1)
    amp = np.abs(fields)
    power = amp ** 2

    std_over_mean = float(power.std() / power.mean())
    ray_ratio = float(amp.mean() ** 2 / (amp ** 2).mean())  # pi/4 for Rayleigh
    ks = float(scipy.stats.kstest(power / power.mean(), "expon").statistic)
    coh_frac = float(abs(fields.mean()) ** 2 / power.mean())

    metrics = {
        "power_std_over_mean": {"value": std_over_mean, "target": 1.0,
                                "threshold": 0.1, "tolerance": "+-0.1",
                                "pass": abs(std_over_mean - 1.0) <= 0.1},
        "rayleigh_moment_ratio": {"value": ray_ratio,
                                  "target": round(np.pi / 4.0, 4),
                                  "threshold": 0.03, "tolerance": "+-0.03",
                                  "pass": abs(ray_ratio - np.pi / 4.0) <= 0.03},
        "ks_stat_power_exponential": {"value": ks, "threshold": 0.05,
                                      "op": "<=", "pass": ks <= 0.05},
        "coherent_power_fraction": {"value": coh_frac, "threshold": 0.01,
                                    "op": "<=", "pass": coh_frac <= 0.01},
    }
    plots.write_metrics(
        OUTDIR / "haynes_speckle" / "metrics.json", "haynes_speckle", metrics,
        group=GROUP,
        notes=f"{n_real} seeded iid-rough realizations (sigma_h = 2 lam, "
              "20x20 lam patch at 150 lam): single-look field amplitude is "
              "Rayleigh / power exponential (KS on mean-normalized power vs "
              "unit exponential), with no coherent mean component.")
    plots.speckle_panels(OUTDIR / "haynes_speckle" / "speckle.png", amp, power)

    assert abs(std_over_mean - 1.0) <= 0.1
    assert abs(ray_ratio - np.pi / 4.0) <= 0.03
    assert ks <= 0.05
    assert coh_frac <= 0.01
