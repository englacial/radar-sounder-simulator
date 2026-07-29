"""1-D gate for the proposed complex-r (H1cr) run -- VERDICT: FAILED, not wired.

Question: does the h1eff construction's discarding of each segment's TMM
reflection PHASE (it matches |r| only) explain the ~2-3 dB deep-band (80-120 m)
overshoot of firn_N40_h1eff vs measured?

Method: realize the complex segment r by dithering each internal interface k by
a sub-wavelength depth offset delta_k, so that the model reflector's phasor at
a common reference plane equals the full-resolution segment's. Then compare the
coherent, pulse-weighted 1-D depth response of (a) the raw 1 mm profile,
(b) the undithered h1eff stack, (c) the dithered stack -- one identical
estimator (hann-weighted 30 MHz compressed pulse, mean power over band,
dB rel the air-firn surface Fresnel power), true velocity structure throughout.

Both TMM sign conventions are derived FROM THE CODE numerically (buried-slab
experiment), and the dither is self-checked against its target complex value
(residual 2e-13).

RESULTS (2026-07-29, N=40, lam 1.5374 m):

  stack                         20-70m      80-120m    (dB rel surface)
  full-res (1 mm)               -17.37      -24.27
  h1eff (|r| only)              -16.59      -24.36     (delta vs full-res -0.09)
  h1eff_cr (complex r)          -16.84      -22.86     (delta vs full-res +1.40)
  h1eff_cr (no ref shift)       -17.54      -23.37     (delta vs full-res +0.89)
  point-sampled N40             -27.53      -38.66

  dither: max |delta| 24.8 cm, median 13.5 cm, all within lam_local/4.

VERDICT: the |r|-only stack ALREADY reproduces the full-resolution deep band to
0.09 dB; there is no discarded-phase energy to recover. A 4.42 m in-firn range
cell holds only ~1.45 layers at N=40, so there is almost nothing to decohere
(coherent vs incoherent summation of the same stack differs by only 0.6-1.0 dB).
Correctly realizing the segment phase moves the deep band 1.4 dB the WRONG WAY.
The deep-band overshoot is not segment phase -- see the two diagnostics at the
bottom: (1) the firn media carry NO attenuation (firn_cfg), worth 2.2-3.6 dB
two-way at 120 m vs 0.8-1.4 dB at 45 m for 9-15 dB/km, i.e. a 1.4-2.3 dB
band differential in exactly the observed direction; (2) the full-res 1-D
itself sits ~8 dB above measured in 80-120 m but only ~1 dB above in 20-70 m,
and only ~1 dB of that is mm-scale densitometer noise.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_firn_investigation as rfi  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_b26_comparison", ROOT / "tools" / "run_b26_comparison.py")
rb = importlib.util.module_from_spec(_spec)
sys.modules["run_b26_comparison"] = rb
_spec.loader.exec_module(rb)

C, F0, B, N = 299792458.0, 195e6, 30e6, 40
LAM = C / F0
K0 = 2 * np.pi / LAM
BANDS = ((20.0, 70.0), (80.0, 120.0))

z, n_full = rb._b26_raw_index()
dz = float(np.median(np.diff(z)))
n_surf = np.sqrt(rfi.point_eps(0.2))
GS = ((1 - n_surf) / (1 + n_surf)) ** 2           # air->firn power, -15.8 dB

# --- pulse: hann-weighted 30 MHz compressed response, 0.5 ns grid -----------
DT = 0.5e-9
_f = np.linspace(-B / 2, B / 2, 4001)
_w = 0.5 * (1 + np.cos(2 * np.pi * _f / B))
_t = np.arange(-400, 401) * DT
PULSE = (_w[None, :] * np.exp(2j * np.pi * _f[None, :] * _t[:, None])).sum(1).real
PULSE /= PULSE.max()


# ========================================================================
# 1. TMM conventions, read numerically out of rb._tmm_r
# ========================================================================
def tmm_conventions():
    """(sign of r vs (n_a-n_b)/(n_a+n_b), sign of the depth-shift phase)."""
    na, nb = 1.5, 1.8
    r0 = rb._tmm_r(np.array([na, na, nb]), dz, LAM)
    s_r = float(np.sign(r0.real / ((na - nb) / (na + nb))))
    m = 200
    rd = rb._tmm_r(np.concatenate([[na], np.full(m, na), [nb]]), dz, LAM)
    s_p = float(np.sign(np.angle(rd / r0)
                        / np.angle(np.exp(2j * K0 * na * m * dz))))
    return s_r, s_p


SIGN_R, SIGN_PROP = tmm_conventions()


# ========================================================================
# 2. stacks, optical phase, dither
# ========================================================================
depths = rfi.equal_depths(N)
eps_eff, r_abs = rb.effective_contrast_eps(depths, LAM)
r_seg = rb.segment_reflectivity(depths, LAM, complex_r=True)
n_eff = np.sqrt(eps_eff)
r_model = SIGN_R * (n_eff[:-1] - n_eff[1:]) / (n_eff[:-1] + n_eff[1:])

phi_true = np.concatenate([[0.0], np.cumsum(K0 * n_full[:-1] * np.diff(z))])
_seg_thick = np.concatenate([[depths[0]], np.diff(depths)])
phi_mid = np.interp(
    np.concatenate([[depths[0]], (depths[:-1] + depths[1:]) / 2.0]),
    z, phi_true)


def phi_model(k, extra=0.0):
    """Model-stack one-way optical phase at depths[k] + extra (the SIM's own
    velocity structure: the synthetic eps, not the true profile)."""
    return K0 * (np.sum(n_eff[:k + 1] * _seg_thick[:k + 1]) + n_eff[k] * extra)


def dithers(ref_shift=True):
    """delta_k so the model reflector's phasor equals the segment's, wrapped
    to +-lam_local/4. ``ref_shift`` accounts for r_seg being referenced to the
    SEGMENT TOP while the model interface sits at the nominal depth."""
    out = np.empty(N)
    for k in range(N):
        want = np.angle(r_seg[k] / r_model[k])
        base = (SIGN_PROP * 2.0 * (phi_model(k) - phi_mid[k])
                if ref_shift else 0.0)
        out[k] = (np.angle(np.exp(1j * (want - base)))
                  / (SIGN_PROP * 2.0 * K0 * n_eff[k]))
    return out


# ========================================================================
# 3. coherent pulse-weighted response
# ========================================================================
_NT = int(2 * phi_true[-1] / (2 * np.pi * F0) * 1.05 / DT) + 1
DEPTH_AXIS = np.interp(np.arange(_NT) * DT,
                       2 * phi_true / (2 * np.pi * F0), z,
                       left=0.0, right=z[-1] + 1e3)


def response(r, phi):
    """|s(t)|^2 for point reflectors of coefficient r at one-way phase phi."""
    g = np.zeros(_NT, complex)
    tau = 2 * np.asarray(phi) / (2 * np.pi * F0)
    np.add.at(g, np.clip(np.round(tau / DT).astype(int), 0, _NT - 1),
              np.asarray(r) * np.exp(SIGN_PROP * 2j * np.asarray(phi)))
    return np.abs(np.convolve(g, PULSE, mode="same")) ** 2


def band_levels(p):
    return [10 * np.log10(p[(DEPTH_AXIS >= lo) & (DEPTH_AXIS < hi)].mean() / GS)
            for lo, hi in BANDS]


def main():
    print(f"TMM: r = {SIGN_R:+.0f}*(na-nb)/(na+nb), depth shift -> "
          f"exp({SIGN_PROP:+.0f}*2ik*delta);  gamma_surf "
          f"{10*np.log10(GS):.2f} dB")
    delta, delta_nr = dithers(True), dithers(False)
    got = np.array([r_model[k] * np.exp(SIGN_PROP * 2j * phi_model(k, delta[k]))
                    for k in range(N)])
    targ = r_seg * np.exp(SIGN_PROP * 2j * phi_mid)
    print(f"dither self-check max rel err {np.abs(got/targ - 1).max():.2e}; "
          f"max |delta| {100*np.abs(delta).max():.1f} cm, median "
          f"{100*np.median(np.abs(delta)):.1f} cm, max frac of lam_loc/4 "
          f"{np.abs(delta / (LAM / n_eff[:-1] / 4)).max():.3f}")

    trend = np.array([rfi.point_eps(d) for d in depths]
                     + [rfi.point_eps(float(depths[-1]) + 1.0)])
    n_pt = np.sqrt(trend)
    runs = {
        "full-res (1 mm)": response(
            SIGN_R * (n_full[:-1] - n_full[1:]) / (n_full[:-1] + n_full[1:]),
            phi_true[:-1]),
        "h1eff (|r| only)": response(r_model,
                                     [phi_model(k) for k in range(N)]),
        "h1eff_cr (complex r)": response(
            r_model, [phi_model(k, delta[k]) for k in range(N)]),
        "h1eff_cr (no ref shift)": response(
            r_model, [phi_model(k, delta_nr[k]) for k in range(N)]),
        "point-sampled N40": response(
            SIGN_R * (n_pt[:-1] - n_pt[1:]) / (n_pt[:-1] + n_pt[1:]),
            K0 * np.cumsum(n_pt[:-1] * _seg_thick)),
    }
    lev = {k: band_levels(v) for k, v in runs.items()}
    print("\n" + f"{'stack':<26}"
          + "".join(f"{f'{lo:.0f}-{hi:.0f}m':>10}" for lo, hi in BANDS)
          + "     delta vs full-res")
    ref = lev["full-res (1 mm)"]
    for k, v in lev.items():
        print(f"{k:<26}" + "".join(f"{x:+10.2f}" for x in v) + "     "
              + "".join(f"{x - r:+8.2f}" for x, r in zip(v, ref)))

    res_firn = 1.44 * C / (2 * B) / np.sqrt(rfi.EPS_MEAN)
    print("\nincoherent (|r|^2 sum) cross-check of the h1eff stack: "
          + ", ".join(
              f"{lo:.0f}-{hi:.0f}m "
              f"{10*np.log10((r_abs[(depths >= lo) & (depths < hi)]**2).sum() * res_firn / (hi - lo) / GS):+.2f} dB"
              for lo, hi in BANDS))

    # --- why: mm-scale noise content, and the missing firn attenuation ------
    print("\npre-smoothing scan (same estimator):")
    lines = (rfi.FIXDIR / "ngt37C95.2_density.tab").read_text().splitlines()
    hdr = next(i for i, ln in enumerate(lines)
               if ln.startswith("Depth ice/snow"))
    rho = np.loadtxt(rfi.FIXDIR / "ngt37C95.2_density.tab", delimiter="\t",
                     skiprows=hdr + 1)[:, 1]

    def sm_(x, m):
        k = int(round(m / dz)) | 1
        box = np.ones(k)
        return np.convolve(x, box, "same") / np.convolve(np.ones_like(x), box,
                                                         "same")

    for s in (0.0, 0.02, 0.05, 0.1, 0.2):
        npf = np.sqrt(rfi.eps_kovacs(rho if s == 0 else sm_(rho, s)))
        ph = np.concatenate([[0.0], np.cumsum(K0 * npf[:-1] * np.diff(z))])
        p = response((npf[:-1] - npf[1:]) / (npf[:-1] + npf[1:]) * SIGN_R,
                     ph[:-1])
        print(f"  smooth {s:4.2f} m: "
              + "".join(f"{x:+10.2f}" for x in band_levels(p)))

    print("\nmissing two-way attenuation (firn_cfg gives the firn media none):")
    for a in (9.0, 15.0):
        print(f"  {a:.0f} dB/km one-way: "
              + ", ".join(f"{2*a*d/1000:.2f} dB at {d:.0f} m"
                          for d in (45.0, 100.0, 120.0)))


if __name__ == "__main__":
    main()
