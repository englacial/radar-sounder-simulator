"""Path B1: effective Gaussian sub-facet surface roughness from measured
(OIB ATM) surface spectra.

The Gerekos kernel is Gaussian-ACF only. The measured surfaces are
self-affine (power-law PSD) or exponential, so no single Gaussian pair
(sigma, l) reproduces them; instead, per pass, we choose the pair whose
2-D PSD is TANGENT to the measured PSD at the Bragg wavenumber of the
clutter angle that matters, k_B = 2 k0 sin(theta_c):

    S_G(k) = sigma^2 l^2 / (4 pi) * exp(-k^2 l^2 / 4)      (int S d2k = sigma^2)
    S_G(k_B) = S_meas(k_B),   d ln S_G / d ln k |_{k_B} = -beta_loc
    =>  l^2 = 2 beta_loc / k_B^2,   sigma^2 = 4 pi S_meas(k_B) e^{beta_loc/2} / l^2

The PSD convention is the ATM analysis's (claude_notes/atm_roughness), which
is the Gaussian-ACF rho(r) = exp(-r^2/l^2) the kernel assumes, so the
first-order (m = 1) Gerekos incoherent term equals the measured PSD at k_B
exactly and follows its slope nearby. Spectra live in
config/roughness/atm_b1.yaml (provenance there).

``resolve_exponential`` (source ``atm_exponential``) instead hands an
exponential-ACF table entry (sigma, l) straight to the kernel's
``acf: exponential`` option (docs/roughness.md) -- no effective pair, no
carrier or clutter-angle dependence -- and refuses power-law entries.
"""
from __future__ import annotations

from pathlib import Path

import warnings

import numpy as np
import yaml

C = 299792458.0
DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config/roughness/atm_b1.yaml"
STRATA_YAML = DEFAULT_YAML.with_name("atm_tier2_strata.yaml")


def load_table(path=DEFAULT_YAML, strata=STRATA_YAML):
    """The site-specific table merged with the Tier 2 per-stratum exponential
    entries (config/roughness/atm_tier2_strata.yaml): their spectra are added
    (site entries win on a name clash) and their ``stratum_lines`` mapping is
    the fallback for lines without an ``exponential`` alternate mapping."""
    with Path(path).open() as fh:
        tab = yaml.safe_load(fh)
    if strata and Path(strata).exists():
        with Path(strata).open() as fh:
            st = yaml.safe_load(fh)
        tab["spectra"] = {**st.get("spectra", {}), **tab["spectra"]}
        tab["stratum_lines"] = st.get("stratum_lines", {})
        for line, m in tab["stratum_lines"].items():
            tab["lines"].setdefault(line, {"default": m["default"]})
    return tab


def spectrum(spec):
    """Callable S(k) [m^4] for one spectrum entry (2-D, int S d2k = sigma^2)."""
    fam = spec["family"]
    if fam == "powerlaw":
        A, beta = float(spec["A_m4"]), float(spec["beta"])
        return lambda k: A * np.asarray(k, float) ** -beta
    if fam == "exponential":
        s2, l = float(spec["sigma_m"]) ** 2, float(spec["l_m"])
        return lambda k: s2 * l ** 2 / (2 * np.pi) * (1 + (np.asarray(k, float) * l) ** 2) ** -1.5
    if fam == "gaussian":
        s2, l = float(spec["sigma_m"]) ** 2, float(spec["l_m"])
        return lambda k: s2 * l ** 2 / (4 * np.pi) * np.exp(-(np.asarray(k, float) * l) ** 2 / 4)
    raise ValueError(f"unknown spectrum family {fam!r}")


def bragg_k(f0_hz, theta_c_deg):
    return 2 * (2 * np.pi * f0_hz / C) * np.sin(np.radians(theta_c_deg))


def gaussian_psd(k, sigma_m, l_m):
    return sigma_m ** 2 * l_m ** 2 / (4 * np.pi) * np.exp(-(k * l_m) ** 2 / 4)


def tangent_pair(S, k_b):
    """(sigma_m, l_m) whose Gaussian PSD matches S in value and log-slope at k_b."""
    h = 1e-3
    beta = -(np.log(S(k_b * (1 + h))) - np.log(S(k_b * (1 - h)))) / (np.log(1 + h) - np.log(1 - h))
    l = np.sqrt(2 * beta) / k_b
    sigma = np.sqrt(4 * np.pi * float(S(k_b)) * np.exp(beta / 2) / l ** 2)
    return float(sigma), float(l)


def _spectrum_id(tab, line, pass_key, alt=None):
    ln = tab["lines"].get(line)
    if ln is None:
        raise KeyError(f"atm_b1: no surface spectrum for line {line!r} "
                       f"(have {sorted(tab['lines'])})")
    if alt is not None and alt in ln:      # alternate-family mapping
        ln = {**ln, **ln[alt]}
    # one surface law per line: every pass (real or synthetic) uses the
    # line's default entry -- the reference pass's spectrum. Per-pass
    # `passes:` overrides are no longer honoured (2026-08-28).
    return ln["default"]


def resolve_exponential(line, pass_key, table=None):
    """(sigma_m, l_m, info) of the line's/pass's EXPONENTIAL-ACF entry, for
    the kernel's acf='exponential' option. Uses the line's ``exponential``
    alternate mapping when present (surfaces whose best family is a power
    law but carry an exponential fit for comparison), else the default
    mapping; a non-exponential entry is an error (use atm_b1 for those)."""
    tab = table or load_table()
    sid = _spectrum_id(tab, line, pass_key, alt="exponential")
    spec = tab["spectra"][sid]
    if spec["family"] != "exponential" and line in tab.get("stratum_lines", {}):
        # no site-specific exponential entry: fall back to the Tier 2 stratum
        sid = tab["stratum_lines"][line]["default"]
        spec = tab["spectra"][sid]
    if spec["family"] != "exponential":
        raise ValueError(f"atm_exponential: spectrum {sid!r} for line "
                         f"{line!r} pass {pass_key!r} is {spec['family']}, "
                         "not exponential -- no exponential-ACF entry; use "
                         "source atm_b1 (effective Gaussian) instead")
    use = spec.get("usability", "use")
    if use == "refuse":
        raise ValueError(f"atm_exponential: stratum {sid!r} ({spec.get('stratum')}) is "
                         f"marked refuse -- Matern nu {spec.get('matern_nu_med_p5_p95', ['?'])[0]}, "
                         f"l at bound in {spec.get('l_capped_frac')} of years; the exponential "
                         "over-predicts wide-angle scatter there (Tier 2 note). Use a site-specific "
                         "entry or wait for the Matern option.")
    sig, l = float(spec["sigma_m"]), float(spec["l_m"])
    info = {"spectrum": sid, "family": "exponential", "acf": "exponential",
            "usability": use, "stratum": spec.get("stratum"),
            "rule": "direct (sigma, l) of the exponential-ACF fit",
            "provenance": spec.get("provenance")}
    if use == "marginal":
        info["warning"] = ("marginal stratum: power law fits better; exponential under-predicts "
                           f"wide-angle scatter by ~{spec.get('misfit_1p5m_dB_med_p5_p95', ['?'])[0]} dB "
                           "(median at 1.5 m) -- see atm_tier2_strata.yaml")
        warnings.warn(f"atm_exponential: {info['warning']} [{sid}]")
    return sig, l, info


def resolve(line, pass_key, f0_hz, theta_c_deg=30.0, table=None):
    """Effective (sigma_m, l_m) for one pass of one line at carrier f0.
    Returns (sigma, l, info) where info records the spectrum used."""
    tab = table or load_table()
    sid = _spectrum_id(tab, line, pass_key)
    spec = tab["spectra"][sid]
    k_b = bragg_k(f0_hz, theta_c_deg)
    sigma, l = tangent_pair(spectrum(spec), k_b)
    info = {"spectrum": sid, "family": spec["family"], "theta_c_deg": float(theta_c_deg),
            "f0_hz": float(f0_hz), "k_bragg_rad_per_m": float(k_b),
            "bragg_wavelength_m": float(2 * np.pi / k_b),
            "S_meas_at_kB_db_m4": float(10 * np.log10(spectrum(spec)(k_b))),
            "rule": "tangent (value + log-slope at k_B)"}
    return sigma, l, info
