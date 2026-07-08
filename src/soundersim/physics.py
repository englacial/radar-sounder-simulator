"""Scalar wave-physics helpers (relative-power conventions, no absolute calibration)."""

from typing import Any, NamedTuple

import numpy as np


def fresnel_normal(eps1, eps2):
    """Normal-incidence scalar Fresnel reflection coefficient from medium 1 into
    medium 2, from relative permittivities. Sign preserved (air->ice ~ -0.2807).
    """
    n1, n2 = np.sqrt(eps1), np.sqrt(eps2)
    return (n1 - n2) / (n1 + n2)


class FresnelTE(NamedTuple):
    """Angle-dependent scalar Fresnel coefficients (see ``fresnel_te``)."""

    gamma: Any       # field reflection coefficient (medium 1 -> 2)
    tau: Any         # field transmission coefficient into medium 2 (= 1 + gamma)
    cos_theta2: Any  # cosine of the refracted angle in medium 2 (0 under TIR)
    tir: Any         # bool: total internal reflection (values clamped, mask me)


def fresnel_te(eps1, eps2, cos_theta1, xp=np):
    """Angle-dependent scalar Fresnel coefficients from medium 1 into medium 2.

    Scalar/unpolarized proxy: the TE (s-polarization) forms are used -- the
    standard choice for scalar sounder simulation (no Brewster angle, |gamma|
    monotone in incidence angle, and the simplest energy-consistent set).
    With n_i = sqrt(eps_i), c1 = cos(theta1), c2 = cos(theta2) from Snell:

        gamma = (n1 c1 - n2 c2) / (n1 c1 + n2 c2)      FIELD reflection
        tau   = 1 + gamma = 2 n1 c1 / (n1 c1 + n2 c2)  FIELD transmission

    Conventions (field vs power):

    - Power reflectance  R = gamma**2.
    - Power transmittance T = (n2 c2)/(n1 c1) * tau**2 = 1 - gamma**2
      (the impedance/angle flux factor makes R + T = 1 exactly).
    - Two-way FIELD transmission through one crossing (down at theta1, back up
      at theta2, Stokes relation): tau_down * tau_up = 1 - gamma**2 = T.
      The coherent kernel multiplies fields by tau_down*tau_up; the incoherent
      kernel multiplies power by (tau_down*tau_up)**2 = T_down * T_up.

    At normal incidence gamma reduces exactly to ``fresnel_normal``. Total
    internal reflection (sin(theta2) > 1, dense -> rare only) is flagged in
    ``tir`` with c2 clamped to 0 (gamma -> 1, finite); callers mask.
    ``xp`` selects the array module (np or jnp); dtype follows the inputs.
    """
    eps1 = xp.asarray(eps1)
    eps2 = xp.asarray(eps2)
    c1 = xp.asarray(cos_theta1)
    n1, n2 = xp.sqrt(eps1), xp.sqrt(eps2)
    sin2_sq = (eps1 / eps2) * (1.0 - c1 * c1)  # Snell: sin^2(theta2)
    tir = sin2_sq > 1.0
    c2 = xp.sqrt(xp.maximum(1.0 - sin2_sq, 0.0))
    gamma = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    return FresnelTE(gamma, 1.0 + gamma, c2, tir)


C = 299792458.0  # speed of light (m/s)


def in_medium_speed(eps_r, c=C):
    """Wave speed in a medium of relative permittivity eps_r: c / sqrt(eps_r)."""
    return c / np.sqrt(np.asarray(eps_r, dtype=np.float64))


def optical_path_length(lengths, eps_r):
    """One-way optical path length Sum_i sqrt(eps_r_i) * s_i over a path's legs.

    ``lengths`` (..., K) are geometric leg lengths (m); ``eps_r`` broadcasts to
    the same shape (per-leg medium permittivity). Sums over the last axis.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    n = np.sqrt(np.broadcast_to(np.asarray(eps_r, np.float64), lengths.shape))
    return np.sum(n * lengths, axis=-1)


def two_way_delay(lengths, eps_r, c=C):
    """Two-way travel time 2 * optical_path_length / c (s)."""
    return 2.0 * optical_path_length(lengths, eps_r) / c


def attenuation_loss_db(lengths, atten_db_per_km):
    """One-way power attenuation (dB, positive = loss) over a path's legs.

    ``lengths`` (..., K) in metres; ``atten_db_per_km`` broadcasts per leg.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    a = np.broadcast_to(np.asarray(atten_db_per_km, np.float64), lengths.shape)
    return np.sum((lengths / 1000.0) * a, axis=-1)


def db_to_linear_power(loss_db):
    """Linear power factor for a positive dB loss: 10 ** (-loss_db / 10)."""
    return 10.0 ** (-np.asarray(loss_db, dtype=np.float64) / 10.0)


def refraction_spreading(lengths, cos_leg, n, xp=np):
    """Effective spreading lengths (L_par, L_perp) for a refracted ray path.

    Standard astigmatic ray-tube geometrical optics through flat dielectric
    interfaces (e.g. Deschamps 1972, Proc. IEEE 60(9); nadir limit h + d/n as
    in Peters et al. 2005, JGR 110): a point source's spherical wavefront
    refracts into an astigmatic wavefront whose two principal effective
    distances differ. With per-leg geometric lengths ``s_i``, leg direction
    cosines ``c_i`` (cosine of the ray angle in medium i w.r.t. the crossed
    interface normals) and refractive indices ``n_i`` (leg 0 = source medium),
    summing over the last axis:

        L_perp = n_0        * sum_i s_i / n_i             (out-of-plane)
        L_par  = n_0 c_0**2 * sum_i s_i / (n_i c_i**2)    (in-plane)

    Derivation (verified by finite-difference ray-tube tracing in the tests):
    in-plane, Snell differentiated gives d(theta_i) = (n_0 c_0)/(n_i c_i)
    d(theta_0) and horizontal spread sum_i (s_i/c_i) d(theta_i); out-of-plane
    the azimuthal width is sum_i s_i sin(theta_i) d(phi) with
    sin(theta_i) = (n_0/n_i) sin(theta_0).

    One-way FIELD spreading from a unit source is 1/sqrt(L_par * L_perp) (with
    the Fresnel ``tau`` field coefficients applied separately per crossing;
    energy in the ray tube is then conserved). The monostatic TWO-WAY spreading
    is (n_0 c_0)/(n_j c_j) / (L_par * L_perp) -- the up-leg's effective lengths
    are (n_j c_j..)/(n_0 c_0..)-scaled copies of the down-leg's, and the flux
    factor (n_0 c_0)/(n_j c_j) is what makes the nadir flat-slab result reduce
    exactly to the image-in-dielectric closed form
    tau_down*tau_up*Gamma / (2*(h + d/n)) (validated in the tests).

    For a single-medium path (one leg) both lengths reduce to s_0, recovering
    the 1/r one-way (1/r**2 two-way) spherical spreading.
    """
    s = xp.asarray(lengths)
    c = xp.asarray(cos_leg)
    n = xp.asarray(n)
    s, c, n = xp.broadcast_arrays(s, c, n)
    n0 = n[..., 0]
    c0 = c[..., 0]
    l_perp = n0 * xp.sum(s / n, axis=-1)
    l_par = n0 * c0 * c0 * xp.sum(s / (n * c * c), axis=-1)
    return l_par, l_perp
