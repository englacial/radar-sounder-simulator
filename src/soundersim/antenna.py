"""Antenna gain patterns (stage 4, M22): per-facet two-way gain in all kernels.

Convention (plan D4-2, stated once here and in physics.py): ``g`` is the
ONE-WAY FIELD gain of the antenna in a departure direction and dimensionless.
Patterns are peak-normalized unless an array specifies element directivity.
Such arrays use a forward-hemisphere cosine-power element pattern whose
integrated directivity matches the scalar; the peak is the numerically
integrated directivity of element times array factor, per side. Monostatic (same antenna transmits and receives), so the
two-way weighting is

    received field  *= g_tx * g_rx = g**2      (coherent / multilayer kernels)
    received power  *= |g**2|**2   = g**4      (incoherent kernel)

which keeps the incoherent kernel consistent with |coherent field|**2
(cross-kernel ensemble test). No polarization: scalar gain only.

Pattern frame: built per trace from the nav track's along-track (u_at) and
cross-track-right (u_ct) unit vectors and the local nadir (0, 0, -1). With
``roll_source="nav"`` every frame vector is rotated (Rodrigues) about u_at by
the per-trace roll angle (radians, positive = right wing down -- a right-handed
rotation about u_at in the local ENU frame, which tilts the nadir boresight
toward the LEFT of travel).

Kernel integration (no per-config recompiles): the pattern KIND is a static
argument of the lru_cached jit factories (it changes the traced graph -- the
isotropic path traces exactly the pre-M22 program), while every run-varying
number -- the per-trace pattern vector, array n_elements/spacing, tabulated
sample arrays -- is a TRACED argument, so changing parameter values (or roll)
reuses the compiled kernel. The gain is evaluated in-kernel from the
facet->platform direction already computed there; the per-trace frame vectors
are precomputed here in NumPy (T x 3, cheap).
"""

import functools

import numpy as np

NADIR = np.array([0.0, 0.0, -1.0])


def rodrigues(v, k, ang):
    """Rotate vectors ``v`` by ``ang`` (rad) about unit axis ``k``, row-wise.

    v/k: (T, 3); ang: scalar or (T,). Right-handed rotation.
    """
    ang = np.asarray(ang, np.float64)[..., None]
    c, s = np.cos(ang), np.sin(ang)
    return (v * c + np.cross(k, v) * s
            + k * (np.sum(k * v, axis=-1, keepdims=True)) * (1.0 - c))


def frame_vectors(antenna, u_at, u_ct, roll=None):
    """Per-trace pattern vector for an AntennaConfig (NumPy float64, (T, 3)).

    dipole -> dipole axis; array -> element axis (cross-track); tabulated ->
    boresight (nadir). ``roll`` (T,) radians is applied about u_at when
    ``antenna.roll_source == "nav"`` (None -> 0; NaNs -> 0).
    """
    u_at = np.asarray(u_at, np.float64)
    u_ct = np.asarray(u_ct, np.float64)
    T = u_at.shape[0]
    if antenna.kind in ("dipole", "finite_dipole"):
        v = u_at if antenna.axis == "along_track" else u_ct
    elif antenna.kind in ("array", "array_tapered"):
        v = u_ct
    elif antenna.kind == "tabulated":
        v = np.broadcast_to(NADIR, (T, 3))
    else:
        raise ValueError(f"no pattern vector for kind {antenna.kind!r}")
    if antenna.roll_source == "nav" and roll is not None:
        phi = np.nan_to_num(np.asarray(roll, np.float64), nan=0.0)
        v = rodrigues(np.ascontiguousarray(v), u_at, phi)
    return np.ascontiguousarray(v)


def pattern_args(antenna, u_at, u_ct, roll=None):
    """Kernel pattern arguments: None (isotropic) or ``(kind, pv, pa, pb)``.

    pv: float64 per-trace pattern vectors (kernels cast to their working
    dtype); arrays carry (T, 2, 3) [cross-track axis, boresight], other kinds
    carry (T, 3). pa/pb are traced parameter arrays --
      dipole:        unused scalar zeros
      array:         pa = n_elements,
                     pb = [spacing_lam, field_scale, element_power_exponent,
                           element_pattern_enabled]
      array_tapered: pa = (2, n) [tx_weights; rx_weights],
                     pb = same array controls
      finite_dipole: pa = length_lam, pb unused
      tabulated:     pa = theta samples (radians, ascending), pb = field gains
    """
    if antenna.kind == "isotropic":
        return None
    pv = frame_vectors(antenna, u_at, u_ct, roll)
    if antenna.kind == "array":
        pv = _array_frame(pv, u_at)
        pa = np.float64(antenna.n_elements)
        pb = np.asarray([antenna.spacing_lam,
                         _array_field_scale(antenna),
                         element_power_exponent(
                             antenna.element_directivity_db),
                         antenna.element_directivity_db is not None], np.float64)
    elif antenna.kind == "array_tapered":
        pv = _array_frame(pv, u_at)
        pa = np.asarray([antenna.tx_weights, antenna.rx_weights], np.float64)
        pb = np.asarray([antenna.spacing_lam,
                         _array_field_scale(antenna),
                         element_power_exponent(
                             antenna.element_directivity_db),
                         antenna.element_directivity_db is not None], np.float64)
    elif antenna.kind == "finite_dipole":
        pa = np.float64(antenna.length_lam)
        pb = np.float64(0.0)
    elif antenna.kind == "tabulated":
        pa = np.deg2rad(np.asarray(antenna.theta_deg, np.float64))
        pb = np.asarray(antenna.gain, np.float64)
    else:  # dipole
        pa = pb = np.float64(0.0)
    return antenna.kind, pv, pa, pb


def _array_frame(axis, u_at):
    """Stack cross-track element axis and nadir/rolled boresight."""
    boresight = np.cross(np.asarray(u_at, np.float64), axis)
    return np.stack([axis, boresight], axis=1)


def element_power_exponent(directivity_db):
    """q for P(theta)/P(0) = cos(theta)^q on the forward hemisphere.

    Its directivity is 2(q + 1). Three dBi is treated as the rounded value
    for a uniform hemisphere (exactly 3.0103 dBi).
    """
    if directivity_db is None:
        return 0.0
    return max(10.0 ** (directivity_db / 10.0) / 2.0 - 1.0, 0.0)


def element_field_pattern(cos_theta, power_exponent, enabled=True, xp=np):
    """Peak-normalized forward-hemisphere element field pattern."""
    c = xp.clip(cos_theta, 0.0, 1.0)
    forward = xp.where(cos_theta >= 0.0,
                       c ** (power_exponent / 2.0), 0.0)
    return xp.where(enabled, forward, xp.ones_like(c))


def _line_array_field(u, cos_bore, weights, spacing_lam, q):
    """Unscaled one-way field of a tapered line array of cosine-power
    elements: u = direction . axis (sin theta_ct), cos_bore = direction .
    boresight. AF is normalized by sum(w); the element factor is
    peak-normalized."""
    w = np.asarray(weights, np.float64)
    m = np.arange(len(w)) - (len(w) - 1) / 2.0
    ph = (2.0 * np.pi * spacing_lam) * np.asarray(u)[..., None] * m
    af = np.hypot(np.cos(ph) @ w, np.sin(ph) @ w) / w.sum()
    return element_field_pattern(cos_bore, q) * af


def _side_directivity(weights, spacing_lam, q, n_theta=720, n_phi=720):
    """Peak power directivity of one side's one-way pattern, integrated
    numerically over the sphere (midpoint rule; theta from boresight)."""
    th = (np.arange(n_theta) + 0.5) * (np.pi / n_theta)
    ph = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    st, ct = np.sin(th)[:, None], np.cos(th)[:, None]
    u = st * np.cos(ph)[None, :]                   # component along the axis
    P = _line_array_field(u, np.broadcast_to(ct, u.shape),
                          weights, spacing_lam, q) ** 2
    d_omega = st * (np.pi / n_theta) * (2.0 * np.pi / n_phi)
    return 4.0 * np.pi * P.max() / np.sum(P * d_omega)


@functools.lru_cache(maxsize=64)
def _array_directivity_scale(tx_weights, rx_weights, spacing_lam,
                             directivity_db):
    q = element_power_exponent(directivity_db)
    d_tx = _side_directivity(tx_weights, spacing_lam, q)
    d_rx = _side_directivity(rx_weights, spacing_lam, q)
    return float((d_tx * d_rx) ** 0.25)


def _array_field_scale(antenna):
    """Absolute peak field gain, or one for legacy normalized arrays.

    g**2 (two-way) equals sqrt(D_tx * D_rx) where each side's directivity is
    the numerically integrated element-times-array-factor pattern. For
    uniform-hemisphere (3 dBi) elements at multiples of half-wavelength
    spacing this reduces to the closed form D = 2 N_eff per side; for
    directive elements it is lower than the product of element and array
    directivities, because both narrow the beam in the cross-track plane.
    Mutual coupling and the platform environment are not modeled: the
    element dBi describes the embedded element.
    """
    if antenna.element_directivity_db is None:
        return 1.0
    if antenna.kind == "array":
        tx = rx = (1.0,) * antenna.n_elements
    else:
        tx = tuple(float(w) for w in antenna.tx_weights)
        rx = tuple(float(w) for w in antenna.rx_weights)
    return _array_directivity_scale(tx, rx, float(antenna.spacing_lam),
                                    float(antenna.element_directivity_db))


def gain_fn(kind):
    """In-kernel one-way FIELD gain evaluator ``g(dhat, pv, pa, pb)`` (jnp).

    ``dhat`` (..., 3) is the DEPARTURE direction, antenna -> facet (i.e.
    -rhat for the kernels' facet -> platform rhat); pv broadcasts against it.
    dtype follows the inputs (f32 in the surface kernels, f64 in multilayer).
    """
    import jax.numpy as jnp

    if kind == "dipole":
        def g(dhat, pv, pa, pb):
            ca = jnp.sum(dhat * pv, axis=-1)          # cos(psi), psi from axis
            s = jnp.sqrt(jnp.maximum(1.0 - ca * ca, 1e-12))
            return jnp.cos((np.pi / 2.0) * ca) / s
    elif kind == "array":
        def g(dhat, pv, pa, pb):
            u = jnp.sum(dhat * pv[0], axis=-1)        # sin(theta_ct)
            ct = jnp.sum(dhat * pv[1], axis=-1)       # cos(theta_boresight)
            x = np.pi * pb[0] * u
            sx = jnp.sin(x)
            near = jnp.abs(sx) < 1e-5                 # main/grating lobe peaks
            safe = jnp.where(near, 1.0, sx)
            af = jnp.where(near, jnp.cos(pa * x) / jnp.cos(x),
                           jnp.sin(pa * x) / (pa * safe))
            ef = element_field_pattern(ct, pb[2], pb[3], xp=jnp)
            return pb[1] * ef * jnp.abs(af)
    elif kind == "array_tapered":
        def g(dhat, pv, pa, pb):
            # AF_w(u) = |sum_m w_m exp(i 2 pi d_lam m u)| / sum(w), element
            # index m centered; g = sqrt(AF_tx * AF_rx) so the kernels'
            # two-way g**2 is the physical AF_tx * AF_rx.
            n = pa.shape[1]
            m = jnp.arange(n, dtype=pa.dtype) - (n - 1) / 2.0
            u = jnp.sum(dhat * pv[0], axis=-1)        # sin(theta_ct)
            ct = jnp.sum(dhat * pv[1], axis=-1)       # cos(theta_boresight)
            ph = (2.0 * np.pi * pb[0]) * u[..., None] * m
            c, s = jnp.cos(ph), jnp.sin(ph)
            def af(w):
                return jnp.sqrt((c @ w) ** 2 + (s @ w) ** 2) / jnp.sum(w)
            ef = element_field_pattern(ct, pb[2], pb[3], xp=jnp)
            return pb[1] * ef * jnp.sqrt(af(pa[0]) * af(pa[1]))
    elif kind == "finite_dipole":
        def g(dhat, pv, pa, pb):
            kh = np.pi * pa                            # k * L / 2
            ca = jnp.sum(dhat * pv, axis=-1)          # cos(psi), psi from axis
            s = jnp.sqrt(jnp.maximum(1.0 - ca * ca, 1e-12))
            return jnp.abs(jnp.cos(kh * ca) - jnp.cos(kh)) / (
                s * (1.0 - jnp.cos(kh)))
    elif kind == "tabulated":
        def g(dhat, pv, pa, pb):
            ct = jnp.clip(jnp.sum(dhat * pv, axis=-1), -1.0, 1.0)
            return jnp.interp(jnp.arccos(ct), pa, pb)
    else:
        raise ValueError(f"unknown pattern kind {kind!r}")
    return g


def field_gain(antenna, dhat, u_at, u_ct, roll=None):
    """NumPy float64 reference: one-way FIELD gain for departure directions.

    dhat: (..., 3) unit departure vectors for ONE trace; u_at/u_ct: (3,) that
    trace's track vectors; roll: scalar radians. Mirrors ``gain_fn`` exactly
    (used by tests and analysis scripts).
    """
    if antenna.kind == "isotropic":
        return np.ones(np.asarray(dhat).shape[:-1])
    kind, pv, pa, pb = pattern_args(
        antenna, np.asarray(u_at, np.float64)[None],
        np.asarray(u_ct, np.float64)[None],
        None if roll is None else np.atleast_1d(roll))
    v = pv[0]
    d = np.asarray(dhat, np.float64)
    if kind == "dipole":
        ca = d @ v
        s = np.sqrt(np.maximum(1.0 - ca * ca, 1e-12))
        return np.cos((np.pi / 2.0) * ca) / s
    if kind == "array":
        axis, boresight = v
        x = np.pi * pb[0] * (d @ axis)
        sx = np.sin(x)
        near = np.abs(sx) < 1e-5
        af = np.where(near, np.cos(pa * x) / np.cos(x),
                      np.sin(pa * x) / (pa * np.where(near, 1.0, sx)))
        ef = element_field_pattern(d @ boresight, pb[2], pb[3])
        return pb[1] * ef * np.abs(af)
    if kind == "array_tapered":
        axis, boresight = v
        n = pa.shape[1]
        m = np.arange(n) - (n - 1) / 2.0
        ph = (2.0 * np.pi * pb[0]) * (d @ axis)[..., None] * m

        def af(w):
            return np.hypot(np.cos(ph) @ w, np.sin(ph) @ w) / w.sum()

        ef = element_field_pattern(d @ boresight, pb[2], pb[3])
        return pb[1] * ef * np.sqrt(af(pa[0]) * af(pa[1]))
    if kind == "finite_dipole":
        kh = np.pi * float(pa)
        ca = d @ v
        s = np.sqrt(np.maximum(1.0 - ca * ca, 1e-12))
        return np.abs(np.cos(kh * ca) - np.cos(kh)) / (s * (1.0 - np.cos(kh)))
    # tabulated
    ct = np.clip(d @ v, -1.0, 1.0)
    return np.interp(np.arccos(ct), pa, pb)
