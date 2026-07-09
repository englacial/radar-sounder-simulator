"""Antenna gain patterns (stage 4, M22): per-facet two-way gain in all kernels.

Convention (plan D4-2, stated once here and in physics.py): ``g`` is the
ONE-WAY FIELD gain of the antenna in a departure direction, dimensionless and
peak-normalized (isotropic g = 1; dipole broadside g = 1; array main lobe
g = 1). Monostatic (same antenna transmits and receives), so the two-way
weighting is

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
    if antenna.kind == "dipole":
        v = u_at if antenna.axis == "along_track" else u_ct
    elif antenna.kind == "array":
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

    pv: (T, 3) float64 per-trace pattern vector (kernels cast to their working
    dtype); pa/pb: traced parameter arrays --
      dipole:    unused scalar zeros
      array:     pa = n_elements, pb = spacing_lam (scalars)
      tabulated: pa = theta samples (radians, ascending), pb = field gains
    """
    if antenna.kind == "isotropic":
        return None
    pv = frame_vectors(antenna, u_at, u_ct, roll)
    if antenna.kind == "array":
        pa = np.float64(antenna.n_elements)
        pb = np.float64(antenna.spacing_lam)
    elif antenna.kind == "tabulated":
        pa = np.deg2rad(np.asarray(antenna.theta_deg, np.float64))
        pb = np.asarray(antenna.gain, np.float64)
    else:  # dipole
        pa = pb = np.float64(0.0)
    return antenna.kind, pv, pa, pb


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
            u = jnp.sum(dhat * pv, axis=-1)           # sin(theta_ct)
            x = np.pi * pb * u
            sx = jnp.sin(x)
            near = jnp.abs(sx) < 1e-5                 # main/grating lobe peaks
            safe = jnp.where(near, 1.0, sx)
            af = jnp.where(near, jnp.cos(pa * x) / jnp.cos(x),
                           jnp.sin(pa * x) / (pa * safe))
            return jnp.abs(af)
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
        x = np.pi * float(pb) * (d @ v)
        sx = np.sin(x)
        near = np.abs(sx) < 1e-5
        af = np.where(near, np.cos(pa * x) / np.cos(x),
                      np.sin(pa * x) / (pa * np.where(near, 1.0, sx)))
        return np.abs(af)
    # tabulated
    ct = np.clip(d @ v, -1.0, 1.0)
    return np.interp(np.arccos(ct), pa, pb)
