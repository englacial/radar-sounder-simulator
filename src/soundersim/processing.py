"""Post-processing (stage 4, M23): presum / unfocused SAR / focused SAR.

Layer-3 processing per docs/tech_stack.md: these functions operate on the
output ``xarray.Dataset`` (docs/output.md) and NEVER inside the kernels. Each
takes a Dataset and returns a NEW Dataset with the step recorded in the
appendable ``processing`` attr -- a JSON list of step descriptors, so a chain
(e.g. presum -> focus) leaves a full provenance trail.

Along-track conventions
-----------------------
* ``presum(ds, n)`` -- non-overlapping coherent boxcar of ``n`` traces
  (CReSIS-style presumming): decimates the along-track sampling by ``n``.
* ``unfocused_sar(ds, ...)`` -- the SAME coherent boxcar applied as a stride-1
  MOVING window (overlapping), so the along-track sampling is preserved
  (CSARP_standard-like unfocused processing). ``presum`` == the decimated
  subset of ``unfocused_sar`` at matched window length.
* ``multilook(ds, n)`` -- INCOHERENT power averaging (speckle reduction); the
  right tool when phase is not wanted. ``presum``/``unfocused_sar`` refuse
  incoherent Datasets (there is no ``field`` to sum coherently) and point here.
* ``focused_sar(ds, aperture_m, ...)`` -- straight-track time-domain
  backprojection through AIR only (surface-referenced). Validation-grade.

All coherent reductions sum ``field``; every other slow_time-indexed variable
(positions, per-trace diagnostics) is averaged, and ``power`` is recomputed as
``|field|**2`` so the docs/output.md contract holds on the returned Dataset.
"""

import json
import warnings

import numpy as np
import xarray as xr

_C_DEFAULT = 299792458.0


def _record(attrs, step):
    """Append ``step`` to the JSON ``processing`` list in an attrs dict."""
    steps = json.loads(attrs.get("processing", "[]"))
    steps.append(step)
    attrs["processing"] = json.dumps(steps)
    return attrs


def _require_coherent(ds, op):
    if ds.attrs.get("mode") != "coherent" or "field" not in ds:
        raise ValueError(
            f"{op}() needs a coherent-mode Dataset with a complex `field` "
            f"(got mode={ds.attrs.get('mode')!r}). Coherent summation of power "
            "is not meaningful; for incoherent speckle reduction use "
            "multilook(ds, n) instead.")


def _speed_of_light(ds):
    try:
        return float(json.loads(ds.attrs["config"])["radar"]["c"])
    except Exception:
        return _C_DEFAULT


def _positions(ds):
    return np.column_stack([ds.x.values, ds.y.values, ds.z.values]).astype(float)


def _along_track(ds):
    """Signed along-track arc length (m), s[0] = 0 (straight-track assumption)."""
    pos = _positions(ds)
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _mean_spacing(ds):
    pos = _positions(ds)
    if pos.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).mean())


def _reduce_slow_time_coord(values, Wn):
    """Weight-average a slow_time coord, handling datetime64."""
    if np.issubdtype(values.dtype, np.datetime64):
        ns = values.astype("datetime64[ns]").astype(np.int64)
        return (Wn @ ns).round().astype(np.int64).astype("datetime64[ns]")
    return Wn @ values.astype(float)


def _apply_along_slow_time(ds, W, coherent, step):
    """Reduce every slow_time-indexed field by the (n_out, T) weight matrix W.

    ``field`` is summed with W (coherent); other slow_time variables/coords are
    averaged with W row-normalised. When ``coherent`` the returned ``power`` is
    recomputed as ``|field|**2``; otherwise ``field`` is dropped and ``power``
    is averaged (incoherent multilook).
    """
    T = ds.sizes["slow_time"]
    if W.shape[1] != T:
        raise ValueError("weight matrix width must match slow_time length")
    Wn = W / W.sum(1, keepdims=True)

    def reduce(da, weights):
        rest = [d for d in da.dims if d != "slow_time"]
        arr = da.transpose("slow_time", *rest).values
        out = np.tensordot(weights, arr, axes=([1], [0]))
        return ("slow_time", *rest), out

    data_vars = {}
    for name, da in ds.data_vars.items():
        if name == "power":
            if coherent:
                continue  # recomputed from the summed field below
            dims, out = reduce(da, Wn)
            data_vars[name] = (dims, out.astype(da.dtype), dict(da.attrs))
        elif name == "field":
            if not coherent:
                continue  # multilook destroys phase coherence: drop the field
            dims, out = reduce(da, W)
            data_vars[name] = (dims, out.astype(np.complex64), dict(da.attrs))
        elif "slow_time" in da.dims:
            dims, out = reduce(da, Wn)
            data_vars[name] = (dims, out, dict(da.attrs))
        else:
            data_vars[name] = (da.dims, da.values, dict(da.attrs))

    coords = {}
    for name, da in ds.coords.items():
        if name == "slow_time":
            coords[name] = ("slow_time",
                            _reduce_slow_time_coord(da.values, Wn), dict(da.attrs))
        elif "slow_time" in da.dims:
            dims, out = reduce(da, Wn)
            coords[name] = (dims, out, dict(da.attrs))
        else:
            coords[name] = (da.dims, da.values, dict(da.attrs))

    attrs = _record(dict(ds.attrs), step)
    out = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    if coherent and "field" in out:
        out["power"] = (out.field.dims, np.abs(out.field.values) ** 2,
                        dict(ds.power.attrs))
    return out


def presum(ds, n):
    """Coherent non-overlapping boxcar sum of ``n`` adjacent traces.

    CReSIS presumming convention: consecutive traces are complex-summed in
    non-overlapping blocks of ``n``, decimating the along-track sampling by
    ``n`` (a trailing remainder of < n traces is dropped). ``field`` is summed;
    positions and per-trace diagnostics are block-averaged; ``power`` is
    recomputed as ``|field|**2``.

    A point/specular target coherent across the block gains ``n`` in field
    amplitude (``n**2`` in power) while incoherent noise grows as ``sqrt(n)``:
    the SNR gain is ``sqrt(n)``. Incoherent Datasets are refused (use
    ``multilook``).
    """
    _require_coherent(ds, "presum")
    n = int(n)
    if n < 1:
        raise ValueError("presum count n must be >= 1")
    T = ds.sizes["slow_time"]
    n_out = T // n
    if n_out < 1:
        raise ValueError(f"presum n={n} exceeds slow_time length {T}")
    W = np.zeros((n_out, T))
    for k in range(n_out):
        W[k, k * n:(k + 1) * n] = 1.0
    step = {"op": "presum", "n": n, "n_traces_in": T, "n_traces_out": n_out,
            "convention": "non-overlapping coherent boxcar (CReSIS presum); "
                          "decimates along-track sampling by n"}
    return _apply_along_slow_time(ds, W, coherent=True, step=step)


def unfocused_sar(ds, aperture_m=None, n_traces=None):
    """Unfocused SAR: stride-1 coherent moving sum along-track.

    Give either ``n_traces`` (window length in traces) or ``aperture_m`` (the
    window converted via the mean trace spacing). This is exactly ``presum``'s
    coherent boxcar applied as an OVERLAPPING moving window, so the along-track
    trace count is preserved (output length ``T - n + 1``, one sample per valid
    window). ``presum(ds, n)`` equals ``unfocused_sar(ds, n_traces=n)`` sampled
    at every n-th output trace.

    "Unfocused" because no range-migration / quadratic-phase correction is
    applied: coherent integration is only valid while the two-way phase across
    the window stays within ~pi/4 (the classic unfocused-aperture limit
    ``L ~ sqrt(lambda*r/2)``); use ``focused_sar`` beyond that.
    """
    _require_coherent(ds, "unfocused_sar")
    spacing = _mean_spacing(ds)
    if n_traces is None:
        if aperture_m is None:
            raise ValueError("give unfocused_sar n_traces or aperture_m")
        n = max(1, int(round(aperture_m / spacing))) if spacing > 0 else 1
    else:
        n = int(n_traces)
    if n < 1:
        raise ValueError("unfocused window must be >= 1 trace")
    T = ds.sizes["slow_time"]
    if n > T:
        raise ValueError(f"unfocused window n={n} exceeds slow_time length {T}")
    # Doppler-aliasing guard for the coherent (point/specular) part.
    if aperture_m is not None and "wavelength" in ds.attrs and "nadir_twtt" in ds:
        r_ref = float(np.nanmedian(ds.nadir_twtt.values)) * _speed_of_light(ds) / 2.0
        _doppler_guard(spacing, ds.attrs["wavelength"], aperture_m, r_ref,
                       "unfocused_sar")
    n_out = T - n + 1
    W = np.zeros((n_out, T))
    for k in range(n_out):
        W[k, k:k + n] = 1.0
    step = {"op": "unfocused_sar", "n_traces": n,
            "aperture_m": aperture_m, "mean_spacing_m": spacing,
            "n_traces_in": T, "n_traces_out": n_out,
            "convention": "stride-1 coherent moving boxcar (CSARP_standard-like "
                          "unfocused SAR); preserves along-track sampling"}
    return _apply_along_slow_time(ds, W, coherent=True, step=step)


def multilook(ds, n):
    """Incoherent multilook: non-overlapping average of ``power`` over ``n``
    traces (speckle reduction). Works on both modes; any ``field`` is dropped
    (multilooking destroys phase coherence). For fully-developed speckle the
    intensity contrast (std/mean) drops as ``1/sqrt(n)``.
    """
    n = int(n)
    if n < 1:
        raise ValueError("multilook count n must be >= 1")
    T = ds.sizes["slow_time"]
    n_out = T // n
    if n_out < 1:
        raise ValueError(f"multilook n={n} exceeds slow_time length {T}")
    W = np.zeros((n_out, T))
    for k in range(n_out):
        W[k, k * n:(k + 1) * n] = 1.0
    step = {"op": "multilook", "n": n, "n_traces_in": T, "n_traces_out": n_out,
            "convention": "non-overlapping incoherent power average; "
                          "speckle contrast ~ 1/sqrt(n)"}
    return _apply_along_slow_time(ds, W, coherent=False, step=step)


def _doppler_guard(spacing, wavelength, aperture_m, r_ref, op):
    """Warn if the along-track spacing under-samples the coherent Doppler
    history for the requested aperture (the lambda/4 criterion from the stage-1
    handoff notes: a point/specular target is coherent across the aperture, so
    adjacent-trace phase must advance < pi -> spacing < lambda/(4 sin theta))."""
    if r_ref <= 0:
        return
    theta = np.arctan((aperture_m / 2.0) / r_ref)
    st = np.sin(theta)
    if st <= 0:
        return
    crit = wavelength / (4.0 * st)
    if spacing > crit:
        warnings.warn(
            f"{op}: along-track trace spacing {spacing:.3f} m exceeds "
            f"lambda/(4 sin theta_max) = {crit:.3f} m for a {aperture_m:.0f} m "
            f"aperture at range {r_ref:.0f} m (half-angle "
            f"{np.degrees(theta):.1f} deg): coherent (point/specular) returns "
            "will alias in Doppler. Densify the track or shrink the aperture.")


def focused_sar(ds, aperture_m, *, window="hann"):
    """Straight-track time-domain backprojection focusing (through AIR only).

    Validation-grade focused SAR (plan D4-3), NOT a production processor: no
    motion compensation, no autofocus, straight-track geometry, and focusing is
    surface-referenced through air only -- in-ice focusing is explicitly out of
    scope (the twtt->range map used here is ``r = c*twtt/2`` in air). Meant for
    point-target validation and simulated-data studies.

    For each output trace ``i`` and fast-time bin (range ``r0 = c*twtt/2``), the
    complex samples of every aperture trace ``j`` within ``|s_j - s_i| <=
    aperture_m/2`` are summed after range-migration correction: the sample is
    read at the migrated delay ``twtt + 2*dr/c`` (linear complex interpolation,
    sub-bin) with ``dr = sqrt(r0**2 + (s_j - s_i)**2) - r0`` and phase-corrected
    by ``exp(+2j*k*dr)`` (aligning each trace's carrier phase to the closest-
    approach range ``r0``). ``window`` weights the aperture (hann default;
    "none" = rectangular). Fully-focused azimuth resolution is
    ``lambda*r/(2*L_ap)`` (rectangular; hann broadens it ~1.44x).

    Requires a 2-D coherent ``field`` (slow_time, twtt): select any ``side`` /
    ``layer`` beforehand. Emits the lambda/4 Doppler-aliasing guard.
    """
    _require_coherent(ds, "focused_sar")
    if set(ds.field.dims) != {"slow_time", "twtt"}:
        raise ValueError(
            "focused_sar needs a 2-D (slow_time, twtt) field; select a single "
            f"side/layer first (got dims {ds.field.dims})")
    if "wavelength" not in ds.attrs:
        raise ValueError("focused_sar needs the `wavelength` attr (coherent run)")

    c = _speed_of_light(ds)
    lam = float(ds.attrs["wavelength"])
    k = 2.0 * np.pi / lam
    twtt = ds.twtt.values.astype(float)
    dt = float(twtt[1] - twtt[0])
    t0 = float(twtt[0])
    nb = twtt.size
    r0 = c * twtt / 2.0
    s = _along_track(ds)
    F = ds.field.transpose("slow_time", "twtt").values.astype(np.complex128)
    T = F.shape[0]
    half = aperture_m / 2.0

    if window == "hann":
        win = lambda d: 0.5 * (1.0 + np.cos(np.pi * d / half))  # d in [-half, half]
    elif window in ("none", "rect", None):
        win = lambda d: np.ones_like(d)
    else:
        raise ValueError(f"unknown window {window!r} (use 'hann' or 'none')")

    spacing = _mean_spacing(ds)
    _doppler_guard(spacing, lam, aperture_m, float(r0.min()), "focused_sar")

    out = np.zeros((T, nb), dtype=np.complex128)
    for i in range(T):
        d = s - s[i]
        J = np.nonzero(np.abs(d) <= half)[0]
        if J.size == 0:
            continue
        dJ = d[J][:, None]                       # (nJ, 1)
        dr = np.sqrt(r0[None, :] ** 2 + dJ ** 2) - r0[None, :]   # (nJ, nb)
        tau = twtt[None, :] + 2.0 * dr / c
        x = (tau - t0) / dt
        i0 = np.floor(x).astype(np.int64)
        frac = x - i0
        valid = (i0 >= 0) & (i0 < nb - 1)
        i0c = np.clip(i0, 0, nb - 2)
        rows = J[:, None]
        samp = F[rows, i0c] * (1.0 - frac) + F[rows, i0c + 1] * frac
        samp = np.where(valid, samp, 0.0)
        w = win(d[J])[:, None]
        out[i] = np.sum(w * samp * np.exp(2j * k * dr), axis=0)

    n_ap = int(np.round(aperture_m / spacing)) + 1 if spacing > 0 else T
    step = {"op": "focused_sar", "aperture_m": aperture_m, "window": window,
            "mean_spacing_m": spacing, "n_aperture_traces_approx": n_ap,
            "model": "straight-track time-domain backprojection through air "
                     "(surface-referenced); in-ice focusing out of scope"}
    attrs = _record(dict(ds.attrs), step)
    new = ds.copy()
    new.attrs.clear()
    new.attrs.update(attrs)
    new["field"] = (("slow_time", "twtt"), out.astype(np.complex64),
                    dict(ds.field.attrs))
    new["power"] = (("slow_time", "twtt"), np.abs(out).astype(np.float32) ** 2,
                    dict(ds.power.attrs))
    return new
