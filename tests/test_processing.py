"""CI-fast tests for post-processing (stage 4, M23): presum / unfocused /
focused SAR / multilook.

Coherent Datasets are built deterministically (known complex field) so the
gains are exact rather than statistical; the focused point-target and speckle
tests use analytic constructions of the kernel's single-facet delta response
and fully-developed speckle respectively.
"""

import json

import numpy as np
import pytest
import xarray as xr

from soundersim import processing as proc

C = 299792458.0
F0 = 195e6
LAM = C / F0


def make_coherent_ds(field, *, twtt, x, y=None, z=None, c=C, wavelength=LAM,
                     mode="coherent"):
    """Minimal Dataset matching docs/output.md that the processors consume."""
    field = np.asarray(field, np.complex64)
    T, nb = field.shape
    x = np.asarray(x, float)
    y = np.zeros(T) if y is None else np.asarray(y, float)
    z = np.full(T, 1000.0) if z is None else np.asarray(z, float)
    twtt = np.asarray(twtt, float)
    nadir = np.full(T, twtt[0])
    ds = xr.Dataset(
        {"power": (("slow_time", "twtt"), np.abs(field) ** 2),
         "field": (("slow_time", "twtt"), field),
         "nadir_twtt": ("slow_time", nadir),
         "dropped_power": ("slow_time", np.zeros(T, np.float32))},
        coords={"slow_time": np.arange(T), "twtt": twtt,
                "trace": ("slow_time", np.arange(T)),
                "lat": ("slow_time", np.full(T, 75.0)),
                "lon": ("slow_time", np.full(T, -40.0)),
                "elevation": ("slow_time", z),
                "x": ("slow_time", x), "y": ("slow_time", y),
                "z": ("slow_time", z)},
        attrs={"mode": mode, "wavelength": wavelength,
               "config": json.dumps({"radar": {"c": c}})})
    if mode != "coherent":
        ds = ds.drop_vars("field")
    return ds


# --------------------------------------------------------------------------
# presum
# --------------------------------------------------------------------------

def test_presum_coherent_point_target_gain():
    """A constant-phase target gains n in field amplitude, n**2 in power."""
    T, nb, n = 12, 5, 3
    v = 0.7 - 0.4j
    field = np.zeros((T, nb), np.complex64)
    field[:, 2] = v                       # same phasor in every trace: coherent
    ds = make_coherent_ds(field, twtt=1e-6 + np.arange(nb) * 1e-8,
                          x=np.arange(T) * 0.2)
    out = proc.presum(ds, n)

    assert out.sizes["slow_time"] == T // n
    np.testing.assert_allclose(out.field.values[:, 2], n * v, rtol=1e-5)
    np.testing.assert_allclose(out.power.values[:, 2], (n * abs(v)) ** 2,
                               rtol=1e-4)
    # power contract preserved on the returned Dataset
    np.testing.assert_allclose(out.power.values, np.abs(out.field.values) ** 2,
                               rtol=1e-5)


def test_presum_snr_gain_vs_noise():
    """Coherent signal grows as n, incoherent noise as sqrt(n): SNR gain sqrt(n)."""
    rng = np.random.default_rng(0)
    T, nb, n = 400, 1, 4
    sig = 0.05 + 0.0j
    noise = (rng.standard_normal((T, nb)) + 1j * rng.standard_normal((T, nb)))
    ds = make_coherent_ds(sig + noise, twtt=[1e-6], x=np.arange(T) * 0.2)
    out = proc.presum(ds, n)
    # signal part is deterministic: amplitude scales exactly by n
    sig_out = out.field.values.mean()            # noise averages toward 0
    assert abs(sig_out - n * sig) < 3.0          # dominated by residual noise/sqrt
    # noise power per presummed trace ~ n (variance adds), vs n**2 for signal:
    # SNR (signal power / noise power) improves by ~n over a single trace pair.
    in_snr = abs(sig) ** 2 / np.var(noise)
    out_noise_var = np.var(out.field.values - n * sig)
    out_snr = (n * abs(sig)) ** 2 / out_noise_var
    assert out_snr / in_snr == pytest.approx(n, rel=0.4)


def test_presum_coords_and_length():
    T, n = 10, 2
    x = np.arange(T, dtype=float) * 5.0
    field = np.ones((T, 3), np.complex64)
    ds = make_coherent_ds(field, twtt=1e-6 + np.arange(3) * 1e-8, x=x)
    out = proc.presum(ds, n)
    assert out.sizes["slow_time"] == 5
    # block-averaged along-track position: mean of (0,5), (10,15), ...
    np.testing.assert_allclose(out.x.values, [2.5, 12.5, 22.5, 32.5, 42.5])
    np.testing.assert_allclose(out.slow_time.values, [0.5, 2.5, 4.5, 6.5, 8.5])
    assert out.twtt.size == 3  # fast-time axis untouched


def test_presum_records_processing_attr():
    ds = make_coherent_ds(np.ones((6, 2), np.complex64),
                          twtt=[1e-6, 1.01e-6], x=np.arange(6) * 0.2)
    out = proc.presum(ds, 2)
    steps = json.loads(out.attrs["processing"])
    assert isinstance(steps, list) and steps[-1]["op"] == "presum"
    assert steps[-1]["n"] == 2 and steps[-1]["n_traces_out"] == 3
    # appendable: chaining adds a second descriptor
    out2 = proc.presum(out, 1)
    assert len(json.loads(out2.attrs["processing"])) == 2


def test_presum_refuses_incoherent():
    ds = make_coherent_ds(np.ones((6, 2), np.complex64), twtt=[1e-6, 1.01e-6],
                          x=np.arange(6) * 0.2, mode="incoherent")
    with pytest.raises(ValueError, match="multilook"):
        proc.presum(ds, 2)


# --------------------------------------------------------------------------
# unfocused SAR
# --------------------------------------------------------------------------

def test_unfocused_matches_presum_on_constant_target():
    """On a constant-phase target the sliding sum equals the presum block at
    every aligned index."""
    T, nb, n = 9, 4, 3
    v = 1.0 + 0.5j
    field = np.zeros((T, nb), np.complex64)
    field[:, 1] = v
    ds = make_coherent_ds(field, twtt=1e-6 + np.arange(nb) * 1e-8,
                          x=np.arange(T) * 0.2)
    uf = proc.unfocused_sar(ds, n_traces=n)
    ps = proc.presum(ds, n)
    assert uf.sizes["slow_time"] == T - n + 1
    # presum block k == unfocused window at output index k*n
    np.testing.assert_allclose(ps.field.values,
                               uf.field.values[::n], rtol=1e-5)
    np.testing.assert_allclose(uf.field.values[:, 1], n * v, rtol=1e-5)


def test_unfocused_aperture_to_traces():
    T = 20
    ds = make_coherent_ds(np.ones((T, 2), np.complex64),
                          twtt=[1e-6, 1.01e-6], x=np.arange(T) * 2.0)
    out = proc.unfocused_sar(ds, aperture_m=8.0)   # spacing 2 m -> 4 traces
    step = json.loads(out.attrs["processing"])[-1]
    assert step["n_traces"] == 4 and out.sizes["slow_time"] == T - 4 + 1


# --------------------------------------------------------------------------
# focused SAR (point target)
# --------------------------------------------------------------------------

def _point_target_ds(H=200.0, aperture=80.0, spacing=0.25, extra=10.0,
                     dt=2e-9, bandwidth=30e6):
    """Analytic single-point-scatterer field along a straight, level track.

    Scatterer at the origin; platform at (s, 0, H). Each trace carries the
    target's range response with exact carrier phase exp(-2jk r): a finite-
    bandwidth (hann-windowed) compressed-pulse envelope in fast time, so the
    range PSF is broad relative to the sub-metre range migration across the
    aperture -- the narrow-band sounder regime where azimuth focusing yields
    the textbook lambda*r/(2L) resolution (a bare range delta would couple
    range and azimuth and give a nonphysical width)."""
    k = 2.0 * np.pi / LAM
    s = np.arange(-(aperture / 2 + extra), aperture / 2 + extra + spacing / 2,
                  spacing)
    T = s.size
    r = np.sqrt(s ** 2 + H ** 2)
    t0 = 2.0 * (H - 30.0) / C
    nb = int(np.ceil((2.0 * r.max() / C + 60.0 * 2.0 / C - t0) / dt)) + 4
    twtt = t0 + np.arange(nb) * dt
    tau = twtt[None, :] - (2.0 * r[:, None] / C)          # (T, nb)
    x = bandwidth * tau
    a = 0.5                                                # hann range weighting
    p = (a * np.sinc(x) + 0.5 * (1 - a) * (np.sinc(x - 1) + np.sinc(x + 1))) / a
    field = (p * np.exp(-2j * k * r[:, None])).astype(np.complex64)
    ds = make_coherent_ds(field, twtt=twtt, x=s, z=np.full(T, H))
    return ds, s, twtt, H


def _minus3db_width(profile_db, coord):
    """Full width where a peak-normalised dB profile crosses -3 dB (linear
    interpolation on each side of the peak)."""
    i0 = int(np.argmax(profile_db))
    tgt = -3.0

    def cross(idx_range):
        prev = i0
        for i in idx_range:
            if profile_db[i] <= tgt:
                f = (tgt - profile_db[prev]) / (profile_db[i] - profile_db[prev])
                return coord[prev] + f * (coord[i] - coord[prev])
            prev = i
        return None

    left = cross(range(i0 - 1, -1, -1))
    right = cross(range(i0 + 1, len(profile_db)))
    return None if left is None or right is None else right - left


def test_focused_point_target_resolution_and_gain():
    L = 80.0
    ds, s, twtt, H = _point_target_ds(aperture=L)
    focused = proc.focused_sar(ds, aperture_m=L, window="none")

    b0 = int(round((2.0 * H / C - twtt[0]) / (twtt[1] - twtt[0])))
    amp = np.abs(focused.field.values[:, b0])
    prof_db = 20.0 * np.log10(np.maximum(amp, 1e-12) / amp.max())

    width = _minus3db_width(prof_db, s)
    pred = LAM * H / (2.0 * L)             # Rayleigh; rect -3 dB is 0.886x this
    assert width is not None
    assert 0.85 <= width / pred <= 1.15, (width, pred)

    # peak coherent gain ~ number of aperture traces (unit-amplitude, in phase)
    n_ap = int(np.sum(np.abs(s - s[np.argmax(amp)]) <= L / 2))
    assert amp.max() / n_ap > 0.7

    # first sidelobe well below the main lobe (rect ~ -13.3 dB; loose gate)
    i0 = int(np.argmax(amp))
    mask = np.abs(s - s[i0]) > 2.0 * pred   # outside the main lobe
    assert prof_db[mask].max() < -10.0

    step = json.loads(focused.attrs["processing"])[-1]
    assert step["op"] == "focused_sar" and step["window"] == "none"


def test_focused_beats_unfocused_azimuth():
    """Focused azimuth response is narrower than the raw (unfocused) peak."""
    L = 80.0
    ds, s, twtt, H = _point_target_ds(aperture=L)
    focused = proc.focused_sar(ds, aperture_m=L, window="none")
    b0 = int(round((2.0 * H / C - twtt[0]) / (twtt[1] - twtt[0])))
    amp_f = np.abs(focused.field.values[:, b0])
    amp_raw = np.abs(ds.field.values[:, b0])
    wf = _minus3db_width(
        20 * np.log10(np.maximum(amp_f, 1e-12) / amp_f.max()), s)
    # raw energy at this bin is spread over the whole aperture; focused is a
    # tight main lobe of a few metres
    assert wf < 0.2 * L


def test_focused_requires_2d_field():
    ds = make_coherent_ds(np.ones((6, 3), np.complex64),
                          twtt=1e-6 + np.arange(3) * 2e-9, x=np.arange(6) * 0.2)
    ds2 = ds.expand_dims(side=["left", "right"]).transpose(
        "slow_time", "twtt", "side")
    with pytest.raises(ValueError, match="2-D"):
        proc.focused_sar(ds2, aperture_m=1.0)


def test_focused_doppler_guard_warns():
    """Coarse along-track spacing trips the lambda/4 aliasing guard."""
    H, L, spacing = 200.0, 80.0, 2.0     # crit ~ 1.96 m at 11.3 deg half-angle
    ds, *_ = _point_target_ds(H=H, aperture=L, spacing=spacing, extra=4.0)
    with pytest.warns(UserWarning, match="alias"):
        proc.focused_sar(ds, aperture_m=L, window="none")


def test_focused_no_warning_when_dense():
    ds, *_ = _point_target_ds(H=200.0, aperture=80.0, spacing=0.25, extra=4.0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        proc.focused_sar(ds, aperture_m=80.0, window="none")


# --------------------------------------------------------------------------
# multilook (incoherent speckle reduction)
# --------------------------------------------------------------------------

def test_multilook_speckle_contrast():
    """Fully-developed speckle: intensity contrast drops as 1/sqrt(n)."""
    rng = np.random.default_rng(1)
    T, nb, n = 1600, 200, 16
    field = ((rng.standard_normal((T, nb))
              + 1j * rng.standard_normal((T, nb))) / np.sqrt(2)).astype(
                  np.complex64)
    ds = make_coherent_ds(field, twtt=1e-6 + np.arange(nb) * 1e-8,
                          x=np.arange(T) * 0.2)

    single = ds.power.values
    c1 = single.std() / single.mean()
    assert 0.9 <= c1 <= 1.1                      # single-look contrast ~ 1

    ml = proc.multilook(ds, n)
    assert "field" not in ml                     # phase dropped
    assert ml.sizes["slow_time"] == T // n
    p = ml.power.values
    cn = p.std() / p.mean()
    assert cn / (1.0 / np.sqrt(n)) == pytest.approx(1.0, rel=0.2)
    assert json.loads(ml.attrs["processing"])[-1]["op"] == "multilook"


def test_multilook_accepts_incoherent():
    rng = np.random.default_rng(2)
    T, nb = 200, 50
    power = rng.exponential(1.0, (T, nb)).astype(np.float32)
    ds = make_coherent_ds(np.sqrt(power).astype(np.complex64),
                          twtt=1e-6 + np.arange(nb) * 1e-8,
                          x=np.arange(T) * 0.2, mode="incoherent")
    ml = proc.multilook(ds, 4)
    assert ml.sizes["slow_time"] == 50 and "field" not in ml
