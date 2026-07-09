"""Waveform / chirp convolution (stage 4, M20).

Textbook references (Harris 1978, "On the Use of Windows for Harmonic
Analysis with the Discrete Fourier Transform", Proc. IEEE 66(1), Table 1;
identical figures in Levanon & Mozeson 2004, "Radar Signals", ch. 5): a
raised-cosine amplitude weighting applied once across a compressed bandwidth
B gives a compressed pulse (the window's Fourier transform) with

    window    peak sidelobe (dB)   -3 dB main-lobe width (x 1/B)
    none      -13.3                0.886
    hann      -31.5                1.44
    hamming   -42.7                1.30

The "power dB" sidelobe equals 20*log10 of the envelope ratio (power =
|p|^2), so the window's spectral sidelobe figures apply to the compressed
power response directly. Range resolution = width * c/(2B).

The point-target pipeline tests run the real coherent kernel on a single
small facet placed at a controlled sub-bin offset ``frac`` inside bin B0 and
convolve with waveform.apply_waveform -- peak bin, peak phase (= the delta-
mode carrier phase, since the symmetric-window kernel is real), resolution,
sidelobes, energy ratio (= sum |p|^2 exactly, by linearity), and the
interp_bins sub-bin peak placement are all checked against closed forms.
"""

import numpy as np
import pytest

import soundersim
from soundersim.config import (FacetConfig, Medium, RadarConfig, SimConfig,
                               WaveformConfig)
from soundersim.kernels.coherent import coherent_cluttergram
from soundersim.nav import nav_to_frame
from soundersim.physics import fresnel_normal
from soundersim.scene import LocalFrame, build_facets
from soundersim import synthetic as syn
from soundersim.waveform import (apply_waveform, compressed_pulse,
                                 convolve_fast_time)

C = 299792458.0
F0, BW, PL = 195e6, 30e6, 3e-6
DT, NSAMP, T0 = 5e-9, 128, 3e-6
B0 = 40  # point-target bin
GAMMA = -0.281
UCT = np.array([[0.0, -1.0, 0.0]])

# Harris 1978 Table 1 (see module docstring): (PSL dB, -3 dB width x 1/B)
TEXTBOOK = {"none": (-13.3, 0.886), "hann": (-31.5, 1.44),
            "hamming": (-42.7, 1.30)}


def _rc(kind="chirp", window="hann", interp=False, **kw):
    wf = WaveformConfig(kind=kind, bandwidth=BW, pulse_length=PL,
                        window=window, interp_bins=interp
                        ) if kind == "chirp" else WaveformConfig()
    return RadarConfig(dt=DT, n_samples=NSAMP, t0=T0, f0=F0, waveform=wf, **kw)


def point_target(frac, *, interp=False, n_samples=NSAMP):
    """Single 0.5 m facet at nadir, delay t0 + (B0 + frac)*dt; returns
    (field, dropped) from the real coherent kernel."""
    h = 0.5 * C * (T0 + (B0 + frac) * DT)
    L = 0.5
    centers = np.zeros((1, 3))
    normals = np.array([[0.0, 0.0, 1.0]])
    e1, e2 = np.array([[L, 0.0, 0.0]]), np.array([[0.0, L, 0.0]])
    return coherent_cluttergram(
        np.array([[0.0, 0.0, h]]), UCT, centers, normals, np.array([L * L]),
        e1, e2, k=2 * np.pi * F0 / C, gamma=GAMMA, t0=T0, dt=DT,
        n_samples=n_samples, c=C, interp_bins=interp)


def _peak_pos(a):
    """Sub-bin peak via 3-point parabolic interpolation of the envelope."""
    a = np.asarray(a, float)
    i = int(np.argmax(a))
    y0, y1, y2 = a[i - 1], a[i], a[i + 1]
    return i + 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)


def test_compressed_pulse_textbook():
    """Dense-sampled kernel: -3 dB width and PSL vs Harris 1978 per window;
    real, peak-normalized, symmetric, physical +-T support."""
    dt = 1.0 / (200.0 * BW)
    for window, (psl_ref, w_ref) in TEXTBOOK.items():
        p, m = compressed_pulse(BW, PL, dt, window=window)
        assert p.dtype == np.float64 and len(p) == 2 * m + 1
        assert p[m] == 1.0 and np.allclose(p, p[::-1])
        a = np.abs(p)
        # -3 dB full width of |p|^2: linear-interp crossings of 1/sqrt(2)
        half = 1.0 / np.sqrt(2.0)
        right = m + np.flatnonzero(a[m:] < half)[0]
        fr = (a[right - 1] - half) / (a[right - 1] - a[right])
        width = 2.0 * (right - 1 + fr - m) * dt * BW  # symmetric
        assert width == pytest.approx(w_ref, rel=0.03), window
        # PSL: max envelope beyond the first local minimum after the peak
        dmin = m + np.flatnonzero(np.diff(a[m:]) > 0)[0]
        psl = 20.0 * np.log10(a[dmin:].max())
        assert abs(psl - psl_ref) <= 1.0, (window, psl)


def test_delta_default_bit_compatible():
    """simulate() with the default (delta) waveform is bit-identical to the
    raw kernel output -- the waveform layer is a strict identity."""
    scene = syn.flat_scene(extent=1200.0, n_traces=3, altitude=1000.0,
                           posting=50.0)
    rc = RadarConfig(dt=2e-8, n_samples=160, t0=6.5e-6, f0=195e6)
    cfg = SimConfig(mode="coherent", radar=rc, facets=FacetConfig(spacing=15.0))
    ds = soundersim.simulate(scene, cfg)
    assert ds.attrs["waveform"] == "delta"

    frame = LocalFrame.centered_on(scene)
    facets = build_facets(scene.dem, scene.transform, scene.crs, frame,
                          spacing=15.0)
    track = nav_to_frame(scene.nav_llh, frame)
    lam = rc.c / rc.f0
    field, dropped = coherent_cluttergram(
        track.positions, track.u_ct, facets.centers, facets.normals,
        facets.areas, facets.e1, facets.e2, k=2.0 * np.pi / lam,
        gamma=fresnel_normal(1.0, 3.17), t0=rc.t0, dt=rc.dt,
        n_samples=rc.n_samples, c=rc.c)
    assert np.array_equal(ds.field.values, field)
    assert np.array_equal(ds.dropped_power.values, dropped)


def test_point_target_peak_bin_phase_and_energy():
    """Compressed peak in the target's bin with the target's carrier phase
    and the delta-mode magnitude (p(0)=1); output/input energy = sum|p|^2."""
    field, _ = point_target(0.25)
    assert np.count_nonzero(field[0]) == 1  # delta response, one bin
    rc = _rc()
    y = apply_waveform(field, rc, "coherent")
    assert y.dtype == np.complex64
    assert int(np.argmax(np.abs(y[0]))) == B0
    a0 = field[0, B0]
    np.testing.assert_allclose(np.abs(y[0, B0]), np.abs(a0), rtol=1e-5)
    dphi = np.angle(y[0, B0] / a0)
    assert abs(dphi) < 1e-4  # real kernel: carrier phase preserved exactly
    p, _ = compressed_pulse(BW, PL, DT, "hann")
    e_ratio = (np.abs(y[0]) ** 2).sum() / (np.abs(field[0]) ** 2).sum()
    np.testing.assert_allclose(e_ratio, (p ** 2).sum(), rtol=1e-4)


def test_point_target_resolution_and_psl_through_pipeline():
    """-3 dB width ~ 1.44/B (hann) and PSL ~ -31.5 dB on the dt-sampled
    convolved trace (window main lobe spans ~10 bins at these parameters)."""
    field, _ = point_target(0.25)
    y = np.abs(apply_waveform(field, _rc(), "coherent")[0]).astype(float)
    a = y / y.max()
    half = 1.0 / np.sqrt(2.0)
    i = int(np.argmax(a))
    right = i + np.flatnonzero(a[i:] < half)[0]
    fr = (a[right - 1] - half) / (a[right - 1] - a[right])
    left = i - np.flatnonzero(a[i::-1] < half)[0]
    fl = (a[left + 1] - half) / (a[left + 1] - a[left])
    width = ((right - 1 + fr) - (left + 1 - fl)) * DT * BW
    assert width == pytest.approx(1.44, rel=0.05)
    dmin = i + np.flatnonzero(np.diff(a[i:]) > 0)[0]
    psl = 20.0 * np.log10(a[dmin:dmin + 40].max())
    assert abs(psl - (-31.5)) <= 1.5


def test_interp_bins_peak_position():
    """Sub-bin envelope placement: parabolic-fit peak error < 0.1 bin with
    interp_bins vs ~0.5 bin (the quantization) without."""
    frac = 0.47
    rc = _rc()
    truth = B0 + frac
    f_no, _ = point_target(frac, interp=False)
    f_in, _ = point_target(frac, interp=True)
    err_no = abs(_peak_pos(np.abs(apply_waveform(f_no, rc, "coherent")[0]))
                 - truth)
    err_in = abs(_peak_pos(np.abs(apply_waveform(f_in, rc, "coherent")[0]))
                 - truth)
    assert err_no > 0.3  # measured ~0.47: the full quantization error
    assert err_in < 0.1  # measured << 0.1


def test_interp_bins_field_sum_and_dropped():
    """Linear splitting preserves each facet's total complex contribution
    (bin-sum of fields equal to the unsplit kernel) and drops split parts
    with |part|^2 accounting at the window edge."""
    # multi-facet scene fully inside the window: bin-sums agree
    rng = np.random.default_rng(7)
    nf = 30
    centers = np.column_stack([rng.uniform(-40, 40, (nf, 2)),
                               rng.uniform(-2, 2, nf)])
    L = 2.0
    e1 = np.tile([L, 0.0, 0.0], (nf, 1))
    e2 = np.tile([0.0, L, 0.0], (nf, 1))
    normals = np.tile([0.0, 0.0, 1.0], (nf, 1))
    areas = np.full(nf, L * L)
    p = np.array([[0.0, 0.0, 700.0]])
    kw = dict(k=2 * np.pi / 1.5, gamma=GAMMA, t0=2 * 640.0 / C, dt=2e-9,
              n_samples=256, c=C)
    f0_, d0 = coherent_cluttergram(p, UCT, centers, normals, areas, e1, e2,
                                   **kw)
    f1_, d1 = coherent_cluttergram(p, UCT, centers, normals, areas, e1, e2,
                                   interp_bins=True, **kw)
    assert d0[0] == 0.0 and d1[0] == 0.0
    np.testing.assert_allclose(f1_[0].sum(), f0_[0].sum(), rtol=1e-5)

    # window edge: the b+1 half of a last-bin facet is dropped as |w*c|^2
    frac = 0.3
    full, _ = point_target(frac, interp=False, n_samples=B0 + 2)
    amp = np.abs(full[0, B0])
    _, dropped = point_target(frac, interp=True, n_samples=B0 + 1)
    np.testing.assert_allclose(dropped[0], (frac * amp) ** 2, rtol=1e-3)


def test_chirp_config_validation():
    with pytest.raises(ValueError, match="bandwidth"):
        WaveformConfig(kind="chirp", pulse_length=1e-6)
    with pytest.raises(ValueError, match="pulse_length"):
        WaveformConfig(kind="chirp", bandwidth=30e6)
    assert WaveformConfig().kind == "delta"  # no params needed


def test_simulate_chirp_end_to_end():
    """Coherent chirp through simulate(): power = |field|^2 of the convolved
    field, waveform attrs recorded, result differs from delta."""
    scene = syn.flat_scene(extent=1200.0, n_traces=2, altitude=1000.0,
                           posting=50.0)
    rc_d = RadarConfig(dt=2e-8, n_samples=160, t0=6.5e-6, f0=195e6)
    wf = WaveformConfig(kind="chirp", bandwidth=30e6, pulse_length=3e-6,
                        interp_bins=True)
    rc_c = rc_d.model_copy(update={"waveform": wf})
    fac = FacetConfig(spacing=15.0)
    ds_d = soundersim.simulate(scene, SimConfig(mode="coherent", radar=rc_d,
                                                facets=fac))
    ds_c = soundersim.simulate(scene, SimConfig(mode="coherent", radar=rc_c,
                                                facets=fac))
    np.testing.assert_array_equal(ds_c.power.values,
                                  np.abs(ds_c.field.values) ** 2)
    assert ds_c.attrs["waveform"] == "chirp"
    assert ds_c.attrs["bandwidth"] == 30e6
    assert ds_c.attrs["waveform_window"] == "hann"
    assert not np.array_equal(ds_c.field.values, ds_d.field.values)
    # delta and chirp surface peaks comparable (p(0) = 1 normalization);
    # the chirped peak collects the leading edge into the main lobe, so it
    # can only exceed the delta per-bin peak by the pulse compression gain
    ratio = float(ds_c.power.max() / ds_d.power.max())
    assert 0.8 < ratio < 20.0


def test_incoherent_envelope_default_off_and_opt_in():
    """Incoherent + chirp leaves power untouched by default (simc parity,
    D4-4); opting in convolves with the |p|^2 envelope."""
    scene = syn.flat_scene(extent=1200.0, n_traces=2, altitude=1000.0,
                           posting=50.0)
    fac = FacetConfig()
    rc_d = RadarConfig(dt=2e-8, n_samples=160, t0=6.5e-6)
    ds_d = soundersim.simulate(scene, SimConfig(mode="incoherent", radar=rc_d,
                                                facets=fac))
    wf = WaveformConfig(kind="chirp", bandwidth=30e6, pulse_length=3e-6)
    rc_c = rc_d.model_copy(update={"waveform": wf})
    ds_c = soundersim.simulate(scene, SimConfig(mode="incoherent", radar=rc_c,
                                                facets=fac))
    np.testing.assert_array_equal(ds_c.power.values, ds_d.power.values)

    wf_on = wf.model_copy(update={"incoherent_envelope": True})
    rc_on = rc_d.model_copy(update={"waveform": wf_on})
    ds_on = soundersim.simulate(scene, SimConfig(mode="incoherent",
                                                 radar=rc_on, facets=fac))
    assert not np.array_equal(ds_on.power.values, ds_d.power.values)
    assert (ds_on.power.values >= 0).all()


def test_quantization_alias_warning():
    """Chirped coherent run with the envelope-quantization carrier alias in
    band (|f0 - round(f0*dt)/dt| < B/2) warns unless interp_bins is set;
    an alias-free dt stays silent (M21 measurement at 195 MHz / 5 ns: the
    in-band alias floors a smooth-surface chirped profile at ~ -18 dB rel
    the surface peak with plain binning, 8-16 dB lower with interp_bins;
    see tests/test_waveform_pedestal.py)."""
    import warnings as w
    scene = syn.flat_scene(extent=1200.0, n_traces=2, altitude=1000.0,
                           posting=50.0)
    fac = FacetConfig(spacing=15.0)

    def run(dt, interp):
        wf = WaveformConfig(kind="chirp", bandwidth=30e6, pulse_length=3e-6,
                            interp_bins=interp)
        rc = RadarConfig(dt=dt, n_samples=160, t0=6.5e-6, f0=195e6,
                         waveform=wf)
        return SimConfig(mode="coherent", radar=rc, facets=fac)

    with pytest.warns(UserWarning, match="alias"):
        soundersim.simulate(scene, run(5e-9, False))  # alias at -5 MHz
    for cfg in (run(5e-9, True),      # interp suppresses -> no warning
                run(4e-9, False)):    # alias at -55 MHz, out of band
        with w.catch_warnings():
            w.simplefilter("error")
            soundersim.simulate(scene, cfg)


def test_multilayer_interp_bins_rejected():
    """interp_bins is coherent-kernel-only for now: multilayer runs refuse it
    up front instead of silently splitting only the surface layer."""
    scene = syn.slab_scene(surface=500.0, depth=300.0, extent=2000.0,
                           n_traces=2, altitude=1000.0)
    from soundersim.config import DemInterface
    wf = WaveformConfig(kind="chirp", bandwidth=30e6, pulse_length=3e-6,
                        interp_bins=True)
    cfg = SimConfig(
        mode="coherent",
        radar=RadarConfig(dt=1e-8, n_samples=1250, t0=0.0, f0=195e6,
                          waveform=wf),
        facets=FacetConfig(spacing=15.0),
        media=[Medium(name="air", eps_r=1.0), Medium(name="ice", eps_r=3.17),
               Medium(name="bed", eps_r=6.0)],
        interfaces=[DemInterface(name="surface"), DemInterface(name="bed")])
    with pytest.raises(ValueError, match="interp_bins"):
        soundersim.simulate(scene, cfg)


def test_convolve_fast_time_matches_numpy():
    """FFT convolution equals np.convolve 'same' semantics with the kernel
    peak at its center index, for trailing extra dims too."""
    rng = np.random.default_rng(11)
    arr = (rng.normal(size=(2, 50, 2)) + 1j * rng.normal(size=(2, 50, 2)))
    ker = rng.normal(size=21)
    out = convolve_fast_time(arr, ker, 10)
    for t in range(2):
        for s in range(2):
            ref = np.convolve(arr[t, :, s], ker, mode="full")[10:60]
            np.testing.assert_allclose(out[t, :, s], ref, atol=1e-12)
