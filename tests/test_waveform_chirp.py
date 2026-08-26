"""Explicit-chirp compressed-pulse construction (``construction="chirp"``).

The analytic windowed sinc is the B*T -> inf limit; the explicit matched
filter carries the finite-TB Fresnel-ripple sidelobe pedestal and the true
+-T support. These tests pin (a) agreement with the analytic form in the
mainlobe at large TB, (b) normalisation and support, (c) the pedestal the
analytic form lacks (the whole point), (d) zero response past the pulse.
"""

import numpy as np
import pytest

from soundersim.config import RadarConfig, WaveformConfig
from soundersim.waveform import apply_waveform, compressed_pulse

B15, DT = 15e6, 8.3e-9   # HAPS 60 MHz design point at the runs' dt_sim


def _lag(m, dt):
    return np.arange(-m, m + 1) * dt


@pytest.mark.parametrize("window", ["none", "hann", "hamming"])
def test_large_tb_matches_analytic_in_mainlobe(window):
    """TB = 3000: mainlobe + first sidelobes agree to the O(1/sqrt(TB))
    Fresnel-ripple level (measured 5e-4 .. 1e-3; tolerance 5e-3)."""
    b, t, dt = 30e6, 100e-6, 5e-9
    p, m = compressed_pulse(b, t, dt, window, "chirp")
    pa, ma = compressed_pulse(b, t, dt, window)
    assert m == ma and len(p) == len(pa) == 2 * m + 1
    sel = np.abs(_lag(m, dt) * b) <= 3.0
    assert np.abs(p[sel] - pa[sel]).max() < 5e-3


def test_peak_and_support():
    for t in (20e-6, 5e-6):
        p, m = compressed_pulse(B15, t, DT, "hann", "chirp")
        assert m == int(np.ceil(t / DT)) and len(p) == 2 * m + 1
        assert p[m] == 1.0 + 0j                     # exactly at lag 0
        assert np.abs(p).max() == 1.0
        assert np.abs(p.imag).max() < 0.03         # ~real (symmetric window)
        assert np.all(np.abs(p[np.abs(_lag(m, DT)) > t]) == 0.0)


def test_pedestal_exists_for_20us_pulse():
    """B = 15 MHz, Hann, T = 20 us: local-max sidelobe ~-53 dB at 10 us lag
    (mission design tool build_sidelobes.py: ~-55 dB at 8-12 us); the
    analytic form is below -100 dB there."""
    p, m = compressed_pulse(B15, 20e-6, DT, "hann", "chirp")
    pa, _ = compressed_pulse(B15, 20e-6, DT, "hann")
    sel = np.abs(_lag(m, DT) - 10e-6) < 0.5e-6
    ped = 20 * np.log10(np.abs(p[sel]).max())
    assert -60.0 < ped < -50.0
    assert 20 * np.log10(np.abs(pa[sel]).max()) < -100.0


def test_short_pulse_is_zero_past_pulse_length():
    p, m = compressed_pulse(B15, 5e-6, DT, "hann", "chirp")
    lag = _lag(m, DT)
    assert np.all(p[lag > 5e-6] == 0.0) and np.all(p[lag < -5e-6] == 0.0)
    # ...and inside the pulse the pedestal is there
    sel = np.abs(lag - 3e-6) < 0.5e-6
    assert 20 * np.log10(np.abs(p[sel]).max()) > -50.0


def test_default_is_analytic_and_apply_waveform_routes():
    assert WaveformConfig().construction == "analytic"
    n = 4096
    field = np.zeros((1, n), np.complex64)
    field[0, 1500] = 1.0
    rc = RadarConfig(dt=DT, n_samples=n, t0=0.0, f0=60e6,
                     waveform=WaveformConfig(kind="chirp", bandwidth=B15,
                                             pulse_length=20e-6,
                                             construction="chirp"))
    y = apply_waveform(field, rc, "coherent")[0]
    assert y.dtype == np.complex64 and np.argmax(np.abs(y)) == 1500
    assert abs(y[1500]) == pytest.approx(1.0, abs=1e-6)
    i10 = 1500 + int(round(10e-6 / DT))
    assert 20 * np.log10(np.abs(y[i10 - 60:i10 + 60]).max()) > -60.0
