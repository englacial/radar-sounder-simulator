"""Multi-frequency synthesis referee vs the chirp convolution (M21, CI part).

The referee (compare/multifreq.py) synthesizes the compressed range profile
exactly per facet across the band; the production path bins a delta trace at
f0 and convolves with the compressed pulse. On an isolated point target the
two must overlay to a fraction of a dB over the main lobe and first
sidelobes (with interp_bins removing the sub-bin envelope quantization); the
band-mean-vs-center amplitude difference is second order thanks to midpoint
band sampling. The full-vs-frozen amplitude referee difference on a nadir
point target is tiny (the k/2pi prefactor is linear in f -> cancels at
midpoint sampling; the flat-surface directivity number lives in the
integration report case tests/test_waveform_pedestal.py).
"""

import numpy as np

from soundersim.compare.multifreq import multifreq_profile
from soundersim.kernels.coherent import coherent_cluttergram
from soundersim.waveform import compressed_pulse, convolve_fast_time

C = 299792458.0
F0, BW, PL = 195e6, 30e6, 3e-6
DT, NSAMP, T0 = 5e-9, 128, 3e-6
B0 = 40
GAMMA = -0.281
TWTT = T0 + np.arange(NSAMP) * DT


def _point_geom(frac):
    h = 0.5 * C * (T0 + (B0 + frac) * DT)
    L = 0.5
    return (np.array([0.0, 0.0, h]), np.zeros((1, 3)),
            np.array([[0.0, 0.0, 1.0]]), np.array([L * L]),
            np.array([[L, 0.0, 0.0]]), np.array([[0.0, L, 0.0]]))


def _conv_trace(frac, interp=True, window="hann"):
    """Kernel point-target trace convolved with the compressed pulse."""
    pos, centers, normals, areas, e1, e2 = _point_geom(frac)
    field, _ = coherent_cluttergram(
        pos[None], np.array([[0.0, -1.0, 0.0]]), centers, normals, areas,
        e1, e2, k=2 * np.pi * F0 / C, gamma=GAMMA, t0=T0, dt=DT,
        n_samples=NSAMP, c=C, interp_bins=interp)
    p, m = compressed_pulse(BW, PL, DT, window)
    return convolve_fast_time(field.astype(np.complex128), p, m)[0]


def test_point_target_convolution_vs_referee():
    """|profile| agreement <= 0.5 dB over the main lobe + first/second
    sidelobe crests (measured: 0.03 dB main lobe, <= 0.4 dB at the crests).
    Null-adjacent bins are excluded from the dB gate (relative error diverges
    where the response -> 0) and covered by an absolute gate instead:
    measured max |y - referee| = 1.8% of peak, on the steepest flank -- the
    second-order residual of the sub-bin linear splitting (two half-weight
    pulses dt apart vs one at the true delay, ~ dt^2 * p''/8)."""
    frac = 0.5
    y = _conv_trace(frac, interp=True)
    pos, centers, normals, areas, e1, e2 = _point_geom(frac)
    yr = multifreq_profile(pos, centers, normals, areas, e1, e2, gamma=GAMMA,
                           f0=F0, bandwidth=BW, c=C, twtt=TWTT, n_freq=96)
    tau = T0 + (B0 + frac) * DT
    region = np.abs(TWTT - tau) <= 3.5 / BW
    pk = np.abs(yr).max()
    m = region & (np.abs(yr) > pk * 10 ** (-32 / 20.0))
    assert m.sum() >= 15
    diff = 20.0 * np.log10(np.abs(y[m]) / np.abs(yr[m]))
    assert np.abs(diff).max() <= 0.5
    assert np.abs(y[region] - yr[region]).max() <= 0.03 * pk


def test_referee_frozen_amplitudes_nadir_point():
    """Frozen-at-f0 amplitudes match the full referee at nadir: exact at the
    peak (< 0.01 dB: midpoint band sampling cancels the linear-in-f k/2pi
    prefactor; the sinc directivity is 1 at nadir), < 0.4 dB down to -35 dB
    (measured 0.23 dB near sidelobe minima -- the quadrature p'/(2*pi*f0)
    term of the in-band amplitude slope, |p'|/(2*pi*f0) ~ 0.3*B/f0)."""
    frac = 0.5
    pos, centers, normals, areas, e1, e2 = _point_geom(frac)
    kw = dict(gamma=GAMMA, f0=F0, bandwidth=BW, c=C, twtt=TWTT, n_freq=96)
    y_full = multifreq_profile(pos, centers, normals, areas, e1, e2, **kw)
    y_froz = multifreq_profile(pos, centers, normals, areas, e1, e2,
                               freeze_amplitudes=True, **kw)
    pk = np.abs(y_full).max()
    diff_pk = 20.0 * np.log10(np.abs(y_froz).max() / pk)
    assert abs(diff_pk) < 0.01
    m = np.abs(y_full) > pk * 10 ** (-35 / 20.0)
    diff = 20.0 * np.log10(np.abs(y_froz[m]) / np.abs(y_full[m]))
    assert np.abs(diff).max() < 0.4


def test_referee_alias_guard():
    pos, centers, normals, areas, e1, e2 = _point_geom(0.0)
    import pytest
    with pytest.raises(ValueError, match="alias"):
        multifreq_profile(pos, centers, normals, areas, e1, e2, gamma=GAMMA,
                          f0=F0, bandwidth=BW, c=C, twtt=TWTT, n_freq=8)
