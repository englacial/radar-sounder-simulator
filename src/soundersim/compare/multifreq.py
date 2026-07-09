"""Multi-frequency synthesis referee for the chirped-convolution model (M21).

Exact (per-facet, float64, tiny scenes only) reference range-profile for the
stage-4 post-convolution waveform model: evaluate the scene's LPA frequency
response

    H(f_k) = sum_i contrib_i(f_k)

at K frequencies spanning the chirp band (kernels/coherent.lpa_contributions
with xp=np -- the k/2pi amplitude prefactor, the sinc facet directivity AND
the exp(-2jkr) phase all vary with k), weight by the same raised-cosine
window the pulse compressor applies (waveform.py convention), and synthesize
the compressed time-domain profile directly at the requested twtt samples:

    y(t) = sum_k Wbar_k * H(f_k) * exp(+2j*pi*(f_k - f0)*t),  Wbar = W/sum(W)

The exp(-2j*pi*f0*t) demodulation puts y in the kernels' baseband-at-carrier
convention (an isolated target at delay tau gives y(t) = A * exp(-2j*pi*f0*
tau) * p(t - tau): constant carrier phase, real envelope), so complex
comparison against the convolved trace is meaningful, not just |y|.

This is a direct Fourier synthesis (no IFFT grid, no binning): y is exact at
the sample times up to the band DISCRETIZATION, which makes y periodic with
period K/B -- contributions alias when (scene delay spread united with the
twtt window) exceeds K/B. ``multifreq_profile`` checks that span and raises.

Normalization matches waveform.compressed_pulse (peak-normalized): an
isolated point target gives |y(tau)| = |band-mean amplitude| while the
convolution path gives |A(f0) * p(0)| = |A(f0)| -- identical to first order
(midpoint band sampling cancels the linear-in-k amplitude term).

``freeze_amplitudes=True`` evaluates every facet amplitude at f0 while
keeping the exact per-frequency phases -- exactly what the post-convolution
model assumes (one delta trace at f0 convolved with a scalar pulse). The
full-vs-frozen difference measures the neglected in-band facet-directivity
variation: THE plan D4-1 decision-gate number.
"""

import numpy as np

from ..kernels.coherent import lpa_contributions
from ..waveform import _WINDOW_A


def band_frequencies(f0, bandwidth, n_freq):
    """Midpoint sampling of [f0 - B/2, f0 + B/2] (K points, no zero-weight
    endpoints; the midpoint rule cancels linear-in-f terms)."""
    return f0 - bandwidth / 2.0 + (np.arange(n_freq) + 0.5) * (bandwidth
                                                               / n_freq)


def window_weights(freqs, f0, bandwidth, window):
    """Raised-cosine compression weights W(f) (waveform.py convention)."""
    a = _WINDOW_A[window]
    return a + (1.0 - a) * np.cos(2.0 * np.pi * (freqs - f0) / bandwidth)


def multifreq_profile(position, centers, normals, areas, e1, e2, *, gamma,
                      f0, bandwidth, c, twtt, window="hann", n_freq=128,
                      freeze_amplitudes=False):
    """Compressed range profile y(twtt) (complex128) for one platform position.

    All facet arrays float64 (N, 3)/(N,); ``twtt`` the fast-time sample times
    (s). K*N per-facet evaluations -- tiny scenes only (module docstring).
    Raises when the K-point band sampling would alias (span > K/B).
    """
    position = np.asarray(position, np.float64)
    twtt = np.asarray(twtt, np.float64)
    r = np.linalg.norm(position - np.asarray(centers, np.float64), axis=-1)
    tau = 2.0 * r / c
    span = max(twtt.max(), tau.max()) - min(twtt.min(), tau.min())
    if span > n_freq / bandwidth:
        raise ValueError(
            f"multifreq referee would alias: delay span {span * 1e6:.2f} us "
            f"> K/B = {n_freq / bandwidth * 1e6:.2f} us; raise n_freq or "
            "shrink the scene/window")

    freqs = band_frequencies(f0, bandwidth, n_freq)
    w = window_weights(freqs, f0, bandwidth, window)
    wbar = w / w.sum()

    if freeze_amplitudes:
        k0 = 2.0 * np.pi * f0 / c
        contrib0, _ = lpa_contributions(position, centers, normals, areas,
                                        e1, e2, k0, gamma, xp=np)
        amp0 = contrib0 * np.exp(2j * k0 * r)  # strip the f0 phase

    H = np.empty(n_freq, np.complex128)
    for j, f in enumerate(freqs):
        k = 2.0 * np.pi * f / c
        if freeze_amplitudes:
            contrib = amp0 * np.exp(-2j * k * r)
        else:
            contrib, _ = lpa_contributions(position, centers, normals, areas,
                                           e1, e2, k, gamma, xp=np)
        H[j] = contrib.sum()

    return (wbar * H) @ np.exp(2j * np.pi * np.outer(freqs - f0, twtt))
