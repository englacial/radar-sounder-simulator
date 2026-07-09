"""Pulse-compressed waveform kernel + fast-time convolution (stage 4, M20).

Model: the simulation kernels produce a delta-response trace whose bins carry
the EXACT per-facet carrier phase exp(-2j*k*r) (float64 path, see
kernels/coherent.py) with only the envelope delay quantized to dt. Pulse
compression of a linear-FM chirp is therefore a post-kernel processing step:
a complex convolution of ``field`` along fast time with the compressed-pulse
kernel p(tau), after the kernel and before build_dataset (power is recomputed
as |field|^2 from the convolved field).

Compressed-pulse construction (documented choice: ANALYTIC windowed sinc).
For a large time-bandwidth-product LFM, the matched-filter output with a
raised-cosine amplitude weighting applied once across the compressed band --

    W(f) = a + (1 - a) * cos(2*pi*(f - f0)/B),   |f - f0| <= B/2
    a = 1 (none) / 0.5 (hann) / 0.54 (hamming)

(the standard weighting-on-receive convention, e.g. CReSIS pulse compression)
-- is, by the stationary-phase approximation, the inverse Fourier transform
of W, which for raised-cosine windows is closed-form:

    p(tau) = [a*sinc(B*tau) + (1-a)/2 * (sinc(B*tau - 1) + sinc(B*tau + 1))] / a

with sinc the normalized sinc. Harris (1978) window figures then apply
directly to the compressed pulse: peak power sidelobe -13.3 dB (none) /
-31.5 dB (hann) / -42.7 dB (hamming); -3 dB main-lobe width 0.886 / 1.44 /
1.30 x (1/B), i.e. range resolution = broadening * c/(2B). The alternative
FFT-of-windowed-chirp-spectrum construction differs only by O(1/sqrt(B*T))
Fresnel ripples; the analytic form is exact in the B*T -> inf limit and is
what the M21 multi-frequency referee independently checks.

Baseband convention: the trace's phase is exp(-2j*k*r) at f0, so p is the
BASEBAND compressed response. A symmetric window centered on f0 makes p REAL
(zero residual phase): the carrier phase in the trace is untouched, and the
peak phase of a compressed point target equals its delta-mode phase.

Normalization: peak-normalized, p(0) = 1, so delta-mode and chirp-mode
surface peaks are directly comparable and the Haynes absolute checks stay
meaningful in delta mode. A chirped absolute check is derivable: the chirped
peak of an isolated point target equals the delta closed form exactly (by
p(0) = 1); extended/overlapping responses integrate the closed form against
p. Energy is NOT preserved by design (windowing trades main-lobe width for
sidelobes): for an isolated point target the output/input energy ratio is
exactly sum_m |p(m*dt)|^2 (tested).

Support: the true matched-filter output vanishes beyond +-T (pulse_length);
p is truncated there. The analytic tails at |tau| ~ T carry O(1/(pi*B*T))
amplitude, negligible for B*T >~ 10 (a warning fires below that, where the
stationary-phase form itself degrades).

``dropped_power`` is a kernel-level (pre-convolution) diagnostic and is not
convolved.
"""

import warnings

import numpy as np

_WINDOW_A = {"none": 1.0, "hann": 0.5, "hamming": 0.54}


def compressed_pulse(bandwidth, pulse_length, dt, window="hann"):
    """Sampled compressed-pulse kernel p(m*dt), m in [-M, M].

    Returns ``(p, M)``: p float64, odd length 2M+1, peak p[M] = 1 at lag 0;
    M = ceil(pulse_length/dt) (the physical +-T support of the matched-filter
    output). See the module docstring for the construction and normalization.
    """
    a = _WINDOW_A[window]
    tb = bandwidth * pulse_length
    if tb < 10.0:
        warnings.warn(
            f"chirp time-bandwidth product {tb:.1f} < 10: the stationary-phase "
            "compressed-pulse model is inaccurate for such short pulses")
    m = int(np.ceil(pulse_length / dt))
    x = bandwidth * dt * np.arange(-m, m + 1)
    p = (a * np.sinc(x) + 0.5 * (1.0 - a) * (np.sinc(x - 1.0)
                                             + np.sinc(x + 1.0))) / a
    return p, m


def convolve_fast_time(arr, kernel, center):
    """'Same'-mode convolution along axis 1 with the kernel peak at lag 0.

    ``arr`` is (traces, n_samples, ...); ``kernel`` 1-D with its zero-lag
    sample at index ``center``. FFT-based, computed in complex128; the caller
    casts back. Returns the same shape as ``arr``.
    """
    arr = np.asarray(arr)
    n, m = arr.shape[1], len(kernel)
    nfft = 1 << (n + m - 2).bit_length()
    kf = np.fft.fft(kernel, nfft).reshape((1, nfft) + (1,) * (arr.ndim - 2))
    y = np.fft.ifft(np.fft.fft(arr, nfft, axis=1) * kf, axis=1)
    return y[:, center:center + n]


def apply_waveform(out, radar, mode):
    """Apply the configured waveform to a kernel output (field or power).

    Coherent mode: complex convolution of the field with p (chirp kind).
    Incoherent mode: |p|^2 power-envelope convolution, only when
    ``waveform.incoherent_envelope`` is set (default OFF -- simc parity).
    Delta kind returns ``out`` unchanged (identity: bit-compatible default).
    """
    wf = radar.waveform
    if wf.kind == "delta":
        return out
    p, m = compressed_pulse(wf.bandwidth, wf.pulse_length, radar.dt, wf.window)
    if mode == "coherent":
        y = convolve_fast_time(np.asarray(out, np.complex128), p, m)
        return y.astype(np.complex64)
    if not wf.incoherent_envelope:
        return out
    y = convolve_fast_time(np.asarray(out, np.float64), p * p, m).real
    return np.maximum(y, 0.0).astype(np.float32)
