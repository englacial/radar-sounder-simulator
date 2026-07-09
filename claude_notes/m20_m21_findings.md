# M20 + M21 findings: chirp convolution, interp_bins, pedestal retirement (2026-07-08)

Implementation: `WaveformConfig` (config.py, on RadarConfig), `waveform.py`
(analytic windowed-sinc compressed pulse + FFT fast-time convolution),
`interp_bins` in kernels/coherent.py (static flag in the jit-factory cache
key; False path traces the exact pre-stage-4 program), `compare/multifreq.py`
(exact per-facet multi-frequency referee, direct Fourier synthesis in the
baseband-at-carrier convention). Regression: all 12 arrays of
`jit_regression_check.py` bit-identical vs `m20_baseline_before.npz`.

## Point target (CI, tests/test_waveform.py, test_multifreq.py)

- Compressed pulse vs Harris 1978: -3 dB width x 1/B measured/textbook =
  0.886/0.886 (none), 1.44/1.44 (hann), 1.30/1.30 (hamming) within 3%;
  PSL within 1 dB of -13.3 / -31.5 / -42.7 dB.
- Through the pipeline (dt-sampled): hann width 1.44/B +-5%, PSL -31.5 +-1.5.
- Peak phase == delta-mode carrier phase (real symmetric kernel), peak
  magnitude == delta magnitude (p(0)=1), energy ratio == sum|p|^2 exactly.
- interp_bins: envelope peak position error 0.47 bin -> <0.1 bin at frac 0.47.
- Convolution vs referee: 0.03 dB main lobe, <=0.4 dB at sidelobe crests,
  max |complex diff| 1.8% of peak (the second-order residual of sub-bin
  linear splitting, on the steepest flank).

## The pedestal, mechanism (M21 measurement, flat scene 195 MHz / 500 m AGL)

The trace carries exact carrier phase but dt-quantized envelope delay ->
quantization noise at the aliased carrier f_a = f0 - round(f0*dt)/dt.

- dt = 5 ns (firn-study value): f_a = -5 MHz, INSIDE the +-15 MHz band.
  Delta pedestal max -17.7 dB rel surface peak over 5-40 m apparent depth
  (the firn finding's -18.5 dB shoulder). Chirped WITHOUT interp_bins: no
  better (the pulse passes the alias). WITH interp_bins: ~16 dB lower in the
  alias-dominated 10-20 m band. NOTE: k*dbin = 0.975*pi at these parameters
  -- close to a NULL of the 2cos^2(theta)|sin(k dbin)| artifact; a generic
  dt makes the delta pedestal far worse (dt = 4 ns: |sin| = 0.64, pedestal
  median ~61 dB above the physical floor).
- dt = 4 ns: f_a = -55 MHz, out of band -> the compressed pulse REJECTS the
  quantization noise regardless of binning. Rule: choose dt so
  |f0 - round(f0*dt)/dt| > B/2. simulate() now warns when a chirped
  coherent run violates this and interp_bins is off.

## Frozen-directivity error (the D4-1 decision-gate number)

Full referee (per-facet amplitudes at every f_k) vs frozen (amplitudes at
f0, exact phases -- the convolution's implicit model):

- 4 m facets: error concentrated at facet sinc-null rings (k0*L*sin(theta)
  = n*pi at ~5.3 / 23.4 / 62.9 m apparent depth): frozen/convolved floor
  features at -23 dB rel peak where the exact referee is -43 dB; median
  (frozen - full) over 5-80 m = 17.9 dB. Independent of dt and of interp.
- 1 m facets (no sinc null in the scene's angular span): median |full -
  frozen| = 0.19 dB. In-band k/2pi-prefactor effect <=0.4 dB near sidelobe
  minima, <0.01 dB at peaks (midpoint band sampling cancels the linear term).
- f32 kernel noise ruled out: f64 numpy rebinning of the same facets gives
  identical profiles.

So the "in-band directivity variation" is a FACET-SIZING cost, not a
convolution defect: keep k0*L*sin(theta_max) < pi (L < ~1.3 m at 195 MHz
for theta up to 35 deg) when the off-nadir floor matters -- the same
direction the LPA Fresnel check pushes.

## End-to-end (report case waveform_pedestal, integration)

Well-sampled config (dt = 4 ns, 1 m facets): chirped convolution matches
the exact multi-frequency referee to 1.06 dB median (p90 ~3 dB, on a floor
60+ dB down); delta pedestal excess over referee 61.3 dB median; chirped
suppression of the delta pedestal 59.5 dB median. Gently rough surface
(0.3 m / 150 m sinusoid): 0.66 dB median vs referee. Firn-parameter config
(dt = 5 ns, 4 m facets): delta shoulder -17.7 dB; chirp+interp suppression
4.2 dB median over 5-40 m (floored by the -23 dB directivity rings).

**D4-1 verdict: post-convolution stays primary.** Its two error terms are
both controllable by simulation hygiene (alias-free dt; facet sizing), and
at recommended parameters it matches the exact synthesis to ~1 dB on the
hardest (smooth, cancellation-dominated) scene. Multi-frequency synthesis
remains the referee. Caveat to carry into M24: the OPR frames use dt = 5 ns
at 195 MHz (in-band alias) and coarse facets -- chirped OPR reruns should
use interp_bins=true and record the residual floor, or resample dt.

## interp_bins default recommendation

Kept default False (delta bit-compatibility; the flag also affects delta
runs if set). For chirped coherent runs: set true whenever the alias is in
band; simulate() warns about exactly that configuration. Not supported for
multilayer runs (refracted kernel unchanged); a chirped multilayer run at
dt = 5 ns / 195 MHz will warn -- M24 should either resample dt (preferred,
works without kernel changes) or port the split binning to multilayer.py.

## Notes

- Compressed pulse: analytic FT of the raised-cosine weighting (weighting-
  on-receive convention, one window application), peak-normalized p(0)=1,
  truncated at +-pulse_length (physical matched-filter support); real kernel
  (symmetric window centered on f0) so the trace carrier phase is untouched.
- Referee cost: 128 frequencies x 360k facets x 2 variants ~ 10 s (numpy
  f64, one trace); aliasing guard at span > K/B.
- Dev probes: claude_notes/m21_pedestal_probe.py (band tables in session
  log); regression baseline claude_notes/m20_baseline_before.npz.
