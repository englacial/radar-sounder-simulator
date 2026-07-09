# Waveform, antenna patterns, and post-processing

Stage 4 adds the instrument and processing layers on top of the field simulation. All three are **default-off**: a config with `waveform.kind="delta"`, `antenna.kind="isotropic"`, and no processing reproduces earlier behavior bit-for-bit (regression-gated).

## Waveform (linear chirp)

`RadarConfig.waveform`: `kind` (`delta`/`chirp`), `bandwidth`, `pulse_length`, `window` (`hann` default, `hamming`, `none`), `interp_bins`. The kernel's binned trace carries exact per-facet carrier phase, so pulse compression is applied as **post-convolution** with the windowed-autocorrelation compressed-pulse kernel — validated against a multi-frequency synthesis referee (kernel run across the band + IFFT) to ~1 dB on flat and rough scenes. Point-target response matches textbook resolution and sidelobe levels for each window (Harris 1978 values, ±1 dB). Incoherent mode is untouched unless `incoherent_envelope=true` (simc parity preserved).

**Sampling hygiene (important):** the delta-response envelope is quantized to `dt`, which manufactures artifacts at the *aliased carrier* `f_a = f0 − round(f0·dt)/dt`. Choose `dt` so `|f_a| > bandwidth/2` (an in-band alias triggers a warning from `simulate()`); `interp_bins=true` is the in-band fallback (single-interface kernels only). Keep facets small enough that `k·L·sinθ_max < π` or in-band facet-directivity variation contaminates off-nadir sidelobe predictions. The previously documented delta-pulse off-nadir pedestal on smooth surfaces is a manifestation of this quantization; the `waveform_pedestal` report case shows the chirped result matching the exact referee ~60 dB below the worst-case delta pedestal.

## Antenna patterns

`RadarConfig.antenna`: `isotropic` (default), `dipole` (axis configurable), `array` (uniform cross-track line array, `n_elements`, `spacing_lam`, nadir boresight), or `tabulated` (1-D g(θ), interpolated). `g` is one-way **field** gain: coherent kernels weight fields by g², the incoherent kernel weights power by g⁴ (consistent by construction; cross-kernel ensemble test covers a steep pattern). `roll_source="nav"` rolls the pattern about the track using the frame's Roll. Verified against closed forms (dipole nulls/values exact; array factor main lobe and null positions). The `antenna_patterns` report case quantifies clutter suppression: a 5-element cross-track array removes 8–9 dB of off-nadir clutter where an along-track dipole manages 2–4 dB.

## Post-processing (`soundersim.processing`)

Layer-3 functions on output Datasets (never inside kernels), each appending a step descriptor to the `processing` attr:

- `presum(ds, n)` — non-overlapping coherent along-track summation (decimates); complex fields only.
- `unfocused_sar(ds, aperture)` — sliding coherent sum (preserves sampling).
- `focused_sar(ds, aperture)` — straight-track time-domain backprojection through air, surface-referenced, validation-grade (no motion compensation; in-ice focusing is future work). Point-target azimuth resolution verified to λr/(2L) within a few %.
- `multilook(ds, n)` — incoherent power averaging (speckle contrast ~1/√n, verified).

A guard warns when trace spacing exceeds λ/4 for the requested aperture (Doppler aliasing). Simulated trace spacing must support the processing you request — subsampled real-frame runs generally cannot be coherently summed; simulate a dense sub-segment for that.

## Real-frame comparisons at processing level

The xOPR coherent+bed cases run with the MCoRDS parameters extracted from the frames' own product files (180–210 MHz chirp, hann compression, 7-element cross-track array, roll from nav; provenance cached in `outputs/cache/mcords_2017P3_params.json`, with unmodeled items — tx taper, rx element subsets, f-k migration — recorded). Simulation runs on an alias-free dt/4 grid decimated exactly onto the frame's time axis. The full-frame comparison applies no simulated along-track processing (subsampled traces are Doppler-aliased; a 5-look multilook provides the speckle analog), while a densified 220-trace sub-segment at λ/4 spacing demonstrates properly-sampled unfocused SAR (~30 dB surface gain). At this processing level the measured-vs-simulated clutter-to-surface ratios agree to ~2–8 dB — the residual now reflects real modeling gaps (sub-facet roughness statistics, exact CReSIS processing chain) rather than idealization artifacts.
