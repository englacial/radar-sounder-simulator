# Waveform, antenna patterns, and post-processing

Stage 4 adds the instrument and processing layers on top of the field simulation. All three are **default-off**: a config with `waveform.kind="delta"`, `antenna.kind="isotropic"`, and no processing reproduces earlier behavior bit-for-bit (regression-gated).

## Waveform (linear chirp)

`RadarConfig.waveform`: `kind` (`delta`/`chirp`), `bandwidth`, `pulse_length`, `window` (`hann` default, `hamming`, `none`), `interp_bins`. The kernel's binned trace carries exact per-facet carrier phase, so pulse compression is applied as **post-convolution** with the windowed-autocorrelation compressed-pulse kernel — validated against a multi-frequency synthesis referee (kernel run across the band + IFFT) to ~1 dB on flat and rough scenes. Point-target response matches textbook resolution and sidelobe levels for each window (Harris 1978 values, ±1 dB). Incoherent mode is untouched unless `incoherent_envelope=true` (simc parity preserved).

**Two compressed-pulse constructions** (`waveform.construction`; instrument YAML `simulated.construction`):

- `analytic` (default) — the stationary-phase windowed sinc `p(τ) = [a·sinc(Bτ) + (1−a)/2·(sinc(Bτ−1) + sinc(Bτ+1))]/a`, the B·T → ∞ limit. Its tails fall like 1/(πBτ), so it has essentially **no pulse-length dependence** beyond the ±T truncation: at B = 15 MHz / Hann the response 10 µs behind the peak is ~−140 dB whether the pulse is 5 or 20 µs long.
- `chirp` — the explicit LFM `exp(jπ(B/T)t²)` matched-filtered against its raised-cosine-weighted conjugate replica (weighting on receive, the convention of the mission design tool's `build_sidelobes.py`), sampled on the simulation `dt` lattice. This keeps the finite-TB **Fresnel-ripple sidelobe pedestal** (~−53 dB at 8–12 µs for B = 15 MHz, T = 20 µs, Hann; TB = 300) and the exact ±T support (a 5 µs pulse has zero response past 5 µs). Mainlobe and near sidelobes agree with the analytic form to the O(1/√(B·T)) ripple level (5e-4 at TB = 3000). Use it whenever the surface return's far sidelobes matter at the bed delay — thin ice under a long pulse. The kernel is complex with a small imaginary part (symmetric window); the peak is normalised to exactly 1.

In the clutter runner the convolution runs inside the cached chunk, so a `chirp` instrument forks the chunk cache key (`waveform: {construction, pulse_length_us}` in `meta_key`, `_wchirp` file suffix); analytic keys are untouched.

**Sampling hygiene (important):** the delta-response envelope is quantized to `dt`, which manufactures artifacts at the *aliased carrier* `f_a = f0 − round(f0·dt)/dt`. Choose `dt` so `|f_a| > bandwidth/2` (an in-band alias triggers a warning from `simulate()`); `interp_bins=true` is the in-band fallback (single-interface kernels only). Keep facets small enough that `k·L·sinθ_max < π` or in-band facet-directivity variation contaminates off-nadir sidelobe predictions. The previously documented delta-pulse off-nadir pedestal on smooth surfaces is a manifestation of this quantization; the `waveform_pedestal` report case shows the chirped result matching the exact referee ~60 dB below the worst-case delta pedestal.

## Antenna patterns

`RadarConfig.antenna`: `isotropic` (default), `dipole` (axis configurable), `array` (uniform cross-track line array, `n_elements`, `spacing_lam`, nadir boresight), or `tabulated` (1-D g(θ), interpolated). `g` is one-way **field** gain: coherent kernels weight fields by g², the incoherent kernel weights power by g⁴ (consistent by construction; cross-kernel ensemble test covers a steep pattern). Arrays are peak-normalized by default. Setting `element_directivity_db` to an element's peak power directivity `D` in dBi adds absolute gain and a forward-hemisphere cosine-power element pattern: `P(θ)/P(0) = cos(θ)^q`, `q = 10^(D/10)/2 - 1` (3 dBi is treated as the rounded uniform-hemisphere limit). The total angular field pattern is the element factor times the array factor, and the peak gain is set by integrating that pattern over the sphere per side (TX weights and RX weights separately), so `g²` at the peak is `sqrt(D_tx · D_rx)`. For 3 dBi elements at multiples of half-wavelength spacing this reduces to `D = 2 N_eff` per side; for directive elements it is below the product of element and array directivities because both narrow the beam in the cross-track plane (about 1 dB one-way for a 20-element λ/2 array of 6 dBi elements). `roll_source="nav"` rolls both the array axis and boresight about the track using the frame's Roll. Verified against closed forms (dipole nulls/values exact; array factor main lobe and null positions). The `antenna_patterns` report case quantifies clutter suppression: a 5-element cross-track array removes 8–9 dB of off-nadir clutter where an along-track dipole manages 2–4 dB.

## Post-processing (`soundersim.processing`)

Layer-3 functions on output Datasets (never inside kernels), each appending a step descriptor to the `processing` attr:

- `presum(ds, n)` — non-overlapping coherent along-track summation (decimates); complex fields only.
- `unfocused_sar(ds, aperture)` — sliding coherent sum (preserves sampling).
- `focused_sar(ds, aperture)` — straight-track time-domain backprojection through air, surface-referenced, validation-grade (no motion compensation; in-ice focusing is future work). Point-target azimuth resolution verified to λr/(2L) within a few %.
- `multilook(ds, n)` — incoherent power averaging (speckle contrast ~1/√n, verified).

A guard warns when trace spacing exceeds λ/4 for the requested aperture (Doppler aliasing). Simulated trace spacing must support the processing you request — subsampled real-frame runs generally cannot be coherently summed; simulate a dense sub-segment for that.

The clutter-study runner can refine slow time with `processing.posting_div`
while choosing the focusing extent independently: `first_fresnel` is a fast
screening aperture, and `product_resolution` keeps the original physical
aperture after refinement. This separation matters because spacing controls
the unaliased look angle, whereas aperture length controls azimuth resolution
and backprojection cost. A facet model adds a third constraint: a 32 m
facet's coherent lobe is only ~λ/L wide (2.8 deg at 195 MHz), so a Doppler
band narrower than the terrain's along-track tilts gates whole cross-track
rows of facets and stripes the image while discarding real clutter.
`fixed_angle` focuses with a wide band (`focus_half_angle_deg`, 5 deg in the
shipped experiments) and then multilooks the power to the
`product_resolution` azimuth resolution, keeping the measured-vs-simulated
comparison at matched resolution.

## Real-frame comparisons at processing level

The xOPR coherent+bed cases run with the MCoRDS parameters extracted from the frames' own product files (180–210 MHz chirp, hann compression, 7-element cross-track array, roll from nav; provenance cached in `outputs/cache/mcords_2017P3_params.json`, with unmodeled items — tx taper, rx element subsets, f-k migration — recorded). Simulation runs on an alias-free dt/4 grid decimated exactly onto the frame's time axis. The full-frame comparison applies no simulated along-track processing (subsampled traces are Doppler-aliased; a 5-look multilook provides the speckle analog), while a densified 220-trace sub-segment at λ/4 spacing demonstrates properly-sampled unfocused SAR (~30 dB surface gain). At this processing level the measured-vs-simulated clutter-to-surface ratios agree to ~2–8 dB — the residual now reflects real modeling gaps (sub-facet roughness statistics, exact CReSIS processing chain) rather than idealization artifacts.
