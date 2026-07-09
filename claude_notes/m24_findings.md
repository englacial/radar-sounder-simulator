# M24 findings: xOPR coherent_bed cases at MCoRDS-like processing level (2026-07-08)

## Parameter sourcing (provenance JSON: outputs/cache/mcords_2017P3_params.json)

Authoritative source: the frames' OWN param structs (param_records /
param_csarp / param_combine) inside the CSARP_standard .mat files, downloaded
and cached (outputs/cache/Data_*_source.mat), plus the CReSIS rds readme
(outputs/cache/cresis_rds_readme.pdf/.txt).

| parameter | value | source |
|---|---|---|
| chirp band | 180-210 MHz (f0 195, B 30 MHz) | param_records.radar.wfs.f0/f1 (both frames) |
| pulse lengths | ANT 20171121_03_005: {1, 3} us; GL 20170422_01_014: {1, 3, 10} us | param_records.radar.wfs.Tpd |
| waveform combine | img_comb: 1 us wf near surface, longest wf for bed (stitch Tpd after surface) | param_combine.combine.img_comb |
| tx window | 20% Tukey (time domain) | wfs.tukey = 0.2 |
| compression window | hanning (frequency domain) | param_csarp.csarp.ft_wind |
| PRF | 12 kHz | param_records.radar.prf |
| hardware presums | NOT in the product file (raw-record level) -- irrelevant: sim starts from CSARP trace positions. ASSUMED absorbed | -- |
| tx antenna | 7-element P-3 center cross-track array, d = 0.5 lambda_c; amplitude weights recorded (wfs.tx_weights) | readme platform/beamwidth table + wfs |
| rx combine | GL: ADCs [7,8,9,11,12] -> center elements [2,3,4,6,7] (element 5 excluded); ANT: ADCs [6..12] -> all 7 center elements; hanning array window | param_combine.combine.imgs + wfs.rx_paths |
| CSARP_standard processing | motion-compensated f-k SAR per channel (sigma_x 2.5 m SLC), delay-and-sum combine, rline_rng [-5..5] = 11 looks, dline 6 -> ~15 m posting (~25 m along-track res) | param_csarp/param_combine + readme |

Modeled as: chirp B=30 MHz + hann compression (Tukey-on-tx unmodeled,
second-order shape effect), pulse = the frame's longest/bed waveform (3 us
ANT / 10 us GL; the compressed windowed-sinc shape is Tpd-independent except
truncation), uniform unsteered 7-element 0.5-lambda array both ways
(g^2 = AF_7^2), roll_source="nav". Recorded-but-unmodeled: tx taper, rx
channel subsets, hanning array window, waveform-playlist gain stitching,
motion comp, focused SAR.

## Alias-dt decision (M21 caveat) -- option (a), measured

The brief assumed dt = 5 ns; the frames' ACTUAL grid is dt = 33.333 ns
(30 MHz), which puts f0*dt = 6.5 -> the envelope-quantization alias at
|f_a| = 15 MHz = B/2 EXACTLY (the hann band edge). Fragile boundary: the
float dt from the twtt axis rounds 6.5000000000000005 -> 7, f_a =
14.99999... MHz, and simulate()'s strict-< in-band warning FIRES.

Measurement (claude_notes/m24_alias_probe.py, Greenland, 25 traces / 1200 m /
64 m facets): quiet-band floor (surface+1.5..3.5 us, clipped above bed) rel
surface peak -- native dt -46.9 dB median (p90 -42.1) vs dt/4 -47.1 dB median
(p90 -45.4). The hann edge nearly nulls the alias (median +0.2 dB only), but
the tail is contaminated (+3.3 dB at p90) and the config would ship with a
standing warning.

DECISION: simulate at dt_sim = dt/4 = 8.333 ns (f0*dt = 1.625 -> f_a = 45 MHz
= 3B/2, warning SILENT), n_samples = 4*(n-1)+1, then decimate [::4] -- t0 = 0
so every 4th simulation bin IS a frame bin (no interpolation; decimating the
critically-sampled complex field is exact). Kernel cost is facet-bound, so
the finer grid is ~free. interp_bins stays off; NO multilayer kernel port
(option (b) avoided).

## Trace spacing / along-track processing decision (hybrid)

Measured product = f-k SAR + 11 hanning looks at 15 m posting. Sim full frame
subsamples 100 traces (~260-530 m spacing >> lambda/4 = 0.384 m): coherent
along-track summation of those traces is Doppler-aliased garbage
(processing.py guard would warn), so the full frame is compared at
chirp+antenna PER-TRACE level, with a 5-look incoherent multilook
(surface-following band) as the speckle-statistics analog (explicitly NOT
resolution-matched; recorded). PLUS a DENSE sub-segment: nav interpolated to
0.35 m spacing (< lambda/4 -> Doppler-unaliased for ALL scattering angles),
220 traces / 77 m of track at the frame center, DEM/bed CROPPED from the
cached full-frame windows (no new network), unfocused_sar with a 20 m
aperture (the unfocused limit sqrt(lambda*r/2) at ~500 m AGL). Surface peak
gain recorded against the coherent (20 log n) and incoherent (10 log n)
references. Note: the recorded gain CAN exceed 20 log n because raw
per-trace peaks are speckled and the ratio uses medians.

## Implementation notes

- tools/run_opr_coherent_bed.py upgraded in place; run_case gained
  dense_traces/dense_spacing/dense_ct/aperture_m kwargs; four-case xOPR
  structure and both original gates unchanged (chirped leading edge: hann
  main lobe 1.44/B ~ 1.4 frame bins, symmetric kernel -> constant shift
  absorbed by the offset removal).
- MultilayerScene has no nav_roll field; the tool attaches it as an attribute
  post-construction (plain dataclass), which _antenna_pattern picks up via
  getattr -- no src change.
- multilook() drops `field` but keeps mode="coherent" attrs -> combine()
  would raise; the multilooked figure panel uses ds_ml.power.sum("layer").
- New recorded metrics: alias_free_dt (incl. alias_warning_fired, asserted
  False in the integration test), clutter_to_surface_db (near 1-2.5 us / mid
  2.5-5 us bands after the surface pick, clipped 1 us above the bed pick;
  sim bands evaluated against offset-shifted picks), speckle_contrast_multilooked
  (+ measured_frame_contrast via the same estimator), unfocused_surface_gain_db
  (+ doppler_guard_warned, asserted False).
- run_opr_comparison.py (incoherent + simc parity cases) untouched.

## Results (full runs, from cache) -- see metrics.json per case

| metric | ANT 20171121_03_005 | GL 20170422_01_014 |
|---|---|---|
| wall time (main + dense) | 701.8 + 13.2 s | 243.7 + 12.8 s |
| facets (11.3 / 12.7 m spacing) | 16.6 M | 5.7 M |
| surface_leading_edge (gate <= 5) | 1.2 bins PASS | 0.7 bins PASS |
| bed_alignment (floor-aware gate) | 63.3 vs thr 70.8 PASS (floor 65.8) | 6.1 vs thr 11.6 PASS (floor 6.6) |
| alias warning fired | False | False |
| C/S near band (meas / sim / diff) | -54.4 / -50.2 / +4.2 dB | -47.7 / -50.1 / -2.4 dB |
| C/S mid band (meas / sim / diff) | -67.4 / -59.1 / +8.3 dB | -60.1 / -64.9 / -4.8 dB |
| speckle contrast (per-trace -> 5-look) | 1.11 -> 1.05 | 0.97 -> 0.82 |
| measured-frame contrast (same estimator) | 1.55 (terrain-dominated) | 0.55 |
| unfocused surface gain (n_win) | 30.3 dB (57; coh ref 35.1, incoh 17.6) | 32.1 dB (58; coh ref 35.3, incoh 17.6) |
| bed/surface energy ratio | -12.3 dB | -36.5 dB |

Reading: the chirped, antenna-weighted sim's clutter-to-surface dynamic range
is now within ~2-8 dB of the measured product in the between-surface-and-bed
bands (vs the delta-pulse product where the comparison was not dynamic-range
meaningful at all); residual diffs carry the unmodeled physics (volume
scatter/internal layers in the measured data, facet-scale roughness
statistics, attenuation constant). The 5-look contrast sits above the
1/sqrt(5) theory because the ~0.5 km-apart looks average non-stationary
terrain (recorded caveat); the unfocused surface gain lands between the
incoherent and full-coherent references (partial coherence of the DEM-scale
rough surface), with the Doppler guard silent at 0.35 m spacing.
