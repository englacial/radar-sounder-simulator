# ATM L1B surface roughness on the study lines: form, scale dependence, along-line variability (plan steps 0-2)

Code: `claude_notes/atm_roughness/{atm_common,atm_pull,atm_roughness,atm_summarize}.py`
(`uv run`), logs in `claude_notes/atm_roughness/logs/`. Data in
`outputs/cache/atm/<line>/<date>/` (granules.json per day). Results in
`outputs/atm_roughness/<line>/`: `blocks_<date>_{1000,500}m.{json,csv}` (per-block
D(r), fits, octave RMS, anisotropy, grid PSD), `summary_<date>.json`,
`summary_years.json` (westcoast), figures `fig_a_octaves_*`, `fig_b_fits_*`,
`fig_c_family_*`, `fig_d_aniso_*`, `fig_years.png`; all tables in
`outputs/atm_roughness/summary_tables.md`. Nothing committed.

## 0. Data pulled (531 MB total; nothing near the 5 GB cap)

ILATM1B via `earthaccess` (netrc creds OK), bbox = line nav envelope +-2 km, temporal =
frame UTC span +-3 min (frame times from the cached OPR frames' `slow_time`). All granules
are HDF5 (v2 layout, incl. 2013/2014 ATM4); no Qfit binary needed. ILNSA1B (narrow swath)
exists only for westcoast 2019 (15 granules, 105 MB, downloaded, not used: 56 m swath).

| line | date | ATM | granules | MB | shots | notes |
|---|---|---|---|---|---|---|
| greenland_westcoast | 2016-05-11 | ATM5 (3 kHz) | 3 | 56 | 3.48 M | 186 m swath, 0.11 shots/m2 |
| greenland_westcoast | 2017-05-10 | ATM6 (9 kHz) | 10 | 115 | 8.50 M | 343 m swath, 0.22/m2 |
| greenland_westcoast | 2019-05-14 | ATM6 (10 kHz) | 15 | 115 (+105 NSA) | 8.82 M | 281 m swath, 0.23/m2 |
| greenland_geikie01_transit | 2014-04-21 | ATM4 (3 kHz) | 2 | 24 | 1.66 M | 0.08/m2 |
| greenland_geikie01_transit | 2017-04-24 | none | 0 | - | - | no ILATM1B/ILNSA1B/ILATM2 anywhere within +20 km that day (2.5 km AGL pass; ATM not ranging) |
| antarctica_getz | 2016-11-05 | ATM6 (DC-8, 2.9 kHz) | 4 | 62 | 3.83 M | low pass only |
| antarctica_getz | 2016-10-28, 10-31 | none | 0 | - | - | 9-10 km AGL passes, beyond ATM range (expected) |
| antarctica_david | 2013-11-19, 11-20 | ATM4 | 3 | 54 | 3.54 M | only OIB ATM over the David bbox in 2009-2019; tracks 4-40 km off the line (median 21 / 15 km); analysed on their own track axis |

Getz/geikie/westcoast ATM tracks sit 85-125 m (median) from the radar reference axis;
s is the radar anchor axis (arc length over the unsliced reference frames), so blocks
line up with the radargram s.

## Method (what differs from the plan, and why)

- Point density is 0.08-0.23 shots/m2, so a 1 m grid is 7-19 % filled and its FFT is
  dominated by the conical-scan mask. The primary estimator is therefore the point-pair
  structure function D(r) = <[h(x)-h(x+r)]^2>, quarter-octave lag bins 0.25-150 m, azimuth
  sectors all / along (+-22.5 deg) / cross, on the detrended cloud (5 m median grid,
  normalised Gaussian high-pass, half-power at 200 m, 93 % retained at 100 m; blunders
  > 6 MAD dropped; weakest 1 % rcv_sigstr and widest 0.5 % pulses dropped). The 1 m median
  grid + mask-corrected Welch PSD is kept as the cross-check (fig_b squares).
- Noise is not white across pair types. Shots within one scan rotation (dt < 30 ms) share
  the attitude solution; pairs from different rotations add a lag-independent
  scan-to-scan term. Both are calibrated per flight from the pairs themselves: crossover
  pairs (lag < 0.5 m, dt > 0.5 s) give the total, the same-vs-cross-scan excess at 2-10 m
  lag gives sigma_scan, the difference in quadrature gives the ranging noise sigma_r.
- Fits in log D(r) over lags 0.25-50 m (wavelength ~2r: 0.5-100 m), same-scan pairs as
  primary on the 9 kHz ATM6 flights (nugget fixed at sigma_r^2, 2 free params), cross-scan
  pairs with a free nugget on the 3 kHz flights (consecutive same-scan shots are 4-5 m
  apart there). Families: Gaussian, exponential, power law D = c r^2H (2-D PSD k^-(2H+2)).
  BIC per block; whiteness = Wald-Wolfowitz runs test on log residuals.
- Octave RMS (non-parametric, nugget cancels): sqrt((D(L2/2)-D(L1/2))/2) for wavelength
  octave [L1, L2]. The 1-2 m octave needs lags 0.5-1 m and is an upper bound only.

### Noise floor per flight (m)

| flight | ranging sigma_r | scan-to-scan | crossover total @0.27 m | class used |
|---|---|---|---|---|
| westcoast 2016 (ATM5) | 0.043 [0.040-0.046] | 0.031 | 0.053 | cross-scan |
| westcoast 2017 (ATM6) | 0.026 [0.018-0.031] | 0.041 | 0.048 | same-scan |
| westcoast 2019 (ATM6) | 0.016 [0.013-0.018] | 0.013 | 0.020 | same-scan |
| geikie 2014 (ATM4) | 0.032 [0.030-0.034] | 0.038 | 0.049 | cross-scan |
| getz 2016 (ATM6/DC-8) | 0.033 [0.023-0.038] | 0.046 | 0.056 | cross-scan |
| david 2013-11-19 (ATM4) | 0.059 | 0.036 | 0.069 | cross-scan |
| david 2013-11-20 (ATM4) | 0.071 | 0.041 | 0.082 | cross-scan |

sigma_r includes any sub-0.3 m roughness, so it is an upper bound on instrument noise.
No flat lake/sea-ice segment was isolated for a cross-check (david 11-20 is over sea ice
and gives the largest numbers, i.e. that surface is not flat at 0.3 m). The gridded PSD's
white floor (cross-scan noise x mask) is -24..-32 dB re m^4, above the surface PSD for
wavelengths < ~15 m on every flight: the grid path cannot see the Bragg band, the pair
path can.

## Question 1: which ACF family?

Per 1 km block, primary class, isotropic sector (fig_c strips):

| line | date | blocks | best by BIC: Gauss / exp / power law (fraction; in parentheses with dBIC > 2) | residual white (p > 0.05): G / E / PL | Gauss sigma cm p5/50/95 | Gauss l m p5/50/95 | H p5/50/95 | beta med |
|---|---|---|---|---|---|---|---|---|
| westcoast | 2016 | 131 | 0.00 / 0.02 / **0.98** (0.79) | 0.06 / 0.18 / 0.31 | 4.5/13.1/33.7 | 5.8/27.9/36.2 | 0.12/0.55/0.73 | 3.09 |
| westcoast | 2017 | 113 | 0.10 / 0.16 / **0.74** (0.67) | 0.00 / 0.03 / 0.05 | 5.0/8.5/16.0 | 1.9/4.1/25.6 | 0.19/0.36/0.71 | 2.72 |
| westcoast | 2019 | 119 | 0.03 / 0.35 / **0.62** (0.48) | 0.00 / 0.03 / 0.03 | 4.9/7.5/14.5 | 2.1/5.1/10.2 | 0.27/0.41/0.54 | 2.83 |
| geikie | 2014 | 76 | 0.04 / **0.83** (0.70) / 0.13 | 0.20 / 0.79 / 0.43 | 4.4/4.8/5.9 | 3.3/5.3/7.3 | 0.08/0.10/0.18 | 2.21 |
| getz | 2016 | 181 | 0.03 / 0.23 / **0.75** (0.55) | 0.12 / 0.37 / 0.42 | 4.0/9.5/25.5 | 4.4/18.7/37.3 | 0.08/0.36/0.66 | 2.71 |
| david 11-19 | 2013 | 95 | 0.05 / 0.24 / **0.71** (0.56) | 0.03 / 0.23 / 0.20 | 4.0/19.3/54.2 | 2.2/10.9/34.1 | 0.13/0.32/0.66 | 2.64 |
| david 11-20 | 2013 | 54 | 0.06 / 0.13 / **0.81** (0.70) | 0.06 / 0.19 / 0.28 | 10.7/51.8/108.3 | 5.7/17.5/35.9 | 0.28/0.55/0.73 | 3.10 |

Pooled whole-line fits: westcoast 2016/2017/2019 power law H = 0.61/0.37/0.44; getz
H = 0.37; david 11-19 H = 0.30; geikie exponential sigma = 4.9 cm, l = 5.3 m
(H = 0.11 if forced to a power law).

Reading:
- A Gaussian ACF is the worst of the three everywhere: best in 0-10 % of blocks, and its
  residuals are white in 0-20 % of blocks. Where it "wins" (a handful of 2017 blocks) it
  does so with l ~ 2 m and produces absurd Bragg-band extrapolations (-400 dB), which is
  why the p5 of some Bragg columns below is a nonsense number.
- Westcoast (percolation zone, 1300-1600 m), Getz (coastal, 10-400 m) and the David
  tracks are self-affine over 1-100 m: D(r) is a straight power law from the smallest
  resolved lag to 50 m with H = 0.35-0.45 (2-D beta = 2.7-2.9; 2016's higher H = 0.6 is
  the noisier cross-scan class with its nugget free, and its 16-64 m band is also the
  most anisotropic, see below). There is no roll-off and no correlation length within the
  measured band: "l" is not a property of these surfaces.
- Geikie 2014 (dry-snow zone, 2400-2700 m, April) is the exception: D(r) saturates
  beyond ~15-20 m lag (fig_b), the exponential ACF is best in 83 % of blocks with white
  residuals in 79 %, sigma = 4.8 cm (p5-p95 4.4-5.9), l = 5.3 m (3.3-7.3). That is a
  finite-correlation sastrugi field; per octave its RMS is flat at ~2.1-2.6 cm from 2 to
  32 m (a k^-2 spectrum, H ~ 0.1), so within the Bragg band it behaves like a power law
  with beta ~ 2.2, and the exponential tail at 1.5 m is 137 dB above the Gaussian one.
- On the precise 2017/2019 same-scan data no family is white (p > 0.05 in <= 5 % of
  blocks): the surface is more structured than any single-parameter tail (two-scale, or a
  curved log-log slope). The power law is the least wrong, at 0.05-0.1 rms in ln D.

## Question 2: scale dependence and along-line variability

### Octave RMS (cm), 1 km blocks: median [p5-p95]; block-to-block CV; correlation length of the block series (1/e, km); between-block std (dB) vs within-block (two 500 m halves, dB) and the implied real fraction of the variance

| flight | 1-2 m (bound) | 2-4 m | 4-8 m | 8-16 m | 16-32 m | 32-64 m |
|---|---|---|---|---|---|---|
| westcoast 2016 | 1.8 [0-3.2] | 2.2 [0.8-3.8] CV .44 Lc 2 (3.5/3.2 dB, 0.59) | 2.4 [1.7-4.2] CV .29 Lc 2 (2.3/1.3, 0.83) | 3.1 [2.0-6.3] CV .38 Lc 2 | 4.9 [1.8-11.5] CV .54 Lc 2 | 8.6 [2.6-20.2] CV .57 Lc 2 |
| westcoast 2017 | 0.9 [0-2.3] | 3.0 [1.7-5.2] CV .36 Lc 4 (3.1/1.2, 0.92) | 2.5 [1.6-6.2] CV .47 Lc 4 (3.6/1.7, 0.89) | 2.9 [1.7-6.7] CV .46 Lc 3 | 3.7 [2.0-7.5] CV .44 Lc 2 | 6.3 [3.1-13.1] CV .47 Lc 2 |
| westcoast 2019 | 1.7 [0.6-4.0] | 2.4 [1.7-4.2] CV .30 Lc 2 (2.3/1.4, 0.82) | 2.4 [1.9-3.7] CV .25 Lc 2 (1.8/1.2, 0.78) | 3.2 [2.3-5.6] CV .33 Lc 2 | 4.3 [2.7-8.5] CV .40 Lc 2 | 6.7 [2.6-15.5] CV .55 Lc 2 |
| geikie 2014 | 1.3 [0-2.6] | 2.1 [0.9-2.9] CV .31 Lc 1 (2.8/2.5, 0.58) | 2.5 [1.8-2.7] CV .11 Lc 1 (1.0/1.1, ~0) | 2.6 [2.0-3.0] CV .12 Lc 1 (1.0/1.5, 0) | 2.3 [1.6-3.0] CV .23 Lc 1 | 1.7 [0-3.1] CV .46 Lc 1 |
| getz 2016 | 1.7 [0-5.1] | 2.2 [0-6.3] CV .80 Lc 3 (5.6/3.6, 0.79) | 2.3 [1.4-6.3] CV .68 Lc 3 (3.8/2.1, 0.85) | 2.7 [1.8-8.9] CV .79 Lc 5 | 3.3 [1.7-12.5] CV .81 Lc 10 | 5.9 [1.2-16.6] CV .76 Lc 11 |
| david 11-19 | 3.5 [0-11.7] | 4.6 [1.0-17.2] CV .89 Lc 3 | 4.4 [1.5-12.4] CV .74 Lc 14 (5.6/1.9, 0.94) | 4.9 [2.1-16.1] CV .83 Lc 15 | 7.2 [2.0-26.1] CV .85 Lc 14 | 12.5 [2.2-32.4] CV .73 Lc 9 |
| david 11-20 | 4.2 [0-18.3] | 8.6 [2.4-22.0] CV .71 Lc 6 | 9.0 [3.1-30.0] CV .78 Lc 6 (7.1/3.3, 0.89) | 15.0 [3.8-45.6] CV .76 Lc 6 | 25.5 [4.8-58.0] CV .69 Lc 4 | 32.3 [7.2-78.8] CV .63 Lc 4 |

Fixture (sigma 4.9 cm, Gaussian l 2.98 m) octave RMS: 1-2 m 0.00, 2-4 m 0.32, 4-8 m 2.45,
8-16 m 3.31, 16-32 m 2.23, 32-64 m 1.21 cm.

Scale dependence: on westcoast/getz/david the octave RMS grows monotonically with
scale (x2.5-3.5 from the 4-8 m to the 32-64 m octave: self-affine, H ~ 0.4); on geikie it
is flat (2.1-2.6 cm per octave over 2-32 m). The fixture matches all lines in the 4-16 m
band (2.4-3.3 cm) but has 7-10x too little at 2-4 m and 3-7x too little at 32-64 m.

Along-line variability:
- Westcoast (all three years) and geikie: the block series decorrelate within 1-4 km
  (Lc), the between-block spread of the Bragg-band octaves is 2-4 dB (std), and a large
  part of that is real (the two 500 m halves agree to 1.2-1.7 dB in 2017/2019; on geikie
  the halves disagree as much as the blocks do, i.e. the line is statistically uniform
  at the block-estimate precision of ~1 dB). Binary segmentation of log RMS(4-8 m) with a
  4x BIC penalty finds NO change point on any westcoast year or on geikie. Spearman
  correlations with elevation are weak (|rho| <= 0.3); with the 100 m-scale slope
  moderate (rho 0.4-0.7 on westcoast: rougher where steeper); with s none. So these lines
  are a stationary random field at the km scale with no regimes.
- Getz: three regimes (s -7-32 km 2.3 cm; 32-72 km 3.6 cm; 72-174 km 2.2 cm, the latter
  floating/ice-shelf with H dropping to 0.18), Lc 5-11 km at >= 8 m, CV 0.7-0.8, with
  slope the strongest correlate (rho 0.62).
- David 2013-11-19 splits at s = 33 km between sea-ice/ice-tongue at sea level (2.2 cm)
  and grounded ice at ~900 m (6.2 cm, H 0.3), Lc ~14 km; 11-20 is sea ice / outer tongue
  (9-32 cm per octave, not a proxy for the grounded line).

### One (sigma, l) / one PSD per line, or per segment?

Measured in the quantity the radar needs, S(k_B), the per-block best-fit PSD scatters
around the pooled whole-line fit by (p5..p95, dB):

| flight | 60 MHz (5 m) | 195 MHz (1.5 m) | 300 MHz (1 m) | 400 MHz (0.75 m) |
|---|---|---|---|---|
| westcoast 2016 | -3.6..+5.2 | -4.3..+9.7 | -4.5..+11.1 | -4.6..+12.3 |
| westcoast 2017 | (-361)..+5.1 | -14.7..+4.9 | -12.4..+5.0 | -13.2..+5.0 |
| westcoast 2019 | -5.1..+3.9 | -5.7..+5.1 | -5.9..+5.2 | -6.2..+5.2 |
| geikie 2014 | -1.7..+3.0 | -1.7..+6.0 | -1.7..+7.4 | -1.7..+8.3 |
| getz 2016 | -8.6..+7.5 | -11.2..+7.7 | -12.0..+7.7 | -12.2..+7.5 |
| david 11-19 | -13.8..+7.7 | -16.5..+6.8 | -17.9..+6.9 | -18.9..+6.7 |

(The 2017 lower tails are the Gaussian-best blocks; with the power law forced the 2017
scatter is like 2019's.)

Conclusion: **westcoast and geikie: one PSD per line** (per year), with a +-5 dB (p5-p95)
block-to-block scatter at 1 km that has a 1-4 km correlation length and no regime
structure; sub-dividing into segments cannot reduce it because it is not organised
along s. Quote the pooled power law (westcoast: H 0.37-0.44, i.e. beta 2.7-2.9; geikie:
exponential 4.8 cm / 5.3 m) and carry the +-5 dB as the roughness uncertainty. For a
Gaussian-only kernel the pooled Gaussian fit is meaningless in the Bragg band (see
below). **Getz and David: per-segment values are needed**, at a 10-15 km segment length
(the Lc of the block series), with the grounding line / ice-shelf boundary as the
natural break (getz s ~ 72 km; the configured GL is 69.7 km): the block scatter there is
+-8-12 dB and organised in 10-40 km runs (fig_a getz/david).

### Anisotropy (along/cross octave-RMS ratio, median [p5-p95]; power-law H along / cross)

| flight | 2-4 m | 4-8 m | 8-16 m | 16-32 m | 32-64 m | H along/cross |
|---|---|---|---|---|---|---|
| westcoast 2016 (cross-scan) | 0.88 | 0.67 | 0.62 [0.39-1.08] | 0.33 [0-0.68] | 0.34 [0.27-0.66] | 0.28 / 0.66 |
| westcoast 2017 (same-scan) | 1.24 | 0.86 | 0.88 [0.25-1.35] | 1.24 [0.46-2.81] | 1.81 [1.15-2.78] | 0.33 / 0.21 |
| westcoast 2019 (same-scan) | 0.84 | 0.83 | 0.88 [0.65-1.08] | 1.10 [0.73-1.83] | 1.36 [0.76-2.05] | 0.38 / 0.26 |
| geikie 2014 | 1.12 | 1.04 | 1.08 [0.67-1.62] | 0.88 | 0.34 [0-1.08] | 0.09 / 0.12 |
| getz 2016 | 0.87 | 0.92 | 0.84 [0.46-1.62] | 0.54 | 0.36 [0-0.98] | 0.16 / 0.46 |
| david 11-19 | 0.97 | 0.78 | 0.81 | 0.63 | 0.52 | 0.24 / 0.41 |

In the Bragg band (2-16 m) the surfaces are isotropic to within ~20 % on every line
(ratios 0.8-1.1, geikie 1.0-1.1). Above 16 m the ratio departs from 1 but with
opposite signs on the same line in different years/instruments (2016 cross-scan 0.33 vs
2017/2019 same-scan 1.4-1.8), so the >= 16 m anisotropy is not trusted: it is at the
mercy of the swath-limited cross-track detrend (280 m swath vs 200 m half-power cut) and
of roll noise that grows with cross-track separation in the cross-scan class. Treat the
surface as isotropic for the kernel; the >= 30 m band is facet tilt anyway.

### Year-to-year spread, westcoast (1 km blocks on the shared axis, ~108 common km)

| pair | RMS 4-8 m: median dB, IQR, corr(log) | RMS 16-32 m | S(1.5 m) best fit: median dB, IQR | S(5 m) | H diff |
|---|---|---|---|---|---|
| 2016 -> 2017 | +1.3 [-0.5, +3.3], r 0.62 | -2.1 [-3.2, -0.7], r 0.81 | +4.8 [-0.3, +8.2] | +3.8 [+0.6, +5.4] | -0.16 |
| 2016 -> 2019 | -0.3 [-1.4, +0.8], r 0.27 | -0.9 [-2.0, +0.5], r 0.88 | +3.0 [-0.9, +6.1] | +2.3 [0, +4.1] | -0.11 |
| 2017 -> 2019 | -1.3 [-4.1, +0.9], r 0.11 | +1.0 [+0.5, +2.4], r 0.77 | -2.8 [-5.3, +0.8] | -1.9 [-3.8, +2.8] | +0.09 |

The 16-32 m octave is a persistent topographic signature (block-to-block correlation
0.8-0.9 between years, fig_years); the Bragg-band 4-8 m octave is not (r 0.1-0.6): it is
a snow-surface property that re-forms each season. Line-median S(k_B) differs between
years by 2-5 dB (60 MHz: -46.9 / -43.6 / -44.2 dB; 195 MHz: -63.1 / -56.7 / -59.3 dB),
part of which is the instrument change (2016 cross-scan class). So the year-to-year
uncertainty on the Bragg-band PSD is ~+-3 dB on the line median, comparable to the
along-line scatter.

## PSD at the Bragg wavelengths (2-D S(k), dB re 1 m^4; k_B = 2 pi / Lambda, theta = 30 deg)

Per-block best-family median [p5-p95] / pooled whole-line fit / fixture. * = below the
smallest resolved lag (0.9 m same-scan, 0.27 m cross-scan): power-law extrapolation by
< 1 octave.

| flight | 60 MHz, 5 m | 195 MHz, 1.5 m* | 300 MHz, 1.0 m* | 400 MHz, 0.75 m* |
|---|---|---|---|---|
| westcoast 2016 | -46.9 [-50.5, -41.7] / -46.9 | -63.1 [-68.0, -54.0] / -63.7 | -68.6 [-73.9, -58.3] / -69.4 | -72.4 [-78.0, -61.0] / -73.4 |
| westcoast 2017 | -43.6 [.., -37.2] / -42.3 | -56.7 [-71.4, -51.7] / -56.6 | -61.4 [-73.9, -56.5] / -61.5 | -64.6 [-78.1, -60.0] / -64.9 |
| westcoast 2019 | -44.2 [-48.6, -39.7] / -43.6 | -59.3 [-64.3, -53.5] / -58.6 | -64.6 [-69.6, -58.5] / -63.7 | -68.2 [-73.5, -62.0] / -67.3 |
| geikie 2014 | -44.5 [-46.3, -41.6] / -44.6 | -60.1 [-62.0, -54.3] / -60.3 | -65.3 [-67.2, -58.2] / -65.5 | -69.1 [-71.0, -60.9] / -69.3 |
| getz 2016 | -45.3 [-50.6, -34.5] / -42.0 | -59.9 [-67.5, -48.7] / -56.4 | -65.0 [-73.2, -53.5] / -61.2 | -68.5 [-76.8, -57.1] / -64.6 |
| david 11-19 (grounded proxy) | -37.1 [-48.6, -27.1] / -34.8 | -51.5 [-64.8, -41.5] / -48.4 | -56.3 [-70.8, -46.1] / -53.0 | -59.7 [-75.1, -49.5] / -56.2 |
| david 11-20 (sea ice) | -36.4 / -30.2 | -51.1 / -45.9 | -56.6 / -51.2 | -60.8 / -54.9 |
| **fixture 4.9 cm / 2.98 m Gaussian** | **-42.9** | **-196.9** | **-408.3** | **-704.4** |
| per-block Gaussian fits (median) | -46..-1331 | -229..-1724 | < -400 | < -700 |

The gridded 1 m PSD confirms the 60 MHz value only where it clears its floor (westcoast
2019, 74 % of blocks above floor: -30 dB after floor subtraction vs -44 model; the
residual mask leakage makes even that an upper bound).

## Comparison with the C&S fixture (sigma 4.9 cm, l 2.98 m, Gaussian)

- Amplitude: the fixture's total sigma and its 4-16 m octave RMS are right to within
  ~1 dB on every Greenland line and on Getz; geikie's exponential (4.8 cm, 5.3 m) is
  numerically almost the fixture. The firn-layer numbers were a lucky surface guess in
  the band where the Gaussian has its power.
- Form: wrong everywhere. Measured octave RMS keeps growing above l (self-affine) and
  does not collapse below 4 m; in the Bragg band the Gaussian tail is 4 dB too high at
  60 MHz (the knee) and then 135-145 dB too low at 195 MHz, ~350 dB at 300 MHz, ~640 dB
  at 400 MHz. The measured S(1.5 m) is -57..-63 dB (Greenland), -60 dB (Getz): the
  195 MHz diffuse surface term computed with the fixture is nonsense, and the "l = 1 vs
  3 m" 90 dB sensitivity is an artefact of the Gaussian tail, as the ICESat-2 note
  anticipated. Plan path B (power-law / tabulated ACF; or B1 effective Gaussian matched
  at each k_B) is required; path A is closed.
- For B1 at 195 MHz, 30 deg, the target is S(2 pi / 1.5 m) = -57 (2017), -59 (2019),
  -63 (2016) dB on westcoast, -60 dB geikie, -60 (line) / -56 (pooled) dB getz, with
  +-5 dB along-line and +-3 dB year-to-year.

## Caveats

- Spot and lag floor: ATM footprint ~1 m; same-scan pairs start at 0.9 m (2017/2019),
  cross-scan at 0.27 m; the 195/300/400 MHz values are power-law extrapolations by 0.3,
  0.9, 1.3 octaves from the last resolved lag; on the 3 kHz flights the 2-4 m octave
  rests on cross-scan pairs with a 3-5 cm nugget and is the least precise entry.
- Noise: the same-scan nugget is fixed at sigma_r from the crossover budget (2017:
  2.6 cm [1.8-3.1]; 2019: 1.6 cm); moving it across its range shifts the 2017 H by
  ~+-0.05 and S(1.5 m) by ~+-2 dB. sigma_r includes sub-0.3 m roughness, so it is an
  upper bound and the small-scale PSD a lower bound.
- Swath 190-340 m: cross-track statistics above ~60 m lag and the >= 16 m anisotropy
  are swath/detrend-limited (see above); the 2-D isotropy test is solid only for
  2-16 m.
- Detrend half-power at 200 m: the 32-64 m octave is ~7 % low; nothing in the Bragg band
  is affected.
- Dates: westcoast 10-14 May (pre-melt, percolation zone), geikie 21 April (dry snow),
  getz 5 Nov (early austral summer), david 19-20 Nov 2013 (not the radar dates, not the
  radar track: 4-40 km away, mostly floating tongue / sea ice; the grounded s > 33 km
  part of 11-19 is the only usable proxy and is 2-3x rougher than getz).
- No family is white on the precise data; the power law is a description, not a
  physical model; ILATM2 platelet roughness was not pulled (not needed once the pair
  statistics carried their own noise budget).
