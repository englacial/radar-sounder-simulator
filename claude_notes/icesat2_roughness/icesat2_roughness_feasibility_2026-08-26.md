# Can ICESat-2 constrain the surface-roughness parametrisation (sigma, l)? — feasibility, 2026-08-26

Context: the surface interface uses the Gerekos 2023 rough-facet model with
sigma = 0.049 m, l = 2.98 m (Gaussian ACF), taken from the C&S 2020 Fig. 11
firn inversion at 0 m depth (`tools/run_altitude_comparison.py:150`), not
from any surface measurement. The 14 km HAPS design point is ~90 dB
sensitive to l (3 m vs 1 m) because diffuse scatter at ~30 deg off nadir goes
as exp(-(k l sin theta)^2/4). (The referenced
`claude_notes/haps_design_study/summary.md` does not exist on disk; the
sensitivity statement is taken from the task brief and the log
`claude_notes/logs/wc_pilot_haps_pulse.log`.)

## Verdict

**No for l; partially for sigma; yes for the ACF/PSD shape at 40-1000 m.**

- The radar needs the surface PSD at the Bragg wavenumber
  k_B = 2 k0 sin(theta). For theta ~ 30 deg that is a surface wavelength
  Lambda = lambda0 / (2 sin theta) = **5 m at 60 MHz, 1.5 m at 195 MHz,
  0.75 m at 400 MHz**. The whole decision (l = 1 m vs 3 m) lives in the
  0.5-5 m band.
- ICESat-2's ~11 m (1/e-ish; 17 m design) footprint is a Gaussian low-pass
  with sigma_fp ~ 4.7 m: power transfer exp(-(k sigma_fp)^2) is **-96 dB at
  Lambda = 3 m, -38 dB at 10 m, -4 dB at 30 m**. Photons within a shot all
  carry the shot's along-track coordinate, so the 0.7 m shot spacing does
  not buy sub-footprint horizontal resolution. Nothing in the 0.5-5 m band
  survives in the height profile, from ATL03 or ATL06.
- What ATL03 **can** do: (a) the within-shot photon height spread gives the
  sub-footprint RMS height sigma_sub (all scales < ~11 m pooled) in
  quadrature with the instrument response (~0.1 m single-photon spread from
  the 1.5 ns pulse + timing); (b) the shot-to-shot profile gives the ACF/PSD
  for scales ~20 m to km with a 2-3 cm noise floor after averaging, i.e.
  whether the surface is Gaussian-correlated, exponential, or a power law
  (beta = 2H+1) down to ~20 m, which is what you extrapolate into the Bragg
  band. (c) For a Gaussian surface with l << 2 sigma_fp, the only combination
  the profile constrains is the low-k plateau **sigma^2 l** (prototype: 7%
  on sigma^2 l, +-53% on l at sigma = 5 cm, l = 3 m; Gaussian and
  exponential fits give identical residuals). With l = 30 m both sigma and
  l are recovered to 1-5% and the forms separate.
- ATL06 (20 m posting, 40 m linear-fit window) adds a sinc^2(k*20 m)
  transfer on top: usable ACF only for lags >= 40 m. Its `h_robust_sprd`
  (robust spread of photon residuals about the 40 m fit) is the same
  sub-window sigma proxy as (a), instrument-noise-inclusive. `dh_fit_dx` is
  the 40 m slope, i.e. a facet-scale tilt, not roughness. ATL08 (100 m
  land/vegetation) and ATL11 (annual crossover-corrected heights) add nothing
  here.

So: ICESat-2 replaces the *un-measured* assumption with a measured
sigma_sub (band-pooled, needs a noise calibration) plus a measured
40-1000 m spectrum, and fixes the ACF family — but l in the 0.5-5 m band
must still come from extrapolation of that spectrum or from a higher-
resolution source. The better source is already coincident with the study
lines: **Operation IceBridge ATM** (ILATM1B point clouds; ~1 m footprint,
sub-m to few-m spacing across a 250 m swath, ~3-7 cm vertical) flew on the
same P-3 flights as the 2016/2017/2019 MCoRDS products of the westcoast and
geikie lines. It reaches the 1-30 m band directly, in 2-D (isotropy check),
on the same day as the radar. That, not ICESat-2, is the recommended path
to l.

## What the model needs, and the estimator

Gerekos/S-IEM take an isotropic Gaussian ACF with (sigma, l) *below the
facet size* (27-70 m here) and above ~lambda/10 (0.1-0.5 m). The diffuse
term is proportional to the 2-D surface PSD at k_B. A 1-D along-track
profile of an isotropic surface has 1-D PSD
W1(k) = sigma^2 l/(2 sqrt(pi)) exp(-(k l)^2/4)   (Gaussian)
W1(k) = sigma^2 l/pi / (1 + (k l)^2)             (exponential)
W1(k) = A k^-beta, beta = 2H+1                    (self-affine)
and the same ACF family and l as the 2-D surface (isotropy assumed; the
1-D transect of an anisotropic sastrugi field measures the along-track
projection only — Greenland sastrugi are wind-aligned, so a single ground
track gives one azimuth). Estimator: quality-filter, resample to uniform
spacing per contiguous run, detrend at a scale >> l (Savitzky-Golay,
>= 25 samples and >= 1 km for ATL06), Welch PSD log-binned in k, then fit
2 W1(k) T(k) + N0 in log space with T the footprint (and segment) transfer
and N0 a white noise floor. Degeneracies: (i) sigma vs l when l < 2 sigma_fp
(only sigma^2 l), (ii) Gaussian vs exponential when only the plateau is
seen, (iii) power-law amplitude vs noise floor when the slope is shallow.
Report `constrained` (relative l error < 1) with every fit.

## Data: what is local, what to fetch

- Local: nothing usable. `~/Documents/hackdays/xover/data/atl06_*.parquet`
  are 8-61 crossover points, not profiles. No ATL03/ATL06 HDF5 anywhere.
- Credentials: `~/.netrc` has `urs.earthdata.nasa.gov`; `earthaccess 0.18`
  is a project dependency and `earthaccess.login(strategy="netrc")`
  authenticates. `h5py 3.16` present; `icepyx`/`sliderule` not installed.
- Fetched today (66 MB) into `outputs/icesat2/`:
  `ATL06_20190515210004_07270305_007_01.h5` (RGT 727, cycle 3, 15 May 2019 —
  one day after the 2019 P-3 westcoast flight). Beams gt3l/gt3r cross the
  pilot box (-49.9..-49.3 E, 70.45..70.70 N) over 28 km at 1465 m elevation;
  gt2l/gt2r clip it for 3-4 km.
- Granules over the westcoast pilot window, Mar-Jun 2019: 8 ATL06 (64-104 MB
  each) and their 8 ATL03 twins (1.3-3.9 GB each; not downloaded). RGTs
  0277/0285/0727/0780/1222/1230.
- Access paths: `earthaccess.search_data(short_name="ATL03", bounding_box=,
  temporal=)` + `download`; for ATL03 use SlideRule (`pip install
  sliderule`, `icesat2.atl03sp` with a polygon: returns only the photons in
  the box, MBs not GBs) or NSIDC/Harmony spatial subsetting; OpenAltimetry
  for a browser look. Same pattern for the geikie, David and Getz lines
  (bbox from each line's reference-pass nav; David/Getz are Antarctic,
  hemisphere-agnostic for ICESat-2).

## First real numbers (westcoast pilot, ATL06 gt3l/gt3r, percolation zone)

`uv run claude_notes/icesat2_roughness/roughness_from_atl.py --atl06
outputs/icesat2/ATL06_20190515210004_07270305_007_01.h5 --bbox -49.90 70.45
-49.30 70.70 --max-lag-m 400` (json/png in `outputs/icesat2/`):

| beam | h_robust_sprd med | h_li_sigma | rms after 1 km detrend | ACF-Gaussian sigma, l | ACF-exp sigma, l | PSD power law |
|---|---|---|---|---|---|---|
| gt3l strong | 0.151 m | 0.012 m | 0.101 m | 0.10 m, 86 +- 20 m | 0.13 m, 49 +- 19 m | (fit unconstrained) |
| gt3r weak | 0.156 m | 0.024 m | 0.093 m | 0.09 m, 92 +- 19 m | 0.11 m, 53 +- 19 m | (unconstrained) |
| gt2l strong (3 km) | 0.157 m | 0.014 m | 0.183 m | 0.18 m, 144 m | 0.21 m, 92 m | beta 2.0 +- 0.3 (H 0.5) |

Reading: at 40-1000 m the surface is decimetre-rough with a ~50-150 m
correlation scale (undulations / percolation-zone topography), and where the
PSD fit converges the slope is beta ~ 1.7-2.0 (H ~ 0.4-0.5, roughly
Brownian). `h_robust_sprd` ~ 0.15 m is an upper bound on the sub-40 m sigma
that is essentially the ATL06 photon noise (0.1-0.15 m is what smooth
interior Antarctica gives too, Brunt 2019); so sigma_sub < ~0.1 m here,
consistent with the 5 cm fixture but far from proving it. Extrapolating
beta = 2 from 40 m to 3 m predicts W(3 m)/W(40 m) ~ 1/180, i.e. much MORE
small-scale power than a Gaussian ACF with l = 3 m would put at 1 m
wavelengths (exp(-(kl)^2/4) is ~-44 dB from plateau at Lambda = 3 m,
-400 dB at 1 m). This is the substantive point: if the real surface is
self-affine down to the Bragg band, the Gaussian-l parametrisation is the
wrong family and the "l = 1 vs 3 m" question is ill-posed; the diffuse tail
would be set by (A, beta) instead and would not collapse exponentially
with angle. The 90 dB swing is an artefact of the Gaussian tail.

## Prototype

`claude_notes/icesat2_roughness/roughness_from_atl.py` (numpy/scipy/h5py
only; `uv run`):
- `--atl06 FILE --bbox ...`: per-beam quality-filtered `h_li`, along-track
  distance, contiguous-run resampling, SG detrend, band RMS, ACF with
  footprint-aware Gaussian and exponential fits (+ `constrained` flags),
  log-binned Welch PSD with Gaussian / exponential / power-law + noise-floor
  fits including the footprint x 40 m-segment transfer; also reports
  `h_robust_sprd`, `h_li_sigma`, `dh_fit_dx` medians. Writes json + png.
- `--atl03 FILE --bbox ...`: land-ice signal photons (conf >= 3), per-shot
  grouping via `pce_mframe_cnt`/`ph_id_pulse`, within-shot residual spread
  (sub-footprint sigma proxy, noise-inclusive), 1 m shot-mean profile
  through the same statistics. Untested on real ATL03 (no granule
  downloaded); path names follow the ATL03 v6/v7 layout.
- `--synthetic [--sigma --l]`: 1-D Gaussian-ACF surface at 0.7 m, observed
  (i) raw, (ii) ATL03-like (11 m footprint + 0.10 m photon noise), (iii)
  ATL06-like (footprint + 40 m boxcar + 20 m posting + 0.03 m noise).
  Results: raw recovers sigma 0.049/l 2.93 (truth 0.05/3.0) from the PSD;
  ATL03-like recovers sigma^2 l to 7% but l only +-53% and cannot tell
  Gaussian from exponential; ATL06-like sees nothing at l = 3 m. At
  sigma 0.2/l 30 all three paths recover l (ATL06 30 +- 10 m).
- Known rough edges: the PSD fit on the ATL06 path is unreliable (the
  segment sinc^2 transfer plus a short periodogram; the ACF fit is the one
  to trust there); no anisotropy handling; no instrument-noise calibration
  for the photon-spread estimate.

## Plan

1. Per line, pull ATL03 photons through SlideRule for a 10 km x ~2 km box
   around the pilot window (westcoast: bbox above; geikie / David / Getz from
   the reference-pass nav), all strong beams, all cycles in the radar
   season +- 1 month (sastrugi migrate seasonally). ~20-50 MB per line.
2. Calibrate the photon-spread instrument term on a flat target of the same
   granule (a frozen lake or, for Antarctica, the flattest 1 km of the
   interior) — the sub-footprint sigma is the excess variance and needs
   sigma_inst to ~1 cm, so treat it as a bound unless the calibration is
   clean.
3. Run the prototype on the 1 m shot-mean profiles: ACF family + (A, beta)
   from 20 m to 1 km, compare across beams (azimuth check) and cycles.
4. Decide the parametrisation family from step 3. If beta ~ 2 persists down
   to 20 m, replace the Gaussian-l surface roughness with a two-scale or
   power-law surface PSD in the diffuse term (model change, not a fit), and
   use ATM (ILATM1B, same P-3 flights) to check the 1-30 m band directly:
   grid the swath at 1 m, 2-D PSD, isotropy, and sigma in the 0.5-5 m band.
   ATM is the only listed source that reaches the Bragg band.
5. Validate: (a) does the ATL03/ATM-derived (sigma_sub, spectrum) reproduce
   the surface-return level and, more sensitively, the measured P-3
   mid-column clutter that the design study used as discriminator
   (`wc_pilot_haps_pulse` decomposition at s = 45 km; 20-70 m below-surface
   band from the B26 work); (b) does it fall within the C&S 0 m point
   (sigma 0.05, l 3 m) — if C&S's l is a firn-layer value it has no reason
   to match the surface, and the comparison tells whether the fixture was
   lucky.
6. Effort: ATL03 pull + prototype run per line, ~1 day for four lines
   including the flat-target calibration; ATM 1-30 m band analysis, ~2 days;
   model change for a non-Gaussian surface PSD, ~3-5 days plus re-running
   the HAPS design points.

## Literature (from memory plus one search; check before citing)

- van Tiggelen et al. 2021, TC 15, 2601 (ICESat-2 aerodynamic roughness,
  K-transect): explicitly finds sastrugi (up to 0.5 m high, horizontal
  extent < footprint) are *not resolved* by ATL03; roughness is derived at
  the ~10-100 m scale only.
- Smith et al. 2019 (ATL06 ATBD/RSE): `h_robust_sprd`, `dh_fit_dx` are 40 m
  window statistics; `h_li_sigma` ~ 1-3 cm for strong beams. Brunt et al.
  2019 GRL: ATL06 vs GPS over interior Antarctica agrees to ~few cm, with
  the surface's own sub-footprint spread ~0.1 m.
- Magruder et al. 2021: footprint ~10.6-12 m (not the 17 m design).
- Yi, Zwally et al. 2005 (GLAS waveform width -> ice-sheet roughness);
  Kurtz & Markus 2012 (sea-ice roughness from ICESat); Farrell et al. 2020
  GRL (sea-ice topography from ATL03 at 5-10 m).
- Lacroix et al. 2008 (ASIRAS/ Antarctic sastrugi): sastrugi heights
  0.1-0.5 m, wavelengths 1-20 m — i.e. structure sitting exactly in the
  Bragg band and below the ICESat-2 footprint.
- Grima et al. 2014 (radar statistical reconnaissance, Greenland MCoRDS) and
  Grima et al. 2019 (RSR vs ATM): inferred surface sigma of cm at 195 MHz
  and correlation lengths of metres; the closest thing to an in-band surface
  (sigma, l) for these lines, and it argues that ATM was already used for
  exactly this comparison.
- Herzfeld et al. (sastrugi/ATM DDA, 2014-2021): ATM at 1 m spacing resolves
  sastrugi fields; roughness length scales of 3-30 m reported.
- Ku/Ka altimetry echo-strength roughness of Greenland (TC 19, 1221, 2025):
  cm-scale sigma maps, but at km footprints.
- Expected values: dry-snow interior sigma 2-10 cm, sastrugi correlation
  1-10 m; percolation/coastal Greenland decimetre sigma at 10-100 m scales
  with melt/crevasse features; so a *surface* l = 3 m with sigma 5 cm is
  plausible as an order of magnitude but there is no measurement behind it
  for these lines, and the ACF family is the bigger uncertainty.

Sources consulted: https://tc.copernicus.org/articles/15/2601/2021/ ,
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019GL084886 ,
https://tc.copernicus.org/articles/19/1221/2025/ ,
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL090708 .
