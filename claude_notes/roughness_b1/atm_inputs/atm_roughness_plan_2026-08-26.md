# Plan: OIB ATM for surface-scattering form and roughness calibration

Goal: replace the Culberg & Schroeder Fig. 11 firn-layer fixture (σ = 4.9 cm,
Gaussian l = 2.98 m, applied to the air–snow surface) with a measured surface
roughness description on each study line, and decide whether the Gaussian-ACF
Gerekos facet law is the right *form* or whether the surface is self-affine
(power-law PSD) over the scales that set off-nadir clutter.

Why ATM: ICESat-2's 11 m footprint kills everything below ~10 m
(claude_notes/icesat2_roughness/…feasibility…md); the radar needs the surface
PSD at the Bragg wavelength Λ = λ/(2 sin θ): 5 m (60 MHz), 1.5 m (195),
1.0 m (300), 0.75 m (400) at the 30° bed-delay clutter angle, and out to
~30–70 m (facet size). ATM L1B (ILATM1B) has ~1 m spot spacing on a ~250 m
swath, ~7–10 cm shot precision, and flew on the same P-3/DC-8 flights as the
radar products — same day, same track.

## 0. Data

| line | radar frames | ATM |
|---|---|---|
| greenland_westcoast | 2016-05-11 (20160511_03), 2017-05-10 (20170510_03), 2019-05-14 (20190514_01) | ILATM1B v2, same flights |
| greenland_geikie01_transit | 2014-04-21, 2017-04-24 | ILATM1B v1 (2014), v2 (2017) |
| antarctica_getz | 2016-10-28 / 10-31 / 11-05 (DC-8) | ILATM1B v2 |
| antarctica_david | 2017/2022/2023 Basler (UTIG/AWI) | **no ATM** — use ICESat-2 for the ≥ 20 m band only, or REMA strips |

Pull via `earthaccess` (creds in ~/.netrc already work for NSIDC): collection
ILATM1B, temporal = flight day, bbox = the line's nav envelope ± 2 km. Expect
~10–20 granules × 100–300 MB per flight day; subset to the frames' time span
from the OPR nav (`cache/…params.json` has frame UTC). Store under
`outputs/cache/atm/<line>/<date>/`. Also pull ILATM2 for the same days as a
free sanity check (250 m platelet RMS roughness).

## 1. Point-cloud preprocessing (per frame, per flight)

1. Read HDF5 (`elevation`, `latitude`, `longitude`, `instrument_parameters/rel_time`, `xmt_sigstr`, `rcv_sigstr`), drop flagged/low-return shots.
2. Project to the line CRS (EPSG:3413 / 3031); compute along-track s from the radar nav so ATM and radargram share an axis.
3. Grid to a 1 m × 1 m raster over the swath with nearest/median binning; keep a shot-count mask (the conical scan leaves gaps on the swath axis — don't interpolate across them).
4. Detrend: remove ≥ 100 m scales with a 2-D Savitzky–Golay or a high-pass; what remains is the sub-facet field h(x, y).
5. **Noise calibration** (essential — 7–10 cm shot noise ≈ the σ we're after): use scan-line crossovers (the conical scan revisits the same ground within seconds) to estimate the white-noise floor per flight; subtract it from all PSDs and variances. Cross-check on a flat lake/sea-ice segment if the flight has one.

## 2. Scattering form: which ACF/PSD family?

For each 1 km along-track block:
- 2-D PSD of h(x, y) (Welch, Hann), radially averaged, plus along- and cross-track 1-D slices (isotropy check — the Gerekos law assumes isotropic; sastrugi are not).
- Fit, with the noise floor as a free additive term, over 2 m ≤ Λ ≤ 100 m:
  - Gaussian ACF: S(k) ∝ σ² l² exp(−k² l²/4)
  - Exponential ACF: S(k) ∝ σ² l² (1 + k² l²)^(−3/2)
  - Power law (self-affine): S(k) ∝ k^(−β), β = 2H + 2 in 2-D
- Decide by AIC/BIC per block and by whether the best-fit residual is white. Report the band-limited RMS in octaves 1–2, 2–4, 4–8, 8–16, 16–32 m: a Gaussian surface's octave RMS collapses above l; a self-affine one decays as a power of scale.
- Output: per line, the fraction of blocks best described by each family, β or l, and the along/cross-track anisotropy ratio.

Expected result (from the ICESat-2 40–1000 m slope β ≈ 1.7–2.0 and the sastrugi literature): power-law down to ~1 m with a possible roll-off near the sastrugi scale (0.5–2 m). If so, "l" is not a property of the surface and the Gaussian tail is the wrong form.

## 3. Roughness calibration: numbers the simulator can use

Two paths depending on §2's answer.

**A. Gaussian form is adequate** (band RMS collapses above some l ≤ ~5 m):
fit σ and l directly from the octave RMS and the ACF, per line, per year;
put them in the line YAML as `surface_roughness: {sigma_m, corr_length_m, provenance}` and let the runner read them instead of `SURF_ROUGH_*` (small change; fork the chunk rid with the values so caches don't collide — today the rid encodes only `_srough`).

**B. Surface is self-affine** (likely): the Gerekos series is Gaussian-only. Options, cheapest first:
1. *Effective Gaussian per frequency*: choose (σ_eff, l_eff) so the Gaussian law reproduces the measured PSD **at the Bragg wavelength of that frequency and angle** (S_meas(2k sin θ) matched at θ = 30°, plus total σ² in the 0.5–30 m band). This keeps the kernel unchanged and is exact where it matters (the bed-delay clutter angle) but wrong elsewhere; document it as frequency-dependent fixtures.
2. *Replace D_Φ's Gaussian ACF with a tabulated one*: the Gerekos incoherent term is a Kirchhoff series whose m-th term uses the m-fold convolution of the ACF; for an exponential or power-law ACF this is not closed-form but can be precomputed numerically on a (k sin θ) grid per interface and fed to the kernel as a lookup. Validate the same way docs/roughness.md validates the Gaussian case (facet-in-isolation Monte Carlo with correlated power-law surfaces, ~0.7 dB target). Bigger change (~1 week), but it is the one that makes "form" a solved problem.
3. *Two-scale*: keep Gaussian small-scale roughness for σ at < 1 m and push the 1–30 m band into the DEM as facet tilt by densifying facets from ATM (facets of ~5 m). Compute cost ~50–200× on HAPS reach; not viable for campaigns, useful as a referee run on one 1 km block.

## 4. Validation against radar

The discriminator already in hand: the measured P-3 mid-column clutter on westcoast (−59.4 dB; sim −117 with the fixture, −72 with l = 1 m) and the measured/sim altitude trend (500 m vs 14 km synthetic is not measurable, but the 2014/2017 geikie and 2016/2017/2019 westcoast passes at different AGL are). Steps:
1. Run `pilot_smoke` on westcoast and geikie with the ATM-derived roughness (path A or B1) and compare mid-column and bed-tail metrics to measured; target the documented 2–8 dB agreement, not the current 40–60 dB miss.
2. Doppler-spectrum check à la C&S: their off-nadir angular scattering function came from the along-track Doppler spectrum of the radar itself (Section II); do the same on the westcoast frames and compare to the ATM-derived PSD evaluated at the same Bragg wavelengths. This is a radar-vs-laser test of the *form* independent of the simulator.
3. Then re-run the HAPS design ladder (claude_notes/haps_design_study, branch waveform-explicit-chirp) at 60/150/225/300 MHz with the calibrated law: this is what collapses the +70 / −19 dB spread.

## 5. Effort and order

1. Data pull + preprocessing + noise calibration, westcoast 2017 and 2019 first: ~1 day.
2. Form analysis (§2) on westcoast, then geikie and getz: ~1 day.
3. Path A or B1 calibration into the runner + pilot_smoke validation: ~1 day.
4. Doppler-spectrum cross-check: ~1 day.
5. Path B2 (tabulated ACF in the kernel) only if B1's frequency-dependent fixtures prove inconsistent across the 195 MHz passes vs the low-frequency Basler/MKB lines: ~1 week.

## Caveats

- ATM swath is ~250 m centred on the flight track; cross-track statistics are limited to that width — fine for 0.5–30 m scales, and the swath gives 2-D so isotropy is testable (ICESat-2 cannot).
- Snow surface changes between the ATM pass and any future HAPS flight; what we calibrate is the spring-campaign surface on these lines, and the year-to-year spread across 2016/2017/2019 is itself the uncertainty to report.
- Below ~1 m the ATM spot (~1 m) and shot noise limit the measurement; at 400 MHz (Λ = 0.75 m) we would be extrapolating the fitted spectrum by less than one octave, acceptable; at 60–300 MHz the Bragg band is inside the measured range.
- David has no ATM: treat its roughness as transferred from Getz (same season/ice type argument) and say so.
