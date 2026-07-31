# B26 measured-vs-simulated comparison (2026-07-09)

Frame 2019_Greenland_P3 / 20190418_01_009 passes 5.75 m from the B26 firn
core (closest trace 434; claude_notes/firn_core_flightlines.md). Tool
`tools/run_b26_comparison.py` (resumable, pilot-budgeted); deliverable
`outputs/b26_comparison/` (self-contained report.html + metrics.json, group
"xOPR clutter" / case "b26_comparison", mirrored into
`outputs/verification/b26_comparison/` for tools/make_report.py); light
integration test `tests/integration/test_b26_comparison.py`.

## 2019 season parameters (frame's own param structs; NOT the 2017 values)

`outputs/cache/mcords_2019P3_params.json` from
`Data_20190418_01_009_source.mat` (param_records/param_sar/param_array):
chirp 180-210 MHz (f0 195, B 30 MHz) -- same band as 2017 -- but Tpd
{1, 3, 10} us (bed waveform 10 us), 20% Tukey, ft_wind = @hanning, rx_paths
all 7 center elements, prf 12 kHz, raw fs 111.11 MHz, and the PRODUCT twtt
grid is dt = 16.667 ns (60 MHz), not 2017's 33.333 ns. CSARP_standard: f-k
SAR sigma_x 2.5 m, 11-look hanning, dline 6 (~15 m posting; measured trace
spacing 14.7 m).

## Configuration (final; NO shrink steps needed)

10 km sub-segment centered on the closest approach, 100 traces (103 m
spacing), surface+bed cross-track +-3 km, firn strip +-600 m; dt_sim =
dt/4 = 4.1667 ns (alias 45 MHz = 3B/2, warning asserted silent; decimate
[::4] exact onto frame bins), window frame bins [119, 2165] (2.25-36.35 us).
Facets: beta=0.5 Fresnel minimized over the stack (deepest firn layer binds)
SNAPPED to 32/3 = 10.667 m so whole-cell crops tessellate on the wide
window's exact facet lattice; LPA nadir error 17% (worst case, recorded).
Media: air / ice(3.17, 15 dB/km one-way, the M24 warm-ice value -- generous
for this cold interior site, recorded) / bed(eps 8); firn = B26 point-sampled
Kovacs eps, equal placement 1-119.66 m, N in {10, 20}.

Firn contribution ran on the narrow strip as 6 ALONG-TRACK CHUNKS cropped
from the wide scene's DEM (one bbox around the whole diagonal segment would
carry ~5x the strip area -- the chunking is what fit the budget), then
field-summed with the wide surface+bed run, EXCLUDING the firn run's own
surface layer (no double count). Seam verified: firn layer-0 field, scaled by
the air->firn0 / air->ice gamma ratio, matches the wide run's surface field
to 1.24e-3 median max-relative-deviation in the first 1.5 us (identical
facet lattice makes this tessellation-noise-free).

## Budget: pilot projection vs actual

2-trace pilots: firn N=20 chunk 23.4 s steady (56.5k facets), wide 2.4 s
(1.76M facets). Projection 1596 s total -> under the 6000 s ceiling with the
FULL configuration. Actual: wide 74.4 s, firn_N10 353.2 s, firn_N20 1458.3 s
= 1886 s (+18% over projection; per-chunk compile). Total incl. pilots
~32.4 min.

## Results (metrics.json)

- surface_pick_alignment: median 0.00 frame bins (p90 1.0, offset -3.0
  bins) PASS; bed_alignment median 4.25 bins vs BedMachine-vs-pick input
  floor 3.94 (threshold 8.9) PASS, offset +8.2 bins.
- Bed depth at site: BedMachine 2600 m vs pick-derived 2597 m.
- Nadir depth-power at the closest trace (dB rel own surface peak, bands
  5-20 / 20-60 / 60-120 m): measured -20.4 / -19.1 / -34.4; surface+bed
  -38.7 / -57.2 / -64.6; N=10 -23.9 / -30.9 / -49.3; N=20 -23.3 / -30.8 /
  -41.8. Profile dB correlation (5-150 m) 0.63 -> 0.88 (N=10) -> 0.90 (N=20).

## Qualitative findings

1. Surface+bed alone cannot reproduce the measured near-surface return: it
   decays ~20-38 dB below the measured level within 60 m of the surface.
   Adding B26 firn layers recovers the measured MORPHOLOGY: a near-surface
   shoulder at ~-20 dB and a slow decay through the firn column.
2. N=10 (13.2 m spacing, resolved vs the 4.4 m in-firn chirped resolution)
   shows a discrete echo train with deep inter-layer nulls -- visibly banded
   in the radargram panels; N=20 (6.25 m, marginally resolved) is smoother
   and raises the 60-120 m band by ~7 dB, tracking the measured profile
   better (r 0.90). Consistent with the firn investigation's
   resolved->unresolved transition; per that study the levels are NOT
   converged in N (3-6 dB per doubling) and random placement (the physical
   case) shifts levels vs equal placement -- morphology comparison only.
3. Both firn sims sit ~8-12 dB below the measured 20-100 m band and drop to
   the surface+bed floor beyond the core's 119.66 m depth range, where the
   measured profile keeps decaying smoothly (real stratigraphy continues;
   the model's layer stack simply ends).
4. Measured-vs-sim processing asymmetry stands (f-k SAR + 11 looks vs
   unfocused per-trace raw at 103 m spacing): structure/relative levels are
   the claim, not texture or absolute calibration.

# CSARP_qlook (unfocused) added: the focusing hypothesis, tested (2026-07-28)

Hypothesis under test: the residual mid-band deficit of the sims is (partly)
an artefact of comparing our UNFOCUSED sims against the SAR-FOCUSED
CSARP_standard product. `CSARP_qlook` -- pulse compression + presums, NO SAR
focusing -- is the like-for-like target, so the tool now loads and compares
against BOTH products.

## How qlook loaded / what had to be aligned

`load_frame(SEASON, FRAME_ID, data_product="CSARP_qlook")` works unchanged
(cache `outputs/cache/frame_2019_Greenland_P3_20190418_01_009_CSARP_qlook.nc`,
15 MB). Differences vs CSARP_standard, and how each is handled:

- FAST TIME: IDENTICAL grid -- t0 = 0.26667 us, dt = 16.667 ns, 3044 samples
  (asserted and recorded as `measured_products.qlook.same_fast_time_grid`).
  No resampling or re-anchoring needed; the radargram panels reuse the same
  windowing code path.
- SLOW TIME: 1265 traces vs standard's 3335 (~39 m vs ~15 m posting). Each
  product gets its OWN sub-frame, own closest-approach trace, own along-track
  axis (`sub_frame` is product-agnostic), and its own panel extent.
- GAIN: qlook is ~11.2 dB hotter in raw amplitude (99.9th pct over the frame)
  -- different presum/multilook normalization. This cancels exactly, because
  every depth profile is already normalized to its OWN surface peak. Radargram
  panels get PER-PRODUCT colour limits (99.5th pct over the sim window), which
  they already did per-panel.
- Layer picks: `load_bottom_pick` is keyed on season/frame and interpolates
  onto whatever slow_time grid it is given, so the qlook panel carries the
  same Surface/Bottom picks.

## Result: the two measured products are nearly identical in depth-power

Nadir depth-power at the closest-approach trace, dB rel own surface peak
(medians; `run_config.json:band_levels_db_rel_surface`):

| profile            | 5-20 | 20-60 | 60-120 | **20-70** | **80-120** |
|--------------------|------|-------|--------|-----------|------------|
| measured standard  |-20.4 | -19.1 | -34.4  | **-20.2** | **-36.2**  |
| measured qlook     |-21.5 | -19.4 | -32.1  | **-20.5** | **-35.3**  |
| surface+bed        |-46.3 | -58.8 | -67.1  | -61.1     | -67.8      |
| firn_N10           |-23.3 | -32.9 | -50.8  | -32.8     | -60.8      |
| firn_N20           |-25.6 | -32.4 | -43.4  | -34.1     | -45.2      |
| firn_N40           |-28.5 | -37.1 | -45.4  | -37.3     | -48.0      |
| firn_N80           |-26.5 | -32.8 | -39.6  | -33.1     | -42.3      |
| firn_N40 rand s0/1/2 | -  | -     | -      | -33.7 / -34.1 / -34.2 | -50.6 / -46.5 / -47.7 |

qlook MINUS standard: **+0.34 dB** in 20-70 m, **+0.88 dB** in 80-120 m;
profile Pearson r (5-200 m) = **0.995**. The two products are the same curve
to within a dB over the entire firn column (see depth_profile.png: the dashed
grey qlook trace sits on top of the solid black standard trace).

Profile correlation of each run against both products (metrics.json
`profile_correlation.corr_standard_* / corr_qlook_*`):

| run | r vs standard | r vs qlook |
|-----|---------------|------------|
| surface+bed | 0.677 | 0.655 |
| firn_N10 | 0.890 | 0.887 |
| firn_N20 | 0.946 | 0.943 |
| firn_N40 | 0.949 | 0.945 |
| firn_N80 | 0.933 | 0.934 |
| firn_N40_s0/s1/s2 | 0.943 / 0.948 / 0.924 | 0.946 / 0.950 / 0.928 |

## Verdict: SAR focusing does NOT explain the mid-band gap

The mid-band (20-70 m) sim deficit is essentially unchanged when the
comparison is moved to the unfocused product (metrics.json
`band_delta_vs_measured`):

| run | vs standard | vs qlook | change |
|-----|-------------|----------|--------|
| firn_N10 | -12.7 dB | -12.3 dB | +0.34 |
| firn_N20 | -14.0 | -13.6 | +0.34 |
| firn_N40 | -17.1 | -16.8 | +0.34 |
| firn_N80 | -12.9 | -12.6 | +0.34 |
| firn_N40 random (3 seeds) | -13.5 / -13.9 / -14.0 | -13.2 / -13.5 / -13.7 | +0.34 |
| surface+bed | -40.9 | -40.5 | +0.34 |

Deep band (80-120 m) moves the other way by 0.9 dB (qlook is the SLIGHTLY
weaker reference there, so the deficits grow: N80 -6.1 -> -7.0 dB, N20
-9.0 -> -9.9 dB).

So the focusing-vs-unfocused asymmetry is worth **~0.3 dB, not ~8-10 dB** --
the hypothesis is rejected. The mechanism is straightforward in hindsight:
the depth profile is a RATIO to the trace's own surface peak, and f-k SAR
focusing raises the surface and the (equally specular, equally along-track
coherent) firn layers by nearly the same factor, so the ratio survives
focusing almost untouched. Focusing changes texture, along-track resolution
and off-nadir clutter rejection -- all visible in radargram panels 1 vs 2,
where the qlook panel is visibly grainier with less along-track continuity --
but not the nadir layer-to-surface ratio that the band levels measure.

Honest caveats on this test:
- qlook is not a perfect analogue of "one raw unfocused trace": it still
  carries incoherent presum averaging, which suppresses speckle much as
  multilooking does. It removes SAR FOCUSING from the comparison, which is
  what the hypothesis was about, but it does not remove all averaging.
- Single trace, single frame. The bands are medians over a 5 m-smoothed
  profile at one closest-approach trace per product (the two products' own
  closest traces, ~6 m and ~88 m from the borehole respectively).
- The remaining 12-17 dB mid-band deficit therefore still wants a physical
  explanation: unconverged N (3-6 dB/doubling per the firn investigation),
  point-sampled rather than volume-integrated permittivity contrasts, no
  volume scatter, and no small-scale (sub-32 m DEM) layer roughness.

## Bookkeeping: wide surface+bed run re-simulated at the recorded config

The cached `runs/wide_surface_bed.npz` had drifted to a 100-trace scene while
all seven firn runs are the pilot-shrunk 60-trace / 4-chunk configuration
recorded in `run_config.json`. `--report-only` now reads that file
(`_recorded_cfg`) instead of the module defaults, so re-assembly uses the
configuration the cache was built with; the wide run was re-simulated once at
60 traces (45.6 s -- the only simulation this touched, no firn run re-ran).
Consequence: the levels and correlations above supersede the earlier section's
values (e.g. profile r for N=20 0.90 -> 0.946, N=80 0.880 -> 0.933; the
surface+bed 5-20 m level -38.7 -> -46.3 dB). Gates still pass on the
re-assembled run: surface leading edge median 0.00 frame bins, bed nadir
median 3.9 bins vs input floor 4.0, firn seam 1.02e-3.

## Rough-firn-layer runs (2026-07-28): roughness hypothesis also NEGATIVE at measured values

Two runs with per-layer Gerekos 2023 roughness depth-interpolated from the
digitized C&S 2020 Fig 11 inversions (surface/bed smooth; equal-placement
N=40): firn_N40_rough_mcords (sigma 2.68-5.50 cm, l 2.53-3.49 m; 5554.4 s)
and firn_N40_rough_ar (sigma 1.51-2.92 cm, l 1.00-2.58 m; 5488.7 s).
Roughness cost only ~4% wall over smooth N=40 (5331 s).

Result: mid-band (20-70 m) gain vs smooth firn_N40 is +0.73 dB (mcords) /
+0.76 dB (ar) -- against a ~17 dB deficit vs measured. Correlations
essentially unchanged (0.9485/0.9484 vs 0.9491 smooth). At sigma ~ 0.03-0.05
lambda_local the diffuse D_Phi contribution barely exceeds the coherent
attenuation exp(-sigma^2 K^2) it replaces at nadir. The sub-wavelength
LAYER-roughness hypothesis, at the roughness values C&S 2020 actually
inverted, does NOT close the gap.

Hypothesis scoreboard for the ~17 dB mid-band deficit: equal-vs-random
placement REJECTED; SAR focusing REJECTED (+0.34 dB); measured layer
roughness REJECTED (+0.7 dB). Leading remaining candidate: the LAYER
CONTRAST AMPLITUDES. We point-sample the Kovacs permittivity at N depths,
which discards the mm-scale density variance the core actually has; C&S
2020's own 1-D model needed the full-resolution profile to reproduce the
plateau level. Next experiment: derive each model layer's effective
reflection coefficient by aggregating the full-resolution profile between
layer boundaries (e.g. thin-film transfer matrix or RMS contrast), instead
of sampling epsilon pointwise. Geometry (conformal smooth copies) is now
verified to matter little at measured roughness; amplitude statistics are
the untested axis.

2026-07-28 follow-up: ranked hypothesis list with a quantitative 1-D
transfer-matrix budget (full-res vs point-sampled reflectivity; the ledger
closes the 17 dB) in `claude_notes/b26_gap_hypotheses.md` (calc:
`claude_notes/b26_contrast_calc.py`).

## H1 effective-contrast run (2026-07-29): CONFIRMED — the gap closes to ~4 dB

firn_N40_h1eff (segment-TMM effective dielectric contrasts replacing
point-sampled eps; identical N=40 geometry; 5345 s, same cost as smooth):

- 20-70 m fair-metric (mean power, all traces): -33.29 -> -21.95 dB rel
  surface = +11.34 dB, vs the 1-D predicted +11.3. Remaining gap vs
  measured: 3.9 dB (standard) / 3.6 dB (qlook), down from 15.2/14.9.
- Old median metric: band delta -17.13 -> -3.80 dB.
- Correlation vs measured RISES to 0.954 (std) / 0.955 (qlook) — best of
  all 11 runs.
- Deep band (80-120 m) now overshoots measured by ~2-3 dB (was -12 dB
  under) — plausibly the |r|-magnitude matching discarding intra-segment
  phase, or real deep-strata decorrelation nature has and we don't.
- The coherent-realization deficit is UNCHANGED at -4.8 dB (realized
  -21.95 vs 1-D expectation -17.2) — exactly as the H2 ledger predicted.
  It is now the whole remaining mid-band story.

Scoreboard final: placement REJECTED / focusing REJECTED (+0.3) /
roughness REJECTED (+0.7) / H2 metric asymmetry PARTIAL (+1.9) /
**H1 contrast sampling CONFIRMED (+11.3)**. Remaining open physics:
the ~5 dB coherent-realization deficit (3-D facet sum under-realizing the
1-D reflectivity — candidates: finite strip truncation of the Fresnel
annuli, conformal-copy lateral coherence, facet-scale phase decorrelation
across the refraction chain), and the ~2-3 dB deep-band overshoot.
Point-sampled permittivity should be considered deprecated for firn
simulation; effective segment contrasts are the way.

## Eff-contrast N-ladder with 15 dB/km firn attenuation (2026-07-29)

Four runs (--only, other caches flagged stale): N=5/10/20/40 h1eff.

| N | corr (std) | fair 20-70 m | fair 80-120 m | old-metric delta 20-70 |
|---|---|---|---|---|
| 5 | 0.758 | -22.14 | -30.83 | -16.57 |
| 10 | 0.907 | -25.06 | -31.46 | -10.38 |
| 20 | **0.963** | -24.59 | -32.95 | -7.38 |
| 40 | 0.960 | -23.46 | -32.91 | -5.06 |
| measured std/qlook | 1 / 0.995 | -18.06 / -18.40 | -31.75 / -32.12 | 0 |

Findings:
- **Correlation plateaus at N=20** (0.963 vs 0.960 at N=40; predicted knee
  N~27 from spacing-vs-range-cell). N=80 is retired; N=20 (22 min) is the
  efficient standard; N=40 (89 min) for flagship figures.
- **Fair-metric level is N-insensitive from N>=10** (within 1.6 dB),
  confirming the 1-D conservation argument in 3-D. The old median metric's
  apparent monotonic improvement with N (-16.6 -> -5.1) is the
  sparse-profile median artifact - one more reason mean-power is the metric.
- **With attenuation the deep band is now essentially exact**: N40
  -32.91 vs measured -31.75/-32.12 (old-metric delta -0.03 dB). The
  attenuation costs ~1.5 dB mid-band, putting the fair mid-band gap at
  5.4 dB - the coherent-realization deficit, the single remaining open
  physics item, now cleanly isolated.
- N=5 shows the sparse pathology in both metrics (2 interfaces in band).

Model state after this arc: shape r = 0.96, deep band exact, mid band
-5.4 dB (realization deficit). Started the week at r = 0.68 (surface+bed
only) and a 17 dB unexplained gap.

## Peak-centered layer placement (2026-07-31): NEGATIVE for shape, and it kills the sparse-median artifact

Question: with N too small to tile the core, does putting the interfaces on
the bright horizons beat a uniform lattice? Two runs, `--only`
(`firn_N10_h1eff_peaks` 329 s, `firn_N20_h1eff_peaks` 1308 s), everything
except the interface DEPTHS identical to the uniform h1eff runs.

### Placement algorithm

Contrast density = the **coherent** reflectivity envelope: the Born reflection
phasors of the raw 1 mm profile, `r_j exp(2i*phi_j)`, summed in a sliding
in-firn range cell (4.418 m), magnitude taken (`FirnCore.contrast_density`).
Chosen over the phase-blind `|d eps/dz|` because it is the same quantity the
segments actually carry: a thick smooth-gradient zone has a large gradient but
its strata cancel at the Bragg wavenumber (0.47 m) and it reflects nothing at
195 MHz. `FirnCore.peak_depths` then takes the N most prominent peaks with
(a) min separation `min(range cell, 0.6*span/N)` = 4.42 m at N=10, 3.56 m at
N=20, and (b) a max-gap repair: while any gap exceeds 1.5x the uniform
spacing, swap the weakest selected peak for the strongest one inside that gap
(swapped-in peaks pinned, so the repair cannot oscillate).

The gap repair is NOT cosmetic. Pure prominence ranking abandons the weakly
contrasted deep firn: at N=10 it left a 30 m hole below 63 m, collapsed 41 m
of profile onto one interface at 93 m, and drove the synthetic eps to 3.47
(above solid ice). Repaired: eps 1.854-3.212 at N=10, 1.697-3.240 at N=20.

Chosen depths (m): N=10 `6.2 11.1 25.5 36.3 44.2 57.0 69.2 81.2 93.4 102.5`;
N=20 adds `1.5 10.1 13.9 21.3 31.3 50.9 63.4 67.2 74.4 85.1 109.2 117.8`.
Segment top edge anchored at z_top = 1.0 m (`segment_reflectivity(top=)`) so
the COVERED EXTENT matches the uniform stack's -- without it the peaks stack
simply drops the 1-6.2 m region and the comparison is unfair.

### 1-D gate: the "conservation" premise is false, and it matters

Segment aggregation is **not** boundary-invariant. Shifting the uniform N=10
segment edges alone -- same interface depths, same everything -- moves the
total `sum|r|^2` over 2.5 dB (-22.49 / -20.03 / -21.41 / -22.24 / -21.22 dB at
edge offsets 0 / 0.1 / 0.25 / 0.4 / 0.5 x spacing). The aggregate is a
COHERENT sum, so where you cut relative to an in-phase horizon changes it.
Peak-centering cuts between horizons instead of through them and therefore
captures more: N=10 total -19.85 dB vs uniform -22.49 dB (+2.6 dB).

So a 1-2 dB band-level difference between placements is expected, not a bug.
Coherent pulse-weighted 1-D band levels (dB rel surface):

| stack | 20-70 m | 80-120 m | sum abs(r)^2 | shape r vs full-res |
|---|---|---|---|---|
| full-res 1 mm | -17.37 | -24.27 | - | 1.000 |
| uniform N10 | -19.15 | -24.12 | -22.49 | 0.484 |
| peaks N10 | -20.02 | -23.79 | -20.74 | 0.540 |
| uniform N20 | -18.71 | -25.03 | -20.12 | 0.774 |
| peaks N20 | -18.90 | -23.50 | -19.25 | 0.689 |

Peaks-minus-uniform: -0.87 / +0.34 dB at N=10, -0.20 / +1.53 dB at N=20 --
inside the intrinsic boundary sensitivity. 1-D SHAPE prediction: peaks better
at N=10 (0.540 vs 0.484), worse at N=20 (0.689 vs 0.774).

### 3-D result

| run | corr std | corr qlook | med@j0 20-70 | med@j0 80-120 | meanP 20-70 | meanP 80-120 |
|---|---|---|---|---|---|---|
| measured std | 1.000 | 0.995 | -20.17 | -36.19 | -18.06 | -31.75 |
| measured qlook | 0.995 | 1.000 | -20.51 | -35.30 | -18.40 | -32.12 |
| N10 uniform | 0.9069 | 0.9096 | -30.55 | -43.81 | -25.06 | -31.46 |
| **N10 peaks** | 0.8994 | 0.9028 | -27.15 | -42.97 | -25.61 | -31.75 |
| N20 uniform | **0.9627** | **0.9652** | -27.55 | -38.78 | -24.59 | -32.95 |
| **N20 peaks** | 0.9564 | 0.9590 | -26.62 | -37.23 | -24.32 | -30.72 |

Correlation gain (peaks minus uniform): **-0.0075 / -0.0067** at N=10,
**-0.0064 / -0.0062** at N=20 (standard / qlook).

Findings:
- **The N=10 prediction is NOT confirmed.** Peak placement does not improve
  the depth-profile correlation at either N; it is marginally (0.006-0.008)
  worse at both. The 1-D gate's shape prediction (better at N=10) did not
  survive into 3-D either.
- **The N=20 prediction IS confirmed**: nothing material changes once uniform
  already beats the range cell (corr -0.006, fair mid-band +0.27 dB). The one
  real change is the deep band, +2.2 dB, which moves it from 0.8 dB below
  measured to 1.0 dB above -- no better, just differently wrong.
- **Peak placement does kill the sparse-median artifact.** The median-metric
  minus mean-power gap at N=10 collapses from 5.49 dB (uniform) to 1.54 dB
  (peaks), and the median-metric mid-band gains +3.40 dB, with essentially NO
  change in actual power (fair mid-band -25.06 -> -25.61). Spreading the same
  reflectivity onto horizon-centred depths fills the interference nulls that
  the uniform comb generates. This is the H2 mechanism seen from the other
  side, and it is a good reason to keep reporting the mean-power estimator.
- **Why the null result is physical**: the B26 firn reflectivity is a
  near-continuum (the C&S plateau), not a sparse set of bright horizons. A
  uniform tiling represents a continuum better than a top-N peak selection,
  which necessarily leaves wide segments whose reflectivity is collapsed onto
  a point. Peak placement would be expected to pay off for a genuinely sparse
  stratigraphy (isolated melt layers / ice lenses), not for polar firn.

**Standard is unchanged: uniform equal placement at N=20.** Deliverables:
`outputs/b26_comparison/placement_profiles.png` (6-curve depth profile) and
`placement_radargrams.png` (firn-zone sections, 2 measured + 4 sim panels).
