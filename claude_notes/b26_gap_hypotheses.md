# Ranked hypotheses for the remaining ~17 dB B26 mid-band deficit (2026-07-28)

Target: firn_N40 sits -16.8 dB below measured qlook (-17.1 vs standard) in the
20-70 m band, dB rel own surface peak, with shape r ~ 0.95. Rejected already
(<1 dB each): random placement, SAR focusing (+0.34), measured layer roughness
(+0.73). This note ranks what is left, with numbers from an actual 1-D
calculation on the B26 fixture (`claude_notes/b26_contrast_calc.py`, run with
`uv run python`; transfer-matrix method identical to C&S 2020 Sec IV-C /
~/Documents/clutter `transfer_matrix.py`).

## The 1-D calculation that frames everything

Full-resolution (1 mm) B26 density -> Kovacs eps -> transfer-matrix effective
|r|^2 per 4.42 m in-firn range cell (hann-broadened MCoRDS res), normalized to
the air->firn Fresnel power (-15.8 dB). Band means over 20-70 m:

| profile representation                    | band level (dB rel surface) |
|-------------------------------------------|------|
| full-res raw 1 mm                         | **-16.7** |
| full-res, 0.1 m pre-smoothed (our pipeline's input) | -18.2 |
| full-res, 0.3 / 0.5 / 1.0 / 3.0 m smoothed | -26.1 / -36.7 / -40.0 / -47.1 |
| N=10 / 20 / 40 / 80 point-sampled interfaces (current sim inputs) | -22.6 / -26.0 / **-28.3** / -26.6 |
| N=20 / 40 / 80 segment-aggregated (TMM |r| of full-res between layer midpoints) | -18.9 / **-17.2** / -17.5 |

Sanity: with C&S's 6.0 m bins the raw profile gives -17.0 vs their digitized
Fig 9b simulated curve at -20.0 (different surface-bin normalization; same
ballpark). Measured is -20.2/-20.5. So the full-res profile SUPPLIES the whole
plateau; the N-point-sampled version does not, by ~11.5 dB.

The smoothing scan localizes the missing power: the Bragg wavelength is
lam_firn/2 ~ 0.47 m, and the reflectivity collapses between 0.1 m and 0.5 m
smoothing (-18 -> -37 dB). The contrast that matters lives at **0.1-0.5 m
vertical scales** -- exactly what point-sampling at 1.5-13 m spacing discards.
(It is NOT densitometer noise: smoothing raw -> 0.05 m costs only 0.5 dB, so
white mm-scale noise is a minor contributor.) Two-way transmission loss to
45 m is 0.57 dB even at full contrast -- stronger contrasts do not eat
themselves.

### The ledger closes

- 1-D full-res minus 1-D N40-point-sampled: **+11.5 dB** (contrast sampling)
- sim firn_N40 realized (-37.3) minus its own 1-D expectation (-28.3):
  **-9.0 dB** (realization/extraction offset; same offset -10.2/-8.1/-6.5 dB
  at N=10/20/80 -- systematic, not N40-specific)
- measured (-20.2) minus 1-D full-res (-16.7): **-3.5 dB** (1-D model
  overshoot vs nature; bounds any lateral-decoherence effect at <=3.5 dB)

11.5 + 9.0 - 3.5 = 17.0 dB = the observed gap. Two mechanisms, H1 and H2
below, plausibly account for essentially all of it.

---

## H1 (rank 1): layer contrast amplitude -- point sampling discards Bragg-scale reflectivity

**Mechanism.** Each real 4.4 m range cell contains the coherently-summed
reflection of hundreds of 0.1-0.5 m-scale density strata (the Fourier
component of the log-index profile at the Bragg wavenumber). We replace that
with 1-2 point-sampled Fresnel steps per cell whose contrast is set by the
smooth Kovacs trend, not the strata. Predicted lift: **+11 dB** (from -28.3
to -17.2 in the 1-D model at N=40). This is the only candidate that can
plausibly supply order-10 dB.

**Experiment.** In `tools/run_b26_comparison.py`'s layer construction, replace
`point_eps(d_i)` with a synthetic eps sequence whose successive Fresnel
contrasts equal the transfer-matrix |r| of the RAW full-res profile aggregated
between layer midpoints: `n_i = n_{i-1} * (1 -/+ r_i)/(1 +/- r_i)`, sign
chosen to track the Kovacs trend. Verified in the calc: eps stays physical
(1.69-3.15, max 0.32 off trend), reproduces the -17.2 dB band level exactly;
median segment |r| = -38.5 dB, max -13.7 dB (near-surface). Use raw density,
not the 0.1 m-smoothed pipeline profile (the pre-smoothing alone costs
1.4 dB). Run one N=40 frame sim (~90 min), compare 20-70 m band vs qlook.

**Outcomes.** Lift ~+8-12 dB (band to ~ -26 rel surface): confirmed; residual
~5 dB is then H2/H3 territory. Lift +4-8 dB: partially confirmed -- the
coherent facet sum realizes less than 1-D (interference/nulls between
conformal layers), combine with H2 diagnostics. Lift <+3 dB: something in the
kernel does not translate interface contrast to received power the way 1-D
optics says -- audit transmission/gamma handling before believing any physics
hypothesis.

## H2 (rank 2): realization/extraction offset -- the sim sits 6.5-10 dB below its OWN 1-D expectation  [TESTED 2026-07-28: PARTIALLY CONFIRMED, worth ~2 dB of gap, not ~9]

**Mechanism.** Independent of input contrasts, every sim run realizes a band
level far below the incoherent 1-D prediction for the same interfaces. Known
contributors: band metric is the MEDIAN of a 5 m-smoothed dB profile at one
trace -- for a sparse echo train with deep interference nulls median <<
mean-power (speckle alone is -1.6 dB; structured nulls between marginally
resolved equal-spaced layers plausibly several more); spreading to 45 m is
only -0.5 dB. The measured profile is smooth (median ~ mean), so this
asymmetry inflates the apparent gap. Worth **up to ~9 dB**, overlapping H1
(fixing contrasts also fills the nulls).

**Experiment (FREE -- no simulation).** From the cached per-layer complex
fields in `outputs/b26_comparison/runs/*.npz`, recompute band levels as mean
linear power over band and traces (and trace-averaged before dB), and compute
each run's own 1-D TMM expectation for its exact eps stack. Compare the
extraction variants for sim AND measured identically.

**Outcomes.** If mean-power extraction closes most of the 9 dB sim-vs-1D
offset, the honest gap is ~11-12 dB and H1's prediction should close nearly
all of it; report band levels with the symmetric metric from now on. If the
offset survives mean-power extraction, the coherent kernel is genuinely
delivering less than geometric optics predicts for unresolved stacks --
investigate before/alongside H1.

### RESULTS (2026-07-28, `claude_notes/b26_h2_metric_recompute.py`, no simulation)

Recomputed from the cached fields in `outputs/b26_comparison/runs/*.npz`, with
the tool's exact depth axis, 5 m power boxcar and per-trace own-surface-peak
normalization, changing ONLY the band estimator. `med@j0` reproduces
`run_config.json`'s `band_levels_db_rel_surface` to the last digit for all 12
rows (sanity check on the reimplementation). Measured means use all traces of
the same 10 km sub-segment (679 standard / 258 qlook); sims use their 60.

**20-70 m band, dB rel own surface peak**

| run | med@j0 (report) | meanP@j0 | med all traces | **meanP all** | 1-D expect (own stack) | residual |
|---|---|---|---|---|---|---|
| firn_N10 | -32.82 | -29.21 | -31.56 | **-28.63** | -22.62 | -6.02 |
| firn_N20 | -34.12 | -32.71 | -32.92 | **-31.88** | -25.99 | -5.88 |
| firn_N40 | -37.30 | -34.37 | -36.26 | **-33.29** | -28.28 | -5.01 |
| firn_N80 | -33.06 | -32.45 | -32.36 | **-31.76** | -26.61 | -5.15 |
| firn_N40_s0 | -33.71 | -33.30 | -33.47 | **-32.52** | -23.72 | -8.80 |
| firn_N40_s1 | -34.05 | -33.31 | -33.50 | **-32.51** | -28.81 | -3.71 |
| firn_N40_s2 | -34.21 | -33.26 | -33.14 | **-32.32** | -26.89 | -5.43 |
| firn_N40_rough_mcords | -36.57 | -34.94 | -36.02 | **-33.74** | -28.28 | -5.46 |
| firn_N40_rough_ar | -36.54 | -34.34 | -35.98 | **-33.38** | -28.28 | -5.10 |
| surface+bed | -61.05 | -58.44 | -60.05 | **-56.64** | - | - |
| measured (standard) | -20.17 | -19.58 | -19.36 | **-18.06** | - | - |
| measured (qlook) | -20.51 | -19.42 | -19.59 | **-18.40** | - | - |
| 1-D full-res TMM | | | | | -16.74 | |

**80-120 m band** (same order): firn_N40 -48.03 / -46.80 / -47.68 / **-45.58**
vs 1-D -39.11 (residual -6.47); measured -36.19 -> **-31.75**, qlook
-35.30 -> **-32.12**; 1-D full-res -22.43. firn_N10 is the outlier
(-60.78 -> -46.38, +14.4 dB): with 10 layers the deep band is a handful of
isolated echoes and the median lands in the nulls -- exactly H2's mechanism,
just at the N where it matters least.

**Headline gap, 20-70 m (measured minus firn_N40)**

| estimator | vs standard | vs qlook |
|---|---|---|
| median dB @ closest trace (report) | 17.13 | 16.79 |
| **mean power, all traces** | **15.23** | **14.89** |

**Verdict: partially confirmed, but the effect is 3-5x smaller than the doc
predicted.** The metric asymmetry is real and in the predicted direction --
the sim gains +4.0 dB going from median-of-dB at one trace to mean-power over
all traces (firn_N40: -37.30 -> -33.29) while the measured gains only +2.1 dB
-- but the differential is just 1.9 dB, not the "up to ~9 dB" the hypothesis
claimed. Decomposed for firn_N40: median->mean-power at the same trace is
+2.9 dB (vs -1.6 dB for pure Rayleigh speckle, so there IS extra null
structure, ~1.3 dB of it) and trace-averaging adds a further +1.1 dB; for the
measured the same two steps give only +0.6 and +1.5 dB, confirming the
"measured profile is smooth, median ~ mean" premise. The sim-vs-own-1-D
residual shrinks from -9.0 to **-5.01 dB** at N=40, and is systematic across
N (-6.0 / -5.9 / -5.0 / -5.2 dB at N=10/20/40/80, -3.7 to -8.8 for the random
placements), so roughly half of that offset was extraction and half is a
genuine coherent-realization deficit that mean-power does NOT explain -- the
facet sum over conformal DEM-offset layers delivers ~5 dB less than incoherent
geometric optics predicts for the same interfaces. The ledger still closes
exactly under the fair metric: 1-D contrast sampling (+11.54) + realization
deficit (+5.01) - 1-D-overshoot-vs-nature (-1.32) = 15.23 dB. **Action:
report band levels as mean power over band and traces from now on; the honest
target for H1 is ~15 dB, and even a full +11.5 dB contrast fix leaves the ~5 dB
realization residual, which is now the second-largest term and deserves its own
diagnostic** (candidate: coherent cancellation between equally-spaced conformal
layers, i.e. the same equal-placement artifact the random runs probe -- note
their residuals scatter -3.7 to -8.8 dB).

## H3 (rank 3): sim surface peak too specular -- normalization denominator too big

**Mechanism.** Everything is dB rel own surface peak. The sim surface is a
32 m-posted DEM, mirror-smooth below that scale; the real snow surface has
cm-dm micro-roughness (sastrugi), which decoheres the measured surface peak by
exp(-(2 k sigma)^2): -2.9 dB at sigma = 10 cm, -6.5 dB at 15 cm (195 MHz).
Layers were tested at their measured sigma (2.7-5.5 cm, <1 dB) but the SURFACE
has stayed smooth in every run. A too-strong sim surface peak suppresses all
sim rel-surface levels. Predicted: **+2-6 dB**, cannot close 17 alone.

**Experiment.** Add Gerekos roughness (sigma 5 / 10 / 15 cm, l ~ meters) to
the surface interface only, in the wide surface+bed run (~1-2 min each) +
report-only reassembly with existing firn fields. Confirm: all sim band levels
rise together by the exp(-(2 k sigma)^2) amount. Constrain sigma from ATM/
ICESat-2 if pursued seriously.

**Outcomes.** A few dB lift stacks with H1; if H1+H2+H3 overshoot, that
overshoot bounds the surface sigma actually consistent with the data.

## H4 (rank 4): measured-product waveform stitching / receiver gain vs depth

**Mechanism.** The 2019 frame mixes Tpd {1, 3, 10} us waveforms; the surface
peak comes from the low-gain short pulse and the firn band possibly from a
different waveform. Imperfect img_comb gain equalization, or receiver/TR
compression at the surface, biases the rel-surface ratio. qlook-vs-standard
agreement (0.995) already shows both PRODUCTS treat it identically, but both
inherit the same combine. Predicted: **0-3 dB** (CReSIS equalizes in overlap).

**Experiment (free, data-only).** Read img_comb / img_comb_weights and the
combine crossover times from `outputs/cache/mcords_2019P3_params.json` /
`Data_20190418_01_009_source.mat`; check whether the 20-70 m band and the
surface peak live in the same waveform. If single-waveform qlook images
(img_01/02/03) are available, compare the band level rel surface per waveform.

**Outcomes.** Same-waveform for surface and band: hypothesis dead. Different
waveforms with visible combine seam in the band: quantify and correct the
measured reference.

## H5 (rank 5): off-nadir diffuse surface clutter filling the firn band

**Mechanism.** Delays of the 20-70 m band map to surface annuli at ~15-30 deg
incidence (~150-300 m radius at 500 m AGL). The sim's smooth DEM produces no
diffuse scatter there. But interior-Greenland VHF backscatter at 20-30 deg is
weak, the AR (750 MHz) sees the same plateau, and r ~ 0.95 against a
density-derived model says the band is stratigraphy. Predicted: **<2 dB**.
The H3 rough-surface run tests this simultaneously (same run, look at the
band contribution of the surface layer's field). Rank low; piggyback only.

## Rejected on paper (do not spend simulations)

- **Interbed multiples**: first-order multiple carries three reflections;
  with median interface gamma ~ -40 to -47 dB, multiples sit ~80-90 dB below
  primaries. Cannot supply anything.
- **Volume scattering from grains/inclusions**: ka ~ 4e-3 for mm grains at
  195 MHz; Rayleigh sigma ~ (ka)^4 -> ~ -100 dB. Cm-scale melt features are
  also deep-Rayleigh. C&S explain both AR and MCoRDS plateaus with layered
  optics alone.
- **Cross-pol return**: nadir plane-stratified media generate no cross-pol;
  the product is co-pol.
- **Hardware range sidelobes of the surface**: would need an integrated
  sidelobe floor at -20 dB (implausible) and would not correlate at r ~ 0.95
  with the density profile shape.
- **Layer lateral coherence vs Fresnel zone** (as a MISSING-power mechanism):
  conformal DEM copies are already fully coherent over the ~28 m Fresnel
  zone; real layers can only be less coherent. This can only work AGAINST H1,
  and the measured-vs-full-res-1-D residual bounds it at <= 3.5 dB. Treated
  as H1's error bar, not a hypothesis.

## Recommended order

1. ~~**H2 extraction recompute (free)**~~ -- DONE 2026-07-28. Gap re-baselined
   at **15.2 dB** (standard) / **14.9 dB** (qlook) with mean-power-over-traces;
   only 1.9 dB of the old 17.1 was metric asymmetry, and a systematic ~5 dB
   sim-vs-own-1-D realization deficit survives. See the H2 RESULTS section.
2. **H1 synthetic-eps N=40 run (~90 min)** -- the headline experiment. It is
   the only mechanism with a demonstrated order-10 dB budget (from this
   repo's own fixture, not literature values), the implementation is a
   drop-in change to the eps list (construction verified physical), and C&S's
   Fig 9 independently shows the full-res profile reproduces the measured
   plateau. Predicted post-fix band: ~ -26 dB rel surface vs measured -20.5.
3. H3 surface-roughness wide runs (~minutes) to account for the residual.

## Post-H1 addendum (2026-07-29): deep-band overshoot resolved analytically

Complex-r gate FAILED in 1-D (claude_notes/b26_complex_r_gate.py): |r|-only
h1eff already matches the full-res deep band to 0.09 dB (at N=40 a 4.4 m
range cell holds ~1.45 layers -- nothing to decohere); realized phase moves
it +1.4 dB the WRONG way. No 3-D run wasted.

Firn attenuation bracket (analytic reweighting of the cached h1eff per-layer
fields, fair metric; alpha one-way, uniform):

| alpha (dB/km) | 20-70 m | 80-120 m |
|---|---|---|
| 0 (current) | -21.95 | -30.07 |
| 5 | -22.46 | -31.02 |
| 10 | -22.96 | -31.97 |
| 15 | -23.46 | -32.91 |
| measured std/qlook | -18.06 / -18.40 | -31.75 / -32.12 |

**alpha ~ 8-12 dB/km one-way eliminates the deep-band overshoot exactly**,
at a physically standard value (tool's own deep-ice number: 15 dB/km; the
firn_cfg docstring's "<0.2 dB one-way at 120 m" claim is ~10x low and
should be corrected). Cost: ~1 dB more mid-band deficit, leaving the
~5 dB coherent-realization deficit as the single remaining open item.
Proper fix: set a defensible temperature/density-based attenuation profile
in firn_cfg (e.g. MacGregor-style Arrhenius) and re-run; the analytic
bracket already fixes the magnitude and direction.
