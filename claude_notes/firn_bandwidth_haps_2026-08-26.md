# Firn plateau vs bandwidth at B26, airborne and HAPS (2026-08-26)

Branch `firn-bandwidth-haps` (on top of `waveform-explicit-chirp`). Tool:
`tools/run_firn_bandwidth.py`; outputs `outputs/firn_bandwidth/{airborne,haps}/`
(metrics.json, profiles.png, plateau_vs_bandwidth.png, haps_arms.png, runs/).
Question: Culberg & Schroeder 2020 (Sec. VI, Fig. 14) -- the firn power
plateau falls with bandwidth and rises with centre frequency -- what does that
do to a 14 km HAPS sounder at 195/300 MHz, which the HAPS design study
(`claude_notes/haps_design_study/summary.md`) ran with NO firn layers?

## 1. Revival of the B26 validation (status: works, numbers reproduce exactly)

`tools/run_b26_comparison.py --only wide_surface_bed,firn_N20_h1eff --force`
on this branch (default analytic pulse; cache staged from
`outputs_backup_20260818/b26_comparison`, frame/qlook/bottom-pick files copied
into the worktree's `outputs/cache`). No code change was needed.

| run | July 2026 wall | now | band levels (meanP, 5-20/20-70/80-120 m) |
|---|---|---|---|
| wide_surface_bed (1.76M facets, 60 tr) | 45.6 s | 18.1 s | identical to 0.01 dB |
| firn_N20_h1eff (4 chunks x 82k facets x 20 layers) | 1308 s | 1736 s (*) | -17.90 / -24.59 / -32.95, identical to 0.01 dB |

(*) shared the CPU with a second simulation for its whole duration; the
kernel-only wide run is 2.5x faster, consistent with the runtime study.
Profile correlation firn_N20_h1eff 0.9627 (unchanged), measured qlook
20-70 m -18.4 dB, i.e. the known ~6 dB mid-band deficit stands. The
2026-08-24 kernel changes (facet windowing, fused bed path) and the chirp
branch (construction opt-in) leave the B26 result bit-for-bit at the reported
precision. The other 14 cached runs show as `cache-stale` only because
`--only` marks every non-simulated run that way; their metrics are unchanged.

## 2. Method for the bandwidth study

The kernels do not depend on the waveform (pulse compression is a post-kernel
fast-time convolution, `waveform.py`), so the tool simulates the DELTA field
once per geometry x carrier at dt = 16.667/8 = 2.083 ns (alias 195 MHz at
f0 = 195, 180 MHz at f0 = 300: every B <= 150 MHz is alias-free, asserted per
pulse) and applies every pulse in post-processing. Firn stack = the validated
standard (uniform N = 20, effective-contrast eps from the raw 1 mm B26 core at
the run's own carrier wavelength, 15 dB/km), field-summed with the wide
surface + BedMachine-bed run without the firn run's own surface (the B26
seam construction). The airborne case is the reference configuration (10 km,
60 traces, +-3 km wide / +-600 m firn strip, beta 0.5 facets = 10.67 m, 7-el
0.5-lambda array with nav roll, smooth interfaces); HAPS is described in Sec. 4.

1-D reference (`oned_profile`): the raw B26 index profile (1 cm resampled)
as a normal-incidence transfer-matrix reflection spectrum r(f) across each
band, Hann-weighted, inverse-transformed -> depth-power profile rel its own
surface peak. This is C&S's 1-D layered-dielectric model at this site with a
real pulse (their Fig. 14 machinery), checked against a single buried
reflector (peak -29.9 vs -29.6 dB expected, B-independent) and a white-noise
profile (plateau -5.5 / -6.4 dB per 10->30->100 MHz, i.e. 1/B).

## 3. Airborne bandwidth demonstration (195 MHz, frame as flown, T = 10 us)

Plateau = mean linear power in the depth band, dB rel own surface peak.
Two estimators: `meanP` (the B26 tool's mean-of-ratios over traces) and the
per-trace MEDIAN of band/peak, which is the cleaner one for a scaling law
(meanP is dominated by the traces with the weakest surface peaks).

| pulse | 5-20 | 20-60 | 60-120 | **20-70 meanP** | 40-100 | **20-70 median** | surf peak abs | firn band abs | midcol |
|---|---|---|---|---|---|---|---|---|---|
| 195/10 | -7.3 | -22.6 | -24.6 | -21.5 | -22.2 | -21.9 | -69.7 | -91.5 | -50.8 |
| 195/30 (as flown) | -18.0 | -25.5 | -28.2 | -24.6 | -25.4 | -25.7 | -70.8 | -96.5 | -54.5 |
| 195/60 | -19.8 | -26.2 | -29.5 | -25.5 | -26.4 | -28.2 | -71.2 | -99.5 | -56.4 |
| 195/97 | -19.5 | -26.1 | -29.5 | -25.5 | -26.4 | -29.7 | -72.0 | -101.5 | -57.8 |
| surface+bed only, 30 | -- | -- | -- | -56.7 | -- | -- | -- | -- | -- |
| measured standard 195/30 | -15.3 | -18.1 | -24.8 | -18.1 | -20.1 | | | | |
| measured qlook 195/30 | -15.7 | -18.4 | -25.1 | -18.4 | -20.4 | | | | |

(explicit-chirp construction at 30 and 97 MHz: identical plateau to 0.1 dB;
"abs" columns are sim units, medians over traces.)

- The firn band power itself follows 1/B exactly: -10.0 dB from 10 to 97 MHz
  vs -9.9 expected (the N = 20 stack is a fixed set of reflectors; the pulse
  energy scales as 1/B). The surface peak also drops 2.3 dB over the same
  range (the DEM-spread surface echo is resolved into more range cells), so
  the plateau REL SURFACE falls 7.8 dB (median) / 4.0 dB (meanP) from 10 to
  97 MHz, and 4.0 dB (median) going from the flown 30 MHz to 97 MHz.
- Mid-column (firn-dominated in this window) falls 7 dB from 10 to 97 MHz.
- The 3-D sim sits ~6 dB below the measured plateau at 30 MHz (the known
  H1/H2 residual), so the sim is a relative, not absolute, demonstration.

1-D full-resolution TMM (the C&S model at this core), 20-70 m / 40-100 m:

| B (MHz) | 5 | 10 | 20 | 30 | 60 | 100 | 150 |
|---|---|---|---|---|---|---|---|
| 195 MHz 20-70 | -9.0 | -11.8 | -14.1 | -15.1 | -16.4 | -16.6 | -17.0 |
| 195 MHz 40-100 | -9.5 | -12.1 | -14.8 | -16.1 | -17.7 | -17.8 | -18.2 |
| 300 MHz 20-70 | -7.5 | -12.2 | -15.2 | -14.1 | -10.6 | -10.2 | -10.5 |
| 300 MHz 40-100 | -10.1 | -13.7 | -16.4 | -16.0 | -13.3 | -12.6 | -12.8 |

The 1-D plateau at 195/30 (-15.1 dB) lands on the measured value (-18.4) and
C&S's per-cell aggregate (-17 dB, claude_notes/b26_gap_hypotheses.md) much
better than the 3-D stack does. Its bandwidth dependence is WEAKER than 1/B:
-3.3 dB from 10 to 30 MHz, only -1.5 dB from 30 to 100 MHz at 195 MHz, and at
300 MHz the plateau RISES again above 30 MHz. The reason is that this single
core's reflectivity spectrum is not flat over the band: widening the band
averages in Bragg wavenumbers away from the carrier (0.27-0.37 m at
300 +- 50 MHz) where the B26 profile carries more contrast, and that gain
cancels the 1/B range-cell shrinkage. C&S's Fig. 14 is an ensemble over six
cores, which smooths this; at one core the bandwidth lever is worth ~3 dB
(10 -> 30 MHz) and then saturates, and the FREQUENCY lever (their Fig. 14a)
is the strong one only below ~200 MHz here: at 195 vs 300 MHz the 30 MHz
plateau is within 1 dB, at 100 MHz the 300 MHz plateau is 6 dB higher.

## 4. HAPS geometry

Synthetic pass: the B26 frame's horizontal track (10 km, 24 traces) at a
CONSTANT ellipsoidal height of 14 000 m (median 11 419 m AGL; surface
~2.58 km), BedMachine bed 2601 m below the surface (median), bed delay 30.9 us
below the surface peak. Instrument: 8-element Hann-tapered (TX and RX)
cross-track array on a 10 m span (`hd_f300_n8_hann.yaml` weights; spacing
0.93 lambda at 195 MHz, 1.43 lambda at 300 MHz), explicit-chirp construction,
T = 8 us, Hann compression window. Physics as the design study: surface
sub-facet roughness sigma 4.9 cm / l 2.98 m, bed roughness 0.1 m / 0.886 m,
grazing fix s_eff 0.05; PLUS the firn stack on the FULL +-12.5 km wide DEM
with C&S Fig. 11 (MCoRDS inversion) roughness on every internal firn
interface. Wide reach +-12.5 km covers surface arrivals to the bed delay
+ 3 us (41 deg off nadir at the bed delay). Facets 50.4 m (195) / 40.6 m (300)
(beta 0.5 Fresnel rule, deepest firn layer binding).

Windows (config/analysis.yaml): bed window bed -0.5 -> +1.5 us, mid-column
surf +1.0 -> bed -0.5 us; all rel the trace's own TOTAL surface peak, medians
over traces; "bed/clutter" = bed arm over (surface+firn) arm in the bed
window (the design study's `bed_visibility`), "bed/surf" the same against
the surface arm alone (what a firn-free run would report).

| f0/B (MHz) | plateau 20-70 (meanP) | 40-100 | no-firn 20-70 | surf arm @bed | firn arm @bed | bed arm | **bed/clutter** | bed/surf only | firn/surf @bed | midcol tot / surf / firn |
|---|---|---|---|---|---|---|---|---|---|---|
| 195/10 | -17.1 | -19.7 | -19.0 | -163.8 | -137.1 | -102.5 | **+35.3** | +60.7 | +26.5 | -45.3 / -48.9 / -48.8 |
| 195/30 (a) | -20.1 | -22.5 | -23.0 | -166.8 | -140.8 | -106.0 | **+35.3** | +61.6 | +26.4 | -50.1 / -54.7 / -52.0 |
| 195/60 | -20.9 | -23.4 | -24.0 | -169.9 | -144.0 | -108.9 | **+35.4** | +62.0 | +26.5 | -52.9 / -58.5 / -54.0 |
| 195/97 (b) | -21.1 | -23.6 | -24.4 | -172.5 | -146.3 | -110.7 | **+35.5** | +62.3 | +26.5 | -54.9 / -61.2 / -55.8 |
| 300/10 | -13.2 | -17.4 | -20.2 | -211.7 | -148.8 | -103.9 | **+44.3** | +106.9 | +62.7 | -44.7 / -48.7 / -46.3 |
| 300/30 (d) | -16.3 | -20.1 | -25.0 | -215.5 | -152.6 | -108.5 | **+44.4** | +107.0 | +62.5 | -50.8 / -56.7 / -52.0 |
| 300/60 | -17.0 | -20.9 | -25.8 | -218.8 | -155.8 | -111.9 | **+44.5** | +107.0 | +62.7 | -54.5 / -62.1 / -55.0 |
| 300/100 (c) | -17.5 | -21.4 | -25.9 | -220.8 | -158.0 | -113.8 | **+44.6** | +107.2 | +62.7 | -57.1 / -65.8 / -57.6 |
| 300/150 | -17.9 | -21.8 | -25.9 | -222.8 | -160.3 | -116.0 | **+44.5** | +107.2 | +62.8 | -59.2 / -68.7 / -59.7 |

(300/150 is alias-legal on the 2.083 ns grid used here, unlike the design
study's 8.33 ns grid.) Absolute firn band power (sim units, 20-70 m):
195: -115.6 / -120.8 / -123.8 / -125.9 dB at 10/30/60/97 MHz (-10.3 dB, 1/B
gives -9.9); 300: -113.9 / -118.3 / -121.3 / -123.4 / -125.1 at
10/30/60/100/150 (-11.2 dB, 1/B -11.8). Figure `haps/haps_arms.png` shows the
four arms vs delay below the surface for every pulse.

### Answers

(i) Plateau at HAPS altitude: 300 MHz sits 3.5-4 dB above 195 MHz at every
bandwidth (the C&S frequency effect, carried by the carrier-wavelength
segment contrasts); bandwidth buys 4-5 dB from 10 to ~100 MHz and ~1 dB from
30 to 100 MHz (meanP estimator). In the HAPS geometry the surface's own
diffuse near-nadir scatter fills the 20-70 m window at -19 to -26 dB (the
"no-firn" column, absent at 500 m AGL where it was -57 dB), so at 195 MHz the
firn is only about half of the plateau power; at 300 MHz it dominates.

(ii) Bed-window clutter at 14 km is FIRN-DOMINATED in this model: the firn
arm at the bed delay is 26 dB above the surface arm at 195 MHz and 63 dB
above it at 300 MHz. Bed over (surface + firn) clutter: +35 dB at 195 MHz,
+44 dB at 300 MHz -- versus +61 / +107 dB that a firn-free run reports. The
firn layers therefore cost 26 / 63 dB of bed visibility relative to the
design study's surface-only picture (the design study's nominal-roughness
numbers were never clutter-limited at these levels, but its l = 1 m
pessimistic rows were, and those did not contain this term).

**How much does bandwidth matter at the basal interface at 14 km?** Nothing:
30 -> ~100 MHz changes bed-over-clutter by +0.2 dB at 195 MHz (35.3 -> 35.5)
and +0.2 dB at 300 MHz (44.4 -> 44.6); 10 -> 150 MHz likewise < 0.3 dB. Both
arms in the bed window are incoherent sums of many scatterers and scale with
the pulse energy (1/B) together, exactly as the design study found for the
surface arm alone; adding the firn arm does not change that. Bandwidth is a
resolution/plateau lever only. Frequency does matter: 195 -> 300 MHz buys
+9 dB of bed-over-clutter (the firn arm drops 12 dB rel the surface peak, the
bed arm 2.5 dB), consistent in direction with C&S Fig. 15(a) -- their
"increasing bandwidth to offset frequency is most effective in HF/VHF" is
about the total near-surface clutter cross-section, which here is B-invariant
relative to the bed.

Why the firn wins at the bed delay while the surface collapses: the
air-firn surface at 41 deg with a Gaussian ACF (l = 3 m, k l sin(theta) ~ 8 at
195 MHz) is exp(-(k l sin theta)^2/m)-suppressed to -164 / -212 dB, but the
same roughness on a buried layer is seen at the REFRACTED in-firn angle
(<= asin(1/n) ~ 39 deg, and sin(theta_in) = sin(theta)/n), so the layers stay
20-60 dB more visible at wide angles than the surface. This is the geometry
C&S build their Fig. 15 clutter model on (in-firn illumination angles).

(iii) Mid-column: -50 dB at 30 MHz at both carriers, falling to -55/-57 dB at
~100 MHz (bandwidth does help here, ~1/B); surface and firn arms contribute
comparably at 195 MHz, and the firn arm is within 1-2 dB of the total at
300 MHz. Compare the design study (no firn, l = 3 m): -45 to -49 dB.

### Comparison to C&S's claims

- Plateau grows with frequency at fixed bandwidth: reproduced (3-D +3.5-4 dB
  195 -> 300; 1-D +6 dB at 100 MHz, ~0 at 30 MHz at this core).
- Plateau suppressed by bandwidth: reproduced in the 3-D stack as a clean 1/B
  in firn power, ~4 dB rel surface from 10 to 100 MHz (airborne median
  estimator 7.8 dB); the 1-D full-resolution model shows the effect
  saturating above ~30 MHz at this core because the reflectivity spectrum is
  not flat across a wide band. "Mitigable with <= 50 % fractional bandwidth
  up to 200-250 MHz, not above" -- at 300 MHz the 1-D plateau at 50 % BW is
  -10 dB rel surface vs -17 dB at 195 MHz, i.e. the wider band does not buy
  the loss back, in line with their statement.
- Off-nadir clutter from layer roughness: with their Fig. 11 roughness the
  layers are the dominant bed-delay clutter source at 14 km; C&S's airborne
  Fig. 13 SCR at the B26 bed (-10 to -25 dB, raw MCoRDS3) is not directly
  comparable (500 m AGL, different geometry, exponential ACF).

## 5. Caveats

- No in-ice focusing / no focusing at all: per-trace unfocused sums. Every
  along-track scatterer at the bed delay is integrated without rejection --
  the same limit as the design study's aliased default posting (round 7:
  where the posting aliases the clutter angle, focusing removes nothing), so
  the bed-window numbers are comparable to the design study's posting_div 1
  rows and are pessimistic for a real platform sampling at centimetres.
  Trace posting here (HAPS 24 traces / 10 km) is irrelevant to per-trace
  metrics; it only sets the number of independent samples in the medians.
- Roughness provenance: surface sigma 4.9 cm / l 2.98 m and the internal
  layer sigma/l are the C&S Fig. 11 MCoRDS3 inversion digitized from the
  paper (tests/fixtures/firn/fig11*.csv, 0-90 m, clamped below), applied with
  the Gerekos GAUSSIAN-ACF rough-facet model. C&S used an EXPONENTIAL
  correlation function in their S-IEM clutter model; the exponential ACF has
  far heavier wide-angle tails, so this sim's wide-angle (30-45 deg) layer and
  surface clutter is a lower bound, exactly as the design study warned
  (l = 1 m rows there are the pessimistic proxy).
- Layer count: N = 20 uniform (6.25 m segments), the validated standard;
  segment aggregation is done at the carrier only, so the 3-D stack cannot
  reproduce the in-band spectral variation that the 1-D full-resolution
  model shows -- in the 3-D sim the firn power is exactly 1/B by
  construction. No decimation was needed (all runs under the budget).
- Surface normalization: the wide run's surface is air->ice (eps 3.17), the
  B26/M24 convention; the physical air->firn + top-metre aggregate is ~2.7 dB
  weaker, so all "rel surface" levels are ~2-3 dB pessimistic and
  bed-over-clutter is unaffected (both arms share the surface).
- Attenuation 15 dB/km one-way (M24 warm-ice constant) over 2.6 km of cold
  interior ice: bed arm ~78 dB two-way; a cold-ice 8 dB/km would lift every
  bed-over-clutter number by ~36 dB. Bed eps 8, no RSSNR calibration.
- Analytic pulse for the airborne sweep (validated model); explicit chirp for
  HAPS. The two agree on every plateau number to 0.1 dB.

## 6. Runtimes (this session, CPU, jax cache warm)

| run | facets/interface | traces | wall |
|---|---|---|---|
| b26 revival wide_surface_bed | 1.76M | 60 | 18 s |
| b26 revival firn_N20_h1eff (4 chunks) | 82k x 21 | 60 | 1736 s (contended) |
| airborne wide_f195 (delta) | 1.76M | 60 | 24 s |
| airborne firn_f195 (4 chunks) | 82k x 21 | 60 | 1431 s |
| haps wide_f195 / wide_f300 | 436k / 669k | 24 | 2.5 s / 8 s |
| haps firn_f195 (1 chunk) | 407k x 21 | 24 | 1926 s |
| haps firn_f300 (3 chunks x 8 traces) | 453k x 21 | 24 | 5303 s |

Every pulse (9 HAPS + 6 airborne) and the 1-D sweeps are post-processing
(seconds). The haps firn_f300 run exceeded the ~30 min guideline (88 min:
1.5x the facets of the 195 run plus per-chunk compile after the run was
restarted chunked); it was allowed to finish rather than decimating the
layer stack so both carriers share the identical N = 20 stack. A repeat can
use `--n-traces 12` (per-trace metrics only need independent samples) or
`--n-layers 10` (uniform decimation; band level N-invariant by construction).
Total simulation wall this session ~2.6 h.

## 7. Reproduce

```
uv run python tools/run_b26_comparison.py --only wide_surface_bed,firn_N20_h1eff --force
uv run python tools/run_firn_bandwidth.py airborne            # 60 traces, 10 km
uv run python tools/run_firn_bandwidth.py haps                # 24 traces, alt 14000 m
uv run python tools/run_firn_bandwidth.py haps --report-only  # re-analyse cached fields
```
