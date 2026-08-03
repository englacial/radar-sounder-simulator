# Basal-clutter altitude triplet — PILOT results (2026-07-31)

Tool: `tools/run_basal_clutter.py` (pilot s=30–40 km, 48 sim traces/pass,
coherent SURFACE+BED only, surf-rough ON, att 15 dB/km). Outputs in
`outputs/basal_clutter/pilot/`. Scout: `claude_notes/basal_clutter_scout.md`.
Stopped after the pilot per plan — the 50 km run needs a go-ahead.

## Derived cross-track reaches (nadir-bed delay + 3 µs, both interfaces)

| pass | surface reach | bed reach (refracted) | ct used | facet spacing | facets/iface | wall |
|---|---|---|---|---|---|---|
| low 442 m | 2493 m | 927 m | ±2493 m | 10.67 m | 1.31 M | 36.4 s |
| mid 9150 m | 6185 m | 2957 m | ±6185 m | 46.26 m | 179 k | 11.3 s |
| high 10684 m | 6643 m | 3181 m | ±6643 m | 49.77 m | 169 k | 4.2 s |

Surface interface always binds (its target delay spans the whole ice
column); no caps applied. The old 6 km default cap would have clipped both
high passes. High passes are ~4.7× coarser-faceted and *cheaper* than the
low pass despite 2.6× the swath (scout quirk 4 confirmed).

## Headline: does the sim reproduce the ~20 dB altitude effect?

Mid-column mean power (surf+1.0 → bed−0.5 µs), dB rel own surface peak:

| pass | measured | sim | sim − meas |
|---|---|---|---|
| low | −54.7 | −71.0 | −16.3 |
| mid | −35.4 | −41.4 | −6.0 |
| high | −35.2 | −41.3 | −6.1 |
| **high − low** | **+19.5** | **+29.7** | +10.2 |

* **YES, the sim shows much more clutter at altitude** (+29.7 dB high−low
  vs +19.5 measured; mid ≈ high in both, as the scout found).
* The sim OVERSHOOTS the trend by ~10 dB, and the overshoot is entirely at
  the LOW pass: at altitude the surface+bed sim comes within ~6 dB of the
  measured mid-column, i.e. high-altitude mid-column clutter is mostly
  surface+bed geometry; at 442 m the sim is 16 dB below measured — the
  measured low-pass mid-column is dominated by englacial returns (internal
  layers/volume scatter, clearly visible in its radargram) that the
  surface+bed sim omits BY DESIGN. So the measured +19.5 dB is compressed
  by an englacial floor the model doesn't carry; the geometric-clutter
  altitude effect itself is larger.
* Measured floors (record-end −12..−8 µs window): −125 / −86 / −76 dB rel
  surface — no pass is noise-limited in the mid-column. (First attempt used
  the pre-surface window: WRONG on the low pass, TX-leakage/img_comb zone
  reads 26 dB ABOVE its mid-column. Record tail last ~4 µs is rolled off
  too — probed 2026-07-31, see FLOOR_TAIL note in the tool.)

## Decomposition (per-interface coherent fields): SURFACE-borne

The mid-column window is **surface-borne at all three altitudes**:
bed-borne mid-column contributions are ≤ −115 dB rel surface (negligible);
sim surface-borne ≈ sim total there. Bed-borne energy dominates only the
bed window (bed−0.5 → bed+1.5 µs), where sim vs measured agree well:
low −50.9/−55.5, mid −47.4/−48.1, high −47.2/−47.7 dB (att 15 dB/km left
the low-pass sim bed ~4.6 dB hot; mid/high within 1 dB). The measured
high-pass "basal" clutter filling the column is off-nadir SURFACE
scattering, not bed scattering — the discriminator the study wanted.

## Sanity

Surface-gate medians 0.39/0.41/0.50 bins (≤5 gate, pass); p90 is 18/27
bins on the high passes purely because the measured high-altitude Surface
pick is ~3.6 bins noisy (scout registration table) — offsets fitted per
pass. Fields all finite; dropped-power fraction ≤ 1.5e-4; bed_clamp 0;
scout-predicted spacings (10.67/47/50.6 m) and reaches (±5.4 km at bed
delay without margin) reproduced by the tool's derivation (tested).

## 50 km projection (s=18–68 km, 240 traces, 5 pilot-sized chunks/pass)

low ~182 s, mid ~57 s, high ~21 s → **~260 s (~4.5 min) total sim time**.
Caveats: per-chunk JAX recompile (chunk shapes differ) could add minutes;
the wider bbox needs new REMA/BedMachine windows (one-time downloads,
few minutes). Realistic end-to-end: **~10–15 min**. No parameter changes
suggested by the pilot; if anything, consider whether to add the firn/
internal-layer stack later to close the low-pass −16 dB englacial gap
(out of scope for this study by explicit design).

## Figure notes

`radargrams.png`: sim reproduces the bright mid-column fill at altitude and
the dark column at 442 m; sim bed arcs are visibly smoother than measured
(BedMachine 500 m texture caveat, expected). `decomposition.png`: orange
(surface-borne) carries the mid-column at all altitudes; green (bed-borne)
switches on at the first off-nadir bed arrival and owns the bed window.

## Full 50 km run (2026-07-31)

All 15 chunks green, 260 s total simulation (36 s/chunk low, 11 mid, 4
high), reaches surface 2.5/6.4/6.9 km. Mid-column mean power rel own
surface peak (meas/sim dB): low -54.6/-72.2, mid -35.5/-45.0, high
-34.8/-44.0 -- sims reproduce the altitude effect (high-low +28.2 dB sim
vs +19.8 measured; mid ~ high in both). Decomposition: surface-borne at
all altitudes; bed-borne is confined to the bed window, where sim bed
arcs reproduce the measured 1-5 km hyperbola trains (s~55-65 km) though
visibly smoother (BedMachine 500 m posting). The measured-over-sim
mid-column residual (~9-18 dB, largest at the low pass) is englacial
scattering excluded by design in this surface+bed study.

## Picked-bed 50 km run (`--picked-bed`, 2026-07-31)

`outputs/basal_clutter/full_pbed/` (vs BedMachine `full/`). Bed =
BedMachine + resid(s), resid(s) = picked_bed(s) - BedMachine at nadir(s) on
the anchor along-track axis; picks from the LOW pass ONLY
(20161105_05_005-007) applied identically to all three sims. Elevations use
the tool's existing convention (surface = Elevation - c*Surface/2,
thickness = (Bottom-Surface)*c/(2*sqrt(3.17)), WGS84-ellipsoidal like
REMA/BedMachine). Nadir matches the picks exactly; BedMachine's cross-track
relief is preserved (a constant cross-track extension of the 1-D profile
would have erased it). Grid/reach/facet spacing unchanged from the
BedMachine run, so the two are directly comparable.

Residual over s=18-68 km: **81.3 m rms, +31.3 m mean, 216 m |max|, gap
fraction 0.0000** (the +31 m mean is the scout's "BedMachine - radar bed
= -30 m"). Along-track bed roughness (rms about a 5 km running mean):
**28.5 -> 60.3 m** (scout: 33.3 BedMachine / 60.5 picks; our 28.5 is the
32 m-grid-resampled BedMachine, slightly smoother than the scout's native
sampling). 264.6 s sim wall (183/62/20 s), 15 chunks, clamp 0.

| dB rel own surface peak | low | mid | high |
|---|---|---|---|
| sim mid-column (bedmachine -> picked) | -72.2 -> -71.3 | -45.0 -> -44.8 | -44.0 -> -43.7 |
| sim bed window | -48.5 -> -50.4 | -47.0 -> -45.2 | -46.2 -> -44.9 |
| measured bed window | -54.3 | -46.0 | -46.1 |
| sim midcol/bed-peak (scout metric) | -56.6 -> -48.3 | -31.3 -> -18.7 | -28.6 -> -18.5 |
| measured midcol/bed-peak | -28.6 | -4.6 | -3.0 |
| **bed-borne** mid-column | -143.7 -> -141.1 | -126.0 -> -58.1 | -124.2 -> -61.4 |

Altitude trend barely moves (high-low +28.2 -> +27.7 dB sim vs +19.8
measured): the mid-column is still surface-borne at all three altitudes.
The real change is at the BED: the bed-borne mid-column contribution jumps
**+63 to +68 dB** at altitude (still 13-16 dB below the surface-borne
term), the specular bed peak drops so the scout contrast metric closes
8-13 dB of its ~25 dB gap to measured, and the low pass's 5.8 dB-hot bed
window drops to 3.9 dB hot. Radargrams: the smooth BedMachine bed band is
replaced by a dense field of overlapping hyperbolae across the whole
segment, the bed envelope now follows the measured bed (by construction),
and the post-bed tail matches measured within a few dB instead of falling
10-20 dB short.

**Caveat, do not tune this away.** The residual is constant along the
cross-track normal, so along-track pick detail becomes cross-track RIDGES
out to +-ct. That is the unavoidable consequence of correcting a 2-D DEM
with a 1-D profile, and it makes the added bed roughness perfectly
correlated cross-track (anisotropic). The +63 dB bed-borne mid-column jump
and the arc density should therefore be read as an UPPER bound on what a
truly 2-D bed of the same rms would give.

## RSSNR-driven bed reflectivity (`--picked-bed --gamma-from-rssnr`, 2026-07-31)

Full 50 km with the bed reflectivity driven along-track by the
required-surface-SNR dataset (claude_notes/required_snr_dataset.md; pinned
icechunk snapshot `3YH47013745B2T5ZZR50`, 111 anchor-line samples at ~1.37 km,
0 censored, cache `outputs/basal_clutter/rssnr_anchor.npz`). Mapping
|Gamma_bed|^2(s) dB = 2*A*H(s) − RSSNR(s) + K, H from the dataset's own
twtts, A = 15 (the run's --att), K **median-anchored** so the segment median
equals the constant Fresnel −12.9 dB. ONE field shared by all three passes,
linear along-track interpolation, cross-track constant (picked-bed caveat
class). Per-facet gamma rides the kernels' blocked scan (commit 7d6292a);
`scene.gamma_maps` plumbing through simulate(). Outputs:
`outputs/basal_clutter/full_pbed_rssnr/` (bed_brightness.png + radargrams +
decomposition + report). Wall 253.5 s (178.7/54.6/20.1), all 15 chunks fresh;
constant-gamma companion pure cache hits.

### Mapping stats

* **K = +11.39 dB**, K_phys = −10.32, **K − K_phys = +21.71 dB**. With
  segment-median H = 641 m this implies an **effective one-way attenuation of
  30.8 dB/km** — independently reproducing run_cross_season's calibrated
  effective 31 dB/km on a different West Antarctic line. Strong evidence the
  median anchoring is absorbing exactly the attenuation the b26 value (15)
  under-books, i.e. the anchoring choice was right.
* Segment G2 (dB): min −32.5 / p5 −29.8 / med −12.9 (by construction) /
  p95 +11.7 / max +15.9 — a **~48 dB along-track dynamic range**. 18.9% of
  segment samples map above 0 dB (unphysical reflectivity): the price of
  holding A = 15 fixed; with A ≈ 31 the whole profile would shift down ~+2AH
  and the brights would land near physical values. Recorded, not tuned away.

### Acceptance: bed-window brightness along-track (1 km smoothed, Pearson r)

| pass | sim const vs meas | **sim RSSNR vs meas** | sanity: bed-layer vs implied | implied vs meas (ceiling) |
|---|---|---|---|---|
| low  | −0.15 | **+0.76** | +0.91 | +0.87 |
| mid  | −0.01 | **+0.84** | +0.88 | +0.85 |
| high | +0.14 | **+0.79** | +0.88 | +0.80 |

**Acceptance criterion met, decisively.** The constant-gamma runs carry
essentially zero along-track bed-brightness information (r −0.15..+0.14: with
gamma constant, the sim profile is geometry-only). The RSSNR-driven runs reach
r = 0.76–0.84 against the measured bed window — close to the data-only
ceiling (implied-vs-measured 0.80–0.87), i.e. the simulator transports nearly
all the information the RSSNR profile contains through the full coherent
chain (refraction, per-facet gamma, picked-bed relief, waveform, SAR-free
windows). The by-construction sanity (bed-borne layer vs implied) is 0.88–0.91
— the residual from 1.0 is geometry + speckle + the 1.4 km sampling, and the
total-field sanity equals it (bed window stays bed-borne-dominated at all
three altitudes given the ±35 dB gamma range).

### Honest caveats

* The RSSNR sim **overshoots the measured highs by ~5–10 dB** at the bright
  end (s > 60 km, the G2 > 0 dB zone) on all passes — consistent with the
  fixed-A error growing away from the median thickness and with the measured
  bed window compressing toward its clutter/noise floor. Median levels are
  close (e.g. low: meas −53.9 vs sim −49.8 dB rel surface).
* RSSNR is surface-referenced: measured surface-power variability leaks into
  the field with opposite sign. Not separable at this stage.
* Sampling is ~1.4 km; sub-km bed-brightness texture is not driven, and the
  bright/dim transitions are linear-in-dB ramps between samples.
* The pilot-segment numbers (10 km, ~7 independent samples) were noisy and
  even sign-flipped (low: +0.19 const vs −0.16 RSSNR) — correlations at that
  scale are not meaningful; the 50 km numbers above are the deliverable.

Mechanically: `--gamma-from-rssnr` composes with `--picked-bed`
(`_pbed_rssnr` cache/output suffix; constant-gamma metas byte-identical to
pre-feature caches). Config records snapshot id, fetch provenance, K,
K−K_phys, censoring policy (floor, not missing-at-random), interpolation and
sharing choices. Tests: tests/test_rssnr_gamma.py (mapping round-trip,
censoring floor, snapshot-pin cache rejection, gamma_maps plumbing
bit-identity); suite 264 unit green.

## Matched processing + 30 km prediction (`--processing standard --add-30km`, 2026-07-31)

Full 50 km, best-model config (picked bed + RSSNR gamma), all four passes
(low/mid/high/**syn30km**) through a CSARP_standard-matching chain, into
`outputs/basal_clutter/full_pbed_rssnr_proc/`.

### Processing chain: real vs applied (recorded per pass in run_config)

REAL (provenance): motion-compensated **f-k**, sigma_x **2.5 m** SLC,
start_eps 3.15 (2016 param structs, scout-verified; ft_wind hanning
hand-verified) -> delay-and-sum combine -> **rline_rng [-5..5] = 11 looks,
dline 6** -> 14.85 m posting, ~25 m effective along-track resolution (11/6 =
M24-verified CReSIS-standard convention; NOT directly read from the 2016
structs -- recorded assumption).

APPLIED: simulate at the **product posting 14.85 m** (every measured trace;
sim columns land on the measured columns) -> first-order nadir mocomp
(dz to smoothed track; residual rms 0.113/0.068/0.049/0.0 m = lambda/14 or
better) -> `soundersim.processing.focused_sar` (straight-track
backprojection, hann) at the **alias-limited aperture**
sin(theta)=lam/(4*ds), theta = 1.52 deg: **87 / 546 / 627 / 1647 m**
(7/38/43/112 traces) at each pass's median optical bed range -> 3-look
stride-1 incoherent average. Az resolution ~21 m (hann) vs the product's
~25 m effective. Derivation recorded: the 2.5 m SLC would need 2.5 m posting
(~40x compute); at posting ds the best unaliased resolution IS ds, so
matching the post-look effective level is the honest optimum. Straight-track
check: the line is dead straight (chord dev p95 0.01-0.06 m); vertical
motion is the real deviation and mocomp leaves <~lambda/14. Gap list
g1-g6 recorded in config (SLC res, looks count, air-only focusing ~1 rad
edge phase at altitude, first-order-only mocomp, near-surface Doppler
aliasing ratio 3.65/1.12/1.10/1.04, combine inside the array pattern).
Unprocessed path untouched (default `--processing none`; `_proc` cache
suffix; const-gamma companions got their own fine-posting `_pbed_proc`
cache).

### Acceptance correlations under matched processing (vs unprocessed)

| pass | const (proc) | RSSNR (proc) | RSSNR (unproc, phase 2) | sanity bed-layer | implied-vs-meas |
|---|---|---|---|---|---|
| low  | -0.17 | **+0.76** | +0.76 | +0.92 | +0.87 |
| mid  | -0.14 | **+0.81** | +0.84 | +0.85 | +0.85 |
| high | +0.02 | **+0.76** | +0.79 | +0.84 | +0.80 |

**The acceptance result HOLDS under matched processing** (mean r 0.78 vs
0.80 unprocessed; the small dip is look-averaging smoothing both sim and
implied against a measured curve that already had its own 11 looks).
Constant gamma stays uninformative. Visual acceptance: the sim radargrams
now show **continuous arcs, not beads** -- the mid/high sim panels are
texture-comparable to the measured f-k product (sim arcs crisper: g2's
fewer looks). Sim mid-column medians barely move vs unprocessed (e.g. high
-43.1 -> -42.9 dB): mean-power metrics are insensitive to coherent
reprocessing, as expected.

### 30 km stratospheric prediction (the design question)

Geometry: 29,858 m median AGL, reach ct **+-11.24 km** (surface 11.24 km /
bed 5.23 km), facet spacing 81.9 m, 91k facets/interface/chunk, fast-time
window to ~215 us, 17 chunks, **249 s wall** -- the reach derivation stayed
compute-feasible (no cap needed). Aperture 1647 m (112 traces).

**Verdict: on this line, at 190 MHz/50 MHz with the 2016 DC-8 system
geometry, the bed remains visible at 30 km -- but marginally.** In the bed
window the bed-borne energy beats the surface-borne clutter by **+11.4 dB**
(median); the bed peak stands **+5.0 dB** above the total mid-column
clutter (scout contrast metric) -- vs measured +28.6 dB at 442 m and
+3.0-4.6 dB at 9-10.7 km, i.e. the 30 km pass sits just past the high pass
on the clutter-limit trend rather than falling off a cliff (the surface
clutter level saturates once the full off-nadir surface is inside the
window). Mid-column stays surface-borne (-33.8 dB rel surface peak, ~1 dB
above the 10.7 km pass). Along-track, the RSSNR-driven bed brightness spans
~40 dB, so the DIM half of the line (s < 40 km, G2 to -32 dB) drops to or
below the clutter at 30 km while the bright half stands well clear --
exactly the along-track discrimination this feature chain was built to
predict. CAVEATS: clutter-limited analysis only (no receiver-noise/link
budget -- at 30 km the extra ~9 dB of two-way spreading loss vs 10.7 km
makes thermal SNR the other binding constraint); surface roughness at the
validated representative values; 500 m BedMachine cross-track texture and
the picked-bed/RSSNR cross-track-constant caveats carry over; 82 m facets
mean sub-facet bed texture is unresolved.

Timings: RSSNR quad 1941.7 s fresh (1038/518/136/249), const companions
1671 s (fine posting), processing+analysis ~6 min/invocation. One harness
kill/resume mid-run (all chunks cache-resumable, no loss). Tests: 269 unit
green (tests/test_basal_processing.py added: aperture math, chain
recording, 30 km construction; pass-table test updated for the synthetic
entry); ruff clean.

## Bed-source ablation: picked bed vs BedMachine vs DEMOGORGN (2026-08-03)

Three-way ablation on the full 50 km, best-model config otherwise identical
(RSSNR gamma + matched CSARP processing), composed as the FOUR-row
`outputs/basal_clutter/full_pbed_rssnr_proc/radargrams.png` (measured /
picked bed / BedMachine / DEMOGORGN seed 0; two-row original kept as
radargrams_tworow.png). Metrics keyed `bedmachine_bed_ablation_*` /
`demogorgn_bed_ablation_*`; DEMOGORGN standalone run in
`outputs/basal_clutter/full_dgn_rssnr_proc/`.

DEMOGORGN plumbing (per claude_notes/demogorgn_scout.md): pinned snapshot
`WG801625MG778C4DS6Y0`, seed 0, `opr.fetch_demogorgn_window` (raw cache +
EIGEN-6C4 geoid added on read from **band 2 of the BedMachine cache** -- the
one real plumbing change; old single-band caches refetch once,
backward-compatibly gated and tested). PLAIN DEMOGORGN only; the picked-bed
hybrid (residual 81.3 -> 43.7 m rms, anisotropy 1.90 -> 1.27) is a
**recorded follow-up**, deliberately not wired. No license found -- internal
use only until the Gator Glaciology group provides one (recorded in config).

### Bed-window power (dB rel own surface peak) and r vs measured (1 km)

| pass | measured | picked bed | BedMachine | DEMOGORGN | r: pbed / BM / DGN |
|---|---|---|---|---|---|
| low  | -54.3 | -48.6 | -50.1 | -46.4 | +0.76 / +0.68 / **+0.85** |
| mid  | -46.0 | -40.6 | -44.8 | -43.4 | +0.81 / +0.73 / +0.78 |
| high | -46.1 | -42.3 | -43.6 | -43.1 | +0.76 / +0.66 / +0.78 |
| 30 km | -- | -42.1 | -43.2 | -41.4 | -- |

Mid-column is bed-source-insensitive (within ~1 dB across all three --
surface-borne everywhere, as established). Every bed source rides the RSSNR
gamma to r >= 0.66: the reflectivity field, not the topography, carries the
along-track brightness pattern.

### What each source gets right/wrong (numbers + visual)

* **Picked bed**: along-track exact at nadir (offset ~0) and the densest
  arc field -- visually closest to the measured texture in the low panel --
  but its roughness is cross-track INERT (ridges artifact, upper-bound
  clutter; unchanged caveat).
* **BedMachine**: smooth everywhere -- a bald continuous bed line with
  sparse gentle arcs, visibly unlike measured; bed-window 1-4 dB dimmer than
  picked bed; nadir bed sits 31-69 m deep vs the passes' own picks (the
  known -29 m bias + the 6% cross-pass pick-thickness spread); lowest r on
  every pass.
* **DEMOGORGN**: statistically right everywhere -- isotropic 2-D texture
  with realistic arc density (between BedMachine and picked bed in the
  figure, and arguably the most measured-like at mid/high); **beats picked
  bed on the acceptance correlation on low (+0.85 vs +0.76) and high (+0.78
  vs +0.76)** -- consistent with the picked bed's cross-track ridges being
  an artifact that DEMOGORGN's honest 2-D roughness avoids. Its bed line is
  ITS OWN realization: locally different wiggles than measured (43.7 m rms
  vs picks), though the **median nadir offset is small: -0.08/+0.21/+0.14 us
  = -7/+17/+12 m** (low/mid/high) -- the scout's ~44 m raw offset was the
  GEOID datum, which the band-2 plumbing removes; what remains is the
  thickness-convention scatter, visible as local bed-line disagreement, not
  a constant shift. Reported, not tuned.

30 km prediction is bed-source-robust: bed-window -41.4..-43.2 dB and
mid-column -33.8..-34.2 dB across all three beds -- the stratospheric
verdict (+11 dB bed-over-clutter in window, ~5 dB peak margin,
clutter-limited) does not hinge on the bed choice.

Timings: BedMachine ablation sims 1933 s fresh (low 1021 / mid 526 / high
137 / syn30 250); DEMOGORGN standalone 1908.6 s fresh (1012/514/134/248,
--no-companion) + one-time BedMachine band-2 cache refetches; composition
run pure cache (+15 process_standard calls, ~12 min). Tests: 276 unit green
(tests/test_demogorgn.py: band-2 gate incl. refetch-on-single-band,
raw+geoid datum path on the staggered grid, snapshot/seed cache pinning,
tag composition, hybrid-raises, nadir-offset math); ruff clean.
