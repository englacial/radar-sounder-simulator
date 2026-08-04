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

## Bed-return tail excess (2026-08-03, assembly only)

Quantifies the observation that **all three simulated bed variants sit above
the measured bed-return decay after the bed echo**. New metric
`bed_return_tail_{pass}` in
`outputs/basal_clutter/full_pbed_rssnr_proc/metrics.json` + `bed_tail.png`
(mirrored to `outputs/verification/basal_clutter_full_pbed_rssnr_proc/`).
Pure cache replay of the composition invocation -- **no new simulations**.

Definition: trace-ensemble mean power (dB rel each trace's OWN surface peak,
the tool's existing gain-free currency), profiled against delay past **each
trace's own bed reference** -- measured on its `Bottom` pick, sims on the
**sim bed-layer nadir twtt** (the per-pass gate registers the SURFACE only,
so no bed alignment is borrowed; residual nadir-bed offsets stay recorded in
`*_bed_ablation_*.nadir_bed_offset_vs_picks`: BedMachine +0.37/+0.82/+0.71 us
= 31/69/60 m deep, DEMOGORGN -0.08/+0.21/+0.14 us = -7/+17/+12 m). Slope = Theil-Sen (robust: one bright
arc crossing the window cannot set it) over **bed+0.5 -> bed+3.5 us**;
excess = sim - measured at bed+1/2/3 us.

### Slopes: delay (dB/us) and refracted bed incidence angle (dB/deg)

| pass | measured | picked bed | BedMachine | DEMOGORGN |
|---|---|---|---|---|
| low 449 m | **-8.22** / -2.01 | -7.46 / -1.62 | -8.26 / -1.81 | -6.51 / -1.54 |
| mid 9150 m | **-5.15** / -2.64 | -4.15 / -1.95 | -5.28 / -2.68 | -4.90 / -2.43 |
| high 10684 m | **-3.17** / -1.65 | -3.63 / -1.98 | -3.05 / -1.54 | -2.90 / -1.62 |
| 30 km (pred.) | -- | -3.36 / -2.91 | -4.82 / -4.11 | -0.80 / -0.72 |

The same post-bed delay probes **very different bed angles per altitude**
(`bed_return_angle_map_deg`, +1/+2/+3 us): low **18/24/27 deg**, mid 6/8/10,
high 5/7/9, 30 km 3/4/6. The dB/us columns are therefore NOT comparable
across passes; the dB/deg columns are, and they are flat-ish (-1.5..-2.7)
across the triplet -- one angular-backscatter law seen through three
geometries.

### Excess (sim - measured, dB) at bed+1 / +2 / +3 us

| pass | picked bed | BedMachine | DEMOGORGN |
|---|---|---|---|
| low | **+10.8 / +9.7 / +11.8** | -5.9 / -9.8 / -5.5 | -14.3 / -14.4 / -11.4 |
| mid | **+13.3 / +14.2 / +11.5** | +10.1 / +7.8 / +6.1 | +10.5 / +7.6 / +11.0 |
| high | **+15.9 / +17.3 / +12.6** | +13.0 / +13.2 / +16.3 | +10.8 / +11.1 / +21.7 |

Measured tail levels (dB rel own surface peak) at +1/+2/+3 us: low
-57.5/-65.2/-73.2, mid -40.3/-46.0/-48.7, high -43.9/-47.1/-50.0.

### Fair-comparison guard and noise-floor caveat

Guard = min over the fit window of (sim **bed returns** - sim **surface
returns**) from the per-interface decomposition; >= 10 dB required for the
total-field tail to be read as bed returns.

| pass | guard pbed | guard BM | guard DGN | meas tail +3 us | meas floor | margin | sim record coverage |
|---|---|---|---|---|---|---|---|
| low | +26.7 ok | **+9.6 FAIL** | **+3.8 FAIL** | -73.2 | -125.8 | **+52.6** | 1.00 / 1.00 / 1.00 |
| mid | +26.7 ok | +16.3 ok | +22.3 ok | -48.7 | -86.1 | **+37.4** | 0.95 / 0.97 / 0.92 |
| high | +29.1 ok | +23.6 ok | +26.9 ok | -50.0 | -75.3 | **+25.3** | 1.00 / 1.00 / 1.00 |
| 30 km | +17.5 ok | +15.0 ok | +13.0 ok | -- | -- | -- | 1.00 |

* The two guard FAILs are the LOW pass's genuinely-2-D beds, whose bed
  returns fall so fast that the (bed-source-invariant) surface returns come
  within 4-10 dB by bed+3.5 us: their already-negative excesses are
  **upper bounds** on the sim bed returns; use the bed-returns-only slopes
  (-8.39 BedMachine, -7.01 DEMOGORGN) instead of the total-field ones.
* **No pass is floor-limited**: the measured tail stands 25-53 dB above the
  pass's own floor estimate at bed+3 us, so the measured decay is real
  returns and the excesses are not a floor artifact. The high pass has the
  least headroom (+25.3 dB), as expected from its -75.3 dB floor.
* Sim coverage < 1 on the mid pass only (0.92-0.97): the fast-time window is
  anchored on the DEEPEST bed + 3.5 us, so a few of the shallowest traces run
  out of record at the very end of the fit window.

### What the numbers say

The excess is real and **grows with altitude**: the best-model (picked bed)
sim runs +10 dB hot at 449 m, +14 at 9.2 km and +17 at 10.7 km at bed+2 us --
roughly **double to quadruple the corresponding bed-WINDOW offset** already
recorded in `clutter_*` (+5.7 / +5.4 / +3.8 dB), i.e. the sim does not merely
put too much power in the bed echo, it puts too much power **off-apex**. The
slope columns say where that comes from: at low and mid the sim tail decays
**0.8-1.0 dB/us too slowly**, at high it decays slightly too fast (-3.63 vs
-3.17) so its whole +16 dB gap is already open by bed+1 us. Two mechanisms
are visible and separable. (1) **Attenuation obliquity**: raising the
one-way attenuation from the run's 15 dB/km to run_cross_season's calibrated
effective 31 dB/km adds 2.15 -> 4.44 dB of decay across the low pass's
window (+0.76 dB/us, essentially the entire low-pass slope deficit) but only
0.08 -> 0.16 dB/us at mid/high, whose tails never leave 10 deg -- so the
attenuation book cannot explain the altitude-growing part. (2) **Bed
roughness/anisotropy**: the bed-source ablation splits cleanly at the low
pass, where the picked bed's cross-track RIDGES (the known 1-D-residual
artifact, anisotropy 1.90) sit +10 dB hot while the two genuinely 2-D beds
sit 5-14 dB LOW -- i.e. at 449 m the tail is a direct read-out of bed
roughness and the picked bed's excess is largely artifact, whereas at 9-11 km
**all three beds are +7..+17 dB hot**, so the altitude part of the excess is
NOT a bed-topography choice. That leaves the RSSNR gamma's recorded
bright-end overshoot (+5-10 dB where G2 > 0 dB), the missing volume/
englacial loss, and the sim's coherent-field speckle statistics (g2: 2-3
looks vs the product's 6-11) as the remaining candidates -- the last of which
raises the ensemble MEAN only weakly, so the level terms are the prime
suspects. Practical consequence for the 30 km prediction: its bed-over-
clutter margin (+11.4 dB in the bed window) is computed from the same
too-hot off-apex bed returns, so the prediction is optimistic in the same
direction, and the DEMOGORGN 30 km tail (-0.80 dB/us, essentially flat) is
the least trustworthy of the four -- a follow-up, not a fix, and nothing
here was tuned.

## Hypothesis campaign on the bed-return tail excess (2026-08-03)

Baseline for every test: **DEMOGORGN bed + RSSNR gamma + matched CSARP
processing** (`outputs/basal_clutter/full_dgn_rssnr_proc`, copied to
`hypothesis_tests/baseline/`), ONE variable changed at a time. Each test
writes `radargrams.png`, `decomposition.png`, `bed_tail.png`, `metrics.json`
and `run_config.json` to `outputs/basal_clutter/hypothesis_tests/<test>/` and
mirrors metrics+figures to `outputs/verification/basal_clutter_<test>/`.

New CLI knobs (all OFF by default, and each contributes to the chunk cache
key ONLY when non-default, so all 68 pre-campaign caches stayed valid -- the
baseline replay re-ran 0 simulations): `--out-name`, `--passes`, `--antenna
{array,isotropic,array8}`, `--bed-rough SIGMA L`, `--bed-rough-extra-db`,
`--posting-div`.

**Read the guard column first.** `bed_return_tail_*` records the total-field
slope AND the bed-returns-only slope, plus the guard (sim bed returns minus
sim surface returns over the fit window). Several variants push surface
clutter up or bed returns down until the total-field tail is no longer bed
returns; where the guard fails only the **bed-returns-only** slope is
interpretable, and that is the column tabulated. T1/T3/T4b were run on **low
+ high only** (the two ends of the altitude trend) to stay inside the compute
budget -- mid tracks high everywhere in this study, and syn30km has no
measured counterpart so it cannot enter a "vs measured" delta.

### Master table (sim = DEMOGORGN bed; excess = sim - measured at bed+2 us)

| test | pass | bed-return slope dB/us | d base | excess +2 us | d base | bed window dB | d base | guard dB | measured slope/bed |
|---|---|---|---|---|---|---|---|---|---|
| baseline | low | -7.01 | +0.00 | -14.4 | +0.0 | -46.4 | +0.0 | FAIL +4 | -8.22 / -54.3 |
| baseline | mid | -4.90 | +0.00 | +7.6 | +0.0 | -43.4 | +0.0 | ok +22 | -5.15 / -46.0 |
| baseline | high | -2.91 | +0.00 | +11.1 | +0.0 | -43.0 | +0.0 | ok +27 | -3.17 / -46.1 |
| baseline | syn30km | -0.76 | +0.00 | -- | -- | -41.4 | +0.0 | ok +13 | -- |
| t2_att31 | low | -7.74 | -0.73 | -24.0 | -9.6 | -67.3 | -20.9 | FAIL -21 | -8.22 / -54.3 |
| t2_att31 | mid | -4.58 | +0.32 | -12.2 | -19.8 | -60.8 | -17.4 | FAIL +1 | -5.15 / -46.0 |
| t2_att31 | high | -2.36 | +0.55 | -10.3 | -21.4 | -60.9 | -17.8 | FAIL +5 | -3.17 / -46.1 |
| t2_att31 | syn30km | -0.78 | -0.02 | -- | -- | -52.5 | -11.1 | FAIL -8 | -- |
| t4_isotropic | low | -5.71 | +1.30 | +8.1 | +22.5 | -44.2 | +2.2 | FAIL +2 | -8.22 / -54.3 |
| t4_isotropic | mid | -4.09 | +0.81 | +13.3 | +5.7 | -37.6 | +5.8 | FAIL +4 | -5.15 / -46.0 |
| t4_isotropic | high | -2.69 | +0.22 | +16.6 | +5.5 | -38.3 | +4.8 | FAIL +8 | -3.17 / -46.1 |
| t4_isotropic | syn30km | -0.22 | +0.54 | -- | -- | -37.6 | +3.8 | FAIL +3 | -- |
| t4b_array8 | low | -6.98 | +0.03 | -14.6 | -0.2 | -46.7 | -0.3 | FAIL +3 | -8.22 / -54.3 |
| t4b_array8 | high | -2.81 | +0.09 | +10.1 | -1.0 | -43.3 | -0.2 | ok +27 | -3.17 / -46.1 |
| t1_bedrough | low | -5.35 | +1.66 | +16.5 | +30.9 | -48.1 | -1.8 | ok +36 | -8.22 / -54.3 |
| t1_bedrough | high | -1.80 | +1.11 | +11.9 | +0.8 | -43.5 | -0.5 | ok +31 | -3.17 / -46.1 |
| t3_posting | low | -6.98 | +0.03 | -13.9 | +0.4 | -48.9 | -2.5 | FAIL +4 | -8.22 / -54.3 |
| t3_posting | high | -2.20 | +0.71 | +11.5 | +0.4 | -47.1 | -4.0 | ok +27 | -3.17 / -46.1 |

### T2 attenuation 15 -> 31 dB/km (`t2_att31`, 32.1 min, all four passes)

`--att 31`. The mapping re-anchors automatically, and the re-anchoring is the
story: **K = +11.39 -> -10.60 dB**, so **K - K_phys = +21.71 -> -0.28 dB** --
at 31 dB/km the median-anchored constant coincides with the physically
derived one, i.e. the 21.7 dB fudge the A = 15 mapping needed disappears.
This independently confirms the ~31 dB/km effective attenuation from
`run_cross_season` on a second line (`implied_eff_att_db_per_km` = 30.8 in
both runs).

**The Gamma^2 > 0 dB fraction does NOT vanish: it is unchanged at 0.189**,
and the bright tail moves UP (segment p95 +11.7 -> +17.4 dB). That prediction
does not survive contact with the mapping: K is set by the segment MEDIAN, so
the median stays pinned at the Fresnel -12.9 dB whatever A is, while the
spread of `2*A*H(s)` grows with A. The unphysical fraction is a property of
the median anchoring, not of the attenuation value.

**Nadir bed level shift: -20.9 / -17.4 / -17.8 / -11.1 dB** (low/mid/high/30
km). This follows analytically: the received bed level goes as
`G2 - 2AH = K - RSSNR(s)`, so it depends on A ONLY through K, and K moved
-22.0 dB. The surface-return decomposition is bit-identical to baseline
(bed-window surface returns -89.83 / -68.86 / -71.95 dB in both), confirming
only the bed layer moved. Consequence: **every level-based tail metric drops
~20 dB and the guard fails on all four passes** -- the total-field tail is now
surface clutter. The interpretable quantity is the level-invariant
**bed-returns-only slope**.

There the result is sharp: **low -7.01 -> -7.74 dB/us (-0.73)**, closing 60%
of the low pass's slope deficit vs measured (-8.22). That is exactly the
attenuation-obliquity term predicted in the tail-metric note: over the low
pass's fit window the refracted path angle runs 13.2 -> 27.9 deg, adding
2*H*(1/cos(phi) - 1) = 0.143 km of two-way ice path, worth 2.15 dB at 15
dB/km and 4.44 dB at 31 -- a predicted +0.76 dB/us steepening against +0.73
measured. At mid/high the same term is worth only 0.08-0.09 dB/us (their
tails never leave 10 deg) and the measured slopes move the other way
(-4.90 -> -4.58, -2.91 -> -2.36): the wider G2 dynamic range re-weights the
trace ensemble toward the bright half of the line, whose arcs reach further.

**Verdict: attenuation is the low pass's missing decay and nothing else's.**
It cannot touch the altitude-growing excess, and as configured it costs a 20
dB bed-level collapse that no longer matches the measured bed brightness. A
level-preserving variant (A = 31 with K held at its A = 15 value) is the
obvious follow-up and was NOT run.

### T4 antenna pattern (`t4_isotropic` + `t4_array8`)

Worst case first: `--antenna isotropic` removes the 5-element cross-track
array factor entirely. The tail metrics move a LOT -- excess at +2 us
+22.5 / +5.7 / +5.5 dB (low/mid/high) -- so by the brief's own rule the
hypothesis is not bounded small and the more-directive bracket was run too.

But the decomposition says exactly WHERE the movement is:

| quantity, bed window | low | mid | high | 30 km |
|---|---|---|---|---|
| surface returns, baseline -> isotropic | -89.8 -> -67.0 (**+22.9**) | -68.9 -> -46.3 (**+22.6**) | -72.0 -> -48.1 (**+23.9**) | -55.4 -> -44.5 (**+11.0**) |
| bed returns, baseline -> isotropic | -46.4 -> -44.2 (+2.2) | -43.5 -> -40.7 (+2.8) | -43.1 -> -41.3 (+1.8) | -42.7 -> -41.9 (+0.8) |

**The pattern is a ~23 dB control on far off-nadir SURFACE clutter and a
~2 dB control on bed returns.** Bed returns arrive within a few degrees of
nadir where the array factor is flat, so they barely notice it; the surface
clutter that fills the record comes from tens of degrees off-nadir where the
array is the only thing suppressing it. Mid-column clutter rises +23.6 /
+7.3 / +5.9 dB, and the guard fails on every pass (surface returns are
brought up to within 2-8 dB of the bed returns).

Since the baseline guard PASSES at mid/high with +22 and +27 dB of margin,
the +11..+17 dB bed-return excess at altitude is measured on genuinely
bed-dominated tails, and a 2 dB pattern sensitivity on bed returns cannot
account for it.

### T1 sub-facet bed roughness (`t1_bedrough`, 42.7 min, low + high)

**Choosing sigma/l from the data.** Structure function of the LOW pass's own
bed picks over the 50 km segment (14.85 m sampling): rms deviation 4.14 m at
30 m lag rising to 113 m at 3.8 km, a clean self-affine law
**sigma(L) = 0.580 * L^0.662 m (Hurst H = 0.66)** fitted over the resolved
59-3803 m band. Extrapolated to the wavelength scale that Gerekos sub-facet
roughness represents this gives **sigma = 0.535 m at l = lambda_ice (0.886 m)**
and 1.12 m at l = 2.7 m -- i.e. **2.4-5x beyond the Gerekos comfortable
ceiling** (sigma = lambda_ice/4 = 0.222 m, where the paper's own validation
measures ~1 dB error; accuracy degrades past ~0.4 lambda). The test was
therefore run AT the ceiling, **sigma = 0.22 m, l = 0.886 m** (l = 1
lambda_ice: <= facet size everywhere -- 10.7 m at the low pass, 49.8 m at the
high -- and giving a diffuse lobe of half-width ~atan(sqrt(2) sigma/l) ~ 19
deg, which is the angular band the tail window probes). The data-implied
roughness is LARGER, so the measured effect below is a LOWER bound.

**The double-count guard needed an empirical term, and finding out why is a
result.** The analytic guard raises G2 by the nadir coherent-term attenuation
`exp(-sigma^2 K^2)` = 42.26 dB. A one-chunk-class pilot (10 km, low pass,
`t1_pilot_base` vs `t1_pilot_rough`) showed that overshoots by **+39.0 dB**:
the bed window went -48.29 -> -9.28 dB. The reason is physical -- at
sigma = lambda/4 the coherent term is annihilated but the INCOHERENT term
inherits nearly all of it, so the true nadir mean-power loss is only
**~3.3 dB**, not 42.3. The full run therefore used
`--bed-rough-extra-db -39.0` (net G2 shift +3.26 dB), and the conservation
check on the full segment is **-1.75 dB (low) and -0.49 dB (high)** -- high
inside the ~1 dB target, low slightly outside because the calibration came
from a 10 km sub-segment with a different bed-brightness mix. Both recorded
in `run_config.bed_roughness.gamma_double_count_guard`.

**Result: roughness moves the tail the WRONG way, hard.** With the nadir
level held, the low pass's bed-return tail rises from -71.9/-80.1/-85.7 to
**-42.8/-48.7/-53.7 dB** at bed+1/2/3 us (+29 dB) and its slope flattens
**-7.01 -> -5.35 dB/us**; the high pass flattens **-2.91 -> -1.80**. Measured
is -8.22 and -3.17. Adding lambda-scale bed roughness makes the simulated bed
LESS specular and its tail flatter, which is the opposite of the measured
behaviour. Surface returns are bit-identical (-89.83 / -71.95 dB), a clean
one-variable control.

**The useful by-product is a bracket on the real bed's wavelength-scale
roughness.** At the low pass the measured tail sits BETWEEN the smooth-bed
sim (14 dB below measured at +2 us) and the sigma = 0.22 m sim (16 dB above),
so the effective lambda-scale roughness is somewhere in between -- a naive
sigma^2 (weak-scattering) interpolation puts it near **3-4 cm**, an order of
magnitude below the 0.53 m the pick spectrum extrapolates. Caveat: the
measured post-bed tail also carries englacial and off-nadir surface
scattering that this surface+bed model excludes by design, so 3-4 cm is an
UPPER bound. Either way the self-affine extrapolation from km-scale picks
badly over-predicts the roughness the radar actually sees at 0.9 m, which is
the physically interesting finding.

### T3 aperture / posting (`t3_posting`, 37.6 min, low + high)

`--posting-div 2`: the SIM along-track grid is refined to **7.43 m** (6729 /
6739 traces vs 3365 / 3370) while the measured frame is untouched, so the
alias-limited aperture doubles -- **87 -> 175 m (low)** and **627 -> 1255 m
(high)**, half-angle 1.52 -> 3.05 deg, hann azimuth resolution **21.4 ->
10.7 m** (the product's effective ~25 m). Fast-time grid, facet spacing,
reach, bed and gamma are bit-identical to baseline.

**The prediction does not hold: the tail excess does not drop.** At +2 us it
moves **-14.4 -> -13.9 (low)** and **+11.1 -> +11.5 (high)** -- 0.4-0.5 dB in
the WRONG direction. Everything falls by a common-mode 2.5-4 dB (bed returns
-2.5 / -4.0, surface returns -3.4 / -2.5), so sim-minus-measured is
preserved. The low pass's bed-return slope is unchanged to **+0.03 dB/us**.

Physically this is the right answer for a nadir-looking sounder: the post-bed
tail is built from **cross-track** off-nadir arrivals, and no along-track
aperture can compress those. The along-track share of the tail excess is
~0.5 dB.

**Caveat on the high pass.** Doubling the aperture quadruples the
air-only-focusing residual phase (recorded gap g3: ~1 rad at the baseline
aperture edge at altitude -> ~4 rad here), so the high pass's +0.71 dB/us
slope flattening is plausibly a focusing artifact of pushing g3 past its
validity rather than physics. The LOW pass (87 -> 175 m, g3 negligible) is
the clean measurement, and it says +0.03 dB/us.

### T4b more-directive bracket (`t4b_array8`, 19.2 min, low + high)

Because the isotropic delta was large, the other side was run: the same
0.5-lambda cross-track array with **8 elements instead of 5** (1.6x aperture)
-- an honest bracket for the fact that real elements (dipoles over structure)
always make the true pattern MORE directive than the bare 5-element array
factor the baseline uses. It is a bracket, not a claim about the real
antenna.

**Everything moves by <= 0.44 dB**: surface returns in the bed window -0.33 /
-0.44 dB, bed returns -0.32 / -0.27, bed-return slope +0.03 / +0.09 dB/us,
excess at +2 us -0.2 / -1.0 dB. So the pattern sensitivity is strongly
ASYMMETRIC: deleting the pattern entirely (physically implausible) costs 23
dB of surface-clutter suppression, while sharpening it in the plausible
direction changes the bed-return tail by ~1 dB. **Pattern fidelity is bounded
as a MINOR contributor to the bed-return excess** -- which is the question
that was asked -- while remaining a first-order control on surface clutter.

### Which hypotheses improved the metrics?

| test | low pass | high pass | verdict |
|---|---|---|---|
| T2 att 31 | slope **-0.73** toward measured (60% of the deficit) | slope +0.55 away | PARTIAL, low only; costs a 17-21 dB bed-level collapse |
| T4 isotropic | slope +1.30 away, excess +22.5 | slope +0.22 away, excess +5.5 | WORSE (worst-case bound only) |
| T4b array8 | +0.03 / -0.2 | +0.09 / -1.0 | NULL (<= 1 dB) |
| T1 bed roughness | slope +1.66 away, excess **+30.9** | slope +1.11 away, excess +0.8 | WORSE, decisively |
| T3 posting/aperture | +0.03 / +0.4 | +0.71 / +0.4 (g3-caveated) | NULL (~0.5 dB) |

**Not one hypothesis reduces the altitude-growing excess.** The only metric
that improved anywhere is the LOW pass's tail slope under T2, and that is the
attenuation-obliquity term already predicted analytically in the tail-metric
note -- it is worth 0.7-0.8 dB/us at 13-28 deg and ~0.1 dB/us at 4-10 deg, so
it is structurally incapable of explaining a gap that GROWS with altitude.
Three candidate mechanisms are now bounded and eliminated for the high pass:
aperture (~0.4 dB), plausible pattern error (~1 dB), and sub-facet bed
roughness (wrong sign, and the data-extrapolated roughness is even larger).
The surviving suspects are the ones this campaign did not vary: the RSSNR
gamma's recorded bright-end overshoot (+5-10 dB where G2 > 0 dB, a LEVEL
error that would propagate straight into the tail), the absent englacial /
volume scattering (which raises the MEASURED curve, not the sim, and so
cannot close a sim-too-hot gap), and the level-preserving attenuation variant
(A = 31 with K pinned at its A = 15 value) that separates T2's two effects.
That last one is the cheapest next test and was NOT run.

Timings (wall, this machine, 24 cores, no GPU): T2 32.1 min (all 4 passes),
T4 isotropic 40.0 (4 passes; its low pass overlapped the T1 pilots, so ~25%
inflated), T1 pilots 5.9 + 12.8 (10 km, low only), T1 42.7 (low+high, the
rough branch costs **2.24x** smooth -- 133 vs 60 s/chunk), T3 37.6 (low+high,
2x traces), T4b 19.2 (low+high). **Campaign total ~3.3 h of simulation**,
inside the 3.5 h budget. Tests: 276 unit green (`tests/test_basal_hypotheses.py`
adds sim_cfg wiring for all three antenna variants and bed roughness, the
cache-key backward-compat lock, the Gerekos nadir-attenuation math, the
gamma-offset/re-anchoring algebra, `upsample_fsub` endpoint/midpoint exactness
and the aperture-doubling identity); ruff clean.
