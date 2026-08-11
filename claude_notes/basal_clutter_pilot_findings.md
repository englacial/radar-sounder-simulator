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

## T5: angle-dependent (specular + diffuse) bed reflectivity (2026-08-03)

The T1 falsification said the tail reshaping cannot come from Kirchhoff
roughness. This implements the alternative the user approved: the
RSSNR-mapped per-facet bed reflectivity |Gamma_bed|^2(x) is SPLIT into

* **specular** `f_s * |Gamma|^2 * G(psi)`, `G(psi) = exp(-tan^2(psi)/(2 s0^2))`
  with psi the facet tilt from horizontal -- "bright because flat"
  (hydraulically ponded water is flat; a facet tilted psi mirrors the
  nadir-looking radar to 2 psi off-nadir and must not inherit the
  nadir-calibrated brightness). Folded into the existing per-facet gamma
  grid: no kernel change.
* **diffuse** `(1 - f_s) * |Gamma|^2 * cos^n(theta_i)`, a NEW incoherent
  per-facet channel in `kernels/multilayer.py` mirroring the gamma-facet and
  roughness-phasor patterns (per-facet amplitude array + frozen unit
  phasors; `cos_t` and `spread` are already in-kernel, `n` is traced).

### Normalization gate (derived, then measured)

The diffuse per-facet amplitude convention is DERIVED, not fitted:

    a_diff = sqrt(A/(2 pi)) * amp * cos_t^(1 + n/2) * spread * att_f

The prefactor `sqrt(A/(2 pi))` is fixed by requiring the split to conserve
total nadir power over a flat interface. The specular coherent sum is the
image-method mirror field -- integrating `(k/2pi) Gamma cos_t spread` over a
flat plane at range r by stationary phase gives `|E|^2 = Gamma^2/(4 r^2)`.
The incoherent sum of the diffuse channel over the same plane is
`sum_i a_i^2 = (1/(2 pi)) amp^2 Int (cos_t spread)^2 dA`, and
`Int (cos_t / r'^2)^2 dA = pi/(2 r^2)` exactly, so `sum_i a_i^2 =
amp^2/(4 r^2)`. With `amp^2 = (1-f_s) Gamma^2` the two channels sum to
`Gamma^2/(4 r^2)` for ANY f_s -- independently of facet size, range and
wavenumber (the k/2pi prefactor cancels, so the diffuse channel is
frequency-flat, as a sigma^0 law should be).

Measured against that closed form (`tests/test_multilayer_diffuse.py`,
flat plane, 6 phasor seeds):

| check | result |
|---|---|
| diffuse total vs `amp^2/(4 r^2)`, 4 m facets | **+0.11 dB** |
| same, 8 m facets | +0.07 dB |
| same, 16 m facets | -0.38 dB (coarse facets under-resolve the angular integral) |
| facet-size independence (4 m vs 8 m) | within 0.10 (test tolerance) |
| cos^n law, single facet, 0-56 deg incidence | exact to 1e-6 |
| power split linearity in (1-f_s) | exact to 1e-6 |
| `diffuse=None` / `f_s=1, s0=0` | traces the pre-feature program; grids bit-identical |

### Trial A (RECORDED, REJECTED): the tilt weight needs the same
### double-count guard the T1 roughness needed

The first pilot pass used `G(psi)` as an absolute weight with the
user-suggested `s0 ~ 1 deg`. It annihilated the bed: this bed's own tilt
distribution on the 32 m scene grid is **median 6.62 deg, p90 13.77 deg,
max 29.9 deg**, so the median facet got `G = -96 dB` and the bed layer fell
**20 dB (mid/high) to 38 dB (low)**. J_shape got WORSE (7.08 -> 11.00).

The fix is the T1 lesson applied again: the RSSNR mapping is calibrated
against the MEASURED bed echo, which already contains the real bed's tilt
mix, so `G(psi)` must enter as a RELATIVE reweighting with unit scene mean,
never as an absolute loss. The tool now divides by `<G>` (recorded as
`mean_normalization_db`: -6.24 dB at s0 = 3 deg, -14.52 at s0 = 1,
-2.50 at s0 = 6.6). With `<G> = 1` the split conserves scene-mean bed power
for every (f_s, s0) by construction (unit-tested).

Residual bed-WINDOW conservation at the pilot scale (bed layer, dB vs the
unsplit run; the window is a narrow delay gate, so the diffuse channel's
delay spreading costs a little even when the total is conserved):

| config | low | mid | high |
|---|---|---|---|
| f_s 1.0, s0 3 | -2.39 | +0.23 | +1.35 |
| f_s 0.9, s0 3 | -2.10 | -0.22 | +0.96 |
| **f_s 0.5, s0 3** | **-1.55** | **-2.47** | **-1.11** |
| f_s 0.9, s0 6.6 | +0.70 | +0.56 | +0.87 |
| f_s 0.0 (all diffuse) | -6.15 | -10.47 | -8.98 |
| f_s 0.9, s0 1 (rejected) | -11.42 | -7.40 | -6.78 |

So the ~1 dB target is met for `s0 = 6.6`, met to 1-2.5 dB for the
`s0 = 3` family, and fails in the pure-diffuse limit -- reported, not tuned.

### s0 and n

* **n = 1** (near-Lambert). Deliberate and almost inconsequential: over the
  fit window the bed incidence angle only reaches 4-10 deg (mid/high) and
  13-28 deg (low), where `cos^1` costs 0.01-0.5 dB. n is NOT a useful shape
  knob here and was not scanned; the shaping comes from `G(psi)` and from
  the specular/diffuse balance.
* **s0 = 3 deg**, chosen between the user's ~1 deg and this bed's own 6.62
  deg median tilt, and then confirmed by trial: it puts the normalized
  weight at p10 -41 dB / median -4.4 dB / max +6.2 dB -- a real
  flat-vs-tilted contrast that is neither a filter (s0 = 1: p10 -413 dB,
  median -82 dB) nor a no-op (s0 = 6.6: -7.2 to +2.5 dB). Both alternatives
  were run and scored (below).

### The objective, and why the brief's version cannot be used alone

    J_abs   = mean_passes(|excess(bed+2 us)| + |slope_sim - slope_meas| * 1 us)
    J_shape = mean_passes(|dR2|             + |slope_sim - slope_meas| * 1 us)
    R2 = tail(bed+2 us) - bed-window level   (the tail against the run's OWN bed peak)

Equal weights, both terms in dB (the slope misfit is multiplied by 1 us).
`J_abs` is the brief's objective, and it is reported -- but with `--att 31`
now the default its first term carries the KNOWN absolute-level offset (the
median-anchoring caveat: the simulated bed sits 6-16 dB below measured
before any splitting), which no bed-angular model can remove. `R2` and the
slope are both invariant to a constant reflectivity/attenuation error, so
**J_shape scores exactly what this model is meant to fix** and selects the
winner. Both are tabulated.

### The pilot scan (10 km, three real passes, DEMOGORGN bed, att 31, n = 1)

Every trial recorded, best first. `dR2` and `dSlope` are sim minus measured
(dB and dB/us); `guard` is the sim bed-returns-minus-surface-returns margin.

| config | J_shape | J_abs | dR2 low/mid/high | dSlope low/mid/high | guard low/mid/high |
|---|---|---|---|---|---|
| **f_s 0.5, s0 3 (WINNER)** | **4.17** | 13.23 | +1.47 / +2.88 / -0.84 | +4.72 / +2.17 / -0.43 | -1.0 / -10.7 / -6.0 |
| f_s 0.9, s0 3 | 4.32 | 14.59 | -1.15 / +1.45 / -1.61 | +5.74 / +1.67 / -1.35 | -7.9 / -14.6 / -8.2 |
| f_s 1.0, s0 3 (tilt weight only) | 4.94 | 15.26 | -2.38 / +1.17 / -1.70 | +6.49 / +1.53 / -1.55 | -28.8 / -16.1 / -9.3 |
| f_s 0.9, s0 6.6 | 5.92 | 12.60 | -3.88 / +3.25 / +2.27 | +5.69 / +1.50 / -1.16 | -7.9 / -6.4 / -2.2 |
| f_s 0.9, s0 1 | 6.88 | 15.95 | +6.59 / +3.81 / -0.81 | +5.76 / +2.81 / +0.87 | -7.9 / -17.6 / -14.4 |
| unsplit (current model) | 7.08 | 12.85 | -4.56 / +4.15 / +3.65 | +6.28 / +1.90 / -0.70 | -27.2 / -4.0 / -1.1 |
| f_s 0.0 (pure diffuse) | 9.23 | 13.15 | +8.64 / +6.93 / +3.62 | +4.36 / +2.96 / +1.18 | +2.0 / -8.2 / -4.9 |
| f_s 1.0, s0 1, UNNORMALIZED (trial A) | 11.00 | 17.03 | -- (bed annihilated, 20-38 dB) | | |

**The split helps, and both ends are excluded.** J_shape falls from 7.08
(unsplit) to 4.17, a **2.9 dB** improvement in the joint shape misfit;
pure diffuse is 5.1 dB WORSE than the winner and 2.2 dB worse than not
splitting at all, so the data require a specular component; the tilt weight
alone (f_s = 1) already buys 2.1 dB of the 2.9. `J_abs` barely moves
(12.85 -> 13.23) exactly as expected: it is dominated by the absolute-level
offset the model cannot touch.

**f_s is only weakly identified.** J_shape is flat between f_s 0.5 and 0.9
(4.17 vs 4.32, i.e. 0.15 dB) and rises steeply outside it. The honest
statement is **f_s = 0.5-0.9 with s0 ~ 3 deg**; f_s = 0.5 is the point
estimate carried forward, f_s = 0.9 is a recorded neighbour that was NOT
validated at full scale (compute).

### Full 50 km validation of the winner (`t5_specdiff_fs0.5`, 32.3 min)

f_s = 0.5, s0 = 3 deg, n = 1, DEMOGORGN bed, RSSNR gamma, matched
processing, att 31, all three real passes + the 30 km prediction. Reference
is the same configuration WITHOUT the split (`t2_att31`, identical in every
other respect).

| pass | dSlope unsplit -> T5 (dB/us) | dR2 unsplit -> T5 (dB) | excess+2 (dB) | bed window vs unsplit |
|---|---|---|---|---|
| low 449 m | +5.80 -> **+3.51** | -10.99 -> **+3.02** | -24.0 -> -12.9 | -2.90 dB |
| mid 9150 m | +1.06 -> +1.13 | +2.65 -> **+0.82** | -12.2 -> -15.0 | -0.96 dB |
| high 10684 m | +0.88 -> **-1.03** | +4.39 -> **+0.63** | -10.3 -> -15.1 | -0.94 dB |
| **J_shape (full segment)** | | **8.59 -> 3.38** | | |

**One scene-constant f_s DOES satisfy all three altitudes on the tail-shape
level.** The tail-vs-bed-peak residual `dR2` collapses from
(-11.0, +2.7, +4.4) dB to **(+3.0, +0.8, +0.6) dB** -- three altitudes
spanning a 24x range in height, one number, all within ~3 dB and two within
1 dB. That is the over-determination test passing on the observable the
model was built for, and it is the strongest evidence so far that the bed's
angular reflectivity, not its topography or its Kirchhoff roughness, is what
the tail was missing.

**It does NOT fix the slopes, and the residual pattern is diagnostic.**
dSlope goes (+5.80, +1.06, +0.88) -> (+3.51, +1.13, -1.03): the high pass
now slightly over-steepens, mid is unchanged, and the low pass is still
+3.5 dB/us too flat -- by far the largest residual left. The guard says why:
at the low pass the sim's post-bed tail is only +7.1 dB above its own
SURFACE returns (and -3.3 / +1.1 dB at mid/high), i.e. a large part of what
the metric is measuring is surface clutter, whose slope no bed-reflectivity
parameter can change. The residual therefore implicates **the absolute bed
level** (the att = 31 / median-anchoring choice leaves the simulated bed
13-15 dB below measured, so the surface-clutter pedestal is relatively too
strong) and, second, the surface-clutter model at low altitude -- NOT f_s,
s0 or n.

### Revised 30 km verdict (the design question)

| model | bed returns - surface returns in the bed window | bed peak over mid-column |
|---|---|---|
| picked bed, att 15 (originally reported) | **+11.4 dB** | +5.0 dB |
| DEMOGORGN, att 31, unsplit | **-7.4 dB** | -0.2 dB |
| DEMOGORGN, att 31, T5 f_s 0.5 | **-8.6 dB** | -0.2 dB |

**The +11.4 dB margin does not survive.** Under the currently adopted
attenuation (31 dB/km) the 30 km bed sits 7.4 dB BELOW the surface clutter
arriving at the same delay, and the specular/diffuse split takes another
1.2 dB off (it moves bed power out of the specular apex into a diffuse
pedestal spread over delay). The bed peak no longer stands above the
mid-column clutter at all (-0.2 dB). Read honestly, the two rows bracket the
answer: **+11.4 dB (att 15) to -8.6 dB (att 31 + T5)**, and the bracket is
set by the unresolved ABSOLUTE bed level, not by the angular model -- the
same 20 dB of attenuation/anchoring ambiguity that T2 exposed. On this line
the stratospheric bed is therefore "marginal to clutter-limited" rather than
"visible with 11 dB to spare", and resolving it requires pinning the
absolute bed level (a level-preserving A = 31 variant with K held at its
A = 15 value is the cheapest next test, still not run).

### What is now wired

`--specular-fraction F_S [--spec-tilt-deg S0] [--diffuse-exponent N]` on
`tools/run_basal_clutter.py` (off by default; requires `--gamma-from-rssnr`),
`SimConfig.diffuse_exponent`, `scene.diffuse_maps` (same grid convention as
`gamma_maps`), and `refracted_cluttergram(..., diffuse=(amp, phasors,
n_exp))`. Every knob is absent from the chunk cache key when off, so all
pre-T5 caches stay valid. Tests: `tests/test_multilayer_diffuse.py` (7: the
analytic normalization gate, facet-size independence, exact cos^n law, split
linearity, OFF-is-untouched, mode/shape guards) and 5 more in
`tests/test_basal_hypotheses.py` (tilt math, the mean-normalization
double-count guard, flat-vs-tilted contrast, scene-mean power conservation,
cache-key isolation). Suite 289 unit green; ruff clean.

Timings: pilot scan 7 configs x ~9 min = 63 min (plus ~20 min spent on the
rejected trial A), full validation 32.3 min (low 1028.8 / mid 518.7 / high
136.5 / syn30km 251.0 s; the diffuse channel costs ~+2% wall). Total ~2 h of
simulation.

## Thickness regression for attenuation (2026-08-06): inconclusive on-line

RSSNR vs H along the anchor line (cached rssnr_anchor.npz; figure
outputs/basal_clutter/rssnr_thickness_regression.png):

- FULL line (111 traces, H 497-1063 m): OLS A = 30.5 +/- 5.2 dB/km
  (Theil-Sen 21.7 [7.4, 33.9]) - numerically reproducing the calibrated
  31, BUT the leverage comes from the floating section (s > 68 km,
  corr(H, RSSNR) = 0.57 there), where the bed reflectivity regime is
  entirely different (ice-ocean interface). Partly coincidental.
- GROUNDED segment only (s <= 68 km, H 554-883 m): A = 2.8 +/- 6.9,
  r = 0.07 - NO thickness signal. The 325 m thickness span gives only
  ~20 dB of expected attenuation dynamic at A=31, swamped by the 13 dB
  rms (40 dB range) reflectivity variance, plus plausible Gamma-H
  anticorrelation (thick troughs holding bright water flattens the
  slope).

Verdict: on a single 50 km grounded segment this regression CANNOT
separate attenuation from reflectivity; the evidence for ~31 remains
the three prior routes (repeat-pass calibration, K-K_phys, obliquity
slope). The statistically sound version is the same regression across
the full 5,646-frame RSSNR dataset (grounded-only masks, km-scale H
dynamic range, reflectivity variance averaging down) - a dataset-level
analysis, recorded as the recommended follow-up.

## Attenuation sweep: 15 / 20 / 26 / 31 dB/km (2026-08-06)

Four identically-configured points (full 50 km, all four passes, DEMOGORGN
bed, RSSNR gamma, matched CSARP processing, **UNSPLIT** -- no T5 flags) so
the progression is directly comparable end to end: `hypothesis_tests/`
`baseline` (15) / `att20` / `att26` / `t2_att31` (31). Each new value-dir
carries `radargrams.png`, `decomposition.png`, `bed_tail.png`,
`metrics.json`, `run_config.json`, mirrored to
`outputs/verification/basal_clutter_att{20,26}/`. A T5-split sweep would
compose on top of this (`--specular-fraction` is orthogonal to `--att`) and
was not run.

### The mapping re-anchors at every value (median anchoring)

K falls 22 dB across the sweep while the median |Gamma|^2 stays pinned at
the Fresnel -12.9 dB by construction:

| att dB/km | K dB | K - K_phys | G2 seg p95 / max dB | G2 > 0 frac | implied eff att |
|---|---|---|---|---|---|
| 15 | +11.39 | +21.71 | +11.7 / +15.9 | 0.189 | 30.8 |
| 20 | +4.36 | +14.69 | +13.4 / +16.5 | 0.162 | 30.7 |
| 26 | -3.73 | +6.59 | +15.7 / +17.7 | 0.189 | 30.8 |
| 31 | -10.60 | -0.28 | +17.4 / +19.0 | 0.189 | 30.8 |

Two things worth stating plainly. **The unphysical-reflectivity fraction is
insensitive to A** (0.162-0.189, no trend): it is a property of the median
anchoring, not of the attenuation. And **`implied_eff_att` is 30.7-30.8 at
every value** -- it is a data quantity (K - K_phys divided by 2 H_med) that
does not know what A the run used, so it is not independent evidence for
A = 31; it is the same single measurement re-expressed.

### The sweep table (sim vs measured, per pass)

| att dB/km | K dB | K-K_phys | G2>0 frac | pass | bed window sim / meas | tail slope sim / meas | excess +2 us | guard |
|---|---|---|---|---|---|---|---|---|
| **15** | +11.39 | +21.71 | 0.189 | low | -46.4 / -54.3 | -7.01 / -8.22 | -14.4 | FAIL +4 |
| | | | | mid | -43.4 / -46.0 | -4.90 / -5.15 | +7.6 | ok +22 |
| | | | | high | -43.0 / -46.1 | -2.91 / -3.17 | +11.1 | ok +27 |
| **20** | +4.36 | +14.69 | 0.162 | low | -53.0 / -54.3 | -7.25 / -8.22 | -20.2 | FAIL -4 |
| | | | | mid | -49.9 / -46.0 | -4.81 / -5.15 | +0.8 | ok +15 |
| | | | | high | -49.5 / -46.1 | -2.72 / -3.17 | +4.0 | ok +20 |
| **26** | -3.73 | +6.59 | 0.189 | low | -60.7 / -54.3 | -7.54 / -8.22 | -23.3 | FAIL -13 |
| | | | | mid | -56.4 / -46.0 | -4.68 / -5.15 | -6.7 | FAIL +7 |
| | | | | high | -56.0 / -46.1 | -2.52 / -3.17 | -4.0 | ok +12 |
| **31** | -10.60 | -0.28 | 0.189 | low | -67.3 / -54.3 | -7.74 / -8.22 | -24.0 | FAIL -21 |
| | | | | mid | -60.8 / -46.0 | -4.58 / -5.15 | -12.2 | FAIL +1 |
| | | | | high | -60.9 / -46.1 | -2.36 / -3.17 | -10.3 | FAIL +5 |

| att dB/km | 30 km bed - surface returns (bed window) | bed peak over mid-column | bed window | mid-column |
|---|---|---|---|---|
| 15 | **+12.72** | +6.40 | -41.4 | -33.9 |
| 20 | **+6.19** | +2.90 | -45.6 | -34.0 |
| 26 | **-1.14** | +0.91 | -49.9 | -34.0 |
| 31 | **-7.41** | -0.20 | -52.5 | -34.0 |

### What the progression says

* **The measured bed BRIGHTNESS is reproduced at att ~ 17-21 dB/km, not
  31.** Interpolating the bed-window level onto the measured value gives
  **17.0 (mid), 17.4 (high), 21.0 (low) dB/km** -- a tight, mutually
  consistent bracket from three independent altitudes. At 31 dB/km the
  simulated bed sits 13-15 dB below measured; at 15 it sits 3-8 dB above.
* **The tail SLOPE is a weak discriminator and it pulls the other way.**
  Mean |slope misfit| over the three passes is 0.57 / 0.59 / 0.60 / 0.62
  dB/us at 15 / 20 / 26 / 31 -- essentially flat. Within it the low pass
  improves monotonically with attenuation (1.21 -> 0.48 dB/us, the
  obliquity term of the T2 finding) while mid and high degrade (0.25 ->
  0.57 and 0.26 -> 0.81). Level, not shape, is what this sweep resolves.
* **The tail excess at bed+2 us nulls at att ~ 21 (mid) and ~ 23 (high)**
  (+7.6 -> -12.2 and +11.1 -> -10.3 across the sweep), consistent with the
  bed-window crossings. The low pass never nulls (-14 to -24 dB at every
  value) because its tail is surface-clutter dominated in the sim at every
  attenuation (guard +4 dB at att 15, -21 dB at att 31).
* **The tail stops being a BED measurement as attenuation rises.** The
  mid/high guards go from ok (+22 / +27 dB of bed-over-surface margin at
  15) through ok (+15 / +20 at 20) to FAIL (+1 / +5 at 31). Any tail metric
  quoted at 26-31 dB/km is therefore partly a surface-clutter metric --
  which is exactly what the T5 fit had to work around.
* **The 30 km verdict crosses zero inside the sweep**, at **att ~ 25
  dB/km**: the bed-minus-surface-returns margin in the bed window is
  **+12.7 / +6.2 / -1.1 / -7.4 dB** at 15 / 20 / 26 / 31, and the bed peak
  over mid-column clutter goes +6.4 / +2.9 / +0.9 / -0.2 dB. The
  stratospheric design answer is therefore decided by precisely the
  parameter this sweep shows to be unresolved -- and the brightness
  evidence (17-21) sits on the VISIBLE side of that crossing while the
  K = K_phys argument (30.7) sits on the invisible side.

**The tension in one line:** matching the measured bed brightness wants
att ~ 17-21 dB/km, while making the mapping's absolute chain
self-consistent (K = K_phys) wants ~30.7 -- a ~10-13 dB absolute-level
disagreement that the median anchoring silently absorbs. Nothing in this
sweep resolves which is wrong (the surface model, the system constants and
the true attenuation are degenerate here); a level-preserving variant
(A = 31 with K pinned at its A = 15 value) separates them and is still the
cheapest next test.

### Figures

`outputs/basal_clutter/hypothesis_tests/att_sweep_strip.png` -- the MID
pass across all four values plus the measured panel, cropped from the four
radargram figures (identical layout, grey scale and twtt axis, so they can
be flipped between directly). The bed arcs fade monotonically while the
upper-column surface clutter is unchanged, which is the whole story in one
image. Per-value `radargrams.png` / `decomposition.png` / `bed_tail.png`
share the same scales and layout across the four dirs.

Timings: att20 1965.4 s and att26 1965.0 s wall (low ~1042 / mid ~527 /
high ~138 / syn30km ~257 s each), i.e. 32.8 min per value, matching the
existing endpoints to ~2%; att26 was interrupted once and resumed from its
chunk cache with no loss. Pilot-projection: the cost is geometry-driven and
was taken from the identically-configured `t2_att31` run rather than spent
again, after a 30 s pilot smoke confirmed the unsplit path still runs at a
non-default attenuation post-T5. Suite 289 unit green; ruff clean.

## Level anchoring at A = 31: the discriminator, and what it kills (2026-08-06)

`--anchor median|level` (default `median`, backward compatible) on the RSSNR
mapping. **Rule:** `K_level = K_median + D`, `D = median(measured bed-window
level) - median(simulated bed-window level)` over the three real passes of
the IDENTICALLY configured median-anchored run. The received bed level moves
dB-for-dB with K (received ~ K - RSSNR, independent of A), so one analytic
step replaces an iteration; `D = 14.8 dB` is the recorded att-31 DEMOGORGN
unsplit measurement (per-pass deficits 13.0 / 14.8 / 14.8 dB), and the
post-run residuals are gated in `rssnr_level_anchor`.

Run: full 50 km, four passes, DEMOGORGN bed, A = 31, matched processing,
unsplit -> `outputs/basal_clutter/hypothesis_tests/att31_klevel/`
(radargrams / decomposition / bed_tail / metrics / run_config), mirrored to
`outputs/verification/basal_clutter_att31_klevel/`. **K_median -10.60 ->
K_level +4.20 dB.** Wall 1954.2 s (32.6 min; low 1037.6 / mid 524.6 / high
137.3 / syn30km 254.7).

### (a) Level verification and the implied reflectivity

| pass | sim bed window | measured | residual |
|---|---|---|---|
| low | -52.5 | -54.3 | **+1.78** |
| mid | -49.2 | -46.0 | **-3.21** |
| high | -48.6 | -46.1 | **-2.50** |

Median residual **-2.50 dB**, max |residual| 3.21 -- the analytic anchor
lands close but **outside the ~2 dB target, and the reason is diagnostic**:
the reference deficits were read off bed windows that were partly SURFACE
returns (at A = 31 median-anchored the bed window sits only ~5 dB above its
own surface-return content), so raising K lifts only the bed part and the
window moves ~11.6 dB, not the full 14.8. One refinement iteration
(D = 17.3 dB) would close it; it was not run, and it would make the
diagnostic below stronger, not weaker.

**The implied reflectivity is the point.** Under level anchoring at A = 31
the mapped bed reflectivity is

    |Gamma_bed|^2 segment: min -17.7 / p5 -14.2 / med +1.9 / p95 +32.2 /
    max +33.8 dB,  fraction above 0 dB = 0.541

**The median is +1.9 dB and 54% of the line is above 0 dB.** Said plainly:
matching the measured bed brightness at 31 dB/km requires a bed that returns
MORE power than it receives over more than half the segment. That is
impossible as a pure power reflectivity, so **A = 31 combined with level
matching is quantitatively refuted under the pure-reflectivity
interpretation** -- it would need real focusing or volume gain (a converging
bed, a resonant sub-ice layer) to survive, which nothing else in this study
supports. For contrast the median-anchored A = 31 run has median -12.9 dB
and 18.9% above 0, and A = 20 median-anchored has -12.9 dB and 16.2%.

### (b) Tails once the levels are matched

| run | pass | bed window (d meas) | bed-return slope / meas | excess +2 us | guard |
|---|---|---|---|---|---|
| **att20** (median, A 20) | low / mid / high | +1.3 / -3.9 / -3.3 | -7.25 / -4.81 / -2.72 vs -8.22 / -5.15 / -3.17 | -20.2 / +0.8 / +4.0 | FAIL-4 / ok+15 / ok+20 |
| **att31_klevel** (level, A 31) | low / mid / high | +1.8 / -3.2 / -2.5 | -7.74 / -4.58 / -2.36 | -20.8 / +1.2 / +3.9 | FAIL-6 / ok+16 / ok+20 |
| t2_att31 (median, A 31) | low / mid / high | -13.0 / -14.8 / -14.7 | -7.74 / -4.58 / -2.36 | -24.0 / -12.2 / -10.3 | FAIL-21 / FAIL+1 / FAIL+5 |

**The guards pass again at mid and high** (+16 / +20 dB of bed-over-surface
margin, vs +1 / +5 median-anchored): with the level restored the post-bed
tail is a genuine BED measurement once more, which is exactly what the T5
fit had to work around. Mid and high now satisfy level AND shape reasonably
(excess +1.2 / +3.9 dB, slope within 0.56 / 0.81 dB/us); the low pass does
not (excess -20.8, guard FAIL) because its simulated tail is
surface-clutter-dominated at every attenuation.

### (c) 30 km margin

Bed minus surface returns in the bed window: **+7.39 dB** (bed peak over
mid-column +3.51), against +6.19 for att20, -7.41 for median-anchored A = 31
and +12.72 for A = 15. So the stratospheric verdict tracks the bed LEVEL,
not the attenuation as such: every level-matched world puts the 30 km bed
6-7 dB clear of the co-arriving surface clutter.

### (d) The actual discriminator: A = 20 + Fresnel-prior vs A = 31 + bright bed

The two level-matched worlds have K within 0.16 dB of each other (+4.36 vs
+4.20), so their bed LEVELS are the same by construction and they differ
only in the obliquity shaping (31 vs 20 dB/km of extra in-ice path off
nadir) and in the reflectivity the mapping implies.

* **On the tails they are indistinguishable.** Mean |slope misfit| **0.59
  (att20) vs 0.62 (att31_klevel) dB/us**; mean |excess| 8.33 vs 8.62 dB.
  Per pass the differences swap sign (klevel better at low, 0.49 vs 0.97;
  att20 better at mid/high, 0.34/0.45 vs 0.56/0.81). At 4-28 deg the
  obliquity term is simply too small to separate 20 from 31 dB/km -- the
  tail shape does NOT discriminate.
* **On the implied physics they are not close at all.** A = 20 keeps the
  median bed reflectivity at the Fresnel -12.9 dB with 16% of samples
  unphysical; A = 31 needs +1.9 dB median with 54% unphysical.

**Verdict: the discriminator fires, and it fires against A = 31.** The two
stories fit the observed tails equally well, so the choice must be made on
the implied reflectivity -- and there A = 31 + level matching demands a bed
brighter than a perfect mirror over half the line. Combined with the sweep's
independent brightness evidence (measured bed levels reproduced at 17-21
dB/km), the coherent story is **A ~ 17-21 dB/km with a near-Fresnel bed**,
and the K = K_phys closure at 30.7 dB/km should be read as the absolute
chain (surface model + system constants) being off by ~10-13 dB rather than
as evidence for high attenuation. The remaining low-pass tail deficit
(-20 dB excess, guard FAIL at every attenuation) is untouched by any of
this and stays pinned on the surface-clutter model.

Mechanics: `--anchor level [--level-deficit-db D]`; mode, D, K_median and
K_level recorded in `run_config.rssnr_gamma.level_anchor` and in the
`rssnr_level_anchor` metric together with the post-run per-pass residuals
and the implied-reflectivity block. Anchor mode is NOT in the chunk cache
file name but IS in the cache key (via `rssnr_k_db`), so a same-directory
anchor change forces a correct re-simulation rather than a silent reuse;
variants still get their own `--out-name` directory by convention. Tests:
3 added (`tests/test_basal_hypotheses.py`: K shifts by exactly D with the
profile shape untouched, median mode bit-identical, deficit override,
unknown mode raises, composition with the T1 roughness guard). Suite 292
unit green; ruff clean.

## The level-anchored family: A20 / A26 / A31 with bed brightness matched (2026-08-06)

Level anchoring run at the lower attenuations -- the "moderate attenuation +
measured bed brightness" combination. Same `--anchor level` machinery, full
50 km, four passes, DEMOGORGN bed, matched processing, unsplit;
`hypothesis_tests/att20_klevel/` and `att26_klevel/` (radargrams /
decomposition / bed_tail / metrics / run_config), mirrored to
`outputs/verification/basal_clutter_att2{0,6}_klevel/`.

### D per member, and the contamination correction the A31 run taught us

The A31 level run missed its target by -2.5 dB because the deficit had been
read off a bed window that was ~3 dB surface-return contaminated: raising K
lifts only the BED layer, so the window moves less than D. Both new members
therefore use a **contamination-aware deficit**, solved analytically per
pass from the recorded bed-window decomposition,

    bed * 10^(D/10) + surface = measured   ->   D_clean = 10 log10((P_meas
    - P_surf) / P_bed)

and then the median over the three real passes. No extra simulation was
needed for this refinement (it is a closed form on numbers already on
record), so the one allowed iteration was not spent.

| member | contamination in the reference window (low/mid/high) | D_clean per pass | **D used** | median-of-deficits (naive) |
|---|---|---|---|---|
| att20_klevel | -0.01 / +0.22 / +0.23 dB (clean) | -1.28 / +4.11 / +3.56 | **+3.56** | +3.34 |
| att26_klevel | +0.00 / +1.18 / +1.20 dB | +6.39 / +11.63 / +11.03 | **+11.03** | +9.84 |
| att31_klevel (already run) | +0.04 / +3.32 / +2.71 dB | +13.02 / +18.15 / +17.44 | 14.8 (naive) | +14.74 |

The correction works: **median bed-window residual +0.07 dB (att20_klevel)
and +0.04 dB (att26_klevel)**, against -2.50 for the naive A31 run. Applying
the same rule to A31 would want D = +17.4 dB (not 14.8), which would push
its implied median reflectivity from +1.9 to +4.6 dB and its unphysical
fraction from 0.541 to 0.595 -- i.e. the A31 refutation below is if anything
understated. That refinement was not run (the conclusion does not turn on
it).

### The family table

| member | D dB | K dB | med G2 | p95 G2 | G2>0 frac | pass | bed-win residual | slope sim/meas | excess +2 us | guard |
|---|---|---|---|---|---|---|---|---|---|---|
| **A20 median (reference)** | -- | +4.36 | -12.9 | +13.4 | 0.162 | low | +1.27 | -7.25 / -8.22 | -20.2 | FAIL -4 |
| | | | | | | mid | -3.91 | -4.81 / -5.15 | +0.8 | ok +15 |
| | | | | | | high | -3.34 | -2.72 / -3.17 | +4.0 | ok +20 |
| | | | | | | **summary** | median -3.34 | mean abs 0.59 | | |
| **A20 level** | 3.56 | +7.92 | -9.3 | +16.9 | 0.270 | low | +4.83 | -7.25 / -8.22 | -17.7 | FAIL -0 |
| | | | | | | mid | -0.48 | -4.81 / -5.15 | +4.3 | ok +19 |
| | | | | | | high | +0.07 | -2.72 / -3.17 | +7.6 | ok +24 |
| | | | | | | **summary** | median +0.07 | mean abs 0.59 | | |
| **A26 level** | 11.03 | +7.30 | -1.8 | +26.7 | 0.459 | low | +4.64 | -7.54 / -8.22 | -18.6 | FAIL -2 |
| | | | | | | mid | -0.56 | -4.68 / -5.15 | +4.0 | ok +18 |
| | | | | | | high | +0.04 | -2.52 / -3.17 | +7.0 | ok +23 |
| | | | | | | **summary** | median +0.04 | mean abs 0.60 | | |
| **A31 level** | 14.8 | +4.20 | +1.9 | +32.2 | 0.541 | low | +1.78 | -7.74 / -8.22 | -20.8 | FAIL -6 |
| | | | | | | mid | -3.21 | -4.58 / -5.15 | +1.2 | ok +16 |
| | | | | | | high | -2.50 | -2.36 / -3.17 | +3.9 | ok +20 |
| | | | | | | **summary** | median -2.50 | mean abs 0.62 | | |

| member | 30 km bed - surface returns | bed peak over mid-column |
|---|---|---|
| A20 median (reference) | **+6.19** | +2.90 |
| A20 level | **+9.75** | +4.57 |
| A26 level | **+9.89** | +4.68 |
| A31 level | **+7.39** | +3.51 |

### The physicality axis

| member | median \|Gamma\|^2 | p95 | fraction > 0 dB | reading |
|---|---|---|---|---|
| A20 median (reference) | -12.9 dB | +13.4 | 0.162 | Fresnel prior by construction |
| **A20 level** | **-9.3 dB** | +16.9 | **0.270** | **physical: 4.7 dB below unity** |
| A26 level | -1.8 dB | +26.7 | 0.459 | bright but still sub-unity |
| A31 level | +1.9 dB | +32.2 | 0.541 | **impossible as pure reflectivity** |

A clean monotone gradient, as predicted. **The unphysical fraction crosses
one half between A26 (0.459) and A31 (0.541)** -- linearly, at
**att ~ 28.5 dB/km**. Above that attenuation, matching the measured bed
brightness requires most of the line to reflect more power than it receives.

### The bed WINDOW and the bed TAIL want different K

Worth stating because it is the one new physical result here. Level
anchoring fixes the bed window (residuals -0.5 / +0.1 dB at mid/high) but
makes the tail excess at bed+2 us WORSE: A20 goes from +0.8 / +4.0
(median-anchored) to +4.3 / +7.6 dB (level-anchored), and A26 behaves
identically. The window is dominated by the near-nadir specular apex and the
tail by off-nadir returns, so a single K cannot satisfy both: the simulated
bed's near-nadir-to-off-nadir ratio is **~3.5 dB too flat**. That is exactly
the angular-reflectivity residual T5 was built to attack, now measured
independently of the attenuation choice, and it is the reason a T5 split
composed on top of A20 level anchoring is the natural next configuration
(not run).

### Verdict

**A20 level anchoring (att 20 dB/km, K = +7.92 dB) is the best member.** It
is the only configuration that satisfies all three requirements at once: it
matches the measured bed brightness (median residual **+0.07 dB**, mid/high
within 0.5 dB), it ties for the best tail-shape agreement in the family
(mean |slope misfit| **0.59 dB/us**, the same as every other member -- the
obliquity term simply cannot separate 20 from 31 dB/km at 4-28 deg), and it
is the only level-matched member whose implied bed reflectivity stays
comfortably physical (**median -9.3 dB**, 27% of samples above 0 dB versus
46% at A26 and 54% at A31). A26 level is a defensible second -- equally good
on level and shape, but it needs a bed whose median reflectivity is -1.8 dB,
i.e. within 2 dB of a perfect mirror over half the line, which no
independent evidence in this study supports. A31 level is refuted on
physicality and is also the worst on level (median residual -2.50 dB) and
shape (0.62 dB/us). The honest caveats on the winner: the LOW pass still
overshoots by +4.8 dB and its tail stays surface-clutter-limited (guard
-0.4 dB), 27% of the RSSNR-mapped samples are still unphysical at the bright
end, and the window-versus-tail disagreement above means A20 level buys the
bed level at the cost of ~3.5 dB of tail excess relative to A20 median. The
30 km design margin under the winner is **+9.75 dB** (bed peak over
mid-column +4.57), the most optimistic of the family and 17 dB above the
median-anchored A31 verdict that the earlier sweep produced.

Timings: att20_klevel 1961.4 s (32.7 min; low 1040.7 / mid 525.7 / high
138.0 / syn30km 256.9), att26_klevel 1982.1 s (33.0 min; 1053.6 / 532.2 /
137.1 / 259.2) -- att26_klevel was interrupted once and resumed from its
chunk cache. Projection basis: seven identically-shaped 50 km runs at
32.6-33.0 min (+-2%), so no separate pilot was spent. No new plumbing was
needed (`--anchor level --level-deficit-db D` already existed); suite 292
unit green, ruff clean.

## A 500 km orbital pass on the A20 level-anchored model (2026-08-06)

`--add-500km` adds `syn500km`, the syn30km construction re-flown at a
constant 500,000 m ellipsoidal height (same line, same picks, same 2016
system parameters, roll 0). Run on the winning configuration ONLY -- A = 20,
`--anchor level --level-deficit-db 3.56` (the recorded K = +7.92 dB, reused
not re-derived), DEMOGORGN bed, matched processing, unsplit -- with the four
existing passes replaying from cache, so
`hypothesis_tests/att20_klevel/` now carries five simulated panels in
`radargrams.png` / `decomposition.png` / `bed_tail.png` and a
`syn500km_bed_visibility` metric beside `syn30km_bed_visibility`.

### Derived geometry (actuals)

| quantity | syn30km | **syn500km** |
|---|---|---|
| median AGL | 29,858 m | **499,858 m** |
| cross-track reach (surface-bound) | +-11,241 m | **+-45,214 m** (bed reach 21,215 m) |
| facet spacing (beta = 0.5) | 81.9 m | **233.05 m** |
| facets / interface / chunk | 91 k | **158 k** (17 chunks) |
| window origin t0 | 196.65 us | **3332.14 us** |
| n_samples (sim grid) | 4211 | 4216 |
| alias-limited aperture | 1647 m / 112 traces | **26,618 m / 1793 traces** (half-angle 1.522 deg) |
| hann azimuth resolution | 21.4 m | 21.4 m |
| DEM window | -- | 4323 x 3814 (66 MB/interface), pixels ~37 x ~21 m |

### Pilot (2 traces) -- the checks that mattered

* **Fresnel/LPA validity FAILED first time and was fixed.** The beta = 0.5
  spacing (332.9 m) built **450 m** facets because `build_facets` strides
  the DEM by ONE integer for both axes and this wide-reach window is
  anisotropic (~37 x ~21 m pixels) -- ratio 1.35 over the in-ice limit
  (332.8 m), with the warning fired. The pass now carries
  `facet_spacing_scale = 0.7` (request 233.05 m), which snaps the stride
  down one notch and clears the check: **second pilot emitted no warnings
  at all**. The scale is attached to this pass only, so no existing cache
  moved.
* **Phase precision at 3.3 ms is a non-issue.** The phase argument is
  2*k0*opl ~ **3.98e6 rad**, where the f64 ulp is **4.7e-10 rad** -- nine
  orders of margin. Empirically the coherent sum is intact: peak-to-median
  power in the layer response is **38.7 dB (surface)** and **58.6 dB (bed)**;
  a decohering phase path would flatten both. Fields finite, dropped
  fraction 1.0e-3, 2-trace simulate 1.1 s.
* **Focuser memory is fine at a 1793-trace aperture**: `focused_sar` loops
  per output trace and its transients are (n_ap x n_bins) ~ 24 MB each, a
  few hundred MB peak, freed each iteration -- no chunking needed.
* **One honest processing caveat**: the alias guard fires by **0.3%** --
  the aperture is sized at the median BED range but the guard checks the
  minimum (surface) range, where lambda/(4 sin theta) = 14.809 m against
  the 14.858 m posting. Recorded in the config
  (`surface_alias_ratio` 1.0, focuser warning kept).

### The 500 km verdict: the margin does NOT survive to orbit

| | syn30km | **syn500km** |
|---|---|---|
| bed returns - surface returns, bed window | **+9.75 dB** | **-26.37 dB** |
| bed-window surface returns | -55.43 | **-33.24** (+22.2) |
| bed-window bed returns | -45.68 | **-59.61** (-13.9) |
| mid-column (surface-borne) | -33.97 | -27.22 |
| bed peak over mid-column | +4.57 | +2.75 (see caveat) |

**At 500 km the bed sits 26 dB BENEATH the surface clutter arriving at the
same delay** -- a 36 dB collapse from the 30 km case, and it comes from both
ends: the co-arriving surface clutter rises **+22 dB** while the bed return
falls **-14 dB**. The mechanism is geometric and unavoidable: the delay-to-
cross-track mapping flattens with altitude, so the annulus of surface that
lands in the bed window grows enormously (the surface reach needed to cover
the bed delay goes 11 -> 45 km), while the bed's return is quasi-specular
and only pays the extra spreading. The `bed peak over mid-column` row must
NOT be read as bed visibility here: the decomposition shows the peak inside
the bed window is surface clutter (-33.2 dB) sitting 26 dB above the actual
bed return (-59.6 dB), so that metric is measuring clutter at this altitude.

**Verdict: on this line, with this system, a 500 km orbital sounder is
clutter-limited to the point of blindness at the bed** -- unlike 30 km,
which retained a ~10 dB margin. Closing 26 dB would need real cross-track
discrimination (a much larger antenna aperture or multi-phase-centre clutter
suppression), not more transmit power; and this is still a purely
clutter-limited analysis, so the thermal link budget at 500 km (another
~+35 dB of two-way spreading loss vs 30 km) is an additional, separate
constraint.

Structural caveat, not new: the BED layer's dropped-power fraction is
0.245 at 500 km -- bed facets are simulated over the SURFACE-driven reach
(45 km) while bed arrivals beyond the bed reach (21 km) fall past the
window end. The same effect is larger at 30 km (0.309) and present at the
high pass (0.024), so it is a property of the reach construction, not of
the orbital geometry, and it does not touch the bed WINDOW itself.

Timings: syn500km simulation **396.0 s** (6.6 min, 17 chunks, 158 k
facets/interface) against a ~5 min projection; DEM/BedMachine/DEMOGORGN
fetches at the +-45 km reach took **194 s** on first call (one-time, cached
thereafter; 66 MB per interface grid, 44.2 MB of chunk cache for the pass);
the four existing passes replayed from cache and the run re-did their
processing. Tests: 2 added (`tests/test_basal_hypotheses.py`: the pass entry
follows the syn30km pattern and carries the cache-safe spacing scale that no
other pass has; the derived reach/spacing/aperture scaling at 500 km).
Suite 294 unit green; ruff clean.

## HIGH-pass bed-source comparison: picked bed vs DEMOGORGN under the winning A20 level-anchored mapping (2026-08-07)

The bed-source question re-asked on the ONE configuration that survived the
level-anchoring campaign, and on the high pass only (20161031_07, 10684 m
AGL). The DEMOGORGN side already existed (`hypothesis_tests/att20_klevel`);
the missing side was run fresh:

    uv run python tools/run_basal_clutter.py --segment full --picked-bed \
      --gamma-from-rssnr --processing standard --no-companion \
      --anchor level --level-deficit-db 3.56 --att 20 --passes high \
      --out <hyp> --out-name high_pbed_klevel

`--passes high` is the tool's own pass restriction, so nothing else was
simulated. The level anchor was **reused verbatim, not re-derived**: both
runs' `run_config.rssnr_gamma.k_db` = **+7.92 dB** (K_median 4.36 + D 3.56),
same A = 20 dB/km, same 49.524 m facets, same ±6.9 km reach, same 4226-sample
window, same matched chain (627 m aperture / 43 traces / 3 looks). The RSSNR
gamma field is anchor-line-derived and bed-DEM-independent, so the only
difference between the two simulations is **bed topography**.

Composed to `outputs/basal_clutter/hypothesis_tests/high_bed_comparison/`
(`radargrams.png` = measured / picked bed / DEMOGORGN triptych on one
surface-referenced twtt axis and one [-90, +5] dB scale; `bed_tail.png` =
(a) tail overlay, (b) per-interface guard context; `metrics.json`), mirrored
to `outputs/verification/basal_clutter_high_bed_comparison/`. Composition
script: `claude_notes/high_bed_comparison.py` (pure cache replay).

### The table (measured high pass = the reference column)

| quantity | measured | picked bed | DEMOGORGN seed 0 |
|---|---|---|---|
| bed-return window level (dB rel own surface-return peak) | **-46.11** | -45.35 (**+0.76**) | -46.04 (**+0.07**) |
| mid-column clutter | -34.76 | -43.23 | -43.60 |
| bed-window decomposition: bed / surface returns | -- | -45.46 / -72.25 | -46.12 / -71.95 |
| nadir bed offset vs this pass's own picks | -- | +0.157 us (+13.2 m) | +0.140 us (+11.8 m) |
| **tail slope (dB/us, Theil-Sen, bed+0.5..+3.5)** | **-3.17** | -3.62 (misfit **0.45**) | -2.72 (misfit **0.45**) |
| tail slope (dB/deg, refracted bed incidence) | -1.65 | -1.98 | -1.51 |
| bed-returns-only slope (dB/us) | -- | -3.62 | -2.72 |
| **excess at bed+1 / +2 / +3 us (dB)** | -- | **+12.3 / +13.6 / +9.2** | **+7.5 / +7.6 / +18.3** |
| guard (min sim bed - surface returns in window) | -- | +25.8 ok | +23.5 ok |
| record coverage / measured floor margin | 1.00 / +25.3 dB (not floor-limited) | 1.00 | 1.00 |
| **bed-brightness r vs measured (1 km smoothed)** | -- | **+0.764** | **+0.783** |
| same, bed-borne layer only | -- | +0.755 | +0.783 |
| post-bed sample contrast p90-p50 / p99-p50 (dB) | 11.07 / 18.79 | 21.45 / 37.88 | 19.43 / 36.12 |
| top-5%-of-traces share of the ensemble mean at +1/+2/+3 us | 0.40 / 0.52 / 0.39 | 0.83 / 0.90 / 0.84 | 0.89 / 0.79 / **0.98** |

### Reading

* **Tail metrics: DEMOGORGN wins on level, and the two TIE on shape.** The
  |slope misfit| is **0.45 dB/us for both**, with opposite signs (picked bed
  decays too fast, DEMOGORGN too slowly), so the decay SHAPE does not
  discriminate the bed sources at this altitude any more than 20-vs-31 dB/km
  did. What does separate them is the tail LEVEL: at bed+1 and +2 us the
  picked bed sits **+12.3 / +13.6 dB** above measured against DEMOGORGN's
  **+7.5 / +7.6 dB** -- a consistent ~6 dB of extra off-nadir bed return,
  which is exactly the signature expected of the picked bed's cross-track
  INERT residual (every along-track bed feature becomes a full cross-track
  ridge). **The cross-track-ridge tail artifact survives the new mapping.**
* **The +3 us column must not be read as a tail level.** New diagnostic
  (`tail_concentration`): 84 % (picked) and **98 %** (DEMOGORGN) of the
  ensemble mean at bed+3 us comes from the brightest 5 % of traces --
  DEMOGORGN's apparent +18.3 dB excess is essentially ONE bright arc at
  s = 57.75 km, not a raised tail. Measured is 0.39-0.52 at every delay,
  i.e. a genuinely broad tail. The robust slope and the +1/+2 us excesses
  are the fair comparison points; this also explains the earlier campaign's
  erratic +3 us numbers (+21.7 dB for DEMOGORGN at high).
* **Brightness correlation: DEMOGORGN wins, +0.783 vs +0.764** -- the
  historical ordering (0.78 vs 0.76 in the different-mapping era) reproduces
  almost to the third digit. The story holds. Both are far above the
  BedMachine value (+0.66) recorded in the three-way ablation; the shared
  RSSNR field, not the topography, still carries most of the along-track
  pattern.
* **The bed-WINDOW level is not a fair discriminator here and is reported as
  such.** D = 3.56 dB was solved on the DEMOGORGN median-anchored run and the
  high pass is the one that set that median, so DEMOGORGN's +0.07 dB residual
  is largely by construction. The informative half is the picked bed's
  **+0.76 dB with the identical reflectivity field**: the ridged bed puts
  ~0.7 dB more power into the bed window and ~6 dB more into the off-apex
  tail, i.e. its excess is overwhelmingly off-apex, as the 2026-08-03
  analysis argued.
* **Visual arc texture: picked bed is denser, DEMOGORGN is cleaner, neither
  is measured-like.** The triptych at s = 30-40 km shows the picked bed
  filling the column with many overlapping, beaded hyperbolic arcs over a
  brighter background; DEMOGORGN gives fewer, sharper, better-separated arcs
  with darker gaps; the measured panel is neither -- a fine-grained jagged
  "mountain range" of short features hugging the bed band with no long clean
  arcs. The numbers agree: the measured post-bed field is by far the least
  peaked (p99-p50 = 18.8 dB vs 37.9 / 36.1 dB), and the picked bed is the
  most peaked AND the brightest at the top end (p99 -21.7 vs -24.8 dB).
  Part of the sim-vs-measured peakedness gap is the known looks mismatch
  (3 sim looks vs the product's 6-11) and is not a bed-source statement; the
  picked-bed-vs-DEMOGORGN difference (2.0 dB in p90-p50) is, since both sims
  share the processing.
* Mid-column is bed-source-insensitive to **0.37 dB** (-43.23 vs -43.60) and
  surface-borne in both, re-confirming the study's central decomposition
  result on the current configuration.
* Honest note on the nadir bed: the picked bed sits **+13.2 m deep** at the
  high pass even though it is built from radar picks -- those are the LOW
  pass's picks, and the scout's 6 % cross-pass pick-thickness spread is what
  remains. DEMOGORGN's +11.8 m is its own thickness-convention scatter. The
  two are within 1.4 m of each other, so neither comparison above is driven
  by a bed-registration difference.

### Verdict

**DEMOGORGN remains the better bed source at the high pass under the current
mapping**: it wins the brightness correlation (+0.783 vs +0.764), it is 6 dB
closer on the tail level at +1/+2 us, and it ties on tail shape. The picked
bed's only advantage is its denser arc field, which is closer to the measured
arc DENSITY but comes with the cross-track-ridge level artifact and the
most-peaked post-bed statistics of the three datasets. Nothing here changes
the `att20_klevel` configuration; the earlier ablation's story is intact.

Timings: picked-bed high-pass simulation **137.7 s** (17 chunks, ~8 s each;
43 MB of chunk cache), full tool invocation ~5.2 min including frame/DEM
prep, matched processing, analysis and figures; composition ~50 s per
invocation (both sides pure cache replay, 0 re-simulations). Tests: suite
**294 unit green**, ruff clean on the changed file
(`claude_notes/high_bed_comparison.py`; no tool/source changes were needed --
every knob this used already existed).

## EXTENDED segment: s = 0 -> 69.7 km, full five-pass set at the winning A20 level-anchored settings (2026-08-07)

The study window grown in both directions -- up-track to the anchor start
(s = 0) and down-track to the **grounding line** (s = 69.7 km; the scout's
quirk 1: beyond it BedMachine's "bed" is the seafloor under a cavity, so the
segment stops there) -- and the whole current-best configuration re-run on
it, with every chunk cached under a distinct segment tag so the plotting can
be iterated without re-simulating.

    uv run python tools/run_basal_clutter.py --segment extended \
      --demogorgn-bed --gamma-from-rssnr --processing standard \
      --add-30km --add-500km --no-companion --anchor level \
      --level-deficit-db 3.56 --att 20 --out outputs/basal_clutter \
      --out-name extended

Deliverables in `outputs/basal_clutter/extended/`: `radargrams.png` (2 x 5,
measured over sim, 0-69.7 km), `decomposition.png` (ensemble),
**`decomposition_trace.png` (NEW: single-trace variant)**, `bed_tail.png`,
`metrics.json`, `run_config.json`, `report.html`; metrics + all four figures
mirrored to `outputs/verification/basal_clutter_extended/`. 305 MB of chunk
cache in `extended/runs/` (115 chunks = 5 passes x 23).

### Coverage / slice verification (derived from nav, not assumed)

`claude_notes/extended_segment_slices.py` projects every candidate frame of
each flight onto the anchor polyline (the same `project_to_track` machinery
`prep_pass` uses) and cuts the s in [0, 69.7] km window:

| pass | parts (increasing s after reversal) | traces | coverage | offset med/max | joins | picks |
|---|---|---|---|---|---|---|
| low | `_005 (0, 3333)`, `_006 (0, 1359)` | 4692 | 0.00 -> 69.70 km | 0 m (it IS the anchor) | +30.7 m | 100 % |
| mid (rev) | **`_007 (0, 216)`**, `_006 (0, 3337)`, `_005 (2194, 3337)` | 4696 | 0.00 -> 69.69 km | 10-23 / 30 m | +32.7, +32.4 m | 100 % |
| high (rev) | `_005 (0, 3033)`, `_004 (1671, 3336)` | 4698 | 0.01 -> 69.70 km | 9-13 / 23 m | +34.1 m | 100 % |

* **Only ONE new frame is needed** (`20161028_05_007`, 216 traces): the high
  pass's extension lives entirely inside the already-used `_005`/`_004`
  (`_005` runs s 45 -> 0 over its 3336 traces), and the low pass's inside
  `_005`/`_006`. `20161031_07_006` covers s -54 .. -4.5 km (off-window),
  `20161028_05_004` s 102-152 km, `20161031_07_003/_002` s 94-194 km -- all
  correctly excluded.
* Every frame's twtt grid matches its pass's, bottom picks are 100 %
  populated, part joins are one-trace gaps (+31..34 m, the same as the 50 km
  segment's), trace counts agree to **0.13 %** across the triplet, and the
  extended parts **contain** the 50 km parts frame-for-frame (unit-tested:
  the window only grows).
* Independent check that the slicing is right: the MEASURED bed-return tail
  slopes recomputed over the s 18-68 km sub-range of the extended run
  reproduce the recorded 50 km values **exactly** (-8.22 / -5.15 / -3.17
  dB/us, low/mid/high).

### The mapping was reused, not re-derived

`--segment extended` pins the RSSNR K anchoring to the **`full` segment**
(new `K_ANCHOR_SEGMENT`, printed at run start and recorded in
`run_config.k_anchor_segment` / `rssnr_gamma.k_anchor_segment`): K_median
+4.36 + D 3.56 = **K = +7.92 dB**, bit-identical to `att20_klevel`. The
implied reflectivity over the NEW extent is recorded separately
(`g2_run_seg_db`): median **-8.6 dB**, 21.2 % above 0 dB over s 0-69.7 km,
against -9.3 dB / 27.0 % over s 18-68 km -- the longer line is slightly MORE
physical, not less.

### Level residuals on the extended segment (honest, not re-anchored)

| pass | measured bed window | sim | residual (extended) | residual (50 km) |
|---|---|---|---|---|
| low | -57.93 | -53.81 | **+4.12** | +4.83 |
| mid | -49.19 | -50.13 | **-0.94** | -0.48 |
| high | -48.84 | -50.65 | **-1.81** | +0.07 |
| **median** | | | **-0.94 (gate <= 2 dB: PASS)** | +0.07 |

The extension adds a **thick-ice, dim-bed, low-relief** zone (the scout's
s 5-15 km "featureless uniform haze"): the MEASURED bed window drops 3.4-3.7
dB and the simulated one drops 4.4-4.6 dB, so the residuals move by -1 to
-1.9 dB and the median stays inside the 2 dB gate. **The 50 km-calibrated
mapping transfers to the longer line without adjustment**, which is the real
test of the level anchor. Mid-column clutter and the altitude trend move the
same modest amount: measured high-low **+17.4 dB** (was +19.8 on the 50 km
window), simulated +25.8 (was +27.7), error **+8.4 dB** (was +7.9) -- the
sim's known over-prediction of the altitude step is unchanged in character.

### Standard metric table (extended, DEMOGORGN bed, A20, K +7.92)

| pass | AGL | midcol meas/sim | bed window meas/sim | tail slope meas/sim | excess +1/+2/+3 us | guard | cov |
|---|---|---|---|---|---|---|---|
| low | 447 m | -53.2 / -71.5 | -57.9 / -53.8 | -8.25 / -6.18 (bed-only -7.32) | -17.8 / -17.9 / -14.2 | **FAIL -1.0** | 1.00 |
| mid | 9080 m | -36.2 / -46.3 | -49.2 / -50.1 | -4.76 / **-4.73** | +10.1 / +2.2 / +5.5 | ok +20.2 | 0.997 |
| high | 10610 m | -35.8 / -45.8 | -48.8 / -50.7 | -3.66 / **-1.23** | +11.0 / +3.5 / +18.5 | ok +23.2 | 1.00 |
| syn30km | 29786 m | -- / -35.8 | -- / -51.3 | -- / -0.70 | -- | ok +10.2 | 1.00 |
| syn500km | 499786 m | -- / -28.5 | -- / -59.1 | -- / -0.79 | -- | FAIL -20.8 | 1.00 |

Prediction panels: **syn30km bed - surface returns in the bed window
+7.12 dB** (was +9.75 on the 50 km segment; bed peak over mid-column +5.75)
and **syn500km -24.74 dB** (was -26.37). Both verdicts survive the longer
line -- 30 km retains a useful margin, 500 km is clutter-blind at the bed --
but the 30 km margin is **2.6 dB smaller** once the dim-bed thick-ice zone
is included, i.e. the design margin is segment-dependent and the 50 km
number was the optimistic end.

### Where the extension changed the tail (`claude_notes/extended_tail_split.py`)

Bed-referenced statistics split into the legacy 50 km sub-range and the two
new pieces (sim / measured level at bed+2 us, dB rel own surface peak):

| pass | zone | sim slope | sim +2 us | meas slope | meas +2 us | excess |
|---|---|---|---|---|---|---|
| low | s < 18 km (NEW) | -5.96 | -91.9 | -8.71 | -90.3 | **-1.6** |
| low | s 18-68 (legacy) | -6.21 | -82.9 | -8.22 | -65.2 | -17.8 |
| mid | s < 18 km (NEW) | -3.09 | -64.0 | -4.27 | -59.1 | **-4.9** |
| mid | s 18-68 (legacy) | -4.77 | -43.8 | -5.15 | -46.0 | +2.2 |
| high | s < 18 km (NEW) | -3.01 | -61.5 | -3.47 | -58.7 | **-2.8** |
| high | s 18-68 (legacy) | -1.55 | -43.3 | -3.17 | -47.0 | +3.8 |

* **The sim's tail excess flips sign in the new zone.** Over the deep-ice,
  dim-bed up-track section the simulation runs 1.6-4.9 dB COLD at bed+2 us
  on all three passes, against +2..+4 dB HOT over the legacy window -- and
  the low pass, whose -18 dB tail deficit has been the study's standing
  anomaly, is within **1.6 dB** of measured there. The deficit is therefore
  not a global property of the surface-clutter model: it is specific to the
  thin-ice, high-relief part of the line.
* **The high pass's whole-segment slope (-1.23) is not a stable statistic.**
  Over the identical legacy sub-range it reads -1.55 here against -2.72 in
  the 50 km run, because (a) the derived cross-track reach is a
  SEGMENT-level quantity -- the extension's thicker ice widens it for every
  trace (high 6943 -> 7415 m, low 2529 -> 2904 m, syn500km 45.2 -> 49.1 km),
  admitting more late off-nadir bed returns, and (b) the ensemble mean at
  bed+2 us is carried by the brightest 5 % of traces (0.66-0.85 in sim,
  0.53-0.77 measured). Cross-segment tail-slope comparisons at high altitude
  are therefore NOT like-for-like; the measured side, which has no reach
  parameter, does reproduce exactly.

### The single-trace decomposition (NEW figure, parameterised location)

`--trace-decomp-s S_KM` (default `DECOMP_S_KM` = 31.0 km on full/extended)
adds `decomposition_trace.png`: the same measured / sim-total / sim-surface-
returns / sim-bed-returns curves at ONE slow-time location instead of the
trace ensemble mean. The chosen s, the per-pass sim AND measured trace
indices, the trace's AGL and bed delay, the per-trace guard and the measured
mid-column percentile are recorded in
`run_config.passes.<key>.trace_decomposition`.

| pass | sim trace | measured trace | s (km) | AGL | bed below surface | bed-window bed - surface returns | measured midcol percentile |
|---|---|---|---|---|---|---|---|
| low | 2087 | 2087 | 31.002 | 491 m | 8.86 us | **+49.5 dB** | 0.05 |
| mid | 2088 | 2088 | 30.996 | 9039 m | 8.88 us | **+21.3 dB** | 0.18 |
| high | 2090 | 2090 | 31.000 | 10565 m | 8.92 us | **+27.3 dB** | 0.32 |
| syn30km | 2087 | -- | 31.002 | 29739 m | 9.04 us | **+14.3 dB** | -- |
| syn500km | 2087 | -- | 31.002 | 499739 m | 9.24 us | **+1.1 dB** | -- |

Why s = 31.0 km is defensible: it is the scout's documented **deep trough**
("one wide bright hyperbola from the deep trough at s ~ 31 km") inside the
30-40 km window with the highest per-km bed relief on the grounded line
(mean 103 m/km), the bed there sits 8.86-9.24 us below the surface against a
segment median of ~8.1 us (i.e. genuinely a trough), and **every real pass
satisfies the bed-window guard with room to spare** (+21 to +49 dB). Its
measured mid-column is NOT the segment's brightest (5th/18th/32nd percentile
low/mid/high) -- recorded rather than hidden; a brightest-clutter location
is one `--trace-decomp-s` away and costs no simulation.

What the single trace shows that the ensemble does not:

* At **low** the sim total is the sim surface returns everywhere except a
  single 40 dB-tall specular spike at the bed -- one sounding is a spike on
  a background, not the smooth decaying tail the ensemble mean draws. The
  measured trace has a second bright return ~1.4 us past its bed (an
  off-nadir bed feature the sim does not reproduce), which is what the
  ensemble buries.
* At **mid/high** the sim bed returns rise out of the surface returns ~2 us
  BEFORE the bed and stay 20-27 dB above them through the bed window, with
  three or four resolved peaks between bed and bed+3 us: at one trace the
  "tail" is visibly a handful of discrete off-nadir arcs, which is exactly
  the concentration diagnostic above, seen directly.
* At **syn30km** the bed returns only overtake the surface returns ~1.5 us
  before the bed (+14.3 dB in the window), and at **syn500km** they never
  really do (+1.1 dB): the orbital panel shows a bed buried in surface
  clutter at a single sounding, which is the 500 km verdict without any
  ensemble averaging.

### Mechanics, timings, tests

* New: `SEGMENTS`/`PASSES[*]["extended"]`, `S0_KM`/`DECOMP_S_KM`/
  `N_TRACES_EXT` entries, `K_ANCHOR_SEGMENT`, `build_rssnr_gamma(
  k_anchor_segment=...)` + `g2_run_seg_db`, `analyze_pass(trace_s_km=...)`
  -> `trace_profs`/`trace_info`, `fig_decomposition_trace`,
  `--segment extended`, `--trace-decomp-s`. The segment is part of the chunk
  cache NAME and KEY, so no 50 km cache could be reused or clobbered
  (unit-tested); all pre-existing caches stayed valid.
* Simulation wall **3830.7 s (63.8 min)**: low 1654.7 (23 x 71.5 s), mid
  740.9, high 357.5, syn30km 358.6, syn500km 719.1. Full first invocation
  ~80 min including one-time DEM fetches and processing; against the ~54 min
  linear projection, the overshoot is the wider derived reach (+7-15 % per
  pass) and the deeper fast-time windows, both consequences of the thicker
  ice the extension adds.
* One-time fetches for the new bounds: **162 MB** (REMA 32 m dominates:
  13.4 + 17.3 + 17.8 + 23.1 + **87.2** MB, the last for syn500km's +-49 km
  reach; BedMachine and DEMOGORGN tiles are 0.1-0.7 MB each at 500 m
  posting).
* **Cache-only replay verified**: re-running the identical command replayed
  all 115 chunks (`skip-exists`) and reproduced every metric bit-identically
  in **19 min**, all of it processing -- so plotting iterations need no
  simulation. Most of that is syn500km's 1793-trace-aperture focuser; adding
  `--passes low mid high` cuts a plotting iteration to ~6 min.
* Tests: **299 unit green** (5 added in `tests/test_basal_hypotheses.py`:
  the extended table is a superset of the full segment with matching trace
  counts and synthetic-pass inheritance; extended cache names/keys are
  distinct; the K anchor reuses the full-segment mapping bit-identically and
  re-deriving it WOULD move K; the single-trace decomposition records a
  parameterised location, guard and profiles and costs nothing when unused;
  the figure renders and skips passes without it). Ruff clean.

## FULL LINE: s = 0 -> 148.45 km across the grounding line, HIGH pass only (2026-08-10)

The study window grown across the GL to the whole overlapping line, with a
HYBRID bed (grounded DEMOGORGN + floating radar-picked shelf base) -- the
first simulation of the FLOATING part, where BedMachine/DEMOGORGN report
the SEAFLOOR under the cavity, not the reflector the radar sees (scout
quirk 1). STAGED by design: the HIGH pass (20161031_07, 10,763 m AGL over
this window) only; low/mid/synthetic passes deliberately not simulated.

    uv run python tools/run_basal_clutter.py --segment full_line \
      --demogorgn-bed --gamma-from-rssnr --processing standard \
      --no-companion --anchor level --level-deficit-db 3.56 --att 20 \
      --passes high --out outputs/basal_clutter --out-name full_line

Deliverables in `outputs/basal_clutter/full_line/` (radargrams with the GL
marked / decomposition / **decomposition_zones (NEW: grounded-vs-floating
ensemble split)** / decomposition_trace (NEW: two panels, grounded s=31 +
floating s=120) / bed_tail / metrics / run_config / report.html), figures +
metrics mirrored to `outputs/verification/basal_clutter_full_line/`. 128 MB
of chunk cache (49 chunks) in `full_line/runs/`.

### Slice verification (derived from nav, not assumed)

`claude_notes/full_line_slices.py` (same projection machinery as the
extended work) over every candidate frame; window s in [0, 148.45] km:

| pass | parts (increasing s after reversal) | traces | coverage | offset med/max | joins | picks (floating) |
|---|---|---|---|---|---|---|
| low | `_005 (0,3333)`, `_006 (0,3333)`, `_007 (0,3327)` | 9993 | 0.00 -> 148.44 km | 0 (IS the anchor) | +30.7, +26.4 m | 100 % (100 %) |
| mid (rev) | `_007 (0,216)`, `_006 (0,3337)`, `_005 (0,3337)`, `_004 (223,3337)` | 10004 | 0.00 -> 148.44 | 12-23 / 30 m | +32.7, +32.4, +32.7 m | 100 % (100 %) |
| high (rev) | `_005 (0,3033)`, `_004 (0,3336)`, `_003 (0,3340)`, `_002 (3044,3341)` | 10006 | 0.01 -> 148.44 | 5-13 / 23 m | +34.1, +28.8, +29.1 m | 100 % (100 %) |

Trace counts agree to 0.13 %; every twtt grid matches its pass's; every
full_line part CONTAINS its extended part (window only grows; unit-tested);
`20161031_07_006` (s -54..-4.5 km) correctly excluded. All frames were
already in `outputs/cache/` from the 07-31 scout -- no new frame downloads.

### The HYBRID bed and its guards

`s < 69.7 km` = DEMOGORGN seed 0, bit-identical source/snapshot to the
extended run; `s > 73.7 km` = the LOW pass's radar basal picks (the
established pick reference), NEAREST-NEIGHBour in anchor s and **constant
cross-track -- the accepted flat-ish shelf-base approximation, stated
plainly: the 1-D picks supply no cross-track relief, and unlike the
grounded picked-bed residual there is no valid 2-D DEM to preserve under
the shelf**. Linear blend over the 4 km ramp GL -> GL+4 (grounded side
stays pure DEMOGORGN). Chunks spanning the GL crop the scene-level hybrid,
so their facets are built from the blended grid.

* **Blend step at the GL**: nadir (on-track) DEMOGORGN - picks = med
  **-17.4 m**, rms 28.3, |max| 67.9 m over 269 track samples -- the
  documented ~10-20 m offset class the ramp absorbs. (The full cross-track
  blend-zone stat reads -162 med / 308 rms m, but that conflates
  DEMOGORGN's genuine 2-D relief with the cross-track-constant picks;
  both are recorded.)
* **Min-clearance guard PASSES with no clamping**: min (REMA surface -
  hybrid bed) = **+239.2 m grounded / +453.2 m floating** (floating median
  720 m); clamp fraction 0.000000 in both zones.
* DEMOGORGN fetched over the grounded(+ramp+2 km) track only (max s
  75.69 km) -- no seafloor data touched; nodata_fill 0.66 of the SCENE grid
  is the zero-weight floating area filled by nearest-edge (recorded).
* Floating pick coverage: 5306/5306 axis picks finite (gap frac 0.0000).

### QC coverage over the floating stretch

`zone_qc_coverage`: bottom picks 100 % on both zones of the high pass
(4699 grounded / 5307 floating measured traces); RSSNR anchor samples
**52 grounded / 59 floating, qc_pass_frac 1.000 in both** (the pinned
snapshot's cached arrays cover the full line, med spacing ~1.37 km).

### Mapping reused verbatim + ZONE-AWARE physicality

`K_ANCHOR_SEGMENT["full_line"] = "full"`: K_median +4.36 + D 3.56 =
**K = +7.92 dB, bit-identical to att20_klevel** (unit-tested). New
`g2_zones_db` judges the implied |Gamma_bed|^2 against each zone's own
Fresnel ceiling (grounded: rock anchor -12.86 dB, hard bound 0 dB;
floating: ice->seawater **-3.5 dB**, a genuine ceiling for a specular
ice-ocean interface):

| zone | n | med G2 | p5..p95 | frac > 0 dB | frac > -3.5 dB |
|---|---|---|---|---|---|
| grounded | 52 | **-8.6 dB** | -25.9 .. +14.7 | 0.212 | 0.385 |
| floating | 59 | **+12.3 dB** | -3.7 .. +19.7 | **0.881** | **0.932** |

**The floating-side mapping is unphysical as a pure reflectivity** -- 93 %
of the shelf sits above the ice-seawater ceiling and the median implied
reflectivity is +12 dB. Read below for why the received-level test still
passes: on the shelf the RSSNR-mapped gamma must be read as an EFFECTIVE
brightness (it absorbs the specular-vs-diffuse spreading difference the
mapping's diffuse normalization assumes away), not as a Fresnel
coefficient.

### The zone-split table (THE deliverable; dB rel own surface-return peak)

| quantity | grounded (s 0-69.7) | floating (s 69.7-148.4) |
|---|---|---|
| bed-window level: sim / measured | -50.71 / -48.84 | **-26.25 / -28.13** |
| **bed-window residual (sim - meas)** | **-1.88 dB** | **+1.88 dB** |
| mid-column: sim / measured (residual) | -45.6 / -35.8 (-9.8) | -48.5 / -35.4 (-13.0) |
| decomposition: surface / bed returns in bed window | -74.5 / -50.8 | -76.4 / -26.25 |
| tail slope sim / measured (dB/us) | -1.54 / -3.66 | -1.72 / **-6.61** |
| tail excess at +1 / +2 / +3 us | +4.7 / +4.3 / +17.3 | +8.4 / **+11.2** / +18.4 |
| tail guard (min bed - surface returns) | ok +24.4 dB | ok +39.9 dB |
| measured floor | -75.7 | -75.5 |

Whole-line: `rssnr_level_anchor` median residual **+1.98 dB (gate <= 2:
PASS, high pass only in this run)**; surface gate 0.35 bins PASS; measured
tail not floor-limited (+27 dB margin at bed+3 us).

### Reading: the specular-regime test

1. **The fixed K reproduces the floating shelf-base brightness within
   +1.9 dB.** The measured shelf base is **20.7 dB brighter** than the
   grounded bed window (-28.1 vs -48.8) and the sim tracks that step
   (22.6 dB, -26.2 vs -50.7) with the grounded-calibrated K = +7.92 dB
   reused verbatim -- no re-anchoring, no per-zone adjustment. The
   grounded and floating residuals are symmetric (-1.9 / +1.9 dB), i.e.
   the single constant splits the difference between the regimes almost
   perfectly on this line. That is the study's specular-regime answer:
   **the RSSNR + level-anchored mapping transfers across the grounding
   line at the received-power level.**
2. **...but NOT at the reflectivity level.** The implied G2 needed for
   that agreement is +12 dB median on the shelf (93 % above the
   ice-seawater ceiling): the mapping's diffuse-spreading normalization is
   wrong for a specular interface, and the gamma field silently absorbs
   the difference. Both facts are recorded; quoting either alone would
   mislead.
3. **The floating tail is the new open misfit.** Measured decay past the
   shelf base is -6.61 dB/us (much steeper than the grounded -3.66 --
   the specular signature); the sim decays at only -1.72 dB/us and runs
   +11.2 dB hot at bed+2 us, with the guard at +39.9 dB confirming this
   is genuine simulated bed-return energy, not surface clutter. Two
   recorded artifact sources: the NN interpolation makes the shelf base a
   ~15 m along-track staircase, and the cross-track-constant extension
   turns every along-track feature into a full-reach ridge (the picked-bed
   ridge artifact, now on the floating side) -- both inject off-nadir bed
   energy a real quasi-planar shelf base would not return. The grounded
   excesses (+4.3 dB at +2 us) are consistent with the extended run's
   (+3.5 at the same reach caveat class).
4. Mid-column stays **surface-borne** in both zones (decomposition
   separation > 35 dB) and under-predicted by ~10-13 dB at this pass --
   the same character as the extended run's high-pass midcol (-10.0 dB);
   the floating zone is 3 dB worse, consistent with the missing
   volume/crevasse scattering of a real shelf.
5. **Single-trace decompositions** (recorded in run_config): grounded
   s = 31.00 km (trace 2090, bed 8.91 us, guard +23.7 dB, measured midcol
   percentile 0.28) and floating **s = 120.00 km** (trace 8089, bed
   6.23 us, guard **+44.7 dB**, percentile 0.79). s = 120 is defensibly
   floating: past the last BedMachine mask flicker at 110 km, mid-shelf.
   The floating sounding is the specular picture directly: a sharp bright
   base rising ~45 dB above the co-arriving surface returns, against the
   grounded trough's multi-peak arc cluster.

### Timings, fetches, mechanics

* Simulation wall **750.1 s** (12.5 min; 49 chunks x ~15.2 s, ~120.5 k
  facets/interface, 855 sim samples/chunk) -- inside the 6-10 min sim
  estimate's ballpark once the 49-vs-projected-chunk count is counted;
  full tool invocation **16 min 49 s** wall (pilot chunk replayed from
  cache; processing 632 m / 44-trace aperture, 3 looks; peak RSS 5.7 GB).
  One-chunk pilot first (16.2 s incl JAX compile; projection 13.2 min --
  landed at 12.5).
* One-time DEM fetches (pilot prep, 307.8 s incl fetch): **61 MB REMA
  32 m** (scene grid 4782 x 3331, ~64 MB/interface in memory), two
  BedMachine windows and one grounded-only DEMOGORGN window < 1 MB each at
  500 m posting. Frames: zero new downloads (cached since the scout).
* The known LPA facet warnings (`ratio 1.03/1.43`) fire on every chunk --
  **identical strings to the recorded att20_klevel and extended high-pass
  caches** (verified from those runs' diag JSONs): a pre-existing property
  of the high pass's 49.5 m facets on this DEM stack, kept unchanged for
  comparability with the recorded family, not a new regression.
* Cache safety: segment name AND a `_hyb` marker are in the chunk cache
  file names, and a `hybrid_bed` block (GL, ramp) is in the cache KEY --
  no pre-existing cache could be reused or clobbered (unit-tested; the
  baseline `_p()` names reproduce byte-identically).
* Tests: **327 green** (`pytest tests -q`: 327 passed, 24 network-marked
  deselected; the unit suite grows 299 -> 308), 9 added in
  `tests/test_basal_hypotheses.py` (full_line
  table containment + trace-count parity; cache name/key distinctness;
  K pinned to 'full' with zone stats + graceful no-sample zones; picks NN
  gap-skipping; zone_g2_stats ceilings; hybrid blend weights/clearance/
  restricted DEMOGORGN fetch on a synthetic scene; multi-location trace
  decomposition; figure fan-out; full_line CLI rejections). One
  PRE-EXISTING stale assertion fixed in
  `tests/integration/test_basal_clutter.py` (PASSES-keys list had not
  been updated for the 08-06 syn500km addition; verified failing on the
  pre-change tree). Ruff clean.

STOPPED HERE by design: low/mid/syn passes on the full line await user
review of the high-pass result.

## FULL-LINE ALTITUDE CAMPAIGN: staged per-pass delivery, low/mid + syn14km/syn300km (2026-08-10)

User-approved continuation after the high-pass review. Two new plot knobs
(the delivered high-pass figures were replotted first, then every other
pass emitted its own figure set the moment it completed):

* `--plot-s-max S` crops the PLOTTED radargram along-track range (the data,
  caches and every metric keep the full 148.45 km); `--fig-width-scale`
  (added by the user in e684f64) scales the radargram panel width. The
  campaign ran `--plot-s-max 100 --fig-width-scale 2`, which preserves the
  delivered 3x-width figure's px-per-km on the 0-100 km crop.
* `--per-pass-figs` (STAGED DELIVERY): each pass's complete figure set is
  written as SEPARATE suffixed files (radargrams_<pass>.png,
  decomposition_<pass>.png, bed_tail_<pass>.png,
  decomposition_trace_<pass>.png, decomposition_zones_<pass>.png)
  immediately after that pass's sim+processing+analysis, with a
  `FIGSET_READY <pass>` marker line; the unsuffixed combined figures are
  skipped in this mode. Every figure top carries the SOURCE-DATA
  provenance (season + frame span + altitude, e.g. "2016_Antarctica_DC8 -
  measured 20161031_07_002-005 (10.8 km AGL)"; synthetics are labeled
  "SYNTHETIC <alt> km constant-altitude pass on the 20161105_05_005-007
  line (no measured data)"). Helpers `frame_span` / `source_label` /
  `emit_pass_figs`; zone_analysis moved into the pass loop so the zones
  figure can be emitted per pass.
* NEW SYNTHETIC ALTITUDES `syn14km` (14,000 m) and `syn300km` (300,000 m)
  -- the syn30km/syn500km constructions at the campaign's altitudes
  (`--add-14km` / `--add-300km`; syn30km/syn500km untouched, not run here).

Each pass ran as its own invocation of the same command family
(`--segment full_line --demogorgn-bed --gamma-from-rssnr --processing
standard --no-companion --anchor level --level-deficit-db 3.56 --att 20
--passes <key> --per-pass-figs --plot-s-max 100 --fig-width-scale 2
--out-name full_line`), sharing one chunk cache; per-pass metrics/config
snapshots are preserved as `metrics_<key>.json` / `run_config_<key>.json`
(the plain metrics.json is the last invocation's). Everything mirrored to
`outputs/verification/basal_clutter_full_line/`.

### 2-trace pilots for the new altitudes (before their full runs)

| quantity | syn14km | syn300km (unscaled -> scaled) |
|---|---|---|
| AGL med | 13,934 m | 299,934 m |
| reach (surface / bed -> ct) | 8,548 / 3,604 -> +-8,548 m | 38,088 / 16,438 -> +-38,088 m |
| facet spacing | 56.20 m (scale 1.0) | 257.9 -> **180.5 m (scale 0.7)** |
| LPA check | ratio 1.26 (KEPT: milder than the accepted airborne 1.43 class) | **1.36 -> 1.03** (the syn500km failure class; 0.7x snaps the stride down) |
| window t0 / n_samples | 89.90 us / 4441 | 1997.88 us / 4441 |
| alias-limited aperture | 800 m / 55 traces | 15,996 m / 1078 traces (half-angle 1.522 deg) |
| phase argument 2k0*opl vs f64 ulp | 1.2e5 rad, margin 8.3e15x | 2.4e6 rad, margin 5.1e15x |
| layer peak-to-median (coherence) | 69.0 / 71.0 dB | 34.8 / 53.3 dB |
| scene grid | 4852 x 3402 (~66 MB/iface) | 6699 x 5248 (~141 MB/iface) |

Both pilots PILOT_OK (fields finite, no alias warning, dropped power
recorded). syn300km's pilot bed-layer dropped fraction is large (0.89 at
the 2-trace crop; 0.245-class at syn500km full runs) -- the documented
reach-construction property (bed facets simulated over the surface-driven
+-38 km reach while bed arrivals beyond +-16 km fall past the window
end); it does not touch the bed window. An earlier syn14km pilot attempt
died silently when the session scratchpad was wiped mid-flight; it was
re-run synchronously and is the one recorded here.

### The zone-split table across altitude (dB rel own surface-return peak)

Bed-window level (sim / meas, residual), tail guard; grounded s 0-69.7 km
vs floating (shelf base) s 69.7-148.4 km:

| pass | AGL | grounded sim/meas (resid) | floating sim/meas (resid) | grounded guard | floating guard | grounded slope sim/meas | floating slope sim/meas | midcol resid g/f |
|---|---|---|---|---|---|---|---|---|
| low | 449 m | -53.9 / -57.9 (**+4.03**) | -28.1 / -40.8 (**+12.63**) | FAIL -0.9 | ok +30.7 | -6.09 / -8.25 | -8.90 / -6.68 | -18.3 / -16.5 |
| mid | 9,080 m | -50.0 / -49.2 (**-0.85**) | -25.2 / -28.2 (**+3.01**) | ok +20.1 | ok +36.3 | -4.72 / -4.76 | -2.75 / -6.53 | -9.9 / -12.7 |
| high | 10,763 m | -50.7 / -48.8 (**-1.88**) | -26.2 / -28.1 (**+1.88**) | ok +24.4 | ok +39.9 | -1.54 / -3.66 | -1.72 / -6.61 | -9.8 / -13.0 |
| syn14km | 13,934 m | -49.9 / -- | -26.1 / -- | ok +21.3 | ok +38.2 | -0.26 / -- | -3.32 / -- | -- |
| syn300km | 299,934 m | -34.1 / -- | -27.0 / -- | **FAIL -12.6** | FAIL +3.4 | -0.51 / -- | +2.09 / -- | -- |

* **The floating (specular-regime) residual is altitude-dependent and
  collapses toward zero with altitude: +12.6 (low) -> +3.0 (mid) -> +1.9
  dB (high).** The high-pass agreement reported at the review was not a
  fluke of the calibration pass, but it does NOT transfer down: at 449 m
  the fixed K overshoots the measured shelf-base brightness by 12.6 dB.
  The measured floating bed window itself moves -40.8 -> -28.2 -> -28.1
  dB (rel own surface peak) from low to mid/high while the sim sits at
  -25..-28 dB everywhere -- i.e. the MEASUREMENT changes with altitude
  (at 449 m the pulse-limited specular surface peak is relatively much
  stronger, and/or the real shelf base decorrelates the specular gain the
  flat-ish NN bed provides), and the effective-gamma mapping only
  reproduces the altitude regime it was anchored in. This is the
  campaign's headline caveat on the specular-regime story.
* Grounded residuals reproduce the known family behavior (low +4.0, the
  standing overshoot; mid/high within 1.9 dB, gate-passing); the
  whole-line level-anchor medians read +9.79 / +2.50 / +1.98 dB
  (low/mid/high) -- the low pass FAILS the 2 dB gate on the full line
  because the floating overshoot now dominates its median (recorded, not
  re-anchored).
* Floating measured tail slopes are steady at -6.5..-6.7 dB/us across all
  three altitudes (the specular signature), while the sim's floating
  slope goes -8.9 -> -2.7 -> -1.7: the NN-staircase/cross-track-ridge
  artifact hurts most at altitude (excess at bed+2 us +12.3 / +10.0 /
  +11.2 dB).
* **syn300km verdict: at 300 km the grounded bed is clutter-buried (zone
  guard -12.6 dB) while the floating specular shelf base still peeks
  above the co-arriving surface clutter (+3.4 dB, below the 10 dB
  fair-comparison threshold).** Whole-line bed-over-clutter -2.8 dB --
  between syn30km (+7.1 on the extended segment) and syn500km (-24.7):
  the clutter-blindness onset sits between 14 km (syn14km: +39.5 dB, still
  wide open) and 300 km on this line. The syn300km grounded-min clearance
  reads -15.9 m with clamp frac 1e-5: the +-38 km window catches a spot
  where DEMOGORGN pokes above REMA; the existing clamp enforces the
  guard (recorded).

### Timings / mechanics

| pass | sim wall | chunks x s/chunk | aperture (traces) | invocation wall |
|---|---|---|---|---|
| high (replay for the figset) | (cached) | 49 (replay) | 632 m (44) | ~9 min |
| low | 3,460.5 s (57.7 min) | 49 x ~70.7 | 86 m (7) | ~66 min |
| mid | 1,563.4 s (26.1 min) | 49 x ~32.2 | 550 m (38) | ~33 min |
| syn14km | 765.8 s (12.8 min) | 49 x ~16.6 | 800 m (55) | ~19 min |
| syn300km | 1,157.7 s (19.3 min) | 49 x ~23.6 | 15,996 m (1078) | ~35 min |

One-time fetches: syn14km REMA window (prep 260 s incl fetch), syn300km's
+-38 km REMA window is the big one (6699 x 5248 at 32 m, ~141 MB/iface in
memory; prep 766 s incl fetch); low's own narrow window fetched during its
run. Chunk cache now 5 passes x 49 chunks under `full_line/runs/`.
Tests: **331 green** (unit 308 -> 312; 4 added: syn14/syn300 pattern +
pilot-verdict scales, geometry scaling, frame_span/source_label format,
emit_pass_figs file set + crop knob), ruff clean. Figure/marker delivery
order: high (replay) -> low -> mid -> syn14km -> syn300km, each announced
with FIGSET_READY when its set hit disk.

## Line map + mask detail (2026-08-11)

Location map: outputs/basal_clutter/full_line/line_map.png (script
claude_notes/basal_clutter_line_map.py; BedMachine v3 mask cached as
bedmachine_mask_antarctic_*.tif). Amundsen sector, lon -117.6..-122.6.
MODELING CAVEAT surfaced by sampling the mask along the track: the
"floating" zone is not purely floating - BedMachine shows grounded
ice-rise patches at s 70.3-72.6, 75.2-87.7, 88.3, 110.4, 143.9-146.5 km.
Our hybrid bed treats everything past 69.7 as shelf (NN picks), so those
patches' bed returns are modeled as shelf base. The radar picks ARE the
right reflector either way (the picks follow whatever the radar saw),
but the flat-ish cross-track assumption is least valid there. Also the
BedMachine grounded-to-floating transition is at s=70.05 km (our 69.7
label sits ~350 m early; visually coincident at map scale).

## Processed-stack cache + floating-zone single-trace four-panel (2026-08-11)

### `--proc-cache`: seconds-fast plot iteration on the full_line campaign

New tool machinery (`process_standard_cached` / `load_proc_pass` /
`proc_rid` / `chunk_digests`, `--proc-cache` on the CLI): the FOCUSED
per-interface stacks (mocomp + backprojection outputs Fs/Fb, complex64 --
the focuser's native dtype and therefore the BIT-EXACT source of the
3-look P/Ps/Pb powers, which are recomputed identically on load) are
persisted per pass under `outputs/basal_clutter/full_line/proc_cache/`
as npz+json pairs following the run-cache meta conventions, together with
the light per-trace arrays (nadir, picks, s axes, twtt) and scalars a
figure needs. STALENESS: the meta key embeds a sha256 digest of every
source chunk's cache meta_key (the sim is deterministic in its meta);
`load_proc_pass` re-derives the digests from the chunk cache as it exists
NOW and declines on any mismatch or missing chunk -- a stale cache is
rebuilt, never silently served. `load_proc_pass` reconstructs a
figure-ready (p_lite, sim_lite, proc) with NO scene prep and NO chunk
replay (measured Data re-sliced from the locally cached frames), so a
four-panel trace figure now takes ~2 s of loading per synthetic pass and
~1 s per measured pass, against ~4-19 min per pass cold.

* **Size: 669 MB total for all five passes** (138-143 MB each: low 140 /
  mid 138 / high 138 / syn14km 143 / syn300km 143) -- above the 500 MB
  estimate because bit-exactness requires the complex per-layer stacks
  (float32 powers would break the bit-compare and cost MORE: 3 arrays vs
  2). Cold population walls: low 211 s, mid 257 s, syn14km 280 s, high
  268 s, syn300km 1121 s (its 1078-trace aperture).
* **VERIFIED bit-exact** (mid): cached-path P stack and the s = 35 km
  extracted trace curves compared against a full direct recompute
  (prep + chunk replay + uncached process_standard):
  `VERIFY mid: P bit-identical=True, trace curves bit-identical=True
  (PASS)`.
* Tests: 2 added (`proc_rid` naming incl. the `_hyb` marker; chunk-digest
  stability, re-simulated-chunk and missing-chunk staleness, cold-cache
  decline; `_proc_from_stacks` equals the process_standard tail exactly).
  Suite **333 green** (unit 312 -> 314), ruff clean.

### Four-panel single-trace figures (low / mid / syn14km / syn300km)

`claude_notes/trace4_fig.py` (supersedes the one-off trace4_s35 script):
per-interface curves at ONE sounding, actual bed marked (red line at the
sim bed-layer nadir; dotted marker at the measured Bottom pick where it
differs visibly), NO window shading/annotations, provenance sub-titles
wrapped (the s=35 overlap defect fixed and that figure regenerated).

* `decomposition_trace4_s35.png` (grounded trough; regenerated):
  low/mid measured+sim at trace 2356/2358, syn14km 2356, syn300km 2356;
  per-trace bed-window bed - surface returns +31.3 / +25.0 / +22.3 /
  **-31.0 dB** (low/mid/syn14/syn300).
* `decomposition_trace4_s100.png` (FLOATING shelf base, s = 100.0 km):
  traces **low 6731 / mid 6739 / syn14km 6731 / syn300km 6731**; bed at
  8.30-8.36 us below the surface returns; per-trace guard (bed-window bed
  - surface returns) **+72.0 / +54.1 / +43.6 / +9.9 dB**.

The floating single sounding IS the specular altitude story in one image:
at 449 m the sim shelf-base spike towers ~45 dB above the measured
RELATIVE level at the same delay (the +12.6 dB zone residual seen at one
trace: the sim's flat-ish NN base mirrors the transmit power back while
the measured relative return is pulled down by the much stronger
pulse-limited surface peak); at 9.2 km the sim spike sits on the measured
peak; and at 300 km the specular base still rises ~10 dB above the
co-arriving surface clutter at a single sounding -- the same monotone
low -> orbital collapse the zone metrics gave (+72 -> +54 -> +44 -> +10 dB
guard), now visible without any ensemble averaging. Note the s = 35 km
grounded counterpart runs the opposite way at orbit (-31 dB: rough-bed
trough fully clutter-buried), which is the grounded-vs-floating verdict
of the syn300km run at trace scale.
