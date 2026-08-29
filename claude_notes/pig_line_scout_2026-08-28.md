# Pine Island candidate lines for the comparison set (2026-08-28)

Scouting only — no simulations. Goal: find Pine Island lines with **coincident
high-altitude (>8 km AGL) and low-altitude (<2 km AGL) sounding**, a straight
50–100 km segment, bed detectable in ≥50 % of the high-altitude data, interesting
basal relief, a surface-roughness regime the single ATM exponential law can carry,
and (bonus) visible off-nadir returns.

Code + figures + tables: `claude_notes/pig_scout/`. Frames cached under
`outputs/cache/`. Nothing in `src/` or `config/` touched.

## Method

1. STAC query (`xopr.query_frames`) over all 24 Antarctic collections for a
   generous PIG box (lon −108…−92, lat −77.5…−74.0) → **1236 frames / 71 flight
   segments**, 9 collections (2004 P3chile, 2009 TO, 2009/2010/2011/2012/2014/2016/2018 DC-8).
2. Per segment, **OPS layer points** (`xopr.ops_api.get_layer_points`, one HTTP
   call per segment, no echogram download) give lat/lon/elev + surface (lyr 1)
   and bottom (lyr 2) twtt for every trace. 67/71 segments returned data
   (2004_P3chile `20041121_01`, 2009_DC8 `20091102_02`, 2009_TO `20100105_02`
   and `20100108_01` return an empty body — not used).
   AGL = c·twtt_surface/2. → **14 segments above 8 km AGL, 52 below 2 km.**
3. Coincidence: each high track thinned to 50 m, KD-tree distance to every low
   track; runs where the separation stays < 800 m inside a PIG box
   (lon −105…−94, lat −77.2…−74.4) and the run is ≥ 45 km.
4. Inside each run, slide 50/60/…/100 km windows and keep the best-scoring one
   with max chord deviation < 300 m and OPS bottom pick present on > 50 %.
   → 13 windows; 5 selected as spatially distinct candidates (A–E).
5. Per candidate: full pass inventory (any segment covering ≥ 85 % of the window
   within 600 m), frames resolved from STAC frame start times, then
   `CSARP_standard` downloaded (`soundersim.opr.load_frame`, 104 frames) and each
   pass trimmed to its single time-contiguous transit of the line.
6. Diagnostics: bed SNR from the data (peak in ±0.5 µs about the OPS pick vs the
   noise floor 3 µs below it), BedMachine v3 mask/bed, ATM Tier-1 10 km roughness
   grid (`outputs/atm_regional/grid_aa.csv`) and Tier-2 site verdicts
   (`outputs/atm_regional/tier2/site_medians.csv`).

## What the data look like over PIG

**All the usable coincidence comes from one survey pattern.** OIB flew a ladder of
~8 parallel rungs, 4–10 km apart, running NNE–SSW across the upper Pine Island
trunk from the interior catchment down through the grounding zone. Two
high-altitude DC-8 flights re-flew the whole ladder — `20091020_06` (median AGL
**10 273 m**) and `20121023_04` (**9 220 m**) — and three low OIB flights re-flew
it at ~450 m: `20141029_05` (456 m), `20161104_05` (451 m), `20181107_01` (447 m).
Median cross-track separation between passes is **10–24 m**; p95 is under 30 m on
most passes. This is a better-instrumented altitude ladder than the existing Getz
line (which has 3 passes); the good rungs here have **five**.

There is essentially nothing else: high-altitude data over the PIG shelf proper,
or crossing the trunk at a different azimuth, either has no low-altitude twin
(window 7, a 70 km down-flow line at lon −99.5, has one low pass at 254 m median
offset and 78 % coverage) or sits outside the PIG basin (window 5, at lat −74.4 /
lon −95).

## The five candidates

Radar parameters are the same for every pass on every candidate:
**2009/2012 DC-8** = MCoRDS 193.9 MHz / **9.5 MHz** bandwidth, product dt 105 ns;
**2014/2016/2018 DC-8** = MCoRDS3 190 MHz / **50 MHz**, dt 20 ns. So each candidate
is simultaneously an altitude ladder (450 m → 9.2 km → 10.3 km) and a bandwidth
trade (50 → 9.5 MHz), exactly as in `cross_season_line_scout.md`.

| | A | B | C | D | E |
|---|---|---|---|---|---|
| length (km) | 100 | 100 | 100 | 100 | 90 |
| start (lat, lon) | −75.350, −100.219 | −75.204, −100.376 | −75.347, −97.315 | −75.172, −97.107 | −75.311, −95.681 |
| end (lat, lon) | −75.560, −96.710 | −75.417, −96.904 | −75.129, −100.764 | −74.959, −100.518 | −75.140, −98.803 |
| max chord deviation (m) | 22 | 34 | 26 | 15 | 16 |
| high passes | **2** (9.2, 10.3 km) | **2** (9.2, 10.3 km) | 1 (9.2 km) | 1 (9.2 km) | **2** (9.2, 10.3 km) |
| low passes (~450 m) | 3 | 3 | 3 | 3 | 3 |
| median ice thickness (m) | 1727 | 1533 | 1094 | **529** | 1413 |
| bed relief p5–p95 (m) | **1116** | 879 | 894 | 319 | 265 |
| OPS bed pick present | 0.90 | 0.87 | 0.91 | **1.00** | 0.54 |
| bed SNR > 5 dB, 9.2 km pass | 0.90 | 0.86 | **1.00** | **1.00** | 1.00 |
| bed SNR median, 9.2 km pass (dB) | 17.7 | 14.8 | 18.4 | **21.9** | 19.2 |
| bed SNR median, 10.3 km pass (dB) | 9.1 | 14.5 | – | – | 8.5 |
| grounding line at s = (km) | 6.7 | 35.0 | 53.2 | 66.1 | none |
| grounded fraction | 0.93 | 0.65 | 0.53 | 0.66 | 1.00 |
| ATM RMS (1–100 m) p50 / p90, whole line (cm) | 8.3 / 104 | 31 / 96 | 36 / 83 | 4.8 / 107 | 4.8 / 8.6 |
| ATM RMS p50 / p90, **grounded part** (cm) | 8.3 / 63 | 5.1 / 31 | 6.8 / 83 | **4.8 / 6.4** | **4.8 / 8.6** |
| nearest Tier-2 site verdict | not adequate | not adequate | 1/2 adequate | adequate (σ 111 cm, l 65 m) | adequate (σ 7.8 cm, l 19 m) |

Frames (`CSARP_standard`, three per pass, already cached):

```
A  2009_Antarctica_DC8 20091020_06_001..003   (reversed)   2012_Antarctica_DC8 20121023_04_019..021
   2014_Antarctica_DC8 20141029_05_024..026   2016 20161104_05_019..021   2018 20181107_01_022..024
B  2009 20091020_06_037..039   2012 20121023_04_011..013
   2014 20141029_05_016..018   2016 20161104_05_011..013   2018 20181107_01_014..016
C  2012 20121023_04_032..034
   2014 20141029_05_033..035   2016 20161104_05_028..030   2018 20181107_01_031..033
D  2012 20121023_04_008..010
   2014 20141029_05_012..014   2016 20161104_05_008..010   2018 20181107_01_011..013
E  2009 20091020_06_022..024   2012 20121023_04_015..017
   2014 20141029_05_019..021   2016 20161104_05_014..016   2018 20181107_01_017..019
```

## Assessment

**D — recommended.** 100 km from the interior catchment (surface 620 m) down onto
the northern Pine Island Ice Shelf (surface 0 m), grounding line at s = 66 km.
Straightest of the set (15 m deviation over 100 km). Thin ice (median 529 m, bed
5–12 µs below the surface) so the bed is **unambiguous in every pass including the
9.2 km one — 100 % of the line above 10 dB SNR, median 21.9 dB**, the only
candidate that fully clears criterion 3. The bed is a continuous train of 3–8 km
hills and valleys with ±250 m relief — not the biggest relief in the set but by far
the most *structured*, which is what exercises a facet/clutter model. Over the
grounded 66 km the surface is the smoothest and most uniform of the set
(ATM RMS 4.8 cm p50 / 6.4 cm p90, and the nearest Tier-2 site is exponential-adequate),
so one exponential law is defensible there; the floating last 34 km is crevassed
(p90 107 cm) and would need to be excluded or given its own law. Off-nadir energy
is obvious: the 451 m pass shows diffraction wings on every bed hill at 60–70 km
and dense crevasse "curtains" past the grounding line, and at 9.2 km the surface
clutter arrives right on top of the bed (only 5 µs separation) — the exact regime
the HAPS study cares about. Weaknesses: only one high pass — the 2009 flight's rung does not reach this
far west. On the plus side its `20121023_04` frames are free of the data-quality
artifact blocks that affect A and B.

**A — recommended as the "hard" case.** The only candidate with a genuine
altitude *ladder* of three levels (450 m / 9.2 km / 10.3 km) plus the biggest bed
relief in the set: the bed runs at 5–13 µs for 35 km and then steps down ~700 m
into the deep trough, sitting at 20–23 µs for the remaining 65 km. That step is a
superb test of the range-window/clutter budget. Bed pick present on 90 %, and the
9.2 km pass keeps 90 % above 5 dB — but honestly, the deep half is marginal at
altitude (the 10.3 km 2009 pass has median SNR 9.1 dB and only 45 % above 10 dB),
so it is really "bed detectable over the shallow 35 km and weakly beyond". The
low-altitude passes show clear double-branch bed returns at 15–35 km — unambiguous
out-of-plane reflections, ideal for criterion 6. Caveats: a rough surface patch
(ATM p90 63 cm even on the grounded part) and three stretches of `20121023_04`
data-quality artifacts (s ≈ 19–26, 58–67, 93–96 km) that would have to be masked.

**C — good middle ground.** 100 km running the other way (interior → shelf, GL at
53 km), ice 1094 m, bed relief 894 m rising from −1300 m to −400 m. Bed SNR at
9.2 km is the second best of the set (median 18.4 dB, 100 % above 10 dB where the
pick exists) and the 2012 pass looks clean — no artifact blocks. The 45–100 km
half has a big diffuse clutter wedge below the bed, good for criterion 6. Loses on
criterion 5: half the line is floating/shear-zone ice with ATM p90 83 cm, and it
has only one high pass.

**B — usable but compromised.** Same 5-pass structure as A, 879 m relief, GL at
35 km, and the grounded part is reasonably smooth (5.1 / 31 cm). But the 2012
high pass carries large artifact blocks across s ≈ 45–58 km and 82–90 km, which is
a third of the line, and the deep half of the bed is faint at both altitudes.

**E — worst on the criteria that matter, best on roughness.** Entirely grounded
interior ice, and the only candidate where a single exponential law is safe over
the *whole* line (ATM RMS 4.8 cm p50, 8.6 cm p90; the nearest Tier-2 site is
adequate with σ = 7.8 cm, l = 19 m). It fails criterion 3 as posed: the OPS bottom
pick exists on only **54 %** of the line and stops entirely at s = 48 km, and the
10.3 km pass is at 8.5 dB median. Basal relief is 265 m of gentle undulation. Take
it only if the roughness regime is the binding constraint.

### Trade to be aware of

Over PIG the criteria fight each other: the interesting bed (trough walls, the
grounding zone, the trunk) is exactly where the surface is crevassed and the
single-exponential law breaks, and the smooth interior where the law holds has a
deep, dull, hard-to-detect bed. **D is the one line where the two do not conflict**,
because its interesting bed is shallow and sits under smooth grounded firn; A gets
the most dramatic bed but pays for it in surface roughness and data artifacts.

### Suggested next steps

* If one line: take **D**, trimmed to the grounded 0–66 km for the roughness law,
  and keep the full 100 km as a grounding-line/crevasse stress case.
* If two: add **A** for the altitude ladder and the deep-trough step.
* Both need a `config/lines/antarctica_pineisland*.yaml` in the new one-law-per-line
  form. `20121023_04` (9.5 MHz, dt 105 ns, tukey 0.2) and the 2014/2016/2018
  MCoRDS3 parameters are already tabulated in `cross_season_line_scout.md`;
  `20091020_06` still needs its param frame read.
* Criterion 6 has not been quantified — only read off the echograms. The C&S
  along-track Doppler method would turn the double-branch returns on A at
  15–35 km into an actual angular scattering function to test against.

---

## Addendum: CSARP_mvdr on the high-altitude passes (same day)

All 8 high-altitude passes (24 frames) were re-pulled as `CSARP_mvdr` and put
through the same assembly. The MVDR product is on the same fast-time grid as
`CSARP_standard` (identical `dt`, same sample count; the 2009 frames start one
bin later and carry slightly fewer traces). Figures `claude_notes/pig_scout/mvdr_[A-E].png`
and `mvdr_zoom.png`; code `dl_mvdr.py`, `assemble_mvdr.py`, `plot_mvdr2.py`,
`zoom_mvdr.py`; numbers `diag_mvdr.csv`, `diag_mvdr2.csv`.

**Answer: yes, clearly — and it changes the criterion-3 verdict on A.**

### Metric

The peak-over-noise-floor SNR used earlier (peak within ±0.5 µs of the pick over
the median power 3 µs *below* it) is the wrong instrument for this question: MVDR
suppresses the off-nadir energy on both sides of the bed, so the ratio can fall
while the bed becomes more visible. It reports MVDR as *worse* on A, E and every
2009 pass — which the echograms flatly contradict.

The right measure is bed over the clutter that competes with it: peak within
±0.5 µs of the pick over the **median power 1.5–5 µs above** it.

| cand | pass | AGL (m) | bed/clutter std → mvdr (dB) | Δ | fraction of picked line > 6 dB, std → mvdr |
|---|---|---|---|---|---|
| A | 20091020_06 | 10 273 | −1.3 → 1.2 | **+2.5** | 0.00 → 0.11 |
| A | 20121023_04 | 9 220 | −1.9 → 5.3 | **+7.2** | 0.00 → 0.46 |
| B | 20091020_06 | 10 273 | −1.2 → 3.1 | **+4.3** | 0.00 → 0.21 |
| B | 20121023_04 | 9 220 | −1.0 → 5.9 | **+7.0** | 0.00 → 0.49 |
| C | 20121023_04 | 9 220 | 0.1 → 6.2 | **+6.1** | 0.12 → 0.50 |
| D | 20121023_04 | 9 220 | −3.9 → −0.9 | **+3.0** | 0.00 → 0.06 |
| E | 20091020_06 | 10 273 | −1.9 → 3.6 | **+5.6** | 0.00 → 0.30 |
| E | 20121023_04 | 9 220 | −1.8 → 8.9 | **+10.7** | 0.00 → 0.67 |

**MVDR improves the bed-over-clutter contrast on every single high-altitude pass**,
by 2.5–10.7 dB. Two caveats on reading the absolute numbers:

* **Thin ice breaks the metric.** On D the bed sits only 5–12 µs below the surface,
  so the 1.5–5 µs reference window lands *inside* the surface return and the
  contrast reads negative even though the bed is obvious. For D use the
  peak-over-noise numbers instead: 21.9 → 23.2 dB, 100 % above 10 dB either way.
  Same effect weakens C's and A's shallow halves relative to their deep halves.
* The `>6 dB` fractions are over the part of each line where an OPS bottom pick
  exists, so E's 0.67 covers only its first 48 km.

### What actually changed on the echograms

* **A, s = 64–100 km (the deep trough, ~1400 m ice, bed at 22 µs) is the headline.**
  In `CSARP_standard` there is no bed at all — a featureless grey wash. In
  `CSARP_mvdr` the 9.2 km pass shows a **continuous, well-defined bed reflector
  from 67 km to the end of the line**, including the 85–88 km rise, and even the
  10.3 km 2009 pass shows a faint but traceable bed over the same stretch. The
  earlier read — "bed clearly detectable over the shallow 35 km, weakly beyond" —
  was a property of `CSARP_standard`, not of the data. On MVDR, A satisfies
  criterion 3 over most of its length.
* **B, 0–32 km**: the smeared bed in the 2009 standard product becomes a sharp,
  textured band. The deep half (32–56 km) stays empty in both. MVDR also
  substantially reduces the large 2012 artifact block at 45–58 km, though the
  82–90 km one survives.
* **C, 52–100 km**: contrast runs 8–15 dB above standard, peaking near 25 dB at
  62 km; the fraction above 6 dB in the thin-ice half goes 0.27 → 0.95. Best
  relative gain of any line where the bed was already visible.
* **D**: a consistent few-dB gain and a visibly thinner near-surface clutter
  blanket, so the 5–6 µs bed separates more cleanly from the surface return over
  45–100 km. The smallest change of the set — because D's bed was never the
  problem.
* **E**: contrast rises the most on paper (+10.7 dB) and MVDR removes the diagonal
  off-nadir streaks that fill the standard product at 20–45 km, but **no bed
  appears past s = 48 km in either product**. E still fails criterion 3.

### MVDR artifacts to be aware of

* Periodic vertical striping in low-signal regions (very visible on C at 0–50 km
  and D at 45–90 km) — the adaptive-weight block structure, not geophysics. It
  would be easy to mistake for englacial layering or crevasse curtains.
* MVDR nulls off-nadir energy by construction, so it is the *wrong* product for
  criterion 6. The double-branch bed returns on A at 15–35 km and the diffraction
  wings on D are exactly what MVDR is designed to remove: keep `CSARP_standard`
  (or `CSARP_qlook`) for the clutter-validation work and use MVDR only for bed
  detection and for picking the bed to compare against.

### Consequence for the recommendation

A moves up. With MVDR its deep trough half is usable, which makes it the strongest
line overall — 1116 m of bed relief, a 700 m step, three altitude levels, and a
bed that is now detectable along most of the line at 9.2 km. D remains the safest
choice on surface roughness and the only one whose bed needs no help at all.

---

## Addendum 2: line configs and pilot runs (same day)

Candidates A and D are now shipped lines. A is the SOUTHERN track
(lat −75.35 → −75.56), D the NORTHERN one (lat −75.17 → −74.96):

* `config/lines/antarctica_pineisland_south.yaml` — A. Five passes, three
  altitude levels: `dc8_2014_0km` / `dc8_2016_0km` / `dc8_2018_0km` (~450 m),
  `dc8_2012_9km` (9.2 km), `dc8_2009_10km` (10.3 km). Reference `dc8_2018_0km`.
  GL at s 6.7 km. Pilot **s 35–45 km** (ice 1498–1734 m, bed −1333…−1084 m:
  the trough floor just past the 35 km step).
* `config/lines/antarctica_pineisland_north.yaml` — D. Four passes, two
  altitude levels (no 2009 rung this far west). Reference `dc8_2018_0km`.
  GL at s 66.1 km. Pilot **s 55–65 km** (ice 313–439 m, 1 km short of the GL).

Both `full` segments are the whole 100 km and declare `crosses_gl`.

### Alignment (tools/line_report.py, pilot segment)

| line | pass | traces | AGL (m) | dt (ns) | lateral offset med / p95 (m) |
|---|---|---|---|---|---|
| south | dc8_2014_0km | 668 | 447 | 20.000 | 7.3 / 24.6 |
| south | dc8_2016_0km | 667 | 398 | 20.202 | 5.9 / 11.2 |
| south | dc8_2018_0km | 669 | 414 | 20.000 | reference |
| south | dc8_2012_9km | 335 | 9301 | 105.210 | 6.8 / 10.1 |
| south | dc8_2009_10km | 819 | 9846 | 105.253 | 2.6 / 7.2 |
| north | dc8_2014_0km | 675 | 495 | 20.000 | 19.2 / 38.1 |
| north | dc8_2016_0km | 674 | 480 | 20.202 | 4.7 / 7.7 |
| north | dc8_2018_0km | 674 | 460 | 20.000 | reference |
| north | dc8_2012_9km | 338 | 9600 | 105.210 | 4.7 / 6.3 |

Shared span 9.91 km (south) / 10.01 km (north) — every pass covers the whole
pilot window.

### Supporting changes

* **Instruments.** `config/instruments/mcords2_dc8_2012.yaml` and
  `mcords_dc8_2009.yaml`: both a 5-element cross-track array at 0.3937 m
  spacing = **0.2546 λ** at 193.9 MHz, from the OPR `lever_arm.m` block that
  groups 2009–2012 DC-8 `rds` (`LArx y = [−0.7874 −0.3937 0 0.3937 0.7874] m`,
  `rxchannel = 1:5`); the alternating along-track x stagger is not modelled.
  The 2012 product's `array_param.imgs` independently confirms 5 combined
  channels, `method 'standard'`, `Nsv = 1`.
  `mcords3_dc8_2016` gained the `20141029_05` / `20161104_05` / `20181107_01`
  segments — `lever_arm.m` puts 2014/2016/2018 in ONE array block, so the
  antenna and every existing chunk cache key are untouched.
* **2009-era param layout.** `run_altitude_comparison._PARAM_LAYOUTS` gained a
  third entry: the 2009 DC-8 product carries the radar struct at the top level
  as `param_radar` (a single `wfs` dict, not a per-waveform list) and `ft_wind`
  directly under `param_csarp`. Reads 193.9/9.5 MHz, Tpd 30 µs, PRF 7800,
  dt 105.253 ns, `kaiser(N,6)`.
* **kaiser window.** `map_window` now names it: modelled as hann, with the
  approximation recorded (kaiser 6 ≈ hann main lobe, −44 vs −31.5 dB
  sidelobes, so the simulated range sidelobes are modestly pessimistic).
* **No attitude in 2009.** The 2009 `CSARP_standard` product has no
  Roll/Pitch/Heading. `opr.frame_scene` already passes `nav_roll=None` (kernel
  treats it as zero); `prep_pass`'s reversed-pass Roll negation is now guarded
  and records why. The instrument declares `roll_source: none`.
* **Test isolation.** `tests/test_basal_hypotheses.py` injected `syn_14km` into
  `run_basal_clutter`'s module globals and never restored, leaking into every
  later module; adding two lines shifted the parametrisation enough to surface
  it as a `test_basal_lines::test_activation_round_trips` failure. Fixed with a
  teardown that re-activates the line clean. The suite is back to the six
  failures that already fail at HEAD (stale `real_low` pass names, per-pass
  roughness mapping, `surface_roughness: true`, a `bedmachine` bed key).

### Axis convention: two mistakes worth recording

Both were mine, both were caught by the first pilot run failing.

1. **`s0_km` and `grounding_line_s_km` are on the ANCHOR axis, not the study
   window.** `run_basal_clutter.ref_bed_picks` builds `s` by concatenating the
   UNSLICED `reference.frames` in flight order, `s = 0` at trace 0 of the first
   one. Getz shows the convention: `pilot.s0_km = 30.0` against a reference
   slice starting at trace 2020 x 14.85 m posting. Our windows start ~15.5 km
   into their reference frames, so the scout note's along-track s maps to
   anchor s + 15.48 km (south) / + 15.83 km (north). Written window-relative,
   the south GL landed at 6.7 km instead of the true anchor 22.13 km and the
   north GL at 66.1 instead of 81.9.

2. **The tool assumes grounded ice is at LOW s.** Three places take it:
   `solve_attenuation_regression` (`floating = s_m > gl_km * 1e3`),
   `apply_hybrid_bed` (`grounded = s_pix < gl_m`) and the zone-split
   reflectivity metrics. The anchor axis runs in the REFERENCE PASS'S FLIGHT
   DIRECTION and is not affected by the per-pass `reversed` flag -- reversing
   passes reorders the DATA, not the axis. The 2018 reference flew the south
   track downstream-to-inland, so grounded ice there sits at high s and the
   floating ice plain at low s. The A regression kept 4 of 97 samples and the
   run aborted.

   Fix chosen: the south line's `full` window starts just past the grounding
   line (anchor s 22.69 km) and the line declares **no** grounding line. It
   drops ~7 km of floating ice plain, which was never the point of that line,
   and leaves every metric meaningful -- better than special-casing three code
   paths every other line depends on. The north line's axis direction was
   already correct and keeps its GL crossing.

   If a future line genuinely needs the other orientation, the honest fix is
   to make those three sites direction-aware (a `gl_side` on the line, or infer
   it from the mask along the anchor track) rather than to orient lines around
   the assumption.

Anchor-axis geometry as shipped:

| | `_south` | `_north` |
|---|---|---|
| anchor axis | 148.28 km | 148.42 km |
| `full` | s0 22.69 km, 92.7 km, all grounded | s0 15.83 km, 100.0 km, `crosses_gl` |
| `pilot` | s0 50.50 km, 9.93 km | s0 70.79 km, 10.03 km |
| grounding line | none (window starts past it) | anchor s 81.9 km |

### Surface roughness: PROVISIONAL

Neither line has a line-specific ATM fit. Both point at the Tier 2
`aa_grounded_500_1500` stratum (σ 10.8 cm, ℓ 13.5 m, usability *marginal*).
That stratum was chosen on a MEASUREMENT, not the elevation rule: the grounded
surface sits at 449 m (south) / 342 m median, straddling the 500 m band edge,
but Tier 1 ILATM2 RMS along the grounded parts is 8.3 cm (south) and 4.8 cm
(north) p50, against the `<500 m` stratum's σ of 24.9 cm — that stratum is
dominated by crevassed margin sites and would over-predict by ~3×. The chosen
stratum still over-estimates the north line by ~1.6×.

`ATM_DAYS` entries for both lines are in `claude_notes/atm_roughness/atm_common.py`,
so the real fit is two commands:

```
uv run claude_notes/atm_roughness/atm_pull.py --lines antarctica_pineisland_north
uv run claude_notes/atm_roughness/atm_roughness.py --line antarctica_pineisland_north --date 2018-11-07
```

(2018-11-07 is the reference pass's day, per the one-law-per-line rule.)

### Pilot results (both lines, 2026-08-28)

`uv run python tools/run_basal_clutter.py --config config/experiments/pilot.yaml --line <line>`.
Reports: `outputs/antarctica_pineisland_{north,south}/pilot/report.html`.
Logs: `claude_notes/logs/pilot_antarctica_pineisland_{north,south}.log`.

Calibration (gamma_surface pinned at −10 dB on both, A solved):

| line | A (dB/km) | CI95 | r | n grounded |
|---|---|---|---|---|
| `_south` | **15.77** | [14.03, 17.38] | **0.93** | 97 (no GL on the line) |
| `_north` | **13.03** | [9.86, 15.88] | 0.767 | 58 (28 floating excluded) |

South's is the tightest A regression of any study line — its 330–1970 m
thickness range gives real leverage. For reference: david 12.8, getz 18.61,
westcoast 34.26 dB/km.

**Bed SCR** = the bed-window simulated bed-returns-over-surface-returns ratio
(`bed_window_bed_minus_surface_returns_db`), at the single `decomp_s_km` trace;
≥ 10 dB is the guard meaning the bed window is a bed measurement. Mid-column is
measured/simulated dB relative to the surface peak; the tail guard is the same
10 dB test over the tail fit window (bed +0.5 → +3.5 µs), segment-wide.

| line | pass | AGL | Bed SCR | mid-column meas/sim | error | tail guard |
|---|---|---|---|---|---|---|
| `_north` (s 75.8, ice ~400 m) | dc8_2014_0km | 447 m | +14.2 | −56.8 / −57.3 | **0.5** | FAIL −3.5 |
| | dc8_2016_0km | 398 m | +12.9 | −53.9 / −57.9 | 4.0 | FAIL −4.0 |
| | dc8_2018_0km | 414 m | +13.1 | −56.3 / −58.5 | 2.2 | FAIL −3.0 |
| | dc8_2012_9km | 9.3 km | **−24.6** | −23.4 / −29.7 | 6.3 | FAIL −26.2 |
| | haps_14km | 14 km | −13.2 | — / −43.1 | — | FAIL −27.7 |
| | haps_20km | 20 km | −14.0 | — / −41.4 | — | FAIL −27.3 |
| `_south` (s 55.5, ice ~1667 m) | dc8_2014_0km | 456 m | +3.3 | −45.8 / −63.8 | **18.0** | ok +18.8 |
| | dc8_2016_0km | 451 m | +6.6 | −43.8 / −64.1 | 20.3 | ok +21.2 |
| | dc8_2018_0km | 447 m | +4.5 | −45.4 / −64.9 | 19.5 | ok +20.7 |
| | dc8_2012_9km | 9.2 km | −9.8 | −22.6 / −31.8 | 9.2 | FAIL −10.1 |
| | haps_14km | 14 km | −0.8 | — / −45.8 | — | FAIL −15.1 |
| | haps_20km | 20 km | −16.7 | — / −45.8 | — | FAIL −18.4 |

#### The result worth keeping

**The mid-column gap scales with ice thickness, not with altitude.** Same
instrument, same protocol, same 450 m altitude, ten days apart, ~30 km apart:
0.5–4.0 dB error through 400 m of ice on `_north`, 18.0–20.3 dB through 1667 m
on `_south`. Getz's low pass is 70.8 dB off through ~1000 m. That is a strong
argument that the standing mid-column under-prediction is a missing ENGLACIAL
scattering term growing with path length, not an error in the surface-clutter
model — on the thin line, where the mid-column is genuinely surface returns,
the model is right to within a few dB.

**Thin ice at 9.2 km is a harsher clutter test than thick ice at 20 km.**
`_north`'s 9.2 km pass has Bed SCR −24.6 dB; getz's 9.2 km pass is +38.1 dB and
even its 20 km HAPS point only reaches −21.2 dB. With the bed ~5 µs below the
surface, the surface clutter that lands at the bed delay arrives from a
near-nadir angle where the surface is still bright. Bed-derived metrics on that
pass are measuring surface returns.

**Bed SCR and the tail guard disagree in opposite directions on the two lines.**
`_north`'s low passes clear the bed window (+12.9…+14.2) but fail the tail
(−3.0…−4.0); `_south`'s clear the tail (+18.8…+21.2) but its single decomp
trace reads +3.3…+6.6. The two are not the same statistic — one trace over
bed −0.5…+1.5 µs versus the whole segment over bed +0.5…+3.5 µs — so the
single-trace number should not be read as a line property. Worth adding a
segment-median Bed SCR before drawing conclusions from it.

#### Caveats on these numbers

* Surface roughness is the PROVISIONAL `aa_grounded_500_1500` stratum
  (σ 10.8 cm, ℓ 13.5 m), which over-estimates `_north`'s measured ATM RMS by
  ~1.6×. Every simulated clutter level above moves when the line fits land.
* `_south`'s pilot has four passes: `dc8_2009_10km` is declared on `full` only.
* Both lines' `full` segments are unrun.

---

## Addendum 3: full runs + the 14 km HAPS configuration study (2026-08-28/29)

`tools/run_basal_clutter.py --config claude_notes/pig_haps_opt/full_haps14opt.yaml --line <line>`
— a ONE-OFF experiment (the shipped `full.yaml` is untouched and stays byte-identical
apart from `run.lines`), verified identical to the shipped `full` protocol except for the
line list and two extra passes. Reports under
`outputs/antarctica_pineisland_{north,south}/full_haps14opt/`; logs
`claude_notes/logs/full_antarctica_pineisland_*.log` and `full_{north,south}_decomp.log`.

17 passes total, no failures. **The 2009 pass ran end to end on `full`** — its first ever
run in this pipeline — exercising the new `_PARAM_LAYOUTS` entry, the no-Roll guard, the
kaiser→hann mapping and `mcords_dc8_2009`. On `full` its pick reaches the true ~22.6 µs
depth (bed reach 3092 m), where on `pilot` the same pass built a window ~8 µs short and
died in `rel_mean_profile`; dropping it from `pilot` only was the right call.

### The configuration study

`claude_notes/pig_haps_opt/` — constrained sweep at 14 km AGL (fc 50–300 MHz, fractional
BW ≤ 40 %, array ≤ 10 m and ≤ 8 elements). Winner `haps14_pig_075` (75/30 MHz, 8 el at
0.3569 λ = 9.99 m); runner-up `haps14_pig_050` (50/20 MHz, 0.2378 λ = 9.98 m).

**Low frequency wins, and not for the reason the beamwidth argument predicts.** Bed SCR
falls monotonically with fc, ~12 dB from 50 to 300 MHz. The driver is the coherent
bed-roughness loss: with the `bed_roughness` fixture (σ 0.10 m) the nadir coherent bed
return loses 0.6 dB at 50 MHz but **21.8 dB at 300 MHz** (= exp(−(2k_ice σ)²), verified
against the tool's own `bed_rough_nadir_db`), while surface Bragg backscatter rises
~7.8 dB. The bed term alone predicts −21.2 dB over that span; the simulator shows
−11.7 (north) / −12.6 (south), so ~9 dB is clawed back by beam and bandwidth.

Nulls worth recording: **pulse length has exactly no effect** (20/3/1 µs bit-identical);
**bandwidth is 0.9 dB per 2×** (the 2 µs bed window spans tens of range cells, so its
clutter is geometric); **grating lobes at 1.43 λ are harmless** (matched the no-lobe
variant to 0.2 dB on both lines).

### Bed SCR, 10 decomposition points per line

Re-derived from the cached chunks after raising `full`'s `decomp_s_km` from 2 to 10
points (decomp is in neither the chunk key nor the proc-cache meta, so 275 cache hits and
zero re-simulation). Medians:

| pass | north grounded | north floating | south (all grounded) |
|---|---|---|---|
| `dc8_2014_0km` (447 m) | +3.5 | +20.8 | +11.4 |
| `dc8_2016_0km` (398 m) | +2.5 | +20.8 | +12.4 |
| `dc8_2018_0km` (414 m) | +4.4 | +22.2 | +13.6 |
| `dc8_2012_9km` (9.2 km) | −29.1 | −9.1 | −23.5 |
| `dc8_2009_10km` (10.3 km) | — | — | −16.9 |
| `haps_14km` (60 MHz, **17.5 m**) | −20.0 | −1.0 | −6.1 |
| `haps_20km` (60 MHz, 17.5 m) | −25.2 | +4.2 | −19.1 |
| `haps14_pig_075` (75 MHz, 9.99 m) | **−20.2** | +5.0 | **−10.0** |
| `haps14_pig_050` (50 MHz, 9.98 m) | −24.6 | +9.7 | −11.7 |

**75 MHz wins on both lines** (by 4.4 dB north grounded, 1.7 dB south), confirming the
sweep. A two-trace sample had shown the opposite on south; that was noise — south's low
passes read +1.5…+4.5 at s 55.5 km against 10-point medians of +11.4…+13.6, an ~9 dB
sampling error. **Two decomposition traces are not enough to rank configurations on a
100 km line.**

**The shipped `haps_60mhz` baseline is not constraint-compliant.** 8 elements at 0.5 λ at
60 MHz is **17.5 m**, 75 % over the 10 m limit, giving it a 3.50 λ aperture against f075's
2.50 λ. It beats both compliant configs on grounded ice (by 0.2 dB north, 3.9 dB south).
Within the rules 75 MHz is the best available; the aperture limit itself costs a few dB.

**Nothing clears the 10 dB guard on grounded ice at 14 km on either line.**

### Two results the pilots could not show

**Ice thickness dominates, and thick ice is EASIER for a high-altitude sounder.** Every
14 km config does 10–14 dB better on south's 1700–1830 m ice than on north's ~400 m.
Surface returns competing at the bed delay arrive at 35.8° on south vs 18.0° on north
(h/cosθ − h = c·Δt/2), and the array suppresses 35.8° far harder. At 14 km on south,
`haps_14km` reaches −6.1 dB against the real 9.2 km MCoRDS pass's −23.5 — a stratospheric
platform beats the existing high-altitude sounding on this line by ~17 dB.

**The mid-column gap scales with ice thickness, not altitude.** Measured/simulated
mid-column error on the ~450 m passes: **3.8–6.8 dB on north (400 m ice)**, **11.7–14.5 dB
on south (1700 m)**, against **70.8 dB on getz (~1000 m)** in the pilot era. On the thin
line, where the mid-column genuinely is surface returns, the model is right to a few dB.
This pair isolates the englacial-scattering deficit better than anything else in the set.

### CAVEAT: the floating bed is cross-track flat by construction

North's floating column should be read as an **upper bound, not a prediction**. The
floating bed is `picks_bed_nn(axis, s_pix)` — indexed by along-track s ONLY, then reshaped
across the grid (`run_basal_clutter.py` ~line 1067) — so past the grounding line the bed
has **zero cross-track relief**: an ideal specular mirror with nothing to scatter off-nadir.

Two symptoms confirm it is doing work. (1) On north, 20 km beats 14 km on the shelf
(+4.2 vs −1.0), which is physically backwards — more altitude should admit more clutter;
on south, which is entirely grounded, altitude hurts monotonically and steeply, as it
should. (2) The floating ordering is anti-correlated with beamwidth, i.e. the *widest*
beam scores best, which is backwards for clutter rejection.

Consequence for metrics: north's `full` window is **34 % floating**, so its segment-wide
bed-tail guard is substantially a floating-ice measurement and its ordering reproduces the
shelf Bed SCR ordering while inverting the grounded one. **Use `zone_split_*` on north, not
the segment-wide guard.** South's guard is clean (no floating zone).

This limitation is not listed in `docs/bed_scattering.md`; it should be.

### Standing caveats

* **Surface roughness is the PROVISIONAL `aa_grounded_500_1500` stratum**, ~1.6× north's
  measured ATM RMS. Every simulated clutter level moves when the line fits land.
* **Bed roughness is an unmeasured study fixture** (σ 0.10 m, ℓ = λ_ice at 190 MHz),
  tuned on getz in 2026-08-21 and never re-validated after the grazing fix. It supplies the
  larger of the two frequency terms, so it is the single biggest lever on the 14 km answer.
  Halving σ moves all levels 5.2 dB but leaves the ranking identical — the CHOICE is robust,
  the numbers are not.
* **Attenuation is frequency-independent in the model.** If real α ∝ f, 75 MHz gains a
  further 6.3 dB (north) / 31.8 dB (south) the model does not credit — the bias runs
  against low fc, so correcting it strengthens the result.
* **No receiver noise or link budget.** This is SCR only; nothing here says 75/30 MHz is
  power-feasible on a stratospheric platform.

### Next

1. Line-specific ATM surface fits (`atm_pull.py` + `atm_roughness.py`, ATM_DAYS entries
   already added) — removes the largest surface-side unknown.
2. Bound the bed roughness. It decides the frequency answer and is currently a fixture.
3. Record the `floating: picked` cross-track-flat limitation in `docs/bed_scattering.md`.
