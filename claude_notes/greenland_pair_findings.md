# Greenland altitude pair: surface+bed clutter simulation (2026-08-11)

Study of the scouted Greenland pair (`claude_notes/greenland_altitude_scout.md`,
committed as f0ebc72): **SURFACE AND BED CLUTTER ONLY** — no firn, no
englacial layers — at three altitudes on one central-west interior line.

| pass | frames | season | median AGL | role |
|---|---|---|---|---|
| `low` | `20140421_01_069` | `2014_Greenland_P3` | **446 m** (pilot) / 465 m (full) | measured |
| `high` | `20170424_01_067` | `2017_Greenland_P3` | **2 488 m** | measured |
| `syn14km` | carrier = `low` | — | **11 539 m** (14 000 m ellipsoidal) | prediction, no measured data |

Segments: pilot **s = 25–35 km** (10 km), full **s = 11–40 km** (29 km), the
scout's slices verbatim.

## BUILD: the study-line registry (`--line`)

`tools/run_basal_clutter.py` was Antarctic-wired. It is now a **line registry**,
not a fork:

* Every line-specific name (`SEASON`, `CRS`, `PASSES`, `ORDER`, `SEGMENTS`,
  `S0_KM`, `DECOMP_S_KM`, `N_TRACES_BY_SEGMENT`, `REF_PASS`/`REF_SEASON`/
  `REF_FRAMES`, `GL_S_KM`, `FC_HZ`, `CASE_PREFIX`, `OUT_DEFAULT`,
  `SYNTHETIC_KEYS`, `MEASURED_CAVEATS`, `UNSUPPORTED`) is a module global.
  `LINES[<name>]` is a dict of overrides and `activate_line()` rebinds them.
* **The Antarctic line IS the module default**, so its registry entry is `{}`
  and `--line antarctic_2016` is a literal no-op. Verified, not assumed: the
  committed Antarctic pilot chunk cache key recomputes **bit-identical**
  (`outputs/basal_clutter/pilot_pbed/runs/low_pilot_pbed_c00_srough.json`
  `meta_key`, `ct_m` 2493.14 / spacing 10.6667 unchanged).
* Total diff: ~150 lines of registry + 9 `EPSG:3031` literals → the `CRS`
  global + a `pass_season()` helper (the pair flies **two seasons**, one per
  pass — the Antarctic line has one, so `spec.get("season", SEASON)` keeps its
  cache keys identical) + `--line` pre-parsed in `main()` so `--segment`'s
  choices and help reflect the active line.
* Shared machinery reused unchanged: `run_altitude_comparison.mcords_params` /
  `map_window` / `pick_oversample` / `facet_spacing` / `base_scene`, and
  `opr.py`'s hemisphere-aware fetches (ArcticDEM v4.1 32 m + **BedMachine
  Greenland v5**, 150 m — selected from the frame's latitude, nothing to
  configure).
* Per-line guards: features not wired for a line raise instead of
  half-working — `gamma_rssnr` and `demogorgn_bed` on the Greenland line, a
  segment the line does not define (`extended`, `full_line`), and a synthetic
  pass the line does not define (`--add-30km`, `--add-500km`, `--add-300km`).

Tests: `tests/test_basal_lines.py`, **21 config-level tests**, no network/no
kernels — registry hygiene (entries may only rebind line globals), the
Antarctic no-op, activation round-trip with a snapshot/restore fixture, the
Greenland invariants (two seasons, one frame per pass per segment, trace
counts matching to ≤1, no reversal, decomposition `s` inside every segment,
5.3× altitude ratio, the scout quirks present in `MEASURED_CAVEATS`), and all
of the guards above.

## Configuration decisions (deliberate, recorded)

| choice | value | why |
|---|---|---|
| bed | **BedMachine Greenland v5 + `--picked-bed`** residual from the LOW pass | scout: BMv5 resolves only 23–46 % of the radar bed roughness. Measured effect on the pilot: along-track bed roughness rms **13.8 → 24.2 m** (matches the scout's 13.7 → 24.0 m); residual rms **25.6 m** (mean −9.4, \|max\| 66.6), **0 % pick gaps**. Nadir bed matches the picks exactly; BedMachine's cross-track relief is preserved. |
| DEMOGORGN | **not used** | Antarctic-only product (Bedmap3 grid). Guarded, not silently ignored. |
| reflectivity | **constant Fresnel γ**, eps_bed 8.0 (−12.9 dB) | RSSNR not wired for this line (see below). |
| attenuation | **15 dB/km one-way** | cold interior nominal. **ASSUMPTION**: the Antarctic effective ~20–31 dB/km was *line-calibrated* on the 2016 DC-8 anchor and is not transferable. Two-way loss here is ~74 dB over 2.46 km of ice — see the bed-visibility result. |
| processing | **CSARP_standard-matching chain** (`--processing standard --proc-cache`) | sims run at the product posting (1 sim column per measured column); aperture 251 m (low) / 356 m (high) / 822 m (syn14km), 3 looks, f-k-matched. |
| cross-track reach | **derived, uncapped** | `run_basal_clutter` applies no cap (the 6 km cap the scout flagged is `run_altitude_comparison`'s). Derived: **±5 607 m** (low), **±7 200 m** (high), **±12 050 m** (syn14km) — all beyond 6 km for the two high passes, so the cap would have clipped them. |
| fast-time | per-pass lattice, never bin-indexed | scout bite 4: dt 33.3859 vs 33.3333 ns (0.16 %) and t0 −0.167 vs 0.000 µs. Every metric is computed in µs/metres on each pass's own grid; `pick_oversample` gives **k = 4** on both. |
| window | `hann` | scout-verified: `ft_wind` decode falls back on BOTH passes; the true value IS `@hanning` (`param_csarp.csarp.ft_wind`). `param_combine` is read for `combine.method` (`standard`), not `param_csarp` (which reads `mvdr` on the 2017 frames). |

### Greenland RSSNR store check (asked for, NOT wired)

`s3://opr-radar-metrics/icechunk/greenland` **exists** and opens anonymously
(snapshot `GEAMAHQ7BRVPG9SQPK20`, same 25 variables as the antarctica store:
`required_surface_snr_dB`, `bed_twtt`, `surface_twtt`, `qc_pass`, …).
**5 698 frames / 182 segments, seasons 2013/2014/2016/2017/2018/2019.**

**Both of our segments are in it**: `20140421_01` contributes **66** frames and
`20170424_01` contributes **65** frames (`processed_frames` entries are
`Data_<frame_id>`). So an RSSNR-driven bed γ on this line is a **live
follow-up**, not a dead end — it needs a K anchoring calibrated on this line
(the Antarctic K = +7.92 dB is not transferable), which is exactly the work the
`UNSUPPORTED` guard currently refuses to fake.

## PILOT (s 25–35 km) — sanity + timing

Simulation wall time **1 381.5 s** (23 min) for three passes over 10 km at the
product posting: **low 655.0 s** (3 chunks, 1.63 M facets/interface),
**high 440.9 s** (3 × 1.10 M), **syn14km 285.6 s** (3 × 475 k).

**syn14km 2-trace geometry/phase gate** (run before the full pass, the
syn500km/syn300km pattern): at the β = 0.5 spacing (54.97 m) the ±12.05 km
window builds **79.2 m** facets and trips the Fresnel-zone LPA check (limit
54.4 m, ratio 1.46) — the known failure class. `facet_spacing_scale = 0.7`
(→ 38.48 m) clears it with **no warnings**, field finite, nadir surface/bed
delays matching the picks to **≤ 15 ns** (< 1 frame bin; DEM-vs-pick, not
phase). Recorded in the registry entry.

### Sanity checks that passed

* **Surface registration** (`leading_edge_gate`): median residual **0.33 bins**
  (low) / **0.24 bins** (high), threshold 5 — both pass.
* **The simulation independently recovers the scout's registration anomaly.**
  The fitted per-frame twtt offsets are **−2.22 bins (low)** and
  **−7.76 bins (high)**; the difference is **5.54 bins × 5.00 m = 27.7 m**,
  i.e. the scout's "~25 m / 5 range bins" surface-pick disagreement, found from
  the DEM side with no knowledge of the scout number. This is the strongest
  confirmation that per-frame registration is mandatory here.
* Picked-bed residual applied identically to all three passes, 0 % gaps.

### Pilot results

Mid-ice-column power relative to each dataset's own surface-return peak:

| pass | AGL | **measured** | **sim** | sim − meas | sim verdict |
|---|---|---|---|---|---|
| low | 446 m | **−44.54 dB** | **−76.05 dB** | −31.5 | surface-borne |
| high | 2 488 m | **−41.74 dB** | **−61.27 dB** | −19.5 | surface-borne |
| syn14km | 11 539 m | — | **−50.91 dB** | — | surface-borne |

**Altitude trend (high − low): measured +2.80 dB, simulated +14.78 dB — the
simulation overshoots the measured altitude trend by +11.98 dB**, and sits
19–32 dB below measured in absolute mid-column level.

Per-interface decomposition (the point of the study): the simulated mid-column
is **surface-borne at every altitude by a huge margin** — bed returns are
**119 dB** (low), **127 dB** (high), **103 dB** (syn14km) below surface returns
in the mid-column window. Geometric *bed* clutter contributes essentially
nothing to the ice column here; what the kernel produces is off-nadir *surface*
clutter.

Bed-window level (rel own surface peak) is, in contrast, reproduced well:

| pass | measured bed window | sim bed window | sim − meas |
|---|---|---|---|
| low | −107.10 dB | −102.03 dB | **+5.1** |
| high | −83.98 dB | −89.38 dB | **−5.4** |
| syn14km | — | −84.72 dB | — |

±5.4 dB on the absolute bed level with a constant Fresnel γ and 15 dB/km is far
better than the Antarctic line's 15–20 dB deficit — but see bite 1: at 465 m
the sim bed window is only +5.6 dB above the sim's own surface clutter, and at
altitude the bed is *below* it.

### Bed-return tail

| pass | measured slope | sim total slope | sim **bed-returns-only** slope | excess @ +2 µs | guard |
|---|---|---|---|---|---|
| low | −0.366 dB/µs | −2.038 | −6.066 | +7.90 dB | **FAIL** (−13.7 dB @ 3.47 µs) |
| high | −1.006 dB/µs | −0.275 | −5.282 | −4.92 dB | **FAIL** (−22.9 dB @ 2.93 µs) |
| syn14km | — | −0.514 | −2.535 | — | **FAIL** (−20.1 dB @ 3.47 µs) |

**The guard fails on all three passes.** That is the guard doing its job, not a
bug: with 2.46 km of ice at 15 dB/km (≈74 dB two-way) and a constant −12.9 dB
Fresnel γ, the simulated bed return is buried under the simulated off-nadir
*surface* clutter across the whole bed+0.5…+3.5 µs fit window, so the
total-field tail slope/excess are **upper bounds** and only the
`bed_returns_slope_db_per_us` column is a bed measurement. Delay→refracted
in-ice incidence: +3 µs = **21.1°** (low), **14.7°** (high), **8.3°** (syn14km).

## FULL SEGMENT (s 11–40 km) — the study result

Simulation wall time **3 896.8 s (65 min)**: **low 1 804.6 s**, **high
1 269.5 s**, **syn14km 822.7 s**, 10 chunks per pass, 1 939 sim columns per
pass (one per measured column). The pilot projection (2.9× the pilot's
1 381.5 s → 67 min) was accurate to **3 %**. Chunk caches are resumable, so a
re-analysis costs only the figures.

Derived geometry (full segment): reach **±5 625 m** (low) / **±7 200 m**
(high) / **±12 050 m** (syn14km); facet spacing 10.67 / 16.00 / 38.39 m;
SAR aperture 251 / 355 / 820 m (18 / 25 / 56 traces), 3 looks; mocomp dz rms
0.084 / 0.060 / 0.000 m. Picked-bed residual rms **41.7 m** (mean +0.6,
\|max\| 136.2), **0 % gaps**, along-track bed roughness **18.5 → 39.8 m rms**
(the scout's 18.4 → 39.8 m for this segment, reproduced exactly).

### Mid-ice-column clutter (rel own surface-return peak)

| pass | AGL | **measured** | **sim** | sim − meas | sim verdict | measured floor |
|---|---|---|---|---|---|---|
| low | 465 m | **−41.96 dB** | **−78.08 dB** | **−36.1** | surface-borne | −115.81 dB |
| high | 2 483 m | **−40.87 dB** | **−61.32 dB** | **−20.5** | surface-borne | −83.36 dB (INVALID, bite 3) |
| syn14km | 11 536 m | — | **−51.05 dB** | — | surface-borne | — |

**Altitude trend (high − low): measured +1.09 dB, simulated +16.76 dB — the
simulation overshoots by +15.67 dB.** (Pilot: +2.80 measured vs +14.78
simulated, error +11.98 dB. Same sign, same size; the 29 km segment is the
number to quote.)

Per-interface decomposition — **the study's discriminator**. Simulated
mid-column power, split by which interface the energy came from:

| pass | surface returns | bed returns | bed − surface |
|---|---|---|---|
| low | −78.08 dB | −195.56 dB | **−117.5 dB** |
| high | −61.32 dB | −185.05 dB | **−123.7 dB** |
| syn14km | −51.05 dB | −135.25 dB | **−84.2 dB** |

**The simulated ice column is surface-borne clutter at every altitude, by
84–124 dB.** Geometric *bed* clutter contributes nothing measurable to the
column on this line.

### Bed window (rel own surface-return peak)

| pass | measured | sim total | sim surface | sim bed | sim − meas |
|---|---|---|---|---|---|
| low | −107.76 | −103.41 | −110.20 | −104.75 | **+4.35** |
| high | −83.95 | −89.62 | −90.19 | −99.28 | **−5.67** |
| syn14km | — | −85.01 | −85.22 | −99.01 | — |

Absolute bed level lands within **±5.7 dB** of measured with a constant
Fresnel γ and 15 dB/km — much better than the Antarctic line's 15–20 dB
deficit. But the decomposition shows why that is fragile: the sim's bed
window is **+5.5 dB bed-over-surface at 465 m**, **−9.1 dB at 2 483 m** and
**−13.8 dB at 11 536 m**. Above ~2.5 km the simulated "bed window" is a
surface-clutter measurement.

### Bed-return tail

| pass | measured slope | sim total | sim **bed-returns-only** | excess @ +2 µs | guard | +3 µs → in-ice angle |
|---|---|---|---|---|---|---|
| low | −0.236 dB/µs | −1.892 | **−5.970** | +6.23 dB | **FAIL** (−12.3 dB) | 20.96° |
| high | −1.014 dB/µs | −0.338 | **−4.479** | −4.30 dB | **FAIL** (−21.4 dB) | 14.70° |
| syn14km | — | −0.450 | **−2.520** | — | **FAIL** (−22.0 dB) | 8.32° |

Guard FAILs on all three: the total-field tail is surface clutter, so the
total-field slope/excess are **upper bounds** and only the bed-returns-only
column is a bed measurement. Sim record coverage 1.000 / 0.998 / 1.000.

### Surface registration (full segment)

Median residual **0.28 bins** (low) / **0.30 bins** (high), threshold 5 — both
pass. Fitted offsets **−2.00** and **−7.70** bins; difference **5.70 bins ×
5.00 m = 28.5 m**, again independently recovering the scout's ~25 m
registration anomaly (pilot: 27.7 m).

### Single-trace decomposition at anchor s = 30.0 km

Sim trace 1270/1271 = measured trace 1270/1271. Bed-window bed − surface
returns: low **+8.6 dB**, high **−12.1 dB**, syn14km **−13.2 dB**.

## Figures (provenance-labeled, Antarctic-campaign convention)

`outputs/greenland_pair/full_pbed_proc/` (the study result) and
`outputs/greenland_pair/pilot_pbed_proc/`, each mirrored to
`outputs/verification/greenland_pair_<segment>_pbed_proc/`
(metrics.json + all four figures):

* `radargrams.png` — measured vs simulated per pass on a surface-referenced
  twtt axis; **no measured panel for syn14km** (it renders as a PREDICTION
  panel, the tool's synthetic convention).
* `decomposition.png` — **surface returns vs bed returns** ensemble profiles
  per pass (the study's discriminator).
* `decomposition_trace.png` — single-trace decomposition at **anchor
  s = 30.00 km** (parameterized: `--trace-decomp-s`; sim column = measured
  column on both passes — trace 1270/1271 full, 334 pilot).
* `bed_tail.png` — bed-referenced tails, fit window shaded, measured floor.
* `metrics.json`, `report.html`, `run_config.json`.

## Bite list — things that will mislead if not read

1. **The simulated bed is invisible under simulated surface clutter at every
   altitude** (guard FAIL on all three passes; mid-column bed contribution
   **84–124 dB** down on the full segment). Any statement about "basal clutter vs altitude" from this
   configuration is really a statement about **surface** clutter. This is the
   scout's bite 9 confirmed quantitatively: the bed here is flat (270 m relief
   / 29 km, ~1.6° slopes) and 2.4–2.5 km deep, so it produces almost no
   off-nadir return, while the surface has a 29 µs-long ice column to fill.
2. **The 15 dB/km assumption is doing a lot of work and is NOT calibrated on
   this line.** 74 dB two-way is the single biggest lever on bed visibility.
   The bed-window levels land within ±5.4 dB of measured, which is encouraging,
   but that is one number reproduced by a two-parameter (γ, A) product — do not
   read it as validation of either factor separately. An RSSNR-driven γ on the
   (now confirmed to exist) Greenland store is the way to break the degeneracy.
3. **The measured noise-floor estimate is INVALID for the high pass.** The
   tool's floor window is `record end −12 … −8 µs` = **43.4–47.4 µs**, and the
   high pass's bed sits at **42.3–47.5 µs** — the "floor" window *is* the bed.
   Its reported `measured_floor_rel_surf_db` (−83.36 dB full, −83.05 pilot,
   i.e. within 0.6 dB of the measured BED window level) is a bed level, which
   is why the measured tail reads 2.45 dB *below* its own "floor" and
   `floor_limited` trips. This is the scout's bite 2 (only 7.9 µs of post-bed
   tail at altitude) biting exactly as predicted. **Fix before trusting any
   high-pass floor/`noise_limited` number**: make the floor window per-line (or
   per-pass, derived from the deepest bed pick). The low pass is fine (floor
   window sits 9 µs past its deepest bed).
4. **The simulated altitude trend is ~15× too steep on the full segment**
   (+16.76 dB simulated vs **+1.09 dB** measured, high−low; the pilot's
   10 km window gave +14.78 vs +2.80). Since the sim mid-column is entirely surface-borne,
   this says the *geometric surface-clutter* model grows with altitude much
   faster than the real ice column's mid-column power does. The measured
   mid-column is essentially FLAT with altitude (+1.1 dB over a 5.3x altitude
   ratio) — consistent with it being dominated by englacial volume scattering
   and internal layers, which are altitude-insensitive and which this study
   explicitly does not model. Do not tune roughness to close a 20-36 dB
   absolute gap that is physically the missing englacial term; the honest
   conclusion is that surface+bed geometric clutter does NOT explain the
   measured ice-column power on this line at either altitude.
5. **Two seasons, two lattices.** dt differs 0.16 % and t0 by 5 bins; the tool
   never index-aligns across passes, but any downstream analysis must not
   either.
6. **`img_comb` blend lengths differ** between the passes (low
   `[3 µs, −∞, 2.64 µs; 10 µs, −∞, 3.5 µs]` vs high `[3, −∞, 1; 10, −∞, 3]`),
   so the first ~3 µs below the surface is not one instrument across passes.
   The sim models the 10 µs bed waveform only, which is right for the bed zone.
7. **`--picked-bed` replicates 1-D pick detail as cross-track ridges** out to
   ±5.6/±7.2/±12.1 km (the tool's own documented caveat). At ±12 km on the
   syn14km pass that is a long extrapolation of a single along-track profile.
8. **syn14km needs `facet_spacing_scale = 0.7`** or it silently violates the
   linear-phase approximation (ratio 1.46). Any new synthetic altitude on this
   line needs its own 2-trace gate.
9. The high pass's sim `record_coverage_frac` is **0.997** at bed+3.5 µs — the
   sim window is fine, but there is essentially no headroom; a longer tail
   window would run off the end of the record.


---

# Part 2 (2026-08-11): attenuation estimated from the measured data

## 2.0 First: the per-pass noise-floor window (bite 3, now fixed)

`floor_window()` derives the measured-floor window PER PASS: keep the
established `record end -[12, 8] us` whenever it clears
`(deepest bed pick + 1.5 us)`, else slide to
`[deepest bed + 1.5 us, record end - 4 us]` (the documented roll-off), and
report `valid: false` if under 1 us is left. Effect:

| pass | window (us) | slid | floor (rel surf) | change |
|---|---|---|---|---|
| Greenland low | [43.254, 47.254] | no | **-116.42 dB** | unchanged (bit-identical) |
| Greenland high | [49.033, 51.400] | **yes** | **-88.30 dB** | was **-83.05 dB** (contaminated) |
| every 2016 DC-8 pass | end -[12, 8] us | no | — | unchanged |

The old high-pass "floor" sat within **0.6 dB of that pass's own measured bed
window** — it was the bed. The corrected floor is **5.25 dB lower**, and the
high pass's bed window is **4.7 dB** above it (not 0.6 dB below). Its bed
level is now usable; it is still the least comfortable margin in the study.

## 2.1 Route (a): per-pass level matching (study segment, s 11-40)

With constant Fresnel gamma the simulated bed level moves dB-for-dB with
`-2*A*H`, so `A_pass = A_run + residual/(2*H_med)`; sensitivity **4.94 dB per
dB/km** (low) / **4.95** (high) at H ~ 2.47 km.

| pass | H_med | meas raw | floor | margin | meas floor-corr | sim (A=15) | resid raw | resid corr | **A raw** | **A floor-corr** |
|---|---|---|---|---|---|---|---|---|---|---|
| low | 2 468 m | -107.76 | -115.81 | 8.05 | -108.50 | -103.41 | **+4.35** | +5.09 | **15.88** | **16.03** |
| high | 2 475 m | -83.95 | -88.67 | 4.72 | -85.73 | -89.62 | **-5.67** | -3.89 | **13.85** | **14.21** |

**The two passes pull in opposite directions.** Spread **2.03 dB/km** raw,
**1.82 dB/km** floor-corrected; the pair's rms-optimal value is **14.87**,
i.e. route (a) on its own says "keep 15". That is not evidence that 15 is
right -- it is evidence that route (a) cannot resolve A here, because the two
passes' *measured* bed levels differ by **23.8 dB** while the simulation says
they should differ by **13.8 dB**. A moves both passes the same way, so no
value of A can close a 10 dB pass-to-pass disagreement.

Route (a) is also structurally degenerate: a constant error in gamma maps
1:1 into A. It constrains the PRODUCT `gamma * 10^(-2AH/10)`, never A alone.

## 2.2 Route (b): bed power vs thickness, full 99.7 km line (measured only)

Bed-window mean power rel own surface peak, **spreading-corrected** by
`(r_bed_eff/r_surf)^2` with `r_bed_eff = r_surf + H/n` (the RSSNR convention,
so the simulator's own spreading is not double-counted). Slope = `-2A`.

Per trace, whole line:

| pass | n | H range | A OLS | A Theil-Sen | r | resid rms |
|---|---|---|---|---|---|---|
| low | 6 664 | 1 937-2 680 m | 6.19 | 6.57 | **-0.496** | **4.47 dB** |
| high | 10 005 | 1 907-2 787 m | **13.50** | **13.13** | **-0.950** | **2.29 dB** |

**Answer to "does this cold interior bed behave better than the Antarctic
one?" -- YES at altitude, NO at low level.** The high pass gives a genuinely
well-determined regression; the low pass does not.

Robustness (all four checks agree on which pass to trust):

* **Along-track binning** (1 / 10 / 50 / 200 traces = 0.015 / 0.15 / 0.75 /
  3 km): low stays **6.19 / 6.28 / 6.22 / 6.60** (r only -0.50 -> -0.65,
  resid 4.47 -> 3.13 dB); high **13.50 / 13.59 / 13.66 / 13.90**
  (r -0.950 -> **-0.977**, resid 2.29 -> **1.56 dB**). Binning does not move
  the low pass, so its shallow slope is **systematic, not speckle**.
* **Thickness sub-ranges** (thin half / thick half / thinnest quartile): low
  5.55 / 5.19 / 6.80 with r -0.134 / -0.334 / -0.131 (uninformative
  everywhere); high 11.35 / 13.07 / 12.53 with r -0.630 / -0.822 / -0.559
  (stable). Noise-floor censoring of the low pass is therefore NOT the
  explanation -- its thin (bright, unclampable) half is just as flat.
* **Floor-pedestal subtraction in POWER** at 0.75 km bins: low 6.22 ->
  **8.04** OLS / **8.29** TS (r -0.60, resid 4.34, TS CI **[6.28, 10.43]**);
  high 13.66 -> **14.24** OLS / **13.91** TS (r **-0.974**, resid **1.72**,
  TS CI **[13.42, 14.40]**).
* **Brightness screening REJECTED as a method.** Keeping only traces
  `>= floor + 10/15 dB` drives the low pass to A = 1.17 / -4.19 dB/km: that
  screen selects on the regression residual (bright bed = low-attenuation
  anomaly), a collider selection. Reported only so it is not retried.

The low pass fails because at 465 m AGL the bed footprint is small, so
trace-to-trace bed-reflectivity variation (4.3-4.5 dB rms, irreducible by
binning) swamps a thickness trend worth ~20 dB over its 744 m of H range. The
high pass's much larger footprint averages that heterogeneity down -- the
same physical reason the Antarctic attempt failed, showing up here as a
per-altitude effect rather than a per-line one.

## 2.3 The chosen A, and the rule

**RULE.** (1) Only route (b) isolates A -- it is a *slope*, independent of the
assumed gamma; route (a) is a *level* match and is degenerate with gamma, so
it is a consistency check, never the primary estimate. (2) A route-(b)
estimate is admissible only if it passes a four-part gate: `|r| >= 0.9`,
residual rms `<= 3 dB`, Theil-Sen 95% CI half-width `<= 1 dB/km`, and slope
stable within `+-2 dB/km` across both thickness sub-ranges and along-track
bin scales. (3) Take the mean of the admissible route-(b) estimate and the
route-(a) estimate **from the same pass**, so the two routes are compared
like for like.

The high pass passes all four gate conditions; the low pass fails all four.
Applying the rule:

| estimate | value |
|---|---|
| route (b), high pass, floor-subtracted, 0.75 km bins, Theil-Sen | **13.91** |
| route (a), high pass, floor-corrected | **14.21** |
| **mean -> adopted** | **A = 14.0 dB/km** |

Two methodologically independent routes -- one slope-based and
gamma-free, one level-based -- agree to **0.30 dB/km** on the pass where both
are well-conditioned. Retained as a systematic envelope, not as estimates:
route (a) low **16.03** (upper) and route (b) low **8.3** (lower), i.e. the
honest per-line uncertainty is roughly **+-2 dB/km**, dominated by the low
pass's disagreement rather than by any statistical error.

**Predicted consequence, stated before the rerun.** dA = -1.0 dB/km brightens
the simulated bed by `2 * 1.0 * 2.47 = +4.94 dB`, so the high pass's residual
should go **-5.67 -> ~-0.7** (nearly zeroed) and the low pass's
**+4.35 -> ~+9.3** (worse). That is expected and is not an argument against
the value: the low-pass residual is a *level* disagreement between the two
measured passes (23.8 dB apart where the sim says 13.8 dB), which only gamma
or a pass-specific effect can explain -- attenuation moves both passes
together.


## 2.4 RERUN at A = 14.0 -- before/after, and a correction to route (a)

Full segment (s 11-40), all three passes, everything else unchanged
(picked bed, constant Fresnel gamma, matched CSARP_standard processing).
Simulation wall **3 934.9 s (66 min)**: low 1 841.4 s, high 1 270.9 s,
syn14km 822.7 s -- within 1 % of the A = 15 run, as expected (attenuation
changes no geometry).

### Bed-window levels, dB rel own surface peak

| pass | component | A = 15 | A = 14 | delta |
|---|---|---|---|---|
| low | **bed returns** | -104.75 | **-99.87** | **+4.88** |
| low | surface returns | -110.20 | -110.20 | +0.00 |
| low | total (metric) | -103.41 | -99.38 | +4.03 |
| low | measured | -107.76 | -107.76 | — |
| low | **residual (total - meas)** | **+4.35** | **+8.38** | +4.03 |
| high | **bed returns** | -99.28 | **-94.33** | **+4.95** |
| high | surface returns | -90.19 | -90.19 | +0.00 |
| high | total (metric) | -89.62 | -88.68 | +0.94 |
| high | measured | -83.95 | -83.95 | — |
| high | **residual (total - meas)** | **-5.67** | **-4.73** | +0.94 |
| syn14km | bed returns | -99.01 | **-94.07** | **+4.94** |
| syn14km | surface returns | -85.22 | -85.22 | +0.00 |
| syn14km | total | -85.01 | -84.61 | +0.40 |

Everything else is untouched by A, exactly as it should be: mid-column levels
(-78.08 / -61.32 / -51.05 dB), the altitude trend (measured +1.09 vs
simulated +16.76 dB), the surface registration and the decomposition verdicts
are all bit-identical between the two runs.

### The correction: route (a) is invalid on the HIGH pass

The **bed-returns** component moved by **+4.88 / +4.95 / +4.94 dB** -- exactly
the predicted `2 * dA * H = 4.94 dB`. But the **total** bed window moved only
**+4.03** (low), **+0.94** (high) and **+0.40 dB** (syn14km), because the
total also contains A-independent SURFACE clutter. So the pre-stated
prediction ("high residual -5.67 -> ~-0.7") was wrong, and the reason matters
more than the number:

> **The bed-window residual is only a calibration target where the bed window
> is bed-dominated.** In the sim that is true for the low pass (bed returns
> +5.5 dB over surface returns at A = 15, +10.3 dB at A = 14) and false for
> the high pass (bed **-9.1 dB below** surface returns at A = 15, -4.1 dB at
> A = 14) and for syn14km (-13.8 dB). On the high pass the total bed-window
> level moves only **0.19 dB per dB/km of A** (0.94 / 4.95), so route (a)
> there is ~26x less sensitive than the `2H` figure assumes and cannot
> constrain A at all.

This inverts the pass selection in the rule of section 2.3: **route (a) is
valid only on the LOW pass, route (b) only on the HIGH pass.** No single pass
supports both routes, so the "mean of the two routes on the same pass" step
cannot be executed as written. The adopted value therefore rests on:

| estimator | pass | value | status |
|---|---|---|---|
| route (b) (slope, gamma-free) | high | **13.91-14.24** | admissible (gate passed) |
| route (a) (level) | low | **15.88-16.03** | admissible but gamma-degenerate |
| route (a) (level) | high | 13.85-14.21 | **INVALID** -- bed window is surface clutter |
| route (b) (slope) | low | 6.2-8.3 | **INVALID** -- gate failed on all four criteria |

**A = 14.0 dB/km is retained**, now justified by route (b) on the high pass
alone (13.91 Theil-Sen / 14.24 OLS, r = -0.974, residual 1.72 dB), with route
(a) on the low pass (16.0) as the upper bound of a **[13.9, 16.0] dB/km**
envelope. Route (b) is preferred because it is measured-only and independent
of gamma, whereas route (a)-low inherits the gamma error 1:1 (a 5 dB gamma
error is 1.0 dB/km of A). Caveat in the other direction: any residual
measured surface clutter in the high pass's bed window is thickness-
independent and flattens its slope, so **route (b)-high is a lower bound**;
that is consistent with it sitting at the bottom of the envelope.

### Net effect of the change

* **High pass**: residual -5.67 -> **-4.73 dB** (improved, but only via the
  0.94 dB the surface-clutter-dominated total allows).
* **Low pass**: residual +4.35 -> **+8.38 dB** (worse, as predicted in 2.3).
* The **23.8 dB** measured pass-to-pass bed-level difference vs the sim's
  **13.8 dB** is unchanged and remains the dominant unexplained discrepancy.
  It is not an attenuation problem -- A moves both passes together. Closing it
  needs either a pass-dependent effect (system calibration, surface-peak
  normalization) or an RSSNR-driven gamma; the Greenland RSSNR store (section
  "Greenland RSSNR store check") contains both segments and is the obvious
  next step.

## 2.5 Bed-pick overlays (and a framing bug they exposed)

`fig_radargrams(..., bed_overlay=True)` (default ON, `--no-bed-overlay` to
disable) now draws the **measured Bottom pick** on every measured panel and
the **sim bed-layer nadir twtt** on every simulated panel, in cyan, from the
same arrays the bed-window metrics use -- so a visible mismatch between line
and echo is a real registration/pick problem, not a plotting artefact.

Adding them exposed a genuine framing bug: the panel window was hard-coded
`-1 .. 13.5 us` and the colour range `-90 .. 5 dB`, both sized for the
Antarctic anchor (bed ~8 us below the surface, ~55 dB down). On this line the
bed is **26-31 us** below the surface and **~108 dB** down, so **every
Greenland radargram figure produced before this change showed only the top
40 % of the ice column and no bed at all.** `RADARGRAM_Y_US` and
`RADARGRAM_DB` are now line globals: Antarctic keeps `(-1.0, 13.5)` /
`(-90.0, 5.0)` (untouched, asserted in tests), Greenland uses
`(-1.0, 34.0)` / `(-120.0, 5.0)`. With the correct framing the overlay lands
exactly on the bed echo on all three passes, and the measured low panel shows
the dense internal layering that the surface+bed simulation has no term for
-- the visual counterpart of the 36 dB mid-column gap.

## 2.6 Deliverables and tests (part 2)

* `outputs/greenland_pair/full_pbed_proc_att14/` -- `radargrams.png` (with
  overlays, corrected framing), `decomposition.png`,
  `decomposition_trace.png`, `bed_tail.png`, `metrics.json`, `report.html`,
  `run_config.json`; mirrored to
  `outputs/verification/greenland_pair_full_pbed_proc_att14/`.
* `outputs/greenland_pair/full_pbed_proc/` (A = 15) retained for the
  before/after comparison.
* Tests: **341 passed**. New this part -- 5 floor-window tests in
  `tests/test_basal_tail.py` (default window kept when the tail is long, slid
  off a bed that reaches into it, `valid: false` when nothing is left, a
  property sweep that the window never lands on the bed, and
  `meas_tail_stats` tolerating a null floor) and 2 framing tests in
  `tests/test_basal_lines.py`.
* Note: `tests/test_refraction_joint.py::test_compile_flat_in_n_and_runtime`
  fails intermittently when a 65-minute simulation is saturating the CPU (it
  asserts on wall-clock); it passes standalone.


---

# Part 3 (2026-08-11): RSSNR-driven bed reflectivity on the Greenland line

## 3.1 Wiring

The Antarctic RSSNR machinery is reused verbatim (`fetch_rssnr_anchor` ->
`rssnr_gamma_profile` -> `build_rssnr_gamma` -> `apply_rssnr_gamma`, mapping
`|Gamma_bed|^2 dB = 2*A*H(s) - RSSNR(s) + K`). Only configuration moved into
the line registry:

| global | Antarctic | Greenland |
|---|---|---|
| `RSSNR_STORE.prefix` | `icechunk/antarctica` | **`icechunk/greenland`** |
| `RSSNR_SNAPSHOT` | `3YH47013745B2T5ZZR50` | **`GEAMAHQ7BRVPG9SQPK20`** |
| `RSSNR_CACHE` | `outputs/basal_clutter/...` | `outputs/greenland_pair/rssnr_anchor.npz` |
| `LEVEL_ANCHOR_DEFICIT_DB` | +14.8 | **-7.89** |

`gamma_rssnr` is removed from the Greenland `UNSUPPORTED` list;
`demogorgn_bed` and `hybrid` stay unsupported. The fetch runs on
`REF_FRAMES` = the **LOW pass** (`20140421_01_069/_070`) -- the same
pick/reference convention the picked bed uses, so gamma and bed share one
along-track axis.

One new API knob was needed: `--companion-name`. The acceptance analysis
compares against a **constant-gamma companion run**, whose directory the tool
derives from `segment + case_tag`. Because this study's constant-gamma run
lives under `--out-name full_pbed_proc_att14` (to keep A = 15 and A = 14
side by side), the derived path would have pointed at the **A = 15** run and
silently re-simulated 66 minutes of chunks at the wrong attenuation.
`--companion-name` names it explicitly, and a missing/mismatched companion
now fails **in milliseconds, before any frame or DEM work**.

## 3.2 What the RSSNR field looks like on this line

76 decimated samples along the 99.7 km anchor line (**1 333 m** median
spacing), **92.1 % QC-pass** (6 censored of 76), **21 samples inside the
29 km study segment**.

* **RSSNR itself**: 72.2 - 100.1 dB, median **83.6**, p5-p95 77.0-95.7 --
  **28 dB of dynamic range** (larger RSSNR = dimmer bed).
* Dataset-internal ice thickness (from its own surface/bed twtts):
  1 937 - 2 670 m, median 2 185 m.
* **Mapped |Gamma_bed|^2 over the segment** (A = 14.0, level-anchored):
  min **-30.8**, p5 -29.2, med **-20.8**, p95 -13.8, max **-11.6 dB**.
  Sample-to-sample swings of 15 dB between adjacent 1.3 km samples --
  genuine along-track structure, not a smooth trend:
  `-17.6 -14.5 -26.9 -27.7 -30.8 -26.3 -20.5 -29.0 -26.5 -21.7 -18.4 -13.8
  -18.5 -27.4 -29.1 -16.7 -16.5 -29.2 -11.6 -16.4 -20.8` dB.
* Censored samples take the segment minimum (-30.8 dB) as a brightness
  FLOOR, never interpolated across.

## 3.3 K and its derivation (contamination-aware level anchoring)

| quantity | value |
|---|---|
| K_median (median |Gamma|^2 = the constant Fresnel value) | **+2.06 dB** |
| deficit D (contamination-aware, LOW pass, bed-returns) | **-7.89 dB** |
| **K_level (adopted)** | **-5.83 dB** |
| K_phys (Fresnel surface + 2-way transmission) | **-10.32 dB** |
| **K - K_phys** | **+4.50 dB** |
| implied effective attenuation | **14.9 dB/km** |

`D = median(measured bed window) - median(simulated BED-RETURNS bed window)`
on the **LOW pass only**, from the A = 14.0 constant-gamma full-segment run:
`-107.76 - (-99.87) = -7.89 dB`. Two deliberate choices, both following the
route-validity finding of section 2.4:

* **Solved through the decomposition**, not the total field: the total also
  contains A- and gamma-independent surface clutter, which would bias D.
* **The HIGH pass is excluded from the solve**: its bed window is
  surface-dominated (bed returns 4.1 dB *below* surface returns at A = 14),
  its total level moves only 0.19 dB per dB/km, so including it would drag D
  toward the clutter floor. Its post-run residual is instead the
  **transfer test**.

**K - K_phys = +4.50 dB is remarkably small.** On the Antarctic family this
gap was large enough to be a headline caveat; here the absolute chain
(Fresnel surface, 14 dB/km attenuation, Fresnel ice->rock bed) is nearly
self-consistent, and the anchoring only has to absorb 4.5 dB. Equivalently,
the level anchoring implies an **effective attenuation of 14.9 dB/km**
against the adopted 14.0 -- an *independent* corroboration of the part-2
estimate to within **0.9 dB/km**, from a completely different constraint
(absolute level, not slope).

## 3.4 Physicality of the implied |Gamma|^2

| diagnostic | value | reading |
|---|---|---|
| fraction of segment samples with |Gamma|^2 > 0 dB | **0.000** | no unphysical reflectivity anywhere |
| fraction above the Fresnel ice->rock ceiling (-12.86 dB) | **0.048** (1 of 21) | one sample only |
| max excess over that ceiling | **+1.31 dB** | marginal |
| segment median vs the ceiling | **7.9 dB below** | comfortably sub-rock |

This is a clean result. The Antarctic runs needed a documented caveat about
a positive-|Gamma|^2 fraction (reflectivity > 1, the price of median-anchoring
on a dim-bed segment); here **nothing exceeds 0 dB**, and essentially nothing
exceeds the rock ceiling either -- the single 1.3 dB excursion is well within
what a wetter or smoother basal patch would give. The mapped field is
physically admissible as it stands, which is the strongest evidence so far
that A = 14 and the constant-gamma baseline are mutually consistent on this
line.

## 3.5 Plot change

The bed overlay is now **dotted** and appears on **measured panels only**.
The sim-panel bed line was removed entirely: the sim's bed nadir twtt is a
model *input*, not an independent pick, so drawing it on the simulated
radargram invites reading a tautology as agreement. The measured Bottom pick
stays, dotted, because there it is a genuine cross-check of registration
against the echo.


## 3.6 RERUN with the RSSNR gamma -- results

Full segment (s 11-40), all three passes, `--gamma-from-rssnr --anchor level`,
everything else identical to the A = 14.0 constant-gamma run (picked bed,
matched CSARP_standard processing). Simulation wall **3 975.7 s (66 min)**:
low 1 875.3 s, high 1 274.4 s, syn14km 825.9 s -- **1.0 % above** the
constant-gamma run, as expected (gamma changes no geometry). The
constant-gamma companion was resolved from `--companion-name` and hit its
cache on every chunk (`[skip-exists]`), so the acceptance analysis cost
nothing.

### Bed-window levels, dB rel own surface peak (constant -> RSSNR)

| pass | component | constant | RSSNR | delta |
|---|---|---|---|---|
| low | **bed returns** | -99.87 | **-108.74** | **-8.87** |
| low | surface returns | -110.20 | -110.20 | +0.00 |
| low | total | -99.38 | -105.32 | -5.94 |
| low | measured | -107.76 | -107.76 | — |
| low | **residual** | **+8.38** | **+2.44** | **-5.94** |
| high | **bed returns** | -94.33 | **-102.88** | **-8.55** |
| high | surface returns | -90.19 | -90.19 | +0.00 |
| high | total | -88.68 | -89.80 | -1.12 |
| high | **residual** | **-4.73** | **-5.85** | -1.12 |
| syn14km | bed returns | -94.07 | **-101.77** | **-7.70** |
| syn14km | total | -84.61 | -85.07 | -0.46 |

**Level-anchor verification**: post-run median residual **-1.70 dB** against a
2 dB gate -- **PASS**. Per pass: low **+2.44 dB** (the solve target; the
residual is positive because the total also carries surface returns that the
bed-only solve did not include), high **-5.85 dB**.

**The transfer test FAILS, and informatively.** K moves only bed returns, and
it moved the high pass's bed returns by the full -8.55 dB -- but the high
pass's *total* bed window moved only **-1.12 dB**, because that window is
surface clutter. The same anchoring that brings the low pass to +2.44 dB
leaves the high pass at -5.85 dB. This is the third independent confirmation
of the study's central structural finding: **above ~2.5 km AGL on this line,
the bed window is not a bed measurement**, so it can neither calibrate a bed
model nor test one.

Mid-column levels (-78.08 / -61.32 / -51.05 dB) and the altitude trend
(measured +1.09 vs simulated +16.76 dB) are **bit-identical** to the
constant-gamma run -- bed gamma touches only bed returns, exactly as it
should.

### ACCEPTANCE: along-track bed-brightness correlation vs measured

Pearson r of the ~1 km-smoothed bed-window power profile (dB rel own surface
peak), same bed geometry in both runs:

| pass | **constant gamma** | **RSSNR gamma** | change | data-only ceiling (`r_implied_vs_measured`) | by-construction check (`r_bedlayer_vs_implied`) |
|---|---|---|---|---|---|
| **low** (465 m) | **+0.193** | **+0.604** | **+0.411** | **+0.897** | +0.823 |
| high (2 483 m) | +0.708 | +0.618 | -0.090 | **+0.070** | +0.784 |

**On the low pass the RSSNR field works: r goes +0.19 -> +0.60** against a
data-only ceiling of +0.897. The constant-gamma profile is nearly flat and
misses every along-track feature; the RSSNR profile reproduces the measured
dips at s ~ 31 and ~ 35.5 km and the rises at ~ 26.5 and ~ 33 km, and drops
the level from ~ -96 to ~ -104 dB against a measured ~ -106 dB
(`bed_brightness.png`, left panel). The Antarctic benchmark went ~0 -> ~0.8;
this line goes +0.19 -> +0.60 with a ceiling of +0.90, i.e. the simulation
captures **67 %** of the achievable correlation.

**The high pass's numbers must not be read as a bed-model score.** Its
`r_implied_vs_measured` ceiling is **+0.070**: the measured high-pass
bed-brightness profile has essentially no relationship to the RSSNR pattern
in the first place (its bed window sits only 4.7 dB above the noise floor and
is surface-clutter crowded). Consistently, `r_sim_rssnr_vs_implied` is
**+0.048** for the high pass against **+0.784** for its bed-layer-only field
-- the imposed gamma is faithfully rendered in the bed layer and then buried
in the total. The constant-gamma +0.708 is therefore a *geometric* agreement
(both sims track the same terrain-driven clutter), not evidence about the
bed; the -0.09 change is noise on a meaningless baseline.

### Verdict on the RSSNR wiring

Accepted for the LOW pass and for any future bed-reflectivity work on this
line; **not** usable as an acceptance metric at altitude, for the same
structural reason that invalidated route (a) in part 2. The recommended
practice on this line is to score bed models on the **bed-layer decomposition
component**, never the total field, whenever the pass is above ~1 km AGL.

## 3.7 Deliverables and tests (part 3)

* `outputs/greenland_pair/full_pbed_proc_att14_rssnr/` -- `radargrams.png`
  (dotted measured-only pick overlay, corrected framing), `decomposition.png`,
  `decomposition_trace.png`, `bed_tail.png`, **`bed_brightness.png`** (the
  acceptance figure), `metrics.json`, `report.html`, `run_config.json`;
  mirrored to
  `outputs/verification/greenland_pair_full_pbed_proc_att14_rssnr/`.
* `outputs/greenland_pair/rssnr_anchor.npz` -- the pinned fetch (76 samples,
  snapshot `GEAMAHQ7BRVPG9SQPK20`) with provenance.
* Constant-gamma A = 14.0 run retained at
  `outputs/greenland_pair/full_pbed_proc_att14/` for the before/after table.
* New metric entries: `rssnr_level_anchor` (gate 2 dB, PASS at -1.70),
  `rssnr_gamma_mapping`, `bed_brightness_correlation`.
* Tests: 6 new in `tests/test_basal_lines.py` (Greenland store pinned to its
  own snapshot and cache, `gamma_rssnr` now wired, the contamination-aware
  deficit's sign and note, the Antarctic RSSNR config untouched, and the two
  companion-resolution guards). The companion-existence check runs in the
  early guard block so a missing companion fails in **milliseconds** rather
  than after 66 minutes of simulation.


## 3.8 Framing sweep: the profile figures had the same bug (and worse)

Fixing the radargram framing in part 2 caught only one of two families. The
**surface-referenced profile figures** carried the same Antarctic sizing, and
in their case it was not merely a crop:

`rel_mean_profile(..., lo_us=-1.5, hi_us=14.5)` is a **DATA window** -- it
builds the profile arrays over that extent. On this line the bed sits at
**~29 us** below the surface, so the bed returns were **never computed**, and
no axis change could have revealed them. Three figures consumed those arrays
and additionally clipped to `-1.0..13.5 us` / `-110..5 dB`:

| figure | was | now (Greenland) |
|---|---|---|
| `decomposition.png` | 0-13.5 us, bed absent | **-1..34 us, -140..5 dB** |
| `decomposition_trace.png` | 0-13.5 us, bed absent (subtitle said "bed at 29.05 us") | **-1..34 us, -140..5 dB** |
| `decomposition_zones.png` (Antarctic-only path) | unchanged | unchanged |

Two derived thresholds were tied to the old -110 dB floor and are now
expressed against `PROFILE_DB`: the surface-borne bed-source-invariance check
(`y0 > -105.0` -> `PROFILE_DB[0] + 5`) and the "median bed" text anchor.

New line globals, Antarctic values unchanged and asserted in tests:

| global | Antarctic | Greenland |
|---|---|---|
| `PROFILE_REL_US` (data) | (-1.5, 14.5) | **(-1.5, 34.5)** |
| `PROFILE_X_US` (plot x) | (-1.0, 13.5) | **(-1.0, 34.0)** |
| `PROFILE_DB` (plot y) | (-110.0, 5.0) | **(-140.0, 5.0)** |

`rel_mean_profile` now resolves its extent from the ACTIVE line at call time
(a regression test proves the default is not bound at import, which would
have silently kept the Antarctic extent after `activate_line`).

**Swept the remaining figures; two were never affected and need no change:**
`fig_bed_tail` plots on `TAIL_PROF_US`, which is **bed-referenced**
(-1..+4 us about the bed pick), so it is correct on any line by construction
-- which is why the bed-tail metrics were sound throughout. `fig_bed_brightness`
plots along-track `s` with an autoscaled y axis. `fig_radargrams` was fixed in
part 2.

### What the corrected figures show

With the bed finally in frame, `decomposition_trace.png` and
`decomposition.png` make the study's central result visible rather than
merely tabulated: on the LOW pass the simulated **bed returns** spike at
29 us to meet and slightly exceed the simulated surface returns (+1.4 dB
single-trace guard) and the sim total tracks the measured bed peak; on the
HIGH pass and syn14km the bed-return curve stays 15-20 dB **below** the
surface-return curve all the way through the bed window (-18.4 / -17.8 dB).
Across the whole column the measured curve sits far above the simulation on
every pass -- the missing englacial term, now visible over 34 us instead of
the first 13.5.


---

# Part 4 (2026-08-12): bed-visibility bisection -- and a correction to A

Triggered by the user's report that the bed is clearly visible in the OPR
product for `20170424_01_067` but not in our figures. Deliverable:
`outputs/greenland_pair/bed_visibility_bisect.png`.

## 4.1 Bisection verdict: NO stage loses the bed

Stage by stage on the high pass (each with its own robust colorscale, full
record, bed-pick overlay):

| stage | bed peak | bed window | floor | verdict |
|---|---|---|---|---|
| 1. raw `load_frame` (whole frame) | -78.14 | -81.62 | -86.87 | present |
| 2. sliced to s 11-40 | -80.54 | -83.95 | -88.67 | present, unchanged |
| 3. dt/t0 lattice handling | — | — | — | **no-op**: the measured array is never resampled (each pass keeps its own `tw`/`dt`); verified array-identical |
| 4. per-trace surface-peak normalization | -80.54 | -83.95 | -88.67 | **exactly unchanged** (it divides each trace by a scalar) |
| 5. final figure arrays | same | same | same | present in the data |

All dB rel own surface peak. **No stage degrades the bed.** Asset checks:
`data_product='CSARP_standard'`, `Data` (3 335 x 1 663), full record
0-55.400 us, no non-positive samples, surface peaks spread only 6.2 dB p5-p95
with **zero** traces at the frame maximum (no saturation, no mispick).

### The bed IS there -- quantified against a proper null

`bed peak` is `max(power over +-0.3 us) / floor mean`. For pure background
that statistic is **not** 0 dB: the max of ~18 exponential samples sits ~2.3 dB
above their mean. Measuring the same statistic at reflector-free delays gives
the null:

| pass | product | at bed | null (in floor window) | **excess over null** |
|---|---|---|---|---|
| low | standard | 16.56 | 2.27 | **+14.30 dB** |
| low | mvdr | 17.85 | 2.38 | **+15.48 dB** |
| high | standard | 8.47 | 2.32 | **+6.16 dB** |
| high | **mvdr** | 13.82 | 2.31 | **+11.51 dB** |

**The high pass's bed is a real +6.2 dB detection in the product we load, and
+11.5 dB in CSARP_mvdr.** The user is right that it is clearly visible -- the
bed-zoom panels show it as a crisp continuous line in `CSARP_mvdr` and as a
faint diffuse band in `CSARP_standard`.

## 4.2 Why it is invisible in our figures (a real, fixed, plotting bug)

The bed stands 6-16 dB above its local background while the shipped panel
ramp spans **125 dB** (`RADARGRAM_DB = (-120, 5)`): a 6 dB feature occupies
4.8 % of the gray range. Worse, a shared ramp **cannot** work here -- the two
passes' beds sit at -99 dB (low) and -81 dB (high) rel their own surface
peaks, 18 dB apart, each only 6-16 dB above its own background, so no single
linear ramp renders both.

Fix: new line global `RADARGRAM_SCALE` (`"shared"` = Antarctic, unchanged and
asserted; `"per_panel"` = Greenland) scaling each panel to its own robust
2-99.8 percentiles and printing the range in the panel title, so
comparability is preserved by annotation rather than by a shared ramp.

## 4.3 CORRECTION: the measured claims were product-limited, not physics

| claim (parts 2-3) | status | corrected |
|---|---|---|
| "bed only 4.7 dB above floor" | window-mean statistic, misleading as a visibility claim | bed **peak** is 8.1 dB over floor and **+6.2 dB over null** -- detected, not absent |
| "data ceiling 0.070 -> not a bed measurement above 2.5 km" | **too strong; product-limited** | ceiling is +0.078 (standard, window), +0.155 (standard, peak), **+0.470 (mvdr, window), +0.531 (mvdr, peak)** |
| acceptance high pass const +0.708 -> RSSNR +0.618 | still uninformative, reason now precise | scored against a standard-product profile whose ceiling is 0.08; on mvdr the ceiling is ~0.5, so the high pass should be re-scored on mvdr before any conclusion |

The **simulation-side** conclusions are untouched: the mid-column is
surface-borne by 84-124 dB, bed returns sit 15-20 dB below surface returns at
altitude, and the altitude trend overshoots by +15.7 dB. None of these depend
on measured bed visibility.

## 4.4 CORRECTION: A = 14.0 dB/km does not survive

Route (b) re-run identically on both products and with both bed measures
(floor-subtracted, 0.75 km bins):

| pass | product | measure | **A (Theil-Sen)** | r | resid |
|---|---|---|---|---|---|
| low | standard | window | 8.29 | -0.602 | 4.34 |
| low | standard | peak | 7.43 | -0.508 | 4.85 |
| low | mvdr | window | 7.93 | -0.640 | 3.85 |
| low | mvdr | peak | 7.06 | -0.549 | 4.22 |
| high | standard | window | **13.91** | -0.974 | 1.72 |
| high | standard | peak | **13.40** | -0.967 | 1.87 |
| high | **mvdr** | window | **6.18** | -0.902 | 1.53 |
| high | **mvdr** | peak | **6.66** | -0.833 | 2.25 |

* The **low pass gives 7.1-8.3 across all four combinations** -- a 1.2 dB/km
  spread. Product-stable and measure-stable, though statistically weak.
* The **high pass swings 13.9 -> 6.2 between products** (7.7 dB/km) while
  passing the statistical gate in BOTH. My part-2 gate tested statistical
  quality but **not product stability**, and admitted the contaminated one.
* On mvdr the high pass (6.2-6.7) **agrees with the low pass** (7.1-8.3).
  Two independent altitudes agreeing is the physically required outcome, and
  it only happens on the clutter-suppressed product.

**The committed A = 14.0 is an artifact of clutter contamination in the
CSARP_standard high-pass bed window.** The clutter's thickness dependence is
steeper than the bed's, faking a high attenuation; that is the same
contamination that gives that product a 0.078 correlation ceiling. The
consistent slope evidence is **A ~ 6.5-8 dB/km**.

**Unresolved tension, stated rather than papered over.** The SLOPE evidence
now says A ~ 7; the LEVEL evidence (part 3's level anchoring, implied
effective attenuation **14.9 dB/km**) still says ~15. They reconcile only if
the bed reflectivity is far dimmer than Fresnel ice->rock: holding the
measured level with A = 7 needs |Gamma|^2 about **30 dB below** the -12.86 dB
rock value, i.e. ~-43 dB. That is a dim, rough, frozen bed -- physically
possible but a different scene than the constant-Fresnel baseline assumes.
Resolving it needs the two constraints fitted jointly, not one anchored on
the other.

**Recommendation (a study decision, not taken here):** re-derive A with a
product-stability gate added, on CSARP_mvdr, and re-run the campaign at the
corrected value. The part-2/3 simulations remain valid as computed -- their
A is an input, and every level-dependent metric scales analytically at
2*dA*H = 4.94 dB per dB/km -- but the *conclusion* that A = 14 is the line's
attenuation should be withdrawn pending that re-derivation.

## 4.5 Also recommended: use CSARP_mvdr for bed-referenced work on this line

`CSARP_standard` is the right measured reference for surface and mid-column
comparisons (it is the product the img_comb/SAR chain we model produces). For
anything *bed-referenced* on the high pass it is the wrong choice: +5.4 dB
less bed, a 0.078 vs 0.470 correlation ceiling, and an attenuation slope
inflated by 7.7 dB/km. The tool loads the product via `load_frame(...,
data_product=...)`, so this is a one-argument change, but it would need its
own metric re-derivation and is left as a recorded follow-up.
