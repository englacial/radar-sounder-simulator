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
