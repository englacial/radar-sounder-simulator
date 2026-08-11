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
