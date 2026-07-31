# Basal-clutter altitude triplet at the 2016 DC-8 anchor (2026-07-31)

Scouting only — no simulations. Goal: find every `2016_Antarctica_DC8` pass that
re-flies the anchor line `20161105_05_005/_006/_007`, characterise the altitude
spread, and pick a common segment for a **basal clutter** study.

Method: anchor nav from `soundersim.opr.load_frame`, projected to EPSG:3031 and
concatenated into one 148.5 km polyline (`s = 0` at `_005` trace 0);
`xopr.query_frames(collections=["2016_Antarctica_DC8"], geometry=<5 km buffer>)`;
every candidate frame downloaded and each trace projected onto the anchor
polyline (perpendicular distance `d`, along-track `s`). STAC frame geometries are
2-point LineStrings — useless for overlap length, so every number below comes
from downloaded nav. Params via `tools/run_altitude_comparison.mcords_params` /
`map_window` / `pick_oversample`. Scripts in the session scratchpad.

## Anchor line

* `20161105_05_005/_006/_007`, 2016-11-05 17:20:37–17:38:52 UTC, 9998 traces,
  **148.51 km**, 14.85 m/trace.
* Endpoints (lat, lon): (−74.58685, −117.56004) → (−74.76515, −122.59817).
  Amundsen Sea sector, Marie Byrd Land coast into the Getz-side shelf.
* **Low pass**: median AGL **442 m** (403–547), Elevation 484–846 m WGS84.
* The line is **grounded ice for s = 0 → 69.7 km** and **floating for
  s ≳ 90 km** (BedMachine mask, transitions at s = 69.7 / 72.3 / 74.8 / 87.4 /
  88.1 / 110.0 km). This constrains everything below — see quirk 1.

## The passes (three, all covering the full 148.5 km)

Two more passes exist exactly as the user reported, and **both re-fly the whole
anchor line end to end**. Verified from nav, not assumed:

| pass | frames spanning the anchor line | overlap with anchor | offset med / p90 / max (m) | **median AGL** | AGL range | n_samples / t0 | PRF | Tpd (µs) | img_comb |
|---|---|---|---|---|---|---|---|---|---|
| **low** `20161105_05` (anchor) | `_005 _006 _007` (+`_004`,`_008` off-line) | 0.00–148.51 km | 0 (reference) | **442 m** | 403–547 | 3367 / 0.000 µs | 12 000 | 1, 3, 10 | 3 img |
| **mid** `20161028_05` | `_007 _006 _005 _004` (`_003` off-line) | 0.00–148.45 km | 18.8 / 27.4 / 28.9 | **9 150 m** | 8 903–9 272 | 4024 / 37.374 µs | 7 500 | 3, 10 | 2 img |
| **high** `20161031_07` | `_005 _004 _003 _002` (`_006` off-line) | 0.00–148.45 km | 10.5 / 17.2 / 22.3 | **10 684 m** | 10 429–10 826 | 4024 / 37.374 µs | 7 500 | 3, 10 | 2 img |

Offsets are over the recommended common segment (s 18–68 km). All three carry
`CSARP_standard`, `CSARP_qlook`, `CSARP_mvdr`, `CSARP_layer`; bottom pick is
**100 % populated** on every frame.

The lateral offsets are near-constant (median ≈ mean, spread of a few m), i.e. a
fixed cross-track shift rather than track wander, and are **far inside one
Fresnel zone** (√(λ·h) = 130 m at 10.8 km, 118 m at 9.2 km). Treat the three as
co-located.

**Both high passes fly the line in the OPPOSITE direction to the anchor**
(decreasing anchor `s` with increasing trace index; frame numbers *increase* as
`s` decreases). Their slow-time slices must be reversed to align with anchor `s`,
and the flight heading is reversed — which flips the sign of any roll-dependent
or fore/aft-asymmetric term.

Also inside a 15 km buffer but **not** repeats: `20161020_09_014–018` (low,
AGL 497 m), `20161028_04_003–006` (high, AGL 8 766 m), `20161102_05_005–009`
(high, AGL 8 756 m). All three run at heading 142.3° — identical to the anchor —
at a **constant 10.0 km lateral offset**: these are the adjacent lines of a 10 km
grid survey, not repeat passes. Useful as independent bed control, not as
altitude repeats.

## Recommended common segment

**Anchor along-track s = 18.0 → 68.0 km (50.0 km)**, anchor traces 1212 → 4577.

* Endpoints (anchor nav): (−74.61470, −118.16400) → (−74.68354, −119.85180).
* Chosen to stay **100 % grounded** (ends 1.7 km short of the first grounding-line
  crossing at s = 69.7 km) while sitting in the high-clutter, high-relief part of
  the line. Mean clutter contrast +24.8 dB (see below); ~all 50 km windows inside
  s ∈ [0, 70] score within 0.6 dB of each other, so the grounding line, not the
  contrast, is the binding constraint.
* Slices (`slow_time` index, half-open) into each full frame:
  * low `20161105_05_005`: `slice(1212, 3333)` — 2121 traces (s 18.00→49.49)
  * low `20161105_05_006`: `slice(0, 1244)` — 1244 traces (s 49.52→67.99)
  * mid `20161028_05_006`: `slice(0, 2341)` — 2341 traces (s 52.72→18.00, **reversed**)
  * mid `20161028_05_005`: `slice(2308, 3337)` — 1029 traces (s 68.00→52.75, **reversed**)
  * high `20161031_07_005`: `slice(0, 1820)` — 1820 traces (s 44.99→18.01, **reversed**)
  * high `20161031_07_004`: `slice(1786, 3336)` — 1550 traces (s 67.99→45.02, **reversed**)
* 3365 / 3370 / 3370 traces per pass — trace counts match to 0.15 %.
* Anchor frame `_007` is **not needed** for this segment (it covers the floating
  s = 99–148.5 km part).

### PILOT sub-segment: s = 30.0 → 40.0 km (10.0 km)

Anchor traces 2020 → 2693; endpoints (−74.63237, −118.56778) →
(−74.64657, −118.90471). **One frame per pass, no frame boundary:**

* low `20161105_05_005`: `slice(2020, 2693)` — 673 traces
* mid `20161028_05_006`: `slice(858, 1532)` — 674 traces (reversed)
* high `20161031_07_005`: `slice(337, 1011)` — 674 traces (reversed)

**Why this window** (eyeballed on the measured high-pass radargrams, and
quantified): over s 30–40 km the low pass shows a sharp bed climbing from 10.5 µs
to 6 µs below the surface with rugged relief, and the two high passes show a
*dense field of overlapping hyperbolic arcs* filling the ice column from ~2 µs
below the surface down past the bed — structured, resolvable off-nadir clutter,
including one wide bright hyperbola from the deep trough at s ≈ 31 km and a
surface-bump diffraction at s ≈ 38 km. Per-km bed relief here is the highest on
the grounded part of the line (mean 103 m/km, max 211 m/km).

Two windows scored *higher* on the raw contrast metric and were **rejected**:

* **s 5–15 km (+29.8 dB, the top score) — rejected.** The high-pass radargram
  there is a nearly featureless uniform haze: the contrast is diffuse
  volume/incoherent fill, not structured off-nadir clutter, and the bed (1030 m
  thick, only 47 m/km relief) is barely visible. A bad target for validating
  geometric clutter.
* **s 46–56 km (+28.8 dB) — good structured clutter and the largest single
  relief (213 m/km), but it straddles a frame boundary in all three passes
  (6 frames instead of 3).** Keep it as the second pilot if s 30–40 km proves too
  easy.

### Clutter-contrast metric (how the windows were ranked)

Per trace: mean power in the mid-ice-column window (3.0 → 0.6 µs *above* the
bottom pick) divided by the peak power within ±0.3 µs of the pick, in dB;
median per 1 km bin. Whole-line medians:

| pass | mid-column / bed-peak |
|---|---|
| low 442 m | **−36.7 dB** |
| mid 9 150 m | **−17.7 dB** |
| high 10 684 m | **−16.1 dB** |

**~20 dB more mid-column energy at altitude** — this is the basal/surface clutter
the study is about, and it is unambiguous. Contrast (high − low) is +20…+35 dB
for s < 80 km and falls to +5…+20 dB over the floating part. On the grounded
rough section (s 10–55 km) the high passes' mid-column power sits within a few dB
of the bed peak itself, i.e. clutter-limited, not noise-limited. The 9.2 km and
10.8 km passes are nearly indistinguishable on this metric (1.6 dB apart) —
expect the altitude *trend* to be carried mainly by low-vs-high, with the two
high passes serving as a repeatability check.

## Radar / processing parameters

| | low `20161105_05` | mid `20161028_05` | high `20161031_07` |
|---|---|---|---|
| f0–f1 (MHz) | 165–215 | 165–215 | 165–215 |
| fc / B (MHz) | 190 / 50 | 190 / 50 | 190 / 50 |
| Tpd waveforms (µs) | **1, 3, 10** | **3, 10** | **3, 10** |
| bed waveform (µs) | 10 | 10 | 10 |
| tukey time window | 0.2 | 0.2 | 0.2 |
| `ft_wind` (verified by hand) | `hanning` | `hanning` | `hanning` |
| mapped soundersim window | `hann` | `hann` | `hann` |
| PRF (Hz) | **12 000** | **7 500** | **7 500** |
| product dt (ns) | 20.2020 | 20.2020 | 20.2020 |
| n_samples / t0 (µs) / t_end (µs) | 3367 / 0.000 / 68.000 | 4024 / 37.374 / 118.646 | 4024 / 37.374 / 118.646 |
| `pick_oversample` k (f_alias) | 5 (57.5 MHz) | 5 (57.5 MHz) | 5 (57.5 MHz) |
| SAR σ_x (m) / sar_type / start_eps | 2.5 / f-k / 3.15 | 2.5 / f-k / 3.15 | 2.5 / f-k / 3.15 |
| `img_comb` (µs) | [3, −∞, 1 ; 10, −∞, 3] | [10, −∞, 3] | [10, −∞, 3] |
| `param_combine.combine.method` | `standard` | `standard` | `standard` |
| \|Roll\| p95 / max (deg) | 1.03 / 1.68 | 0.97 / 1.32 | 0.55 / 0.70 |
| products | std, qlook, mvdr, layer | same | same |

The fast-time lattice is **identical across all three passes** (20.2020 ns,
k = 5) — a genuine like-for-like altitude trade, unlike the 2012-anchor
cross-season case where bandwidth and dt both changed. Only the geometry, the
window origin, the PRF and the waveform set differ.

Twtt headroom is generous everywhere: the high-pass window runs to 118.6 µs while
the deepest bed return lands near 82.6 µs, leaving ~36 µs of recorded post-bed
clutter tail. The low pass records to 68 µs with the bed at ≤14.2 µs.

### Surface-pick vs REMA registration (the cross-pass pitfall)

`Elevation − c·Surface/2` minus REMA v2.0 32 m, both WGS84-ellipsoidal (no geoid
term), over the recommended segment:

| pass | bias (m) | σ (m) | p5..p95 (m) |
|---|---|---|---|
| low `20161105_05` | **+13.2** | **2.45** | +10.6 … +18.3 |
| mid `20161028_05` | **+17.1** | **10.80** | +11.9 … +44.0 |
| high `20161031_07` | **+17.7** | **10.92** | +11.9 … +41.6 |

Over the 10 km pilot the high passes degrade further (bias +22 m, σ 16.3 m,
p5..p95 spanning −11 → +55 m). One range bin is 3.03 m, so the high-altitude
surface pick is ~3.6 bins noisy and the low-vs-high bias disagreement is ~1.5
bins. **Fit the twtt offset per frame** (as `leading_edge_gate` already does);
never carry one pass's registration to another.

The same disagreement shows in the picked ice thickness over the identical
segment: **683 m (low) vs 644 m (mid) vs 643 m (high)** — a 6 % systematic
difference in the *measured* thickness of the same ice, from the same processing
chain, purely from altitude. Do not treat any of them as truth.

## BED context (this is a basal-clutter study)

BedMachine Antarctica v3 (NSIDC-0756), **native posting 500 m**, EPSG:3031, bed
converted to WGS84-ellipsoidal (bed + EIGEN-6C4 geoid) by
`opr.fetch_bedmachine_window`.

Over the recommended 50 km segment (s 18–68 km):

| quantity | value |
|---|---|
| BedMachine bed, nadir | −812 … −351 m (median −506), peak-to-peak **461 m** |
| BedMachine bed, ±6 km cross-track window | −1268 … −230 m, peak-to-peak **1038 m** |
| REMA surface, nadir | 41 … 390 m (median 127) |
| radar ice thickness (anchor picks, ε=3.15) | median **683 m**, p5–p95 528–858, min–max 452–896 |
| radar bed elevation | median −480 m, min–max −806 … −208 |
| BedMachine − radar bed | median **−30 m**, MAD 57 m, p5..p95 −171 … +100 m |
| nadir bed twtt below surface (clutter delay window) | median **8.09 µs**, range 5.35–10.61 µs |
| bed slope at 500 m posting | median 2.43°, p90 5.96°, max 20.0° |
| along-track bed roughness (residual about a 5 km mean) | **33.3 m rms** from BedMachine vs **60.5 m rms** from the radar picks |

Pilot window (s 30–40 km): bed −691 … −404 m nadir (p-p 286 m), −1073 … −239 m
over ±6 km; thickness median 669 m (468–894); bed twtt below surface median
7.92 µs (5.54–10.58); BedMachine − radar bed median −3.9 m, MAD 74.5 m.

**Honest limits of the 500 m bed.** At the ~50 m facet spacing the high passes
want (below), the bed DEM is being bilinearly upsampled 10×: everything finer
than 500 m in the simulated bed is interpolation, not data. The measured
consequence is in the table above — BedMachine reproduces only **55 %** of the
along-track bed roughness rms that the radar picks show (33 m vs 61 m), and its
nadir bed disagrees with the radar bed by ±100–170 m at the 5–95 % level.
Expect the simulated basal clutter to be **systematically smoother and weaker in
fine texture** than the measured cluttergrams; the large-scale hyperbolic arcs
from the 1–5 km-scale troughs should reproduce, the speckle-scale texture will
not. Do not tune roughness parameters to close that gap without saying so.

## Data quirks that will bite the simulation stage

1. **The line goes afloat at s ≈ 70 km, and BedMachine's "bed" there is the
   SEAFLOOR, not the reflector the radar sees.** Beyond the grounding line the
   basal echo is the ice/ocean interface; BedMachine `bed` is the bathymetry
   under the cavity, and its own ice thickness is a hydrostatic inversion (515–
   580 m vs the radar's 577–686 m over s 90–140 km). Simulating "bed" clutter
   there would model the wrong surface entirely. **The recommended segment stops
   at s = 68 km for exactly this reason** — if anyone extends the study past
   s ≈ 70 km, the layer stack must change (ice → seawater, ε ≈ 80, and the
   ice-base geometry taken from surface × hydrostatic ratio, not from `bed`).
2. **The two high passes fly the line backwards** relative to the anchor. Slices
   must be reversed; heading, roll sign convention and any fore/aft asymmetry
   flip with them.
3. **Cross-track reach scales ~3.3× with altitude.** For surface clutter to reach
   the nadir-bed delay (720 m ice → 8.53 µs) requires ±1.66 km at 442 m AGL but
   ±5.02 km at 9.2 km and ±5.40 km at 10.8 km (±6.5 km for the 1020 m-thick
   spots). Size `ct_dist` per pass; the default 6 km cap in
   `run_altitude_comparison` is *just* adequate for the high passes and much more
   than needed for the low one.
4. **…but the high passes are CHEAPER, not more expensive.** β=0.5 Fresnel facet
   spacing (fc 190 MHz, 720 m ice) is 10.67 m at 442 m AGL, 47.05 m at 9.2 km and
   50.58 m at 10.8 km. The ~4.7× coarser facets more than pay for the 3.3× wider
   swath. The catch is item 3 of the bed section: at 50 m facets the 500 m bed is
   pure interpolation.
5. **Waveform set differs.** The low pass composites 1/3/10 µs
   (`img_comb = [3, −∞, 1 ; 10, −∞, 3]`); both high passes composite only 3/10 µs
   (`[10, −∞, 3]`). The repo convention simulates the longest (10 µs / bed)
   waveform only, which is right for the bed zone in all three, but the *measured*
   shallow product is built differently between low and high — do not compare the
   near-surface few µs across passes as if it were one instrument.
6. **PRF differs 12 000 vs 7 500 Hz**, yet the along-track posting is identical
   (14.83–14.86 m/trace) because the products are decimated to a fixed spacing.
   Doppler/SAR aperture bookkeeping differs even though the output grid does not.
7. **`ft_wind` decode fails in `mcords_params` for all three passes** (the known
   2016 quirk): `param_records.radar` exists so `_PARAM_LAYOUTS` selects the
   `param_sar` entry, then finds no `param_sar` struct and falls back to the
   readme default string `"hanning (ft_wind decode failed; CReSIS readme
   default)"`. **Verified by hand: the true value is
   `param_csarp.csarp.ft_wind` = MATLAB `@hanning` in all three**, so the
   fallback is correct — but the provenance string is misleading and must not be
   quoted as measured.
8. **`param_csarp.combine.method` reads `"mvdr"` in all three frames while
   `param_combine.combine.method` reads `"standard"`.** This resolves the
   ambiguity flagged in `cross_season_line_scout.md` item 8: the `param_csarp`
   copy is a stale struct overwritten by the later CSARP_mvdr run; the
   `param_combine` struct is the one describing the CSARP_standard product. Read
   `param_combine`.
9. **Registration disagrees between passes by ~4.5 m in bias and 4× in scatter,
   and the picked ice thickness by 6 %** (see the registration table). Per-frame
   twtt alignment is mandatory.
10. **The dt grid is 20.2020 ns, not 20.000 ns** on all three (the 2016
    fs/ft_dec bookkeeping), consistent with the earlier 2016 finding. All three
    agree, so this is not a cross-pass hazard here — only a cross-season one.
11. Roll is small everywhere (p95 ≤ 1.03°, max 1.68°), so `roll_source="nav"`
    matters less than it did for the 2012 anchor.
12. The anchor's own frames `_004` and `_008`, and the far-end frames
    `20161028_05_003`, `20161031_07_006`, sit at the polyline ends and contribute
    no usable overlap — do not include them.
