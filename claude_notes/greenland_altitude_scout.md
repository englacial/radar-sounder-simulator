# Greenland altitude pair at the 2014 P-3 anchor (2026-08-11)

Scouting only — no simulations. Goal: find OPR passes that overlap the ground
track of `20140421_01_069/_070` at a **different platform altitude**.

Method (same as `basal_clutter_scout.md` / `cross_season_line_scout.md`): anchor
nav from `soundersim.opr.load_frame`, projected to **EPSG:3413** and concatenated
into one 99.7 km polyline (`s = 0` at `_069` trace 0);
`xopr.query_frames(collections=[<season>], geometry=<buffered track>)` run over
**all 26 Greenland collections in the OPR catalog** at a 5 km buffer (54 frames)
and again at 25 km (105 frames); every candidate's nav downloaded and each trace
projected onto the anchor polyline (perpendicular distance `d`, along-track `s`).
STAC frame geometries are 2-point LineStrings and are useless for overlap length
— every number below comes from downloaded nav. Nav came from the small
`CSARP_layer` files via `xopr.load_layers_file` (lat/lon/elev + surface/bottom
twtt); pre-2005 seasons have no `CSARP_layer` and fell back to
`load_frame_url(CSARP_standard)`. Params via
`tools/run_altitude_comparison.mcords_params` / `map_window` /
`pick_oversample`. Scripts in the session scratchpad.

## Anchor line

* **Season string verified: `2014_Greenland_P3`** (frames `20140421_01_069`,
  `20140421_01_070`; `opr:frame` 69/70 of segment `20140421_01`).
* 2014-04-21 **17:45:10–17:57:41 UTC**, 6664 traces, **99.72 km**, 14.97 m/trace.
* Endpoints (lat, lon): (70.88774, −44.77977) → (70.28829, −46.53560); extent
  70.288–70.888 N, 46.536–44.780 W. Heading ~225° (SW).
* **Central-west Greenland interior**, on the western flank of the main divide
  roughly midway between Summit and the west coast, upstream of the
  Jakobshavn/central-west drainage. Surface 2156–2578 m (WGS84), ice
  **2.0–2.6 km** thick, bed near sea level.
* **Low pass**: median AGL **475 m** (`_069`) / **495 m** (`_070`), range
  430–525 m; Elevation 2641–3051 m WGS84.
* Bottom pick **100 % populated** on both frames; thickness median 2475 m
  (`_069`) / 2122 m (`_070`) at ε = 3.15.

## The altitude search — what exists

| pass | frames on the line | overlap w/ anchor | offset med / p90 / max | **median AGL** | verdict |
|---|---|---|---|---|---|
| **anchor** `2014_Greenland_P3 / 20140421_01` | `_069 _070` | 0.00–99.72 km | 0 (reference) | **465–503 m** | reference |
| **HIGH** `2017_Greenland_P3 / 20170424_01` | `_066 _067 _068` | 0.0–46.1 and 74.2–99.7 km within 300 m | **14 / 38 / 45 m** (s 11–40) | **2 483 m** (2 413–2 538) | **the one genuine altitude repeat** |

**That is the whole list.** Everything else that touches the line is at the
standard IceBridge survey altitude:

* 15 seasons / 54 frames within 5 km; 22 seasons / 105 frames within 25 km.
  Median AGL of every one of them is **450–560 m** except the 2017 pass.
* The only other >600 m AGL frames anywhere within 25 km are
  `1996_Greenland_P3 19960520_01_012–014` (~1 000 m AGL) and a handful of 1993 /
  2001 / 2006_TO / 2008_TO frames at 600–740 m — **all ≥ 9 km, mostly ≥ 19 km,
  from the track**. No usable overlap.
* Six low passes look like long overlaps in a naive `d < 300 m` test
  (`20110422_03_001`, `20120429_01_020`, `20150423_06_022`, `20140414_02_046`,
  `20190507_01_066`, `20110422_02_015`, 15–32 km "overlap"). **All six are
  shallow-angle crossings, not repeats** — their `d` oscillates 400–1600 m across
  the segment and the perpendicular-distance distribution is uniform (median
  ≈ half the threshold, p90 ≈ 0.9× the threshold), the signature of a line
  crossing the anchor rather than following it. Do not use them as repeats; they
  are fine as independent bed control.
* `2010_Greenland_DC8`, `2016_Greenland_Polar6`, `2016_Greenland_TOdtu`,
  `2016_Greenland_G1XB`, `2005_Greenland_TO`, `2009_Greenland_TO` return
  **zero** frames even at 25 km.

**Answer to the framing question: yes, there is a genuine multi-altitude
opportunity here, but it is a PAIR, not a triplet.** 465 m vs 2 483 m AGL, a
**5.3× altitude ratio**. It is weaker than the Antarctic 442/9 150/10 684 m
triplet (23×) but it is real, well-registered, and — unusually — carries the
**identical radar configuration** (see params). Greenland high-altitude coverage
is as sparse as feared: one transit leg, found once in 26 seasons.

### Why the 2017 pass is high

`20170424_01` is a single long sortie: high-altitude transit out (frames 001–010,
~2.0–2.2 km AGL), a low-altitude survey in East Greenland (frames 020–060,
~485–510 m AGL), then a **high-altitude transit home** — frames ~066 onward at
2.2 → 3.2 km AGL, climbing steadily. The anchor line is overflown on that return
transit. AGL therefore **drifts within the overlap** (2 413 → 2 538 m over the
recommended segment; 2 223 / 2 523 / 2 723 / 2 943 / 3 173 m at frames
066/067/068/069/070). It is not a constant-altitude survey line.

Both flights are **late April, ~17:45–18:00 UTC, three years apart** — nearly
identical season and time of day, so firn state and surface conditions are about
as comparable as a 3-year separation allows.

Both passes fly the line in the **same direction** (anchor `s` increases with
trace index). No slice reversal needed.

### Where the 2017 pass departs the line

Offset vs anchor `s` (median per 2 km bin, from nav):

* s 2 → 44 km: **≤ 45 m** (mostly 2–40 m) — frames `_066` (s 0–10.45) and
  `_067` (s 10.46–61.0).
* s 44 → 74 km: bulges out to a **1.3 km maximum at s ≈ 55 km** (the transit
  takes a slightly different great circle). Unusable.
* s 74 → 98 km: back to **10–116 m** (frame `_068`).
* Roll: the 2017 aircraft **starts banking at s ≈ 41 km** (|roll| 0.3° → 9.8° by
  s = 44). The main segment must stop at s = 40 km, not 44.

## Recommended common segment

**Anchor along-track s = 11.0 → 40.0 km (29.0 km)** — chosen so that it is
**one frame per pass with no frame boundary**, offsets stay ≤ 45 m, and both
aircraft are wings-level.

* Endpoints (anchor nav): (70.84793, −45.05577) → (70.74061, −45.77654).
* Slices (`slow_time` index, half-open) into each full frame:
  * low `2014_Greenland_P3 / 20140421_01_069`: `slice(736, 2675)` — **1939** traces
  * high `2017_Greenland_P3 / 20170424_01_067`: `slice(36, 1976)` — **1940** traces
* Trace counts match to **1 trace (0.05 %)**; both at 14.95 m/trace.

| | low `_069` s 11–40 | high `_067` s 11–40 |
|---|---|---|
| median AGL | **465 m** (430–520) | **2 483 m** (2 413–2 538) |
| offset from anchor med / p90 / max | 0 | **14 / 38 / 45 m** |
| \|roll\| med / p95 / max | 0.50 / 1.37 / 1.96° | 0.67 / 2.03 / 2.40° |
| picked ice thickness (ε 3.15) | median **2 476 m** (p5–p95 2 276–2 614) | median **2 483 m** (2 278–2 607) |
| bed twtt below surface | 26.12–31.22 µs (med 29.31) | 26.32–31.11 µs (med 29.40) |
| surface twtt | 2.87–3.47 µs | 16.10–17.00 µs |
| bed twtt (absolute) | 28.26–34.19 µs | 42.33–47.53 µs |
| **post-bed recorded tail** | **≥ 21.1 µs** | **≥ 7.9 µs** |
| pre-surface record | 3.0 µs | 16.1 µs |

The picked thickness agrees to **7 m (0.3 %)** between the two altitudes — a much
cleaner like-for-like than the Antarctic triplet's 6 % disagreement.

### PILOT sub-segment: s = 25.0 → 35.0 km (10.0 km)

Anchor traces 1672 → 2341; endpoints (70.79650, −45.40490) → (70.75906, −45.65254).

* low `20140421_01_069`: `slice(1672, 2341)` — 669 traces, AGL med **446 m**
* high `20170424_01_067`: `slice(973, 1641)` — 668 traces, AGL med **2 488 m**
* Offset med / p90 / max **10 / 21 / 22 m** — the tightest stretch on the line;
  |roll| ≤ 1.53° (low) / 2.40° (high); thickness 2 473 vs 2 474 m.
* Bed here is smooth (nadir p-p 140 m over 10 km, mean 20 m relief/km): a good
  *first* test but a weak basal-clutter target — see the bed caveat below.

### Second segment (optional, s = 79.0 → 97.0 km, 18.0 km)

* low `20140421_01_070`: `slice(1948, 3150)` — 1202 traces, AGL med **503 m**
* high `20170424_01_068`: `slice(1205, 2407)` — 1202 traces, AGL med **2 732 m**
* Offset med / p90 / max 30 / 92 / 108 m — 1.7× the high pass's first Fresnel
  radius, so treat as "near-repeat", not co-located. Thinner ice (2 149 vs
  2 135 m), more post-bed tail on the high pass (10.7 µs). Useful as an
  independent second altitude ratio (5.4×) with a slightly different geometry.

## Radar / processing parameters

Read from each frame's own param structs through xopr.

| | **low** `2014_Greenland_P3 20140421_01_069/_070` | **high** `2017_Greenland_P3 20170424_01_066/_067/_068` |
|---|---|---|
| f0–f1 | 180–210 MHz | 180–210 MHz |
| fc / B | **195 MHz / 30 MHz** | **195 MHz / 30 MHz** |
| Tpd waveforms | **1, 3, 10 µs** | **1, 3, 10 µs** |
| bed waveform | 10 µs | 10 µs |
| tukey **time** window | **0.1** | **0.2** |
| `ft_wind` (verified by hand) | `@hanning` (`param_csarp.csarp.ft_wind`, scipy `mat_struct`, MATLAB 2012a) | `@hanning` (decoded `function_handle`, MATLAB 2015a) |
| mapped soundersim window | `hann` | `hann` |
| PRF | 12 000 Hz | 12 000 Hz |
| **product dt** | **33.3859 ns** | **33.3333 ns** |
| n_samples / t0 / t_end | 1661 / **−0.1669 µs** / 55.2537 µs | 1663 / **0.0000 µs** / 55.4000 µs |
| range bin (air) | 5.006 m | 5.000 m |
| `pick_oversample` k (f_alias) | **4** (44.62 MHz) | **4** (45.00 MHz) |
| SAR σ_x / sar_type / start_eps | 2.5 m / f-k / 3.15 | 2.5 m / f-k / 3.15 |
| `param_combine.combine.img_comb` | **[3 µs, −∞, 2.64 µs; 10 µs, −∞, 3.5 µs]** | **[3 µs, −∞, 1 µs; 10 µs, −∞, 3 µs]** |
| `param_combine.combine.method` | `standard` | `standard` |
| `param_csarp.combine.method` | `standard` | **`mvdr`** (stale struct — see quirk 6) |
| along-track posting | 14.94–14.99 m | 14.95–14.97 m |
| products | std, qlook, mvdr, layer | std, qlook, mvdr, layer |
| qlook grid | 250 traces, 1661 samples, 33.3859 ns | 252 traces, 1663 samples, 33.3333 ns |
| bottom pick coverage | **100 %** | **100 %** |

**This is the cleanest instrument parity of any pair scouted so far**: same
centre frequency, same 30 MHz bandwidth, same three-waveform set, same PRF, same
SAR parameters, same window family, k = 4 on both. Only the geometry, the
window origin, the fast-time bin (0.16 %), the transmit tukey and the `img_comb`
transitions differ.

### Surface-pick vs ArcticDEM registration (the cross-pass pitfall)

`Elevation − c·Surface/2` minus ArcticDEM v4.1 32 m mosaic, both WGS84-ellipsoidal
(no geoid term), over the recommended segment:

| pass | bias (m) | σ (m) | p5..p95 (m) |
|---|---|---|---|
| low `20140421_01_069` (s 11–40) | **−1.5** | **1.23** | −3.5 … +0.5 |
| high `20170424_01_067` (s 11–40) | **−26.8** | **1.47** | −29.4 … −24.4 |
| low `20140421_01_070` (s 79–97) | −5.3 | 1.95 | −8.8 … −2.2 |
| high `20170424_01_068` (s 79–97) | −30.1 | 3.28 | −35.4 … −25.8 |

**The two passes disagree with each other by ~25 m — five range bins — while
their picked ice thicknesses agree to 7 m.** The offset is therefore
*common-mode* in the twtt/elevation reference of the 2017 high-altitude frames
(a constant shift of Surface and Bottom together, or of Elevation), not a
surface-tracker error. The scatter is small and stable (σ 1.2–1.5 m, ~0.3 bin),
so a single fitted constant per frame removes it cleanly — but **it must be
fitted per frame** (`leading_edge_gate` already does this). Never carry the
anchor's registration to the 2017 frames. Real interior-Greenland elevation
change over 3 years is ~0.1–0.3 m; 25 m is instrumental.

## Bed context over the segment

BedMachine **Greenland v5 (IDBMG4)**, native posting **150 m**, EPSG:3413, bed
converted to WGS84-ellipsoidal (bed + EIGEN-6C4 geoid) by
`opr.fetch_bedmachine_window`. Surface from ArcticDEM v4.1 32 m.

| quantity | main s 11–40 km | pilot s 25–35 km | tail s 79–97 km |
|---|---|---|---|
| BedMachine bed, nadir | −104 … +166 m (p-p **270 m**) | −99 … +41 m (p-p 140 m) | +27 … +149 m (p-p 122 m) |
| BedMachine bed, ±6 km cross-track | −104 … +166 m (p-p 270 m) | −103 … +128 m (p-p 231 m) | −21 … +226 m (p-p 247 m) |
| ArcticDEM surface, nadir | 2 411 … 2 534 m | 2 411 … 2 469 m | 2 186 … 2 256 m |
| radar ice thickness (ε 3.15) | med **2 476 m**, p5–p95 2 276–2 614 | med 2 473 m | med 2 149 m |
| radar bed elevation | med −26 m (low) / −57 m (high) | −39 / −62 m | +65 / +50 m |
| BedMachine − radar bed | +13 m (low) / +37 m (high), MAD 21–25 m | +15 / +29 m, MAD 17–21 m | +20 / +30 m, MAD 28–30 m |
| nadir bed twtt below surface | med **29.3 µs**, 26.1–31.2 | med 29.3 µs | med 25.4 µs |
| bed slope at 150 m posting | med 1.64°, p90 3.03°, max 6.07° | med 1.48°, max 3.09° | med 1.04°, max 4.39° |
| along-track bed roughness (rms about a 5 km mean) | **18.4 m** BedMachine vs **39.8 m** radar picks | 13.7 vs 24.0 m | 10.1 vs 43.9 m |

**This is a FLAT, DEEP interior bed** — 270 m of relief over 29 km, mean 27 m of
relief per km, slopes ~1.6°. Compare the Antarctic basal-clutter anchor: 461 m
p-p over 50 km with 103 m/km in the pilot window and 20° max slopes. Whatever
altitude-dependent clutter shows up here will be **surface- and volume-driven far
more than bed-driven**. The BedMachine−radar bed offset (+13 to +37 m, and it
differs between the two passes by exactly the 25 m registration bias above) is
mostly that registration bias, not a bed error.

BedMachine v5 reproduces only **46 %** (main) / **23 %** (tail) of the
along-track bed roughness rms the radar picks show. At the ~11–16 m facet
spacing these altitudes want, the 150 m bed is being upsampled 10×: everything
finer than 150 m in a simulated bed is interpolation.

### Measured altitude payoff (mid-ice-column clutter contrast)

The `basal_clutter_scout.md` metric — per trace, mean power in the mid-column
window (3.0 → 0.6 µs *above* the bottom pick) over the peak power within ±0.3 µs
of the pick, median over traces:

| segment | low | high | **contrast** |
|---|---|---|---|
| main s 11–40 | −17.45 dB | **−1.54 dB** | **+15.9 dB** |
| pilot s 25–35 | −18.45 dB | **−1.67 dB** | **+16.8 dB** |
| tail s 79–97 | −20.71 dB | **−1.07 dB** | **+19.6 dB** |

**~16–20 dB more mid-column energy at 2.5 km than at 0.47 km**, and at altitude
the mid-column power sits **within 1.5 dB of the bed peak itself** — the high
pass is clutter-limited over the bed here. The effect is unambiguous and about
as large as the Antarctic 442 → 10 684 m case (~20 dB), from only a 5.3×
altitude ratio, because the ice is 3.5× thicker (the clutter has 29 µs of ice
column to fill). This is the strongest single argument for the pair.

Do **not** read the shallow band (3–6 µs below the surface) as clutter: it sits
at +39 dB (low) / +34 dB (high) relative to the bed peak simply because internal
layers at 250–500 m depth are far brighter than a 2.5 km-deep, heavily
attenuated bed.

### Geometry sizing

λ = 1.538 m at fc 195 MHz; ε_ice 3.15; thickness 2 476 m.

| | low (h = 472 m) | high (h = 2 488 m) |
|---|---|---|
| first Fresnel radius √(λh) | **26.9 m** | **61.8 m** |
| β = 0.5 facet spacing (bed binds at low, surface at high) | 13.5 m → snapped **10.67 m** | 30.9 m → snapped **16.0 m** |
| cross-track reach for surface clutter to reach the nadir-bed delay | **±4.82 km** | **±6.39 km** |

Note the consequences: the facets are only **1.5× coarser** at altitude (vs 4.7×
in the Antarctic case) and the cross-track reach is **1.33× wider**, so the high
run is *not* cheaper here — expect roughly comparable or slightly higher cost
than the low run. The default `--ct-cap 6000` in `run_altitude_comparison`
**clips the high pass** (needs 6.39 km); raise it to ~7 km.

## Data quirks that will bite the simulation stage

1. **Only ONE alternative altitude exists.** There is no mid-altitude pass, no
   third point, and no second high pass to serve as a repeatability check. Any
   altitude *trend* here is a two-point line. Report it as such.
2. **The high pass's recorded window barely clears the bed.** Its post-bed tail
   is **7.9 µs** on the main segment (10.7 µs on the tail segment) versus
   **21.1 µs** for the anchor. A study of the post-bed clutter tail is
   ~2.7× shorter at altitude, and any window margin (`POST_BED_US = 2.0`) plus
   the 10 µs bed chirp eats most of it. Meanwhile the high pass wastes 16.1 µs of
   record *before* the surface (t0 = 0). Check the twtt window per level before
   assuming a common surface-referenced axis.
3. **The 2017 surface pick is offset ~25 m (5 range bins) from the anchor's**,
   while the ice thickness agrees to 7 m. Fit the twtt offset per frame; never
   assume a common registration. (σ is only ~1.4 m, so the fit is easy — but it
   is mandatory.)
4. **dt differs by 0.16 %** (33.3859 vs 33.3333 ns) and **t0 differs by 167 ns
   (5 bins)**. Over the 55 µs window the grids drift 2.6 bins apart. Express any
   depth binning in metres, not bins, and resample rather than index-align.
   `pick_oversample` gives k = 4 on both, so the *simulation* lattices are
   compatible in structure but not identical in value.
5. **`img_comb` differs**: low = `[3 µs, −∞, 2.64 µs; 10 µs, −∞, 3.5 µs]`, high =
   `[3 µs, −∞, 1 µs; 10 µs, −∞, 3 µs]`. Both composite 1/3/10 µs, but the blend
   lengths differ, so the *measured* shallow product is built differently between
   the two passes. The repo convention (simulate the 10 µs / bed waveform only)
   is right for the bed zone in both; do not compare the near-surface few µs
   across passes as if it were one instrument. Also the transmit tukey time
   window differs (0.1 vs 0.2).
6. **`ft_wind` decode fails in `mcords_params` for BOTH passes** — the known
   quirk. `param_records.radar` exists so `_PARAM_LAYOUTS` selects the
   `param_sar` entry, then finds no `param_sar` struct and falls back to the
   readme default string `"hanning (ft_wind decode failed; CReSIS readme
   default)"`. **Verified by hand: the true value is `@hanning` in both**
   (`param_csarp.csarp.ft_wind`; 2014 arrives as an undecoded scipy `mat_struct`
   pointing at MATLAB 2012a's `hanning.m`, 2017 as a decoded `function_handle`
   from MATLAB 2015a), so the fallback is correct — but the provenance string is
   misleading and must not be quoted as measured. Separately,
   `param_csarp.combine.method` reads `"mvdr"` in all three 2017 frames while
   `param_combine.combine.method` reads `"standard"` (the stale-overwrite pattern
   already documented for 2016 Antarctica). **Read `param_combine`.** The 2014
   frames read `"standard"` in both structs, and `param_records.combine.img_comb`
   in 2014 is a *third*, different, stale value (`[2.5, −∞, 1; 10, −∞, 3]`) —
   use `param_combine`.
7. **The high pass is a climbing transit, not a survey line.** AGL rises
   2 413 → 2 538 m across the 29 km segment (and 2 223 → 3 173 m across frames
   066–070). There is no constant-altitude assumption to lean on; use the real
   nav (`--levels real`) or accept a ±60 m spread.
8. **The 2017 aircraft banks to 9.8° at s ≈ 41–44 km** and diverges to 1.3 km
   offset by s ≈ 55 km. Do not extend the main segment past s = 40 km.
   `roll_source="nav"` matters more for the high pass (p95 2.03° vs 1.37°).
9. **The bed is flat and deep** — 270 m relief over 29 km, ~1.6° slopes, ice
   2.3–2.6 km thick. Bed-driven off-nadir clutter will be weak; the measured
   +16 dB mid-column contrast is likely dominated by **surface** clutter reaching
   a 29 µs-long ice column, plus volume scattering. If the study is specifically
   about *basal* clutter, this line is a poor target and the Antarctic
   `20161105_05` anchor is far better; if it is about *surface* clutter and
   volume return vs altitude in thick interior ice, this is a good one.
10. **BedMachine v5 (150 m) resolves only 23–46 % of the along-track bed
    roughness the radar picks show.** At 10–16 m facets the bed is pure
    interpolation. Expect simulated basal texture to be systematically smooth.
11. **The high run is not cheaper.** Facets are only 1.5× coarser while the
    cross-track reach is 1.33× wider (±6.39 km, above the tool's 6 km default
    cap). Budget the high run at roughly the low run's cost, and raise
    `--ct-cap`.
12. **No floating ice, no grounding line, no coverage gaps** — the whole line is
    grounded interior ice, bottom pick 100 % populated on all five frames, all
    four products (`CSARP_standard/qlook/mvdr/layer`) present on every frame.
    This is a much cleaner scene than the Antarctic basal case.
13. **Legacy-season nav caveat** (only relevant if anyone revisits the rejected
    candidates): pre-2005 Greenland collections have **no `CSARP_layer` asset**,
    and `2003_Greenland_P3 20030511_01_014` is a 28 361-trace whole-segment file
    whose `Surface` implies a nonsensical 75 m AGL. Treat legacy `Surface` as
    unvalidated.
