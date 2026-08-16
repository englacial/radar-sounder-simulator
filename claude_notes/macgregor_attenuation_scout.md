# MacGregor englacial attenuation as a source of `--att` (2026-08-16)

Scout + prototype. No `src/` or `tools/` changes, nothing committed. Deliverables:
this note, cached sample data under `outputs/cache/`, and numbers.

**Headline.** On our Greenland anchor line (`2014_Greenland_P3 / 20140421_01_069+_070`,
70.3–70.9 N, 44.8–46.5 W) the MacGregor depth-averaged **one-way attenuation rate
over the traced part of the column is 14–15.5 dB/km** — **14.3** re-derived from
the archived reflection intensities on our exact frames (§6) and **15.5**
digitized from the published map (§3). Correcting for the unsampled warm bottom
third of the column gives a *full-column* rate of **~16–19 dB/km**. Both numbers
**support the level-matching estimate (14.9) and reject the
bed-power-vs-thickness slope estimate (~7)**. The withdrawn A = 14.0 was, on this
evidence, essentially right; the value that replaced it is not. Recommended for
this line: **A ≈ 16 ± 2 dB/km one-way, full column.**

---

## 1. Dataset identity: what exists, what does not

| thing | status |
|---|---|
| MacGregor et al. (2015), *Radar attenuation and temperature within the Greenland Ice Sheet*, JGR-ES 120, 983–1008, `doi:10.1002/2014JF003418` | the canonical source. Open-access copy: <https://escholarship.org/uc/item/17r372tq> |
| **a gridded / along-track digital product of Na** | **does not exist publicly.** Not at NSIDC, not on Zenodo, not in the author's repos |
| NSIDC **RRRAG4** v1 (`10.5067/UGI2BGTC4QJA`, retired) / v2 (`10.5067/SZSVA3CWV3U4`) — *Radiostratigraphy and Age Structure of the GrIS* | ages/isochrone depths only. **v1 additionally archives per-reflection `echo_intensity` (dB)** — the raw ingredient of the attenuation fit |
| **Zenodo `10.5281/zenodo.14182641`** (concept) → record `15547327`, MacGregor et al. 2025 ESSD 17, 2911 — *A revised and expanded deep radiostratigraphy of the GrIS 1993–2019* | the successor. Per-campaign `.mat` with **`int` = recorded reflection power** per traced reflection, plus depth/age/twtt/lat/lon/thick. CC-BY-4.0. **This is the practical route to real Na numbers.** |
| NSIDC **RDBTS4** v2 = **GBaTSv2** (MacGregor et al. 2022, TC 16, 3033) | basal thermal state, melt, basal water, speed ratio at 5 km. **No temperature, no attenuation.** Also on GitHub as a single 17 MB netCDF |
| `github.com/joemacgregor/atten` | the *code* (`atten.m`, `hfcond.m`, `hfcondprop.m`) that produced the 2015 estimates — conductivity model with published Arrhenius parameters. No data |

The 2015 paper's data statement is only *"Echo-intensity data will be archived at
the National Snow and Ice Data Center"* — which is what happened (RRRAG4 carries
echo intensity), so the derived Na field was never archived. A CMR/Earthdata
keyword sweep for attenuation collections returns nothing for Greenland; the only
CMR hits are Antarctic and local (see §7).

**Variables and conventions in the 2015 paper** (these matter):

* `Na` — depth-averaged **one-way** attenuation rate, dB/km, positive, from a
  weighted least-squares fit of geometrically corrected reflection power against
  depth over 1 km along-track bins: `Δ[Prc] = 2 Na Σ Δz + b`, with
  `[Prc] = [Pr] + 20 log10(h + z/√ε′)`, `ε′ = 3.15`, `h` = aircraft height above
  the surface (eqs. 1–5).
* It is **radar-inferred**, not modelled — the only "modelled" fields in the
  paper are the comparison temperature `Tm` from a thermo-mechanical model
  (Fig. 8b) and the σ∞(T, chemistry) model used to convert Na → `Ta′`.
* Constraints: ≥ 5 traced reflections, `zmin ≥ 200 m`, `zmax ≤ 0.85 H`,
  `zmax − zmin ≥ 0.25 H`, heading change < 2°/km. **So Na samples only the
  middle of the column, never the bottom ~15–35 %** — the part that matters most
  for a bed echo.
* `Na = 0.9218 σ∞` with σ∞ in μS/m (eq. 6; reproduced exactly — the paper's
  NGRIP pair 10.1 dB/km ↔ 11.0 μS/m checks out).
* Ice-sheet-wide: `< 10 dB/km` along the central divide to `> 25 dB/km` within
  ~100 km of the margin; crossover scatter `0.3 ± 3.2 dB/km` (the honest error
  bar), formal `Ña < 1 dB/km` over most of the ice sheet.
* No successor attenuation product is tied to GBaTSv2 or the 2025
  radiostratigraphy release — both explicitly exclude attenuation.

## 2. Access recipe

```
# 1. the paper (figures are the only published form of the Na field)
curl -L https://escholarship.org/content/qt17r372tq/qt17r372tq.pdf
# 2. the ingredients to recompute Na yourself (CC-BY-4.0, no login):
curl -L "https://zenodo.org/records/15547327/files/Greenland_radiostratigraphy_v2_2014_P3.mat?download=1"
#    (560 MB for 2014_P3; per-campaign. The zipped GeoPackage sibling is
#     smaller but carries depths only, no intensity.)
#    Segment list without downloading: HTTP range-request the .zip's central
#    directory. Our line is segment `20140421_01_058-072` (frames 058-072,
#    i.e. our _069/_070 are inside it).
# 3. basal thermal state context (17 MB, no login):
curl -L https://raw.githubusercontent.com/joemacgregor/GBaTSv2/main/GBaTSv2.nc
```

No Earthdata login is needed for any of this (RRRAG4 at NSIDC does need one, but
it is superseded by the Zenodo release). Cached in this repo:

| file | what |
|---|---|
| `outputs/cache/gbatsv2_greenland_basal_thermal_state.nc` (+ `.json`) | GBaTSv2, 5 km, EPSG:3413 |
| `outputs/cache/macgregor2015_attenuation_20140421_01_digitized.json` | Na, Ña, zmin/H, zmax/H, h/H, Ta′, Ta′ uncertainty digitized from Figs. 4a–e and 7a–b, sampled every 10th trace along the anchor line, with method + validation in the sidecar fields |
| `outputs/cache/macgregor_radiostrat_v2_20140421_01.npz` (+ `.json`) | per-reflection depth / intensity / age for segment `20140421_01_058-072`, extracted from the 560 MB campaign file (the big file itself was not kept) |

### How the digitization works (and why it is trustworthy)

The published maps are the only form the Na field exists in, so it was digitized
rather than guessed:

1. `pdfimages` the figure raster out of the article PDF (300 dpi embedded JPEG).
2. Georeference by least-squares affine fit of the **magenta deep-ice-core
   markers** (Camp Century, NEEM, NGRIP, GRIP/GISP2, DYE-3) to EPSG:3413.
   Max residual **0.5 px = 1.9 km**; scale 3.84 km/px (Fig. 4), 2.35 km/px
   (Fig. 7). Fitting EPSG:3413 vs. other polar-stereographic variants makes no
   difference at this residual, confirming the paper's projection.
3. Read the discrete colorbar patches out of the figure itself (20 patches,
   5→25 dB/km in 1 dB steps) and classify the nearest coloured pixel to each
   trace within 3 px.
4. **Validation:** at NGRIP the digitizer reads 9.5 dB/km against the paper's
   published borehole value of **10.1 dB/km** — one colour bin.

Scripts live in the session scratchpad (`digitize_panels.py`, `column_model.py`,
`robin_column.py`, `rederive_na.py`); they are throwaway, not repo code.

## 3. Values along the Greenland anchor line

Sampled every 10th trace; `s` is arc length from `_069` trace 0 (the study
segment is s = 11–40 km).

| quantity (Fig.) | full 99.7 km line | **s = 11–40 km** | s = 79–97 km |
|---|---|---|---|
| **Na, one-way (4a)** | median **15.5**, all reads in 14.5–16.5 | **15.5** (bins 14.5 / 15.5 only) | 15.5 |
| *(same, re-derived from archived intensities, §6)* | *14.2 (IQR 13.6–15.1)* | ***14.3** (IQR 13.8–14.8)* | *15.9* |
| Ña, formal uncertainty (4b) | 1.1–3.9 | **1.1** | 3.9 |
| zmin/H (4c) | 22–30 % | **29.5 %** | 22.5 % |
| zmax/H (4d) | 65–74 % | (sparse) **~71.5 %** | 71.5 % |
| h/H, fraction sampled (4e) | 30–57 % | **47.5 %** | 50.5 % |
| Ta′, depth-averaged T (7a) | −32.5 … −19.5, median −27.5 | **−25.5** | −32.5 |
| Ta′ uncertainty (7b) | 3.4–3.8 K | **3.4 K** | 3.8 K |

Spatial structure along the line is **flat**: one colour bin of variation over
100 km (15.5 ± 1), against a regional field that ranges 5→25 dB/km. The 5 km
figure bins and the 3.8 km pixel do smooth real structure — but the re-derivation
in §6, which has no such smoothing, agrees: **IQR 13.6–15.1 dB/km over the whole
100 km**, with a mild rise (~+1.5 dB/km) toward the far end of the line
(s = 79–97 km, i.e. downstream/west). Along-line structure is smaller than the
dataset's own ±3.2 dB/km crossover scatter.

**Consistency check of the digitized pair.** Porting `hfcond.m`/`hfcondprop.m`
to Python and asking what scale factor β on the W97 σ∞ model reproduces
Na = 15.5 dB/km at Ta′ = −25.5 °C with GRIP mean chemistry gives **β = 2.93**,
against the paper's own corrected model **β ≈ 2.6** — a 13 % agreement between
two independently digitized panels through a model that was never fit to them.
The digitization is not producing nonsense.

**Basal context (GBaTSv2, sampled along the same line):** `likely_basal_thermal_state
= +1` (**likely thawed**) everywhere on the line, `basal_water = 2` published
identifications on the study segment, `speed_ratio ≈ 0.29` (slow,
deformation-dominated, not an ice stream).

## 4. Na is NOT `--att`: the full-column correction

`Na` averages only z/H ≈ 0.20–0.72 here (§3, §6). A bed echo traverses the whole
column, including the warm bottom quarter-to-third that the traced reflections
never reach, where attenuation is largest. Prototype: Robin (1955) steady-state
column (Nye vertical velocity, κ = 34.4 m²/yr, k = 2.1 W/m/K, H = 2400 m),
surface temperature tuned so the modelled mean over the *sampled band* equals
Ta′ = −25.5 °C, σ∞ model scaled so the band-mean rate equals the observed
**14.3 dB/km over z/H = 0.20–0.71**, then integrated over the full column:

| accum (m/yr) | G (mW/m²) | Ts (°C) | Tb (°C) | **full-column Na** | ratio to band | two-way loss @2.4 km |
|---|---|---|---|---|---|---|
| 0.25 | 40 | −26.2 | −12.5 | 17.5 | 1.22 | 83.8 dB |
| 0.25 | 60 | −26.6 | −6.0 | 20.2 | 1.41 | 97.1 dB |
| 0.35 | 50 | −25.9 | −11.4 | **17.6** | **1.23** | **84.3 dB** |
| 0.45 | 40 | −25.7 | −15.4 | 16.2 | 1.13 | 77.7 dB |
| 0.45 | 60 | −25.7 | −10.4 | 17.6 | 1.23 | 84.6 dB |

(Recovered Ts ≈ −26 °C matches RACMO-class surface temperatures for this site —
another check that fell out rather than being imposed. If the bed really is at
the pressure-melting point, as GBaTSv2 says, the ratio rises further, to ~1.5.)

**So: `Na_sampled ≈ 14.3` → `Na_full-column ≈ 16–19 dB/km` (central 17.6), i.e.
the number a simulator should use for a bed echo is ~15–25 % larger than the
published/derived map value.** This is a general caveat for any use of this
dataset, not a line-specific one — and it is the single most important thing to
carry into any `--att` wiring.

## 5. Arbitration

Our three candidate values for this line, against MacGregor:

| estimate | value (one-way dB/km) | verdict |
|---|---|---|
| route (b) slope, bed power vs thickness (part 4.4, mvdr, both passes) | **6.5–8** | **rejected.** Off by a factor of ~2 from two independent, physically-grounded measurements; below the *floor* of the whole GrIS map (< 10 dB/km only on the central divide, and we are ~200 km west of it on the flank) |
| route (a) level anchoring (part 3.3, implied effective) | **14.9** | **supported.** It sits between the re-derived 14.3 and the digitized 15.5 — and unlike them it is already a *full-column* number, so it is on the low side of the full-column estimate (16–19) rather than dead on |
| adopted-then-withdrawn A = 14.0 (part 2.3) | 14.0 | within 0.3 dB/km of the re-derived sampled-band Na. The withdrawal was justified on method grounds (the route-(b) high-pass number was contaminated); the *value* it produced was right for the wrong reason |

**Why the slope route is biased low, physically.** Route (b) regresses bed power
on ice thickness and reads the slope as `−2A`. That is only valid if bed
reflectivity is uncorrelated with thickness. On a thawed-bed line it is not:
thicker ice ⇒ warmer bed ⇒ more basal water ⇒ *brighter* bed, which is a positive
`|Γ|²`–H correlation that cancels part of the attenuation slope. GBaTSv2 says
this line is likely thawed with basal water identified, so the confounder is
present, and it biases A **downwards** — exactly the direction and roughly the
magnitude of the discrepancy. The part-4 conclusion that the two passes agreeing
on mvdr at ~7 is "the physically required outcome" is right about *product
stability* but does not rule out a shared confounder: both passes see the same
bed, hence the same `|Γ|²`–H correlation.

**What it implies for bed reflectivity (the joint-fit tension).** Part 3 anchored
the level with median `|Γ|² ≈ −20.8 dB` (7.9 dB below the Fresnel ice→rock
ceiling of −12.86 dB) at an implied effective A = 14.9. Moving A moves the
required bed brightness by `2 ΔA H = 4.94 dB per dB/km` at H = 2.47 km:

| A (full column) | required median `|Γ|²` | physicality |
|---|---|---|
| 7 (slope route) | ≈ **−60 dB** | absurd for a wet, thawed bed; ~47 dB below rock |
| 14.3 (re-derived, *sampled band* — a lower bound on the full-column value) | −23.7 dB | comfortable |
| 14.9 (level anchor as run) | −20.8 dB | comfortable |
| **16** (recommended) | **−15.4 dB** | **2.5 dB below the rock ceiling — fine** |
| 17.6 (central full-column estimate) | −7.5 dB | above the rock ceiling; needs a wet bed everywhere |
| 19 | −0.6 dB | **unphysical** |

So the level constraint and the MacGregor sampled-band Na agree beautifully, and
the joint fit is *not* the one part 4 feared: the reconciliation is **a
normal-brightness bed with high attenuation**, not a very dim bed. The `−43 dB`
dim-bed scenario that A ≈ 7 would require is dead.

A residual, much smaller tension remains: the level constraint tolerates A up to
~16.5 before the median `|Γ|²` crosses the Fresnel rock ceiling, while the
full-column correction centres on 17.6. Something in that chain is ~1–3 dB/km
optimistic — candidates: our absolute level chain (system loss, K), the Robin
profile's deep gradient (the low-G / high-accumulation corner gives 16.2), or
MacGregor Na being slightly high on this flank. The admissible window from
combining everything is **A ≈ 15–17 dB/km one-way, full column**; the honest
recommendation is **A = 16 ± 2**.

## 6. Re-deriving Na ourselves on the exact line

The digitized map value is a *read of a figure*. The archived reflection
intensities let us skip the figure entirely and apply MacGregor's own method to
our own frames. Segment `20140421_01_058-072` (750 km, 129 traced+dated
reflections) is in `Greenland_radiostratigraphy_v2_2014_P3.mat`; 6688 of its
49 980 traces fall within 200 m of the anchor-line nav (i.e. our `_069`+`_070`).

Method as published: 1 km along-track bins on the anchor line; per reflection,
the bin's intensity `[Pr]`; geometric correction
`[Prc] = [Pr] + 20 log10(h + z/√3.15)` with `h` from the cached OPR `Surface`
twtt (median **480 m**, matching the known 475 m AGL); least-squares fit of
`[Prc]` against depth over reflections with `z ≥ 200 m`, `z ≤ 0.85 H`, ≥ 5
reflections and `zmax − zmin ≥ 0.25 H`; `Na = −slope/2`.

| | full 99.7 km line | **s = 11–40 km** | s = 79–97 km |
|---|---|---|---|
| **Na (dB/km, one-way)** | median **14.2**, IQR 13.6–15.1, range 10.3–18.0 | **14.3**, IQR 13.8–14.8 (29 bins) | 15.9, IQR 15.0–16.5 |
| reflections per fit | 12 | 13 | 10 |
| fit residual rms | 2.3 dB | 2.6 dB | 2.3 dB |
| z/H sampled | 0.22–0.72 | 0.20–0.71 | 0.24–0.74 |

**Robustness:** swapping the bin statistic from a median to MacGregor's
70th–95th-percentile mean moves the answer by **0.1–0.2 dB/km** (14.0 vs 14.2);
tightening `zmin` to 300 m or `zmax` to 0.80 H moves it by **0.0**. The one thing
that *does* matter is `h`: using the aircraft's WGS-84 elevation instead of its
height above the surface inflates Na by 1.5 dB/km, which is worth knowing if
this is ever wired up.

**Agreement with the published map: 14.3 vs 15.5 = 1.2 dB/km**, well inside the
paper's own ±3.2 dB/km crossover scatter. The gap is expected: v2 (2025) is a
retraced, expanded stratigraphy, my fit omits the along-track slope correction,
and the map read is a 5 km bin from a neighbouring transect. **Three independent
routes — re-derived 14.3, digitized 15.5, our own level anchoring 14.9 — land
within 1.2 dB/km of each other, and none of them is near 7.**

## 7. Antarctic equivalent (secondary)

**There is no MacGregor-equivalent observational attenuation map for Antarctica.**
What exists:

* **Matsuoka, Pattyn, Callens & Conway (2012), EPSL 359–360, 173–183,
  *Predicting radar attenuation within the Antarctic ice sheet*** — the closest
  thing: a continent-wide **predicted** one-way depth-averaged attenuation field
  from an ensemble of 24 thermomechanical model runs (Pattyn 2010) + the same
  Arrhenius conductivity physics. Not archived as a grid (paper figures only,
  Fig. 7a). Reported values (one-way, dB/km):
  * continental mean **15–16 ± ~6** for essentially all boundary conditions;
  * **majority of the WAIS 10–15**; ice shelves 15–20; Siple Coast / Peninsula
    / coastal margins > 25–30;
  * grounded ice that is neither fast-flowing nor over a lake: **10.7 ± 7.3,
    median 9.2**; grounded ice thicker than 2 km: **8.5 ± 3.0**;
  * **Fig. 7 is the pure-ice contribution only.** Acidity adds **30–70 %** on
    top in West Antarctica (their Fig. 9b).
* **NSIDC-0470 / USAP-DC 609470** (Matsuoka et al.), *Englacial Layers and
  Attenuation Rates across the Ross and Amundsen Sea Ice-Flow Divide* — a
  *local* product near WAIS Divide (−79.5, −112), .mat via usap-dc, not near our
  line and not a grid.
* Dawson et al. (2025, J. Glaciol.), *Ice sheet attenuation from radar sounding
  in the frequency domain* — a newer method applied to interior Greenland and
  Antarctica; worth reading if the frequency-domain route ever becomes relevant
  to us, but it is not a product.

**For our Amundsen line (`20161105_05`, −74.6, −118…−120):** Matsuoka's pure-ice
prediction for the WAIS interior is 10–15 dB/km; adding the West Antarctic
acidity contribution (+30–70 %) gives **~13–25 dB/km**. **The adopted 20 dB/km
sits inside that range** — high-ish but not contradicted. Unlike Greenland, we
cannot do better than a modelled range there without deriving it ourselves.

**And the §6 route does not transfer.** Re-deriving Na needs ≥ 5 traced,
depth-resolved reflections per bin. Antarctic radiostratigraphy is far sparser
than Greenland's: the best regional product near the Amundsen sector, Bodart et
al. (2021, JGR-ES, `10.1029/2020JF005927`) over Pine Island Glacier, traces
**four** IRHs — below MacGregor's own threshold, and over a catchment east of our
line. SCAR's AntArchitecture is the effort that would eventually fix this. So for
the Antarctic line the realistic options are (a) keep 20 dB/km with the Matsuoka
range as its justification, or (b) build a Matsuoka-style prediction ourselves
with the σ∞ + Robin machinery already prototyped here (§4), driven by a local
accumulation / geothermal-flux / surface-temperature triple. Option (b) is maybe
half a day and would give the Antarctic line the same footing as the Greenland
one, minus the observational anchor.

## 8. Integration design for `--att-from-macgregor` (paper only)

**Recommendation: per-line constant, from a small checked-in table — not a raster
lookup, and not per-facet.**

Reasoning:

1. **The along-line variation does not justify per-facet plumbing.** Our 100 km
   line varies by **±1 dB/km in the published field and IQR 1.5 dB/km in the
   re-derivation**, against a ±3.2 dB/km crossover systematic and a ±2 dB/km
   full-column modelling uncertainty. Per-trace spatial attenuation would model
   structure smaller than the dataset's own noise. A per-facet `att_facet` map
   (mirroring `gamma_facet`) is real work in the kernel — the multilayer kernel
   takes per-medium constants and the attenuation enters the per-facet path
   integral — and it would buy nothing here. Revisit only for a line that
   *crosses* the gradient (margin-to-interior, where the field really does go
   10 → 25 dB/km over ~100 km); on such a line the honest first step is
   per-trace, not per-facet, since the along-track gradient is what varies.
2. **There is no raster to look up.** Any implementation has to either digitize
   the figures (§3) or re-derive Na from the Zenodo reflection intensities
   (§6 — the better route, and cheap once the campaign file is in hand). Neither
   belongs behind a CLI flag that implies a live dataset fetch; the campaign
   files are 0.5–0.6 GB each and cover a whole season, not a line. So: a
   **static table in the study-line registry**, one row per
   line, with `att_one_way_db_per_km`, `att_source`, `att_uncert`, and a
   provenance string — the same shape as the other per-line constants, and
   reviewable.
3. **The depth-averaging caveat has to be applied at the table, not hidden.**
   The dataset value is a *partial-column* rate (z/H ≈ 0.3–0.7 here) and our
   `--att` is applied over the whole column. Store *both*: the published
   `Na_sampled` and the `Na_full_column` we adopt, with the ratio and the
   assumptions (Robin profile, basal thermal state source) recorded. Otherwise
   the next person will compare our `--att` against the published map and think
   we are 20 % high.
4. **H varies along the line (2.0–2.6 km) but the *rate* does not** — the
   depth-averaged rate depends on the column's temperature profile, which is
   itself thickness-dependent (thicker ⇒ warmer base ⇒ higher rate). Our column
   model says the effect is ~1 dB/km across our thickness range, i.e. inside the
   noise. Another reason a constant is the right resolution.
5. If a spatially varying option is ever built, the natural interface is a
   **per-trace** `att(s)` array attached to the line (same length as the nav),
   applied to the two-way path of each trace's facets by their nadir column —
   NOT a per-facet field. That keeps the kernel's per-medium constants intact
   and matches how the physical quantity actually varies (laterally, on ~10 km
   scales, not within a footprint).

Concretely, if wired: `--att-from-macgregor` would (a) look up the line in the
table, (b) fail loudly for a line with no entry rather than silently
interpolating a figure, and (c) log the sampled-band value, the full-column
value, and the uncertainty into the run manifest.

## 9. Open questions

1. **The ~1–3 dB/km gap** between the central full-column estimate (17.6) and the
   physicality ceiling from the level constraint (~16.5). Resolving it needs the
   joint (A, `|Γ|²`) fit that part 4 recommended — now with a *prior* on A of
   16 ± 2 rather than a free parameter, which makes the fit far better posed.
2. **Is the bed thawed?** GBaTSv2 says likely thawed; our Robin column with
   nominal G = 50 mW/m² and a = 0.35 m/yr gives Tb ≈ −11 °C (frozen), and needs
   G ≈ 75–80 to reach the pressure-melting point. That inconsistency changes the
   full-column correction by ~2 dB/km and the expected `|Γ|²` by a lot. A
   published Greenland geothermal-flux grid (e.g. the ones GBaTSv2 ingests)
   would close it.
3. **The slope correction.** My re-derivation omits MacGregor's along-track
   reflection-slope correction (their eq. 1 "slope-corrected `[Pr]`"). On this
   gently dipping interior line it should be small, but it is the one published
   step not reproduced, and it is the likeliest source of the 1.2 dB/km gap
   between 14.3 and the map's 15.5.
4. **The slope-route confounder is testable.** If bed brightness correlates with
   thickness on this line for basal-thermal reasons, then binning the route-(b)
   regression by GBaTSv2 basal state / basal water should change the slope
   systematically. That is a cheap experiment with data already cached, and it
   would convert "the slope route is biased" from an argument into a measurement.
5. Whether to apply the same treatment to the Antarctic line: there we would have
   to *build* the Matsuoka-style prediction (temperature model + chemistry)
   rather than read it, since no product exists.
