# Tier 1 status: whole-archive ILATM2 roughness screen, Greenland + Antarctica (2026-08-27)

Code `claude_notes/atm_regional/`: `atm2_search.py` (archive inventory), `atm2_pull.py`
(per-campaign download -> parquet, raw CSV deleted), `atm2_masks.py` (1 km ice / shelf /
distance-to-margin rasters), `atm2_build.py` (QC, masking, per-flight floors),
`atm2_grid.py` (10 km grids, maps, NetCDF/GeoTIFF), `atm2_analyze.py` (variograms,
histograms, relations, GMM; needs `uv run --with scikit-learn`), `atm2_sites.py` (Tier 2
site list). Logs in `claude_notes/atm_regional/logs/`. Nothing committed; simulator source
untouched.

## 1. Data pulled

ILATM2 (Icessn platelets) via `earthaccess`, 2009-2019, bbox Greenland (-75..-10 E,
58..84 N) and Antarctica (< -60 S). NSIDC serves one CSV layout for every year (there is no
separate v1 ASCII to parse: 2009-2012 files are the same `# comment + 11 column` CSV, the
2013+ files add `_V01` / ATM6 headers). Columns kept: UTC s, lat, lon, WGS84 h, S-N and W-E
slope, RMS_Fit (cm), n used, n removed, cross-track distance of the block, track id
(0 = 80 m nadir block, 1..3 or 1..5 = cross-track segments). Platelet footprint is
~80 m wide x 65-130 m along track (0.25 s output, 0.5-1 s smoothing): the ILATM2 RMS is a
planar-fit residual over 1-100 m, not a 250 m product.

48,035 granules, 10.3 GB (the plan's "~2 GB" was wrong by 5x; 2017-2019 are split into
thousands of 10-min files). Downloaded campaign by campaign, parsed to
`outputs/cache/atm2/platelets_<year>_<gl|aa>.parquet` (4.2 GB total), raw CSVs deleted;
per-campaign inventory in `outputs/cache/atm2/granules_<year>_<hemi>.json`.

| year | GL granules / MB / platelets / flight-days | AA granules / MB / platelets / flight-days |
|---|---|---|
| 2009 | 1250 / 363 / 4.3 M / 13 | 1232 / 536 / 6.4 M / 16 |
| 2010 | 1516 / 480 / 5.8 M / 19 | 524 / 148 / 1.8 M / 6 |
| 2011 | 1595 / 745 / 8.9 M / 29 | 1058 / 506 / 6.1 M / 20 |
| 2012 | 1815 / 863 / 10.3 M / 33 | 581 / 282 / 3.4 M / 13 |
| 2013 | 942 / 458 / 5.5 M / 18 | 381 / 189 / 2.2 M / 7 |
| 2014 | 1926 / 941 / 11.3 M / 36 | 678 / 422 / 5.0 M / 20 |
| 2015 | 1785 / 844 / 10.1 M / 41 | none (no ATM in the 2015 Antarctic campaign) |
| 2016 | 1184 / 461 / 5.5 M / 21 | 1005 / 394 / 4.7 M / 22 |
| 2017 | 7219 / 783 / 9.3 M / 31 | 1020 / 125 / 1.5 M / 8 |
| 2018 | 5541 / 396 / 4.7 M / 14 | 4868 / 390 / 4.0 M / 23 |
| 2019 | 9141 / 757 / 7.8 M / 28 | 2774 / 229 / 2.4 M / 19 |

(flight-days = days with on-ice platelets after masking). Total after QC and masking:
77.5 M platelets (23.6 M centre), 430 flight-days.

## 2. QC thresholds

n_used >= 100; n_removed <= 10 % of n_used; 0 < RMS_Fit < 5 m; |slope| < 0.5; on ice
(Natural Earth 10 m glaciated areas for Greenland; land + ice-shelf polygons for
Antarctica). The middle cross-track segment (same footprint as the nadir block) is dropped;
"centre" = track 0, "off-centre" = segments with |cross-track offset| >= 30 m. QC removes
1-7 % per campaign, masking a further 10-35 % (sea ice, Arctic Canada, ocean transits).

## 3. Noise floors

Per flight-day floor = p3 of centre-platelet RMS_Fit over on-ice platelets (>= 200 needed,
else campaign median; all 430 days had enough). `outputs/atm_regional/flights.csv`.

| | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GL floor cm, median (p10-p90 of days) | 6.7 (6.3-7.2) | 6.4 (5.6-7.2) | 4.6 (4.1-6.4) | 4.3 (3.7-5.3) | 4.6 (4.3-5.2) | 4.9 (4.5-5.3) | 5.8 (4.6-7.4) | 5.1 (3.9-5.6) | 3.8 (3.0-4.3) | 3.4 (3.0-4.3) | 3.2 (2.5-4.2) |
| AA floor cm | 5.4 (4.8-6.3) | 8.6 (8.5-9.4) | 4.8 (4.2-5.5) | 4.2 (3.9-5.3) | 4.1 (3.9-4.5) | 4.8 (4.5-5.8) | - | 4.2 (3.4-5.1) | 3.5 (3.3-5.1) | 4.3 (3.6-5.1) | 4.3 (3.2-6.2) |

Cross-check against the ATM1B crossover totals on the study flights (ranging + scan-to-scan,
`atm_roughness_results_2026-08-26.md`):

| flight | ILATM2 p3 floor | ATM1B crossover total @0.27 m |
|---|---|---|
| westcoast 2016-05-11 (ATM5) | 5.5 | 5.3 |
| westcoast 2017-05-10 (ATM6) | 4.3 | 4.8 |
| westcoast 2019-05-14 (ATM6) | 3.4 | 2.0 |
| geikie 2014-04-21 (ATM4) | 4.6 | 4.9 |
| getz 2016-11-05 (DC-8 ATM6) | 4.6 | 5.6 |
| david 2013-11-19 / 20 (ATM4) | 4.1 / 4.4 | 6.9 / 8.2 (sea ice, not a floor) |

Agreement within ~1 cm for the 2014-2017 flights. On the low-noise 2019 flight the p3 floor
(3.4 cm) exceeds the instrument total (2.0 cm): the p3 is instrument noise plus the
smoothest real sub-80 m surface on that day (~2.7 cm in quadrature), so the "floor" is an
upper bound on noise and r a lower bound on roughness for the smoothest interior. The
2010 Antarctic floor (8.6 cm) is anomalous (ATM4 on the DC-8 at high altitude); those 6
days are essentially blind.

r = sqrt(max(RMS^2 - floor^2, 0)); at_floor = RMS < floor x 10^(3/20).

## 4. Maps and grids (`outputs/atm_regional/`)

`map_gl.png`, `map_aa.png` (r median, r p90, at-floor fraction, platelet slope, years
of coverage, off/centre ratio); `grid_<hemi>.nc` (all fields incl. r p10/p90, n, n_days,
n_years, log r mean/std, elevation, slope, distance to margin/grounding line, shelf
fraction, month), `grid_<hemi>_r_med.tif`, `grid_<hemi>.csv`. 10 km cells with >= 30
centre platelets: Greenland 13,442 cells (71 % of the ice-sheet 10 km cells), Antarctica
13,340 (10 % of the ice area: West Antarctica, Peninsula, Siple Coast, Transantarctic
outlets, a few East Antarctic transits; the plateau interior is essentially unsampled).

The 100 m-scale slope is the platelet planar-fit slope itself (80 x 65-130 m fit). The
"anisotropy" panel is the off-centre / centre r ratio: it is 1.0-1.2 almost everywhere
and rises to > 1.3 only at margins and on the Peninsula, where it measures the
swath-edge geometry (larger incidence angle, more slope leakage) rather than surface
anisotropy. The five-across pattern exists only in 2009-2010 Antarctica; the ILATM2
platelet geometry cannot give a usable along/cross statistic. Drop it; Tier 2 (ATM1B
pair azimuth sectors) is the anisotropy measurement.

## 5. Covariates

At the 10 km grid: elevation (platelet median), platelet slope, distance to margin
(Greenland: to the nearest non-glaciated cell; Antarctica: grounded ice to the nearest
non-grounded cell, i.e. grounding line or coast; shelf cells have d = 0 and a shelf flag),
latitude, month. No MAR / RACMO / ERA5 fields are cached anywhere in the repo (`outputs/cache`
has BedMachine windows, ArcticDEM/REMA tiles, SUMup, OPR frames only); per the brief I
did not hunt beyond that. Facies / wind / SMB are therefore represented by the proxy
(elevation, distance to margin, latitude); this must be stated in any regime naming.
Facies masks are the first covariate to add in the Tier 2 grouping.

## 6. Spatial structure and grouping (`analysis.json`, `variogram_*.png`, `hist_*.png`, `relations_*.png`, `clusters_*.png`)

Along-track variogram of log10 RMS_Fit on the centre-platelet series (0.1-160 km lags):
semivariance rises as a power law from 0.1 km to ~40 km (Greenland) / ~60 km (Antarctica)
with no plateau below that; 1/e correlation length 11 km (GL) / 7 km (AA). Consistent
with the ATM1B block series (1-4 km on Greenland lines, 10-15 km Antarctic) once the
platelet noise is removed, and it says the field is multi-scale, not a patchwork of
uniform regions.

10 km grid variogram of log10 r_med: Greenland keeps rising to ~1000 km (elevation
trend; sill 0.15 dex^2, 1/e length 141 km all cells, 45 km above-floor cells);
Antarctica flattens by ~50-100 km (sill 0.10 dex^2, 1/e 35 km / 22 km) - i.e. no
continental-scale trend where the archive samples it.

Histograms: log r is unimodal with a heavy high tail (margins, crevassed outlets) in both
ice sheets. 1-D GMM BIC keeps decreasing to 5 components on 200 k platelets / 10^4 cells,
which is skew, not separate modes; there is no visible second peak.

Relations (Spearman on cells; all / above-floor):
Greenland: elevation -0.63 / -0.64, distance to margin -0.68 / -0.67, slope +0.67 / +0.68,
latitude ~0. Median r of above-floor cells by elevation band: 36 cm (< 500 m), 14 (500-1000),
6.7 (1000-1500), 6.2, 5.7, 4.9 cm (> 2500 m). Month: Mar 4.5 / Apr 4.8 / May 4.5 cm (no
spring trend); the few Jul (7.6 cm, n small) and Sep/Oct cells are summer/autumn sea-ice-season
flights over the SW margin.
Antarctica: elevation +0.22 / +0.15, distance +0.16 / +0.12, slope +0.34 / +0.24 - weak.
Median r of above-floor cells 6.4 cm (< 500 m) rising to 8.5 cm (> 2500 m); shelves
4.4 cm vs grounded 5.9 cm; Oct 5.7 vs Nov 5.4 cm.
Season: the Greenland spring (Mar-May) vs Antarctic spring (Oct-Nov) contrast is the
GL-interior 5 cm vs AA-interior 8.5 cm difference, but it is confounded with sastrugi
climate; the archive has no Greenland autumn / Antarctic summer to separate them.

GMM on (log r, log slope, elevation, log distance), k by the "< 10 % of total BIC drop"
rule (BIC min is at the k = 6 cap in both):

Greenland, k = 3 (`grid_gl_clusters.csv`):
| regime | n cells | r_med cm (p10-p90 of cells) | at floor | slope | elev m | dist km | note |
|---|---|---|---|---|---|---|---|
| 0 interior | 6152 | 3.7 (2.3-5.7) | 0.67 | 0.004 | 2354 | 194 | dry snow / upper percolation; mostly at floor |
| 1 percolation belt | 4646 | 5.1 (3.4-7.7) | 0.40 | 0.013 | 1772 | 46 | westcoast line sits here |
| 2 margin | 2644 | 17 (5-103) | 0.13 | 0.05 | 905 | 4 | ablation zone, outlets, peripheral ice |
Spatially these are three concentric bands (clusters_gl.png); the interior/belt boundary
is the noise floor as much as a surface change.

Antarctica, k = 2: 0 = shelves / sea level (3417 cells, r 4.5 cm, 48 % at floor);
1 = everything grounded (9923 cells, r 5.8 cm (3.0-11.5), slope 0.011, 34 % at floor).
The grounded set does not split on the covariates offered: the Siple Coast, Pine
Island/Thwaites catchments, Peninsula plateau and Transantarctic outlets sit in one
lognormal cloud 3-12 cm. Sub-strata for Tier 2 are therefore taken by elevation band,
not by the GMM.

Year-to-year: at cells with >= 4 years above the floor (GL 4961, AA 1030 cells) the std of
the annual r_med is 1.6 dB median, 3.7-3.9 dB p90, in both ice sheets - the same +-3 dB
the ATM1B westcoast years gave.

## 7. Regime candidates and Tier 2 site list

Candidates (covariate signatures above): GL-interior (h > 2000 m, d > 100 km, slope < 0.005,
r <= 4 cm, at floor), GL-percolation belt (1200-2000 m, 20-80 km, r 3-8 cm),
GL-margin (< 1200 m, < 15 km, slope > 0.02, r 10-100 cm, self-affine expected),
AA-shelf (r 3-6 cm), AA-grounded coastal (< 1500 m), AA-grounded interior (> 1500 m, r 8 cm,
above floor - the sastrugi regime the geikie line hinted at, but rougher). Six strata that
the ILATM2 screen can *name*; whether they are distinct scattering regimes is Tier 2's
question.

`outputs/atm_regional/tier2_sites_draft.csv`, 709 rows: 459 stratified random 10 km cells
(strata = GMM regime x elevation band {0-500, 500-1500, 1500-2500, > 2500 m}, >= 20 per
stratum, repeat-year and above-floor cells weighted 3x / 2x; 259 GL + 200 AA), 200
most-repeated cells (GL >= 9 years, AA >= 6, capped at 100 each, spread across strata),
the 4 study lines (start/mid/end), 18 Greenland cores (B26, Camp Century, DYE-2, Summit,
NGT B16-B29, SE-Dome, FA13A) + 6 approximate EGIG-line points, 20 Antarctic SUMup cores.
Columns: lat/lon/x/y (median platelet position, so the site is on a flown line), regime,
stratum, years available, r_med, at-floor flag, covariates, priority (0 = ground truth,
1 = repeat + above floor, 2 = rest). Coverage of the fixed sites: all Greenland cores and
both Greenland lines have 1-11 ILATM2 years within 5 km (B18/B21 only at 10-15 km); Getz
has 2016 (+2009/10/14 at the start); David has nothing within 15 km (as ATM1B found);
of the 20 Antarctic cores only WDC06A, the Peninsula OPTV sites, BER11 (Berkner) and RICE
are within 15 km of a flown line - the DML / Dome Fuji cores are 300-900 km from any ATM.

Tier 2 budget from the study-line pull (a 5 km site touches ~3 ILATM1B granules of
10-25 MB): ~50-100 MB per site-year; 709 sites x median 3 years ~ 100-200 GB if
every repeat is taken. Take one year per stratified site (~40 GB) and all years only at the
200 repeat + 50 ground-truth sites (~50 GB).

## 8. Honest limits

- At the floor: 46 % of Greenland centre platelets (8 % below 500 m, 39 % at 1000-1500 m,
  54 % at 1500-2000, 59 % at 2000-2500, 65 % above 2500 m); 48 % of Greenland cells and
  67 % of the interior regime. Antarctica 39 % of platelets, 35 % of cells, but only
  20-25 % above 1500 m: the Antarctic grounded interior *is* above the floor (8-9 cm at
  80 m scale), the Greenland interior is not (< 4 cm, ATM6-era floor 3-4 cm).
- So the ILATM2 screen shows Greenland structure only as the concentric elevation /
  distance bands; inside the dry-snow zone it is blind (r <= floor, lognormal 2-5 cm with
  no spatial organisation resolvable). It shows Antarctic structure as shelf vs grounded
  and a weak elevation increase; no basin-level grouping.
- ILATM2 RMS mixes 1-100 m. The Bragg band (1-5 m) contributes ~2-3 cm on every line
  measured at ATM1B (octave RMS 2-3 cm at 2-8 m), i.e. it is *below* the ILATM2 floor by
  construction. The screen therefore ranks sites by their 10-100 m roughness, which the
  ATM1B study showed is the persistent topographic part, not the Bragg-band part that
  re-forms each season. Regime candidates here are 10-100 m regimes.
- Coverage bias: Antarctica 10 % of ice area, all West Antarctic / coastal; East Antarctic
  plateau, DML and the Dome Fuji cores are unsampled. Greenland 71 % but the NE interior
  sector has gaps (map_gl.png).

## 9. Recommendation for Tier 2

Proceed with Tier 2 on the stratified list, but change the emphasis from "find the
regime boundary" to "measure the Bragg-band statistics inside each stratum", because
the screen shows (a) a continuous, elevation-organised field rather than modes, and (b)
blindness exactly where the simulator most needs the answer (interior firn). Concretely:

1. Run the ATM1B pipeline first on the 200 repeat sites + ground truth (priority 0-1,
   ~50 GB): these give the Bragg-band S(k) with its own noise budget (crossover method),
   the persistence partition, and the geikie-vs-westcoast (exponential vs power-law)
   question at 100+ interior sites instead of 1.
2. The thinned-ATM1B fallback from the plan is needed for the Greenland interior
   regardless: budget = the interior stratified sites (GL regime 0: ~85 sites x 1 year
   x ~60 MB ~ 5 GB) plus a 25 km-spaced thinning along 3 interior transects (the
   Summit-Camp Century, EGIG and the 2011/2017 NE lines: ~60 further 5 km sites, ~4 GB).
   That is < 10 GB and covers the blind zone at the ATM1B noise (1.6-4 cm ranging,
   Bragg-band octaves resolved to ~1 cm).
3. Do not use ILATM2 for anisotropy or for the interior; keep it as the covariate /
   stratification layer and as the 10-100 m persistent-roughness map.
4. Add MAR/RACMO facies and wind before the Tier 2 grouping; the current proxy cannot
   distinguish "dry snow" from "far from the coast".
