# Tier 2: Bragg-band ATM1B roughness at 726 sites, both ice sheets; where the exponential ACF works (2026-08-27)

Code `claude_notes/atm_regional/tier2/`: `sites.py` (site list), `visits.py` (flight visits from the
ILATM2 platelets), `pull.py` (earthaccess search/download per visit), `roughness.py` (per-site
pipeline, reuses `claude_notes/atm_roughness/atm_roughness.py`), `batch.py` (sharded driver,
checkpoint per site-visit), `collect.py`, `covariates.py` (RACMO / MEaSUREs sampling, by a
sub-agent), `analysis.py`. Logs in `tier2/logs/`. Data `outputs/cache/atm/tier2/<site>/<date>/`.
Results `outputs/atm_regional/tier2/`: `rows.parquet` (one row per site-visit incl. the pooled
D(r) arrays as JSON), `site_medians.csv`, `tables.md` (by stratum / facies / year / pair class /
kind + recommendation), `recommendation_by_stratum.csv`, `analysis.json` (year partition,
variograms, GMM, GBT), `map_gl.png`, `map_aa.png`, `clusters_{gl,aa}.png`,
`fig_misfit_vs_covariates.png`, `covariates.csv`, `pull_log_*.csv`. Nothing committed; simulator
source untouched.

## 1. Data pulled

Sites: the 709-row draft minus 19 sites with no ATM within 15 km (David line, DML / Dome Fuji /
Berkner-FRI cores) plus 65 Greenland interior transect sites (25 km thinning along Summit-Camp
Century, EGIG and Summit-NE; cells within 8 km of the transect) = 755. Visits from the ILATM2
platelets (centre track within 2.5 km, passes split at 10 min gaps, >= 40 platelets): 743 sites,
5032 visits. Selection: ground-truth / study-line sites all dates; repeat sites all 2013-2019
years + two of 2009-2012; stratified and transect sites the two (phase 2 stratified: one)
best-noise years in the order 2019 > 2018 > ... > 2009. Processing order phase 1 (repeat +
ground truth) -> phase 3 (GL interior stratified + transects) -> phase 2 (stratified), 3 shards,
~3.5 h wall.

Pulled: 2906 ILATM1B granules, 59.5 GB (612 Qfit `.qi` v1 files 2009-2012 at ~53 MB, 2294 HDF5
v2 at 7-25 MB); no ILNSA1B used. 2111 site-visits OK at 726 sites (GL 423: 100 repeat, 258
stratified, 47 transect, 18 ground truth; AA 303: 100 repeat, 198 stratified, 5 ground truth).
Skipped: 12 sites with no usable visit (B17, B18, B21, RICE, BER11, egig_T01/T03/T04, gl_r0_48328,
gl_tr_summit_0400, aa_r1_61976, aa_r1_206145) and 15 site-dates on the Summit-Camp Century transect
where ILATM2 exists but no ILATM1B granule is served. Repeat sites have a median 9 (GL) / 5 (AA)
years; stratified sites 1 year. Per site-visit: median 116 k shots after QC in the 5 km x ~300 m
swath (0.08 shots/m2); same-scan pair class (ATM6 9-10 kHz) on 40 % of site-visits, cross-scan on
the rest. Crossover noise totals (median per year): GL 7.5 cm (2009) -> 2.3 cm (2019); AA
5-7 cm, 11.8 cm in 2010 (blind), 2.7 cm in 2019.

## 2. Method (changes vs the study-line pipeline)

Same detrend (5 m median grid, Gaussian high-pass, half-power 200 m), pair structure function
D(r) in quarter-octave lags 0.25-110 m, azimuth sectors, crossover / scan-to-scan / ranging
noise budget per site-visit, octave RMS 1-128 m, 2-D anisotropy. New: track axis = principal
axis of the cloud (no radar anchor); the fit is on the pooled 5 km D(r) (1 km blocks kept as
diagnostics in `blocks_json`); **fit band 1 <= r <= 30 m lag** with a free nugget bounded by the
crossover total (same-scan: by the ranging noise); families Gaussian, exponential, power law and
**Matern with nu free** (nu = 1/2 is the exponential); BIC per family; runs-test whiteness. The
Bragg S(k) at 5 / 1.5 / 1.0 / 0.75 m is the closed form of each fitted family; 1.0 and 0.75 m
are extrapolations below the fit band. **Reference family** for the misfit = best BIC among
exponential / power law / Matern (the Gaussian wins 0-18 % of site-visits, only at the roughest
margins and shelves, and where it wins its Bragg tail is -1300 dB, i.e. unusable, so it is
excluded from the reference). Misfit = S_exp(k_B) - S_ref(k_B) in dB, zero by construction when
the exponential is itself the reference; the misfit against the Matern fit is reported alongside
as the continuous measure. sigma reported two ways: the exponential fit parameter (`e_sigma`)
and the non-parametric band-limited RMS below 30 m wavelength (`sigma_bl30`, = sqrt(D(15 m)/2 -
n0)); for l < 30 m (68 % of site-visits) e_sigma / sigma_bl30 = 1.15 (median); fixing the nugget
at the crossover budget instead of freeing it changes sigma and l by < 1 % (median).
**Adequacy** = exponential residuals white (runs p > 0.05) AND |misfit| < 3 dB at both 1.5 m and
1.0 m; "Bragg-only adequacy" drops the whiteness condition (on precise same-scan data no family
is white, as on the study lines).

## 3. Covariates

RACMO2.3p3 FGRN11 SMB + snowmelt (2009-2018, 11 km, zenodo 4013856), RACMO2.3p2 ZGRN11 10 m wind
1961-1990 climatology (zenodo 3368405), NSIDC-0533 MEaSUREs Greenland melt days (25 km, only
2010-2012 available); Antarctica RACMO2.3p2 ANT27 SMB / snowmelt / wind 2009-2019 (zenodo
7845736; melt days = proxy from snowmelt months). MAR (ftp.climato.be) timed out. Facies used
in the analysis: elevation/latitude rule (ELA 1600 m at 65 N -> 1000 m at 80 N; dry-snow line
2500 m at 66 N -> 2000 m at 78 N) with MEaSUREs melt > 30 d/yr flagging `wet_snow` inside the
percolation band; Antarctica shelf (d = 0 or h < 100 m) / coastal < 1500 m / interior, `_melt`
where RACMO implies > 30 d. (The sub-agent's melt-day override was inverted by no-data zeros on
peripheral ice and by the 2012 season in the interior; `covariates.csv` keeps its columns but the
facies column is recomputed in `analysis.py`.) Greenland facies counts (sites): ablation 132,
wet snow 89, percolation 52, dry snow 151; Antarctica: shelf 80, coastal 113 (+28 melt),
interior 82.

## 4. Family verdicts (`tables.md` by_stratum / by_facies)

| stratum (Tier 1 regime x elevation) | sites / site-yrs | best of G/E/PL % | Matern best % | nu p5/50/95 | H p50 | exp white % |
|---|---|---|---|---|---|---|
| GL interior > 2500 m | 46 / 168 | 7 / **76** / 18 | 40 | 0.08 / 0.51 / 1.29 | 0.15 | 68 |
| GL interior transects (Summit-CC, EGIG, NE) | 47 / 93 | 2 / **72** / 26 | 39 | 0.09 / 0.50 / 0.84 | 0.31 | 62 |
| GL interior 1500-2500 | 51 / 233 | 0 / **55** / 45 | 26 | 0.14 / 0.42 / 0.74 | 0.28 | 62 |
| GL interior 500-1500 | 31 / 137 | 1 / 31 / **69** | 17 | 0.20 / 0.40 / 0.67 | 0.36 | 45 |
| GL percolation belt > 2500 / 1500-2500 / 500-1500 / < 500 | 32 / 39 / 39 / 22 | 36 / 33 / 7 / 16 % exp, rest PL | 24 / 24 / 4 / 16 | 0.37 / 0.39 / 0.38 / 0.40 | 0.28-0.39 | 41 / 37 / 13 / 31 |
| GL margin > 2500 / 1500-2500 / 500-1500 / < 500 | 21 / 30 / 34 / 31 | 3 / 12 / 39 / 41 % exp; G 3 / 4 / 18 / 18 | 0 / 4 / 59 / 65 | 0.33 / 0.39 / 0.65 / 0.73 (p95 1.2-2.2) | 0.33-0.51 | 17-33 |
| AA grounded > 2500 / 1500-2500 / 500-1500 / < 500 | 20 / 62 / 81 / 65 | 75 / 17 / 27 / 22 % exp, rest PL | 40 / 9 / 22 / 19 | 0.46 / 0.35 / 0.38 / 0.45 | 0.31-0.38 | 55 / 34 / 43 / 38 |
| AA shelf / sea level | 75 / 170 | 15 / 30 / 55 | 28 | 0.13 / 0.59 / 2.50 | 0.46 | 51 |

By facies (GL): dry snow 60 % exponential (nu 0.45, p5-p95 0.08-0.99), percolation 59 %
(nu 0.42), wet snow 27 % (0.39), ablation 29 % exp + 10 % Gaussian (nu 0.51, p95 1.2). AA:
interior 23 % exp (nu 0.36), coastal 25 % (0.39), shelf 31 % (0.59).

Reading: the exponential is the preferred 3-parameter family only in the Greenland dry-snow /
upper-percolation interior (> ~2000 m) and at the few AA sites above 2500 m; everywhere else the
power law wins. But the Matern nu, which measures *how far* from exponential, sits at 0.35-0.5 in
every stratum below the margins: within 0.35-0.65 of 1/2 in 51 % (GL > 2000 m), 43 % (GL < 1200 m),
44 % (AA > 1500 m), 40 % (shelves); the per-fit nu uncertainty is 0.13-0.23. The interior
distribution is centred on 1/2 with a low tail (nu 0.1-0.3 = power-law-like, H 0.1-0.3); the
margin / shelf distributions are centred at 0.6-0.75 with a high tail (nu > 1: steeper than
exponential, i.e. smoother at the Bragg scales than their 10-30 m roughness implies).

## 5. Exponential (sigma, l) (`recommendation_by_stratum.csv`; maps `map_gl.png`, `map_aa.png`)

| stratum | sigma cm p5 / 50 / 95 | sigma_bl30 cm p50 | l m p5 / 50 / 95 | l at bound (300 m) % | S(1.5 m) ref dB p50 | S(5 m) dB | octave RMS cm 2-4 / 4-8 / 16-32 m | aniso 4-8 m |
|---|---|---|---|---|---|---|---|---|
| GL interior > 2500 | 3.5 / **5.0** / 6.9 | 4.5 | 1.5 / **7.3** / 19.5 | 0 | -62.4 | -46.7 | 1.5 / 1.9 / 2.2 | 0.96 |
| GL interior transects | 3.1 / 5.6 / 10.0 | 4.9 | 1.2 / 8.9 / 18.4 | 0 | -61.6 | -46.2 | 1.4 / 2.0 / 2.5 | 0.99 |
| GL interior 1500-2500 | 4.8 / 7.1 / 23 | 6.0 | 3.9 / 10.5 / 300 | 6 | -59.4 | -44.0 | 2.1 / 2.6 / 3.0 | 0.96 |
| GL interior 500-1500 | 6.1 / 11.8 / 67 | 8.4 | 5.9 / 17 / 300 | 15 | -56.6 | -42.0 | 2.6 / 3.0 / 4.6 | 0.97 |
| GL percolation belt > 2500 | 4.3 / 7.1 / 25 | 5.4 | 1.2 / 10.5 / 300 | 7 | -58.6 | -44.8 | 2.1 / 2.3 / 3.1 | 1.04 |
| GL percolation belt 1500-2500 | 4.7 / 9.4 / 45 | 7.2 | 2.5 / 11.3 / 300 | 15 | -57.0 | -42.5 | 2.6 / 2.9 / 4.4 | 0.89 |
| GL percolation belt 500-1500 | 6.6 / 15.5 / 61 | 9.1 | 4.9 / 18 / 300 | 26 | -56.0 | -41.6 | 3.1 / 3.1 / 5.8 | 0.79 |
| GL percolation belt < 500 | 10 / 23 / 250 | 11.3 | 3.5 / 23 / 300 | 19 | -52.0 | -38.1 | 4.0 / 3.8 / 7.4 | 0.61 |
| GL margin > 2500 | 9.1 / 25 / 119 | 13.4 | 2.3 / 13.7 / 279 | 7 | -50.8 | -36.9 | 5.6 / 5.1 / 10.9 | (0.1) |
| GL margin 1500-2500 | 9.0 / 25 / 177 | 13.4 | 2.5 / 19 / 300 | 23 | -52.9 | -38.0 | 7.2 / 4.8 / 10.0 | 0.55 |
| GL margin 500-1500 | 29 / 109 / 610 | 53 | 7.1 / 64 / 300 | 29 | -46.0 | -28.4 | 17 / 19 / 36 | 0.89 |
| GL margin < 500 | 37 / 428 / 1700 | 207 | 7.8 / 300 / 300 | 52 | -39.7 | -19.5 | 18 / 30 / 78 | 0.88 |
| AA grounded > 2500 | 6.6 / 10.4 / 15.4 | 9.1 | 1.9 / 11.2 / 16.1 | 0 | -55.5 | -41.2 | 3.1 / 3.5 / 4.6 | 1.04 |
| AA grounded 1500-2500 | 4.9 / 11.3 / 37 | 6.8 | 0.7 / 12.2 / 300 | 17 | -57.0 | -43.1 | 2.4 / 2.6 / 4.1 | 0.86 |
| AA grounded 500-1500 | 5.3 / 10.8 / 94 | 7.6 | 3.7 / 13.5 / 300 | 12 | -56.6 | -42.2 | 2.6 / 2.9 / 4.6 | 0.86 |
| AA grounded < 500 | 6.9 / 25 / 345 | 12.1 | 3.9 / 24 / 300 | 26 | -53.4 | -38.7 | 3.7 / 3.8 / 7.0 | 0.86 |
| AA shelf / sea level | 4.2 / 27 / 839 | 9.5 | 2.0 / 68 / 300 | 41 | -56.9 | -41.8 | 2.5 / 2.9 / 4.3 | 0.84 |

Ground-truth sites (site medians over years; `site_medians.csv`): Summit 4.9 cm / 9.8 m, nu
0.53, exponential best 78 % of years, Bragg-adequate 78 %, S(1.5 m) -62.7 dB; B26 5.1 / 6.9, exp
86 %; Camp Century 6.1 / 7.0, exp 85 %; B19 7.5 / 6.5; B29 4.5 / 4.7; B16 (one year, 2019)
3.5 / 12, nu 1.29, exp +10 dB; DYE-2 5.7 / 12.6, power law 80 % (exp -2.1 dB); FA13A 9.1 / 10.1,
PL 70 %; SE-Dome 5.9 / 8.2 (one year, exp -5.6 dB); EGIG T02 5.9 / 6.9 (exp 57 %), T05 6.0 / 7.8
(exp 86 %), T00 13.4 / 16.7 (PL); geikie mid 6.7 / 8.9 (exp 71 %; the 2014 line result 4.8 / 5.3
sits at its low end), geikie end 11.9 / 36; westcoast start / mid / end 10 / 39 / 24 cm, PL 83-93 %
(as in the line study); Getz start 49 cm / uncapped, mid 8.2 / 12.6, end 11.9 / 22; WDC06A
4.8 / 4.7, PL. The Greenland interior sigma of 5 cm / l 7-10 m is the geikie number reproduced
at ~100 sites; interior AA sits at 10-11 cm / 11-12 m (rougher, as ILATM2 suggested).

**Year-to-year partition** (`analysis.json`, additive site + year model at sites with >= 3 years,
dB): GL (112 sites, 1002 site-years) 20 log10 sigma: total 13.3, site 12.6, year 1.1, residual
4.0; S(1.5 m): total 9.4, site 8.0, year 2.2, residual 4.9; S(5 m): 10.0 / 9.2 / 1.5 / 3.8; log l:
13.5 / 9.5 / 3.1 / 9.2; nu: 0.32 / 0.19 / 0.08 / 0.25; misfit at 1.5 m: 4.9 / 2.8 / 1.7 / 3.8. AA
(93 sites) the same pattern (sigma site 10.2, year 0.8, resid 4.2 dB; nu site 0.29, resid 0.31).
Year effects on S(1.5 m) are within +-2 dB except 2009 (+4.3 GL, ATM4 noise) and 2010 (-3.9). So
sigma and S(k_B) are persistent site properties (site variance 80-90 %) with a +-4-5 dB
residual; the *family* (nu, misfit) is not: its residual exceeds its site component, i.e. the
exponential-vs-power-law verdict at a site changes from year to year. At the 183 repeat sites
the fraction of years that are Bragg-adequate is 0.44 / 0.61 / 0.78 (p25 / 50 / 75); 10 % of
sites are adequate every year, 2 % never.

**Along-ice-sheet variograms of site medians** (`analysis.json`): log10 sigma GL: 0.07 dex^2 at
7 km rising to the sill 0.35 by ~60 km (20 % nugget: sigma is spatially organised at the
10-100 km scale, the elevation / distance gradient); AA sill 0.24 by ~30 km. log l: 55 % (GL) /
50 % (AA) of the variance already at 7 km; nu and misfit: 45-55 % at 7 km and flat beyond
~30 km. So l and the family have essentially no spatial organisation above the site scale.

## 6. Where the exponential fails, and how

Adequacy (`tables.md`; maps `map_*.png` panels 4-6; `fig_misfit_vs_covariates.png`):

| region | site-yrs | reference exp / PL / Matern | Bragg-adequate (|misfit| < 3 dB at 1.5 and 1.0 m) | + white | misfit at 1.5 m when a non-exp family is the reference, p10 / 50 / 90 dB | at 1.0 m | over-predicts (+) % | nu p50 |
|---|---|---|---|---|---|---|---|---|
| GL > 2500 m (dry snow) | 363 | 34 / 33 / 33 | 71 % | 51 % | -4.6 / **-0.9** / +6.9 | -5.6 / -1.4 / +8.4 | 44 | 0.44 |
| GL 2000-2500 | 99 | 36 / 36 / 27 | 78 % | ~50 % | -3.3 / -1.2 / +4.5 | -4.0 / -1.5 / +5.5 | 40 | 0.44 |
| GL < 1200 m (ablation / lower percolation) | 379 | 10 / 46 / 45 | 63 % | ~20 % | -2.1 / **+1.6** / +8.8 | -2.7 / +1.9 / +10.7 | 62 | 0.57 |
| AA grounded > 1500 m | 203 | 15 / 73 / 12 | 75 % | 30 % | -3.5 / -1.2 / +4.9 | -4.3 / -1.9 / +4.1 | 35 | 0.36 |
| AA grounded < 1500 m | 330 | 12 / 67 / 21 | 76 % | 27 % | -4.0 / -1.5 / +3.3 | -4.8 / -2.0 / +3.6 | 27 | 0.40 |
| AA shelves | 164 | 21 / 46 / 32 | 60 % | 37 % | -2.9 / **+2.5** / +13.8 | -3.7 / +2.8 / +15.7 | 70 | 0.59 |

Per Tier 1 stratum the Bragg-only adequacy is 64-68 % in the GL interior strata and transects,
57-75 % in the percolation belt, 79-90 % in the two high margin strata (rough but in-band the
power law and the exponential with l > 30 m coincide), 52 % at GL margin 500-1500 m, **32 % at
GL margin < 500 m**, 64-70 % AA grounded, 52 % AA shelves. The strict score (with whiteness) is
45-53 % interior, 9-25 % belt / margin, 20-45 % AA.

Two failure modes with opposite signs:
1. **Interior (GL > 2000 m, AA grounded)**: the exponential *under*-predicts the wide-angle
   scatter by 1-2 dB (median) and up to 5 dB (p10) at 195 / 300 MHz: the surface is a shallow
   power law (H 0.15-0.35, nu 0.1-0.4), so beyond k ~ 1/l the true spectrum falls as k^-2.3..-2.7
   rather than the exponential's k^-3; the D(r) misfit inside the band is < 0.2 dB at 1-20 m, so
   the error is in the extrapolation, and it grows with frequency (-0.9 dB at 1.5 m, -1.4 at
   1.0 m, ~-2 at 0.75 m). A minority of years (nu > 0.8; 10-15 % of GL > 2500 site-years, e.g.
   Summit 2017 / 2019, B16 2019) go the other way by +7-10 dB: a smooth spring surface whose
   Bragg-band content is below the exponential tail.
2. **Margins (< 1200 m) and shelves**: the exponential *over*-predicts by +1.6 (GL) / +2.5 (AA
   shelf) dB median and +9 / +14 dB p90 at 1.5 m (more at 1.0 m); l runs to the bound (30-50 %
   of site-years) so sigma is the 30 m tilt (0.5-4 m) and the exponential's k^-3 tail carries
   too much power into the Bragg band relative to the measured k^-3.2..-3.6 (nu 0.6-0.8,
   H 0.5-0.7): crevassed / undulating surfaces that are smoother at 1-3 m than their large-scale
   amplitude implies. At 195 MHz the median error there is +2 dB with a +10 dB tail; at 300 MHz
   +3 / +13 dB.

Covariate dependence (GBT 5-fold CV R^2 and Spearman rho, `analysis.json` `gbt`): sigma is
predictable (R^2 0.59 GL, 0.33 AA, 0.62 pooled; slope rho +0.85, distance -0.81, elevation
-0.68 in GL); the misfit, nu and adequacy are not (R^2 <= 0.03; strongest rho: GL adequacy vs
distance to margin +0.38, vs slope -0.36; misfit vs MEaSUREs melt days -0.35, vs wind -0.21;
SMB -0.23). Elevation gradient of the GL site medians: adequacy 0.21 (< 500 m), 0.19, 0.24,
0.33, 0.47 (2000-2500), 0.43 (> 2500); nu 0.55, 0.46, 0.36, 0.40, 0.43, 0.48; median misfit +0.9
-> -1.4 -> 0 dB; l-capped fraction 40 % -> 3 %. AA: adequacy 0.38 (shelf), 0.28, 0.23, 0.51,
0.30, 0.45 with elevation; no monotonic trend.

## 7. Grouping: regimes or gradient?

GMM on the site medians (log sigma, log l, nu, misfit at 1.5 m; `clusters_*.png`,
`site_clusters.csv`): BIC keeps falling to k = 6; the "< 10 % of total drop" rule gives k = 4:
cluster 3 (404 sites, 58 % GL) sigma 7 cm, l 9 m, nu 0.39, misfit -0.6 dB, h 1800 m, 94 km from
the margin - the interior of both ice sheets; cluster 2 (168) sigma 24 cm, l 31 m, nu 0.42,
-1.3 dB, 16 km from the margin; cluster 0 (60) sigma 26 cm, l 7 m, nu 0.61, +2.3 dB, mixed;
cluster 1 (94) sigma 73 cm, l at the bound, nu 0.61, +2.5 dB, 6 km from the margin, shelves and
GL margin < 500 m. The clusters separate on amplitude and l (i.e. on distance to the margin /
slope), not on the family: nu is 0.39-0.61 across them and the misfit changes sign only in the
two small margin clusters. Facies reproduces the interior cluster at 65-75 % (dry snow, AA
interior / coastal, GL percolation) and nothing else. Conclusion: there is a **gradient in
amplitude** (sigma, S(k_B): x 10 from interior to margin, organised at 30-100 km, 80-90 %
persistent) and **no regime boundary for the family**: the exponential-vs-power-law verdict
varies site to site and year to year with a ~50 % nugget, centred on nu ~ 0.45 inland and
drifting to nu ~ 0.6-0.7 only in the outermost margin / shelf band where l is unconstrained
anyway. The two "regimes" of the 5-line study (geikie exponential vs westcoast self-affine) are
the two halves of one nu distribution, not two populations.

## 8. Recommendation for the simulator's exponential option

Per-stratum table for `config/roughness/` (exponential ACF, band 1-30 m, sigma = fit parameter,
sigma_bl30 = band-limited RMS < 30 m for reference; uncertainty = p5-p95 of site-years, which
is dominated by the site-to-site spread; year-to-year on a given site is +-2 dB on sigma,
+-4-5 dB on S(1.5 m)):

| stratum | sigma cm (p5-p95) | l m (p5-p95) | S(1.5 m) dB | verdict | Matern nu if not exponential |
|---|---|---|---|---|---|
| GL dry snow / interior > 2500 m (incl. Summit, NGT, B26, EGIG upper) | 5.0 (3.5-6.9) | 7.3 (1.5-19.5) | -62 | **use exponential**; expect -1 dB (to -5) at 195 MHz, -1.5 dB at 300 MHz | 0.45 (0.3-0.6) |
| GL interior transects | 5.6 (3.1-10) | 8.9 (1.2-18) | -62 | use exponential | 0.50 |
| GL interior 1500-2500 m | 7.1 (4.8-23) | 10.5 (3.9-300) | -59 | use exponential | 0.42 |
| GL percolation belt > 1500 m | 7-9 (4.5-45) | 11 (1.2-300) | -57..-59 | usable; power law (H 0.3-0.4) fits better in 2/3 of years; exp under-predicts 1-1.5 dB | 0.38 |
| GL percolation belt / interior 500-1500 m (westcoast, DYE-2, FA13A) | 12-16 (6-67) | 17-18 (5-300) | -56..-57 | marginal: PL wins 70-90 %, l uncapped in 15-26 %; use PL H 0.36-0.39 or Matern | 0.38-0.40 |
| GL margin 1500-2500 / > 2500 m (crevassed high outlets) | 25 (9-180) | 14-19 | -51..-53 | in-band both fit; exp -1.2 dB at 1.5 m; l uncapped 7-23 % | 0.33-0.39 |
| GL margin 500-1500 m | 109 (29-610) | 64 (7-300) | -46 | do not rely on it: +2 dB median, +10 dB p95 over-prediction; PL H 0.47 / Matern | 0.65 (0.3-1.2) |
| **GL margin < 500 m** | 428 (37-1700) | at bound | -40 | **do not use**: +4 / +5 dB median, +25 dB p95 at 1.5 / 1.0 m | 0.73 (0.45-2.2) |
| AA grounded > 2500 m | 10.4 (6.6-15) | 11.2 (1.9-16) | -56 | use exponential (75 % exp-best, 65 % adequate) | 0.46 |
| AA grounded 500-2500 m (Siple, PIG / Thwaites catchments, Peninsula plateau) | 11 (5-40..95) | 12-14 (1-300) | -57 | usable with a -1..-2 dB bias; PL H 0.31 fits better in 70-80 % | 0.35-0.38 |
| AA grounded < 500 m | 25 (7-345) | 24 (4-300) | -53 | marginal; l uncapped 26 % | 0.45 |
| **AA shelves** | 27 (4-840) | 68 (2-300) | -57 | do not use where l is at the bound (41 %): +2.5 dB median, +14 dB p90 | 0.59 (0.13-2.5) |

How to carry it: (a) an exponential ACF with the interior table is adequate to +-3 dB at
195 / 300 MHz for 2/3 of site-years everywhere inland of the ablation zone, with a systematic
-1 to -2 dB under-prediction that could be absorbed as a bias term; (b) for the outer margin
and shelf strata the exponential has no defensible l and over-predicts by 2-5 dB (tails to
+10-25 dB) - use the power-law / Matern option there (H 0.45-0.55, nu 0.6-0.75) once it exists,
or cap l at 30 m and accept the error; (c) the amplitude should be site- or stratum-specific
(x 10 gradient), the family need not be: a single Matern nu ~ 0.45 would beat the exponential
by ~1 dB inland and by 2-4 dB at margins, but is not required inland.

## 9. Caveats

- Pair-lag floor 1 m: 1.0 and 0.75 m Bragg values are extrapolations of the fitted forms below
  the band (0.5 and 0.7 octaves); 1.5 m sits at the band edge (lag 0.75 m) and 5 m inside it.
- The "vs best" misfit is 0 when the exponential wins; use the "vs Matern" columns in
  `tables.md` for a continuous measure (interior median -0.4 to -1.1 dB, margins +1.2 to +4.4).
- Cross-scan pairs with a free nugget on 60 % of site-visits (ATM4 / ATM5 and the 2009-2012
  Qfit years); 2010 AA (12 cm crossover noise) and 2009 GL (7.5 cm) are the noisy years and
  carry +-4 dB year effects; the interior 1-2 m octave is a bound.
- Greenland facies is an elevation / latitude rule (MEaSUREs melt days cover 2010-2012 only and
  the 25 km grid misses peripheral ice); Antarctic melt days are a RACMO-month proxy; wind is a
  1961-1990 climatology. Facies sampled at 11-27 km grids.
- Phase 2 stratified sites carry one year (budget); the year-to-year partition rests on the
  200 repeat + 30 ground-truth sites, which over-represent the OIB repeat lines (SW / NW
  Greenland, Siple Coast, Amundsen sector). The Summit-Camp Century transect lost 15 dates to
  missing ILATM1B granules and has 19 sites instead of ~25. Antarctic coverage remains West
  Antarctica / Peninsula / Transantarctic outlets; East Antarctic plateau unsampled.
- The whiteness condition is too strict on precise same-scan data (no family is white), which
  is why the strict adequacy is ~half the Bragg-only adequacy; the numbers to use are the
  Bragg-only ones plus the misfit magnitudes.
- Surface only; the buried-layer roughness question (plan section 5) is untouched.
