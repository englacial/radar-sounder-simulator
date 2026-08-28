# Plan: ice-sheet-wide surface roughness regimes from OIB ATM, to set the simulator's roughness representation

Goal: decide how sub-facet roughness (surface and firn layers) should be
represented in the simulator going forward — one law, a few regional
regimes, or a per-site table — by mapping the 1–100 m roughness statistics
across Greenland and Antarctica from the OIB ATM archive and testing for
groupings, then choosing the representation that the data support.

Inputs already in hand: per-line ATM analysis (claude_notes/atm_roughness/,
5 lines): coastal Greenland and Getz are self-affine (H 0.35–0.45, no outer
scale) while geikie dry snow is exponential with l ≈ 5 m; the Gaussian family
fits nowhere; block statistics decorrelate in 1–4 km (Greenland) or 10–15 km
(Antarctic) and shift 2–5 dB year to year. The open questions are whether
these are two regimes with a mappable boundary, what drives them, and how
they relate to the buried-layer roughness that C&S 2020 showed dominates
firn clutter.

## 1. Scope and two-tier data design

The ATM archive is ~10 TB of ILATM1B; we do not need it all.

**Tier 1 — coarse screen, whole archive (cheap):** ILATM2 (Icessn) gives per
250 m platelet RMS roughness and slope for every OIB flight 2009–2019, both
ice sheets, ~2 GB total. Use it to map the 250 m-scale roughness field and
its spatial structure, flag regime candidates, and select where Tier 2 goes.
Note ILATM2 RMS mixes scales 1–250 m; it is a screen, not the answer.

**Tier 2 — Bragg-band statistics, stratified sample (the answer):** ILATM1B
point clouds for ~300–500 sampling sites of 5 km each (≈ 1–2 GB per 50
sites), chosen by stratified random sampling over the covariates in §3 so
every regime candidate has ≥ 20 sites, plus every site with radar ground
truth (§5). Per site: the existing pipeline (structure function D(r) from
point pairs, noise from crossovers, 2-D anisotropy, family fits over 1–50 m,
octave RMS 1–64 m, S(k) at 5/1.5/1.0/0.75 m). Reuse
claude_notes/atm_roughness/atm_roughness.py; add a site-list driver and a
results database (one parquet row per site-year).

**Repeat coverage:** OIB re-flew many lines yearly (Greenland spring
campaigns 2009–2019; Antarctic Nov 2009–2018). Take every repeat at the Tier
2 sites — the year-to-year spread is part of the regime description and
tells us which bands are persistent topography vs seasonal.

## 2. Statistics to carry per site-year

- Octave RMS 1–2 … 32–64 m (band-limited, noise-subtracted)
- S(k) at the Bragg wavelengths for 60/150/195/300/400 MHz at 30° (plus 20°/40°)
- Family verdict (Gaussian / exponential / power-law) with ΔBIC, whiteness
- Power-law H and amplitude; if a bend-over exists, the outer scale l_out
- Anisotropy (along/cross-track ratio 2–16 m; scan direction vs prevailing wind)
- Noise floor, point density, swath, QC flags
- Covariates (§3) sampled at the site

## 3. Covariates and hypotheses

Candidate drivers of the surface roughness regime:
- **Glacier facies**: dry snow / percolation / wet snow / ablation (MAR or RACMO melt days and SMB; Greenland facies masks); hypothesis: dry-snow sastrugi surfaces have an outer scale (exponential-like), melt/ablation and crevassed surfaces are self-affine.
- **Wind**: mean 10 m wind speed and directional constancy (RACMO/MAR, or ERA5); sastrugi amplitude and orientation.
- **Accumulation rate** (RACMO/MAR): high-accumulation coastal zones bury roughness faster.
- **Elevation, surface slope (100 m–1 km), distance to coast/grounding line, ice velocity** (MEaSUREs): crevassing and flow-stripe texture.
- **Season/date** of the flight (spring vs autumn).
Each site-year row carries these; the grouping analysis (§4) tests which explain the variance.

## 4. Grouping analysis

1. Unsupervised: cluster the site-years in the space (octave RMS vector, H, l_out, anisotropy) — k-means/GMM on log-RMS + H, with the number of clusters chosen by BIC; map the clusters; inspect whether they form spatially contiguous regions.
2. Supervised: regress log S(k_B) and H on the covariates (gradient-boosted trees + a plain linear model for interpretability); rank drivers; test whether facies alone reproduces the clusters.
3. Spatial structure: variogram of the site statistics across the ice sheet — at what distance do sites decorrelate (the earlier 1–15 km line-scale result vs a regional 100 km scale)? This sets whether a regime map or a per-site table is the right product.
4. Temporal: for repeat sites, partition variance into site (persistent), year, and residual; report which bands are stable.
Deliverable: a regime map (raster or polygon set per ice sheet) with per-regime PSD parameters and uncertainty, or — if no clean grouping — a gridded S(k_B) field with its spatial correlation length.

## 5. Radar ground truth and the layer question

ATM measures the surface; the simulator also needs the roughness of buried density layers, which C&S showed controls firn clutter and which our HAPS/B26 study found dominates the bed-delay clutter at 14 km.
1. **Surface check with radar**: at every Tier 2 site that has a coincident MCoRDS/AR frame, derive the off-nadir angular scattering function from the along-track Doppler spectrum (C&S §II) and compare with the ATM S(k) at the same Bragg wavelengths. Also use the RSSNR store (Greenland/Antarctic season-wide surface-return statistics) as a coarse regional cross-check of surface backscatter vs the ATM roughness map.
2. **Layer roughness hypothesis**: buried layers inherit the surface roughness spectrum at burial, smoothed by densification — i.e. the layer S(k) at depth z is the surface S(k) of that regime attenuated at high k. Test with the C&S method: Doppler spectra vs depth at sites in each regime (AR 750 MHz where available for the fine band; MCoRDS for the 1.5 m band), plus firn-core density profiles where they exist (B26, Camp Century, DYE-2, Summit, EGIG line; SUMup for Antarctica). This yields a per-regime layer roughness spectrum and its depth decay — replacing the single Fig. 11 profile.
3. Where no radar exists, carry the surface regime as the prior for the layers with the depth decay fitted at the calibrated sites.

## 6. Feeding the simulator (decision this investigation makes)

The outcome decides among:
- **(a) A few regimes, each a family with parameters**: e.g. dry-snow interior = exponential (σ, l, depth decay), coastal/melt = power-law (A, H) with a Bragg-band cutoff. Implemented as a regime attribute on the line YAML (`surface_roughness: {regime: dry_snow_interior}`) plus a per-regime table; kernel needs the tabulated-ACF branch (roughness.py `area_only` path with interpolated W_m(k), plan B2) so exponential and power-law forms are both exact.
- **(b) No clean grouping**: a gridded S(k) field per ice sheet; the line YAML points at a site-specific PSD; same kernel change.
- **(c) One law suffices** (unlikely from the 5-line result): keep a single family with per-line amplitude.
In every case: retire the Gaussian fixture; make the roughness law, its provenance, and its uncertainty explicit in run_config; carry layer roughness as regime × depth.

## 7. Effort and order

1. Tier 1 ILATM2 pull + roughness/slope map, both ice sheets, facies/wind/SMB covariate stack: 2 days.
2. Site selection (stratified) + Tier 2 pull (~5–10 GB) + batch run of the existing pipeline: 3 days incl. compute.
3. Grouping analysis, regime map, temporal partition: 2 days.
4. Radar Doppler cross-check at ~10 sites and the layer-roughness depth test at 3–5 core sites: 3–4 days (needs OPR frame access for Doppler; AR data for the fine band).
5. Kernel change to tabulated ACF (B2) + validation + regime plumbing: 1 week, in parallel with 3–4.
Total ~3 weeks elapsed; the regime map and the "one law / regimes / per-site" decision arrive after step 3 (~1 week).

## Caveats

- ATM samples the flight lines, which over-represent margins and outlet glaciers; the stratified sampling and the interior Antarctic (few flights) need REMA/ArcticDEM strips or ICESat-2 for the ≥ 20 m band as a fallback.
- Sub-metre scales: ATM's ~1 m spot limits the 0.75 m (400 MHz) point to a half-octave extrapolation; where the family is power-law that is fine, where an outer scale sits near 1 m it is not.
- Surface changes between the ATM year and any future mission; regimes are the durable product, absolute values are seasonal.
- Layer roughness from Doppler inversion is model-dependent (C&S used S-IEM with its own ACF); we should fit the same three families there as for the surface before assuming a form.
