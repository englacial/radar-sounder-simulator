# Handoff Summary: Radar Sounder Simulation for Ice-Penetrating Radar

**Purpose of this document.** Context package for a future agent continuing work toward an open-source radar sounder simulator for Antarctic/Greenland mission and instrument design. It summarizes the fidelity hierarchy of simulation approaches, key references, candidate benchmarks, and the identified open-source opportunity. It deliberately avoids prescribing implementation choices.

---

## 1. The fidelity hierarchy

Radar sounder simulation divides into five tiers by physics captured and computational cost, plus hybrid chains that combine them.

**Tier 1 — Radar equation / link budget.** Closed-form SNR/signal-to-clutter budgets with terms for spreading, ice attenuation, antenna gain, and statistical clutter. Answers frequency-band, power, antenna-gain, and altitude feasibility questions in seconds. No imaging, no geometry.

**Tier 2 — 1-D layered media.** Plane-wave/propagator-matrix propagation through a stratified firn/ice/bed column, convolved with the waveform. Captures layer interference, firn gradients, attenuation, dispersion, and (with extensions) birefringence. Answers bandwidth/vertical-resolution and layer-detectability questions. Strictly nadir; no clutter.

**Tier 3 — Ray tracing / geometric optics / geospatial survey emulators.** Snell's-law propagation and DEM-based echo-source geometry. Captures refraction, off-nadir arrival geometry, and survey measurement bias (e.g., bed-elevation under-measurement). Infinite-frequency limit: no phase, no interference, no diffraction.

**Tier 4 — Facet-based DEM-driven simulators (the workhorse tier).** Real DEMs tessellated into facets; each facet's contribution is computed under the Kirchhoff/tangent-plane approximation and binned by two-way traveltime. This tier splits internally into two families, and the split matters more than the tier boundary:

- **Incoherent (power-summing):** facet powers in dB summed into range bins. Correct arrival-time geometry and rough relative power; no phase. Sufficient for clutter *identification* and survey planning. All currently open tools live here.
- **Coherent (field-summing):** complex fields with phase summed via the Stratton–Chu integral under the linear phase approximation (LPA), which permits facets several wavelengths across. Phase enables interference structure, speckle, Fresnel-zone effects, roughness-relative-to-wavelength behavior, and SAR-compatible along-track phase histories. Required for frequency/bandwidth trades, post-focusing clutter competition, and echo statistics. All published coherent tools are closed or on-request.

The defining physics distinction between Tier 3 and Tier 4 (and between the two Tier-4 families) is whether the electromagnetic phase is tracked: rays vs. waves.

**Tier 5 — Full-wave solvers (FDTD, PSTD, MoM, FEM).** Direct Maxwell solution; captures volume scattering, diffraction, and multiple scattering the Kirchhoff approximation misses. Cost scales steeply with domain size in wavelengths; practical only for 2-D or small 3-D scenes at HF/VHF. Role: small-scene detectability studies and validation referee for lower tiers.

**Hybrids and end-to-end chains.** Documented patterns combine facet/ray large-scale scenes with full-wave small-scale inserts, apply SAR focusing to simulated raw data (raw simulator output underestimates detection performance without focusing), and generate radargram databases across parameter combinations for invertibility studies (Trento-school work: Cortellazzi, Sbalchiero, Thakur & Bruzzone).

Known limits even at coherent-facet parity: scalar fields (no polarization/birefringence), no volume scattering, single-bounce interfaces, and no tractable path to thousands of explicit englacial layers (stratigraphy requires a Tier-2 hybrid).

---

## 2. Key references

### Coherent facet method lineage
- **Nouvel, Herique, Kofman & Safaeinili (2004), Radio Science, DOI 10.1029/2003RS002903** — founding square-facet coherent method (MARSIS); analytic LPA facet integral.
- **Berquin et al. (2015)** — triangular facets, vectorial Huygens–Fresnel (the polarimetric reference point).
- **Gerekos et al. (2018), IEEE TGRS 56(12), DOI 10.1109/TGRS.2018.2851020** — coherent *multilayer* generalization: Snell ray tracing through the layer stack + per-facet LPA; validated against SHARAD and Kaguya LRS. The de facto specification of the target capability.
- **Gerekos, Haynes, Schroeder & Blankenship (2023), Radio Science 58(6), DOI 10.1029/2022RS007594** — analytic phase response of a *rough* rectangular facet (Gaussian roughness/correlation), giving coherent and incoherent power at arbitrary bistatic angles. Fixes the "too coherent" bias of coarse-DEM simulations (excess specular, deficient diffuse power).
- **Ferro & Bruzzone (2013), IEEE TGRS** — incoherent facet formulation (basis of SOPA's simulators; the cheap baseline mode).
- **Ilyushin et al. (2017), CLUSIM, Radio Science, DOI 10.1002/2017RS006265** — continuous piecewise-surface alternative; catalogs facet-gridding (Bragg) artifacts to test against.

### The simc / cluttergram lineage
- **Holt et al. (2006)** — echo-source discrimination in airborne sounding, Dry Valleys, Antarctica (method origin).
- **Choudhary, Holt & Kempf (2016), IEEE GRSL 13(9), DOI 10.1109/LGRS.2016.2581799** — closest peer-reviewed methods citation for what simc computes.
- **Christoffersen & Holt (2020), LPSC 51, abstract #2881** — the generalized multi-planet simulator description (no journal paper exists).
- **Christoffersen, Holt, Kempf & O'Connell (2022)** — SHARAD Surface Clutter Simulations PDS4 archive (`urn:nasa:pds:mro_sharad_simulations`, DOI 10.17189/nbdh-2k53); user's guide at pds-geosciences.wustl.edu/mro/urn-nasa-pds-mro_sharad_simulations/document/userguide.pdf. Note: this documents the archived *products*, not how to run the software. Note also two SHARAD simulator lineages exist (UT Austin and U. Arizona); the UA products are the ones in the PDS.

### Verification anchors
- **Haynes, Chapin & Schroeder (2018)** — closed-form geometric power fall-off / first-Fresnel-zone nadir power across roughness regimes (the 2023 rough-facet paper validates against it).
- **Grima et al. (2014)** — radar statistical reconnaissance; coherent/incoherent power partition and echo statistics.
- **Warren, Giannopoulos & Giannakis (2016)** — gprMax (FDTD), the full-wave cross-validation referee.
- **Ulaby, Moore & Fung (Vol. II); Ogilvy (1991)** — Kirchhoff/tangent-plane validity limits.

### Application, validation, and design exemplars
- **Pierce et al. (2024), The Cryosphere 18, DOI 10.5194/tc-18-1495-2024** — Gerekos simulator adapted to UTIG MARFA (60 MHz) for subglacial hydrology; documents the practical pain points: aperture length limited by compute, and BedMachine's ~500 m smooth bed as the off-nadir geometry limit. Template for an Earth validation study; derived data on Zenodo (10.5281/zenodo.8165343).
- **Culberg & Schroeder (2020), IEEE TGRS 58(9), DOI 10.1109/TGRS.2020.2976666** — Tier-1 exemplar: firn-clutter constraints on orbital ice-sounder design (HF/low-VHF favored; noise-limited).
- **Bartlett et al. (2020), Annals of Glaciology 61(81), DOI 10.1017/aog.2020.35** — Tier-3 exemplar: geospatial survey simulation quantifying bed-elevation under-measurement bias.
- **Lei et al. (2020, IEEE TGRS, DOI 10.1109/TGRS.2019.2960751; 2022 validation vs. SHARAD)** — 2-D PSTD full-wave sounding simulator (open code, MATLAB).
- **Mission-design precedents:** RIME 9 MHz frequency selection via loss-and-clutter modeling (Bruzzone et al.); REASON interferometric cluttergrams via a Stratton–Chu simulator generalized to multi-antenna sounders (Gerekos et al., 2022 abstracts); Devon Ice Cap canyon study (reflectivity anomalies vs. backscatter simulations).

### Input physics
- **MacGregor et al. (2015), JGR-ES, DOI 10.1002/2014JF003418** and **Matsuoka et al. (2012)** — Greenland/Antarctic attenuation models (beyond constant dB/m).
- **Fujita et al. (2006)** — ice dielectric anisotropy/birefringence propagation.
- **DEMs:** REMA and ArcticDEM (surface, meter-class), BedMachine (bed; smooth mass-conserving interpolation, ~500 m effective — the key geometric input limitation for bed clutter).

---

## 3. Software landscape (as of mid-2026)

**Open, Tier 4, incoherent:**
- **simc** (github.com/lpl-tapir/simc; Christoffersen & Holt, U. Arizona TAPIR). Python, pip-installable CLI with .ini config and a supported editable-install workflow. Surface-only, multi-planet. Cleanest extensibility starting point; geometry plumbing reusable.
- **Snow4Flow survey_planning** (github.com/Snow4Flow/survey_planning; MacGregor, NASA/GSFC). Notebook wrapper around a SIMC-derived `combogram` module. The only open tool with a bed interface: air/surface (REMA or ArcticDEM via PGC STAC) + ice/bed (BedMachine), dipole antenna pattern, nadir Fresnel coefficients, constant attenuation. Tuned to CReSIS frames; research script, not a library.
- **SOPA dRSsim/sfRSsim** (github.com/adamoferro/sopa; Ferro). Python 3, includes the SOFA SAR focuser; square-facet cos^exp weighting; almost no documentation; SHARAD-specific readers.

**Closed or on-request, Tier 4, coherent:** Gerekos 2018/2023 multilayer simulator (reported MATLAB, GPU status publicly unconfirmed — verify with authors); Berquin; CLUSIM; Trento (Thakur/Sbalchiero) tools. The RadSPy authors' published complaint that most sounder simulators are instrument-specific and not publicly available remains accurate for the coherent family.

**Open, other tiers:** gprMax, MEEP, openEMS (Tier 5 FDTD); JPL/Caltech PSTD (github.com/leiyangleon/PSTD, MATLAB, Tier 5); empymod (github.com/emsig/empymod, Tier 2); ImpDAR and CReSIS Toolbox / Open Polar Radar (processing and validation-data access, not simulators).

**The gap:** the open-source × coherent quadrant of Tier 4 is empty.

---

## 4. Candidate benchmarks and validation resources

**Analytic ground truth:** point-target range response (resolution/sidelobes); flat-plate and first-Fresnel-zone nadir power vs. Haynes 2018 across roughness regimes; speckle statistics and coherent/incoherent power partition vs. Grima 2014; Rayleigh-criterion behavior of specular vs. diffuse returns; convergence of the LPA facet integral against brute-force sub-wavelength facet summation (a stronger correctness check than transcribing published formulas).

**Cross-code:** gprMax on small 2-D/3-D canonical scenes (refraction, layer interference, rough interfaces); empymod for nadir layered-column returns; parity with simc in incoherent mode on shared inputs; the PDS SHARAD cluttergram archive as fixed reference outputs for a SHARAD configuration; the Gerekos 2023 internal test (coarse DEM + analytic roughness vs. oversampled explicit-roughness realizations) automated as a regression benchmark.

**Real data (Earth advantage — all open):** Open Polar Radar / CReSIS MCoRDS frames (the Snow4Flow notebook already targets these); UTIG MARFA lines with the Pierce et al. 2024 study as a published reproduction target; SHARAD radargrams + archived cluttergrams for planetary heritage checks. Inputs: REMA, ArcticDEM, BedMachine, MacGregor/Matsuoka attenuation fields.
Open Polar Radar data can be access through xOPR

**Reference scenario used for feasibility numbers:** 10 km along-track segment, 14 km platform altitude, 60 MHz (MARFA-class), 3 km ice: clutter-competing swath ≈ ±13 km cross-track; ~2×10⁷ facets at one-wavelength facet size; ~8×10³ traces at λ/4 spacing; ~10¹¹ facet–trace evaluations ≈ 10–100 TFLOP-equivalents.

---

## 5. The open-source opportunity

1. **Empty quadrant, ready market.** Every open Tier-4 tool is incoherent; every coherent tool is unavailable. Instrument-design questions that need phase (frequency/bandwidth trades, post-focusing clutter competition, echo statistics, interferometric concepts) currently require collaboration access or reimplementation.
2. **The physics is fully published.** Closed-form LPA facet integrals (square, triangular, rough) plus the multilayer refraction scheme are all in the open literature; nothing proprietary blocks a reimplementation.
3. **Compute is no longer the barrier.** The facet–trace summation is embarrassingly parallel, compute-bound, transcendental-heavy, and memory-light — near-ideal GPU work (expect ~10–50× over a well-vectorized CPU node; orders of magnitude over naive interpreted code). The reference scenario runs in minutes on a single modern datacenter GPU (~a dollar of cloud time); a 1000 km campaign line is a modest bill. Work scales roughly as f³ (facets f², traces f), so VHF/UHF is ~35× HF cost — the argument for GPU-native design. "Supercomputer problem" framing in the literature reflects implementations, not the method.
4. **Earth is the easy validation target.** Open radargrams (OPR/CReSIS/UTIG), meter-class surface DEMs, and published attenuation models make terrestrial validation more accessible than the planetary context these simulators were born in.
5. **Reference-implementation opening.** simc runs on an abstract plus a README; Gerekos has papers but no code. A tool shipping both code and a methods paper (e.g., Geoscientific Model Development, Radio Science, Annals of Glaciology) with the verification suite as supplementary material would become the citable standard.
6. **Design-shaping findings (challenges, not prescriptions):** the multilayer refraction two-point ray solve through rough interfaces is the numerically delicate core; sub-facet roughness statistics (especially for the bed) are required inputs and poorly constrained; bed-DEM fidelity (BedMachine smoothness) caps off-nadir bed realism regardless of simulator quality — geometric uncertainty should be expressed, not hidden; validation demands emulating the instrument and processing chain (waveform, antenna incl. installation effects, stacking, focusing); and scope honesty is needed about scalar fields, absent volume scattering, and the 1-D-hybrid path for englacial stratigraphy.
