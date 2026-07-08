# Implementation plan: stage 2 (coherent simulation)

Per `docs/overview.md` stage 2: Stratton–Chu / coherent field summation, benchmarked against Haynes et al. 2018 for smooth and rough surfaces, plus a coherent extension of the xOPR comparison. Builds on the stage-1 codebase (committed at 5d1274e). **Status: complete (2026-07-07 evening). M8–M13 done: 83 CI + 14 integration tests green; report has 14 cases across the three groups, 0 fails. Key measured results: parity re-baselined at 2.0554–2.0613 (predicted 2·(1/k)²); LPA breakdown at large-facet × near-nadir, 5% at L≈0.23√(λr); phase precision λ/664 at 20 km/195 MHz via reference-range subtraction (orbital would need f64); Haynes absolute constants to 0.1% (parameter-free), coherence-loss ≤0.25 dB coherent-regime residual, speckle Rayleigh/exponential confirmed; xOPR coherent on both frames (facets at β=0.5 ⇒ ~26% worst-case nadir LPA error, recorded; coherent is specular-dominated vs incoherent diffuse — measured radargram sits between; Gerekos-2023 rough facet is the natural follow-on). docs/coherent_simulation.md written. Not yet committed.**

## Physics approach (what "coherent" means here)

Per facet, instead of a power `(A·cosθ)²/r⁴`, the kernel accumulates a **complex field** contribution at the carrier: amplitude from the analytic linear-phase-approximation (LPA) facet integral for a rectangular facet (Nouvel et al. 2004; Gerekos et al. 2018 monostatic case — closed-form sinc×sinc in the facet-plane phase gradient), scalar Fresnel reflection coefficient, spherical-spreading `1/r²`, and phase `exp(−j·2k·r)`. Contributions are complex-summed into the same `floor((twtt−t0)/dt)` fast-time bins (delta-pulse response at carrier `f0`; waveform/chirp convolution stays deferred to stage 4). Scalar fields only — no polarization — per the documented scope.

Output per `docs/output.md`, already specified: `field` (complex64) + precomputed `power = |field|²`, `combine()` applying field-level summation across `side`, `save()` handling complex via h5netcdf `invalid_netcdf` with a strict real/imag split option.

## Two physics constraints that shape the design

1. **Facet size vs Fresnel zone.** LPA drops the quadratic phase term; the error across a facet of size L at range r is ~`k·L²/(4r)`. At MCoRDS (195 MHz, λ ≈ 1.54 m) with 32 m DEM posting and 500 m AGL, that's ~2 rad — LPA is invalid; the facet must be ≲ half a Fresnel-zone radius (`√(λr/2)` ≈ 19 m there). Scene building therefore gains a facet-subdivision path (bilinear DEM interpolation to a target facet size) plus a loud validity check in the coherent path (warn/error when `L > β·√(λr_min)`, β configurable ≈ 0.5). Subdivision manufactures smoothness the real surface doesn't have — acceptable for envelope/speckle statistics, stated in output notes. High-altitude cases (the actual mission focus) are much gentler: at 14 km AGL, Fresnel radius ≈ 100 m > any DEM posting we use.
2. **Phase precision in float32.** Phase needs range to ~λ/100. Float32 eps at r = 20 km is ~2.4 mm — marginal at 195 MHz (λ/100 = 15 mm), broken for orbital ranges. Candidate fixes, decided by measurement in M10: (a) compute the range→phase path in float64 (JAX x64 for that op only; cheap on CPU, slow on consumer GPU), or (b) per-trace reference-range subtraction: accumulate phase from `2k·(r − r_ref)` with `r_ref` the f64 nadir range, keeping magnitudes small in f32. A dedicated f32-vs-f64 phase-error test gates the choice.

## Milestones (each ends green: `uv run pytest`)

### M8 — Config + geometry extensions
- `RadarConfig` gains `f0` (Hz; `wavelength` derived). `SimConfig.mode="coherent"` becomes runnable end-to-end only at M11.
- **Media config**: user-facing dielectric spec is relative permittivity per medium, not reflection coefficients — a `MediumConfig` (or ordered media list, anticipating stage-3 layers) with stage-2 defaults `air ε_r = 1`, `ice ε_r = 3.17`. Fresnel Γ is *derived* at the interface (normal-incidence scalar: `Γ = (√ε₁ − √ε₂)/(√ε₁ + √ε₂)` ≈ −0.281 for air→ice); this matches docs/overview.md ("dielectric permittivity specified between each layer") and keeps the config physical while the kernel stays scalar.
- `scene.py`: rectangular-facet builder — per DEM cell: center, two edge vectors (mean-plane fit to the 4 corners), unit normal, area; plus the subdivision path (target facet size → bilinear DEM refinement) and the Fresnel-zone validity check helper.
- **Decision D2-1 (revised per review)**: rectangular facets **throughout** — the incoherent kernel switches to the same rect facets, so geometric optics (and the stage-3 multilayer ray solve) has exactly one path, and the M11 cross-kernel test compares identical facets. The triangle builder is removed once parity is re-baselined.
- **Parity re-baseline**: simc fixtures are unchanged; our tessellation granularity changes the constant power ratio by exactly 2 (one rect of area A yields `(A·cosθ)²` vs two triangles' `2·(A/2·cosθ)²`), so expected median ratio becomes ≈ 2·(1/k)² ≈ 2.06. Rerun all five parity scenes; shape metrics expected to move only marginally (cell triangle pairs are near-coplanar on the test scenes); update recorded values in docs/incoherent_simulation.md and the report. Any shape-metric failure is investigated, not tolerated away.
- Tests: rect facet area/normal vs triangle-pair aggregate before the triangle path is deleted (agreement on smooth scenes, documented divergence on rough); plane-fit residual bounded; area/normal *convergence* with posting on curved surfaces (replaces stage-1 triangle exactness tests); subdivision converges as target size shrinks.

### M9 — Brute-force reference simulator (verification before the kernel)
- Pure NumPy float64 reference: sub-wavelength point scatterers (λ/10 spacing) with exact spherical phase, direct complex summation. Tiny scenes only — this is ground truth, not a tool.
- Analytic anchors for the reference itself: point-target response; flat-plate return vs the image-method solution (Haynes Eq. 19–21); Fresnel-zone oscillation of a growing disk (Haynes Eq. 15 — `|I|²` oscillates with `cos(ka²/h)`).
- The handoff notes call LPA-vs-brute-force the strongest correctness check available (stronger than transcribing published formulas); this milestone builds that referee first.

### M10 — Coherent JAX kernel
- `kernels/coherent.py` reusing `kernels/geometry.py` ranges/binning: LPA rect-facet amplitude, Fresnel Γ derived from the configured permittivities (scalar, normal-incidence for now), `exp(−j2kr)/r²`, complex `segment_sum` into bins, `dropped_power` from `|contribution|²`, optional side split (fields kept per side; `combine()` sums fields).
- Phase-precision strategy chosen here per constraint 2 (dedicated test: kernel phase vs f64 brute force across ranges/frequencies; require < λ/50 equivalent error at 20 km, 195 MHz).
- Tests: single-facet LPA amplitude+phase vs M9 brute force across incidence angles (document the breakdown angle); multi-facet small scenes (flat, tilted, sinusoid at fine posting) — complex field agreement with brute force (envelope AND phase); energy bookkeeping.

### M11 — Coherent output + mode integration
- `simulate(scene, cfg)` dispatches on mode; Dataset gains `field`, `power=|field|²`; `combine()` and complex `save()` per docs/output.md (round-trip test: save → load → bit-identical field, both h5netcdf-native and strict split modes).
- Cross-kernel consistency test: on an ensemble of rough-surface realizations (phases decorrelated), trace-averaged `|field|²` converges to the incoherent kernel's power (same facets fed to both) within statistical tolerance — ties the two kernels together above the level of either paper.

### M12 — Haynes 2018 benchmark suite (the headline verification)
New report cases under "Radar equation comparison":
- **Smooth-surface R⁻² fall-off**: flat scene altitude sweep with the coherent kernel → leading-edge power slope −2 (completing the −4/−3/−2 triad started in stage 1; same anchored-window method).
- **Coherent constants** (Haynes Table II): flat-plate nadir return vs the plane-wave / spherical-wave / infinite-mirror closed forms — absolute check of the amplitude normalization, not just slopes.
- **Coherence loss vs roughness**: Gaussian rough surfaces (rms σ_h, correlation length l) as explicit fine-posting realizations, sub-Fresnel facets, ensemble of ~100–200 seeded realizations over a Fresnel-zone-scale disk at a few altitudes (cheap: the disk is only ~10²–10³ λ across) → `F(h)` and the R²→R³ transition vs Haynes Eqs. (34)–(36) and Fig. 5–6, including the σ_h ≈ λ/4 transition point. Fit exponent tolerance set after first run, recorded like the OPR thresholds.
- **Speckle statistics**: rough-surface trace ensemble → amplitude Rayleigh / power exponential tests (KS or moment-based), coherent/incoherent power partition sanity (Grima-style, qualitative at this stage).

### M13 — xOPR coherent extension
- Rerun the two cached frames (20171121_03_005, 20170422_01_014) in coherent mode with facet subdivision to meet the Fresnel criterion at each frame's AGL and MCoRDS f0 (~195 MHz); record what subdivision was needed.
- **Explicit kernel comparison in the report (per review)**: each frame's "xOPR clutter" case grows a dedicated coherent-vs-incoherent comparison — a figure row with the measured radargram, the incoherent cluttergram, the coherent `|field|²` cluttergram, and the facet-scale-smoothed coherent-minus-incoherent difference (dB), all on shared axes; plus metrics: envelope agreement (Pearson / dB residual of smoothed coherent power vs incoherent power, thresholds set after observation), speckle contrast of the coherent product, and the two kernels' wall times. Same treatment on both frames.
- Honesty note carried into the report: at 32 m posting the DEM cannot supply true λ-scale phase, so coherent output on real frames is statistically meaningful (speckle, envelope) but not deterministically phase-accurate — consistent with why simc's authors stayed incoherent, and with Gerekos 2023's motivation for analytic sub-facet roughness (a candidate stage-2.5/3 enhancement, out of scope here).

## Decisions I'll make unless redirected
- **D2-1** rect facets throughout; incoherent parity re-baselined (expected ratio ≈ 2.06; see M8). Approved direction from review.
- **D2-2** Delta-pulse-at-carrier response; no waveform/chirp in stage 2 (stage 4 per overview).
- **D2-3 (revised per review)** User-facing config specifies relative permittivity per medium (air = 1, ice = 3.17); the kernel derives the scalar normal-incidence Γ ≈ −0.281 from the contrast. Angle-dependent Fresnel and multi-layer media arrive with stage 3 geometric optics on the same config structure.
- **D2-4** Roughness benchmarks by explicit realizations, not the Gerekos-2023 analytic rough facet (deferred as an enhancement).
- **D2-5** Agent split mirrors stage 1: well-specified geometry/output/report work to cheaper agents; M9/M10 physics core and M12 to the strongest model; I review each wave.

## Known risks
- LPA validity at low altitude + coarse DEM (constraint 1): mitigated by subdivision + hard validity check; the brute-force referee quantifies residual error.
- f32 phase precision (constraint 2): two candidate mitigations, test-gated.
- Speckle stochasticity in tests: fixed seeds + ensemble tolerances (JAX PRNG keys derived from explicit constants; no wall-clock randomness).
- Haynes constant checks are *absolute* — they will expose any normalization slip in the LPA amplitude that relative metrics can't see. Budget investigation time there.
- Coherent memory: complex64 doubles bin-array size; per-side fields double again. Same blocking structure should hold; watch the 4000-sample OPR windows.

## Definition of done
Coherent kernel validated against brute-force sub-λ summation and Haynes smooth/rough closed forms (R⁻² slope, Table II constants, coherence-loss curves); cross-kernel ensemble consistency test green; report shows the completed −4/−3/−2 triad, coherence-loss curves against Haynes analytics, and coherent xOPR cluttergrams with envelope agreement vs incoherent; CI stays fast (< ~5 s), full integration suite green; docs/incoherent_simulation.md gets a sibling docs/coherent_simulation.md and docs/output.md's coherent promises (field, combine, complex save) become real.
