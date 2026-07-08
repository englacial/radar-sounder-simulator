# Implementation plan: stage 3 (subsurface layers)

Per `docs/overview.md` stage 3: geometric optics through arbitrary internal layers, bed clutter (BedMachine) added to the xOPR comparison, and synthetic firn layers vs the Culberg & Schroeder 2020 firn power plateau. Builds on stage 2 (committed a33b9d8): single rect-facet geometry path, ordered `media` config, both kernels, brute-force referee. **Status: complete (2026-07-08). M14–M19 done: 141 CI + 20 integration tests green; report has 19 cases across 4 groups, 0 fails. Key results: refraction solve validated vs Fermat referee (error = facet-anchoring, quadratic in offset; sin(θ_rare) iteration avoids critical-angle singularity; ~58 ns/pair f32); slab absolute closed form to 0.7%/0.64° parameter-free; bed fall-off slope −2.00 in (h+d/n); xOPR bed clutter input-limited (Greenland: sim adds ~0.5 bins over BedMachine's own 6.1-bin floor vs picks, corr 0.984; Peninsula: BedMachine v3 vs picks ±370 m, corr 0.461 — measured, floor-aware gate); geoid is EIGEN-6C4 not EGM2008; firn plateau REPRODUCED in 3-D (20 uniform offset layers, B26 core, plateau 10.5 dB/40 m then 15.6 dB rolloff, tracks decimated-γ², coherent-only — incoherent flat; uniform decimation required, depth-graded aliases the compaction trend; compile ~0.18·N² s). docs/multilayer_simulation.md written. Not committed.**

## Physics scope

A scene becomes an ordered stack: media `[air, ice, ..., substrate]` and interfaces `[surface, layer_1, ..., bed]`, each interface its own DEM grid (potentially different resolutions — surface from REMA/ArcticDEM, bed from BedMachine). Per interface, each facet contributes via the ray path refracted through all interfaces above it (Snell / geometric optics, Gerekos 2018 scheme adapted to our facets):

- **Delay**: sum of optical path lengths, `Σ n_i·s_i / c` (in-medium wave speed `c/√ε_r`).
- **Amplitude**: product of downward/upward Fresnel transmission coefficients at each crossed interface, reflection Γ at the target interface, refraction-modified geometric spreading (divergence factor), and accumulated attenuation (new per-medium property).
- **Phase (coherent)**: `k_i = k₀·√ε_r` along each leg; same LPA facet integral evaluated in the local incidence geometry at the target facet.
- Single-bounce only (no multiples), scalar fields, angle-dependent scalar Fresnel coefficients now (replacing stage 2's normal-incidence constant — the machinery needs angles anyway).

**The delicate core** (flagged in the handoff notes): the two-point refraction solve — for each (trace, subsurface facet) pair, the Snell-stationary crossing point on each interface above. Stage-3 approach: **local-plane approximation** — solve the flat-interface Fermat problem (1-D root find, vectorized Newton in JAX) against each interface's local mean plane at the horizontal midpoint, then evaluate transmission/incidence angles there. Exact for flat interfaces; error grows with interface roughness at the crossing point (quantified in M15 against brute-force Fermat minimization on the true faceted surface). Total internal reflection / shadowed paths → contribution dropped and accounted (extends `dropped_power`).

## Milestones

### M14 — Multi-interface scene + config
- `Medium` gains `attenuation_db_per_km: float = 0` (constant per medium; MacGregor/Matsuoka fields are a later data-driven upgrade). Config validation: N media ⇒ N−1 interfaces.
- Scene container generalizes to an ordered list of interfaces; **each interface specified one of three ways (per review)**: (a) its own DEM, (b) a flat surface at constant elevation, or (c) **a reference DEM plus a constant vertical offset** (e.g. `surface − 2 m`) — the firn-layer case. Config-level (pydantic `InterfaceConfig` with a dem-source/flat/offset-of union). Offset interfaces shift elevation before the projected→ECEF pipeline (a constant-elevation offset ≈ translation along local up; normals/areas unchanged to first order — implementer may fast-path this, with a correctness test against a full rebuild).
- `build_facets` per interface (existing code, unchanged); synthetic scenes gain two-interface variants (flat slab, tilted bed under flat surface, rough bed under flat surface, flat bed under rough surface) plus an offset-stack variant (surface + N offset copies).
- Layered-delay utilities (optical path length, in-medium dt per leg). Tests: config validation; slab nadir delay `2h/c + 2d√ε/c` exact.

### M15 — Refraction solve (verification-first, like M9)
- Brute-force Fermat referee (NumPy f64): travel-time minimization over a finely sampled true surface for (platform, subsurface point) pairs — the ground truth.
- JAX vectorized flat-local-plane Snell solve: 1-D Newton on the horizontal offset of the crossing point; robust bracketing/fallback, TIR detection.
- Tests: exact agreement with the analytic flat-interface solution; vs Fermat referee on tilted and gently rough surfaces (document error vs surface slope/roughness at the crossing point); Snell's law satisfied at the returned point to tolerance; TIR cases dropped not NaN'd; convergence within fixed iteration budget over the geometry range we use (airborne + stratospheric).

### M16 — Kernel extensions (both modes, one geometry path)
- Per-interface contribution pipeline: refraction solve → angles → per-crossing angle-dependent scalar Fresnel T↓/T↑ (and Γ at target) → attenuation → spreading (divergence factor for refraction) → delay; incoherent sums power, coherent runs the LPA facet integral with in-medium k and total phase.
- Output per docs/output.md: optional `layer` dimension (`surface`, `bed`, ...); per-trace `nadir_twtt` per layer; `dropped_power` gains the TIR/shadow channel.
- Tests: with ε_r(ice)→1 and zero attenuation, the multilayer bed reduces exactly to a surface-only run at the bed geometry (both kernels); Fresnel angle-dependence vs textbook curves; energy bookkeeping.

### M17 — Analytic + referee verification (report cases, "Radar equation comparison")
- **Flat slab closed form**: nadir bed echo delay and absolute amplitude vs the image-in-dielectric-halfspace solution (T↓T↑Γ_bed with refraction spreading — derive and cite in the test; the stage-2 normalization makes this parameter-free like haynes_constants). Sweep depth and ε_r.
- **Two-media brute force**: extend the referee to two flat/gently-rough interfaces via the Fermat solve per sample pair (tiny scenes); coherent kernel field vs referee (envelope + phase), same style as M10.
- **Bed fall-off**: flat surface + flat bed altitude/depth sweep — smooth bed coherent `r_eff⁻²`-analog and rough-bed `r⁻³`-analog with the refraction-corrected effective range (state the expected forms in the test; this extends the Haynes triad below the surface).

### M18 — xOPR bed clutter
- BedMachine (Greenland v5 / Antarctica v3) bed + our existing surface DEMs for both cached frames; bed pick from the frame (xOPR layer products) as the measured reference.
- Incoherent + coherent bed cluttergrams; gates in the stage-1 style: median |simulated bed nadir − frame Bottom pick| after constant-offset removal (attenuation constant chosen/recorded; absolute bed power recorded, not gated). Report case per frame under "xOPR clutter"; honesty note: BedMachine's ~kilometer-scale effective resolution caps off-nadir bed-clutter realism (handoff notes) — expect timing fidelity, not clutter-texture fidelity.
- Cross-check available: Snow4Flow `combogram` (nadir bed interface, dipole pattern) as a qualitative timing/power sanity reference — optional, cheap.

### M19 — Firn power plateau (Culberg & Schroeder 2020, Fig. 9 analog in 3D)
- **Goal (per review): reproduce the general behavior of the Fig. 9 depth–power profile — the near-surface "power plateau" (elevated, slowly-decaying layer return power over the upper tens of meters, then rolloff) — from a full 3-D facet simulation. NOT a numerical match to the paper's 1-D layered dielectric model.**
- Scene: firn layers as flat interfaces or surface-offset copies (the M14 offset-of-DEM spec), decimated to a reasonable count (O(tens)); layer permittivities from a borehole density profile via the standard density–permittivity relation.
- **Source material**: the paper (now in `reference_papers/`) and the user's reimplementation at `/home/thomasteisberg/Documents/clutter` (READ-ONLY; never referenced at runtime or in tests by external path): reuse its density→permittivity approach (`src/firn_clutter/density.py`), and use its 1-D transfer-matrix depth-power curve and/or the digitized Fig. 9 CSVs as context lines on our plot — for shape comparison, not a gate. **Any resources the tests/case need (borehole density `.tab` files, digitized `fig09*_digitized.csv`, a precomputed 1-D context curve) are COPIED into this repo** under `tests/fixtures/firn/` with a provenance note (source repo + original data citation + C&S 2020 citation), so the suite is self-contained.
- Output: depth–power profile extracted from the simulated trace (layer-return power vs depth, both kernels; coherent is the interesting one — interference between closely spaced layers is what a 1-D model captures and an incoherent sum does not), plotted against the context curves. Gate: qualitative/structural (plateau present: e.g. power within the upper ~X dB band over the upper tens of meters, then monotonic rolloff — exact criterion set after first run and recorded, per repo convention).
- **Scoping caution** (handoff notes): thousands of explicit layers are not tractable in a facet method — O(tens) of decimated layers is a physics check of the multilayer machinery, not a stratigraphy capability claim. If plateau behavior demonstrably requires far more layers than is tractable, the finding is "needs the 1-D hybrid (future tier-2 coupling)", reported honestly with the evidence.

## Decisions I'll make unless redirected
- **D3-1** Local-plane Snell solve (per interface, per facet pair) with brute-force Fermat as referee; full faceted-surface search deferred unless M15 errors demand it.
- **D3-2** Angle-dependent scalar Fresnel coefficients everywhere (upgrade from normal-incidence constants); still scalar/no polarization.
- **D3-3** Constant per-medium attenuation (config), spatial attenuation fields deferred.
- **D3-4** Single-bounce only; TIR/shadow dropped and accounted.
- **D3-5** Agent split as before: config/scene plumbing and xOPR data plumbing to Opus; refraction solve, kernel extensions, and analytic verification to the strongest model; I review each wave.

## Known risks
- Refraction-solve robustness on rough interfaces (multi-valued stationary points, grazing rays) — the M15 referee quantifies where the local-plane answer diverges; grazing/TIR paths dropped with accounting.
- Runtime: the solve adds Newton iterations per (trace × facet × interface); budget measured in M15 before M16 commits to a structure (blocking already in place helps).
- BedMachine realism cap (above) — gates on timing only.
- Bed pick availability/quality on the two cached frames (Peninsula bed may be patchy) — checked early in M18; fallback is choosing one additional frame with a clean Bottom pick.
- Firn milestone may exceed facet-method scope (above) — explicitly allowed to conclude "hybrid needed".

## Definition of done
Multilayer geometric optics validated: refraction solve vs Fermat referee, flat-slab absolute closed form (parameter-free), two-media brute-force field agreement, ε→1 reduction test; bed clutter simulated for both xOPR frames with bed-pick timing gates and honest BedMachine caveats; firn plateau case run in 3-D showing the qualitative Fig. 9 plateau behavior against sourced context curves (or a documented scope finding); `layer` dimension in outputs per docs/output.md; docs page for multilayer simulation; report grown accordingly; CI stays < ~10 s.
