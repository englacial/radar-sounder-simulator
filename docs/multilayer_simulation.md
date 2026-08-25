# Multilayer (subsurface) simulation

`soundersim` can simulate returns from subsurface interfaces — an ice bed, internal layers, firn stratigraphy — through geometric-optics refraction. Both kernels ([incoherent](incoherent_simulation.md) and [coherent](coherent_simulation.md)) share one multilayer geometry path.

## Specifying a layered scene

A scene is an ordered stack of media and the interfaces between them:

- `SimConfig.media`: top-down list of `Medium(name, eps_r, attenuation_db_per_km)` — e.g. air (ε_r = 1), ice (ε_r = 3.17, optional constant one-way attenuation).
- `SimConfig.interfaces`: top-down list, one fewer than media. Each interface is one of:
  - **`dem`** — its own DEM (the surface from REMA/ArcticDEM, a bed from BedMachine, ...). Interfaces may use different resolution grids.
  - **`flat`** — a constant ellipsoidal elevation.
  - **`offset`** — another interface plus a constant vertical offset, e.g. `surface − 2 m`. This is how firn layers are specified; offsets may chain. (Implemented as a translation along local up — exact to the ~10⁻⁵ ellipsoidal-radius correction, verified against a full rebuild.)

A single `dem` interface with two media reproduces the surface-only behavior of stages 1–2 exactly (the surface path is unchanged code).

## Physics

Per interface j, each facet contributes via the ray refracted through the interfaces above it:

- **Path**: the two-point Snell problem is solved per crossing against each interface's local facet plane (vectorized Newton on sin θ of the rarer medium — singularity-free at the critical angle). This local-plane approximation was validated against brute-force Fermat minimization: the error is an *anchoring* error, quadratic in the anchor offset, negligible when facets are small against the interface's roughness wavelength (see `tests/test_refraction.py` and the twomedia_field report case for measured scalings and where it degrades).
- **Delay/phase**: optical path length with in-medium speed `c/√ε_r`; the coherent kernel uses in-medium wavenumbers per leg and computes path lengths in float64 inside the kernel.
- **Amplitude**: angle-dependent scalar Fresnel transmission (TE convention) at each crossing (τ↓·τ↑ two-way), normal-incidence reflection Γ at the target interface, refraction-corrected geometric spreading (the layered-GO divergence factor `L_par·L_perp`, validated against ray-tube tracing and the image-in-dielectric closed form to 0.05%), and per-medium attenuation. The incoherent kernel applies the power-domain equivalents and keeps its no-target-reflectivity simc convention on the surface layer.
- **Sub-facet roughness** (coherent mode): a per-interface `roughness` config adds the Gerekos et al. 2023 rough-facet response — coherent attenuation with the local-medium `K = 2k_j cosθ` plus the incoherent `√D_Φ·φ_r` speckle term at the target reflection, and the two-way transmission attenuation `exp(−2σ²K_t²)` at crossings. See [roughness.md](roughness.md).
- **Accounting**: total-internal-reflection, shadowed, non-converged, and out-of-window contributions go to per-layer `dropped_power` — never silently lost, never NaN. Facets provably outside the fast-time window (horizontal distance ≥ `c·t_end/2`, which bounds every optical path from below) are skipped per trace by the along-track windowing described in [coherent_simulation.md](coherent_simulation.md) and are not counted.
- **Cost**: O(traces × facets in each trace's window) per chunk, independent of the sample count. The sequential (surface + bed) path runs its two Newton solves as a loop-carried `fori_loop` in component form so XLA:CPU fuses each iteration into one loop; the kernel is memory-bandwidth-bound, not compute-bound (see `claude_notes/runtime_reduction_proposals_2026-08-24.md`).

Single-bounce only (no multiples), scalar fields, and the sequential interface-by-interface chaining is an approximation whose error vanishes with layer contrast (fine for firn; quantified for rough interfaces in the report).

## Output

The Dataset ([output.md](output.md)) gains an optional `layer` dimension on `power`/`field` (named from the interface list, e.g. `surface`, `bed`), per-layer `nadir_twtt` and `dropped_power`. `combine(ds, "layer")` applies the mode-correct rule (field summation for coherent).

## Verification (all in the report)

- **Refraction core**: analytic flat/tilted-interface cases exact; vs brute-force Fermat on rough surfaces with documented error scaling; Snell residual < 10⁻¹²; TIR masked; up/down reciprocity to 3×10⁻¹² m.
- **slab_absolute**: parameter-free flat-slab closed form `τ↓τ↑·Γ_bed·e^(−2jk₀(h+nd))/(2(h+d/n))` through `simulate()`: magnitude to 0.7%, phase to 0.64°, delay exact to the bin, attenuation law to 0.02%, across depth/permittivity/altitude sweeps.
- **twomedia_field**: coherent kernel vs sub-wavelength two-media brute force on flat and gently rough scenes — ~0.1–0.2% envelope, <1° phase; degradation on rougher surfaces measured and recorded.
- **bed_falloff**: nadir bed power fits `(h + d/n)⁻²` to slope −2.00 in both altitude and depth sweeps.
- **ε→1 reduction**: with ice ε_r = 1, the bed layer reproduces a surface-only run at the bed geometry (both kernels).
- **xOPR bed clutter**: BedMachine (Greenland v5 / Antarctica v3; EIGEN-6C4 geoid corrected to ellipsoidal heights) under the cached frames, gated on bed-pick timing with a measured *input-bed error floor* — on the Greenland frame the simulation adds only ~0.5 bins over BedMachine's own disagreement with the picks; on the Peninsula frame BedMachine v3 itself diverges from the picks by ±370 m, so timing fidelity there is input-limited (recorded, not hidden).
- **Firn power plateau** (Culberg & Schroeder 2020, Fig. 9 analog): a B26-core density profile decimated to 20 offset-interface layers at 195 MHz reproduces the plateau in full 3-D — coherent per-layer power elevated over the upper ~40 m then rolling off ~15 dB, tracking the decimated-γ² closed form and qualitatively consistent with the paper's 1-D curves. The incoherent kernel shows no such structure: the plateau is coherent/specular physics. Practical note recorded in the case: layer decimation must be uniform-in-depth — depth-graded spacing aliases the smooth compaction trend into spurious deep contrasts.

## Operating constraints

- Compile cost grows ~quadratically with interface count (one refraction chain per layer); O(20) layers ≈ 1 minute of XLA compile. Fine for firn checks; hundreds of layers need the future 1-D hybrid, not this path.
- BedMachine's native resolution (150/500 m, smooth interpolation) caps off-nadir bed-clutter realism: timing fidelity is the supported claim, not clutter texture.
- The refraction solve adds ~an order of magnitude per-facet cost over the surface-only kernels (~0.8 µs/facet/trace f64 on CPU); still minutes-scale for real-frame runs.
