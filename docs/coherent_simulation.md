# Coherent surface clutter simulation

`soundersim` can perform coherent (complex field-summing) clutter simulation. It shares the scene representation, geometric setup, and fast-time binning with the incoherent mode ([incoherent_simulation.md](incoherent_simulation.md)); only the kernel differs. Output follows [output.md](output.md): a complex `field` variable plus precomputed `power = |field|²`, with `combine()` applying field-level summation and `save()`/`load()` handling complex data.

## Physics

Per rectangular mean-plane facet (center, edge vectors `e1`/`e2`, area A, normal n̂), the kernel accumulates the physical-optics field under the linear phase approximation (LPA; Nouvel et al. 2004):

```
field = (j·k/2π) · Γ · cosθ · A · sinc(k·r̂·e1) · sinc(k·r̂·e2) · exp(−2j·k·r) / r²
```

with `sinc(x) = sin(x)/x` and `k = 2πf₀/c`. Contributions are complex-summed into fast-time bins by `floor((2r/c − t0)/dt)` (drop-not-wrap, `dropped_power` accumulates `|contribution|²`). The response is a delta pulse at the carrier; waveform/chirp convolution is a later processing stage.

**Normalization** is absolute, not relative: the convention (documented in the brute-force reference module) makes the total return from an infinite flat interface at nadir range h equal `Γ·exp(−2jkh)/(2h)` — the image-method result. This is what lets the Haynes verification check *constants*, not just slopes.

**Reflectivity** comes from the configured media (`SimConfig.media`, relative permittivities, default air ε_r = 1 → ice ε_r = 3.17): the normal-incidence scalar Fresnel coefficient Γ = (√ε₁ − √ε₂)/(√ε₁ + √ε₂) ≈ −0.281, sign (π phase flip) preserved. Scalar fields throughout — no polarization.

## Constraints to respect (enforced or warned)

- **Facet size vs Fresnel zone.** LPA drops the quadratic phase term; the per-facet error at nadir reaches ~5% at facet size `L ≈ 0.23·√(λr)` and grows steeply beyond (measured against brute force; the breakdown is at *near-nadir*, not oblique — off-nadir facets are sinc-suppressed). `simulate()` runs `check_facet_size` (β = 0.5 default) and warns; use `FacetConfig.spacing` to bilinearly subdivide the DEM below its native posting. For ~1–2% nadir accuracy target `L ≈ 0.1·√(λ·r_min)`.
- **Phase precision.** The kernel computes carrier phase via per-trace reference-range subtraction (f32 hot loop on `2k(r − r_ref)`, f64 constant folded back), measured at λ/664 equivalent error at 20 km range and 195 MHz. Adequate for airborne/stratospheric geometries; truly orbital ranges would need a float64 range path (noted in the kernel docstring).
- **DEM phase honesty.** A 32 m DEM carries no λ-scale surface information, so coherent output on real terrain is *statistically* meaningful — speckle, envelopes, interference structure — not deterministically phase-accurate at any given pixel.

## Coherent vs incoherent on the same facets

On identical facet grids the two kernels differ by the deterministic per-facet factor `(kΓ/2π)²·sinc²` and, physically, by what they assume about sub-facet scattering: the coherent LPA facet is a specular plate whose sinc² directivity concentrates the return at the leading edge, while the incoherent kernel's Lambertian-like `cos²θ` fills the diffuse off-nadir field. On rough surfaces with decorrelated facet phases, ensemble-averaged coherent power converges to the (scaled) incoherent power — a cross-kernel consistency test in CI. On real frames the measured radargram sits between the two kernels' predictions; the diffuse/specular partition per facet from sub-facet roughness statistics (Gerekos et al. 2023) is available via the per-interface `roughness` config — see [roughness.md](roughness.md).

## Verification (all in the report)

- **Brute force referee**: a float64 sub-wavelength physical-optics summation (`compare/brute_force.py`), itself validated against closed forms (point target, image-method flat plate to 0.07%, Fresnel-zone oscillation, first-Fresnel-zone 4× power). The LPA kernel is tested against it facet-by-facet and scene-by-scene, including the small-facet limit.
- **Haynes et al. 2018** ("Radar equation comparison" report section): smooth-surface fall-off slope −2 (completing the −4/−3/−2 triad with stage 1); parameter-free absolute constants (magnitude to 0.1%, end-to-end through `simulate()` to ~1%); coherence-loss curves vs Eqs. 34–36 across σ_h/λ and correlation length (≤0.25 dB residual in the coherent regime, ~1 dB through the σ_h ≈ λ/4 transition), with the σ = 0 limit exact; speckle statistics (Rayleigh amplitude / exponential power).
- **xOPR frames** ("xOPR clutter" section): coherent cluttergrams for both cached frames on subdivided facets, compared side-by-side with the measured radargram (Surface pick overlaid, leading-edge gate after constant-offset removal), with speckle contrast ≈ 1 (fully developed speckle) and recorded subdivision choices and estimated LPA error.

One subtle verification lesson, recorded in the tests: on a *smooth* surface the coherent leading edge is the total Fresnel-zone return — individual range bins are cancellation-dominated ring integrals — so coherent power comparisons gate on total/windowed field, and coherent-vs-incoherent range-binned ensemble comparisons need bin widths that are integer multiples of λ/2.
