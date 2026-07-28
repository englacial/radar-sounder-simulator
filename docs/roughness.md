# Sub-facet roughness (rough rectangular facets)

The coherent kernels model each facet as a perfectly smooth plate, so a DEM-only scene is "too coherent": all the power the real λ-scale roughness would scatter diffusely stays in the specular sinc² lobe. The `roughness` option adds the **Gerekos et al. 2023** rough rectangular facet response (Radio Science 58, doi:10.1029/2022RS007594): each facet carries Gaussian roughness with RMS height σ and isotropic Gaussian correlation length *l* **below the facet scale**, and its ensemble-exact phase response splits into

- a **coherent (mean) term** — the usual LPA sinc·sinc response attenuated by `exp(−σ²K²/2)`, with `K = 2·k·cosθ` in the facet's **local medium** (buried facets use the in-ice wavenumber and the refracted arrival angle), and
- an **incoherent term** — a per-facet field `√D_Φ · φ_r` with the same Fresnel/transmission/spreading/antenna factor as the coherent term, where `D_Φ` is the paper's Eq. 21 series (the exact variance of the facet phase integral) and `φ_r` a frozen per-facet random phasor. One realization is drawn per run (`roughness_seed`), so radargrams show correctly-distributed speckle that decorrelates along track through the geometry, and runs are reproducible.

At interface **crossings** (transmission into the firn/ice), the mean-field attenuation `exp(−2σ²K_t²)` with `K_t = k₁cosθ₁ − k₂cosθ₂` is applied two-way (the up- and down-going rays cross the same interface point, so their perturbations add coherently). The diffusely *re-transmitted* field is not modeled — for low-contrast firn layers `K_t` is a few percent of the reflection `K`, making the whole crossing effect near-negligible there.

## Usage

Roughness is a per-interface property (coherent mode only):

```python
from soundersim.config import DemInterface, RoughnessConfig, SimConfig

cfg = SimConfig(
    mode="coherent", roughness_seed=0,
    radar=..., facets=..., media=...,
    interfaces=[
        DemInterface(name="surface",
                     roughness=RoughnessConfig(sigma_m=0.05, corr_length_m=2.0)),
        DemInterface(name="bed"),   # roughness=None -> smooth (exact pre-roughness path)
    ])
```

`roughness=None` (the default) is guaranteed **bit-identical** to the pre-roughness kernels — the rough branch is never traced, so unused it costs nothing. `sigma_m=0` through the rough branch is also bit-identical (regression-gated).

## Validity limits

- **Facet size and *l* up to a few λ** (in the local medium — for buried firn layers λ_ice ≈ λ₀/1.78). The LPA facet-size constraint (see [coherent_simulation.md](coherent_simulation.md)) still applies.
- **l ≤ facet size**: roughness with correlation length above the facet scale is really facet tilt and belongs in the DEM (a warning is emitted).
- Accuracy is worst around σ ≈ λ/4 with large *l* (measured ~1 dB); sub-λ/10 roughness — the firn case — is the easy regime (≤ ~0.3 dB).
- The series length is sized automatically from σ²K² (10 terms at σ ≤ λ/20, up to 300 near σ ≈ λ); cost scales with it, and only rough interfaces pay it.

## Numerical notes

Eq. 22's `erfi(A_m)` grows like `exp(Re(A_m)²)` and is never evaluated unscaled: the implementation folds the analytic prefactor into the Faddeeva function `w(z)` (bounded in the upper half-plane), evaluated by the branch-free Weideman rational approximation (N=32, measured 3×10⁻¹³ vs `scipy.special.wofz`). The Poisson-like series weights are computed in log space, so large σ²K² never overflows — including in the float32 surface kernel, where the measured D_Φ error is ~10⁻⁷ of (L_xL_y)², i.e. ~−70 dB in incoherent amplitude.

## Verification (report case `rough_facet` + CI tests)

- **Facet-in-isolation Monte Carlo** (paper Fig. 4 setup: 4×7 λ facet, λ/40 sampling, 150 correlated-Gaussian-surface realizations/point): ensemble ⟨|Φ|²⟩ matches `|⟨Φ⟩|² + D_Φ` to **0.72 dB max** over σ/λ ∈ [0.02, 0.3], l/λ ∈ {0.5, 1, 2}, nadir/oblique/off-axis geometries (1.04 dB including σ = 0.4 λ, the validity edge).
- **Series algebra**: the D_Φ series matches float64 brute-force quadrature of the underlying integral to ~10⁻¹⁵ and mpmath (50 digits) to 2×10⁻¹¹ with just 10 terms at σ ≤ λ/20.
- **Haynes 2018 rough Fresnel zone** (nadir facet-disk vs the closed forms in `compare/haynes.py`): total power to **0.09 dB**, coherent part vs `exp(−(2kσ)²)` to 0.03 dB.
- **Kernel wiring**: kernel speckle-ensemble mean power matches the float64 referee to 0.15 dB; ε→1 reduction of a rough bed through the refracted path matches the single-interface rough kernel to 4×10⁻⁴ of peak; buried-facet attenuation verified to use the **local** wavenumber (9×10⁻⁵ relative); transmission attenuation through `simulate()` matches `exp(−2σ²K_t²)` to 8×10⁻⁴.
- **Smooth limit**: `roughness=None` and `sigma_m=0` both bit-identical to the pre-roughness kernels.
