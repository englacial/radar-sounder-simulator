# Sub-facet roughness (rough rectangular facets)

By default, the coherent kernels model each facet as a perfectly smooth plate, so all the power the real λ-scale roughness would scatter diffusely stays in the specular sinc² lobe. The `roughness` option adds a rough rectangular facet response parameterization derived by Gerekos et al. 2023 (Radio Science 58, doi:10.1029/2022RS007594): each facet carries Gaussian roughness with RMS height σ and isotropic Gaussian correlation length *l* below the facet scale, and its ensemble-exact phase response splits into

- a **coherent (mean) term** — the usual LPA sinc·sinc response attenuated by `exp(−σ²K²/2)`, with `K = 2·k·cosθ` in the facet's local medium (buried facets use the in-ice wavenumber and the refracted arrival angle), and
- an **incoherent term** — a per-facet field `√D_Φ · φ_r` with the same Fresnel/transmission/spreading/antenna factor as the coherent term, where `D_Φ` is the paper's Eq. 21 series (the exact variance of the facet phase integral) and `φ_r` a frozen per-facet random phasor. One realization is drawn per run (`roughness_seed`), so radargrams show correctly-distributed speckle that decorrelates along track through the geometry, and runs are reproducible.

At interface **crossings** (transmission into the firn/ice), the mean-field attenuation `exp(−2σ²K_t²)` with `K_t = k₁cosθ₁ − k₂cosθ₂` is applied two-way (the up- and down-going rays cross the same interface point, so their perturbations add coherently). The diffusely *re-transmitted* field is not modeled — for low-contrast firn layers `K_t` is a few percent of the reflection `K`, making the whole crossing effect near-negligible there.

## Usage

Roughness is a per-interface property (coherent mode only); `acf` selects the correlation function (`"gaussian"` default, `"exponential"` see below):

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

## Exponential correlation function (`acf: exponential`)

The Gaussian ACF `ρ(r) = exp(−r²/l²)` has a spectrum that collapses like `exp(−k²l²/4)` beyond `k ~ 2/l`, so a Gaussian pair fitted to a measured surface at one Bragg wavenumber is tens to hundreds of dB low half an octave away (path B1, `claude_notes/roughness_b1_2026-08-26.md`). Measured ice-sheet surfaces (OIB ATM, `config/roughness/atm_b1.yaml`) and the Culberg & Schroeder 2020 firn-layer inversion (their Fig. 11, an S-IEM inversion with an **exponential** ACF) have `k⁻³`-like tails instead. `RoughnessConfig(acf="exponential")` switches the correlation function to `ρ(r) = exp(−r/l)`.

The ACF enters the rough facet model only through the incoherent variance (Eq. 21 in Gerekos et al., 2023): `D_Φ = e^{−σ²K²} Σ_m (σ²K²)^m/m! · I_m`, with `I_m` the Fourier transform of `ρ^m` over the facet. The coherent term (Eq. 20, `exp(−σ²K²/2)`) and the Poisson weights are ACF-independent, so the nadir coherent power is unchanged by the option. In the area-only (grazing-fix) form `I_m ≈ L_x L_y · W_m(k_B)`, `k_B² = A0² + B0²` (the transverse Bragg wavenumber, `2k sinθ` for a horizontal facet), and

- Gaussian: `W_m = π (l²/m) exp(−k_B² l²/(4m))`
- exponential: `W_m = 2π (l/m)² [1 + (k_B l/m)²]^(−3/2)` (Culberg & Schroeder 2020 Eq. 6 with `n = m`, `L = l`; the m-th term of the infinite-surface Kirchhoff law, `roughness.acf_spectrum`).

The finite-facet edge terms (Eqs. 22–24, `_f_factor`) have no closed form for the exponential ACF, so **`acf: exponential` requires the area-only D_Φ**, i.e. `SimConfig.grazing_fix` (the config validator, `d_phi` and the kernels all refuse otherwise). The same fixed series length (`n_terms_for`) suffices: `W_m` decays only polynomially in `k_B` but the Poisson weight still truncates the series in `m` (CI-asserted < 10⁻⁶ dB vs a 600-term float64 sum for `k_B l` up to 84, i.e. 400 MHz, `l = 5 m`, 90°; `tests/test_roughness_exponential.py`).

**Validity / when to use.** The option is the right choice whenever the measured (σ, l) come from an exponential-ACF fit: the geikie ATM line (best family in 83 % of blocks; `{source: atm_exponential}` in an experiment spec hands the table's σ, l straight to the kernel) and the C&S Fig. 11 firn-layer profile (`tools/run_b26_comparison.py --rough-runs N:src:exponential`). Its validity is narrower than the Gaussian's at the same (σ/λ, l/λ): an exponential surface has divergent slope variance, so the Kirchhoff tangent-plane assumption behind Eq. 21 is weakest at small `l/λ` and wide angles. Facet-in-isolation Monte Carlo with exponential surfaces (300 realisations/point, `claude_notes/roughness_exponential_2026-08-27.md`): on a 16×28 λ facet (edge remainder negligible) the area-only D_Φ is within 0.5 dB at nadir and at 50° for `l ≤ 2 λ` and all `σ ≤ 0.2 λ` (0.9 dB at `l = 4 λ`, `σ = 0.2 λ`); at 30° it is low by +0.8 (l = 0.5 λ) to +2.5 dB (l = 2 λ) and +6 dB at `l = 4 λ` at small σ, shrinking with σ — a finite-facet spectral-window effect on the steep part of the k⁻³ spectrum, not the tangent-plane failure. On the paper's 4×7 λ facet the dropped O(1) edge remainder adds 2–3 dB more at 30° and up to −5 dB at nadir for `l = 4 λ` (the facet is one correlation length wide); that remainder is what the grazing fix removes on purpose (facet-size dependent), so use the option with facets many λ across. The campaign spans 7.5–85 m facets at 1.5–5 m λ (firn layers ~12 λ_firn): the HAPS/high passes sit comfortably at 14–17 λ, but the low passes run 7.5 m ≈ 5 λ facets at 195 MHz, where the 30° spectral-window bias above applies most strongly — mitigated in practice by the adopted area-mean entries' small correlation lengths (l ≈ 0.8–1.6 λ, the ≤ +0.8..+2.5 dB end of the measured range). The Kirchhoff degradation for a divergent-slope surface shows up as a σ-dependent +1 dB drift at 50° for `l ≤ λ` on the small facet. For self-affine (power-law) surfaces neither ACF is right; an exponential fit through the Bragg band (`westcoast_*_exp` entries) is within ±0.5 dB of a `β ≈ 2.5–3.3` power law over 0.75–5 m, far better than any Gaussian, but it is a fit, not the family.

## Validity limits

- ***l* up to a few λ** (in the local medium — for buried firn layers λ_ice ≈ λ₀/1.78). Facet size is governed separately: the LPA facet-size constraint (see [coherent_simulation.md](coherent_simulation.md)) bounds it above, and the edge/spectral-window effects above favor facets many λ across.
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

### ATM exponential parameters and aggregation (per line and per stratum)

`config/roughness/atm_tier2_strata.yaml` carries per-stratum exponential (σ, l)
entries per facies/elevation stratum on both ice sheets (726 ATM sites), merged
into the `atm_b1.yaml` table by `tools/surface_roughness_b1.load_table`.

**Aggregation**: the entries used by
`source: atm_exponential` are fits through the linear-domain area mean of
the per-block (per-site-year) S(k_B) at the four Bragg points, not the median.
The radar's ensemble-mean clutter integrates the area average of the local
spectrum over the footprint, and the measured σ² fields are heavily
right-skewed — a minority of rough blocks carries most of the scattered power
— so the median under-represents the clutter by 1–13 dB depending on the
population (largest for the coastal <500 m Antarctic stratum). Median-fit
entries remain in the tables for reference. Lines with their own ATM coverage
(westcoast, geikie, getz, david) use their own line's area mean rather than a
stratum. Validation across all six pilot lines (median |mid-column error|
9.2 → 3.8 dB): `claude_notes/experiments_2026-08-31/atm_area_mean/results.md`. Each entry
has a `usability`: `use` (exponential is the best 3-parameter family, ν ≈ 0.5),
`marginal` (a power law fits better; the exponential under-predicts wide-angle
scatter by ~1–2 dB at 195/300 MHz — a warning is emitted), or `refuse`
(margins < 1500 m and ice shelves: ν ≥ 0.6, `l` at the fit bound; the
exponential over-predicts by +2…+25 dB — `source: atm_exponential` raises).
Site-specific entries take precedence over strata; `stratum_lines` supplies
the fallback mapping for lines without one. One surface law per line: every
pass, real or synthetic, uses the line's default entry (the reference pass's
spectrum); the effective-Gaussian `atm_b1` pair still varies with each pass's
carrier frequency, the underlying spectrum does not.
