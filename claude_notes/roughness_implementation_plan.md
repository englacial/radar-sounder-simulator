# Gerekos 2023 rough-facet response — implementation spec

Status: IMPLEMENTED (2026-07-28) -- measured verification numbers at the
bottom of this file. Code: src/soundersim/roughness.py,
compare/gerekos.py (referees), kernels/coherent.py + kernels/multilayer.py
(integration), config RoughnessConfig / roughness_seed, docs/roughness.md,
report case rough_facet (tools/run_rough_facet.py), tests
test_roughness.py + test_roughness_kernel.py.

Source: Gerekos, Haynes, Schroeder, Blankenship (2023), "The Phase Response of a
Rough Rectangular Facet for Radar Sounder Simulations of Both Coherent and
Incoherent Scattering", Radio Science 58, doi:10.1029/2022RS007594
(`reference_papers/Gerekos et al. - 2023 - ...pdf`). Equation numbers below are
the paper's.

## Why

Our facets are perfectly smooth at sub-facet scales, so the simulated response is
"too coherent": excess specular power, missing diffuse power. This is the leading
candidate for the ~8–10 dB mid-band deficit in the B26 firn comparison
(`b26_comparison_findings.md`) — DEM-copied firn layers scatter only specularly.
Gerekos 2023 derives the exact ensemble-average LPA phase response of a facet with
Gaussian roughness (RMS height σ, isotropic Gaussian correlation length l),
splitting it into a coherent and an incoherent term.

## Formulation

Facet with LPA phase coefficients A0, B0, D0 (Eqs 4–5; our kernel already computes
the equivalent quantities for its sinc·sinc response) and in-plane projected edge
lengths Lx, Ly (Eq 8). Perturbation δ ~ N(0, σ²) along the facet normal, Gaussian
correlation C(u) = exp(−|u|²/l²) (Eq A9).

Phase-perturbation wavenumber (Eq 15):

    K = k_i·cosθ_i + k_s·cosθ_s        (reflection; monostatic: K = 2 k cosθ)

k is the wavenumber in the medium containing the facet (for buried firn facets,
the local medium). cosθ from the facet normal and the (refracted) ray direction —
the kernel already has these.

**Coherent (mean) response** (Eq 20) — the existing LPA response times an
attenuation:

    ⟨Φ̃⟩ = e^{−iD0} · Lx·Ly·sinc(Lx·A0/2)·sinc(Ly·B0/2) · exp(−σ²K²/2)

**Incoherent variance** (Eq 21):

    D_Φ = e^{−σ²K²} Σ_{m=1}^{∞} (σ²K²)^m / m! · (l⁴/m²) · F_A(m)·F_B(m)

with (Eqs 22–24), erfi(z) ≡ −i·erf(iz):

    F_A(m) = 1 − e^{−Lx²m/l²}·cos(Lx·A0)
             + √π·e^{−A0²l²/(4m)}·[Re{A_m·erfi(A_m)} − Re{A_m}·erfi(Re{A_m})]
    A_m = (A0·l² + i·2·Lx·m) / (2·l·√m)          (F_B analogous with B0, Ly)

Derivation route (Appendix A): D_Φ is the center-difference integral (Eq A8)

    D_Φ = ∫_{−Lx}^{Lx}∫_{−Ly}^{Ly} (Lx−|u1|)(Ly−|u2|)·e^{i(A0u1+B0u2)}
          · (e^{−σ²K²[1−C(|u|)]} − e^{−σ²K²}) du1 du2

whose Taylor expansion in m factorizes into products of 1-D integrals
∫_{−Lx}^{Lx}(Lx−|u|)·e^{iA0u−m u²/l²} du = (l²/m)·F_A(m).

**Speckle reproduction** (Eqs 25–28): per-facet random phasor
φ_r = (ε1+iε2)/√2, ε ~ N(0,1); total field per facet

    E = F·⟨Φ̃⟩ + F·√(D_Φ)·φ_r

which has the correct ensemble-average power (Appendix D). Use a deterministic
per-facet seed so runs are reproducible.

**Convergence** (Appendix B): series is absolutely convergent; in practice
~10 terms for σ ≲ λ/20, up to ~250 for σ ≈ λ. Fixed term count chosen from
σ²K² (JAX: fixed-size scan; overshooting terms is cheap and harmless since
terms decay factorially).

**Numerical stability**: erfi(A_m) grows like e^{Re(A_m)²}; the prefactor
e^{−A0²l²/(4m)} = e^{−Re(A_m)²} exactly cancels it. Implement the *scaled*
combination e^{−Re(z)²}·erfi(z) directly (relate to the Faddeeva function
w(z) = e^{−z²}·erfc(−iz), for which stable rational approximations exist —
Humlíček / Poppe–Wijers — or use per-m change-of-variable Gauss–Legendre
quadrature of the 1-D integral above, substituting t = u·√m/l so fixed nodes
resolve the Gaussian at every m). Either route must be validated against a
float64 scipy/mpmath reference. Do NOT evaluate unscaled erfi.

**Simpler fallback for reference only** (Appendix C, Eq C1): the classical
Kirchhoff law σ_K (l ≪ L assumption) — useful as an independent cross-check in
tests, not as the production formula (breaks down for l ≳ 0.2·L).

## Validity limits (Section 4)

- Facet size and l up to a few λ; l should be ≤ facet size (roughness with l > L
  is really facet tilt and is misrepresented).
- Accuracy worst around σ ≈ λ/4 with large l; sub-λ/10 roughness (our firn case)
  is the easy regime.
- λ here is the wavelength in the local medium.

## Integration points

1. `src/soundersim/roughness.py` (new): `mean_attenuation(sigma, K)` and
   `d_phi(sigma, l, K, A0, B0, Lx, Ly)` (JAX, vmap-able over facets), plus the
   scaled-erfi/Faddeeva helper. Fixed series length static arg.
2. Config: roughness is a per-interface property, user-facing where interfaces /
   media are specified (surface, internal layers, bed): `roughness: (sigma_m,
   corr_length_m)` or None (default → bit-identical smooth path, kernels must
   skip the rough branch entirely when all interfaces are smooth). Plus
   `roughness_seed: int` for the speckle phasors.
3. `kernels/coherent.py` and `kernels/multilayer.py`: multiply the facet phase
   response by exp(−σ²K²/2); add the incoherent field term √D_Φ·φ_r with the
   same F factor (Fresnel/spreading/pattern) as the coherent term. For buried
   facets use local-medium k and refracted incidence angle. Transmission
   roughness at crossings: coherent attenuation exp(−σ²K_t²/2) with
   K_t = k₁cosθ₁ − k₂cosθ₂ (sign flip vs reflection) — near-negligible for
   low-contrast firn; include the attenuation factor (it is one exp), skip the
   transmission D_Φ term (document this).
4. Docs: `docs/roughness.md` user-facing page (formulation summary, config
   usage, validity limits, verification results).

## Verification plan

a. **Facet in isolation vs Monte Carlo** (paper Section 4.1 / Fig 4): discretize
   a facet at Δx = λ/40, generate Gaussian-roughness realizations (spectral
   synthesis, surface 10× the largest l), compute the numerical phase integral
   (Eq 29) over ≥100 trials, compare ⟨|Φ|²⟩ = |⟨Φ̃⟩|² + D_Φ. Sweep σ/λ at a few
   l values, nadir + oblique + off-principal-axis directions. Report case
   `rough_facet` with pass thresholds (~1 dB where the paper's Fig 4 agrees).
b. **Smooth limit**: σ=0 (and roughness=None) reproduces the existing kernels
   bit-identically (regression test on an existing scene).
c. **Haynes 2018 rough Fresnel zone** (paper Section 4.2): flat rough disk at
   nadir vs the Haynes closed forms already in `compare/haynes.py`
   (coherence-loss case) — coherent term must match exp(−4k²σ²)-style coherence
   loss; total power within ~1 dB over the σ scan.
d. **Series convergence test**: fixed-count series vs mpmath high-precision
   reference across the (σ/λ, l/λ, angle) grid; verify the σ ≲ λ/20 → 10-term
   claim before choosing the default term count.

## Follow-up experiment (separate, after landing)

Re-run the B26 comparison with roughness on the firn layers (σ of a few cm,
l of a few m — check C&S 2020 / the clutter repo for measured values) to test
whether diffuse layer scattering closes the mid-band gap. New cache keys —
plan compute accordingly.

## Measured verification numbers (2026-07-28, first run)

Implementation choices: scaled-erfi via the Faddeeva function w(z) --
e^{-x^2} erfi(x+iy) = i[e^{-x^2} - e^{-y^2} e^{2ixy} w(x+iy)] with w evaluated
ONLY in the closed upper half-plane by the branch-free Weideman (1994) N=32
rational approximation (chosen over Humlicek w4: 3.2e-13 vs ~1e-4 rel err, no
region branching, one polyval). Series in log space (Poisson weight
exp(m log x - lgamma(m+1) - x)); fixed length via n_terms_for(x) =
clip(ceil(x + 6 sqrt(x) + 5), 10, 300), a lax.scan over m (O(facets) memory).
Speckle: frozen per-facet phasors, seeded default_rng((roughness_seed, j)).
Transmission crossings: two-way exp(-2 sigma^2 K_t^2) (down/up cross the same
point -> perturbations add coherently; one exp per crossing), D_Phi skipped.

Math (tests/test_roughness.py):
- faddeeva vs scipy wofz: 3.15e-13 max rel (gate 1e-11)
- d_phi f64 vs scipy referee: 4.5e-10 max rel (gate 1e-8; Weideman-limited)
- d_phi f32 vs f64 referee: 4.1e-8 absolute / (LxLy)^2 (gate 1e-6)
- referee vs Eq-A8 brute quadrature: ~2e-15 rel (gate 1e-9)
- 10-term series vs mpmath(50 dps) at sigma <= lam/20: 1.98e-11 rel
  (gate 1e-9) -> the ~10-term claim holds; default floor kept at 10
- MC core (3 points, N=120): residuals -0.02/+0.24/-0.05 dB (gate 1 dB)

Report case rough_facet (tools/run_rough_facet.py):
- (a) MC sweep (150 realizations, 4x7 lam facet, lam/40 grid, 3 geometries,
  l/lam in {0.5, 1, 2}): 0.715 dB max residual for sigma <= 0.3 lam
  (gate 1.0); 1.04 dB including sigma = 0.4 lam (gate 1.5)
- (c) Haynes rough Fresnel-zone disk (h = 8000 lam, 4-lam facets, l = 2 lam):
  total 0.088 dB max (gate 1.0), coherent-only vs exp(-(2k sigma)^2)
  0.029 dB (gate 0.5, sigma <= 0.15 lam)
- (d) series_10term 1.98e-11 / full 1.98e-11 / faddeeva 3.15e-13

Kernels (tests/test_roughness_kernel.py):
- (b) smooth limit: roughness=None AND sigma=0 bit-identical (array_equal)
  to the pre-roughness kernels, coherent + multilayer
- coherent kernel speckle ensemble (64 seeds) vs f64 referee: -0.149 dB
  (gate 0.5); coherent part via zero phasors: -0.0016 dB (gate 0.05)
- multilayer eps->1 rough-bed reduction to the single-interface rough
  kernel: 4.3e-4 of peak (gate 5e-3)
- buried-facet local-k attenuation exp(-sigma^2 (2 k0 n_ice)^2 / 2):
  |ratio/pred - 1| = 8.9e-5 (gate 5e-3)
- transmission through simulate(): exp(-2 sigma^2 K_t^2) to 7.9e-4
  (gate 0.02)

- Haynes core in pytest (3 sigmas): total <= 0.09 dB, coherent <= 0.03 dB

Unit suite: 217 passed (199 pre-existing + 18 new), integration deselected.
