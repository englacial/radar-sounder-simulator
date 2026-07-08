# Coherent field normalization convention (M9)

Established by the brute-force reference (`src/soundersim/compare/brute_force.py`);
the coherent facet kernel (M10) and the Haynes benchmarks (M12) MUST adopt the
same convention so absolute-constant checks are meaningful.

## The convention

Per surface element (sub-wavelength sample in the reference; facet in the kernel):

```
field = Σ_i  (j·k / 2π) · Γ · cosθ_i · dA_i · exp(−2j·k·r_i) / r_i²
```

- `r_i`: platform → element distance; `cosθ_i = r̂·n̂`, `r̂` pointing element → platform
  (same convention as `kernels/geometry.py::ranges_and_cos`).
- Phase: `exp(−2j·k·r)` — two-way delay phase, *negative* sign, plus the constant
  `+π/2` from the `j` prefactor. All complex sums in float64/complex128 in the
  reference (the kernel's f32-vs-f64 strategy is the M10 decision).
- `Γ`: scalar normal-incidence Fresnel coefficient (≈ −0.281 air→ice); real and
  negative, so it flips phase by π.

## Why j·k/2π

This is the physical-optics (Kirchhoff) scalar backscatter integral for a
monostatic sounder over a locally specular interface,

```
E_s = (j·k / 2π) ∫ Γ cosθ · e^{−2jkr} / r² dA ,
```

i.e. each element re-radiates the incident spherical wave with the aperture-like
obliquity factor cosθ; the j·k/2π makes the stationary-phase evaluation land
exactly on the image-method result with no leftover constant (derivation below).
Equivalent to Haynes 2018 Eq. 13–14 up to the overall normalization, which
Haynes leaves inside the radar equation; we fix it so `|field|²` is an absolute
(relative-to-transmit) power.

## Image-method anchor (the number everything is checked against)

For an infinite flat interface at nadir range `h` (normals up, platform above):
cosθ = h/r, dA = 2πρ dρ, ρ dρ = r dr →

```
field = j·k·h ∫_h^∞ e^{−2jkr} / r² dr
      ≈ j·k·h · e^{−2jkh} / (2j·k·h²)          (endpoint/stationary-phase, kh ≫ 1)
      = Γ · exp(−2j·k·h) / (2h)
```

**Flat-plate return: `field = Γ·e^{−2jkh}/(2h)`, `|field|² = |Γ|²/(4h²)`.**

This is the classic image method (Haynes Fig. 2, Eq. 19–21): a mirror source at
range 2h with amplitude spreading 1/(2h) — the "infinite mirror" row of Haynes
Table I, `P_r ∝ |Γ|²/(2⁶π²)·λ²/R²` once antenna gains/aperture are attached.
R⁻² fall-off in power, R⁻¹ in field.

Useful corollaries in this normalization (validated in `tests/test_brute_force.py`):

- **Hard-edged disk of radius a** (h ≫ a):
  `|field(a)|² ≈ (Γ²/2h²)·(1 − cos(k·a²/h))` — Haynes Eq. 15's oscillation.
  Zeros where √(h²+a²) − h = n·λ/2; maxima at half-integers.
- **First Fresnel zone** (a = r_f = √(λh/2)): `|field|² = 4·|Γ|²/(4h²)` —
  exactly 4× the infinite plate (Haynes Eq. 16–17).

## Practical notes for M10+

- A **hard-edged finite plate does not converge** to the image value as it grows —
  the rim ringing (the Eq. 15 cosine) never decays. Benchmarks comparing against
  `Γ/(2h)` must taper the scene edge (raised-cosine area weights over many
  Fresnel zones work; see `flat_disk_samples(taper_start=...)`) or average over
  the oscillation.
- λ/10 sampling is comfortably converged: halving to λ/20 moved the tapered
  flat-plate field by ~2×10⁻⁸ relative (cell-centered sampling ≈ midpoint rule).
- The reference is O(N) dense NumPy — keep scenes ≤ ~10⁶ samples.
- The LPA facet kernel's per-facet closed form must reduce to
  `(j·k/2π)·Γ·cosθ·A·e^{−2jkr}/r²` in the small-facet limit — that limit is the
  cross-check against `brute_force_field` in M10.
