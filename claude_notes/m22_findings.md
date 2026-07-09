# M22 antenna patterns — implementation notes (2026-07-08)

## What was built

- `AntennaConfig` on `RadarConfig` (config.py, additive): kinds isotropic
  (default) / dipole / array / tabulated; `roll_source: none|nav`. Array
  spacing is in CARRIER WAVELENGTHS (`spacing_lam`, dimensionless — chosen
  over metres so the config is f0-independent and matches array-factor math);
  element axis cross-track, boresight nadir (MCoRDS-like), unsteered.
  Tabulated is 1-D g(theta), rotationally symmetric about the (rolled) nadir
  boresight, linear interp, clamped at table ends.
- `antenna.py`: pattern frame from u_at/u_ct/nadir (+ Rodrigues roll about
  u_at), `pattern_args()` producing the kernel tuple `(kind, pv, pa, pb)`,
  `gain_fn(kind)` (jnp, in-kernel) and `field_gain()` (f64 NumPy reference).
- All three kernels take `pattern=None` and weight per facet: coherent /
  multilayer FIELD × g², incoherent POWER × g⁴. Multilayer evaluates g at the
  AIR-leg departure direction (platform → first crossing).

## Convention (D4-2)

g = ONE-WAY FIELD gain, peak-normalized. Monostatic two-way: field ∝
g_tx·g_rx = g², power ∝ |g²|² = g⁴. Stated in physics.py + antenna.py
docstrings. Cross-kernel ensemble test extended with a steep tabulated
pattern (per-bin g⁴ spans ×445): coherent ensemble/|ref|² mean 1.045
(min/max 0.779/1.397, speckle s.e. ~0.18), total vs C0·incoherent 1.011,
mean sinc² 0.991–0.994 — conventions consistent.

## JIT / recompile behavior (M19 rule respected)

Pattern KIND is a static in the lru_cache factory keys
(`_coherent_fn(split, n, interp, pattern)`, `_incoherent_fn(split, n,
pattern)`, `_refracted_fn(coh, split, n, n_crossed, pattern)`); everything
run-varying is traced: per-trace pattern vector pv (vmap axis 0), array
n_elements/spacing_lam and tabulated theta/gain arrays (broadcast). Verified
by `claude_notes/m22_recompile_probe.py` via `fn._cache_size()`:

- coherent dipole: 2 axes × roll values → 1 executable
- incoherent array: (5,0.5)/(7,0.35)/(3,0.6)/(15,0.5) → 1 executable
- tabulated: same table length, different values → 1; new length → +1
  (shape retrace, expected)
- multilayer tabulated: 3 gain tables → 1 executable

Isotropic default: the factory's "isotropic" branch adds NO ops (gain code
not traced); dummy pv/pa/pb args are dead. Regression
(`jit_regression_check.py compare claude_notes/m22_baseline.npz`, saved
pre-change): all 12 arrays bit-identical (worst rel diff 0.0).

## Numerical edges

- Dipole on-axis null: exact 0 in f64; in the f32 surface kernels the null
  bottoms at g ≈ 4e-2 (cos(pi/2) rounds to -4.4e-8, sin-psi clamp 1e-6) →
  ≥54 dB down in g⁴ power. Fine physically; tests assert accordingly.
- Array factor evaluated as sin(Nx)/(N sin x) with an |sin x|<1e-5 guard
  switching to cos(Nx)/cos(x) (L'Hôpital at main/grating lobes).
- Roll sign: positive = right wing down = right-handed rotation about u_at
  in ENU; nadir boresight then tilts LEFT of travel (belly faces port).
  OPR frames: `opr.frame_scene` now stores `frame.Roll[idx]` (radians) as
  `scene.nav_roll`; NaN→0; synthetic scenes roll=0.

## Report case `antenna_patterns` (integration, Radar equation comparison)

Flat + hill incoherent cluttergrams, isotropic vs along-track dipole vs
5-element 0.5λ cross-track array, 1000 m AGL. First-run measured values
(gates recorded from these):

- flat 25–55° off-nadir band suppression: array 8.35 dB (gate ≥6),
  dipole 2.26 dB (gate ≤3.5 — along-track dipole barely touches
  cross-track clutter, only the ring's along-track azimuths)
- hill-echo band (10.8–12 µs, hill 1500 m cross-track ≈56° off nadir):
  array 9.45 dB (gate ≥7), dipole 3.79 dB (gate ≤5)
- 27–33° flat band iso-normalized ratio vs analytic azimuth average of g⁴:
  array 0.28 dB, dipole 0.01 dB (gates ≤1). NOTE: a SINGLE range bin holds
  only ~4–5 of the 50 m facets (3 m annulus), so single-bin ring checks are
  azimuth-sampling noisy for the array (first attempt: 1.5 dB); band
  averaging fixed it.
- Flat-scene split_sides L/R is inherently ~7% asymmetric (the discrete
  near-nadir r⁻⁴-dominant facets land on one side); the roll test therefore
  compares rolled/unrolled per side (measured left ×1.62, right ×0.035 for
  25° roll with a cos⁶ beam).

## Follow-ups / M24 notes

- The xOPR rerun should use `roll_source="nav"` + the cross-track array;
  Roll plumbed and ready.
- Tabulated is theta-only (axisymmetric). If a real 2-D pattern table is
  ever needed, extend with a phi grid + bilinear interp (new kind or a
  shape-static flag).
