# gamma_surface = solve: design note (2026-08-20)

User decision: "Allow both options... setting a physical constant based on
minimizing residual from RSSNR is fine. Default to setting it to minimize
the residual." This note records how the solve works and why its first
implementation was replaced within the hour.

## The problem with the naive residual

The obvious target — zero the median of (sim − measured) total-field
bed-window levels — fails two ways:

1. **Contamination**: at high altitude the sim's bed window is dominated by
   off-nadir surface returns, whose level does not move with gamma. Feeding
   that residual back diverges or lands on a clutter-level match.
2. **Circular gating**: gating on "sim bed returns >= 10 dB above sim
   surface returns in the bed window" is itself gamma-dependent (the bed arm
   moves dB-for-dB with gamma). At the seed (−10 dB) only getz's low pass
   qualified; three of four lines would have refused to solve even though
   their data contain the answer.

## The power-sum inversion (implemented)

The simulated bed window is a power sum `S + B(gamma)`: surface returns S
fixed, bed returns B moving dB-for-dB with the mapping constant. Given the
measured level M (all three in dB rel own surface peak, from the standard
decomposition), invert per pass:

    gamma_required = gamma_seed + (M ⊖ S) − B      (⊖ = power subtraction)

Properties:
- **Exact at any contamination level** — the modeled clutter floor is
  subtracted before reading the bed.
- **Seed-invariant** (B scales with the seed; gamma_required does not), so
  the loop is one seed evaluation + one verification, and the verify run is
  a genuine linearity check, not a formality.
- **Honest refusal**: a pass whose measured level sits below the modeled
  clutter floor (M < S) can never be reproduced by ANY gamma — it holds no
  bed information and must not vote. Qualification = measured headroom
  above the modeled clutter floor >= `min_headroom_db` (1.0; conditioning
  degrades as 1/(1 − 10^(−h/10)), ~5x at 1 dB, so the cross-pass agreement
  is the real check).
- **Missing-physics detector**: qualifying passes demanding gammas more
  than `spread_warn_db` (6) apart mean the sim under-models something in
  one pass's regime. Warned loudly, median still recorded.

Settings in `config/analysis.yaml: gamma_surface_solve`; per-pass numbers in
metric `rssnr_level_residuals.gamma_solve`; history in
`run_config.json: calibration_resolution.gamma_surface_solve_history`.

## Retro-predictions from the gamma=−10 pilots (before the solve runs)

| line | per-pass gamma_required (headroom dB) | prediction |
|---|---|---|
| antarctica_getz | low +4.34 (32.5), 9km +3.84 (16.8), 10km +6.19 (18.4) | **+4.34**, spread 2.4 — consistent; carries the un-audited surface-reference anomaly (open item #1), now ~+15.4 dB vs Fresnel |
| greenland_westcoast | 2016 +8.68 (17.3), 2017 −3.73 (2.1), 2019 −4.98 (1.5) | **−3.73** — 2017/2019 independently reproduce the retired level-era −3.69; 2016 is the known-odd censored-gap pass |
| antarctica_david | Basler +7.41 (9.1); both MKB60 windows BELOW modeled clutter floor | **+7.41** from one pass; MKB60 disqualification is itself a finding (sim surface clutter ~10 dB above the whole measured level there — instrument diversity, worth a look) |
| greenland_geikie01_transit | low −12.96 (1.8), high +8.88 (5.8) | **median −2.04 with a 21.8 dB spread warning** — the englacial-scattering signature: the measured window holds column power the sim does not model. The solved value is suspect on this line by construction; the warning says so |

The naive total-field zeroing gamma (`gamma_surface_level_match_db`) stays
recorded as a diagnostic; on getz it gives +2.7 vs the inversion's +4.3 —
the difference is exactly the clutter contamination of the 9/10 km passes.

Commits: b23fc0c (solve allowed + default), 830255e (power-sum inversion).
