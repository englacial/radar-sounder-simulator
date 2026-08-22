# Facet convergence + calibration refit (getz, 2026-08-21 evening)

Sweep: real_10km, DEMOGORGN + adopted physics, facet_spacing_scale
1.0/0.7/0.5/0.35 (4/7/13 min for the refined runs).

| scale | bed win dB | below-bed | dB-std | p95-p50 |
|---|---|---|---|---|
| 1.0 | -51.76 | 0.394 | 7.74 | 13.3 |
| 0.7 | -51.51 | 0.356 | 7.02 | 10.9 |
| 0.5 | -51.65 | 0.346 | 6.76 | 10.0 |
| 0.35 | -50.86 | 0.352 | 6.57 | 9.4 |

Energy metrics converge by scale 0.5 (level +-0.5 dB, fraction flat);
texture creeps ~0.2 dB/step toward an extrapolated dB-std ~6.1 (glint
sharpening; saturates only if the surface spectrum is enriched).
CONVERGENCE POINT ADOPTED: scale 0.5.

Refit at scale 0.5, picked bed + adopted scattering, all 3 passes
(pilot_refit; 77 min seed+verify):
- A = 18.61 dB/km unchanged (dataset-only regression, sim-independent).
- gamma_surface solves to **+1.20 dB** (was +4.34 pre-adoption), spread
  1.91 dB over ALL THREE passes (-0.15/+1.20/+1.76, headrooms 13-34 dB),
  verify exact. ~3 dB of the getz "chain anomaly" was missing bed
  scattering physics; residual anomaly vs Fresnel +12.2 dB (surface
  -reference audit still open). Residuals at +1.20: +1.35/+0.06/-0.50.
- Bed windows now match measured to ~1 dB on every pass; 9/10 km midcol
  within 0.7-3 dB; low-pass midcol gap (-54.7 vs -68.8) persists (not bed
  physics; the known getz mid-column question).
- gamma remains PINNED at -10 manual in config (user decision stands);
  +1.20 is recorded solve evidence, not adopted.
