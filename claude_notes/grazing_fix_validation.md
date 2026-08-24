# Grazing-angle facet-lattice fix: design + validation (kernel-grazing-fix branch)

Diagnosis (S7 reconciliation, session scratchpad): the bed-window
surface-clutter floor on antarctica_david (annuli at 80-84 deg incidence)
is ~100% the COHERENT smooth-facet channel at both 60 and 195 MHz.
Mechanism: facet-lattice spatial aliasing -- at grazing the LPA phase ramp
across one facet is 2kL sin(theta) ~ 28-60 rad >> pi, the sinc tails/grid
lobes stop converging (non-monotonic +-10 dB vs facet size), 95.8% of the
floor power in a 4-fold cross on the facet-grid axes. Separately (S4), the
D_Phi facet-edge remainder gives sigma0 ~ 1/L (facet-size dependent),
+32 dB over physical PO at 60 MHz grazing.

## The fix (one opt-in switch: SimConfig.grazing_fix = GrazingFixConfig)

1. Coherent off-specular taper (coherent.py `_off_specular_taper`,
   multilayer.py target reflection): smooth/specular facet FIELD *=
   T(alpha) = exp(-tan^2(alpha)/(2 s_eff^2)), alpha off the facet normal
   (refracted arrival in the multilayer kernel). Removed power is DROPPED,
   not re-booked: it is grid aliasing, not physical power; the physical
   off-specular return is the D_Phi channel (below) plus the optional
   spec/diffuse split. s_eff default 0.05 (~sqrt(2) sigma/l scale).
2. Area-term-only D_Phi (roughness.d_phi `area_only`): per series term
   F_A*F_B -> pi l^2 Lx Ly/m * exp(-(A0^2+B0^2) l^2/(4m)) -- the Gerekos
   Appendix-C infinite-surface PO law per facet; sigma0 exactly facet-size
   invariant.

One switch for both because they are two faces of the same lattice-aliasing
artifact and the acceptance criterion (facet-size-invariant sigma0) needs
both. Statics in the jit factories; OFF traces the legacy program
bit-identically (full 419-test suite green untouched).

Plumbing: `--grazing-fix [S_EFF]` / spec `physics.grazing_fix: <s_eff>`;
threaded sim_cfg -> chunk_rid (`_gfx{s_eff:g}` suffix) -> chunk_meta
(`grazing_fix: {s_eff}` key, conditional like spec_diffuse/bed_rough) ->
chunk_digests -> process_standard_cached; companion + ablation arms share
it. Caches fork only when ON; composes with the antenna-realism branch's
instrument keys.

## Acceptance tests (tests/test_grazing_fix.py, 9 tests, gates inline)

- 60-85 deg band sigma0: 0.05 dB spread over facet-size x4 with fixes ON;
  OFF falls ~1/L by 4.3 dB (D_Phi edge remainder) and the on-grid-axis
  coherent sinc tails swing 8.8 dB NON-monotonically (7 sign flips over
  L = 4..6 m at 80-81 deg) -- the documented artifact; taper zeroes them.
- Nadir GO limit (sigma 3 m, l 30 m, 60 MHz): area-only D_Phi matches the
  Gaussian-slope GO law (per-axis msq slope 2 sigma^2/l^2) to 0.08 dB at
  nadir, <= 0.7 dB to 15 deg.
- Infinite-surface PO law: area-only D_Phi/(Lx Ly) == closed-form series to
  3e-9 dB at 0-84 deg, both frequencies.
- Bit-exactness: fixes OFF == legacy (default-kwarg guard + full suite);
  alpha = 0 arrivals bit-identical with taper ON; sigma = 0 still exact.
- S2 wall constraint: 40-deg tilted facet glint bit-identical (T=1 at
  alpha=0); off-glint level exactly T^2 = exp(-tan^2 alpha/s_eff^2), set by
  s_eff. Flat buried bed mirror field through the refracted path preserved
  to +0.41 dB (s_eff 0.05), shrinking with s_eff.

## Chunk-level validation (production david chunks, s7_run metric:
surface-arm bed-window mean power rel own surface peak, median)

| chunk | baseline | fixes ON | R's prediction (both fixes) |
|---|---|---|---|
| mkb60_2023 c03 (60 MHz, annulus 80.9-81.6 deg) | -66.0 dB | -137.3 dB | ~-134 dB |
| basler195_2017 c05 (195 MHz, 82.8-83.3 deg)    | -80.4 dB | -320.1 dB | physical-PO negligible |

Surface peak medians move <= 0.4 dB (taper leaves the specular surface
return alone). The 60 MHz post-fix floor lands within ~3 dB of the
predicted area-only-D_Phi (physical PO) level; the 195 MHz floor collapses
to numerical zero, as predicted.

## Pilot (mkb60_2023 pilot, standard processing; rel own surface peak, dB)

outputs/antarctica_david/pilot_kfix (s_eff 0.05) and pilot_kfix_s03
(s_eff 0.3) vs pilot_smoke and measured. pilot_smoke additionally carried
--bed-rough 0.1 0.886 and the spec_diffuse 0.5/3.0 split (BED-arm numbers
are not physics-identical; the surface arm is clean -- its only physics
delta is the fix).

| metric              | smoke   | kfix .05 | kfix .30 | measured |
|---------------------|---------|----------|----------|----------|
| sim midcol          | -40.19  | -60.31   | -58.90   | -44.73   |
| sim bed window      | -63.01  | -80.69   | -80.69   | -75.86   |
| surf-arm midcol     | -40.19  | -60.31   | -58.90   |          |
| surf-arm bed window | -63.05  | -122.90  | -123.25  |          |
| bed-arm bed window  | -82.57  | -80.70   | -80.69   |          |
| scout midcol/bedpk  | -2.40   | -39.46   | -39.29   | -23.52   |

measured noise floor (upper bound) -122.24 rel surface.

Findings:
- The bed-window surface-clutter floor drops ~60 dB to the measured noise
  floor (-122.9 vs -122.2): post-fix the sim no longer claims surface
  clutter above the bed, and the bed window flips to bed-dominated
  (-80.7, i.e. 4.8 dB below measured bed -- bed reflectivity physics, and
  this run carries no bed_rough/spec split).
- MIDCOL: drops 20 dB below smoke, now ~15.6 dB below measured -- and
  widening the taper to s_eff = 0.3 recovers only 1.4 dB. So the previous
  midcol "match" was NOT taper-shoulder facets (alpha <~ 25 deg would have
  come back at s_eff 0.3): it was the same far-off-normal sinc-tail
  aliasing channel. Genuine glints (alpha ~ 0 walls) are preserved exactly
  (test-gated), so the remaining measured-minus-sim midcol gap is real
  missing physics (firn/volume scattering, surface diffuse tails), not a
  taper-width tuning -- recorded as the follow-up implied by S2's
  constraint, which held only as long as the artifact supplied the power.
- Surface peak, surface alignment (2.16 bins) and bed_return_tail gates
  unchanged/passing; wall clock per chunk comparable to baseline.
