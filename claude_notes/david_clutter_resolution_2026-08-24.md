# David surface-clutter discrepancy: root cause and resolution (2026-08-24)

Six parallel investigations (S1-S6) + reconciliation (S7) + two fix
branches. Question: why does simulated surface clutter obscure the bed on
antarctica_david when the measured products show none?

## Root causes (both confirmed, composable)
1. **Facet-lattice spatial aliasing of the COHERENT channel** (kernel
   artifact): at grazing (70-84 deg) the LPA phase ramp across a facet is
   2kL sin(theta) >> pi; the sinc/grid-lobe tails stop converging
   (non-monotonic +-10 dB in facet size; 95.8% of the floor power in a
   4-fold cross on the facet-grid axes). ~100% of the bed-window surface
   floor at BOTH 60 and 195 MHz. The incoherent Gerekos facet-edge term
   (sigma0 ~ 1/L, Dawson tail of Eqs 22-24) sits 28-36 dB below it but
   would become the next artifact floor. Fix: opt-in --grazing-fix =
   off-specular taper exp(-tan^2 a / 2 s_eff^2) on the coherent field
   (power dropped, not re-booked) + area-term-only D_Phi. Acceptance:
   sigma0(theta) facet-size-invariant to 0.05 dB; chunk floors matched
   predictions (-137 vs ~-134 @60; -320 @195). Branch kernel-grazing-fix.
2. **Isotropic-antenna placeholders for the david instruments**: real
   systems carry ~-58..-61 dB (195 MHz: 8-el tapered array + hann combine,
   from the products' own param structs) and -23..-42 dB (60 MHz: MARFA
   wing flat-plate dipole, right wing only, no beamforming) of two-way
   rejection at the clutter angles. Fix: array_tapered + finite_dipole
   antenna kinds, YAMLs updated with provenance, roll_source nav enabled
   (nav Roll verified all seasons), instrument-antenna fingerprint in the
   chunk keys (legacy states exempt byte-for-byte). Branch antenna-realism.

## Rejected: terrain occlusion (<=0.1 dB), wall glints (zero glint-capable
area; wall zone already matched), normalization (~0, wrong sign);
"different product processing" true only for 195 MHz (the beamformer).

## Validation (integration-fixes = main + both branches; 440 tests green)
pilot_fixed vs pilot_smoke (kept side by side on disk):
- david: surf-arm bed-window -63/-64/-75 -> -128/-134/-305 dB; every
  pass's bed window flips bed-dominated; ALL THREE passes now qualify for
  the gamma solve. New honest gaps: sim bed 7-22 dB dim (gamma_req spread
  15.6 dB, Basler vs MKB disagree -- element-pattern/absolute-cal
  uncertainty), midcol underpredicts everywhere (missing englacial/other
  physics -- now the getz/geikie-class open question).
- getz: artifact also removed (surf-arm -25..-83 dB); calibration robust
  (gamma_req 1.11 -> 1.16); the 9/10 km midcol match degrades to a 4-8 dB
  underprediction and the low-pass midcol becomes unexplained -- part of
  the old altitude-clutter closure was artifact; trend survives, numbers
  need re-derivation.

Branches (NOT merged, awaiting decision): kernel-grazing-fix (93b59e2,
091ff9f), antenna-realism (718ad25, b2ecb72), integration-fixes (merge of
both). Older parked branches: h2-bed-crosstrack, t2-bed-spectrum.
