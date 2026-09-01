# Area-mean full campaign — results and analysis (2026-09-01)

Six full lines reprocessed on branch atm-area-mean (pilots first, then
fulls; ~12.3 h total wall, detached driver after a WiFi-drop killed the
first session-attached attempt). Old pilots preserved at
outputs/<line>/pilot_pre_areamean_20260901; fixture-era full metrics at
outputs/<line>/full_pre_areamean_20260901_snapshot. PIG-N and PIG-S got
their FIRST full runs. All deltas dB rel own surface peak, sim - meas.

## Baseline caveat

The old fulls were FIXTURE-era (2026-08-28, pre-exponential): their
low-pass midcol errors were -60..-97 dB — unusable. The old-vs-new full
comparison therefore spans exponential + area-mean combined; the pilot
comparison isolates the area-mean step (9.2 -> 3.8 dB median |err|).

## Full-campaign state after area-mean (midcol, low passes)

| line | midcol delta | bedwin delta | tail m vs s | guards (low) |
|---|---|---|---|---|
| getz | -3.7/-4.1/-4.4 (all alt) | -0.4 low | -7.2/-6.4 | +29.0 clean |
| PIG-N | +1.3/-1.8/+0.3 | +2.0..-1.2 | -5.4..-6.0 / -6.7..-6.9 | +13..+14.5 |
| PIG-S | -6.8/-9.4/-6.9 | -1.2..-3.3 | -6.6..-7.0 / -6.0..-6.3 | +13.5..+14.2 |
| david | +0.8 / -18.8 / -7.4 | -7..-20 | ~match | -9.6/-2.1/+1.9 |
| geikie | -22.4 / -14.2 | -3.9 | (flat meas) | -6.8 |
| westcoast | -10.7/-4.5/-4.6 | -6..-16 | ~match | +2.3/-3.2/+1.8 |

Key numbers: getz altitude trend 80.2 (fixture) -> **17.9** vs scout ~20;
geikie 37.6 -> 12.5; PIG-N 26.1, PIG-S 27.2 (first measurements).

## Per-line reading

- **getz**: the campaign's best full line. Uniform -4 dB midcol offset at
  ALL altitudes (a single scatter-level tweak away), bed window exact at
  low altitude with a +29 dB clean guard, tail slope -6.4 vs measured
  -7.2, altitude trend on the scout value. Floating-zone shelf-base
  residual +2.8 dB on the low pass (the specular-regime test PASSES
  there). High passes: tail too fat (-0.8 vs -6.0) — the guard says the
  total-field tail is surface clutter there, so it is a clutter-shape
  issue, not bed physics.
- **PIG-N**: excellent first full — midcol within 2 dB, bed windows
  within 2 dB, clean guards. Sim tails now slightly STEEPER than measured
  (-6.8 vs -5.6 typ): the old "tails too shallow" defect is gone at full
  scale, slightly overshot here.
- **PIG-S**: midcol -7..-9 on the full line vs -13..-16 on the pilot
  window — the crevasse excess (E4) is diluted over 148 km, exactly as
  the along-track localization predicted. Both 9-10 km passes -5.3/-5.4.
  Guards clean. The per-facet map remains the lever for the residual.
- **david**: basler_2017 +0.8 (closed), but the three passes now DISAGREE
  with each other more than with the sim: measured midcol -50.0 / -33.9 /
  -42.6 for 2017/mkb22/mkb23 over the same ice. The sim is one consistent
  answer inside a 16 dB measured spread — the measured-product
  normalization audit (E5's instrument split) is now david's whole story.
- **geikie**: -22/-14, unchanged — englacial layers (parked). Also the
  only line whose altitude trend is far LOW (12.5): layer energy does not
  scale with altitude like surface clutter.
- **westcoast**: -4.5 on the two consistent seasons; p3_2016 still ~6 dB
  worse than its siblings (measured-side question stands).

## Cross-line findings

1. **The area-mean law generalizes from the 10 km pilot windows to the
   full 100-150 km lines**: no line degraded moving to full; getz/PIG-S
   actually improved (window-vs-line sampling). The 10 km pilot window is
   a usable but noisy proxy — PIG-S's window sits on its crevassed
   section, getz's window was slightly unrepresentative in the other
   direction.
2. **Remaining midcol residuals are the known non-roughness causes**:
   layers (geikie), crevasse map (PIG-S ~-7), measured-product
   normalization (david mkb passes, westcoast p3_2016). The
   surface-roughness level itself is now within ~4 dB wherever those
   don't apply.
3. **Floating-zone shelf bases run dim** (-6..-21 dB) on david and PIG-N
   under the grounded-anchored K, while getz's low pass passes (+2.8).
   Consistent with E5's attenuation-family bias and/or shelf-base
   reflectivity — the stratified re-solve is the next lever here.
4. **Bed-return tails now match at low altitude** (getz -6.4 vs -7.2,
   PIG-S -6.0 vs -6.7, david/westcoast within ~1 dB): the fixture-era
   "tails too shallow" defect is largely resolved by the roughness change
   + full-scale apertures; PIG-N slightly oversteep.
5. **HAPS design points**: getz's HAPS bed windows are now the only CLEAN
   ones in the campaign (guards +9.3/+6.5; all other lines -5..-40).
   HAPS midcol predictions dropped 4-7 dB on the lines whose entries got
   quieter (getz/geikie/westcoast) and rose on david. The 2026-08 HAPS
   studies (getz stratum sigma 24.9) need re-reading against these runs.

## Follow-ups (in rough priority)

1. Measured-product normalization audit: david's 16 dB inter-pass spread
   and westcoast p3_2016 (data QC, no modeling).
2. Per-facet surface scattering map for PIG-S (bed diffuse-channel
   machinery, crevasse mask) — now a ~7 dB full-line effect.
3. Stratified attenuation re-solve + shelf-base K: the floating-zone
   dimness (david -8..-21, PIG-N -6..-11).
4. High-altitude clutter tail shape (getz 9/11 km total-tail too fat).
5. Getz uniform -4 dB: within reach of the l-capped fit residual /
   pilot-window sensitivity — worth one look before calling it physics.

Logs: claude_notes/logs/areamean_{pilot,full}_*.log; branch atm-area-mean
(4 commits) still unmerged pending final adoption call.
