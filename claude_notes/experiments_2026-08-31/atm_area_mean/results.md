# ATM area-mean re-aggregation — validation results (branch atm-area-mean)

2026-08-31/09-01. All six pilots re-run with the *_expmean entries
(outputs/experiments_atm_areamean/<line>/, baseline outputs/<line>/pilot).
Mid-column deltas are sim minus measured, dB rel own surface peak.

## Headline

Median |mid-column error| across the 19 real passes: **9.2 -> 3.8 dB**
(mean 10.2 -> 7.1). Every line improved or held; no serious regression.

| line | midcol delta old -> new (low passes) | comment |
|---|---|---|
| david | -10.4/-15.4/-11.4 -> **+2.0/-5.5/-1.6** | the +10 dB line-ATM correction validated; residual spread is the instrument split (E5) |
| getz | -0.1 -> -2.2 (low); +4.9/+6.0 -> +2.3/+3.7 (high) | trades 2 dB at low for the high passes; bedwin low -15.5 -> -5.8 and the bed-window guard FLIPS CLEAN (-9.5 -> +8.1) |
| PIG-N | -0.5/-4.0/-2.2 -> +4.2/+0.8/+2.4 | slight overshoot; ~a wash (|err| ~2.2 -> 2.5) |
| PIG-S | -17.9/-20.3/-19.5 -> -13.1/-15.6/-14.3 | +4.7 as predicted; remaining ~14 dB = crevasse map (E4) |
| geikie | -23.5/-17.2 -> -21.4/-16.3 | layers, as expected — not this lever |
| westcoast | -14.2/-4.9/-5.3 -> -12.8/-3.6/-3.8 | +1.3 as predicted; p3_2016 gap is measured-side |

Altitude trends: getz 25.6 -> 25.4, PIG-N 28.8 -> 27.2, PIG-S 32.3 -> 30.4
(scout ~20; direction right, still steep).

## Trade-offs and side effects

- **david guards degrade** (bed-window bed-surface margin -0.0..-4.8 ->
  -9.6..-16.9): the brighter surface law pushes more clutter into the bed
  window. The bed-window *level* residuals barely moved.
- **getz low pass**: -0.1 -> -2.2 midcol (cost), but its bed window went
  from -15.5 dim to -5.8 and is now a clean bed measurement (guard +8.1).
  Mechanism note: dropping sigma 24.9 -> 4.6 cm removes the ~9 dB nadir
  coherent surface-peak suppression and the 8.7 dB double-count-guard gamma
  raise; the surface-peak normalization change moves every rel-dB metric.
- **HAPS design numbers move**: getz haps_14km midcol -37.5 -> -48.8
  (surface quieter at altitude under the line law). Any HAPS conclusions
  drawn from the getz stratum entry (sigma 24.9/l 24.5) should be
  revisited if this is adopted.
- PIG-N tips slightly positive (+4.2 worst) — within the campaign's noise
  but worth remembering it was the best-matched line.

## Status

Branch atm-area-mean holds: config/roughness/atm_b1.yaml +
atm_tier2_strata.yaml (new *_expmean entries + line mappings switched),
this analysis dir. Standard outputs untouched; validation runs live in
outputs/experiments_atm_areamean/. ADOPTION DECISION PENDING (user).
If adopted: re-run standard pilot+full, revisit HAPS studies for getz,
and consider refreshing the aa_grounded_1500_2500 / Greenland strata the
same way for future lines.
