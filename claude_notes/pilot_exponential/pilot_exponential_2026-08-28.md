# Pilot benchmark with the ATM exponential-ACF surface law (2026-08-28)

One question: what happens to all four study lines when the surface interface's
sub-facet roughness stops being the C&S firn-layer Gaussian fixture (sigma
4.9 cm, l 2.98 m) and becomes the measured ATM **exponential-ACF** pair
resolved per line and pass (`physics.surface_roughness: {source:
atm_exponential}`)?

Short answer: the long-standing **mid-column clutter under-prediction largely
closes** -- by 5 to 79 dB per pass, and to within 0.1 dB of measured on the
getz low pass -- and the **altitude trend**, the key deliverable, goes from a
+67 dB error to +6 dB on getz and from +30 to +6 dB on geikie. The cost is
that the simulated bed return is now *below* the simulated surface clutter in
the bed window on **every real pass of every line** (the bed-tail guard flips
True -> False, 11/11), so the bed-tail slope/excess numbers become upper
bounds, and the HAPS 14/20 km bed visibility drops 1-8 dB.

Code at HEAD `7584e2c` ("Roughness: Tier 2 per-stratum exponential table").

## Where the task description and the repo had drifted

The task named `config/experiments/pilot_smoke.yaml` ("4 lines, segment pilot,
real passes only, `surface_roughness: true`"). That file no longer exists:
commit `1090ccd` (2026-08-27, "Config simplification") folded it and 24 other
study specs into exactly two shipped experiments, `full.yaml` and
`pilot.yaml`, renamed the passes and re-audited the instruments
(`mcords5_p3_2016` 7 -> 2 elements, `mcords3_dc8_2016` 7 -> 3 at 0.45 lambda).
`pilot.yaml` IS the old pilot_smoke plus the cross-line HAPS 14/20 km design
points, and `tests/test_experiment_specs.py` asserts that exactly `full` and
`pilot` ship and that they differ only in segment -- so re-creating
`config/experiments/pilot_smoke.yaml` would have broken the test suite, and
`config/README.md` says outright that a one-off study is "a copy of one of
these with one thing changed; it does not get committed" to that directory.

So the study spec lives here instead:
**`claude_notes/pilot_exponential/pilot_exponential.yaml`**, byte-identical to
`config/experiments/pilot.yaml` except for the one line under `physics:`, and
verified so:

```
$ diff of RunSpec.run.model_dump(), pilot.yaml vs pilot_exponential.yaml
physics.surface_roughness:  True  ->  {source: atm_exponential}
(nothing else differs)
```

Run: `uv run python tools/run_basal_clutter.py --config
claude_notes/pilot_exponential/pilot_exponential.yaml --line <line>`; outputs
land in `outputs/<case>/pilot_exponential/`.

**If the exponential law is adopted for the campaign**, the change is the same
one line in BOTH `config/experiments/pilot.yaml` and `config/experiments/
full.yaml` (the test requires them to differ only in segment), plus the
`docs/roughness.md` statement of what the shipped experiments use. That edit
is deliberately NOT made here -- see the open issues at the end.

## Baselines preserved

The like-for-like fixture control is the 2026-08-27 `pilot` run of each line:
same code, same instruments, same passes, same spec but for the roughness
line. Its `run_config.json` differs from the exponential run's only in
outputs (case name, per-pass wall times, `trace_decomposition`,
`dropped_power_fraction`, `full_projection`) -- every input is identical.

| line | copied from | run date | copy |
|---|---|---|---|
| antarctica_david | `outputs/antarctica_david/pilot` | 2026-08-27 18:30 | `outputs/antarctica_david/pilot_fixture_2026-08-28` |
| antarctica_getz | `outputs/antarctica_getz/pilot` | 2026-08-27 23:05 | `outputs/antarctica_getz/pilot_fixture_2026-08-28` |
| greenland_geikie01_transit | `outputs/greenland_geikie/pilot` | 2026-08-27 18:19 | `outputs/greenland_geikie/pilot_fixture_2026-08-28` |
| greenland_westcoast | `outputs/greenland_westcoast/pilot` | 2026-08-27 18:05 | `outputs/greenland_westcoast/pilot_fixture_2026-08-28` |

The older `pilot_smoke` directories (the pre-rename, pre-instrument-audit runs
of **2026-08-24 20:59-21:24**, all four lines) were copied too, to
`outputs/<case>/pilot_smoke_fixture_2026-08-28/`. They are NOT used as the
control: the instrument audit alone moved getz mid-column by up to 4 dB.
Both copies carry the figures, `metrics.json`, `run_config.json` and
`report.html`; `runs/` and `proc_cache/` were deliberately NOT copied (tens of
GB, and the originals stay in place and usable).

Nothing was overwritten: the exponential runs went to a new directory.

## What the exponential law resolved to

From each run's `run_config.json` -> `surface_roughness.passes` (identical for
that line's HAPS passes, which ride the reference pass's spectrum):

| line | pass | sigma (cm) | l (m) | spectrum / stratum | usability |
|---|---|---|---|---|---|
| greenland_westcoast | p3_2016 | 4.78 | 10.00 | `westcoast_2016_exp` (site) | use |
| greenland_westcoast | p3_2017, haps_* | 3.33 | 1.038 | `westcoast_2017_exp` (site) | use |
| greenland_westcoast | p3_2019 | 3.77 | 2.590 | `westcoast_2019_exp` (site) | use |
| greenland_geikie01_transit | all | 5.15 | 5.276 | `geikie_2014` (site) | use |
| antarctica_david | all | 10.80 | 13.500 | Tier 2 `aa_grounded_500_1500` | **marginal** |
| antarctica_getz | all | 24.90 | 24.500 | Tier 2 `aa_grounded_lt500_m` | **marginal** |

Warnings, both expected and both from `surface_roughness_b1.resolve_
exponential`, once per pass (11 in total): *"marginal stratum: power law fits
better; exponential under-predicts wide-angle scatter by ~-1.6 dB (david) /
~-0.7 dB (getz) (median at 1.5 m)"*. The only other warnings in the logs are
36 `divide by zero encountered in log10` from `run_basal_clutter.py:927-929`
(the gamma-grid dB conversion), which the fixture runs emit as well -- benign.

Chunk cache rids forked as expected, e.g.
`..._srough_sr0.249_24.5_exp_att18.61_...`; nothing was reused from the
fixture chunks. Note the westcoast p3_2016 / david / getz correlation lengths (10.0, 13.5,
24.5 m) exceed the 7.47 m nominal facet spacing of the low passes, which is
the documented validity edge -- see the open issues.

## Runtimes (wall, whole run; sim wall in brackets)

| line | fixture sim wall | exponential sim wall | run wall |
|---|---|---|---|
| antarctica_getz | 202.8 s | 215.6 s (+6.3 %) | 4.0 min |
| antarctica_david | 312.4 s | 322.9 s (+3.4 %) | 5.6 min |
| greenland_westcoast | 514.1 s | 518.0 s (+0.8 %) | 9.0 min |
| greenland_geikie01_transit | 753.6 s | 763.9 s (+1.4 %) | 13.2 min |

All four sequential: 10:12 -> 10:44 (31.7 min). The feared 1.4x cost did not
appear: the Poisson series length only grows where sigma is large (getz needs
22 terms at 195 MHz vs 10 for the fixture, david 12), and the exponential
`W_m` is cheaper per term than the Gaussian's exponential.

## Metrics

Full tables: `outputs/pilot_exponential_comparison/metrics_comparison.md` and
`.csv` (210 rows: every numeric metric value, the clutter mid-column and
bed-rel-surface sub-fields measured vs both sims, the altitude-trend pairs,
per-pass tail slope/excess and the tail guard, surface alignment, RSSNR level
residuals, HAPS bed visibility, per-pass wall time). Compact headline below;
`err` = sim - measured (dB).

### Mid-column clutter (the headline)

| line | pass | measured | fixture (err) | exponential (err) |
|---|---|---|---|---|
| getz | dc8_2016_0km | -54.70 | -125.53 (-70.83) | **-54.79 (-0.09)** |
| getz | dc8_2016_9km | -35.39 | -39.64 (-4.25) | -30.50 (+4.89) |
| getz | dc8_2016_11km | -35.18 | -36.88 (-1.70) | -29.21 (+5.97) |
| david | basler_2017 (195 MHz) | -51.38 | -117.82 (-66.44) | **-61.76 (-10.38)** |
| david | baslermkb_2022 (60 MHz) | -44.57 | -65.31 (-20.74) | -59.95 (-15.38) |
| david | baslermkb_2023 (60 MHz) | -44.73 | -61.62 (-16.89) | -56.14 (-11.41) |
| geikie | p3_2014_low | -44.54 | -119.85 (-75.31) | **-68.03 (-23.49)** |
| geikie | p3_2017_high | -41.74 | -87.30 (-45.56) | **-58.93 (-17.19)** |
| westcoast | p3_2016 | -48.91 | -148.11 (-99.20) | **-69.47 (-20.56)** |
| westcoast | p3_2017 | -59.42 | -117.54 (-58.12) | **-64.29 (-4.87)** |
| westcoast | p3_2019 | -59.00 | -118.48 (-59.48) | **-67.01 (-8.01)** |

Every real pass improves. Median |error| over the 11 real passes: 58.1 dB
(fixture) -> 10.4 dB (exponential); worst case 99.2 -> 23.5 dB.

### Altitude trend (KEY DELIVERABLE)

| line | pair | measured | fixture (err) | exponential (err) |
|---|---|---|---|---|
| getz | 9km - 0km | 19.31 | 85.89 (+66.58) | 24.29 (+4.98) |
| getz | 11km - 0km | 19.52 | 88.65 (+69.13) | 25.58 (+6.06) |
| geikie | high - low | 2.80 | 32.55 (+29.75) | 9.10 (+6.30) |

### Bed level in the bed window, rel own surface peak

| line | pass | measured | fixture (err) | exponential (err) |
|---|---|---|---|---|
| getz | dc8_2016_0km | -55.51 | -61.87 (-6.36) | -71.05 (-15.54) |
| getz | dc8_2016_9km | -48.11 | -55.91 (-7.80) | -41.20 (+6.91) |
| getz | dc8_2016_11km | -47.72 | -56.09 (-8.37) | -39.75 (+7.97) |
| david | basler_2017 | -66.65 | -86.35 (-19.70) | -87.92 (-21.27) |
| david | baslermkb_2022 | -74.46 | -82.55 (-8.09) | -82.66 (-8.20) |
| david | baslermkb_2023 | -75.86 | -80.34 (-4.48) | -79.92 (-4.06) |
| geikie | p3_2014_low | -107.10 | -105.58 (+1.52) | -105.53 (+1.57) |
| geikie | p3_2017_high | -83.98 | -101.43 (-17.45) | -90.18 (-6.20) |
| westcoast | p3_2016 | -80.03 | -92.35 (-12.32) | -92.39 (-12.36) |
| westcoast | p3_2017 | -84.11 | -91.58 (-7.47) | -90.32 (-6.21) |
| westcoast | p3_2019 | -85.32 | -91.57 (-6.25) | -91.38 (-6.06) |

(These are the same numbers as `rssnr_level_residuals.per_pass_residual_db`.)
The bed level is a *bed* quantity and should not have moved: it is unchanged
to within 0.4 dB on david, geikie-low and westcoast. Where it does move --
getz all three passes, geikie high -- the bed window has filled with surface
clutter, which is the guard story below.

### Bed-return tail

| line | pass | slope meas | fix | exp | excess +2us fix -> exp | guard fix -> exp |
|---|---|---|---|---|---|---|
| getz | dc8_2016_0km | -8.93 | -5.73 | -4.46 | -2.85 -> -10.20 | +97.2 -> **-9.5 FAIL** |
| getz | dc8_2016_9km | -4.45 | -2.47 | -1.20 | -0.55 -> +10.03 | +30.7 -> **-26.8 FAIL** |
| getz | dc8_2016_11km | -2.91 | -1.94 | -1.03 | -3.98 -> +9.96 | +26.7 -> **-28.5 FAIL** |
| david | basler_2017 | -7.62 | -6.21 | -4.54 | -15.79 -> -18.41 | +198.9 -> **-4.9 FAIL** |
| david | baslermkb_2022 | -5.40 | -3.98 | -3.28 | -11.89 -> -12.10 | +36.8 -> **+0.8 FAIL** |
| david | baslermkb_2023 | -5.18 | -3.72 | -3.18 | -8.87 -> -7.80 | +35.9 -> **-0.0 FAIL** |
| geikie | p3_2014_low | -0.37 | -5.78 | -4.79 | -0.07 -> +0.74 | +177.0 -> **-0.6 FAIL** |
| geikie | p3_2017_high | -1.01 | -4.43 | -0.78 | -25.36 -> -6.10 | +147.8 -> **-21.0 FAIL** |
| westcoast | p3_2016 | -2.75 | -6.12 | -4.64 | -21.48 -> -19.77 | +190.4 -> **-2.7 FAIL** |
| westcoast | p3_2017 | -4.44 | -7.04 | -3.40 | -4.73 -> +2.72 | +186.5 -> **-10.5 FAIL** |
| westcoast | p3_2019 | -7.69 | -6.78 | -4.41 | -10.69 -> -7.38 | +186.5 -> **-6.2 FAIL** |

The guard (`sim bed returns - sim surface returns`, minimum over the fit
window, threshold +10 dB) flips True -> False on **all 11 real passes**. This
is the one flag that flipped anywhere in the metric set; every threshold-op
metric (`surface_alignment_*`, <= 5 bins) still passes on both arms and barely
moves (max change 0.35 bins).

### HAPS design points

Mid-column clutter goes UP on getz (+6.6/+7.3 dB at 14/20 km) and david
(+0.1/+1.2), DOWN on westcoast (-7.8/-8.2) and geikie (-2.8/-2.3): the
direction follows the resolved sigma, not the altitude. Bed visibility
(`bed_over_surface_clutter_in_bed_window_db`) degrades on 7 of 8 points:
david -10.8 -> -14.3 and -16.2 -> -17.0; getz -4.0 -> -7.8 and -11.8 ->
-15.2; geikie -13.6 -> -21.9 and -17.1 -> -20.4; westcoast -19.4 -> -20.5 but
-24.3 -> -21.7 at 20 km.

## Figures

Stacked fixture-over-exponential pairs (the runs' own PNGs, no re-rendering),
under `outputs/pilot_exponential_comparison/`:

```
antarctica_david_{radargrams,decomposition,decomposition_trace}.png
antarctica_getz_{radargrams,decomposition,decomposition_trace}.png
greenland_geikie01_transit_{radargrams,decomposition,decomposition_trace}.png
greenland_westcoast_{radargrams,decomposition,decomposition_trace}.png
```

Built by `claude_notes/pilot_exponential/compare.py` (also writes the metrics
tables). `antarctica_getz_decomposition_trace.png` is the clearest single
picture of the whole result: in the top (fixture) panel the simulated surface
tail sits 40-70 dB below the measured trace and the bed peak stands 105 dB
above the surface clutter; in the bottom (exponential) panel the simulated
tail lies on the measured trace on the low pass, and the bed peak is buried
19-20 dB below the surface clutter on the 9 and 11 km passes.

## Per-line reading

**antarctica_getz** (Tier 2 `aa_grounded_lt500_m`, sigma 24.9 cm / l 24.5 m,
marginal). The most dramatic line. The 478 m low pass's mid-column clutter
lands within 0.09 dB of measured (was 70.8 dB low), and the altitude trend --
the deliverable the whole line exists to test -- goes from +67/+69 dB error to
+5.0/+6.1 dB. But the high passes now *over*-predict by ~5-6 dB, the bed level
over-predicts by +7/+8 dB where it under-predicted by -8, the tail excess at
+2 us swings from -0.5 to +10 dB, and the guard fails on all three passes: the
sim's bed window is now surface clutter, not bed. Read together, the
exponential at sigma 24.9 cm is clearly too much scatter at the high passes'
wider angles while being about right at the low pass -- consistent with the
stratum's own "marginal, power law fits better" flag, and with sigma being 2.5x
the non-parametric `sigma_bl30` of 12.1 cm. This line needs a site-specific
ATM fit (there IS ATM on the getz line: `getz_2016` is a power law in
atm_b1.yaml) or the Matern option, not the Tier 2 stratum median.

**antarctica_david** (Tier 2 `aa_grounded_500_1500`, sigma 10.8 cm / l 13.5 m,
marginal; no ATM on the line). The 195 MHz basler pass improves by 56 dB
(-66.4 -> -10.4 dB error) and the two 60 MHz MARFA passes by ~5 dB each
(-20.7 -> -15.4, -16.9 -> -11.4). The frequency ordering is the interesting
part: the fixture's error was strongly frequency-dependent (66 dB at 195 MHz,
17-21 dB at 60 MHz) and the exponential removes most of that asymmetry,
leaving a fairly uniform 10-15 dB deficit -- which now looks like a
*non-surface* residual (englacial, or the bed-arm diffuse tail) rather than a
surface-law artifact. Bed levels are untouched (<= 0.4 dB) except the 195 MHz
pass drifting 1.6 dB further low. The tails all get shallower and thus move
AWAY from the measured slopes (-7.6 meas vs -6.2 fix vs -4.5 exp), and the
guard fails on all three passes -- on baslermkb_2023 by a hair (min +0.0 dB
against a +10 dB threshold), so david is the line where "bed vs clutter in the
bed window" is now a coin flip.

**greenland_geikie01_transit** (site entry `geikie_2014`, sigma 5.15 cm /
l 5.28 m, usability "use" -- the best-evidenced surface in the campaign,
exponential ACF best in 83 % of 1 km blocks). Both passes improve by ~28-52 dB
but remain 17-23 dB low, the largest residual deficit of the four lines. The
altitude trend improves from +29.8 to +6.3 dB error. This line also shows the
cleanest *secondary* win: the high pass's bed level residual halves (-17.4 ->
-6.2 dB), its tail slope goes from -4.43 to -0.78 against a measured -1.01,
and its +2 us excess from -25.4 to -6.1 dB -- i.e. the high pass's bed
window stops being anomalously empty. Given that this is the site where the
exponential is best justified and it still leaves 17-23 dB, the mid-column
deficit is NOT purely a surface-roughness-law problem.

**greenland_westcoast** (site entries `westcoast_*_exp`, sigma 3.3-4.8 cm /
l 1.0-10.0 m, "use" but fitted to surfaces whose best family is a power law).
The largest raw improvements: p3_2016 by 78.6 dB, p3_2017 by 53.3, p3_2019 by
51.5, leaving -20.6/-4.9/-8.0 dB errors. p3_2017 and p3_2019 are the closest
any pass gets to measured on a "use" surface. Bed levels barely move (<= 1.3
dB), which is the reassuring control. Tail slopes improve on 2016 and 2017 and
worsen on 2019. The oddity is the HAPS points going 8 dB *quieter*: they ride
the reference pass p3_2017, whose exponential fit has l = 1.04 m -- an order
of magnitude shorter than the other two years -- so the HAPS design points on
this line are now anchored to the year with the least wide-angle scatter. That
is a per-pass-mapping artifact worth fixing before quoting HAPS numbers here.

## Open issues / recommendations

1. **Do not adopt globally on this evidence.** The mid-column verdict improves
   everywhere, but the bed-tail guard fails on 11/11 real passes, which means
   every bed-tail slope and excess in an exponential run must be read as an
   upper bound. The bed-arm results the campaign has already published were
   obtained on guard-passing runs.
2. **getz and david are running Tier 2 stratum medians flagged marginal**, at
   sigma 24.9 and 10.8 cm -- both far above the fixture's 4.9 cm and above
   their own `sigma_bl30` references. getz has real ATM on the line
   (`getz_2016`, power law); a site-specific exponential fit for it (like the
   `westcoast_*_exp` entries) would be a better test than the stratum median.
3. **The correlation lengths are past the validity edge on the low passes**:
   l = 10.0 m (westcoast p3_2016), 13.5 m (david) and 24.5 m (getz) against a
   nominal facet spacing of 7.47 m on every low pass (11.2 m on david's MARFA
   and geikie's high pass; the high getz passes are at 32-35 m). No
   `corr_length_m exceeds the facet size` warning fired, because
   `simulate._roughness_args` compares l against the MAXIMUM facet edge in the
   scene and the far-off-nadir facets are much larger than the nominal
   spacing -- so the check passed while most facets are in fact smaller than
   l. Roughness at those scales is facet tilt and belongs in the DEM; the getz
   over-prediction at the high passes may be exactly this double count.
4. **westcoast HAPS passes inherit p3_2017's l = 1.04 m.** Either pin the HAPS
   design point to a stated spectrum or use the line default deliberately.
5. If adopted, edit BOTH `config/experiments/pilot.yaml` and `full.yaml` (the
   test requires them to differ only in segment) and update
   `docs/roughness.md`; do not re-introduce a `pilot_smoke.yaml`.

## Files

- spec: `claude_notes/pilot_exponential/pilot_exponential.yaml`
- comparison script: `claude_notes/pilot_exponential/compare.py`
- logs: `claude_notes/pilot_exponential/logs/{driver,<line>}.log`
- runs: `outputs/<case>/pilot_exponential/`
- fixture control copies: `outputs/<case>/pilot_fixture_2026-08-28/`
- legacy 2026-08-24 copies: `outputs/<case>/pilot_smoke_fixture_2026-08-28/`
- comparison: `outputs/pilot_exponential_comparison/`
