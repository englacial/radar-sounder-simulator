# Path B1: effective Gaussian surface roughness from measured ATM spectra, validated on the real passes (2026-08-26/27)

Branch `roughness-b1` (off `waveform-explicit-chirp`), worktree
`.claude/worktrees/agent-a546a4adcd12397d0`. Not merged.

Inputs (copied to `claude_notes/roughness_b1/atm_inputs/`): the plan
(`atm_roughness_plan_2026-08-26.md`), the ATM results note, `summary_tables.md`,
the per-line `summary_<date>.json` and 1 km block CSVs from `outputs/atm_roughness/`.
Code: `tools/surface_roughness_b1.py` (rule + resolver), `config/roughness/atm_b1.yaml`
(spectra + provenance), `claude_notes/roughness_b1/fit_b1.py` (builds the YAML, the
tables `outputs/roughness_b1/b1_table.md` and `outputs/roughness_b1/residuals.png`),
`claude_notes/roughness_b1/tabulate.py` (measured vs fixture vs B1 tables),
`tests/test_surface_roughness_b1.py`.

## 1. The matching rule

Convention (same as the ATM note and the Gaussian ACF rho = exp(-r^2/l^2) the Gerekos
kernel assumes): 2-D two-sided PSD with int S d^2k = sigma^2,

    S_G(k) = sigma^2 l^2 / (4 pi) exp(-k^2 l^2 / 4)      (Gaussian)
    S(k)   = A k^-beta                                     (power law, beta = 2H + 2)
    S(k)   = sigma^2 l^2 / (2 pi) (1 + k^2 l^2)^-3/2       (exponential, geikie)

(the task's "pi sigma^2 l^2 exp(...)" is the same PSD in the (2 pi)^2 convention.)
The first-order (m = 1) term of the Gerekos incoherent series D_Phi is the facet area
times this PSD at the transverse wavenumber 2 k sin(theta); with sigma^2 K^2 << 1 (true
for every B1 pair below, sigma ~ 1 cm) the higher terms are negligible, so matching S_G
to the measured S at k_B = 2 k0 sin(theta_c) makes the kernel's wide-angle term exact at
theta_c. Coherent (nadir) term: exp(-sigma^2 K^2 / 2) changes by < 0.1 dB, so the
surface peak the metrics normalise to is unchanged.

Measured spectrum per line: westcoast (per year) and getz = power law through the
LINE-MEDIAN per-block best-family S(k_B) at 5 / 1.5 / 1.0 / 0.75 m (the ATM note's
recommendation; on getz the pooled fit is 3-5 dB higher and is not used); geikie = the
block-median exponential ACF (sigma 4.9 cm, l 5.3 m; best in 83 % of blocks). Values
and provenance are in `config/roughness/atm_b1.yaml`.

Three second constraints were compared (RMS of 10 log10(S_G/S_meas), theta_c = 30):

| rule | over Lambda 1-5 m (150-300 MHz) | over the theta_c = 20-40 deg Bragg band | comment |
|---|---|---|---|
| **tangent**: S and d ln S/d ln k at k_B -> l = sqrt(2 beta_loc)/k_B, sigma^2 = 4 pi S(k_B) e^{beta_loc/2}/l^2 | 3.3-8 dB (60 MHz: 42-55 dB, see below) | **0.5-0.7 dB** at every f | adopted |
| band variance 0.5-30 m (sigma = band RMS) + S(k_B) | 5-11 dB | 2-8 dB | sigma fixed at 2.8-4.2 cm forces l onto the Gaussian tail (l >= 0.9 m at >= 225 MHz), wrong slope |
| two wavenumbers: k_B and 2 pi/5 m | 3.6-12 dB | 1.7-7 dB | exact at two points, wrong slope between |

Full table: `outputs/roughness_b1/b1_table.md` ("Matching-rule comparison"). The tangent
rule wins over 1-5 m at 150-225 MHz (the Bragg wavelengths sit inside the band), ties at
300 and loses at 400 MHz and at 60 MHz over 1-5 m -- but at 60 MHz the 1-5 m band lies
entirely ABOVE k_B (Lambda_B = 5 m), where any Gaussian tail collapses; the band that
matters for a 60 MHz sounder's clutter angle is Lambda 3.6-7.3 m, where the tangent is
within 0.7 dB. One rule for all frequencies keeps the fixtures comparable, and the
tangent is the only one exact in value AND slope at theta_c, so the theta_c = 20-40 deg
spread is < 1 dB by construction. Band variance: only the sub-facet band 0.5-30 m is a
legitimate constraint (>= 30 m is facet tilt already in the DEM; facets are 27-70 m in
these runs); the 0.5-100 m band would inflate sigma by +1.5 (geikie), +3.0 (wc 2017),
+5.2 (getz), +5.4 (wc 2019), +6.8 dB (wc 2016) -- recorded in `b1_table.md`, not used.
The adopted tangent rule has no band dependence, so no run depends on this choice.

### (sigma_eff, l_eff) at theta_c = 30 deg (cm / m); S_meas(k_B) in dB re m^4

| spectrum | 60 MHz (5 m) | 150 (2 m) | 195 (1.54 m) | 225 (1.33 m) | 300 (1 m) | 400 (0.75 m) |
|---|---|---|---|---|---|---|
| westcoast_2016 | 1.47 / 2.04 (-48.6) | 0.81 / 0.82 (-61.7) | 0.69 / 0.63 (-65.5) | 0.62 / 0.54 (-67.5) | 0.52 / 0.41 (-71.6) | 0.43 / 0.31 (-75.7) |
| westcoast_2017 | 1.61 / 1.79 (-47.3) | 1.26 / 0.72 (-57.4) | 1.18 / 0.55 (-60.3) | 1.13 / 0.48 (-61.8) | 1.05 / 0.36 (-65.0) | 0.97 / 0.27 (-68.2) |
| westcoast_2019 | 1.89 / 1.96 (-46.2) | 1.19 / 0.78 (-58.2) | 1.04 / 0.60 (-61.7) | 0.96 / 0.52 (-63.5) | 0.83 / 0.39 (-67.3) | 0.72 / 0.29 (-71.1) |
| geikie_2014 (exp.) | 2.39 / 1.93 (-44.1) | 1.54 / 0.78 (-55.9) | 1.35 / 0.60 (-59.3) | 1.26 / 0.52 (-61.2) | 1.09 / 0.39 (-64.9) | 0.95 / 0.29 (-68.7) |
| getz_2016 | 1.60 / 1.95 (-47.6) | 1.01 / 0.78 (-59.6) | 0.89 / 0.60 (-63.0) | 0.83 / 0.52 (-64.8) | 0.72 / 0.39 (-68.6) | 0.62 / 0.29 (-72.3) |
| getz pilot window s 30-40 (sensitivity) | 4.80 / 1.84 (-37.9) | 3.53 / 0.74 (-48.5) | 3.24 / 0.57 (-51.5) | 3.09 / 0.49 (-53.2) | 2.80 / 0.37 (-56.5) | 2.55 / 0.28 (-59.9) |
| fixture (S_G at k_B) | 4.95 / 2.98 (-42.9) | (-121) | (-197) | (-238) | (-408) | (-704) |

The runs resolve per pass carrier: getz at 190 MHz gives 0.90 cm / 0.614 m, geikie
1.35 / 0.599, westcoast 2016 (200 MHz) / 2017 / 2019 0.67 / 0.612, 1.18 / 0.551,
1.04 / 0.602. theta_c sensitivity of the pair: l scales as 1/sin(theta_c) (20 deg:
x1.46, 40 deg: x0.78 vs 30 deg), sigma by +-10-25 % (full table in `b1_table.md`); a
20 deg pair evaluated at the 30 deg wavenumber is 0.7 dB off, a 40 deg pair 0.6 dB.
Year-to-year (westcoast): S(1.54 m) = -65.5 / -60.3 / -61.7 dB for 2016 / 2017 / 2019
(the ATM note's +-3 dB): the 2016 pair is 5 dB weaker than 2017 in the Bragg band.

### Residual of the effective Gaussian vs the measured spectrum (theta_c = 30, dB)

| spectrum / f | 0.75 m | 1 m | 1.5 m | 2 m | 3 m | 5 m | 10 m | 20 m | 30 m |
|---|---|---|---|---|---|---|---|---|---|
| wc 2017 / 60 | -218 | -114 | -42 | -19 | -4.1 | 0 | -3.5 | -10 | -14 |
| wc 2017 / 195 | -9.7 | -2.8 | 0 | -0.6 | -3.3 | -8.0 | -15 | -23 | -27 |
| wc 2017 / 300 | -1.1 | 0 | -1.4 | -3.5 | -7.2 | -12 | -20 | -28 | -32 |
| geikie / 195 | -11.5 | -3.3 | 0 | -0.8 | -3.9 | -9.3 | -18 | -25 | -28 |
| getz / 195 | -11.5 | -3.3 | 0 | -0.8 | -3.9 | -9.4 | -18 | -27 | -32 |
| fixture vs wc 2017 | -637 | -344 | -137 | -66 | -17 | +4 | +8 | +3 | 0 |

(all rows in `b1_table.md`; figure `outputs/roughness_b1/residuals.png`.) The B1 pair is
exact within ~1 dB for +-0.3 octave around k_B, 3-4 dB low one octave away on the
long-wavelength side (angles below theta_c), 10 dB low at 0.75 m for the 195 MHz pair
(angles above ~45 deg, the grazing part of the surface response that sets the
LOW-altitude mid-column clutter). The fixture is +4..+8 dB high at 5-20 m and 66-640 dB
low at <= 2 m.

## 2. Plumbing

- `physics.surface_roughness` in an experiment spec: `true`/`false` (fixture on/off,
  unchanged), `{sigma_m, corr_length_m}` (explicit pair), or `{source: atm_b1[,
  theta_c_deg]}` (resolved per line, pass and carrier f0 at run time from
  `config/roughness/atm_b1.yaml`; synthetic passes use the line default, westcoast per
  year by pass name). `run()` receives `surf_rough` as bool / [sigma, l] / dict;
  `resolve_surf_rough()` turns it into a pair per pass inside `simulate_pass` and
  `process_standard_cached`.
- Cache keys: `chunk_rid` appends `_sr<sigma>_<l>` and `chunk_meta` adds
  `surf_rough_sigma_l` ONLY for a non-fixture pair; `True` and the fixture pair give
  byte-identical rid/meta (test `test_fixture_and_bool_keep_legacy_keys`). The fixture
  pilot_smoke replay on this branch hit every cached chunk (8.7 s).
- `run_config.json` gets `surf_rough` (as specified) and `surface_roughness` = fixture +
  per-pass resolved pair with spectrum id, family, theta_c, k_B, Lambda_B, S_meas(k_B);
  the report dumps the config. The runner prints the resolved pair per pass.

## 3. Validation on the real passes (segment `pilot`, adopted `pilot_smoke` physics)

`pilot_smoke` (fixture; the 2026-08-25 outputs copied in and replayed from cache) vs
`pilot_smoke_b1` (identical except `surface_roughness: {source: atm_b1}`; calibration,
bed, reflectivity, processing untouched). Mid-column clutter = simulated mid-column
power rel. own surface peak vs the measured product. Figures:
`outputs/<line>/pilot_smoke_b1/{radargrams,decomposition,decomposition_trace,bed_tail}.png`
(fixture: `outputs/<line>/pilot_smoke/`); line dirs greenland_westcoast, greenland_geikie,
antarctica_getz (worktree outputs, not committed).

### greenland_westcoast (P-3, ~500 m AGL; low-altitude control)

| pass | AGL m | f0 | B1 pair | measured midcol | fixture (err) | B1 (err) |
|---|---|---|---|---|---|---|
| p3_2016 (200 MHz system) | 484 | 200 | 0.67 cm / 0.61 m | -48.9 | -149.6 (-100.6) | -80.9 (-32.0) |
| p3_2017 | 500 | 195 | 1.18 / 0.55 | -59.4 | -117.5 (-58.1) | -69.7 (-10.2) |
| p3_2019 | 476 | 195 | 1.04 / 0.60 | -59.0 | -118.5 (-59.5) | -72.2 (-13.2) |

Nothing else moved: bed level rel. surface -94.2 / -90.7 / -92.1 dB (fixture -94.3 /
-91.6 / -91.6; measured -80.0 / -84.1 / -85.3), bed-tail excess at +2 us -26.2 / -2.4 /
-10.4 dB (fixture -26.5 / -4.7 / -10.7), surface alignment p90 0.85 / 0.67 / 1.37 bins
(unchanged). The 2017/2019 mid-column gap closes from -58 to -10..-13 dB; the remaining
deficit is the expected form error: at 500 m AGL the mid-column delay maps to surface
angles of 60-70 deg (Lambda 0.8-0.9 m), where the 30 deg B1 pair is 3-10 dB below the
measured spectrum (residual table), plus the along-track aliasing of the 15 m posting.
p3_2016 stays 32 dB low: its measured mid-column is 10 dB above the other two years on
the same window with a different (200/100 MHz) system, while the 2016 ATM spectrum is
the WEAKEST of the three years (-65.5 dB) -- that pass's excess is not a
surface-roughness signal in the ATM data (product radiometry / img_comb, see the
line's measured_caveats).

### greenland_geikie01_transit (P-3 2014 low 465 m vs 2017 high 2483 m; altitude pair)

| pass | AGL m | B1 pair | measured midcol | fixture (err) | B1 (err) |
|---|---|---|---|---|---|
| low | 465 | 1.35 cm / 0.60 m | -44.5 | -119.8 (-75.3) | -69.6 (-25.1) |
| high | 2483 | 1.35 / 0.60 | -41.7 | -87.3 (-45.6) | -60.3 (-18.6) |

Altitude trend high - low: measured +2.8 dB, fixture +32.5 (err +29.8), **B1 +9.3 (err
+6.5)**. Bed level rel. surface: low -104.0 (fixture -105.6, measured -107.1), high
-97.0 (fixture -101.4, measured -84.0); bed-tail excess at +2 us: low +2.5 (fixture
-0.1), high -14.5 (fixture -25.4; coverage-limited pass); surface alignment p90 0.92 /
0.83 bins (0.79 / 0.85). The high pass's bed tail moved 11 dB towards measured because
the tail there is surface clutter at the bed delay, which B1 now supplies. The 2017
high pass has no ATM (the 2014 spectrum is applied to both passes; the ATM note found
geikie statistically uniform along the line, dry-snow April surface in both years).

### antarctica_getz (DC-8 2016; 442 m vs 9150 / 10684 m -- the real high-altitude data)

| pass | AGL m | B1 pair | measured midcol | fixture (err) | B1 (err) |
|---|---|---|---|---|---|
| real_low | 442 | 0.90 cm / 0.61 m | -54.7 | -125.8 (-71.1) | -71.3 (-16.6) |
| real_9km | 9150 | 0.90 / 0.61 | -35.4 | -43.1 (-7.7) | -49.3 (-13.9) |
| real_10km | 10684 | 0.90 / 0.61 | -35.2 | -39.4 (-4.2) | -48.0 (-12.8) |

Altitude trend (high - low): measured +19.3 / +19.5 dB; fixture +82.7 / +86.4 (err +63 /
+67); **B1 +22.0 / +23.3 (err +2.7 / +3.8)**. Bed level rel. surface at 9 / 10 km: B1
-54.4 / -53.5 (fixture -59.3 / -60.2, measured -48.1 / -47.7), 5 dB closer; bed-tail
excess at +2 us: +6.3 / +3.6 (fixture -5.0 / -7.9): B1 now slightly OVERshoots the
post-bed tail at altitude where the fixture undershot it; low pass unchanged (-2.3 vs
-3.0). Surface alignment p90 at 9/10 km is 21-29 bins in BOTH runs (pre-existing, not
roughness). The pilot window s 30-40 km straddles the ATM regime boundary at 32 km
(2.3 cm -> 3.6 cm per octave); its own block-median spectrum is +11.5 dB above the
line median at 1.5 m (pilot-window pair 3.2 cm / 0.57 m at 195 MHz, in the YAML as
`getz_2016_pilot_window`, not used). Since the clutter term is linear in S(k_B), using
it would raise the B1 mid-column by ~+11 dB: the 9/10 km passes would then land at
~-38 / -37 dB (measured -35.4 / -35.2) and the low pass at ~-60 (measured -54.7). The
high-altitude absolute level is therefore bracketed by the line-median (13 dB low) and
the window-median (~2 dB low) spectra; the along-line scatter of the ATM PSD
(+-8-12 dB on getz) is the dominant uncertainty, not the matching rule.

### Verdict: does B1 reproduce the real high-altitude clutter better than the fixture?

Yes, decisively for the altitude DEPENDENCE, which is what a 14-20 km prediction
extrapolates: the fixture's high-minus-low error is +30 dB (geikie 2.5 km) and +63-67 dB
(getz 9-10 km); B1's is +6.5 and +3-4 dB. For the ABSOLUTE high-altitude level the
fixture was accidentally close at getz 9-10 km (-4..-8 dB, by being 70 dB wrong at the
low pass and ~80 dB wrong in slope) and B1 is -13..-14 dB with the line-median spectrum,
~-2 dB with the pilot-window spectrum. Low-altitude mid-column errors go from
-58..-75 dB to -10..-25 dB on every line, with bed level, bed tail and surface
alignment unchanged at low altitude.

What B1 does NOT fix:
- Form away from theta_c: the pair is exact at k_B and 3-10 dB low half to one octave
  away on either side (residual table); the low-altitude mid-column (60-70 deg surface
  angles, Lambda ~0.8 m) is under-predicted by 10-25 dB, and a pass whose clutter angle
  differs from 30 deg needs its own theta_c (or path B2, a tabulated power-law ACF in
  the kernel).
- Along-track aliasing of the 15 m posting at the clutter angle (HAPS summary, round 7)
  is untouched; part of the residual mid-column error is sampling, not physics.
- Firn / englacial layers are absent (one surface interface); the p3_2016 westcoast
  excess and the geikie absolute levels are not explained.
- Getz per-segment roughness (10-15 km regimes) is not resolved: one spectrum per line.
- Runtime: B1 chunks take 1.7-1.8x the fixture (l ~ 0.6 m needs more D_Phi terms per
  facet): westcoast pilot 14.4 min (fixture ~8), geikie 20.5, getz 5.6 min.

## 4. HAPS 14 km frequency ladder with B1 (westcoast pilot, `wc_hd_b1`)

Same ladder as the design study: 8-element Hann-tapered array on a 10 m span, T = 8 us,
Hann compression, explicit-chirp construction, B = f/2 (100 MHz at 300 MHz, the alias
rule), posting_div 1, riding the p3_2017 geometry at 14 km. B1 resolves per carrier
from the westcoast_2017 spectrum. Metric `<pass>_bed_visibility` = bed arm over
surface arm in the bed window (dB); fixture / l = 1 m rows from
`claude_notes/haps_design_study/summary.md`; the fixture column is the same 8-el ladder re-run here (`wc_hd_b1fix`: -23.5 / +17.3 / +45.5 / +69.6, midcol -47.5 / -45.2 / -49.3 / -53.8), matching the summary
(the `l = 1 m` column keeps the summary element counts).

| design | B1 pair (cm / m) | bedvis B1 | surf arm @bed | bed arm | midcol B1 | bedvis fixture | bedvis l = 1 m |
|---|---|---|---|---|---|---|---|
| 60 MHz, 8 el | 1.61 / 1.79 | **-20.6** | -68.1 | -88.7 | -54.5 | -23.5 (8 el, wc_hd_b1fix; summary 5 el -22.9) | -29.1 (5 el) |
| 150 MHz, 8 el | 1.26 / 0.72 | **-21.1** | -63.1 | -84.2 | -52.4 | +17.3 (8 el; summary 11 el +17.2) | -30.3 (11 el) |
| 225 MHz, 8 el | 1.13 / 0.48 | **-22.4** | -61.2 | -83.6 | -51.4 | +45 | -23.6 |
| 300 MHz, 8 el | 1.05 / 0.36 | **-24.4** | -60.9 | -85.3 | -51.2 | +69.6 | -18.7 |

Under the measured spectrum the frequency lever is GONE: bed visibility is -21 to -24 dB
at every carrier (the surface arm at the bed delay rises 7 dB from 60 to 300 MHz while
the bed arm gains ~3 dB), and the ordering even reverses (300 MHz is the worst by 4 dB).
The +90 dB spread of the fixture ladder was the Gaussian tail; the l = 1 m proxy had the
right order of magnitude but still favoured high frequency by 10 dB, because a fixed
l = 1 m understates the 60 MHz Bragg PSD (Lambda 5 m) and overstates the 300 MHz one.
Mid-column clutter is -51..-54 dB at every carrier (l = 1 m: -52 -> -37). With B1 the
design decision cannot be made on clutter-limited bed visibility vs frequency in this
sampling regime: the remaining levers are along-track sampling/focusing (the 15 m
posting aliases the 29 deg clutter at every carrier here; round 7 showed ~26 dB
recoverable when unaliased) and the ~+-5 dB along-line / +-3 dB year-to-year spectrum
uncertainty, which is now larger than the inter-frequency differences.

## 5. Sensitivity to theta_c (pilot_smoke_b1_th20 / _th40)

Westcoast (195 MHz; p3_2016 / p3_2017 / p3_2019 pairs at 20 deg: 0.86 / 0.90, 1.30 /
0.81, 1.26 / 0.88 cm / m; at 40 deg: 0.57 / 0.48, 1.10 / 0.43, 0.91 / 0.47):

| pass | measured | B1 theta 20 (err) | B1 theta 30 (err) | B1 theta 40 (err) |
|---|---|---|---|---|
| p3_2016 | -48.9 | -91.0 (-42.0) | -80.9 (-32.0) | -78.8 (-29.9) |
| p3_2017 | -59.4 | -77.7 (-18.2) | -69.7 (-10.2) | -68.0 (-8.6) |
| p3_2019 | -59.0 | -81.5 (-22.5) | -72.2 (-13.2) | -70.3 (-11.3) |

Bed level, bed-tail excess and surface alignment are identical to +-0.3 dB / 0.02 bins
across the three theta_c. The mid-column moves -8..-9 dB (20 deg) and +1.7..+2.1 dB
(40 deg) relative to 30 deg, far more than the < 1 dB the PSD residual at k_B(30 deg)
suggests, because at 500 m AGL the mid-column window is set by 60-70 deg surface
angles (Lambda ~0.8 m) where the longer 20 deg l (0.8-0.9 m) collapses the Gaussian
tail and the shorter 40 deg l (0.43-0.48 m) does not. For a high-altitude pass, whose
mid-column and bed-delay clutter come from 20-35 deg, the theta_c choice matters by the
< 1 dB the residual table gives. So: pick theta_c from the pass geometry (the angle at
the delay of interest) -- 30 deg for the 9-14 km cases, 40+ deg if the low-altitude
mid-column is the target.

## 6. Runtimes and commands

Wall per invocation (this box, warm data cache; `claude_notes/roughness_b1/logs/`):
fixture pilot_smoke westcoast replay 9 s (all chunks cached); pilot_smoke_b1 westcoast
14.4 min, geikie 20.5 min, getz 5.6 min; theta 20/40 variants the same; wc_hd_b1 (4 HAPS
passes + p3_2017) 11.6 min, wc_hd_b1fix 10.6 min. Per-pass B1 chunk time is 1.7-1.8x the
fixture. No `full` segment was run (the pilot results were unambiguous and a full
westcoast/geikie B1 run is ~1.5-2.5 h each at this ratio; the getz `full_line` would be
~2.5 h).

    uv run python claude_notes/roughness_b1/fit_b1.py                      # YAML + tables + residual plot
    uv run pytest -q tests/test_surface_roughness_b1.py tests/test_basal_hypotheses.py tests/test_experiment_specs.py
    bash claude_notes/roughness_b1/run_pilots.sh pilot_smoke greenland_westcoast          # fixture replay
    bash claude_notes/roughness_b1/run_pilots.sh pilot_smoke_b1 greenland_westcoast greenland_geikie01_transit antarctica_getz
    bash claude_notes/roughness_b1/run_pilots.sh pilot_smoke_b1_th20 <line>; ... _th40
    bash claude_notes/roughness_b1/run_pilots.sh wc_hd_b1 greenland_westcoast; ... wc_hd_b1fix
    uv run python claude_notes/roughness_b1/tabulate.py [cases]              # measured vs fixture vs B1 tables
    uv run python claude_notes/haps_design_study/gen.py b1 designs_b1.json --tabulate-only

Fixture pilot outputs were copied from the main tree's `outputs/<line>/pilot_smoke`
(2026-08-25, sha 8ab38b7, on this branch's history) so the fixture columns are replays,
not re-simulations.

Geikie (195 MHz; pairs 1.63 / 0.87 (20 deg), 1.35 / 0.60 (30), 1.19 / 0.47 cm / m (40)):

| pass | measured | B1 theta 20 (err) | B1 theta 30 (err) | B1 theta 40 (err) |
|---|---|---|---|---|
| low (465 m) | -44.5 | -78.9 (-34.4) | -69.6 (-25.1) | -67.6 (-23.1) |
| high (2483 m) | -41.7 | -60.5 (-18.8) | -60.3 (-18.6) | -61.5 (-19.7) |
| high - low | +2.8 | +18.4 (+15.6) | +9.3 (+6.5) | +6.2 (+3.4) |

The 2.5 km pass is insensitive to theta_c (+-1 dB: its mid-column angles are near
30 deg), the 465 m pass moves -9 / +2 dB as on westcoast, so the altitude-trend error
is 3-16 dB depending on theta_c, with the low-altitude grazing form error the cause.
Bed level and tail on the high pass: theta 20 -101.5 dB / -24.8 (+2 us), theta 30
-97.0 / -14.5, theta 40 -95.6 / -12.0 (measured -84.0): the bed-window clutter at
altitude is set by the ~30-40 deg band and the fixture's -25 dB tail deficit shrinks to
-12..-15 dB.
