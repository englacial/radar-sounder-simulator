# Cross-season repeat-line comparison (2026-07-30)

Tool: `tools/run_cross_season.py`; line + params from
`claude_notes/cross_season_line_scout.md`; deliverable `outputs/cross_season/`
(report.html, radargrams/profiles/diff_matrix figures, metrics.json with the
pairwise capture numbers, mirrored to outputs/verification/cross_season/).

## Setup

Common 30.65 km window of the 2012_Antarctica_DC8 / 20121023_04_008
high-altitude anchor (9217 m AGL, 9.5 MHz, tukey(0.2)->rect model, 105.21 ns
critically-sampled grid) and its ~450 m-AGL 50 MHz repeats 20141029_05_013 /
20161104_05_008 / 20181107_01_011 (hann, ~20 ns grids; 2016 on 20.202 ns).
Each frame simulated at its REAL altitude/nav (incl. roll — the 2012 frame
rolls to 5 deg) and its own chirp/window/fast-time grid, 100 sim traces over
the window, REMA 32 m + BedMachine, coherent surface+bed + B25-proxy firn
N=10 (effective contrasts, 15 dB/km in the strip) + representative surface
roughness (`--surf-rough` values validated in the altitude arc), alias-free
dt/k per frame (k=6/5/5/5), per-frame constant-offset surface registration
(scout pitfall: seasons disagree by up to 23 m), metre-domain profile
smoothing (10 m), bed cross-track per frame (2012 needs the 6 km cap; the
repeats ~2.7-2.8 km).

Measured reference: CSARP_standard. Scout pitfalls honored and recorded:
2014/2016 ft_wind provenance is a decode-fallback string (scout verified the
real value IS hanning); 2014+ products are 1/3/10 us img_comb composites
(sims carry the 10 us bed waveform only); qlook unused (per-season decimation
incomparable).

## Metric design (calibration reality)

CReSIS products are not radiometrically cross-calibrated between seasons, so
measured ABSOLUTE level differences between flights mix geometry with unknown
per-season gains. Currencies used:

- **bed-minus-surface level** per frame (within-record ratio, gain-free);
- **surface-peak-normalized mean-power depth profiles** and their
  inter-flight DIFFERENCE curves (gain-free);
- raw surface-level pair deltas recorded as UNCALIBRATED alongside the r^-2
  expectation.

## The two discoveries that shaped the analysis

1. **The 2012 frame does not actually detect the bed here.** Its measured
   "bed peak" equals its own mid-column level: bed SNR ~0 dB; the whole ice
   column below ~the firn sits at the frame's noise floor, -42 dB rel
   surface. The repeats genuinely detect the bed at -70 dB rel surface with
   21-26 dB SNR. So the dominant REAL inter-flight difference on this line is
   "bed visible at 450 m / 50 MHz, invisible at 9.2 km / 9.5 MHz" — and a
   noise-free simulator cannot express that. The tool therefore estimates
   each frame's own noise floor (rel surface, gain-free, from the mid-column
   quiet window 2-3 us above the bed pick) and produces NOISE-AWARE metric
   rows where the sim profile/bed level is floored at that frame's measured
   floor. (An early hypothesis — that the 28 dB bedsurf gap was the img_comb
   composite's section gains — was tested and mostly rejected: the
   slope-corrected gain step at the first waveform boundary is only ~-5.7 dB
   in the repeats, ~0 in single-image 2012.)
2. **15 dB/km underestimates this line's column loss by ~16 dB/km.** With
   the b26/altitude 15 dB/km, the sim bed sat ~28 dB hot relative to surface
   on all three repeats (which measure it cleanly). A single effective-loss
   knob fitted on the repeats — **31 dB/km one-way** — closes all three
   frames to 0.5-2.3 dB AND pushes the 2012 sim bed below that frame's own
   noise floor exactly as observed. It is an EFFECTIVE value (absorbs true
   warm-ice attenuation, bed-roughness scattering loss, any unmodeled column
   loss), promoted into the tool as `--att` (default 31, provenance
   recorded; the firn strip keeps the b26-validated 15 — <= ~6 dB
   inconsistency over its 178 m, recorded).

## Per-frame sim-vs-measured (final config: firn10 + surf-rough + att31)

| frame | AGL / B | surface gate (bins) | profile corr | bedsurf sim / meas (dB) | floor (dB rel surf) | bed SNR |
|---|---|---|---|---|---|---|
| 2012 | 9217 m / 9.5 MHz | 0.95 PASS | **0.994** | -42.9 (floored -41.9) / -42.1 | -41.9 | **-0.2 dB — NOISE-LIMITED** |
| 2014 | 465 m / 50 MHz | 0.46 PASS | 0.933 | -68.1 / -70.0 | -95.4 | +25.4 |
| 2016 | 446 m / 50 MHz | 0.50 PASS | 0.947 | -68.3 / -70.6 | -92.8 | +22.2 |
| 2018 | 447 m / 50 MHz | 0.78 PASS | 0.949 | -68.8 / -69.3 | -95.7 | +26.4 |

## Cross-flight difference matrix (the key deliverable)

delta = value(i) − value(j); bedsurf rows gain-free; "simfl" = sim with each
frame's own measured noise floor applied; capture = 1 − |sim−meas|/|meas|.

| pair | bedsurf delta meas / simfl (dB) | capture | profile-diff r (20-500 m) | rms (dB) | surf delta meas / sim / r^-2 (UNCAL) |
|---|---|---|---|---|---|
| **2012-2014** | +27.9 / +26.2 | **0.94** | 0.77 | 11.2 | -16.8 / -27.1 / -25.9 |
| **2012-2016** | +28.5 / +26.4 | **0.93** | 0.83 | 11.0 | -20.3 / -27.3 / -26.3 |
| **2012-2018** | +27.2 / +26.8 | **0.99** | 0.83 | 9.6 | -22.3 / -27.5 / -26.3 |
| 2014-2016 | +0.6 / +0.2 | 0.33 | -0.04 | 1.3 | -3.6 / -0.2 / -0.4 |
| 2014-2018 | -0.8 / +0.6 | -0.80 | -0.15 | 2.5 | -5.6 / -0.4 / -0.3 |
| 2016-2018 | -1.4 / +0.4 | -0.30 | -0.20 | 2.0 | -2.0 / -0.2 / +0.0 |

Reading:

- **Headline (2012 vs repeats): the sims capture 93-99 % of the ~28 dB
  bed-visibility difference** once each frame's own noise floor is applied —
  the dominant real difference on this line (bed seen at 450 m / 50 MHz,
  noise-buried at 9.2 km / 9.5 MHz) is reproduced, with the calibrated
  column loss shared by all four frames.
- Headline profile-difference curves: shape r 0.77-0.83; the sims
  under-predict the measured difference by ~10 dB rms at 150-500 m because
  the sim column between the firn stack's end and the bed is EMPTY (no
  englacial reflectors) while the repeats' measured columns have structure
  there and the 2012 measured column is at its floor.
- Repeat-vs-repeat pairs: real differences are tiny (<= ~1.4 dB bedsurf,
  <= ~2 dB profiles); the sims correctly predict near-zero differences
  (rms 1.3-2.5 dB), and the correlation/capture numbers there are
  noise-on-noise — reported for completeness, not evidence either way.
- Surface deltas: sims sit on the r^-2 expectation by construction of the
  physics (-27 vs -26); measured deltas are 4-9 dB smaller — consistent
  with uncalibrated per-season gains (recorded only).

## Iteration log

KEPT:

1. **Noise-aware analysis** (assembly-only): per-frame measured noise floor
   (gain-free rel surface) applied to the sim; bed-SNR qualifier on every
   bedsurf row (2012 flagged NOISE-LIMITED: its measured bedsurf is an upper
   bound). Without it the headline bedsurf capture is meaningless (the sim
   was being compared against a noise floor as if it were a bed echo).
2. **Effective attenuation 31 dB/km** (4 bed re-sims, 1281 s): per-frame
   repeat corr 0.79 -> 0.93/0.95/0.95, repeat bedsurf error +28 -> +0.5-2.3
   dB, 2012 floored bedsurf -41.9 vs measured -42.1; headline pair bedsurf
   capture 0.12/0.14/0.13 -> **0.94/0.93/0.99**.
3. **Firn N=10 stays on** (tested off, assembly-only): removing it drops
   repeat frame corr 0.79 -> 0.76 and the headline profile-diff r
   0.77/0.83/0.83 -> 0.40/0.54/0.54 (the near-surface response difference
   between 9.5 and 50 MHz lives in the firn zone). Cost to keep: it was
   already simulated.

REVERTED / RECORDED ONLY:

4. **img_comb section-gain correction — investigated, NOT adopted**: the
   hypothesis that the repeats' -70 dB bedsurf was a composite-product gain
   artifact was tested with a slope-corrected boundary-step estimate; the
   step at the first waveform boundary is only ~-5.7 dB (2014 -5.83 / 2016
   -5.91 / 2018 -5.48; 2012 single-image -0.37) — real but secondary, so no
   correction was applied (recorded as a caveat instead).
5. **Surface-roughness-off and 2012-roll-off variants — not run**: planned
   as documentation iterations (~22 min sims) but dropped at the
   coordinator's compute stop after the attenuation iteration landed;
   surf-rough stays on the strength of its two-frame altitude-arc
   validation. Untested here — recorded as such.

## Honest caveats

- The B25 firn core is a REPRESENTATIVE Antarctic proxy (Berkner Island);
  this line is upstream Thwaites/Pine Island — no cored site.
- 31 dB/km is an effective, line-calibrated loss, not a physical attenuation
  measurement; it deliberately absorbs bed roughness. Do not reuse it
  elsewhere without re-calibration.
- The sims carry no receiver noise; the noise-aware rows use each frame's own
  MEASURED floor, so they cannot predict the floor of a hypothetical flight —
  they test whether the modeled column+bed levels are consistent with what
  each real system could see.
- The sim column between the firn stack's 178 m bottom and the bed is empty
  (no englacial reflectors); the measured profiles keep structure there.
  This is most of the residual profile-difference rms (~10 dB) on the
  headline pairs.
- Measured cross-season surface deltas are uncalibrated (recorded vs r^-2
  only); the 2014+ products are img_comb composites (surface section from
  the 1 us waveform, ~-6 dB slope-corrected step at the first boundary).
- 2012 is critically sampled (105.21 ns = one range cell): its measured
  surface peak/profile carry quantization scatter (scout pitfall).

## Timings (simulation wall)

- 2-trace pilot (all four frames, incl. one FSTimeoutError retry and a
  window pre-cache pass): ~6 min of simulation; the network pre-cache
  (frames + REMA + BedMachine windows with retry/backoff) is not sim wall.
- Baseline (100 traces, firn10 + srough, att 15): 3040 s = 50.7 min
  (2012: 61+89 s; each repeat: ~320 s bed + ~645 s firn).
- Attenuation iteration (4 beds at att31 under scratch meta): 1281 s.
- Final run (4 beds re-simulated at att31 under tool meta; firn cached):
  1281 s. Recorded final total in metrics: 3043 s for the assembled runs.
- Cross-season total ~1.9 h; combined with nothing else re-run, well inside
  the 4-5 h ceiling. No window or trace-count reduction was needed
  (100 traces, full 30.65 km window).

## Tests

- tests/integration/test_cross_season.py (config-level, no sim/network):
  scout-table match, per-era window mapping incl. the ft_wind-fallback
  string, pair-matrix math on synthetic profiles (exact-sim capture = 1,
  degraded capture, r^-2 row, NOISE-LIMITED flag), peak_db. 4 tests.
- Full unit suite green; ruff clean on the new/changed files.
