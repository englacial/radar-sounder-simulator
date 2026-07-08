# Firn power plateau investigation (exploratory redo of M19) — 2026-07-08

Deliverable: honest observation, not a gate. Everything in
`outputs/firn_investigation/` (report.html is self-contained); sweep tool
`tools/run_firn_investigation.py` (resumable), integrity test
`tests/test_firn_investigation.py` (integration-marked). The old case
`tests/test_firn_plateau.py` got ONE surgical repair: digitized Fig. 9 curves
are now depth-sorted before plotting (fig09a had 59 depth reversals →
self-intersecting path); its gates/logic are untouched and its framing is
under review separately.

## Design (user-specified)

Flat 600 m scene, 500 m AGL, 195 MHz, 4 m facets, 3 traces, coherent mode.
N ∈ {10, 20, 40, 80} offset flat layers over 1–119.66 m; equal spacing plus 3
seeded random placements per N (sorted uniform, min sep 0.25 m,
`default_rng((seed, n))`). eps convention: POINT sampling — medium below
interface at depth d takes the closest 0.1 m-smoothed B26 sample's Kovacs
eps(rho); substrate = eps(d_N + 1 m). This preserves the local contrast the
flawed 5 m slab-MEAN decimation destroyed. Observable: layer-summed coherent
field, |.|^2, trace mean, twtt→depth via per-layer in-firn nadir times, 5 m
boxcar, dB rel. surface peak. Reference: surface-only run.

## Findings (details + figures in outputs/firn_investigation/report.html)

1. **Plateau morphology partially present; the paper's operational criterion
   (near-zero/nonnegative gradient over >10 m) is met by NO run.** Longest
   contiguous interval at −0.05 dB/m: 2.4–9.6 m (best: random N=40 s1,
   26.9–36.5 m). For N≥40 the interval consistently sits in the 27–42 m
   rising limb. Threshold sensitivity (0 / −0.1 dB/m) < 2.5 m.
2. Common core-driven shape in all 13 runs: shoulder over ~10–50 m (mean
   ≈ −20 dB, span 12–16 dB) with a dip at ~25 m and local max at ~40–45 m;
   decay onset ~50–60 m (paper: sharp decay below ~60 m); −40..−50 dB at
   100–119 m; 8–11 dB mean excess over the surface-only reference (40–100 m).
3. Secondary max: 5.1–5.5 m depth, −6.0..−13.7 dB rel. surface in every run
   (paper Fig. 8: typically 10–15 dB below surface — same order).
4. Resolved→unresolved transition visible: N=10/20 (spacing 12/6 m > 3.7 m
   in-firn resolution) = trains of isolated boxy echoes; N=40/80 merge into
   continuous profiles and the deep floor rises (60–100 m mean −35 → −31.6 dB,
   100–119 m −49 → −40 dB at N=80). Trend points toward the plateau emerging
   in the many-layers-per-resolution-cell limit (paper: mm-scale layers).
5. Equal vs random: band means within ~2 dB; isolated bright echoes are
   placement-specific (~3 dB seed scatter 60–100 m); the 25 m dip / 40 m bump
   are not.
6. Realized per-interface |gamma| median: −40.5/−46.2/−50.5/−52.9 dB (equal
   N=10/20/40/80), p90 −30.7..−41.7 dB, vs full-res 1 mm adjacent-sample
   median −90.7 / p90 −79.9 dB — even N=80 is ~30–40 dB above the mm-adjacent
   statistics: nowhere near a converged discretization of the continuous
   profile.
7. **Bug found & fixed in this tool's loader** (the old test still has the
   benign-for-it version): `np.convolve(..., mode='same')` boxcar zero-pads
   the core bottom, halving the deepest ~5 cm density (882→446 kg/m³, eps
   3.05→1.90). Point-sampling the deepest layer there planted a spurious
   −18 dB bottom reflector (fake "secondary max" −3.7 dB @ 108 m in the first
   sweep pass). Fixed by edge-normalized smoothing; whole sweep rerun.

## Runtime (compile-inclusive first call per N, then in-process cached)

N=10: 14.0 s / 0.6 s; N=20: 39.0 s / 2.1 s; N=40: 235.0 s / 7.9 s;
N=80: 1568.6 s (26.1 min) — breached the 25 min first-per-N cutoff, so
random_N80_s0/s1/s2 were skipped per protocol (recorded in skipped.json).
13/16 layered runs + reference completed; total sim wall ≈ 32 min.
