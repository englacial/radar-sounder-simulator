# Altitude comparison with firn layers (2026-07-29)

Extension of `tools/run_altitude_comparison.py` with `--firn N` (effective-
contrast firn stacks, the b26-validated standard) plus the iteration arc that
followed. Deliverables re-run for both frames at their previously recorded
levels with `--firn 10`.

## What was promoted (item 1)

`src/soundersim/firn.py` — the effective-contrast construction, promoted from
`tools/run_b26_comparison.py`:

- `eps_kovacs`, `load_density_tab` (PANGAEA .tab parser), `tmm_reflection`
  (Yeh transfer matrix);
- `FirnCore` (raw + 0.1 m edge-normalized-boxcar-smoothed profile, `point_eps`
  trend, `equal_depths`, `segment_reflectivity`, `effective_contrast_eps`);
- `firn_stack` (media + conformal OffsetInterfaces, shared attenuation,
  optional per-layer roughness).

`run_b26_comparison.py` now delegates (`B26_CORE = FirnCore(...)`; its
`segment_reflectivity` / `effective_contrast_eps` / `firn_cfg` are thin
wrappers). BYTE-IDENTITY VERIFIED: eps/|r| arrays for N=5/10/20/40 are
`np.array_equal` to the pre-refactor golden values; the recorded cache-key
components (`eps` sum rounded to 6 dp, `depths_hash`) of all four cached
`firn_N*_h1eff` runs reproduce exactly; the b26 config-level tests
(`tests/integration/test_b26_comparison.py -k "not tiny"`, 4 tests) pass.
No b26 cache was touched.

New fixture: `tests/fixtures/firn/BER11C95_25_density.tab` (B25 Berkner
Island summit core, PANGAEA doi:10.1594/PANGAEA.227732, 1.139-178.213 m at
3 mm), copied from the read-only `~/Documents/clutter` repo with provenance in
the fixtures README. **B25 is a REPRESENTATIVE Antarctic firn proxy** — the
2012 frame is nowhere near Berkner Island; the tool prints/records this
caveat everywhere.

## Tool extension (item 2)

`--firn N` (default off; surface+bed path and caches byte-identical when
absent — the deliverable runs below all hit the existing `level_*` caches):

- region-keyed core (lat > 0 → B26, else B25), depths `equal_depths(N)` over
  [1 m, core zmax] (B26 119.66 m; B25 178.21 m), eps from
  `FirnCore.effective_contrast_eps` at the frame's own wavelength;
- firn media (and substrate) at 15 dB/km one-way (= the ice medium constant);
- conformal `OffsetInterface`s of the surface DEM; simulated per level on a
  narrow ±600 m strip in ~17-trace along-track chunks (b26 `firn_scenes`
  construction), platform height overridden per level; field-summed with the
  cached wide surface+bed run EXCLUDING the firn run's own surface layer;
- the firn strip has its own facet spacing (deepest layer binds, 32 m-divisor
  snap) — unlike b26 there is no exact-lattice seam GATE; a gamma-scaled
  surface-field agreement diagnostic is recorded instead (approximate when
  the two spacings differ);
- new per-level nadir depth-power panel (`firn_profiles.png`) vs measured,
  b26 `profile_vs_depth` pattern, plus band levels
  (5-20/20-60/60-120/20-70/80-120 m) and Pearson r (5-200 m) before/after.

## Compute (item 3): pilot projection vs actual

2-trace pilots per frame (scratch out dirs), then facet-count-exact
projections at the full 100-trace configs, rate ~5-10e-5 s/(trace·facet) at
N=10:

| frame | projected firn | actual firn | levels |
|---|---|---|---|
| 2019 (B26) | 12.5 min | 905 s = 15.1 min (361/361/183 s) | real/1000agl/5000msl |
| 2012 (B25) | 23.5 min | 1072 s = 17.9 min (187/660/166/59 s) | real/1000agl/10000msl/15000msl |

All surface+bed level runs were cache hits (recorded configs reused, as
required). Total simulation wall across baseline + pilots + all iterations +
final rough re-runs ≈ 1.6 h — no trace-count reduction was needed (100 traces
kept everywhere).

## Baseline results: --firn 10, mean-power metric (final convention)

Nadir depth-power, dB rel own surface peak; PRIMARY metric = trace-averaged
mean power (see iteration C); delta = sim − measured; corr = Pearson r
5-200 m.

### 2019_Greenland_P3 / 20190418_01_009 (B26; measured 20-70 m = -17.5 dB)

| level | corr before → after | 20-70 m delta before → after |
|---|---|---|
| real (462 m AGL) | 0.721 → 0.921 | -41.2 → -9.8 dB |
| 1000agl | 0.807 → 0.918 | -32.5 → -9.4 dB |
| 5000msl (2390 m AGL) | 0.922 → 0.926 | -24.4 → -9.0 dB |

The firn stack recovers the measured near-surface morphology at every
altitude; the residual ~-9 to -10 dB mid-band deficit at N=10 matches the b26
ladder (N=10 old-metric delta -10.4 there) — largely the known N=10
picket-fence shape + the ~5 dB coherent-realization deficit.

### 2012_Antarctica_DC8 / 20121023_04_006 (B25 proxy; measured 20-70 m = -6.7 dB)

| level | corr before → after | 20-70 m delta before → after |
|---|---|---|
| real (8953 m AGL) | 0.860 → 0.868 | -13.4 → -13.1 dB |
| 1000agl | 0.880 → 0.884 | -13.7 → -13.6 dB |
| 10000msl | 0.865 → 0.861 | -12.9 → -12.7 dB |
| 15000msl | 0.859 → 0.856 | -12.6 → -12.7 dB |

**The firn stack changes almost nothing on this frame (<1 dB, corr ±0.01).**
Physics: 9.5 MHz bandwidth (15.8 m range res), near-rect window and ~1 km
pulse-limited surface footprint at 9 km AGL — the measured near-surface skirt
is dominated by the surface response, not by firn strata; the B25 segment
reflectivities (~-35 dB/interface) are far below that skirt. The ~-13 dB
residual was then attacked by iteration D (surface roughness), which is where
the real 2012 gap closed.

## Iteration log (item 4)

KEPT (each validated by numbers, then wired in):

1. **Mean-power profile as the primary metric** (iteration C; analysis-only).
   Mean power over all traces, each aligned/normalized on its own surface
   peak (the b26 findings' fair-metric convention) instead of one
   representative trace. 2019 with-firn corr 0.894→0.921 (real),
   0.888→0.918 (1000agl), 0.892→0.926 (5000msl); level-to-level band deltas
   stabilized (single-trace values swung by up to 6 dB between levels from
   speckle alone — e.g. 2012 15000msl 20-70 m delta -18.6 single-trace vs
   -12.6 mean). Single-trace numbers remain recorded under
   `firn.per_level.*.repr_trace`.
2. **`--surf-rough`: representative surface roughness** (iteration D).
   Gerekos rough-facet response on the wide run's surface interface at the
   C&S 2020 Fig. 11 mcords inversion's 0 m clamp (sigma 4.95 cm, l 2.98 m;
   REPRESENTATIVE, not site-measured; default OFF so the smooth path/caches
   stay byte-identical; rough runs cache under `level_<spec>_srough`).
   Mean-power with-firn effect at the real levels: 2012 corr 0.867→0.917,
   20-70 m delta -13.1→-9.6 dB (surface+bed-only: 0.860→0.926, -13.4→-8.8);
   2019 corr 0.921→0.930, -9.8→-7.7 dB. Improves both frames → kept and used
   for the final deliverable runs.

3. **Firn N=20 for the 2019 frame** (iteration A; 2019 real level, scratch,
   1941 s firn wall — contention-inflated). Mean-power with-firn: corr
   0.921→0.968, deltas 5-20 m -11.6→-9.7, 20-70 m -9.8→-9.0, 60-120 m
   -12.5→-3.8, 80-120 m -10.7→-6.4 dB. The b26 ladder's N=20 correlation
   plateau (0.963) reproduces on this tool; the deep firn band nearly
   closes. KEPT for the 2019 deliverable (final run at --firn 20). NOT
   extended to 2012: the firn stack moves that frame by <1 dB at any N (its
   near-surface skirt is surface-dominated), so N=20 there would be ~70 min
   of compute for nothing — 2012 deliverable stays at N=10.
4. **Internal-layer roughness — REVERTED** (iteration B; C&S Fig. 11 mcords
   roughness depth-interpolated onto every internal firn interface, 2019
   real, N=10, scratch): corr 0.9213→0.9335 (+0.012) but 20-70 m delta
   -9.77→-9.85 dB (nil) and the other bands 0.4-0.6 dB WORSE, at ~2x the
   firn wall (699 vs 361 s). Consistent with the b26 rejection (+0.7 dB
   there); not worth the complexity/cost — no code path added to the tool.

## Final deliverable runs

`outputs/altitude_comparison/` (mirrored to `outputs/verification/altitude_*`,
both mirrors diff-verified; main report rebuilt via tools/make_report.py):

- **2019**: `--firn 20 --surf-rough` (both kept improvements). Mean-power
  with-firn vs measured (before = ROUGH surface+bed of the same run):

  | level | corr before → after | 20-70 m delta | 60-120 m | 80-120 m |
  |---|---|---|---|---|
  | real | 0.767 → **0.970** | -34.8 → **-7.0** | -33.4 → **-1.7** | -29.7 → **-4.6** |
  | 1000agl | 0.831 → **0.971** | -28.2 → **-7.4** | -28.8 → **-2.1** | -26.3 → **-4.7** |
  | 5000msl | 0.899 → **0.978** | -18.8 → **-6.8** | -19.2 → **-1.8** | -17.6 → **-4.4** |

  The remaining ~-7 dB mid-band deficit is the b26 arc's known
  coherent-realization deficit (~-5 dB there) plus this frame's residuals; the
  deep firn band is now within ~2 dB. Seam diagnostic ~0.25-0.31 with
  --surf-rough (expected: the wide surface field is coherently attenuated +
  diffuse while the firn strip's surface stays smooth; it was ~1e-3-0.15 in
  the smooth baseline).
- **2012**: `--firn 10`, smooth surface (numbers in the baseline table above).
  --surf-rough was validated at this frame's real level (iteration D) but the
  full-frame rough re-run was not performed (compute stop by the coordinator
  after the iteration phase); the tool flag is ready and the iteration numbers
  say it is worth ~+4 dB / corr +0.06 there.

The 2019 N=10 baseline (task item 3's literal deliverable) was fully run and
is tabulated above; its `level_*_firn10` caches remain in runs/ alongside the
N=20 ones, so `--firn 10` re-assembles without simulation.

## Timings (simulation wall, from the runs' recorded diagnostics)

- Pilots (2-trace, both frames, all levels incl. wide + firn): ~11 min.
- 2019 firn N=10: 361 + 361 + 183 s = 15.1 min (surface+bed: cache hits).
- 2012 firn N=10: 187 + 660 + 166 + 59 s = 17.9 min (surface+bed: cache hits).
- Iteration A (2019 real firn N=20): 1941 s (CPU-contended).
- Iteration B (2019 real N=10 + layer roughness): 699 s. REVERTED.
- Iteration D (surface-rough wides, 2012 real + 2019 real): 108 + 234 s.
- Final 2019: srough wides 178 + 86 + 82 s, firn N=20 1000agl/5000msl
  1448 + 710 s (real N=20 reused from iteration A).
- Total ≈ 1.9 h, under the ~4 h ceiling; **no trace-count reduction was
  needed** (100 traces everywhere, full recorded levels).

## Tests

- Unit suite: **231 passed** (223 pre-existing + 7 tests/test_firn.py +
  1 firn-config test in tests/integration/test_altitude_comparison.py).
  One timing-sensitive test (test_compile_flat_in_n_and_runtime) failed once
  under CPU contention with a background simulation and passes in isolation
  and in the final clean run.
- Integration (config-level, no sim): b26 rough/effective-contrast/
  attenuation/only-flag tests + altitude firn-config test all pass; the
  cached b26 h1eff meta keys (eps sums, depths hashes) reproduce exactly
  post-refactor; eps/|r| byte-identity vs pre-refactor golden values
  verified for N=5/10/20/40.
- ruff clean on all changed files.

## Honest caveats

- **B25 is a proxy.** The 2012 frame does not pass a cored site; B25
  (Berkner Island summit) supplies a plausible Antarctic firn density
  profile, nothing more. Recorded in the fixtures README, the tool's console
  warning, run_config, metrics note and the report.
- **N=10 shape limitation.** Band-integrated firn power is ~N-independent by
  construction, but N=10 (12-18 m layer spacing vs ~3-6 m in-firn range
  cells) leaves a picket-fence profile (visible in firn_profiles.png for
  2019); the b26 ladder puts the correlation plateau at N=20.
- **Surface roughness values are representative.** The Fig. 11 numbers are a
  Greenland INTERNAL-layer inversion clamped to 0 m depth, borrowed as a
  stand-in for unmeasured cm-scale surface roughness; treat the rough-surface
  runs as "plausible roughness" not "measured roughness".
- **Seam diagnostic is approximate.** The firn strip and the wide run use
  different facet spacings, so the gamma-scaled surface-field agreement is
  loose where lattices differ (recorded per level; tight, ~1e-3, when the
  spacings coincide, e.g. 2019 real).
- Measured products are SAR-focused + multilooked; sims are unfocused
  per-trace (the b26 qlook test bounded this asymmetry at ~0.3 dB for
  surface-referenced depth profiles).
- 2012 sims model the tukey(0.2) compression window as rect (recorded
  window approximation), and 'MSL' levels are ellipsoidal heights.
