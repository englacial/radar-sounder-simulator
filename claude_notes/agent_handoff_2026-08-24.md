# Agent handoff — clutter studies state (2026-08-24)

Read this first. Supersedes `agent_handoff_2026-08-17.md` (kept for the
pre-calibration-era chronology). Everything below is committed on `main`
(no remote; the user creates any GitHub repo themselves). 441 unit tests
green (`uv run pytest tests -q --ignore=tests/integration`).

## Architecture (config-driven since 2026-08-18)

Four study lines, each fully declared in YAML:
- `config/lines/` — geometry, passes, frame windows (omitted slice = whole
  frame), segments, calibration block, figure framing.
- `config/instruments/` — the radars; real ones read every simulated
  parameter from their own OPR frames; antennas modeled per instrument and
  fingerprinted into chunk cache keys.
- `config/experiments/` — one full experiment per line (`ant_full_line`,
  `ant_david_full`, `gl_std_benchmark` x2) + the shared `pilot_smoke`; the
  cross-line HAPS design points (same 60/15 MHz stated instrument at 14 and
  20 km on every line, `carrier: reference`).
- `config/analysis.yaml` — measurement conventions + study physics
  constants (facet scale <= 0.7; grazing_fix s_eff; solver settings).

Lines: antarctica_getz (3 alt. passes, 148 km, GL at 69.7), antarctica_david
(frequency diversity: 195 MHz Basler + 60 MHz MKB x2, GL at 95.4;
mkb60_2022 is an off-corridor stress case — ~900 m offset, terrain-hugging),
greenland_geikie01_transit (englacial-dominated column), greenland_westcoast
(3-way P-3 repeat; mid-May wetness question).

## Physics era (adopted 2026-08-21/24)

1. **Bed scattering** (docs/bed_scattering.md): tilt-gated specular/diffuse
   split (f_s 0.5, s0 3 deg) + Gerekos sub-facet bed roughness sigma 0.10 m
   at l = lambda_ice — continuous bed horizons, matched tails.
2. **Grazing-angle facet-lattice fix** (default ON — a bug fix): coherent
   off-specular taper (s_eff 0.05) + area-only D_Phi. Removes a facet-size
   -dependent grazing clutter floor that was ~100% of david's spurious
   bed-window surface clutter and part of getz's mid-column "match".
   `--no-grazing-fix` = legacy artifact path, debug only. Root cause +
   validation: `david_clutter_resolution_2026-08-24.md` (the S1-S7
   campaign), `grazing_fix_validation.md`.
3. **Antennas**: array_tapered (2017 Basler, from the product's own param
   structs), finite_dipole (MKB wing plate), nav roll enabled; isotropic =
   declared upper bound only.

## Kernel era "2026-08-24-cull" (runtime work, merged 2026-08-24)

Per-trace along-track facet windowing in both kernels (exact: skipped facets
are provably outside the window; `dropped_power` no longer counts them),
component-form + `fori_loop` Newton bed path, adaptive facet block size,
and `KERNEL_VERSION` in every chunk cache key (all caches re-simulated on
2026-08-24/25). Greenland `std_benchmark` companion re-sim is OFF. Full
campaign now 5.5 h (getz 82 / david 127 / geikie 35 / westcoast 84 min) vs
30.6 h; all results in the standard experiment dirs are from this run.
The box is DRAM-bandwidth-bound — local multi-process/pmap fan-out gains
nothing; next lever if needed is the Newton budget (proposal 2b). Evidence:
`runtime_reduction_proposals_2026-08-24.md`, harness `runtime_opt/`.

## Calibration (per line, `calibration:` block)

gamma_surface: pinned -10 dB manual everywhere (solved values were
unphysical — they absorb chain anomalies; power-sum-inversion solve exists,
runs as a per-run diagnostic, history in `gamma_solve_design_2026-08-20.md`).
A (dB/km one-way): getz 18.61 solve, david 12.8 solve, westcoast 34.26
solve (Theil-Sen RSSNR vs 2H), geikie 16 manual (MacGregor; regression
rejected there — thawed-bed confounder). Received bed level is A-invariant
and moves dB-for-dB with gamma (the identity behind both solvers).

## State of results

- ALL chunk caches predate the grazing fix -> every next run re-simulates.
- Side-by-side on disk: `outputs/*/pilot_fixed` (fixed physics, getz+david)
  vs `outputs/*/pilot_smoke` (pre-fix); `outputs/_before_facet07_20260822/`
  (pre-adoption archive).
- Post-fix getz: calibration robust (gamma_req +1.16, three passes tight);
  9-10 km midcol now 4-8 dB under-predicted, low-pass midcol unexplained.
- Post-fix david: bed window bed-dominated on every pass; all 3 passes
  qualify for the gamma solve; sim bed 7-22 dB dim, Basler-vs-MKB
  disagreement (element pattern / absolute cal uncertainty).

## Open items (priority order)

1. Re-run the benchmark protocol everywhere at fixed physics; re-derive the
   getz altitude-clutter and design-ladder numbers.
2. The universal mid-column under-prediction (englacial term / surface
   roughness spectrum beyond the C&S sigma=0.05, l=3 m statistics).
3. David: element-pattern uncertainty + cross-instrument absolute
   calibration (gamma_req spread 15.6 dB); mkb60_2022 measured bed-window
   energy source.
4. Parked branches (user decides): `h2-bed-crosstrack` (pick-relief
   cross-track decorrelation — good pre-adoption evidence, untested with
   new physics), `t2-bed-spectrum` (bed spectral fill — right at altitude,
   amplitude needs tuning).
5. Bed sigma=0.10 re-check at fixed physics (the sweep partly traded
   against the artifact).
6. Older: geikie englacial term; westcoast winter-crossover test; DEMOGORGN
   licensing before publication; RSSNR store re-point when reprocessed.

## Key notes index

- `david_clutter_resolution_2026-08-24.md` — the S1-S7 campaign (START HERE
  for the current physics rationale).
- `gamma_solve_design_2026-08-20.md` — calibration era + solve design.
- `rebaseline_2026-08-23.md` — adopted-physics re-baseline, all 8 runs.
- `facet_convergence_refit_2026-08-21.md` — facet convergence + gamma refit.
- `grazing_fix_validation.md`, `antenna_realism_pilot.md` — fix validation.
- `bed_comparison_*.py`, `eval_below_bed.py`, `before_after_fig.py` —
  reusable analysis scripts from the campaign.
- Pre-calibration chronology: `agent_handoff_2026-08-17.md` and the notes
  it indexes (basal_clutter_pilot_findings, greenland_pair_findings,
  macgregor_attenuation_scout, b26_*).
