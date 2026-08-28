# Config simplification plan (2026-08-27)

Goal: two experiments (`full`, `pilot`) identical across lines; instruments named
`<instrument>_<platform>_<year>`; lines audited; prose trimmed everywhere.
Decisions from the user are marked **[D]**.

## 1. Structural changes (code)

1. **Bed source split into data (line) and method (experiment)** [D].
   - Line: `bed_dem: demogorgn | bedmachine` (AIS lines demogorgn, GrIS bedmachine);
     replaces `unsupported: [demogorgn_bed, hybrid]`.
   - Experiment: `bed: {nadir: picked, floating: picked, demogorgn_seed: 0}`.
     `nadir: picked` = DEM + along-track residual to the reference-pass picks on
     the grounded side; `floating: picked` = reference picks as shelf base on
     `crosses_gl` segments (ignored otherwise).
   - Wire DEMOGORGN + pick residual (currently refused at
     `run_basal_clutter.py:1729,3368`): apply `apply_picked_bed` after
     `apply_demogorgn_bed` / grounded side of `apply_hybrid_bed`.
   - Cache tags already separate pbed/dgn/hyb; add a tag for dgn+pbed so old
     chunks are never reused. All Antarctic runs re-simulate.
2. **Experiment schema trim** (`tools/clutter_spec.py`): `meta` → `{name, description}`
   only (drop status/role/backs/runtime/note/expected/requires); `out_name` derived
   from `meta.name`; `bed` per (1). Filename == `meta.name`.
3. **Line schema**: add `bed_dem`; drop `synthetic_passes`; drop
   `identity.season` (per-pass `season` required instead).
4. Tests: `test_experiment_specs.py` (hardcoded spec names, `RECORDED` pin on
   `outputs/antarctica_getz/full_line`), `test_instruments.py` (5 instrument names),
   `test_analysis_config.py`, `test_surface_roughness_b1.py`, `test_roughness_exponential.py`
   (reference `pilot_smoke*` files).
5. `config/README.md` rewritten to match; `docs/clutter_studies.md`, `docs/bed_scattering.md`
   name fixes; stale docstring `run_basal_clutter.py:48`.

## 2. Experiments → `full.yaml`, `pilot.yaml`

Both: `lines: [all four]`, HAPS 14/20 km extras [D], fixture surface roughness [D],
adopted reflectivity/bed_roughness/processing block, `att_db_per_km: solve`.
Differ only in `segment`. Delete the other 24 files (git history).
Getz-only `figures` block → getz line (`plot_s_max_km`, `width_scale`) or dropped.
Existing output dirs (`pilot_smoke`, `full_line`, `std_benchmark`, `david_full`) left as history.

## 3. Instruments

| old | new | fix |
|---|---|---|
| mcords3_p3_greenland | mcords3_p3_2014 | tx_power 1011 → 1050 W |
| mcords_p3_2019 | mcords3_p3_2019 | note L/R 18° steered boxcar beams (unmodeled) |
| mcords_p3_2016_200mhz | mcords5_p3_2016 | **antenna: 2 elements, 0.61 m aperture** (was 7 × 0.5λ) |
| mcords3_dc8_2016 | (same) | **antenna: 3 cross-track × 2 along-track, 6 ch** (was 7 × 0.5λ); spacing unknown |
| basler195_2017 | mcords5_basler_2017 | ok |
| mkb60_basler | marfa_baslermkb_2022 [D] | not in OPR readme; 2023 same instrument |
| haps_60mhz | keep | |
| haps_60mhz_{5us,chirp,5us_chirp}, hd_* (24) | delete | regenerable from claude_notes/haps_design_study |

Prose: one-line description + antenna-model source line; drop provenance essays.

## 4. Lines

- Pass names `<platform>_<year>[_<agl>]` [D]: getz `dc8_2016_0km/9km/11km`, geikie
  `p3_2014_low/p3_2017_high`, westcoast `p3_2016/p3_2017/p3_2019`, david
  `basler_2017/baslermkb_2022/baslermkb_2023`.
- Segments: exactly `pilot` + `full` per line. Geikie `full` = 139 km transit [D];
  getz `full` = old `full_line`; drop `extended`, old getz/geikie `full`.
- Every pass carries `season`; delete `synthetic_passes`; `facet_spacing_scale`
  dropped where = analysis default 0.7.
- Calibration `why` → one line; `provenance` → 2-3 lines; header essays removed.
- Recorded asymmetries kept in README: david reference is the 60 MHz pass;
  radargram dB floors differ per line.

## Open
- DC-8 2016 element spacing: resolved from OPR lever_arm.m (0.45 lam), 2026-08-27.

## Execution log (2026-08-27)
Executed as planned. Deviations: `synthetic_passes` stays in the line schema
(tests use it) but no shipped line declares one; `identity.season` dropped,
per-pass `season` required; getz-only figure knobs (plot_s_max_km 100,
width_scale 2, per_pass) dropped rather than moved. DC-8 2016 element spacing
assumed 0.5 lam (readme silent). Pre-existing failure noticed, not mine:
tests/test_basal_processing.py::test_chunk_digests_forwards_the_hypothesis_knobs
(KeyError rc_sim; HEAD already fails).
