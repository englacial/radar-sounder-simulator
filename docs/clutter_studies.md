# Clutter studies: multi-altitude line comparisons

`tools/run_basal_clutter.py` compares coherent surface+bed simulations
against measured OPR data for repeat passes of the same line at different
platform altitudes, with matched CSARP-style processing. It grew out of the
Antarctic basal-clutter study and now supports multiple study lines via a
line registry (`--line`).

## What it models

- Surface (ArcticDEM/REMA) and bed interfaces, coherent facet method, with
  per-pass geometry derived from the real navigation (or synthetic
  constant-altitude passes for design studies).
- Bed topography sources: BedMachine, radar-picked bed applied as an
  along-track residual correction (`--picked-bed`), DEMOGORGN geostatistical
  realizations (`--demogorgn-bed`, Antarctica only), and a grounded/floating
  hybrid with a grounding-line blend.
- Bed reflectivity: constant Fresnel, or per-facet values mapped from the
  required-surface-SNR (RSSNR) dataset (`--gamma-from-rssnr`):
  `|Γ_bed|²(x) = 2·A·H(x) − RSSNR(x) + (γ_surface − T²)`, where γ_surface
  is the line's effective surface power reflectivity (the RSSNR dataset is
  surface-referenced) and T² is the two-way Fresnel transmission
  (~−0.71 dB, computed, never configured). The mapping is anchoring-free:
  the former constant K and its median/level anchoring are gone. γ_surface
  in each line's `calibration:` block is either manual `{value, why}` or
  `solve`: the config driver matches the measured
  bed-window level by power-sum inversion — the modeled surface-clutter
  floor is subtracted from the measured level before reading the bed, so
  the solve is exact at any contamination level and needs only a seed run
  plus a verify run (the bed returns move dB-for-dB with the constant).
  Passes whose measured window has no headroom above the modeled clutter
  floor do not vote, and qualifying passes that disagree are flagged as
  missing physics. It cannot come from the attenuation regression's intercept
  (degenerate with mean bed reflectivity), which is why it needs a
  simulation. Its offset from smooth Fresnel (−11.03 dB) is recorded as a
  per-line surface anomaly, and the solve history lands in
  `run_config.json`. The 2026-08-20 solve sweep returned +4.3 (getz),
  +7.4 (david), −3.7 (westcoast), −2.0 with a 21.8 dB pass-disagreement
  flag (geikie) — none physically plausible as true surface
  reflectivities (they absorb chain/model anomalies), so **every line
  pins γ_surface = −10 dB manually** and the anomaly stays visible in
  the recorded residuals and the per-run `gamma_solve` diagnostic
  (per-pass numbers: `claude_notes/gamma_solve_design_2026-08-20.md`).
  An optional specular/diffuse split with a
  tilt-gated specular component models angle-dependent bed scattering.
- Englacial attenuation: constant one-way dB/km per line, set in the
  line's `calibration:` block as either a manual `{value, why}` pair or
  `solve` — a Theil–Sen regression of RSSNR on 2H over the line's own
  store samples (dataset-only; censored samples excluded; floating samples
  excluded when the line has a grounding line). Current values
  (2026-08-20): antarctica_david **12.8 solved** [CI 11.5–14.1, r = 0.89];
  antarctica_getz **20 manual** (regression diagnostic 18.6 [5.2–30.4] —
  consistent but weak leverage); greenland_geikie01_transit **14 manual**
  (the regression's γ_bed–thickness independence assumption is rejected
  there: a thawed-bed Γ–H confounder gives A ≈ 0.7, r = 0.11);
  greenland_westcoast **34.3 solved** [29.6–38.4, r = 0.85].
- Processing: the simulated stacks can be passed through a chain matched to
  CSARP_standard (product-posting simulation, motion compensation,
  time-domain backprojection at the alias-limited aperture, multilook)
  so texture and levels are comparable to the measured product.

## Analysis conventions

- Decomposition: per-interface fields let every figure and metric separate
  **surface returns** from **bed returns** — essential because at high
  altitude the "bed window" of the total field is often dominated by
  off-nadir surface clutter.
- Metrics use incoherent mean power over windows and traces. Bed models are
  scored on the bed-return component when the total is surface-dominated.
- All figure framing (time/dB windows, color scaling) is per-line; per-panel
  robust scaling is used where passes' dynamic ranges differ widely.
- Simulation chunks and focused stacks are cached (`--proc-cache`), so
  figure iteration and re-analysis do not re-simulate.

## Headline results to date

- The measured growth of ice-column clutter with platform altitude is
  reproduced by surface+bed geometry on the Antarctic line, and the
  decomposition attributes it to off-nadir surface returns arriving at
  bed-range delays at every altitude.
- Design ladder (Antarctic line, best-model config): the bed remains
  visible above the clutter at ~14-30 km platform altitudes, while at
  orbital altitudes (300-500 km) the grounded bed is buried by 13-26 dB;
  specular targets (ice-shelf base) remain detectable longest.
- RSSNR-driven reflectivity reproduces measured along-track bed-brightness
  structure (correlation ~0.6-0.8 against data ceilings of 0.9) on both a
  West Antarctic coastal line and a Greenland interior line.
- On the thick Greenland interior line, surface+bed geometry cannot explain
  the measured column power (flat with altitude where geometry predicts
  +17 dB): the column there is englacial-scattering dominated.

## Reproducing a study

Each named result is a committed declarative spec:

```
uv run python tools/run_basal_clutter.py --config experiments/<name>.yaml
```

`config/README.md` indexes them with status, dependencies, and runtime;
`tests/test_experiment_specs.py` asserts that every spec reproduces the
`run_config.json` of the directory it claims to build.

The line-level ground truth is reproducible without simulating anything:
`tools/line_report.py` surveys a line's real passes (map, aligned
radargrams, pass-agreement metrics), and `tools/calibrate_line.py` reports
every line's calibration block and attenuation-regression diagnostics
straight from the RSSNR store.

Detailed chronology, per-study findings, and data-source scouting notes
live in `claude_notes/` (see `agent_handoff_2026-08-17.md` for the index).
