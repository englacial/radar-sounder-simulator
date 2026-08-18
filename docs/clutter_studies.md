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
  `|Γ|²(x) = 2·A·H(x) − RSSNR(x) + K`, with the calibration constant K set
  by median (Fresnel-prior) or level (match measured bed brightness)
  anchoring (`--anchor`). An optional specular/diffuse split with a
  tilt-gated specular component models angle-dependent bed scattering.
- Englacial attenuation: constant one-way dB/km per line (`--att`).
  Antarctic 20161105_05 line: **20 dB/km, adopted** (level-anchored family
  analysis). Greenland 20140421_01 line: **14 dB/km is what the current runs
  use**; the MacGregor 2015 method re-applied to archived reflection
  intensities (with a Robin-profile full-column correction) arbitrates for
  16 ± 2 dB/km, but adopting it means re-deriving the level-anchor deficit
  against a new constant-gamma run, so it is pending rather than in force.
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

Detailed chronology, per-study findings, and data-source scouting notes
live in `claude_notes/` (see `agent_handoff_2026-08-17.md` for the index).
