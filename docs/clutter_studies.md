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
  tilt-gated specular component models angle-dependent bed scattering,
  and Gerekos sub-facet bed roughness broadens each facet's angular
  response; both are ON in every experiment spec since 2026-08-21
  (specular_fraction 0.5, tilt 3 deg, sigma 0.10 m at l = lambda_ice) --
  the evidence is `docs/bed_scattering.md`.
- Englacial attenuation: constant one-way dB/km per line, set in the
  line's `calibration:` block as either a manual `{value, why}` pair or
  `solve` — a Theil–Sen regression of RSSNR on 2H over the line's own
  store samples (dataset-only; censored samples excluded; floating samples
  excluded when the line has a grounding line). Current values
  (2026-08-20): antarctica_david **12.8 solved** [CI 11.5–14.1, r = 0.89];
  antarctica_getz **18.6 solved** [5.2–30.4, r = 0.41 — weak leverage,
  consistent with the earlier manual 20]; greenland_geikie01_transit
  **16 manual** (MacGregor arbitration, adopted 2026-08-20; the
  regression's γ_bed–thickness independence assumption is rejected there:
  a thawed-bed Γ–H confounder gives A ≈ 0.7, r = 0.11);
  greenland_westcoast **34.3 solved** [29.6–38.4, r = 0.85].
- Antennas: per-instrument patterns (cross-track arrays with nav roll for
  the MCoRDS3 systems; an 8-element amplitude-tapered array for the 2017
  Basler; a finite wing-plate dipole for the 60 MHz MKB), fingerprinted
  into the chunk cache keys. Isotropic is a declared clutter upper bound,
  not a default.
- Grazing-angle facet-lattice fix (default ON since 2026-08-24): the
  coherent facet response is tapered off-specular and the sub-facet
  roughness variance keeps only its area term, removing a facet-size
  -dependent (unphysical) grazing clutter floor. This is a bug fix, not a
  model option; `--no-grazing-fix` exists only for artifact demonstration.
  Root-cause record: `claude_notes/david_clutter_resolution_2026-08-24.md`.
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
- All figure framing (time/dB windows, color scaling) is per-line. Shared
  scaling is the default everywhere: per-panel robust percentiles stretch a
  simulated panel down to its numerical floor (there is no receiver-noise
  model), which renders −100 dB bed clutter as mid-grey next to a measured
  panel whose noise floor is black at the same level.
- Simulated radargram panels can be coloured by energy source (surface vs
  bed; brightness unchanged) with `figures.radargram.source_color: true` —
  see [source_color_radargrams.md](source_color_radargrams.md). Off by
  default.
- Simulation chunks and focused stacks are cached (`--proc-cache`), so
  figure iteration and re-analysis do not re-simulate. Chunk cache keys
  carry the kernel numerics era (`soundersim.kernels.KERNEL_VERSION`);
  bumping it re-simulates everything.
- Runtime (2026-08-25, all four full experiments, sequential on the
  9900X): getz 82 min, david 127, geikie 35, westcoast 84 — 5.5 h total
  (30.6 h before the per-trace facet windowing / fused bed path of
  2026-08-24). Cost per pass falls with altitude because the Fresnel-scaled
  facet spacing coarsens faster (`∝ √h`) than the reach grows; low-altitude,
  high-frequency passes dominate. The shipped experiments no longer
  runs the constant-gamma companion simulation (its bed-brightness
  correlation row is therefore absent from new reports).

## Headline results to date

> **Re-derivation pending (2026-08-24).** The grazing fix removed a
> clutter artifact that was present in every run behind the quantitative
> results below. Validation pilots show the qualitative findings survive,
> but the numbers need re-deriving at fixed physics: on getz the 9–10 km
> mid-column "match" becomes a 4–8 dB under-prediction and the low-pass
> mid-column is unexplained; on david the bed is no longer obscured (the
> original artifact symptom) but sits 7–22 dB dim with the two instruments
> disagreeing. The universal **mid-column under-prediction** is now the
> primary open physics question (candidates: englacial scattering, surface
> roughness spectrum beyond the C&S statistics).

- The measured growth of ice-column clutter with platform altitude is
  qualitatively reproduced by surface+bed geometry on the Antarctic line,
  and the decomposition attributes it to off-nadir surface returns arriving
  at bed-range delays at every altitude. (Quantitative closure was partly
  artifact — see the note above.)
- Design ladder (Antarctic line, best-model config): the bed remains
  visible above the clutter at ~14-30 km platform altitudes, while at
  orbital altitudes (300-500 km) the grounded bed is buried by 13-26 dB;
  specular targets (ice-shelf base) remain detectable longest. (Pre-fix
  numbers; the ladder should be re-run.)
- RSSNR-driven reflectivity reproduces measured along-track bed-brightness
  structure (correlation ~0.6-0.8 against data ceilings of 0.9) on both a
  West Antarctic coastal line and a Greenland interior line. (From the
  constant-gamma companion runs, switched off in the shipped experiments on
  2026-08-24 for runtime; re-enable `processing.companion` to reproduce.)
- On the thick Greenland interior line, surface+bed geometry cannot explain
  the measured column power (flat with altitude where geometry predicts
  +17 dB): the column there is englacial-scattering dominated. The post-fix
  results extend this class of gap to every line's mid-column.
- The David-line investigation (2026-08-24) resolved why simulated surface
  clutter obscured the bed there when measurements showed none: a kernel
  discretization artifact plus isotropic-placeholder antennas. With both
  fixed, every David pass's bed window is bed-dominated and all three
  passes qualify for the γ solve for the first time.

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
live in `claude_notes/` (see `agent_handoff_2026-08-24.md` for the index).
