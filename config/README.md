# Configuration

Three kinds of file, split by what varies independently:

```
config/lines/<name>.yaml         GEOMETRY   — where: frames, slices, segments, framing
config/instruments/<name>.yaml   THE RADAR  — carrier, bandwidth, pulse, window, antenna
config/experiments/<name>.yaml   THE GLUE   — which line, which observations, which radars,
                                              reflectivity, bed source, processing
config/analysis.yaml             THE RULER  — what the metrics MEAN (study-wide)
```

Run one:

```
uv run python tools/run_basal_clutter.py --config config/experiments/<name>.yaml
```

Only `--out` and `--force` may accompany `--config`; every physics knob comes
from the file, so a published result and the spec that names it cannot drift.

## Why three files

Two mission-design questions need to vary **independently**:

- *Would a different radar still see this bed?* — swap the instrument, hold
  the geometry.
- *What happens from higher up?* — hold the instrument, change the altitude.

If altitude lived in the instrument (as it does in the mission design tool's
presets) those axes would be welded together. So here **altitude is a property
of the observation**, and the instrument is only the box.

## Lines

Geometry and nothing else: CRS, the real passes with their frame slices and
measured AGL, the study segments, figure framing, the RSSNR store pin, and the
provenance prose recorded into every run built on the line.

Each pass names the instrument that **actually flew it** — a default an
experiment may swap. A synthetic instrument cannot be a line default: claiming
a design point flew a real line would be false provenance, and it is refused
at import.

## Instruments

Field names follow
`radar_return_statistics_postprocessing/mission_design_tool`, so one config can
describe a system to both tools. Two kinds:

- `source: {kind: opr_frame}` — a **real** system. Every simulated parameter is
  read from the OPR frame the pass was flown on. `segments:` lists the
  `YYYYMMDD_SS` segments it covers; data from one segment is one instrument, so
  a pass whose `param_frame` falls outside its instrument's segments is a
  mis-pinned config and fails at import rather than 40 minutes in.
- `source: {kind: stated}` — a **synthetic** system. Values are the design; no
  OPR frame is consulted, and every simulated field must be given.

This simulator is **clutter-limited** — no receiver-noise model, no link
budget — so it consumes only `frequency_Hz`, `bandwidth_Hz`, `pulse_length_s`,
`window` and the antenna. Link-budget fields (tx power, gains, losses, noise
figure) live under `recorded:`: carried into the run config as provenance,
consumed by nothing here *yet*, so wiring a link budget later needs no
re-authoring.

Stating a value on an `opr_frame` instrument is legal but never silent — it is
reported in the run config as `deviations_from_recorded_system`.

> **Quote your ids.** YAML 1.1 treats `_` as a digit separator, so an unquoted
> `20161105_05` becomes the integer `2016110505`. Segment and frame ids must be
> quoted; the loader says so explicitly if you forget.

## Experiments

Two swap axes, both optional:

```yaml
run:
  line:    greenland_2014_2017
  passes:  [low, high, haps14km]

  instruments:                   # axis 1: same geometry, different radar
    high: haps_60mhz

  extra_passes:                  # axis 2: an observation this run invents
    haps14km:
      carrier:    low            # its line geometry, picks and nav
      altitude_m: 14000.0
      instrument: haps_60mhz     # ...and a different radar too
```

Anything not named keeps the line's default. A swapped instrument **forks the
chunk cache key**, so it can never silently reuse the real instrument's
simulated chunks.

## Analysis conventions

`config/analysis.yaml` holds the measurement definitions: where the
mid-column window starts, which delays the bed tail is fitted over, what
counts as a trustworthy noise floor, the cross-track coverage margin.

**An experiment cannot set these.** A per-run window is an invitation to move
the bed window until the residual looks good, which is metric shopping rather
than measurement — the spec schema rejects an `analysis:` block outright.

**A line may override a subset**, because some are properties of the data
rather than of the study: a pass recording only ~8 µs of post-bed tail cannot
use a floor window sized for 21 µs. An override is merged over the defaults,
and both the resolved values and a `line_overrides` diff land in every run
config, so a line that measures differently says so out loud. Neither shipped
line currently overrides anything, and a test asserts that — so the first one
has to be deliberate.

### Calibration

The RSSNR → bed-reflectivity mapping is **anchoring-free**:

```
|Γ_bed|²(x) = 2·A·H(x) − RSSNR(x) + (γ_surface − T²)
```

γ_surface is the line's **effective** surface power reflectivity — the
RSSNR dataset is surface-referenced, so the mapping constant is direct and
segment-independent (the old `K_ANCHOR_SEGMENT` machinery is obsolete). T²
is the two-way Fresnel transmission (~−0.71 dB), computed, never
configured.

Each line file carries a `calibration:` block with exactly two parameters:

- `gamma_surface_db` — manual `{value, why}` or the literal `solve`
  (**the study default**). The solve zeroes the qualifying-median bed-level
  residual against the measured passes: the `--config` driver runs the sim
  at a seed γ (`analysis.yaml: gamma_surface_solve.seed_db`, −10), shifts by
  the median residual over the **qualifying** passes (those whose sim bed
  window is bed-dominated by ≥ `min_bed_over_surface_db`), and verifies with
  one more run (|residual| ≤ `tolerance_db`) — exact in one step because the
  received bed level shifts dB-for-dB with the mapping constant. It cannot
  come from the RSSNR regression intercept (degenerate with the mean bed
  reflectivity), which is why this solve needs simulations while the A solve
  does not. The solve history is recorded in `run_config.json`
  (`calibration_resolution.gamma_surface_solve_history`).
- `att_db_per_km` — either manual `{value, why}` or the literal `solve`.

`solve` is a Theil–Sen linear regression of RSSNR on 2H over the line's own
store samples — dataset-only, no simulation needed. Censored samples are
excluded (never floored), and it is GL-aware by default
(`calibration.gl_aware`, default true): floating samples are excluded when
the line has a grounding line. Its declared assumption: γ_bed uncorrelated
with thickness, γ_surface and A constant along the line. That assumption is
**rejected on the geikie line** — a thawed-bed Γ–H confounder gives the fit
A ≈ 0.7 with r = 0.11 — so geikie pins A = 14 manually. The regression is
still computed as a **diagnostic** on every line, manual or not, and
recorded.

The old quantities survive only as diagnostics. K is recorded as
`k_db = γ_surface − T²`; what used to be "K − K_phys" is now the **surface
anomaly** (γ_surface minus the smooth-Fresnel −11.03 dB). The level deficit
D and level anchoring are deleted — post-run bed-level residuals are
recorded as chain diagnostics (metric `rssnr_level_residuals`), never
absorbed into the mapping. The old `chain_closure` attenuation rule was
proven vacuous on 2026-08-19 (K − K_phys is invariant in A when the level
is absorbed) and is replaced by the regression; `analysis.yaml` now carries
`attenuation_regression` settings instead of `attenuation_rule`.

Current calibrations (2026-08-20, `outputs/line_reports/calibrations.json`):

| line | γ_surface (dB) | surface anomaly | A (dB/km) |
|---|---|---|---|
| `antarctica_david` | −11.03 (Fresnel default) | — | **12.8 solved** [CI 11.5–14.1, r = 0.89] |
| `antarctica_getz` | +7.21 effective | +18 dB, un-audited | **20 manual** (diagnostic 18.6 [5.2–30.4] — consistent but weak leverage) |
| `greenland_geikie01_transit` | −10.21 | +0.8 dB, chain honest | **14 manual** (regression rejected, see above) |
| `greenland_westcoast` | −3.69 | +7.3 dB, suspected product radiometry | **34.3 solved** [29.6–38.4, r = 0.85] |

Report every line's calibration and regression diagnostics without
simulating:

```
uv run python tools/calibrate_line.py
```

Note `compute.chunk_m` is tuning rather than science, but it sets the chunk
count, which is part of the cache key: changing it re-simulates everything.

## Benchmarks

`meta.role: benchmark` marks an experiment kept for fidelity regression — the
"did a simulator change help or hurt" loop — with an `expected:` block of
acceptance numbers to score against. Real-instrument lines are the natural
benchmarks: their measured data is the reference.

A **segment is a window on the line, not a roll-call**: a pass may omit
windows it does not reach. Requiring every pass to cover every window caps a
multi-year line at the extent of its shortest flight — on the west coast,
15.2 km instead of 49.8. A window still needs at least two passes, one of
which must be the reference (there has to be an axis to project onto).

## Surveying a line

```
uv run python tools/line_report.py config/lines/<name>.yaml [--segment S]
```

Writes to `outputs/line_reports/<line>/<segment>/`: a **map** of the flight
data used (shared span bold, whole frames faint), **radargrams** trimmed to
the span every pass shares and aligned on each pass's own surface pick, and
**metrics.json** — lateral offset, along-track coverage, surface-elevation
and ice-thickness agreement against the reference pass, and whether the
passes were even flown on the same fast-time lattice.

Offsets come from each frame's **own nav**, never from the STAC geometry:
STAC carries a coarse decimation of the track and can misplace it by hundreds
of metres, which is fine for discovering candidates and useless as a metric.
Radargrams are normalised by **one scalar per pass** — that pass's median
surface return over the segment — on a common depth-in-µs axis, because
absolute product scaling and bin indexing are not comparable across seasons.
Per-trace normalisation would flatten exactly the along-track variation the
radargram exists to show.

## Index

### Lines

| line | kind | passes | notes |
|---|---|---|---|
| `antarctica_getz` | altitude | 3 real (0.4/9.2/10.7 km) + 2 synthetic | grounding line at s 69.7 km |
| `antarctica_david` | **frequency diversity** | 60 vs 195 MHz, plus a same-instrument repeat pair (2022/2023) | David Glacier / Drygalski; grounding line at s 95.4 km |
| `greenland_geikie01_transit` | altitude | 2 real (0.5/2.5 km) + 1 synthetic | altitude pair; `transit` is one 139 km path; it contains a turn the two aircraft flew on different radii (up to 1.3 km apart over s 40-80) |
| `greenland_westcoast` | **instrument** | 3 real, all ~460 m AGL | one altitude, three radars (195/30 MHz twice + 200/100 MHz) over one 49.8 km window |

### Multi-line protocols

A benchmark experiment may state `run.lines: [...]` instead of one line and
be selected at run time:

```
uv run python tools/run_basal_clutter.py \
    --config config/experiments/gl_std_benchmark.yaml --line <name>
```

Outputs cannot collide: each line's `case_prefix` gives the same experiment
name its own directory and cache. The spec itself carries no calibration
numbers: γ_surface comes from each line's `calibration:` block, and with
`att_db_per_km: solve` A comes from the line's own Theil–Sen regression —
both recorded in the run config.

### Experiments

| experiment | line(s) | instruments | status |
|---|---|---|---|
| `ant_full_line` | antarctica_getz | as flown | **adopted** |
| `gl_std_benchmark` | geikie OR westcoast (`--line`) | as flown | **adopted** |
| `pilot_smoke` | all four lines (`--line`) | as flown | benchmark |

`pilot_smoke` is the cheap end of the fidelity loop: a ~10 km `pilot`
segment per line — real passes, picked bed, RSSNR reflectivity, matched
processing — so a simulator change can be scored against every study line
before committing to a full-line run.

## Guarantee

`tests/test_experiment_specs.py` asserts every spec reproduces the
`run_config.json` of the directory it claims to build.
`tests/test_instruments.py` asserts the default instrument path leaves cache
keys byte-identical and that a swap forks them.
