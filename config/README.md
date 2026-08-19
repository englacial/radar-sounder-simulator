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

`level_anchor` is the one rule the whole study shares for solving the
level-anchor deficit D — contamination-aware (`bed·10^(D/10) + surface =
measured`), median over the passes whose bed returns stand ≥ 10 dB clear of
their surface returns. The exclusion is **derived** from the decomposition
rather than hand-listed, and one threshold reproduces both lines' previously
hand-made pass selections. Regenerate D from a completed constant-gamma run
with:

```
uv run python tools/derive_level_deficit.py <run_dir>
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
Radargrams are in dB rel each trace's own surface peak on a common
depth-in-µs axis, because absolute product scaling and bin indexing are not
comparable across seasons.

## Index

### Lines

| line | kind | passes | notes |
|---|---|---|---|
| `antarctica_getz` | altitude | 3 real (0.4/9.2/10.7 km) + 2 synthetic | grounding line at s 69.7 km |
| `greenland_geikie01_transit` | altitude | 2 real (0.5/2.5 km) + 1 synthetic | `transit` is one 139 km path; it contains a turn the two aircraft flew on different radii (up to 1.3 km apart over s 40-80) |
| `greenland_westcoast` | **instrument** | 6 real, all ~470 m AGL | four radars, 195/30 to 315/270 MHz; `full` = all six over 15.2 km, `long` = the three that reach 49.8 km |

### Experiments

| experiment | line | instruments | status |
|---|---|---|---|
| `ant_att20_klevel` | antarctic_2016 | as flown | **adopted** |
| `ant_extended` | antarctic_2016 | as flown | **adopted** |
| `ant_full_line` | antarctic_2016 | as flown | **adopted** |
| `gl_full_pbed_proc_att14` | greenland_2014_2017 | as flown | **adopted** |
| `gl_full_pbed_proc_att14_rssnr` | greenland_2014_2017 | as flown | **adopted** |
| `gl_haps60_at_14km` | greenland_2014_2017 | swap → `haps_60mhz` @ 14 km | exploratory |

`gl_full_pbed_proc_att14` must exist before `gl_full_pbed_proc_att14_rssnr`
(it supplies both the cached constant-gamma companion and the run D is solved
against). `requires:` is validated, never executed.

## Guarantee

`tests/test_experiment_specs.py` asserts every spec reproduces the
`run_config.json` of the directory it claims to build.
`tests/test_instruments.py` asserts the default instrument path leaves cache
keys byte-identical and that a swap forks them.
