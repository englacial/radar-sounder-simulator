# Configuration

```
config/lines/<name>.yaml         DATA    — where: frames, slices, segments, bed DEM, calibration
config/instruments/<name>.yaml   RADAR   — carrier, bandwidth, pulse, window, antenna
config/experiments/<name>.yaml   METHOD  — bed construction, reflectivity, physics, processing
config/analysis.yaml             RULER   — what the metrics mean (study-wide, never per run)
config/roughness/atm_b1.yaml     measured ATM surface spectra (opt-in surface roughness source)
```

Run one experiment on one line:

```
uv run python tools/run_basal_clutter.py --config config/experiments/pilot.yaml --line <line>
```

Only `--line`, `--out` and `--force` may accompany `--config`; every physics
knob comes from the file. Outputs land in `outputs/<line case_prefix>/<experiment>/`.

## Experiments

Exactly two, identical apart from the segment, valid on every line:

| experiment | segment | what |
|---|---|---|
| `full` | the line's full overlap window | the study result |
| `pilot` | 10 km, 48 traces | the same protocol, minutes per line |

Both simulate the real passes as flown plus the cross-line design points
(`haps_60mhz_6el_halflambda` and `haps_60mhz_6el_lambda` at 14 km riding
each line's reference pass). Anything
line-specific (DEM, calibration, geometry) is read from the line, so the
two files carry no per-line numbers. A one-off study is a copy of one of
these with one thing changed; it does not get committed.

Schema (`tools/clutter_spec.py`): `meta {name, description}`; `run` with
`lines`/`line`, `segment`, `passes`, `instruments` (swap the radar of a
real pass), `extra_passes` (invent an observation: carrier, altitude,
instrument), `bed`, `reflectivity`, `physics`, `processing`, `figures`.
Typos fail at load; an `analysis:` block is rejected outright.

## Bed: data on the line, method in the experiment

```yaml
# line                              # experiment
identity:                           bed:
  bed_dem: demogorgn | bedmachine     nadir: picked | dem
                                      floating: picked
```

- `bed_dem` — the DEM available for the grounded ice: DEMOGORGN (Antarctic
  ensemble realization; seed in the experiment) or BedMachine.
- `nadir: picked` — the grounded bed is the DEM plus an along-track residual
  pinning nadir to the reference pass's radar picks (cross-track structure of
  the DEM preserved).
- `floating: picked` — on a segment that declares `crosses_gl`, the floating
  bed is the reference pass's picks (the ice-ocean interface; DEMs report the
  seafloor), blended over `analysis.yaml: hybrid_bed.gl_ramp_km` past the GL.

So `nadir`/`floating` mean the same thing on every line; the only cross-line
difference is the DEM, and it is data.

## Lines

Geometry and data only. Pass names are `<platform>_<year>[_<agl>]`. Each pass
names the instrument that actually flew it (a synthetic instrument as a line
default is refused at import). Every line has exactly two segments, `pilot`
and `full`; a segment needs at least two covering passes including the
reference pass.

| line | kind | passes | bed DEM |
|---|---|---|---|
| `antarctica_getz` | altitude | `dc8_2016_{0,9,11}km` | DEMOGORGN; GL at s 69.7 km |
| `antarctica_pineisland_south` | altitude | `dc8_{2014,2016,2018}_0km`, `dc8_2012_9km`, `dc8_2009_10km` | DEMOGORGN; GL at s 6.7 km; three radar generations |
| `antarctica_pineisland_north` | altitude | `dc8_{2014,2016,2018}_0km`, `dc8_2012_9km` | DEMOGORGN; GL at s 66.1 km; thin ice (529 m median) |
| `antarctica_david` | frequency | `basler_2017` (195 MHz), `baslermkb_{2022,2023}` (60 MHz) | DEMOGORGN; GL at s 95.4 km; reference is the 60 MHz 2023 pass |
| `greenland_geikie01_transit` | altitude | `p3_2014_low`, `p3_2017_high` | BedMachine; `full` contains a turn flown on different radii (s 40–80 km) |
| `greenland_westcoast` | instrument | `p3_2016` (MCoRDS5), `p3_2017`, `p3_2019` (MCoRDS3) | BedMachine |

The two Pine Island lines share one OIB survey ladder over the upper trunk and
were selected in `claude_notes/pig_line_scout_2026-08-28.md`; `_south` (lat
−75.35 → −75.56) carries the trough step and a third altitude level, `_north`
(lat −75.17 → −74.96) carries thin ice with the bed visible in every pass.
Both take their surface-roughness law from the Tier 2 `aa_grounded_500_1500`
stratum as a PROVISIONAL stand-in — no line-specific ATM fit yet, see the note
in `config/roughness/atm_tier2_strata.yaml`.

`calibration:` holds the two physical mapping parameters, `gamma_surface_db`
(manual `{value, why}` or `solve`) and `att_db_per_km` (manual or `solve` =
Theil–Sen regression of RSSNR on 2H over the line's store samples). Every
line pins γ_surface at −10 dB; geikie pins A = 16 because the regression's
independence assumption fails there. `A` and γ are recorded in every run
config together with the regression diagnostic. `tools/calibrate_line.py`
reports them without simulating.

A line may override a subset of `analysis.yaml` (`analysis:`); none does.

## Instruments

Real systems are `<instrument>_<platform>_<year>` (first year when several
share one configuration), `source: {kind: opr_frame}`: carrier, bandwidth,
pulse and window are read per pass from its OPR frame; `segments` lists the
`YYYYMMDD_SS` segments it covers and a pass pinned outside them fails at
import. Only the antenna is modelled here; `recorded:` carries link-budget
numbers no code consumes.

| instrument | OPR readme | modelled antenna |
|---|---|---|
| `mcords3_p3_2014` | MCoRDS3 on P-3, 2014 + 2017 Greenland | 7 el, 0.5 λ |
| `mcords3_p3_2019` | MCoRDS3 on P-3, 2019 (steered L/R beams unmodelled) | 7 el, 0.5 λ |
| `mcords5_p3_2016` | MCoRDS5 on NOAA P-3, 2 channels, 0.61 m | 2 el, 0.41 λ |
| `mcords3_dc8_2016` | MCoRDS3 on DC-8 2014/2016/2018, 3 × 2 element array | 3 el, 0.45 λ (OPR lever_arm.m) |
| `mcords2_dc8_2012` | MCoRDS2 on DC-8, 2012 (193.9/9.5 MHz) | 5 el, 0.2546 λ (OPR lever_arm.m) |
| `mcords_dc8_2009` | MCoRDS on DC-8, 2009 (193.9/9.5 MHz) | 5 el, 0.2546 λ, no roll (product has no attitude) |
| `mcords5_basler_2017` | MCoRDS5 on Basler, 8 channels, 3.7 m | 8 el tapered, from the product param structs |
| `marfa_baslermkb_2022` | not in the readme (2022/2023) | right-wing 0.4 λ dipole, no beamforming |
| `haps_60mhz` | synthetic (`kind: stated`) | 8 el, 0.5 λ, no roll |
| `haps_60mhz_6el_halflambda` | synthetic constrained design | 6 el, 0.5 λ, central pair TX, Taylor 40 dB RX, 3 dBi elements, no roll |
| `haps_60mhz_6el_lambda` | synthetic constrained design | 6 el, 1.0 λ (25 m span), central pair TX, Taylor 40 dB RX, 3 dBi elements, no roll |

The resolved antenna is fingerprinted into the chunk cache key; a swapped
instrument forks the key. Quote segment and frame ids (`'20161105_05'`):
YAML 1.1 reads `_` as a digit separator.

For matched processing, `processing.posting_div` refines the simulated
slow-time grid before focusing. `processing.focus_aperture` is
`alias_limited` (use the refined grid's full unaliased band),
`product_resolution` (retain the aperture and resolution of the original
product posting), or `first_fresnel` (one monostatic first-zone diameter).

## Analysis conventions

`config/analysis.yaml` defines the measurement windows, noise-floor rules,
bed-tail fit, the grazing-angle facet-lattice fix (`grazing_fix.s_eff`, a
bug fix that is on for every run), the γ/A solver settings, the hybrid-bed
ramp and the compute tuning (`chunk_m`, `facet_spacing_scale`; both are in
the cache key).

## Surveying a line

```
uv run python tools/line_report.py config/lines/<name>.yaml [--segment S]
```

writes a map, aligned radargrams and `metrics.json` (offsets, coverage,
surface/thickness agreement) to `outputs/line_reports/<line>/<segment>/`.

## Guarantees (tests)

`tests/test_experiment_specs.py`: exactly `full` and `pilot` ship, they
differ only in segment, and they cover every line with the same HAPS points.
`tests/test_instruments.py`: the default instrument leaves cache keys
byte-identical and a swap forks them. `tests/test_basal_lines.py`: line
activation is total and reversible.
