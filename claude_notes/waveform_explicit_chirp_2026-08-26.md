# Explicit-chirp compressed pulse (2026-08-26)

Branch `waveform-explicit-chirp` (worktree
`.claude/worktrees/agent-a70c6d1cabd5d33f1`), not merged.

## Why

`compressed_pulse` was the analytic stationary-phase windowed sinc: the
B*T -> inf limit, with tails ~1/(pi*B*tau). It has essentially no pulse-
length dependence beyond the +-T truncation, so the HAPS 20 us vs 5 us
experiment (`wc_pilot_haps_pulse`) produced fields differing by 6.6e-7.
A real finite-TB chirp leaves an O(1/sqrt(TB)) Fresnel-ripple pedestal
(mission design tool `build_sidelobes.py`: ~-55 dB at 8-12 us for
B = 15 MHz, T = 20 us, Hann), and a 5 us pulse has no response past 5 us.

## What changed

- `src/soundersim/config.py`: `WaveformConfig.construction:
  Literal["analytic","chirp"] = "analytic"`.
- `src/soundersim/waveform.py`: `compressed_pulse(..., construction=)`;
  `_chirp_pulse` builds `exp(j*pi*(B/T)*t^2)` at the simulation dt
  (internally oversampled so the rate is >= 2B), `fftconvolve`s it with
  the raised-cosine-weighted conjugate replica (weighting on receive),
  decimates onto the dt lattice, zeroes |lag| > T, normalises the lag-0
  sample to exactly 1+0j. Returns complex128, same `(p, M)` shape, so
  `convolve_fast_time`/`apply_waveform` are unchanged (incoherent
  envelope now uses |p|^2). Module docstring documents both constructions.
- `tools/clutter_instruments.py`: `Simulated.construction` (default
  analytic; never read from an OPR frame). `resolve()` adds
  `pulse_compression_construction` to the waveform dict ONLY when not
  analytic; `provenance_block` records it likewise.
- `tools/run_basal_clutter.py`: `radar_grid` passes the construction into
  `WaveformConfig`; new `wave_meta(p)` feeds `chunk_meta` and `chunk_rid`.
- `docs/processing_simulation.md`: "Two compressed-pulse constructions".
- Tests: `tests/test_waveform_chirp.py` (new), one test appended to
  `tests/test_instruments.py`.
- Configs: `haps_60mhz_5us.yaml`, `wc_pilot_haps_pulse.yaml` (copied from
  the untracked main-repo files), new `haps_60mhz_chirp.yaml`,
  `haps_60mhz_5us_chirp.yaml`, `wc_pilot_haps_chirp.yaml`.

## Cache-key handling

The waveform convolution runs INSIDE the cached chunk (`simulate.py` ->
`apply_waveform`, before `run_level` saves `field`), so it must be in the
key. `wave_meta(p)` returns `{}` when `rc_sim.waveform.construction ==
"analytic"` and `{"waveform": {"construction": "chirp",
"pulse_length_us": 20.0}}` otherwise; `chunk_meta` splats it in and
`chunk_rid` appends `_wchirp` to the file name. Every existing meta_key
string and rid is therefore byte-identical (asserted in
`test_chirp_construction_plumbs_through_and_forks_the_cache_key` and the
pre-existing `test_default_instrument_leaves_the_cache_key_untouched`).
Pulse length for analytic instruments is still keyed only via the
instrument name, as before.

## Numerics (B = 15 MHz, Hann, dt = 8.3 ns)

| lag behind peak | chirp T=20 us | chirp T=5 us | analytic |
|---|---|---|---|
| 3 us  | -55.7 dB | -41.4 dB | -104.9 dB |
| 5 us  | -53.0 dB | -66.9 dB | -120.1 dB |
| 10 us | -53.0 dB | 0 (past T) | -139.3 dB |

Mainlobe/first-sidelobe agreement with the analytic form at TB = 3000:
max |diff| 5e-4 (hann), 9e-4 (none), 5e-4 (hamming). Max imaginary part
0.005 (20 us) / 0.02 (5 us).

## Tests

`uv run pytest tests/test_waveform_chirp.py tests/test_waveform.py
tests/test_waveform_pedestal.py tests/test_instruments.py
tests/test_experiment_specs.py tests/test_config.py tests/test_basal_lines.py -q`
-> 101 passed, 1 skipped, 1 deselected (2.5 s).

## Run: wc_pilot_haps_chirp

`uv run python tools/run_basal_clutter.py --config
config/experiments/wc_pilot_haps_chirp.yaml`, log
`claude_notes/logs/wc_pilot_haps_chirp.log`. Passes: p3_2017 (real,
3 x 55 s), haps_14km_chirp, haps_14km_5us_chirp (3 x ~5 s each). All nine
chunks printed `[ok]` (fresh), none `[skip-exists]`; the chirp run jsons
carry `meta.waveform = {construction: chirp, pulse_length_us: 20.0 / 5.0}`
and `_wchirp` rids.

Field difference, 20 us vs 5 us chirp, max|dF|/max|F| per chunk:
1.58e-2, 1.68e-2, 1.70e-2 (was 6.6e-7 with the analytic form).

Per-pass metrics (`outputs/greenland_westcoast/wc_pilot_haps_chirp/metrics.json`),
with the analytic `wc_pilot_haps_pulse` run for reference (identical for
both pulse lengths there):

| metric | haps_14km_chirp (20 us) | haps_14km_5us_chirp (5 us) | analytic (either) |
|---|---|---|---|
| clutter midcol_rel_surf_db | -48.14 | -46.16 | -49.24 |
| clutter bed_rel_surf_db (surface returns at the bed) | -60.29 | -67.22 | -67.30 |
| scout_midcol_over_bedpeak_db | 1.36 | 0.27 | 0.26 |
| bed_visibility: bed_over_surface_clutter_in_bed_window_db | -26.48 | -19.44 | -- |
| bedpeak_over_midcol_db | -1.36 | -0.27 | -- |
| bed_return_tail slope (dB/us, picked_bed) | -3.08 | -2.63 | -2.58 |
| bed_return_tail level_rel_surf_db +1/+2/+3 us | -59.8 / -62.8 / -66.8 | -61.6 / -65.9 / -69.7 | -61.7 / -66.0 / -69.7 |

Reading: the 20 us pulse's Fresnel pedestal raises the surface-returns
floor at the bed delay by 7 dB (-60.3 vs -67.2 dB) and worsens bed
visibility by 7 dB (-26.5 vs -19.4 dB); the 5 us chirp lands within
0.1 dB of the analytic form at the bed (its pedestal ends before the bed
window) but is 3 dB higher in the mid-column, where the near-in
sidelobes are now real. Mid-column verdict remains "surface returns" for
all passes. p3_2017 (analytic, unchanged) is a context pass only.

Figures (worktree):
- /home/thomasteisberg/Documents/coherent-radar-simulator/.claude/worktrees/agent-a70c6d1cabd5d33f1/outputs/greenland_westcoast/wc_pilot_haps_chirp/radargrams.png
- /home/thomasteisberg/Documents/coherent-radar-simulator/.claude/worktrees/agent-a70c6d1cabd5d33f1/outputs/greenland_westcoast/wc_pilot_haps_chirp/decomposition.png
- /home/thomasteisberg/Documents/coherent-radar-simulator/.claude/worktrees/agent-a70c6d1cabd5d33f1/outputs/greenland_westcoast/wc_pilot_haps_chirp/decomposition_trace.png
- also bed_tail.png, report.html in the same directory

Setup note: `outputs/cache` in the worktree is a symlink to the main
repo's data cache (runs write only under outputs/<line>/<case>/).
