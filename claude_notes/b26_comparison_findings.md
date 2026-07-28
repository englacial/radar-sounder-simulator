# B26 measured-vs-simulated comparison (2026-07-09)

Frame 2019_Greenland_P3 / 20190418_01_009 passes 5.75 m from the B26 firn
core (closest trace 434; claude_notes/firn_core_flightlines.md). Tool
`tools/run_b26_comparison.py` (resumable, pilot-budgeted); deliverable
`outputs/b26_comparison/` (self-contained report.html + metrics.json, group
"xOPR clutter" / case "b26_comparison", mirrored into
`outputs/verification/b26_comparison/` for tools/make_report.py); light
integration test `tests/integration/test_b26_comparison.py`.

## 2019 season parameters (frame's own param structs; NOT the 2017 values)

`outputs/cache/mcords_2019P3_params.json` from
`Data_20190418_01_009_source.mat` (param_records/param_sar/param_array):
chirp 180-210 MHz (f0 195, B 30 MHz) -- same band as 2017 -- but Tpd
{1, 3, 10} us (bed waveform 10 us), 20% Tukey, ft_wind = @hanning, rx_paths
all 7 center elements, prf 12 kHz, raw fs 111.11 MHz, and the PRODUCT twtt
grid is dt = 16.667 ns (60 MHz), not 2017's 33.333 ns. CSARP_standard: f-k
SAR sigma_x 2.5 m, 11-look hanning, dline 6 (~15 m posting; measured trace
spacing 14.7 m).

## Configuration (final; NO shrink steps needed)

10 km sub-segment centered on the closest approach, 100 traces (103 m
spacing), surface+bed cross-track +-3 km, firn strip +-600 m; dt_sim =
dt/4 = 4.1667 ns (alias 45 MHz = 3B/2, warning asserted silent; decimate
[::4] exact onto frame bins), window frame bins [119, 2165] (2.25-36.35 us).
Facets: beta=0.5 Fresnel minimized over the stack (deepest firn layer binds)
SNAPPED to 32/3 = 10.667 m so whole-cell crops tessellate on the wide
window's exact facet lattice; LPA nadir error 17% (worst case, recorded).
Media: air / ice(3.17, 15 dB/km one-way, the M24 warm-ice value -- generous
for this cold interior site, recorded) / bed(eps 8); firn = B26 point-sampled
Kovacs eps, equal placement 1-119.66 m, N in {10, 20}.

Firn contribution ran on the narrow strip as 6 ALONG-TRACK CHUNKS cropped
from the wide scene's DEM (one bbox around the whole diagonal segment would
carry ~5x the strip area -- the chunking is what fit the budget), then
field-summed with the wide surface+bed run, EXCLUDING the firn run's own
surface layer (no double count). Seam verified: firn layer-0 field, scaled by
the air->firn0 / air->ice gamma ratio, matches the wide run's surface field
to 1.24e-3 median max-relative-deviation in the first 1.5 us (identical
facet lattice makes this tessellation-noise-free).

## Budget: pilot projection vs actual

2-trace pilots: firn N=20 chunk 23.4 s steady (56.5k facets), wide 2.4 s
(1.76M facets). Projection 1596 s total -> under the 6000 s ceiling with the
FULL configuration. Actual: wide 74.4 s, firn_N10 353.2 s, firn_N20 1458.3 s
= 1886 s (+18% over projection; per-chunk compile). Total incl. pilots
~32.4 min.

## Results (metrics.json)

- surface_pick_alignment: median 0.00 frame bins (p90 1.0, offset -3.0
  bins) PASS; bed_alignment median 4.25 bins vs BedMachine-vs-pick input
  floor 3.94 (threshold 8.9) PASS, offset +8.2 bins.
- Bed depth at site: BedMachine 2600 m vs pick-derived 2597 m.
- Nadir depth-power at the closest trace (dB rel own surface peak, bands
  5-20 / 20-60 / 60-120 m): measured -20.4 / -19.1 / -34.4; surface+bed
  -38.7 / -57.2 / -64.6; N=10 -23.9 / -30.9 / -49.3; N=20 -23.3 / -30.8 /
  -41.8. Profile dB correlation (5-150 m) 0.63 -> 0.88 (N=10) -> 0.90 (N=20).

## Qualitative findings

1. Surface+bed alone cannot reproduce the measured near-surface return: it
   decays ~20-38 dB below the measured level within 60 m of the surface.
   Adding B26 firn layers recovers the measured MORPHOLOGY: a near-surface
   shoulder at ~-20 dB and a slow decay through the firn column.
2. N=10 (13.2 m spacing, resolved vs the 4.4 m in-firn chirped resolution)
   shows a discrete echo train with deep inter-layer nulls -- visibly banded
   in the radargram panels; N=20 (6.25 m, marginally resolved) is smoother
   and raises the 60-120 m band by ~7 dB, tracking the measured profile
   better (r 0.90). Consistent with the firn investigation's
   resolved->unresolved transition; per that study the levels are NOT
   converged in N (3-6 dB per doubling) and random placement (the physical
   case) shifts levels vs equal placement -- morphology comparison only.
3. Both firn sims sit ~8-12 dB below the measured 20-100 m band and drop to
   the surface+bed floor beyond the core's 119.66 m depth range, where the
   measured profile keeps decaying smoothly (real stratigraphy continues;
   the model's layer stack simply ends).
4. Measured-vs-sim processing asymmetry stands (f-k SAR + 11 looks vs
   unfocused per-trace raw at 103 m spacing): structure/relative levels are
   the claim, not texture or absolute calibration.
