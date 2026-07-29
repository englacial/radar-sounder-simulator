# OPR flight lines near the clutter-repo firn cores (2026-07-09)

Method: core coordinates from PANGAEA `.tab` headers in `~/Documents/clutter/data/`;
xOPR `query_frames` with a core-centered azimuthal-equidistant buffer; exact
frame-geometry distances in the same frame; frames deduped to flight lines
(season/segment), closest frame per line. Script archived in session scratchpad;
trivially reproducible with xopr.

## ngt37C95.2 (B26) — NGT, 77.2533 N, 49.2167 W — 5 lines within 1 km

| Flight line | Closest frame | Distance |
|---|---|---|
| 2019_Greenland_P3 / 20190418_01 | 20190418_01_009 | 10 m |
| 2017_Greenland_P3 / 20170328_01 | 20170328_01_055 | 356 m |
| 2015_Greenland_C130 / 20150515_03 | 20150515_03_021 | 358 m |
| 2014_Greenland_P3 / 20140508_01 | 20140508_01_061 | 375 m |
| 2011_Greenland_P3 / 20110506_01 | 20110506_01_026 | 929 m |

B26 is the core behind the firn-plateau investigation; 20190418_01_009 passes
10 m from the borehole — the natural measured-data comparison frame.

### Cross-year measured depth-power at B26 (2026-07-28)

`tools/run_b26_overflights.py` → `outputs/b26_overflights/` (metrics.json,
overflight_profiles.png). All five frames load from `CSARP_standard`;
`CSARP_qlook` exists for four (2011 has no qlook asset — only CSARP_standard,
data, flight_path, thumbnail, CSARP_layer). Profile method imported unchanged
from `run_b26_comparison` (closest-approach trace, own surface peak, dB rel
that peak, 5 m boxcar, depth via c/√ε̄, 5–200 m). Distances here are recomputed
in EPSG:3413 from the frame nav and run ~15 m short of the table above (2019:
6 m vs 10 m; different projection/segment). Note the products have different
fast-time postings (16.7 / 20 / 33.3 / 33.4 ns), so the 5 m boxcar floors at
3 bins and the 33 ns frames are effectively ~9 m smoothed; the frame datasets
expose no waveform attrs, so center frequency/bandwidth could not be recorded
(only dt/n_samples).

| Year / platform | Frame | Dist (m) | 20–70 m std (dB) | 20–70 m qlook (dB) | 80–120 m std (dB) | 150–200 m floor (dB) | r vs 2019 std | Products |
|---|---|---|---|---|---|---|---|---|
| 2019 P-3 | 20190418_01_009 | 6 | −20.2 | −20.5 | −36.2 | −59.8 | 1.000 | std + qlook |
| 2017 P-3 | 20170328_01_055 | 340 | −18.0 | −19.4 | −34.8 | −55.9 | 0.983 | std + qlook |
| 2015 C-130 | 20150515_03_021 | 355 | −14.7 | −21.1 | −28.8 | −48.9 | 0.973 | std + qlook |
| 2014 P-3 | 20140508_01_061 | 348 | −16.1 | −18.8 | −32.2 | −55.2 | 0.989 | std + qlook |
| 2011 P-3 | 20110506_01_026 | 905 | −10.8 | — | −21.7 | −45.8 | 0.979 | std only |

Interpretation. The *shape* of the profile is essentially instrument-independent:
every frame correlates r = 0.97–0.99 with the 2019 standard profile over
5–200 m, so the plateau-then-rolloff morphology at B26 is a property of the
firn, not of one product. The *level* is robust too once the processing
asymmetry is removed: the four unfocused qlook profiles sit at −18.8 to
−21.1 dB (median −20.0, spread 2.3 dB, σ 0.9 dB) across two platforms and
three different fast-time grids, i.e. the ~−20 dB mid-band plateau reproduces
independently four times. The wider 9.3 dB spread in the SAR-focused
CSARP_standard column tracks each season's dynamic range almost one-for-one —
the 150–200 m noise floor spans 14 dB in the same order, and mid-band-above-floor
is much tighter (34–40 dB) — so the hot 2011/2015 standard levels look like
per-season surface-peak/multilook gain differences (and, for 2011, a 905 m
offset and a coarse 33 ns grid), not more firn scattering. Bottom line for the
gap analysis: the measured target is not a fluke of frame 20190418_01_009, and
firn_N40 sits −17.3 dB below the cross-instrument qlook median (−21.2 dB below
the standard median), so the deficit is real and its magnitude is uncertain by
only a couple of dB on the measurement side — the remaining explanation has to
be on the simulation side.

## ngt27C94.2 (B21) — NGT, 80.0000 N, 41.1374 W — none within 1 km

Nearest: 1999_Greenland_P3/19990514_01 (frame _020) at 2.2 km; next nearest
12.5 km (2018_Greenland_P3/20180418_06).

## BER11C95_25 (B25) — Berkner Island, 79.6142 S, 45.7243 W — none within 1 km

Nearest: 2011_Antarctica_DC8/20111021_02_005 and 2014_Antarctica_DC8/20141108_04_003,
both ~7.3 km.
