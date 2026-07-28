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

## ngt27C94.2 (B21) — NGT, 80.0000 N, 41.1374 W — none within 1 km

Nearest: 1999_Greenland_P3/19990514_01 (frame _020) at 2.2 km; next nearest
12.5 km (2018_Greenland_P3/20180418_06).

## BER11C95_25 (B25) — Berkner Island, 79.6142 S, 45.7243 W — none within 1 km

Nearest: 2011_Antarctica_DC8/20111021_02_005 and 2014_Antarctica_DC8/20141108_04_003,
both ~7.3 km.
