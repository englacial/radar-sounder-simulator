# Firn fixtures (stage-3 M19: Culberg & Schroeder 2020 firn power plateau)

Copied 2026-07-07 from the user's read-only reimplementation repo at
`~/Documents/clutter` (per explicit direction: tests must be self-contained,
never referencing external paths at runtime).

## Files

- `ngt37C95.2_density.tab` — B26 ice-core density profile (North Greenland
  Traverse, 77.2533 N, 49.2167 W, elevation 2598 m; 0.2–119.66 m depth at 1 mm
  sampling, gamma-ray attenuation densitometry). PANGAEA tab format (comment
  block in `/* ... */`, then a tab-separated header + data).
  Citation: Miller, Heinrich; Schwager, Matthias (2000): Density of ice core
  ngt37C95.2 from the North Greenland Traverse. PANGAEA,
  https://doi.org/10.1594/PANGAEA.57798 — License CC-BY-3.0.
  Related: Schwager (2000), Berichte zur Polarforschung 362,
  https://doi.org/10.2312/BzP_0362_2000.
  This is the density profile Culberg & Schroeder (2020) use for their 1-D
  layered dielectric model (their Section IV-C).

- `BER11C95_25_density.tab` — B25 ice-core density profile (Berkner Island
  summit, Antarctica, 79.6142 S, 45.7243 W, elevation 886 m; 1.139–178.213 m
  depth at 3 mm sampling, gamma-ray attenuation densitometry). PANGAEA tab
  format. Copied 2026-07-29 from the same read-only repo
  (`~/Documents/clutter/data/`).
  Citation: Gerland, Sebastian; Wilhelms, Frank (1999): Continuous density log
  of icecore BER11C95_25. PANGAEA, https://doi.org/10.1594/PANGAEA.227732 —
  License CC-BY-3.0. Related: Gerland et al. (1999), Annals of Glaciology 29,
  https://doi.org/10.3189/172756499781821427.
  USE NOTE: serves as a REPRESENTATIVE Antarctic firn proxy for frames that do
  not pass a cored site (e.g. the 2012_Antarctica_DC8 altitude-comparison
  frame, which is NOT at Berkner Island) — a plausible-firn stand-in, never a
  site-specific truth.

- `fig09a_digitized.csv` / `fig09b_digitized.csv` — digitized (WebPlotDigitizer,
  by the source repo's author) from Figure 9(a)/(b) of Culberg & Schroeder
  (2020): the ORANGE curves, i.e. the paper's *simulated* normalized reflection
  coefficient vs depth from their 1-D transfer-matrix layered dielectric model,
  for (a) the Accumulation Radar (AR, ~750 MHz) and (b) MCoRDS3 (~195 MHz).
  These simulated curves track the paper's empirical (blue) depth–power
  profiles closely over 0–100 m (that agreement is the paper's Fig. 9 result),
  so they serve as the context/shape reference here. Used for qualitative
  shape comparison only, never as a numerical gate.

- `fig11a_rms_height.csv` / `fig11b_correlation_length.csv` — copied 2026-07-28
  from the same read-only repo (`~/Documents/clutter/data/`), digitized there
  with WebPlotDigitizer from Figure 11(a)/(b) of Culberg & Schroeder (2020)
  (their Section IV-C-2) and interpolated onto a common 0–90 m depth grid at
  5 m posting. Columns give the depth-resolved *inverted* internal-layer
  roughness for the three inversion sources: `ar` (Accumulation Radar, dotted
  curve), `mcords` (MCoRDS3, dashed) and `joint` (solid). Ranges: RMS height
  sigma 1.5–2.9 cm (ar) / 2.5–5.6 cm (mcords); correlation length l 1.0–2.9 m
  (ar) / 2.4–3.5 m (mcords). Used as the measured sub-facet roughness input
  for the B26 rough-layer comparison runs (`tools/run_b26_comparison.py`,
  `firn_N40_rough_{mcords,ar}`); a physical-plausibility input, never a gate.

## Primary citation

Culberg, R., & Schroeder, D. M. (2020). Firn Clutter Constraints on the Design
and Performance of Orbital Radar Ice Sounders. IEEE Transactions on Geoscience
and Remote Sensing, 58(9), 6344–6361. https://doi.org/10.1109/TGRS.2020.2976666

## Density → permittivity relation

The paper's Eq. (4) (from Kovacs et al., 1993): eps' = (1 + 0.845 * rho)^2 with
rho in g/cm^3. Reimplemented in `tests/test_firn_plateau.py` (mirroring
`src/firn_clutter/density.py` in the source repo); imaginary part neglected per
the paper (conductivity-driven contrast negligible vs density-driven).
