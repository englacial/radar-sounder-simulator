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

- `fig09a_digitized.csv` / `fig09b_digitized.csv` — digitized (WebPlotDigitizer,
  by the source repo's author) from Figure 9(a)/(b) of Culberg & Schroeder
  (2020): the ORANGE curves, i.e. the paper's *simulated* normalized reflection
  coefficient vs depth from their 1-D transfer-matrix layered dielectric model,
  for (a) the Accumulation Radar (AR, ~750 MHz) and (b) MCoRDS3 (~195 MHz).
  These simulated curves track the paper's empirical (blue) depth–power
  profiles closely over 0–100 m (that agreement is the paper's Fig. 9 result),
  so they serve as the context/shape reference here. Used for qualitative
  shape comparison only, never as a numerical gate.

## Primary citation

Culberg, R., & Schroeder, D. M. (2020). Firn Clutter Constraints on the Design
and Performance of Orbital Radar Ice Sounders. IEEE Transactions on Geoscience
and Remote Sensing, 58(9), 6344–6361. https://doi.org/10.1109/TGRS.2020.2976666

## Density → permittivity relation

The paper's Eq. (4) (from Kovacs et al., 1993): eps' = (1 + 0.845 * rho)^2 with
rho in g/cm^3. Reimplemented in `tests/test_firn_plateau.py` (mirroring
`src/firn_clutter/density.py` in the source repo); imaginary part neglected per
the paper (conductivity-driven contrast negligible vs density-driven).
