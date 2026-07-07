# simc code-reading notes (2026-07-07)

Read from github.com/lpl-tapir/simc @ main (shallow clone; pin a commit before building fixtures). These back the claims in `docs/incoherent_simulation.md`.

## Algorithm as implemented (`src/simc/sim.py`)

- Per trace, `genGrid` builds a track-aligned ECEF grid: `±atNum/±ctNum` steps of `atStep`/`ctStep` along `nav.uv` (along-track) and `nav.ul` (cross-track right). Grid rides at platform altitude; z replaced by DEM sample.
- DEM sampling: transform grid to DEM CRS, pixel coords via inverse geotransform, then **`astype(np.int32)` truncation** — nearest-neighbor-ish, biased toward pixel origin. `demBump=True` clamps out-of-bounds indices to edge pixels; else marks invalid. Optional `demInterp`: coarse grid at `sqrt(2)*xres` spacing + `scipy.interpolate.griddata(method="cubic")`.
- `genFacets`: two triangles per grid cell, ordered so cross product gives consistent normals; tagged left (0) / right (1) of track; cross-track index kept for echo map.
- `calcFacetsFriis`: centroid `m`, `r = |p−m|`, area from cross product /2, `ct = r̂·n̂` (unnormalized dot / (r·2·area)), then:
  - **`power = |((area*ct)**2) / r**4|`** — i.e. (A cosθ)²/r⁴.
  - `twtt = 2r/c`.
- **Paper/code discrepancy — resolved via Haynes et al. 2018**: Choudhary Eqs. (1)–(4) (Friis with G=4πA_eff/λ² *and* σ=4πA_eff²/λ²) algebraically give A_eff⁴/(λ⁴r⁴) — the letter double-counts the facet aperture by using it as both antenna gain and RCS. The code's (A cosθ)²/r⁴ is the standard result: fixed-area coherent flat-plate RCS σ=4πA_eff²/λ² in the plain radar equation (Haynes 2018 Table I, "coherent/fixed" row; λ² dropped by normalization), equivalently Haynes's "antenna approach". Incoherence enters only in the power summation. Corollary (Haynes): per-facet r⁻⁴ + annulus facet count ∝ r ⇒ aggregate leading-edge fall-off r⁻³ over a flat surface — used as an analytic verification anchor.
- Optional `half_wave_dipole` antenna pattern (drone GPR): gain `(cos(π/2·cosψ)/sinψ)²` about `nav.uv` as dipole axis, plus percentile clipping (0.25/99.75) of facet powers.

## Binning (`src/simc/output.py`)

- `cbin = int32((twtt − nav.datum[i]) / dt)` then **`np.mod(cbin, traceSamples)`** — late (and early/negative) arrivals wrap around the trace. `np.bincount(cbin, weights=pwr)` → column.
- `datum` is a per-trace time offset defined by each navfunc (e.g. ARISE: `2h/c − dt·N/2`).
- Side products: left/right cluttergrams, `combined_center/sides` split by angle-of-return theta vs `swathAngle` (needs `centerplane=True`), fret = facet(s) at min adjusted twtt, echo power map at DEM window resolution / {2,8}.

## Gotchas for the benchmark harness

- **`GetNav_simpleTest` is broken**: references undefined `gdf` (`df["datum"] = np.zeros(gdf.shape[0])`). Supply our own navfunc (editable install) or a nav CSV matching another loader.
- Nav loaders are instrument-specific; simplest path is a custom `GetNav_*` returning `x, y, z, datum` in `xyzsys` (ECEF proj string in configs).
- Config is `.ini` (see `config/oib_ak.ini` for an Earth airborne example: dt=10ns, 5000 samples, at/ctDist=6000, at/ctStep=30, demBump on, demInterp off, `binary` output on).
- Outputs: `binary` writes the combined array (h5/np format — check `output.save` when building fixtures); images are contrast-normalized, don't compare against PNGs.
- Facet count per trace = 2·(2·atDist/atStep)·(2·ctDist/ctStep); oib_ak example → 320k facets/trace.

Local clone lives in the session scratchpad (ephemeral); re-clone and pin when building the harness.
