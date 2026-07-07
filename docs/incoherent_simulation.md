# Incoherent surface clutter simulation

`soundersim` can per perform incoherent (power-summing) clutter simulation. This mode uses the same geometric optics and setup as coherent simulation, but with a different kernel.

In this mode, the behavior of `soundersim` should roughly match the existing open-source `simc` simulator (Christoffersen & Holt, U. Arizona; algorithm per Choudhary et al., 2016) for single-layer (i.e. surface-only) cases.

## Initial design

### Scene representation

Facets come directly from the projected DEM grid (Nouvel-style rectangular cells, each split into two triangles for exact area/normal on non-planar cells), built once per scene in a local ENU/scene frame.

Scene building (CPU, float64): DEM window → facet centers `(N,3)`, unit normals `(N,3)`, areas `(N,)` in the local frame; nav → per-trace platform positions + track unit vectors in the same frame.

### Kernel (JAX)

For each trace: ranges and incidence cosines against all facets in the scene window, per-facet power `(A cosθ)²/r⁴`, two-way time, scatter-add into fast-time bins (`floor((twtt − t0)/dt)`, out-of-window facets **dropped, not wrapped**). `vmap` over traces; a facet cull by max range keeps the per-trace working set bounded. Left/right split kept (sign of facet offset against `u_ct`).

Optional multiplicative hooks: per-facet reflectivity, antenna gain pattern, `R⁻²` vs `R⁻⁴` exponent experiments.

### Outputs

Cluttergram (combined, left, right) as an xarray Dataset with nav coordinates; first-return time/location per trace; power-in-window diagnostics. Written to `outputs/` as a NetCDF file created with `xarray`. The Dataset structure is documented in [output.md](output.md).

# Verification against `simc`

Note: Read from the simc source (`sim.py`, `output.py`), which in one place differs from the 2016 letter. The code is the parity target.

Per trace (platform position `p`, along-track unit vector `u_at`, cross-track unit vector `u_ct`):

1. **Grid**: a track-aligned grid centered on nadir, extents ±`atDist`/±`ctDist`, spacing `atStep`/`ctStep`, constructed in ECEF by offsetting `p` along `u_at`/`u_ct` (so the "grid" rides at platform altitude; only its x,y footprint matters).
2. **DEM sampling**: grid points are transformed to the DEM CRS and sampled **nearest-neighbor via `int32` truncation of pixel coordinates** (optional mode: coarse grid + cubic `griddata` interpolation). Off-DEM points are either clamped (`demBump`) or dropped.
3. **Facets**: the sampled surface is tessellated into **triangles** (2 per grid cell), tagged left/right of track.
4. **Per-facet physics**, with `m` the triangle centroid, `r = |p − m|`, `A` the triangle area, `n̂` the facet normal, `cosθ = r̂·n̂`:

   ```
   P = (A · cosθ)² / r⁴        twtt = 2r / c
   ```

   No Fresnel reflectivity, no wavelength dependence, no antenna pattern (an optional half-wave-dipole gain exists for drone GPR), and no absolute calibration — output is **relative power**.

   **Interpreting the formula (via Haynes et al., 2018, Table I):** `(A·cosθ)²/r⁴` is the fixed-area *coherent flat-plate* RCS, `σ = 4πA_eff²/λ²`, with the λ² dropped by relative normalization — equivalently Haynes's "antenna approach" (facet intercepts `∝ A cosθ/r²`, re-radiates with aperture gain `4πA cosθ/λ²`). Each facet is treated as a specular plate; the *incoherence* enters only in how facet powers are summed. Choudhary Eqs. (1)–(4) algebraically give `A⁴cosθ⁴/(λ⁴r⁴)` because they apply the facet-as-antenna gain *and* the flat-plate σ in the same Friis equation, double-counting the facet aperture; the code is the physically standard form. A consequence Haynes makes explicit: per-facet `r⁻⁴` plus facet count growing `∝ r` in the pulse-limited annulus yields aggregate leading-edge fall-off `∝ r⁻³` over a flat rough surface — a testable prediction (§3).
5. **Binning**: `bin = floor((twtt − datum) / dt) mod traceSamples`, powers summed per bin (`np.bincount`) → one cluttergram column. The `mod` silently **wraps** late arrivals into the top of the trace. Side products: left-only/right-only cluttergrams, first-return (fret) coordinates, nadir delay, echo power map.

### Explicit divergences from simc

| simc behavior | ours | why |
|---|---|---|
| Per-trace track-aligned regrid, nearest-neighbor int-truncation DEM sampling | fixed facet grid from projected DEM | needed for coherent stage; less sampling noise |
| `mod traceSamples` bin wrap | drop out-of-window returns | wrap is aliasing, never physical |
| `int32` truncation in binning near zero | `floor` | correctness for negative pre-datum times |
| Paper Eqs. 1–4 imply `(A cosθ)⁴/λ⁴`; code does `(A cosθ)²` | match the **code** | the code is the standard flat-plate/antenna-approach form (Haynes 2018, Table I); the letter double-counts the facet aperture |

### Harness

- `simc` installed from a pinned commit as a dev dependency. The harness supplies nav via a small custom navfunc / patched loader.
- Scene generator emits a matched input pair: GeoTIFF DEM + nav for simc (`.ini` config), and the same arrays natively for us. Facet spacing set equal to the DEM posting on both sides so simc's regrid degenerates to (nearly) the same facets.
- simc outputs for each case are cached as fixtures (with the generating commit + config recorded) so CI never runs simc; a separate script regenerates fixtures.

### Test scenes (small: ~10–40 traces, few-km windows; airborne geometry ~500–14000 m altitude)

1. **Flat plane** at constant elevation — analytic leading edge `twtt = 2h/c`; pulse-limited annulus falloff shape.
2. **Cross-track tilted plane** — first return displaced off-nadir by the predicted amount; left/right products separate correctly.
3. **Gaussian hill / ridge offset cross-track** — the canonical clutter hyperbola; arrival time vs. trace matches simc and matches geometric prediction `twtt(x) = 2√(h² + d(x)²)/c`.
4. **Sinusoidal surface** — distributed clutter, exercises binning statistics.
5. **Crater/valley** (negative relief) — first-return-vs-nadir divergence, the Choudhary headline effect.

### Metrics and thresholds (each scene, numeric pass/fail)

- **Peak alignment**: per-trace argmax twtt within ±1 range bin of simc.
- **First-return time**: within ±1 bin; first-return ground location within one facet.
- **Profile shape**: Pearson correlation of per-trace binned power (linear domain) ≥ 0.99 on synthetic scenes.
- **Power ratio**: total in-window power ratio ours/simc within a few % (both are relative-power tools; a constant scale factor is acceptable and recorded, bin-to-bin scatter is not).
- **dB residual**: RMS difference over bins above a −40 dB (rel. peak) floor ≤ ~1 dB, after constant-offset removal.

Thresholds start at these values and get tightened empirically once we see actual residuals; any loosening requires a written justification in the test.

### Analytic checks independent of simc (unit tests, CI-fast)

- Single facet: power and twtt exact against hand-computed `(A cosθ)²/r⁴`, `2r/c` over a grid of geometries.
- Facet geometry: areas/normals of the tessellation sum to analytic values on planar and quadratic surfaces.
- Flat-plate leading edge `twtt = 2h/c` vs. closed form.
- **Geometric fall-off regime (Haynes et al., 2018)**: flat scene simulated at a sweep of altitudes; per-facet nadir power fits `r⁻⁴`, aggregate leading-edge power fits `r⁻³` (rough/pulse-limited row of Haynes Table I, since facet count in the annulus grows `∝ r`). This checks the summation geometry end-to-end, independent of simc. (Stage 2 counterpart: the coherent kernel on the same scene must recover `r⁻²`.)
- Binning: energy conservation (sum of binned power = sum of facet power in window).

### Suite structure

- **CI (fast)**: unit/analytic tests + one tiny simc-fixture comparison (flat plane, ~5 traces). Seconds.
- **Integration**: all scenes, full metrics, plots (ours vs. simc vs. difference, per-trace profiles) + HTML summary with pass/fail table. Run via `uv run pytest -m integration-incoherent`.