# Incoherent surface clutter simulation

`soundersim` can perform incoherent (power-summing) clutter simulation. This mode uses the same geometric optics and setup as coherent simulation, but with a different kernel.

In this mode, the behavior of `soundersim` should roughly match the existing open-source `simc` simulator (Christoffersen & Holt, U. Arizona; algorithm per Choudhary et al., 2016) for single-layer (i.e. surface-only) cases.

## Initial design

### Scene representation

Facets come directly from the projected DEM grid (Nouvel-style rectangular cells, each split into two triangles for exact area/normal on non-planar cells), built once per scene in a local ENU/scene frame.

Scene building (CPU, float64): DEM window → facet centers `(N,3)`, unit normals `(N,3)`, areas `(N,)` in the local frame; nav → per-trace platform positions + track unit vectors in the same frame.

**Projected area handling.** Facet *sizing* is defined in projected map units (the DEM posting, e.g. 50 m in EPSG:3413), but every vertex is carried through projected → geodetic → ECEF → local ENU before tessellation, so the stored areas and normals are *true ground* quantities — the projection's scale distortion never enters the physics. The consequence is that a "50 m" facet is 50 m on the map, not on the ground: at 75°N the EPSG:3413 point scale is k ≈ 0.98666, so a 50 m projected cell is ≈ 50.68 m of true ground and its area is (1/k)² ≈ 2.7% larger than 50 × 50 m².

> **What this means for simc comparison:** simc steps its per-trace grid in true (ECEF) meters, so at nominally matched 50 m settings its facets are (1/k)² smaller in area than ours. Total incoherent power scales with per-facet area (power ∝ A², facet count ∝ 1/A over fixed ground), so soundersim/simc total power sits at a *constant* ratio of (1/k)² ≈ 1.028 at the test scenes' latitude. Both tools produce relative power, so the parity metric gates on the ratio's *constancy* across traces (coefficient of variation), and the median ratio is recorded and checked against this explanation rather than forced to 1.

### Kernel (JAX)

For each trace: ranges and incidence cosines against all facets in the scene window, per-facet power `(A cosθ)²/r⁴`, two-way time, scatter-add into fast-time bins (`floor((twtt − t0)/dt)`, out-of-window facets **dropped, not wrapped** — the dropped power is accumulated per trace and reported as `dropped_power`). `vmap` over traces; facets are processed in fixed-size blocks (`lax.scan`) to bound memory (a max-range cull can be added later for large scenes). Optional left/right split (sign of facet offset against `u_ct`). Kernel runs in float32 on local-frame coordinates; a float64 NumPy reference implementation lives in the tests.

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

- `simc` installed as a dev dependency pinned to commit `bac8b97` (note: its default branch is `master`). The harness (`compare/simc_harness.py`) drives simc **programmatically** — it imports `simc.prep`/`simc.sim`/`simc.output`, builds the confDict directly with the lowercased keys the code actually reads, and supplies nav as the DataFrame simc expects (ECEF x/y/z + datum) — bypassing the CLI, `.ini` parsing, and the broken `GetNav_simpleTest` loader entirely.
- The scene generator emits a matched input pair: GeoTIFF DEM for simc, the same arrays natively for us. Facet step = DEM posting on both sides (but see the projected-area note above: "matched" spacing still differs by the map scale factor); `demBump=False`, `demInterp=False`; the time window is sized from scene geometry so simc's mod-wrap provably never triggers.
- simc outputs are cached as fixtures in `tests/fixtures/` (npz + json sidecar recording the simc SHA, full confDict, scene params, and binning convention) so CI never runs simc; `tools/make_fixtures.py` regenerates them.

### Test scenes (small: ~10–40 traces, few-km windows; airborne geometry ~500–14000 m altitude)

1. **Flat plane** at constant elevation — analytic leading edge `twtt = 2h/c`; pulse-limited annulus falloff shape.
2. **Cross-track tilted plane** — first return displaced off-nadir by the predicted amount; left/right products separate correctly.
3. **Gaussian hill / ridge offset cross-track** — the canonical clutter hyperbola; arrival time vs. trace matches simc and matches geometric prediction `twtt(x) = 2√(h² + d(x)²)/c`.
4. **Sinusoidal surface** — distributed clutter, exercises binning statistics.
5. **Crater/valley** (negative relief) — first-return-vs-nadir divergence, the Choudhary headline effect.

### Metrics and thresholds (each scene, numeric pass/fail)

**Comparison scale.** At the fixture sampling (dt = 10 ns) a range bin is 1.5 m while a facet is 50 m, and simc's per-trace regrid is a *different tessellation of the same surface* — so raw per-bin power is facet-placement shot noise (observed raw Pearson 0.77–0.92 on every scene, raw per-bin ratios scattering 0.3–3×), not a physics signal. No simulator that doesn't clone simc's exact facet geometry can match at raw bin scale. The shape metrics are therefore evaluated on per-trace profiles power-summed along fast time to the facet scale, `posting/(c·dt/2)` ≈ 33 bins; the raw-bin values are recorded alongside every aggregated metric (`raw_pearson`, `raw_bin_diff`, `raw_rms_db`).

Metrics as implemented in `compare/metrics.py` (all five scenes pass):

- **Peak alignment**: per-trace argmax within ±1 *facet-scale* bin. Observed: exact (0) on all scenes.
- **First-return time**: raw-bin comparison against simc's fret, ±1 bin on flat/hill; **±3 bins on the sloped scenes (tilted, sinusoid, crater)** — verified against a 10× upsampled ground-truth surface, soundersim's first return is within ±1 bin of truth on all scenes while simc's is off by up to 3 bins there, a direct consequence of its int32-truncation DEM sampling displacing heights by up to one pixel on slopes. First-return ground locations agree within 36–93 m (simc's fret facet is a near-tie among adjacent facets on gentle geometry).
- **Profile shape**: per-trace Pearson correlation (linear power, facet scale) ≥ 0.99. Observed ≥ 0.993.
- **Power ratio**: gate on the per-trace total-power ratio's coefficient of variation ≤ 3% (constancy is the physics check); the median ratio is recorded, not forced to 1 — observed 1.028 = (1/k)², explained by the projected-area note above. Observed CV ≤ 1.6 × 10⁻⁴.
- **dB residual**: RMS over facet-scale bins above −40 dB (rel. peak), after removing the constant dB offset, ≤ 1 dB. Observed ≤ 0.64 dB.

Any change to these thresholds or evaluation scales requires written justification in the test (the facet-scale evaluation and the sloped-scene fret tolerance above are the two such justifications to date, documented in `tests/test_parity.py`).

### Analytic checks independent of simc (unit tests, CI-fast)

- Single facet: power and twtt exact against hand-computed `(A cosθ)²/r⁴`, `2r/c` over a grid of geometries.
- Facet geometry: areas/normals of the tessellation sum to analytic values on planar and quadratic surfaces.
- Flat-plate leading edge `twtt = 2h/c` vs. closed form.
- **Geometric fall-off regime (Haynes et al., 2018)**: flat scene simulated at a sweep of altitudes; per-facet nadir power fits `r⁻⁴`, aggregate leading-edge power fits `r⁻³` (rough/pulse-limited row of Haynes Table I, since facet count in the annulus grows `∝ r`). This checks the summation geometry end-to-end, independent of simc. (Stage 2 counterpart: the coherent kernel on the same scene must recover `r⁻²`.)
- Binning: energy conservation (sum of binned power = sum of facet power in window).

### Suite structure

- **CI (fast)**: unit/analytic tests + the flat-scene simc-fixture comparison with full metrics. Runs in ~1 s; `uv run pytest` (integration tests deselected by default).
- **Integration**: all five scenes, full metrics, via `uv run pytest -m integration`. Planned additions: per-scene plots (ours vs. simc vs. difference, per-trace profiles) + HTML summary with pass/fail table, and the Haynes altitude-sweep check from the analytic list above (the rest of that list is implemented).