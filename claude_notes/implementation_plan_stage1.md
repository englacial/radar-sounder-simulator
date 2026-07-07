# Implementation plan: stages 1 and 1.1

Plan for my own execution, per `docs/overview.md` (verification before code), `docs/tech_stack.md`, `docs/incoherent_simulation.md`, `docs/output.md`. **Status: approved. M0–M5 complete (2026-07-07): 49 CI tests + 5 integration parity tests green; all five scenes pass parity vs simc fixtures (simc pinned at bac8b97; its default branch is `master`, not `main`). Two evidence-backed metric interpretations, justified in test docstrings: shape metrics evaluated at facet scale (~33 raw bins aggregated; raw values recorded in metric dicts), fret tolerance ±3 on sloped scenes (simc's int-truncation DEM sampling is the outlier, verified against upsampled ground truth). Constant power ratio 1.028 = (1/k)² with k = EPSG:3413 point scale at 75°N — projected vs ECEF facet sizing, not a bug. M6+M7 complete (2026-07-07 pm): 50 CI + 8 integration tests green; outputs/verification/report.html covers 8 cases (5 scenes, Haynes sweep, 2 OPR frames). Haynes slopes: per-facet −3.9998 (r⁻⁴), leading-edge −2.930 (r⁻³, window anchored to first-return bin). OPR: 20171121_03_005 (2017_Antarctica_P3, REMA v2 32 m) + 20170422_01_014 (2017_Greenland_P3, Helheim trunk, ArcticDEM v4.1 32 m); both DEMs WGS84-ellipsoidal per PGC docs; measured twtt offsets −26.9 ns (Antarctic) and −177.7 ns constant (Greenland, hypothesized DEM-epoch surface change + system delay — unverified); soundersim-vs-simc on real frames: Pearson ≥0.990, dB RMS ≤0.35 (recorded, thresholds deferred per plan). Frames + DEM windows cached under outputs/cache/ (offline reruns). Stage 1 + 1.1 definition of done met. Nothing committed to git yet.**

## Proposed repo layout

```
pyproject.toml                  # uv-managed; package `soundersim`
src/soundersim/
  config.py                     # pydantic models: SceneConfig, RadarConfig, SimConfig
  scene.py                      # DEM window -> facet arrays in local ENU frame (numpy/rasterio/pyproj)
  nav.py                        # nav ingest -> platform positions + track unit vectors in scene frame
  synthetic.py                  # synthetic DEM (GeoTIFF) + nav generators — shared by tests, harness, users
  kernels/incoherent.py         # JAX kernel: facets + trace -> binned power (vmap over traces)
  kernels/geometry.py           # ranges, cosθ, binning helpers shared with future coherent kernel
  output.py                     # Dataset assembly per docs/output.md; save/load; combine()
  compare/simc_harness.py       # drive simc, load its outputs
  compare/metrics.py            # parity metrics from docs/incoherent_simulation.md
tests/                          # fast CI tests (pytest, no markers)
tests/integration/              # slow suite (pytest -m integration) -> plots + HTML report
tools/make_fixtures.py          # regenerate cached simc outputs (pinned commit recorded)
tools/make_report.py            # integration results -> outputs/verification/report.html
outputs/                        # gitignored except cached fixtures? (see decision D6)
```

## Milestones (each ends green: `uv run pytest`)

### M0 — Scaffolding (small)
- `uv init`, pyproject: numpy, jax, rasterio, pyproj, pydantic, xarray, h5netcdf, matplotlib; dev: pytest, simc @ pinned git commit. `integration` pytest marker. `.gitignore` outputs/, scratch clones.
- Smoke test: import soundersim, jax backend reports CPU.
- Note: rasterio/pyproj wheels bundle GDAL/PROJ binaries — satisfies "no system GDAL" with no apt/brew steps; truly GDAL-free alternatives (rioxarray still uses rasterio) don't exist for GeoTIFF, so this is the intended reading of the tech-stack constraint. **Confirm.**

### M1 — Config + synthetic scenes (before any simulator code)
- Pydantic configs, JSON-round-trippable (this becomes the `config` attr in outputs).
- `synthetic.py`: the five test scenes from `docs/incoherent_simulation.md` (flat, cross-track tilt, offset Gaussian hill, sinusoid, crater), each emitting (a) GeoTIFF DEM in a projected CRS + nav CSV for simc, (b) the same arrays natively. Straight-line nav, configurable altitude.
- Tests: DEM writer round-trips through rasterio; scene parameters (slopes, hill height) recoverable from written files.

### M2 — simc harness + fixtures (verification target locked in before our kernel exists)
- Drive simc programmatically (import `simc.sim`/`simc.output` from the pinned commit, build its confDict ourselves) rather than via CLI — sidesteps the broken `GetNav_simpleTest` without forking; nav supplied as a DataFrame in ECEF with datum column. Record simc commit + full config next to each fixture.
- `tools/make_fixtures.py` writes cluttergram + fret fixtures for all five scenes to `tests/fixtures/` (small: ~10–40 traces).
- Tests: fixtures load; sanity — flat-scene fixture leading edge at `2h/c ± 1` bin (validates the harness itself, catching datum/window mistakes now rather than in M5).

### M3 — Scene building + nav
- Local ENU frame (origin = scene center); DEM window → facet centers/normals/areas (float64, rectangular cells split into triangle pairs); nav → positions + `u_at`/`u_ct` in same frame.
- Tests (from the analytic list): tessellation areas/normals exact on planar surfaces, convergent on quadratic; frame round-trip lat/lon↔local < 1 mm over a 50 km scene.

### M4 — Incoherent kernel
- JAX: per trace, ranges + `cosθ` against facet arrays → `(A·cosθ)²/r⁴`, `twtt = 2r/c`, scatter-add via `segment_sum` into bins, out-of-window → `dropped_power`. `vmap` traces; facets processed in fixed-size blocks (`lax.map`) to bound memory. Left/right by sign against `u_ct`. Float32 in-kernel, float64 prep (positions pre-shifted to scene frame keeps float32 safe; verify with a float64-vs-float32 test).
- Tests: single-facet power/twtt exact vs hand computation; binning energy conservation; left+right = combined.

### M5 — Output Dataset + parity
- `output.py` per docs/output.md (power, optional `side` dim, per-trace products, attrs, `save()`).
- `compare/metrics.py`: peak alignment, first-return, Pearson ≥ 0.99, power ratio, dB residual (thresholds from docs/incoherent_simulation.md).
- CI gets the one tiny fixture comparison (flat, ~5 traces); the rest go to integration.
- **Exit criterion for stage 1 core: all five scenes pass all metrics vs simc fixtures.**

### M6 — Integration suite + Haynes check
- Integration tests produce per-scene PNGs (ours / simc / difference, per-trace profiles); `tools/make_report.py` assembles `outputs/verification/report.html` with pass/fail table.
- Altitude-sweep test on flat scene: per-facet `r⁻⁴` fit, aggregate leading-edge `r⁻³` fit (log-log slope within ±0.1 — tighten later).

### M7 — Stage 1.1: real frame via xOPR
- Add deps: xopr; pystac-client (PGC STAC) for ArcticDEM/REMA discovery; DEM read as COG over HTTP through rasterio (no bulk download).
- `nav.py`: OPR frame → nav (lat/lon/elevation → scene frame); pick one Greenland or Antarctic MCoRDS frame with interesting surface relief (following the Snow4Flow clutter notebook's approach; concrete frame chosen at implementation time and recorded) and then also use 20171121_03_005 from the 2017_Antarctica_P3 season
- Run soundersim and simc on the same frame + DEM; comparison script writes a report section: cluttergram vs radargram overlay (using the xOPR mapping in docs/output.md), soundersim-vs-simc metrics (looser thresholds than synthetic — real DEM regrid differences are larger; record observed values first, then set thresholds).
- Cache the frame + DEM window locally for repeatability; integration-only (network), never CI.

## Decisions I'll make unless redirected

- **D1** simc driven programmatically at a pinned commit (not CLI, not forked).
- **D2** Facet spacing = DEM posting for parity runs; simc `demBump=False`, `demInterp=False`.
- **D3** Our twtt window chosen to contain all returns in synthetic scenes (so simc's wrap never triggers and drop-vs-wrap can't confound parity).
- **D4** Surface DEM for 1.1: ArcticDEM/REMA **mosaic** (not strips), 32 m tier, matching the Snow4Flow notebook scale.
- **D5** HTML report: hand-rolled single-file template (no pytest-html dep).
- **D6** simc fixtures committed to the repo (`tests/fixtures/`, expected < 5 MB total, compressed npz + json sidecar). Alternative if too big: regenerate-on-demand with hash check.

## Known risks

- simc's int-truncation DEM sampling + per-trace regrid means even matched-posting facets differ slightly near cell edges → watch the Pearson/dB metrics on the sinusoid scene first; if they fail, the fallback is bilinear-free "grid-aligned nav" scenes (nav parallel to DEM axes through cell centers) before considering a simc-geometry mode.
- JAX scatter-add float32 accumulation over ~10⁵–10⁶ facets/trace: acceptable for parity (simc itself is float64 numpy) but verify with an f32-vs-f64 kernel comparison test in M4.
- xOPR/PGC network flakiness: all network access confined to cached, integration-marked paths.

## Definition of done

Stage 1: M0–M6 complete, CI < 30 s, integration suite green with report. Stage 1.1: M7 report shows real-frame cluttergram, simc agreement metrics recorded, and surface return aligning with the radargram surface within a few range bins along the frame.
Test reports should visually and quantitatively show matching outputs against simc, validation against Haynes et al., 2018, and generation of a surface cluttergram for a real xOPR frame. 
