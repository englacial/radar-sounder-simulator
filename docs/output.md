# Simulation output structure

Every simulation run produces a single `xarray.Dataset`, the same shape in coherent and incoherent modes. Dimension and coordinate names follow xOPR's conventions for OPR/CReSIS frames (`slow_time` × `twtt`) so a simulated cluttergram and a real radar frame can be plotted, resampled, and differenced with the same code.

## Dimensions and coordinates

| Name | Kind | Description |
|---|---|---|
| `slow_time` | dimension coord | One entry per trace. `datetime64` when the nav source has timestamps (e.g. an OPR frame); otherwise integer trace numbers. |
| `twtt` | dimension coord | Fast-time axis: two-way travel time in **seconds since transmit** (not relative to a per-trace datum), uniformly spaced by `dt`, common across all traces. |
| `trace` | coord on `slow_time` | Integer trace index, always present (stable even when `slow_time` is datetime). |
| `lat`, `lon`, `elevation` | coords on `slow_time` | Platform position (WGS84 / meters above ellipsoid). |
| `x`, `y`, `z` | coords on `slow_time` | Platform position in the local scene frame (meters) — the frame the kernels ran in, useful for debugging geometry. |

Returns whose two-way time falls outside the `twtt` window are **dropped** (never wrapped, unlike simc); the dropped energy is reported per trace (see diagnostics).

## Data variables

### The radargram

| Variable | Dims | Dtype | Present |
|---|---|---|---|
| `power` | `(slow_time, twtt, [side, ...])` | float32 | always |
| `field` | `(slow_time, twtt, [side, ...])` | complex64 | coherent mode only |

`power` is linear **relative** power (unitless; convert with `10*log10`). In incoherent mode it is the direct power sum over facets. In coherent mode it is `|field|²`, precomputed so that downstream tools (plots, comparison metrics) never need to care which mode produced the file.

`field` is the complex summed field (relative amplitude, consistent with `power`'s scale). It is what stage-2+ processing (focusing, interferometry, speckle statistics) consumes.

### Optional split dimensions

Products that simc exposes as separate arrays (left/right) are optional *dimensions* here, present only when requested in the config:

| Dim | Values | Introduced |
|---|---|---|
| `side` | `left`, `right` | stage 1 (cross-track ambiguity analysis) |
| `layer` | `surface`, `bed`, ... | stage 3 (per-interface contributions) |
| `channel` | instrument-defined | stage 4 (multi-channel / interferometric) |

**Combination rule — this is the one place the modes differ.** Incoherent power is additive: `ds.power.sum('side')` is the combined cluttergram. Coherent power is not: combine fields first, `abs(ds.field.sum('side'))**2`. A helper (`soundersim.combine(ds, dim)`) applies the correct rule for the file's mode so users don't have to remember.

### Per-trace products (dims: `slow_time`)

| Variable | Description |
|---|---|
| `nadir_twtt` | Two-way time to the surface directly below the platform. |
| `first_return_twtt` | Earliest in-window arrival (simc's "fret"). |
| `first_return_lat`, `first_return_lon` | Ground location of that first return. |
| `dropped_power` | Total power of evaluated facet contributions that fell outside the `twtt` window (validity check on window choice). Facets that provably cannot reach the window (horizontal distance ≥ `c·t_end/2`) are skipped by the per-trace windowing and not counted. |

## Metadata (attrs)

Global attrs carry full provenance — enough to rerun the simulation:

- `mode`: `"incoherent"` or `"coherent"`
- `config`: the complete pydantic config as a JSON string (waveform, facet spacing, window, physics flags)
- `soundersim_version`, `git_commit`, `created`
- `dem_source`, `dem_crs`, `scene_frame`: DEM provenance and the local-frame definition (origin lat/lon, orientation)
- `frequency`, `wavelength`: coherent mode only

Each variable carries CF-style `units` and `long_name` attrs (`twtt`: `"s"`; `power`: `"1"` with `comment: "relative linear power"`).

## On disk

Datasets are written to `outputs/*.nc` with `soundersim.save(ds, path)`:

- Incoherent files are plain NetCDF4, readable anywhere.
- Coherent files contain complex data, which classic NetCDF4 doesn't support; they are written via `h5netcdf` with `invalid_netcdf=True` (a valid HDF5 file that xarray reads back transparently). For strict-NetCDF interchange, `save(..., strict=True)` splits `field` into `field_real`/`field_imag`.

## Mapping to xOPR / CReSIS frames

For side-by-side comparison with a frame loaded by xOPR:

| soundersim | xOPR frame |
|---|---|
| `power` (dB via `10*log10`) | `Data` |
| `slow_time`, `twtt` | `slow_time`, `twtt` (identical) |
| `lat`, `lon`, `elevation` | `Latitude`, `Longitude`, `Elevation` |
| `nadir_twtt` / `first_return_twtt` | `Surface` (measured, so it corresponds to whichever the tracker locked onto) |

Because the dimension names and `twtt` definition match, `sim.power.interp(twtt=frame.twtt, slow_time=frame.slow_time)` aligns the two directly.

## Example

```text
<xarray.Dataset>
Dimensions:            (slow_time: 256, twtt: 4000, side: 2)
Coordinates:
  * slow_time          (slow_time) datetime64[ns] 2018-11-03T14:02:11 ...
  * twtt               (twtt) float64 9.2e-05 9.21e-05 ... 1.32e-04
  * side               (side) <U5 'left' 'right'
    trace              (slow_time) int64 0 1 2 ... 255
    lat                (slow_time) float64 ...
    lon                (slow_time) float64 ...
    elevation          (slow_time) float64 ...
Data variables:
    power              (slow_time, twtt, side) float32 ...
    field              (slow_time, twtt, side) complex64 ...   # coherent only
    nadir_twtt         (slow_time) float64 ...
    first_return_twtt  (slow_time) float64 ...
    dropped_power      (slow_time) float32 ...
Attributes:
    mode:                coherent
    config:              {"scene": {...}, "radar": {...}, ...}
    soundersim_version:  0.1.0
    ...
```

## Not in this Dataset (future companion products)

simc's georeferenced **echo power map** (power per ground cell) is a map-space product with different dimensions; when implemented it will be a separate Dataset on ground coordinates, not extra variables here. Waveform-convolved / focused products (stage 4) will reuse this same structure with processing steps recorded in `attrs`.
