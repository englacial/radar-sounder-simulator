# Tech stack

Package name `soundersim`

Python ≥3.11, managed with `uv` (`pyproject.toml`, run everything via `uv run`).

## Core dependencies

* **JAX** for all core numerics to support GPU acceleration while keeping CPU compatibility
* **NumPy** of course
* **rasterio** and **pyproj** for DEM ingest, projections, etc
* **pydantic** for configuration and serializability for cloud submission
* **xarray** for output files
* **matplotlib** for plotting
* **pytest** for CI and integration tests
* **xOPR** for comparison radar data

Benchmarking for incoherent simulation against `simc` library.

## Architecture

Three layers with one-way dependencies:

1. **Scene building** (CPU, NumPy/rasterio/pyproj): DEM → facet arrays (centers, normals, areas) in a local frame; nav → platform positions in the same frame.
2. **Kernels** (JAX): facet arrays + trace positions → binned power. Incoherent and coherent modes are alternative kernels over the same scene representation.
3. **Instrument/processing** (NumPy/xarray): time-window handling, waveform convolution (later), output products, plotting.
