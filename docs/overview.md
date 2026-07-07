## Why build a new simulator?

There are a number of radar sounder simulators already in existance, so why bother to create a new one?

The immediate need is an engineering tool to assess high-altitude radar sounding, particularly using stratospheric platforms to observe and monitor Earth's ice sheets. This simulator would ideally be:
* Focused on Earth-observing radar sounding of ice sheets
* Capable of correctly simulating clutter from surface and subsurface layers
* Intended for instrument engineering and mission design
* Able to bring in data-driven and semi-empirical constraints from various data sources
* Computationally efficient enough to solve over pulse-limited footprints from high altitudes

The last point largely eliminates FDTD methods that are simply too expensive for the frequencies and domain sizes of interest here.

Several facet-based simulators exist. (For example, Holt et al., 2006, Choudhary et al., 2016, Gerekos et al., 2018 and 2023.) All of these were originally built for planetary applications (though some have been validated or used on Earth-based applications). Among these, there is a split between coherent and incoherent methods. Incoherent methods are probably not suitable for exploring multi-layer clutter that will be important to high-alitude sounders on Earth.

## Initial design choices

* Stratton-Chu simulator, using geometric optics to find the transmission and reflection 
* Support for multiple subsurface layers, potentially specified on different resolution grids. Dielectric permittivity specified between each layer.
* Rectangular facets, directly derived from projected DEMs (ArcticDEM, REMA, BedMachine, etc) as in Nouvel et al., 2004
* Support both incoherent and coherent modes (with incoherent designed to match behavior of `simc`)
* Built on JAX and desigend to easily submit runs to cloud compute

## Design stages and testing

Stage 1: Surface only, incoherent power summation
* Benchmark against the `simc` simulator for a small synthetic case

Stage 1.1: xOPR connection plumbing
* Bring in a real radar frame using xOPR, simulate surface clutter, do the same with `simc` and compare
(inspired by https://github.com/Snow4Flow/survey_planning/blob/main/clutter_cresis_frames.ipynb)
* Use ArcticDEM or REMA for surface DEM

Stage 2: Implement Stratton-Chu integral and coherent summation of multiple returns
* Benchmark against theoretical results of Haynes 2018 for rough and smooth surfaces
* Extend xOPR simulation and compare results against incoherent version

Stage 3: Add subsurface layers
* Expand geometric optics to handle arbitrary numbers of internal layers
* Add bed clutter (using BedMachine topography) to xOPR comparison
* Create synthetic firn layers and try to match Culberg and Schroeder 2020 firn power plateau results

Stage 4: Antenna patterns and post-processing
* Add support for varying antenna beampatterns and various levels of post-processing (unfocused SAR, focused SAR)

## Testing and verification

Build the verification before writing the code.

Keep a small set of CI tests that run quickly. Most of the real verificaiton tests will take longer. Those should be part of an integration test suite that prodcues plots and an HTML summary. Each one should still have numerical tests to see if it passes.