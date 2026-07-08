# M19 firn power plateau — session findings (2026-07-07)

Case: `tests/test_firn_plateau.py` -> `outputs/verification/firn_plateau/`,
group "Firn clutter (Culberg & Schroeder 2020)". Fixtures + provenance:
`tests/fixtures/firn/README.md`. Dev scripts: `m19_compile_probe.py`,
`m19_dev_run.py` (profile arrays in `m19_dev_profile.npz`).

Key measured findings (details in the test docstring):

1. **Compile scaling** (kernels/multilayer.py warning confirmed): simulate()
   multilayer compile time ~0.18*N^2 s (N interfaces; 10 -> 16 s, 20 -> 69 s,
   30 -> 167 s cold). Chosen adaptation: 20 coherent layers + 10 incoherent
   nodes (~2:15 test). More layers needs a scanned solve or persistent
   compile cache in the kernel.
2. **Uniform decimation spacing is load-bearing**: depth-graded nodes (fine
   near surface, coarse deep) alias the smooth compaction trend into large
   deep-interface contrasts and ERASE the rolloff (measured 0.5 dB). Uniform
   5 m slab-mean nodes give the Fig. 9 structure (plateau span 10.5 dB over
   upper 40 m, rolloff 15.6 dB to 70-100 m).
3. Per-layer coherent power tracks the decimated-gamma^2 image-method closed
   form to 1.6 dB in band means; individual weak deep layers (gamma < -50 dB)
   carry up to ~6 dB trace-dependent aperture/facet-quantization residual
   (600 m scene, 4 m facets) — gates use band means.
4. Incoherent kernel is reflectivity-blind by convention: 1.6 dB band
   rolloff, no structure — the plateau is coherent/specular physics; this
   contrast is gated (coh − inc rolloff ≥ 8 dB, measured 14.0 dB).
5. mm-scale within-range-bin thin-film interference (the paper's transfer
   matrix) is out of scope at O(tens) layers; a 1-D hybrid remains the tier-2
   route for stratigraphy-scale realism. Not needed for the qualitative
   plateau, which appears at N=20.
