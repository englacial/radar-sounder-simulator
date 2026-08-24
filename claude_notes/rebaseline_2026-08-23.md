# Re-baseline at adopted scattering physics + facet <= 0.7 (2026-08-22/23)

All 4 pilots + all 4 full experiments re-run from cold (facet-scale change
forked every chunk key). 8/8 rc=0; runtimes ~2-2.6x the old lattice
(pilots 15-56 min; fulls 264-700 min; total ~31 h wall). 'Before' figures
and metrics preserved in outputs/_before_facet07_20260822/.

| line / run | residuals before -> after | notes |
|---|---|---|
| getz pilot | -14.0/-11.4/-12.8 -> -9.5/-9.7/-9.6 | altitude-UNIFORM now; gamma_req 4.4 -> 1.1 |
| getz full | -7.4/-14.6/-15.1 -> -3.5/-10.6/-9.7 | floating shelf-base -4.6 -> -0.8 dB (specular regime fixed) |
| david pilot | ~unchanged | MKB60 puzzle intact; only Basler qualifies (req 9.7) |
| david full | 2022: -4.5 -> -17.3 | NOT a regression: 2022's old match was spurious coarse-facet SURFACE clutter (surf-arm dropped 17.6 dB, 2023's unchanged); honest gap replaces accidental agreement |
| geikie pilot/full | ~unchanged | englacial gap untouched (midcol -42 vs -77); brightness r 0.54 -> 0.70 |
| westcoast pilot | 2016: -13.4 -> -10.8; others ~0 | pass disagreement 13.7 -> 9.9 dB |
| westcoast full | 2016: -19.2 -> -13.0; others ~ -2 to -3.5 | brightness r 0.83 holds; tails still shallow (ridge arcs -> h2 branch) |

Reading: the adopted physics is surgical -- big gains exactly where the
bed dominates (getz: uniform residuals, specular floating zone closes),
clean nulls where the physics gap is elsewhere (geikie englacial, david
MKB60), no regressions. Facet refinement additionally removed a
geometry-specific discretization artifact (david 2022 surface clutter).
Standing decisions: gamma pin vs +1.1 solve evidence (getz); h2/t2 branch
merges; the geikie englacial term; david MKB60 measured energy source.
