# Agent handoff — clutter studies state (2026-08-17)

> **SUPERSEDED by `agent_handoff_2026-08-24.md`.** This note predates the
> config-driven refactor, the (gamma_surface, A) calibration era, the bed
> scattering adoption, and the grazing-angle fix; its K/D anchoring
> machinery and "best config" values no longer exist. Kept for the
> pre-2026-08-18 chronology and the notes index.

Read this first, then the per-topic notes it links. Everything below is
committed through `61540c0`; working tree clean; 352 unit tests green
(`uv run pytest tests -q --ignore=tests/integration`).

## The two active study lines

### Antarctic: 20161105_05 (Amundsen sector) — MATURE
Three real passes over one 148.45 km line (449 m / 9,150 m / 10,684 m AGL,
identical 190/50 MHz systems) + synthetic passes (14/30/300/500 km flavors
exist across campaigns). Grounded s=0-69.7, floating beyond (hybrid bed).
- **Best config** (`att20_klevel` / `full_line`): DEMOGORGN bed seed 0
  (snapshot WG801625MG778C4DS6Y0), RSSNR gamma (antarctica store, snapshot
  3YH47013745B2T5ZZR50) level-anchored **K=+7.92**, **A=20**, matched CSARP
  processing (14.85 m posting, backprojection at alias-limited aperture,
  3 looks), picked-bed hybrid for the floating zone (NN low-pass picks,
  4 km GL blend).
- **Established results**: altitude clutter effect reproduced and
  decomposed (mid-column is SURFACE returns at all altitudes); bed-source
  ablation (DEMOGORGN > picked bed > BedMachine; picked bed has a
  cross-track-ridge artifact); T5 specular/diffuse split (f_s~0.5, s0=3°,
  mean-normalized tilt gate) fits all three altitudes' tail shapes at once;
  attenuation family analysis (level evidence 17-21 vs K=K_phys closure 31;
  A=20+level anchoring adopted — only member physical+level+shape);
  design ladder: bed visible +7-10 dB at 30 km, grounded bed buried at
  300 km (-12.6)/500 km (-26) while the floating specular base survives to
  300 km (+3.4); floating-zone fixed-K residual collapses with altitude
  (+12.6 low → +1.9 high) — effective-gamma is altitude-regime-bound on
  specular targets.
- **Open**: (1) the ~10-13 dB reference-chain question (K−K_phys=+18 at
  A=20) — surface-reference audit never done; (2) T5-composed-on-A20-level
  capstone run never executed; (3) cross-track decorrelation of the
  picked-bed residual (floating tail +11 dB hot); (4) ice-rise patches
  inside the "floating" zone (BedMachine mask, see findings); (5) DEMOGORGN
  has NO LICENSE — contact Gator Glaciology/Englacial before publication.

### Greenland: 20140421_01 s=11-40 km (central-west interior) — ACTIVE
Pair: low 20140421_01_069 (465 m, traces 736-2675) / high 20170424_01_067
(2,483 m, traces 36-1976; single frames, no reversal) + syn14km. Best
instrument parity of any pair (identical fc/B/waveforms/PRF).
- **Current runs** (`full_pbed_proc_att14_rssnr`): BMv5+picked-bed residual,
  RSSNR gamma (greenland store, snapshot GEAMAHQ7BRVPG9SQPK20)
  level-anchored **K=−5.83** at **A=14**, matched processing.
- **Established**: geometric surface+bed CANNOT explain the measured
  column (englacial-scattering dominated; measured altitude trend +1.1 dB
  vs sim +16.8 — the flat trend is the diagnostic); RSSNR acceptance low
  pass 0.19→0.60 (data ceiling 0.897); chain nearly self-consistent
  (K−K_phys=+4.5); the measured high-pass bed IS real (+6.2 dB peak over
  null; +11.5 on CSARP_mvdr) — earlier "not a bed measurement" corrected
  to PRODUCT-limited; slope-regression A~7 REJECTED (thawed-bed Γ–H
  confounder, see macgregor note); **MacGregor arbitration: A = 16 ± 2
  full-column** (three routes converge 14.3/15.5/14.9 in the sampled band;
  Robin column correction → 16-19).
- **NEXT ACTION (user-pending)**: adopt A=16, re-derive K (level anchor,
  low pass, contamination-aware rule), rerun 3 passes (~66 min), re-score
  acceptance. Consider switching bed-referenced measured work to
  CSARP_mvdr (mvdr ceiling 0.47-0.53 vs standard 0.078).
- **Open**: englacial/firn term is the missing physics (effective-contrast
  machinery in soundersim.firn is ready; needs a density source decision —
  B26 is far away; densification model?); stale truncated decomposition
  figures in the older run dirs (A=15, A=14 const-γ) — regenerate from
  proc caches if ever shown.

## Foundation (validated, stable)
- **B26 firn methodology** (docs/ + b26 findings): effective segment
  contrasts solved the 17 dB plateau gap (+11.3 dB exactly as 1-D
  predicted); uniform N=20 standard; ~5 dB coherent-realization deficit
  still open; attenuation 15 dB/km + firn attenuation defaults.
- **Kernels**: Gerekos sub-facet roughness, per-facet gamma, per-facet
  diffuse channel (T5), joint refraction (O(1) compile). All
  regression-gated bit-identical on legacy paths.
- **run_basal_clutter.py**: line registry (`--line`; Antarctic entry is a
  no-op literal), bed sources (BedMachine/picked/DEMOGORGN/hybrid), RSSNR
  mapping (--anchor median|level, --level-deficit-db, contamination-aware
  D), --att, matched processing (--processing standard, --proc-cache =
  bit-exact focused-stack cache; figure iteration in seconds), per-pass
  figures (--per-pass-figs, --plot-s-max, --fig-width-scale,
  --trace-decomp-s, --passes, --only-style guards, --companion-name).
  ALL figure windows/scales are line globals (PROFILE_REL_US etc.) —
  the Antarctic-hardcoding bug class is dead, keep it that way.
- **Terminology**: "surface returns"/"bed returns" (user-mandated).
- **Metrics practice**: mean-power over windows+traces (never median-of-dB
  at one trace); score bed models on the bed-layer decomposition component
  for any pass whose bed window is surface-dominated; tail stats need the
  brightest-5%-share check (single-arc domination).

## Approximate runtimes (local 9900X)
- Antarctic 50 km 5-pass set: ~39 min sim + ~6 min processing.
- Antarctic full-line (148 km) high pass: ~13 min; low ~28; all five ~65.
- Greenland pair full segment (29 km, 3 passes): ~66 min + processing.
- Figure-only iteration: seconds (proc cache) to ~10 min (full replay).
- GPU: not worth it (V100 1.6×, serialization-bound); cloud value is
  CPU fan-out across independent runs (Batch spec in tools/gcp/).

## Key notes index
- claude_notes/basal_clutter_pilot_findings.md — the Antarctic saga, all
  phases appended chronologically.
- claude_notes/greenland_pair_findings.md — Greenland incl. part 4
  (bed-visibility bisection + A withdrawal).
- claude_notes/macgregor_attenuation_scout.md — attenuation arbitration.
- claude_notes/{basal_clutter,greenland_altitude,cross_season_line,
  demogorgn,required_snr_dataset}_scout.md — data/source scouts.
- claude_notes/b26_gap_hypotheses.md + b26_comparison_findings.md — firn.
- docs/clutter_studies.md — user-facing overview (written 2026-08-17).

## Global open items (beyond per-line ones above)
1. Antarctic surface-reference audit (the 10-13 dB K−K_phys residual).
2. T5-on-A20-level capstone (one ~35 min run).
3. Greenland A=16 adoption + rerun (user said investigate; adoption pending).
4. Englacial term for the Greenland line.
5. RSSNR stores: re-point to rebuilt `main` when the user's reprocess
   finishes; interpolated product wiring when user provides it.
6. DEMOGORGN licensing contact before any publication.
7. outputs/verification/report.html is >40 MB — consider splitting.
8. B26 realization deficit (~5 dB) — strip-width test is the cheap probe.
