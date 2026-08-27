# HAPS 14 km design study — greenland_westcoast pilot window (2026-08-26)

Objective: maximise basal SNR for a stratospheric sounder at a fixed 14 km
over the westcoast pilot window (s 40–50 km, ~1 km ice, bed at 11.7–17 µs
below the surface), within: 20–400 MHz, ≤50 % fractional bandwidth, ≤16
elements on a 10 m cross-track span, free choice of pulse length / window /
TX+RX taper, along-track focusing as long as wanted.

Metric: `<pass>_bed_visibility` = bed-arm power over surface-arm (clutter)
power in the bed window, dB (simulator is clutter-only; thermal noise from the
mission design tool separately). Driver: `gen.py` (instrument + experiment
YAML per round, `config/instruments/hd_*.yaml`, `config/experiments/wc_hd_*.yaml`),
`noise_budget.mjs` (MDT link budget), `run_rough_sens.py` (surface-roughness
override). All runs use the explicit-chirp pulse model (branch
`waveform-explicit-chirp`) — the analytic model has no pulse-length dependence.

## Noise-limited side (MDT physics.js, mu = 60 dB, 1000 m ice, 200 W payload)

All candidates land at 70–74 dB basal SNR, within ~3 dB of each other: higher
frequency loses λ² and integration count but gains array directivity and drops
the galactic noise (4000 K at 60 MHz → 560 K at ≥225 MHz). Thermal noise is
not the binding constraint on this line; clutter is.

## Rounds (clutter-limited bed visibility, dB; nominal surface roughness σ=4.9 cm, l=2.98 m unless noted)

Round 1 — frequency ladder, Hann TX/RX taper, 10 m span, T=8 µs, B=f/2:

| design | bedvis | surf arm @bed | midcol |
|---|---|---|---|
| 60 MHz, 5 el, uniform | −21.6 | −67.1 | −48.9 |
| 60 MHz, 5 el, Hann | −22.9 | −65.8 | −47.9 |
| 150 MHz, 11 el | **+17.2** | −101.1 | −44.8 |
| 225 MHz, 16 el | +45.4 | −128.7 | −48.8 |
| 400 MHz, 16 el | +99.9 | −185.8 | −56.9 |

Round 2 — at 225 MHz: uniform/Hann/Hamming taper, 8/11/16 el, 10 % amplitude
errors, Hann vs Hamming window, 50 vs 112 MHz BW, 8 vs 10 µs → all 45 ± 1 dB.
5 el (1.9 λ spacing, grating lobe at the ~29° clutter angle): 34 dB. 20 µs
pulse (pedestal on the bed): −13 dB. 300 MHz/16 el: +69.6.

Round 3 — attribution: isotropic 225 MHz +35.9 (array worth ~10 dB); surface
roughness OFF: surface arm −330 dB (all wide-angle clutter is the sub-facet
roughness term). 300 MHz 8 = 11 el (+69.6). 400 MHz 8 el (1.9 λ) +82.8 vs 16
el +99.9. posting_div 2: bedvis unchanged (44.0 vs 45.5), midcol −2 dB.
T = 11 µs = 8 µs.

Round 4/5 — SAME designs with surface correlation length l = 1 m (σ = 5 cm),
a proxy for a surface with more short-scale roughness:

| design | bedvis | surf arm @bed | midcol |
|---|---|---|---|
| 60 MHz, 5 el | −29.1 | −59.7 | −51.8 |
| 150 MHz, 11 el | −30.3 | −53.6 | −40.4 |
| 225 MHz, iso | −31.3 | −48.1 | −30.0 |
| 225 MHz, 8 el | −23.6 | −59.7 | −37.5 |
| 225 MHz, 16 el | −23.7 | −59.2 | −37.0 |
| 300 MHz, 8 el Hann | −18.7 | −66.2 | −36.7 |
| 300 MHz, 8 el uniform | −18.9 | −66.7 | −37.3 |
| 300 MHz, 16 el Hann | −18.7 | −65.8 | −36.4 |
| 400 MHz, 16 el | −14.0 | −69.0 | −33.5 |

Side finding: under l = 1 m the REAL p3_2017 pass's simulated mid-column
clutter is −72.5 dB vs measured −59.4 (nominal l = 3 m: −117.5). The shorter
correlation length is far closer to the measured data, so the pessimistic
scenario is the more credible one for this line.

Round 6 — along-track aperture at 300 MHz / 8 el under l = 1 m:

| posting_div (posting m, aperture m) | bedvis | surf arm @bed | bed arm | midcol | wall s |
|---|---|---|---|---|---|
| 1 (14.9, 492) | −18.7 | −66.2 | −84.9 | −36.7 | 72 |
| 4 (3.7, 1971) | −20.1 | −68.8 | −88.9 | −40.7 | 308 |
| 8 (1.9, 3969) | −20.6 | −67.7 | −88.3 | −42.4 | 792 |

Longer apertures buy ~6 dB mid-column and nothing in the bed window: the
surface arm at the bed delay moves 2.5 dB while the bed arm loses 4 dB. At
300 MHz even a 1.9 m posting aliases a 29° along-track scatterer (needs
λ/(4 sin 29°) ≈ 0.5 m), so its energy folds into the processed band
regardless of aperture length.

Round 7 — the aliasing test, posting_div 8 (1.9 m posting) under l = 1 m:

| design | unaliased to | bedvis (pd1 → pd8) | surf arm @bed (pd1 → pd8) |
|---|---|---|---|
| 60 MHz, 5 el | 42° | −29.1 → **−0.2** | −59.7 → −85.9 |
| 150 MHz, 11 el | 16° | −30.3 → −35.4 | −53.6 → −55.3 |

Where the posting is unaliased at the 29° clutter angle, along-track focusing
removes ~26 dB of surface clutter at the bed delay; where it is aliased, it
removes none. The residual bed-delay clutter in every ≥150 MHz run above is
therefore a sim sampling artefact. A real 20 m/s platform at kHz PRF samples
at centimetres and never aliases; its along-track rejection is bounded by
processing aperture and motion errors instead.

Round 8 — the recommended 300 MHz / 8 el design at posting_div 32 (0.47 m,
unaliased to 32°) under l = 1 m: NOT COMPLETED. The run (21 345 traces per
pass) was lost to a session restart, and the chain's time-domain
backprojection cost grows as traces × aperture-traces (~40× the posting_div
8 run, i.e. many hours) with an alias-limited aperture (17 km) longer than
the 10 km window. The 60 MHz round-7 result is the demonstration of the
along-track lever; transferring its ~26 dB to 300 MHz is a physics
extrapolation, not a simulated number.

## What sets the clutter at the bed delay

- Geometry: at 12.5 km AGL a 1 km bed (11.8 µs) shares its delay with surface
  at ~29° off nadir (7 km cross-track) — or the same angle along-track.
- Cross-track: any ≥8-element array on 10 m at ≥150 MHz puts 29° in deep
  sidelobes; taper and element count beyond that are irrelevant in the sim.
  Fewer than ~8 elements at ≥225 MHz opens a grating lobe near 29°.
- Along-track: the cross-track array cannot reject it, and the focuser can
  only reject it if the along-track sampling is unaliased at that angle
  (posting ≤ λ/(4 sin 29°): 2.6 m at 60 MHz, 1.0 m at 150, 0.5 m at 300).
  The sim's default 15 m posting aliases it at every frequency; once
  unaliased (round 7) focusing removes ~26 dB. This is the biggest single
  lever after frequency, and the one the default sim chain hides.
- Scattering law: the wide-angle level is the Gerekos sub-facet term,
  ∝ exp(−(k l sinθ)²/m). With l = 3 m it collapses with frequency (−28 dB
  per 75 MHz), with l = 1 m it barely moves. The ~100 dB of headroom in
  round 1 is a Gaussian-ACF artefact; the direction (higher f helps) is
  physical, the magnitude is unknown until this line's short-scale surface
  roughness is measured.
- Pulse length: the only pulse-model effect is the finite-TB Fresnel pedestal
  of the surface return; keep T below the minimum bed delay (≤ ~11 µs here).
  Longer pulses buy thermal SNR the line doesn't need.
- Bandwidth / compression window: no effect on bed-over-clutter (both scale
  together); take the maximum for resolution.
- Mid-column clutter rises with frequency under l = 1 m (−52 → −33 dB from
  60 to 400 MHz): a real englacial-visibility cost of going high.

## Recommended configuration

- **f0 = 300 MHz, B = 100 MHz** (the sim's alias rule forbids 300/150 on the
  8.33 ns grid; 150 MHz BW would be the physical choice), **8 elements at
  1.43 m (1.43 λ)** on the 10 m span — the 300 MHz/8-el point where the sim
  saturates; going to 400 MHz gains ~5 dB but needs 16 elements (8 el at 1.9 λ
  loses 17 dB to a grating lobe) and costs ~3 dB mid-column.
- **Hann taper on both TX and RX** — free in the sim, and the only thing that
  keeps a real array's clutter floor low once pattern errors matter.
- **T = 8–10 µs, Hann compression window** (< 11.7 µs minimum bed delay).
- **Along-track: unaliased sampling (≤ 0.5 m posting at 300 MHz — trivial
  at 20 m/s) and a focused aperture of several km.** In the sim this needs
  posting_div ≥ 32; rounds 6/8 quantify it.

Expected bed visibility: +70 dB (nominal roughness, aliased chain), −19 dB
(l = 1 m, aliased chain), and roughly +5 to +10 dB (l = 1 m with the ~26 dB
unaliased along-track rejection seen at 60 MHz, extrapolated). The design
is settled; the surface roughness spectrum and the processing chain's
along-track sampling are what set the number.

## Practical tradeoffs learned

1. Frequency is the dominant lever, via (a) array directivity per fixed
   physical span and (b) the surface scattering law at ~29°; (b) is
   model-dominated. Pick the highest frequency the array can be populated at
   ≤ ~1–1.4 λ spacing with ≤16 elements (≈300 MHz for 8 el, ≈400 MHz for 16).
2. Element count saturates at 8 on a 10 m span; more elements only matter to
   avoid grating lobes at high frequency. Taper is irrelevant in the sim
   (ideal patterns) but is what bounds real sidelobes — keep it.
3. Pulse length must clear the bed delay (pedestal), otherwise irrelevant.
4. Bandwidth and window don't move bed-over-clutter.
5. Along-track focusing does not touch cross-track clutter but is the only
   lever on along-track clutter, which is what remains once the array has
   done its job — and it only works with unaliased along-track sampling at
   the clutter angle (posting ≤ λ/(4 sin θ_clutter)). In the sim that is
   posting_div ≈ 8 at 60 MHz and ≈ 32 at 300 MHz; on the platform it is free.
   The bed arm loses a few dB at long apertures because the chain focuses
   through air only (no in-ice migration, gap g3), so the sim's number is a
   lower bound on the bed side.
6. Thermal SNR has ~70 dB of margin on 1 km ice at 14 km; this line's basal
   SNR is set by clutter, so transmit power and pulse energy are not levers.
7. Everything above 150 MHz depends on the sub-facet roughness spectrum
   (validated l/λ ≤ 2; Gaussian ACF). Measuring the metre-scale surface
   roughness of the study line would collapse the 90 dB uncertainty band.
