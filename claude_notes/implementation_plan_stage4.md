# Implementation plan: stage 4 (waveform, antenna patterns, post-processing)

Per `docs/overview.md` stage 4 (antenna beampatterns, unfocused/focused SAR) plus the review directive that stage 4 must handle **linear chirp effects** — motivated concretely by the firn investigation's finding that the delta-pulse-at-carrier idealization produces a nonphysical off-nadir pedestal (`2cos²θ·|sin(kΔ)|` unless the bin width is an exact multiple of λ/2; measured −18.5 dB shoulder). Builds on stages 1–3 (141 CI + 19 integration tests). **Status: complete (2026-07-08). M20–M24 done: 183 CI + 22 integration tests green. Key results: chirp post-convolution validated vs multi-frequency referee (1.06 dB median; D4-1 verdict = convolution stays primary with hygiene rules |f_a|>B/2 and k·L·sinθ<π); pedestal mechanism = aliased-carrier envelope quantization, chirped result ~60 dB below worst-case delta pedestal; antenna patterns closed-form verified (5-elt array suppresses 8–9 dB off-nadir vs dipole 2–4 dB); processing layer validated to textbook (focused width 0.874× vs 0.886× theory); xOPR coherent_bed at MCoRDS level (params from product files, alias-free dt/4 grid, dense λ/4 sub-segment for unfocused; measured-vs-sim C/S ratios agree ~2–8 dB; NOTE the frames' native dt is 33.3 ns, not 5 ns as the plan brief assumed). docs/processing_simulation.md written. Firn plateau revisit (chirp + surface-field subtraction) completed 2026-07-09 — see claude_notes/firn_investigation_findings.md: operational plateau criterion now MET (11/16 subtracted, 6/16 raw); D+ joint-refraction-solve trigger condition 1 met. Committed with this stage. NEXT: D+ joint refraction solve (user-approved 2026-07-09).**

**Deferred item this stage unblocks:** the firn plateau follow-up (surface-field subtraction + chirped traces) runs immediately after stage 4 — first post-stage entry, per review.

## Physics scope

Three additions, all default-off so every existing test/fixture stays green (delta pulse, isotropic antenna, no processing = today's behavior, bit-compatible):

1. **Waveform (linear chirp)**: the kernel's binned complex trace is a delta-response with *exact* per-facet carrier phase (`e^{−2jkr}` from true range) and dt-quantized envelope delay. The primary method is **post-convolution**: convolve the complex trace with the pulse-compressed chirp kernel (windowed autocorrelation of the LFM waveform — analytic or FFT-built), optionally splitting each facet's contribution linearly between adjacent bins to suppress envelope-delay quantization. This is Gerekos-2018-style, costs nothing per facet, and replaces the monochromatic bin-cancellation pedestal with the physical (windowed) sidelobe floor. Its accuracy is validated (M21) against **multi-frequency synthesis** — running the kernel at K frequencies across the band and IFFTing — which is exact but K× the cost; the referee also measures what the convolution neglects (facet sinc-directivity variation across the band: ±7.5% in k for 30 MHz at 195 MHz).
2. **Antenna patterns**: per-facet two-way gain G²(θ,φ) applied in both kernels (the stage-1 "multiplicative hooks" made real). Analytic patterns first (isotropic, half-wave dipole, uniform cross-track array of n elements — the MCoRDS-like case), plus a tabulated-pattern option. Pattern frame from nav: track-aligned by default, optionally rolled using the frame's Roll (xOPR provides attitude).
3. **Post-processing**: unfocused SAR (presumming / coherent along-track summation, the CReSIS-style processing our CSARP_standard comparisons implicitly assume) and basic focused SAR (time-domain backprojection over a configurable aperture on simulated fields). Processing steps recorded in Dataset attrs per docs/output.md's original promise.

## Milestones

### M20 — Waveform config + chirp convolution
- `WaveformConfig` on RadarConfig: `kind: "delta" | "chirp"`, bandwidth, pulse length, window (hann/hamming/none/taylor), `interp_bins: bool`. Delta default = exact current behavior (regression-gated).
- Compressed-pulse kernel construction + complex convolution on `field` (and optional power-envelope convolution for incoherent mode, default off — simc parity untouched).
- Tests: point-target range response — resolution c/(2B) to the bin, peak sidelobe level matching the chosen window's textbook value (±1 dB), phase at peak preserved; delta-default bit-compatibility; energy bookkeeping under convolution.

### M21 — Multi-frequency referee + the pedestal fix, verified
- Small-scene referee: kernel at K frequencies → IFFT range profiles (reuses the cached-jit factory; K ~ 64–128 on tiny scenes).
- Validate M20 convolution vs referee: point target, flat surface (the pedestal case — chirped convolution must reproduce the referee's sidelobe-floor behavior; quantify the residual from sub-bin quantization with/without `interp_bins`), gently rough surface. Document the error budget incl. neglected in-band directivity variation.
- **Decision gate**: if convolution errs > a few dB in the physically important regions, promote multi-frequency synthesis to the primary path for coherent runs (cost recorded); otherwise it remains the referee.
- Report case (Radar equation comparison): flat-surface off-nadir response, delta vs chirped vs referee — the definitive retirement of the monochromatic pedestal.

### M22 — Antenna patterns
- `AntennaConfig`: pattern kind + params, element axis/boresight convention, roll source (`none` | `nav`). Two-way field gain per facet per trace in both kernels (traced argument, no recompile per pattern change where feasible).
- Tests: isotropic ≡ today (bit-compatible); dipole null/gain vs closed form on synthetic geometry; array pattern main-lobe/null positions vs analytic; pattern-weighted flat-surface integrand vs 1-D analytic weighting.
- Report case: flat + hill scene cluttergrams under isotropic vs dipole vs array — the cross-track clutter suppression story (directly relevant to why the measured Helheim frame shows less off-nadir clutter than our stage-2 cluttergram).

### M23 — Post-processing (unfocused + basic focused SAR)
- `ProcessingConfig`: presum count / unfocused aperture length; focused backprojection (aperture length, straight-track assumption documented). Operates on the output Dataset (processing layer per the three-layer architecture — NOT inside kernels); steps recorded in attrs.
- Tests: point target — unfocused gain ≈ N_presum on coherent target, azimuth resolution after focusing ≈ λ/(2·L_ap)·r (±10%); speckle statistics under multilooking behave per theory (contrast ~ 1/√N_looks).
- Trace spacing guard: warn when along-track sampling is too coarse for the requested aperture (Doppler aliasing, λ/4 criterion from the handoff notes).

### M24 — xOPR upgrade + report + docs
- Re-run the two coherent+bed xOPR cases with: MCoRDS-like chirp (params sourced from CReSIS documentation/frame metadata — record exactly what's used), cross-track array pattern, unfocused presumming approximating CSARP_standard's processing level. The measured-vs-simulated comparison finally becomes dynamic-range-meaningful: revisit envelope metrics (thresholds still record-first), speckle, and the clutter-fan visual comparison.
- Existing report cases re-verified untouched (delta defaults); new waveform/antenna cases added under "Radar equation comparison".
- `docs/processing_simulation.md` (waveform + antenna + processing, conventions and verification summary); stale-bullet sweep over the other docs.
- **Post-stage hook (first follow-on task)**: firn plateau revisit — chirped traces + surface-field subtraction (save complex fields this time), diagnostics re-extracted; this is the user-deferred item and the D+ trigger evidence re-evaluation.

## Decisions I'll make unless redirected
- **D4-1** Post-convolution as the primary chirp method, multi-frequency synthesis as referee (promoted only if M21 says so).
- **D4-2** Scalar antenna gain (no polarization, consistent with scope); two-way as G² field weighting.
- **D4-3** Focused SAR = straight-track time-domain backprojection only (no motion compensation, no range migration beyond what backprojection inherently handles); enough for point-target validation and simulated-data studies, not a production processor.
- **D4-4** Incoherent mode stays delta + no antenna default (simc parity fixtures untouched); waveform/antenna available as opt-in there too.
- **D4-5** Agent split as before: config/plumbing and report/xOPR reruns to Opus; waveform physics, referee, and processing validation to the strongest model; I review each wave.

## Known risks
- Sub-bin envelope quantization may need `interp_bins` on by default for chirped runs (M21 measures it); interpolated binning slightly changes the delta-mode-adjacent code paths — regression tests guard the default.
- In-band directivity variation (±7.5% k-span) is neglected by convolution — M21 quantifies; if it matters, that's a point for multi-frequency primary mode on the affected scenes only.
- CReSIS processing chain realism: CSARP_standard involves steps we won't replicate exactly (motion comp, deconvolution details); M24 comparisons stay record-first, framed as "processing-level-matched", not exact emulation.
- Focused SAR on multilayer scenes: backprojection assumes a propagation model; stage 4 focuses through AIR only (surface-referenced), with in-ice focusing noted as future work.

## Definition of done
Chirped point-target response matches textbook resolution/sidelobes; flat-surface pedestal demonstrably replaced by windowed sidelobe floor and validated against the multi-frequency referee; antenna patterns verified against closed forms with a clutter-suppression report case; unfocused + focused processing validated on point targets; xOPR coherent+bed cases re-run at MCoRDS-like processing level with recorded metrics; all existing cases/fixtures green under delta defaults; docs page written; report grown accordingly. Then, immediately after: the firn plateau revisit.
