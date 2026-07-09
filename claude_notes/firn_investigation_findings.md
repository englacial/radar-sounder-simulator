# Firn power plateau investigation — MCoRDS-matched chirp redo (2026-07-09)

Deliverable: honest observation, not a gate. Everything in
`outputs/firn_investigation/` (report.html self-contained; the earlier
DELTA-pulse sweep — 2026-07-08, summarized at the bottom — is archived in
`outputs/firn_investigation/delta_runs/`). Sweep tool
`tools/run_firn_investigation.py` (resumable), integrity test
`tests/test_firn_investigation.py` (integration-marked).

## Design changes vs the delta sweep (user-directed)

Same scenes (flat 600 m, 500 m AGL, 195 MHz, 4 m facets, 3 traces, coherent;
N ∈ {10,20,40,80} point-sampled B26 layers over 1–119.66 m, equal + 3 seeded
random placements, surface-only reference), but:

1. **Waveform**: chirp B = 30 MHz, hann compression window, Tpd = 3 µs (the
   MCoRDS 2017 P-3 parameters, `outputs/cache/mcords_2017P3_params.json`).
   Chirped range resolution (hann 1.44·c/2B): 7.20 m air / 4.42 m in-firn —
   N ≥ 40 (spacing ≤ 3 m) is fully unresolved; N = 10/20 resolved.
2. **Alias-free dt**: dt 5 → 4 ns (f0·dt = 0.78 → f_a = 195−250 = −55 MHz;
   |f_a| = 55 MHz > B/2 = 15 MHz); n_samples 512 → 640 keeps the 2560 ns
   window. simulate()'s in-band-alias warning asserted silent on every call.
   interp_bins off (unsupported for multilayer; unnecessary at alias-free dt).
3. **Cutoff**: 30 min per simulation (compile-inclusive), replacing 25/15.
4. **Complex fields saved** per run (complex64, incl. reference) and a
   **surface-subtracted profile** |E_total − E_surface|² per layered run
   (kernel + convolution are linear so the layer sum is exact). E_surface =
   the run's own layer-0 field; cross-checked per run against the reference
   field scaled by the surface-gamma ratio: agreement ≤ 1.4e-6 of the
   reference peak (exactly 0 for equal placements). Both profiles share the
   raw-surface-peak normalization and 5 m smoothing.

## Findings (figures + table in outputs/firn_investigation/report.html)

1. **The paper's operational criterion (≥ −0.05 dB/m over > 10 m) is now
   MET**: raw profile 6/16 runs (max 18.0 m, random N=80 s1 @ 42–60 m;
   13.3 m s2 @ 30–43 m), surface-subtracted 11/16 (10.3–18.7 m; all random
   N=20/40, 3 of 4 N=80). Delta sweep: 0/13, best 9.6 m. Subtracted
   intervals at N ≥ 20 often start at the surface ([0.1, 16–18 m]) — the
   paper's near-surface plateau. Threshold sensitivity < 1 m on the long
   intervals. Caveat: equal-placement raw intervals at 99–117 m ride the
   periodic stack's flat deep tail; random placement is the physical case.
2. **Attribution**: (a) alias-free chirp dropped the surface-only reference
   floor 6–17 dB over 5–45 m (10–20 m band: −17.3 → −34.3 dB) — most of the
   delta sweep's "upper-45 m surface contamination" was the in-band
   quantization alias, so raw ≈ subtracted beyond ~10–15 m now; (b) the
   4.4 m pulse integrates several interfaces per cell, raising deep bands
   2–6 dB vs delta (equal N=80 100–119 m: −40.4 → −34.1 dB) and flattening
   the decay; (c) subtraction is decisive only in the top ~15 m (raw
   secondary max is always the ~5 m surface shoulder at −4.0..−8.8 dB;
   subtraction removes it exactly — random N=20 s0 5–10 m: −20.2 → −59.0 dB)
   and exposes the near-surface plateaus; (d) the newly-completed random
   N=80 seeds supply the only raw-criterion passes of real length.
3. **Secondary max (subtracted)**: −10.2..−23.0 dB rel. surface (median
   ≈ −17.5) at 5–65 m — the paper's 10–15 dB regime, slightly weak on
   average. Outlier: random N=10 s2 (−1.4 dB; real layers at 3.98/4.34 m).
4. **Resolved→unresolved**: subtracted pass rate 1/4 at N=10 vs 10/12 at
   N ≥ 20; raw passes of length only at N=80. Deep floor still rising at
   N=80 (50–100 m random-mean ≈ −31 → −27 dB; 100–119 m −46 → −36 dB):
   3–6 dB per doubling of N — NOT converged. Realized |γ| medians
   −40.5..−52.9 dB (N=10→80) remain ~30–40 dB above the 1 mm
   adjacent-sample stats (−90.7 median).
5. **D+ (joint refraction solve) trigger verdict**: condition 1 of
   `claude_notes/joint_refraction_solve_note.md` is met — the phenomenon
   strengthens with N through the compile-reachable range without converging,
   and O(N²) compile (26.6 min at N=80) blocks N ~ 150–300 convergence runs.
   Nuance: the operational plateau no longer needs larger N (chirp +
   subtraction demonstrate it at N=20–80); schedule D+ when converged
   absolute levels / fine-layer statistics vs a 1-D transfer-matrix referee
   matter, not for morphology alone.

## Runtime (compile-inclusive first call per N / cached same-N)

N=10: 14.0 / 0.6 s; N=20: 69.8 / 2.1 s; N=40: 318.6 / 8.0 s;
N=80: 1593.5 s (26.6 min, under the 30 min cutoff) / 31.2 s — all 16 layered
runs + reference completed, zero skips (the delta sweep had skipped
random_N80 s0–s2). The 512→640-sample grid recompiled everything (n_samples
is in the kernel cache key); the persistent XLA cache reused the integration
test's N=10 compile. Total sim wall ≈ 35 min.

## Delta sweep history (2026-07-08, archived in delta_runs/)

13/16 layered runs completed (random_N80 skipped on the old 25 min cutoff);
NO run met the >10 m criterion (best 9.6 m, random N=40 s1); secondary max
pinned at 5.1–5.5 m / −6.0..−13.7 dB; upper ~45 m contaminated by what this
sweep identified as mostly the dt = 5 ns in-band envelope-quantization alias
(f_a = −5 MHz), not physical surface response. The loader's edge-normalized
0.1 m density smoothing (fixing a spurious deep reflector from zero-padded
`mode='same'`) carries over unchanged. The old case `tests/test_firn_plateau.py`
previously got one surgical repair (depth-sorted digitized Fig. 9 curves);
its framing remains under review separately.
