# Cross-season repeat line at the 2012 DC-8 anchor (2026-07-30)

Scouting only — no simulations. Goal: find 2014/2016/2018 Antarctica DC-8 frames
that re-fly the line of `2012_Antarctica_DC8 / 20121023_04_008`, pick the best
common segment, and collect the parameters a follow-up simulation needs.

Method: anchor nav from `soundersim.opr.load_frame`, projected to EPSG:3031;
`xopr.query_frames(collections=[season], geometry=<3 km buffer of the anchor
track>)`; candidate frames downloaded and each trace projected onto the anchor
polyline (perpendicular distance `d`, along-track coordinate `s`, `s=0` at anchor
trace 0). STAC frame geometries are only 2–18 point LineStrings (usually just the
endpoints) and are useless for overlap length — the exact numbers below all come
from the downloaded nav. Params via `tools/run_altitude_comparison.mcords_params`
/ `map_window` / `pick_oversample`. Scripts in the session scratchpad.

## Anchor: 2012_Antarctica_DC8 / 20121023_04_008

* 2012-10-23 17:11:14–17:15:06 UTC, 1732 traces, 51.30 km of track, ~29.6 m/trace.
* Endpoints (lat, lon): (−75.18422, −96.91905) → (−75.08093, −98.68331).
  West Antarctica, upstream Thwaites/Pine Island area.
* **High-altitude flight**: Elevation 9690–9703 m (WGS84), median AGL
  **9217 m** (from `Surface` twtt 61.49 µs). Surface elevation ~520 m.
* Ice thickness from the CSARP_layer bottom pick (ε=3.15): median 856 m,
  range 533–1119 m; 97.9 % of traces picked.
* Product grid: 787 samples, dt **105.21 ns**, t0 **25.221 µs** (the window
  starts at 3.78 km range because the platform is 9.2 km up).
* Products present: CSARP_standard, CSARP_qlook, CSARP_mvdr, CSARP_layer.

All three target seasons exist as DC-8 collections (`2014_Antarctica_DC8`,
`2016_Antarctica_DC8`, `2018_Antarctica_DC8`) and **all three re-fly this line
end to end**. No substitute seasons were needed.

## Per-season selection

Each season's repeat covers the whole 51.3 km anchor line but splits it across a
frame boundary. Using ONE frame per season, the frame that carries the longest
piece is:

| Season | Frame | Anchor s covered (km) | Overlap in common window | offset med / mean / p90 / max (m) | median AGL (m) | fc (MHz) | B (MHz) | Tpd (µs) | product dt (ns) | ft_wind | products |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2012 (anchor) | 20121023_04_008 | 0.00–51.30 | 30.65 km | 0 (reference) | **9217** | 193.90 | **9.5** | 1, 10 | **105.21** | tukeywin(N,0.2) → modeled `none` | std, qlook, mvdr, layer |
| 2014 | 20141029_05_013 | 8.46–51.30 | 30.65 km | 12.3 / 15.0 / 30.3 / 80.0 | 465 | 190.0 | 50 | 1, 3, 10 | 20.000 | hanning → `hann` | std, qlook, mvdr, layer |
| 2016 | 20161104_05_008 | 0.00–42.20 | 30.65 km | 11.9 / 14.7 / 30.2 / 78.5 | 446 | 190.0 | 50 | 1, 3, 10 | 20.202 | hanning → `hann` | std, qlook, mvdr, layer |
| 2018 | 20181107_01_011 | 0.00–39.11 | 30.65 km | 4.5 / 9.2 / 16.7 / 71.0 | 447 | 190.0 | 50 | 1, 3, 10 | 20.000 | hanning → `hann` | std, qlook, mvdr, layer |

Offset stats are over the recommended common window only. All four tracks run in
the **same direction** (increasing anchor `s` with increasing trace index).

Adjacent frames if the full 51.3 km is wanted (two frames per season):
2014 `20141029_05_012` (s 0.00–8.42, mean offset 82 m — the noisiest of the set),
2016 `20161104_05_009` (s 42.23–51.30, mean 25 m),
2018 `20181107_01_012` (s 39.13–51.30, mean 16 m).

Also inside the 3 km buffer but **not** usable as repeats: `20161103_07_003`
(2016, a *crossing* line, only 0.31 km within 500 m — but note it is another
high-altitude pass, 10.3 km elevation), `20181116_02_014` (closest 5.1 km),
`20121016_03_003` (closest 5.3 km), and anchor-segment neighbours
`20121023_04_007` / `_009`.

## Recommended common segment

**Anchor along-track s = 8.46 → 39.11 km (30.65 km)**, set by the 2014 frame
boundary at the start and the 2018 frame boundary at the end. Every selected
frame stays within 80 m of the anchor track over the whole window (median 4.5–12.3 m).

* Endpoints (anchor nav): (−75.16644, −97.21066) → (−75.10646, −98.26529).
* Slices (`slow_time` index, half-open) into each full frame:
  * 2012 `20121023_04_008`: `slice(286, 1320)` of 1732 — 1034 traces, 29.6 m/trace
  * 2014 `20141029_05_013`: `slice(1, 2067)` of 3332 — 2066 traces, 14.8 m/trace
  * 2016 `20161104_05_008`: `slice(1058, 3124)` of 3333 — 2066 traces, 14.8 m/trace
  * 2018 `20181107_01_011`: `slice(1267, 3333)` of 3334 — 2066 traces, 14.8 m/trace

Pairwise overlaps are not a constraint here (every pair shares ≥30.65 km; the
2012↔2014 pair shares 42.84 km, 2012↔2016 42.20 km, 2012↔2018 39.11 km).

## Radar / processing parameters

| | 2012 | 2014 | 2016 | 2018 |
|---|---|---|---|---|
| f0–f1 (MHz) | 189.15–198.65 | 165–215 | 165–215 | 165–215 |
| center / bandwidth (MHz) | 193.90 / **9.5** | 190 / 50 | 190 / 50 | 190 / 50 |
| Tpd waveforms (µs) | 1, 10 | 1, 3, 10 | 1, 3, 10 | 1, 3, 10 |
| tukey time window | 0.2 | 0.1 | 0.2 | 0.2 |
| ft_wind (compression) | `tukeywin(N,0.2)` | `hanning` | `hanning` | `hanning` |
| mapped soundersim window | `none` (approx., recorded) | `hann` | `hann` | `hann` |
| PRF (Hz) | 8 000 | 12 000 | 12 000 | 12 000 |
| ADC fs (MHz) | 111.111 | 150 | 150 | 150 |
| product dt (ns) / n_samples / t0 (µs) | 105.21 / 787 / 25.221 | 20.000 / 3402 / −0.02 | 20.202 / 3367 / 0.000 | 20.000 / 3154 / 0.020 |
| `pick_oversample` k (dt/k alias-free) | 6 | 5 | 5 | 5 |
| SAR σ_x (m) / sar_type | 5 / f-k | 2.5 / f-k | 2.5 / f-k | 2.5 / fk |
| start_eps | 3.15 | 3.15 | 3.15 | 3.15 |
| img_comb (µs) | none (single image) | [3, −∞, 1 ; 10, −∞, 3] | same | same |
| param layout (`_PARAM_LAYOUTS`) | `param_csarp` | `param_sar` label, actually `param_records.radar` | same | `param_sar` (true) |
| qlook grid | 787 / 105.21 ns / 178 traces | 3402 / 20.0 ns / 302 traces | 3368 / 20.202 ns / 706 traces | 3154 / 20.0 ns / 1335 traces |
| bottom pick over full frame | 97.9 % picked, 856 m median | 100 %, 850 m | 100 %, 1011 m | 100 %, 1044 m |
| |Roll| p95 / max (deg) | 2.72 / 5.18 | 0.65 / 1.39 | 0.73 / 1.05 | 0.45 / 0.66 |

Ice-thickness medians above are per *full frame* (different spatial extents), not
over the common window.

## Data quirks that will bite the simulation stage

1. **The anchor is a ~9.2 km-AGL, 9.5 MHz-bandwidth pass; the repeats are all
   ~450 m AGL, 50 MHz.** This is by far the dominant difference — a 20× altitude
   ratio and a 5× bandwidth ratio (unwindowed range resolution c/2B: 15.8 m vs
   3.0 m in air, 8.9 m vs 1.7 m in ice at ε=3.17 — and note the 2012 product's
   105.21 ns bin is 15.77 m, i.e. exactly one resolution cell, so that frame is
   critically sampled with no oversampling headroom). The comparison is *not* a
   same-instrument repeat; it is an altitude **and** waveform trade. Plan the
   twtt windows, facet spacing (β=0.5 Fresnel scales with √(λ·r)) and cross-track
   reach per frame — the 2012 case needs a much wider cross-track extent for the
   same clutter population.
2. **Fast-time posting differs 5×** (105.21 ns vs ~20 ns). Any depth-profile
   smoothing/binning must be expressed in metres, not bins (the b26-overflight
   lesson). `pick_oversample` gives k=6 for 2012 and k=5 for the others.
3. **Surface-pick vertical registration is inconsistent between seasons.**
   Comparing `Elevation − c·Surface/2` against REMA v2.0 32 m over the common
   window (both WGS84-ellipsoidal, so no geoid term):
   * 2012: **−33.0 m bias, σ 14.1 m** — the bias is ~2 range bins and the scatter
     is ~1 bin, i.e. the high-altitude coarse-grid surface pick is both offset and
     quantization-noisy.
   * 2014: +13.1 m, σ 1.35 m 2016: +13.6 m, σ 1.90 m 2018: **−9.4 m**, σ 2.75 m
   So the seasons disagree with each other by up to 23 m and with REMA by up to
   33 m. Simulated-vs-measured twtt alignment must be done per frame (fit a
   constant offset, as `leading_edge_gate` already does), never assumed common.
4. **`ft_wind` decode quirk in `mcords_params` for 2014 and 2016.** Both frames
   carry `param_records.radar`, so `_PARAM_LAYOUTS` selects the `param_sar` entry
   and then fails to find `param_sar/radar/wfs/ft_wind` (there is no `param_sar`
   struct), falling back to the readme default string
   `"hanning (ft_wind decode failed; CReSIS readme default)"`. Verified by hand:
   the real value is in `param_csarp.csarp.ft_wind` and **is** `hanning` in both
   cases (2014 arrives as an undecoded `scipy` `mat_struct`, 2016 as a decoded
   `function_handle` dict), so the fallback happens to be correct — but the
   provenance string is misleading and should not be quoted as measured. 2018
   decodes properly through `param_sar`; 2012 decodes properly through
   `param_csarp` (`tukeywin(N,0.2)`).
5. **2016 uses a 20.202 ns grid, not 20.000 ns** (fs 150 MHz / different ft_dec
   bookkeeping), so the three "50 MHz" seasons are not on an identical fast-time
   lattice.
6. **CSARP_qlook is decimated very differently per season** (178 / 302 / 706 /
   1335 traces for the same ~50 km frames), and the 2012 qlook has the *same*
   coarse 105.21 ns grid as its standard product. A qlook-based like-for-like
   comparison across seasons is not apples-to-apples in along-track posting.
7. **Image combination**: 2014/2016/2018 CSARP_standard is a 3-waveform composite
   (1/3/10 µs, `img_comb = [3 µs, −∞, 1 µs; 10 µs, −∞, 3 µs]`), so the effective
   pulse length varies with depth in the *measured* product; the repo convention
   simulates only the longest (10 µs / bed) waveform. 2012 has just 2 waveforms
   and no `img_comb`.
8. Minor: `param_*.combine.method` reads `"mvdr"` in the 2014 and 2016
   CSARP_standard files (vs `"standard"` in 2018). Most likely the stored param
   struct was overwritten by the later CSARP_mvdr run rather than describing the
   standard product — low confidence, do not build on it.
9. The 2012 frame has noticeably more roll (p95 2.72°, max 5.18°) than the
   low-altitude repeats (p95 ≤0.73°), so `roll_source="nav"` matters more there.
10. All four frames have CSARP_standard, CSARP_qlook, CSARP_mvdr and CSARP_layer
    assets — nothing missing.
