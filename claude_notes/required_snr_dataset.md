# Required-surface-SNR dataset: definition, access, and the bed-reflectivity mapping

Scouted 2026-07-31 from `~/Documents/opr/radar_return_statistics` (read-only) for
driving per-facet bed reflectivity along-track in the basal-clutter study.
Companion deliverable: per-facet `gamma` support in the coherent + multilayer
kernels (`tests/test_gamma_facet.py`); tool wiring is a follow-up task.

## 1. Exact definition

`required_surface_snr_dB` (RSSNR) is computed per decimated trace by
`compute_rssnr_dB` in
`opr/radar_return_statistics/src/radar_return_statistics/processing.py`:

```
r_surf     = c * surface_twtt / 2                      # air range
H          = (c/n) / 2 * (bed_twtt - surface_twtt)     # ice thickness, n = sqrt(3.17)
r_bed_eff  = r_surf + H / n                            # refraction-corrected effective bed range
RSSNR_dB   = 10 log10( P_surf * r_surf^2 / (P_bed * r_bed_eff^2) )
```

with `P_surf`/`P_bed` the *peak* linear powers within a margin of the OPR
surface/bed picks (`peak_power_in_window`), and `ice_permittivity = 3.17`
(config default; matches our `EPS_ICE`). It matches the RSSNR definition from
`github.com/thomasteisberg/required_surface_snr`.

Sign/reference conventions:

* **Larger RSSNR = dimmer bed.** It is the surface-to-bed peak-power ratio in
  dB after *crediting the bed for its extra spherical spreading* (dividing by
  `(r_bed_eff/r_surf)^2`). It is referenced to the *local observed surface
  return*, not to an absolute level and not to the noise floor -- system gain,
  transmit power, pulse compression, and (at nadir) antenna gain cancel in the
  ratio.
* **Removed (do not re-remove):** differential geometric spreading only, using
  exactly `r_bed_eff = r_surf + H/n`. Note this equals our kernel's refracted
  nadir spreading length for a flat interface (`L_par = L_perp = s_air + s_ice/n`),
  so the dataset removes precisely the spreading the simulator re-applies --
  no double counting.
* **Folded in (everything else):** two-way englacial attenuation, two-way
  surface transmission loss `(1-g_surf^2)^2`, the surface/bed reflectivity
  ratio, surface and bed roughness/scattering losses, birefringence -- all of
  it lands in RSSNR. Englacial attenuation is *not* corrected, and the surface
  reference means surface-reflectivity variability leaks into RSSNR with
  opposite sign.

## 2. Store access

Public icechunk stores (zarr v3, anonymous S3 read):
`s3://opr-radar-metrics/icechunk/{antarctica,greenland,ase,utig,crosssystem}`
(region `us-west-2`). Recipe (verified working from our env):

```python
import icechunk, zarr
storage = icechunk.s3_storage(bucket="opr-radar-metrics",
                              prefix="icechunk/antarctica",
                              region="us-west-2", anonymous=True)
repo = icechunk.Repository.open(storage=storage)
session = repo.readonly_session(branch="main")          # or snapshot_id=...
root = zarr.open_group(session.store, mode="r")
rssnr = root["required_surface_snr_dB"][:]
```

`xr.open_zarr` does NOT work (the `processed_frames` array lacks xarray dim
metadata) -- read arrays via zarr and build coordinates by hand.

Environment: added `icechunk` (2.1.2) and `zarr` (3.3.0) to our **dev**
dependency group (`uv add --dev icechunk zarr`; pyproject/uv.lock updated).
`import icechunk` previously failed in our env.

### IMPORTANT: pin the snapshot

The `antarctica` store's `main` branch is **mid-rebuild right now** (a
full reprocess for the new bed-pick/noise schema; tip had 800/7320 frames on
2026-07-31 and did NOT contain our frames). The completed 5,646-frame version
is snapshot **`3YH47013745B2T5ZZR50`** (2026-07-29, message
"[run] 2023_Antarctica_BaslerMKB ..."). The tool must open
`repo.readonly_session(snapshot_id="3YH47013745B2T5ZZR50")` (or re-resolve
`main` after the rebuild finishes and re-verify frames). That snapshot uses
the **old schema**: no `record_tail_noise_dB` / `bed_pick_*` /
`qc_surface_pass`; `qc_pass == False` implies all metrics NaN (no censored-
trace bookkeeping). The `ase` store does NOT cover our line (its G-H region
cut only contains other frames of the same segments).

## 3. Data model

Flat per-trace arrays over a single `slow_time` dimension (append-ordered by
frame, all seasons concatenated; 10 s decimation for `antarctica`):

* Coordinates: `latitude`, `longitude`, `elevation` (platform), `slow_time`
  (CF ns-since-epoch ints; see attrs for units/calendar).
* Selection: `frame_id` per trace (e.g. `"Data_20161105_05_005"`); also
  `frame_index` into root attrs `frame_names`/`frame_collections`.
* Variables (old-schema snapshot): `surface_power_dB`, `bed_power_dB`,
  `surface_twtt`, `bed_twtt`, `surface_elevation`, `bed_elevation`,
  `required_surface_snr_dB`, `pre_surface_noise_dB`, `post_bed_noise_dB`,
  `qc_pass`, `processed_frames`.
* Mask with `qc_pass & isfinite(rssnr)`.

## 4. Coverage of the basal-clutter passes (verified by fetch)

All 11 frames present in snapshot `3YH47013745B2T5ZZR50`
(collection `2016_Antarctica_DC8`). Per-frame RSSNR
[min / p5 / med / p95 / max] dB, median ice thickness, median trace spacing:

| frame | n | RSSNR (dB) | H med (m) | spacing (m) |
|---|---|---|---|---|
| 20161105_05_005 (anchor) | 37 | 33.2 / 39.0 / **50.7** / 69.0 / 73.2 | 740 | 1372 |
| 20161105_05_006 (anchor) | 37 | 15.5 / 16.1 / **27.6** / 40.3 / 48.7 | 650 | 1372 |
| 20161105_05_007          | 37 | 12.9 / 14.7 / **20.5** / 31.5 / 41.3 | 624 | 1364 |
| 20161028_05_004..007 (mid) | 22 ea | med 20.5 / 22.3 / 47.2 / 37.1 | 554-673 | ~2330 |
| 20161031_07_002..005 (high) | 21 ea | med 16.5 / 15.4 / 25.9 / 49.1 | 532-680 | ~2400 |

All fetched traces on these frames are `qc_pass` with finite RSSNR (111/111 on
the anchor line). Sampling is ~1.4 km (low pass) / ~2.3-2.4 km (DC8 at higher
altitude, same 10 s decimation) -- far coarser than facet spacing, so the tool
interpolates along-track. Note the anchor frames' RSSNR medians differ by
>20 dB (005 has a genuinely dim bed) -- exactly the along-track structure this
feature is meant to inject. Anchor-line arrays cached at
`scratchpad/anchor_line.npz` for the wiring phase.

## 5. Proposed mapping (per-facet bed reflectivity, attenuation held fixed)

At nadir, with the simulator's constant one-way attenuation `A` (dB/km),
local ice thickness `H(x)` (km), Fresnel surface `g_surf` and two-way power
transmission `T2_dB = 20 log10(1 - g_surf^2)`, the simulator's
spreading-removed bed-to-surface power ratio is

```
P_bed/P_surf |no-spreading, dB = G2(x) + T2_dB - |G_surf|^2_dB - 2 A H(x)
```

with `G2(x) = |Gamma_bed(x)|^2` in dB. Equating to the dataset
(`-RSSNR(x)` is the same quantity by definition) and solving:

```
G2(x) [dB] = 2*A*H(x) - RSSNR(x) + K
gamma_facet(x) = -10 ** (G2(x) / 20)        # FIELD coefficient, sign of Fresnel ice->bed
```

* Physical anchoring: `K_phys = |G_surf|^2_dB - T2_dB = -11.04 - (-0.71)
  = -10.32 dB` (eps_ice 3.17). No other terms belong in K -- spreading is the
  only thing the dataset removed, and it equals what the kernel re-applies.
* **Recommended: median anchoring.** Choose K so the *study-segment median*
  G2 equals the current constant -12.9 dB (= Fresnel power reflectivity of
  eps 3.17 -> 8.0, the validated level):
  `K = -12.9 - median_seg(2*A*H - RSSNR)`. Measured on the anchor frames
  005-007 with A = 15: `median(2AH - RSSNR) = -11.5 dB` -> `K = -1.4 dB`
  (compute per actual study segment in the tool, not per line).

Why median anchoring: RSSNR is surface-referenced and attenuation-inclusive,
so `K_phys` transfers every absolute unknown -- the real (non-Fresnel, rough)
surface reflectivity, and above all the attenuation mismatch -- straight into
the bed level. The gap is material: with A = 15 dB/km, K_phys puts the segment
median G2 at -21.8 dB (9 dB below the validated -12.9); with the
cross-season-calibrated effective A = 31 it would swing ~+20 dB. Median
anchoring absorbs all of that at the median thickness (residual error only
`2*dA*(H(x) - H_med)`, a few dB across this line's 496-1063 m range) and makes
the dataset supply what we actually trust it for: the along-track *relative*
structure, while the absolute level stays continuous with the current
constant-gamma results. Report `K - K_phys` as a diagnostic of the
attenuation/surface-model consistency.

Double-counting checklist for the tool: keep the simulator's surface gamma,
transmission `(1-g^2)`, attenuation `A`, and refracted spreading exactly as
they are (all are on the *simulator* side of the equation above); do NOT add
any extra spreading or attenuation correction to RSSNR. If per-facet *bed
roughness* (Gerekos sub-facet) is enabled on the bed, its mean power loss is
then counted twice -- either keep the bed smooth (current basal-clutter
default) or subtract the roughness mean-power attenuation from G2.

Dynamic range note: on frame 005 the mapping reaches G2 ~ -45 dB (dim bed
under thin ice). `gamma_facet` is per-facet in the kernel, so this is purely a
data-preparation question.

## 6. Open questions for the wiring phase

1. **Interpolation/extent**: nearest vs linear in along-track distance from
   ~1.4-2.4 km samples to per-facet values; how far to extrapolate across the
   cross-track DEM extent (constant along cross-track normal to the line?).
   The bed DEM chunking in `run_basal_clutter.py` gives each facet an (x, y);
   project onto the line's along-track coordinate.
2. **Which pass supplies RSSNR**: the three passes fly the same line at
   different altitudes and their RSSNR profiles differ (e.g. anchor med 50.7
   vs mid-line frames covering the same ground). Anchor-line (low-altitude)
   RSSNR is presumably cleanest (best SNR); decide whether all three sims
   share the anchor-derived gamma field (recommended: yes, one ground truth).
3. **H(x) source**: use the dataset's own `bed_twtt - surface_twtt` (self-
   consistent with RSSNR) vs the simulator's DEM thickness at the facet.
   Recommended: dataset thickness for the `2*A*H` term (keeps the mapping
   internally consistent), DEM only for geometry.
4. **Censored traces**: the pinned old-schema snapshot has no
   `bed_pick_available`; NaN/missing RSSNR (none on these frames, but possible
   elsewhere) means "bed too dim to pick" -- treat as a *floor* on RSSNR (cap
   G2 low), not as missing-at-random.
5. **Snapshot refresh**: after the antarctica rebuild completes, switch to the
   new `main` (gains censoring columns + `qc_surface_pass`) and re-verify the
   frame values match the pinned snapshot.
6. **K scope**: one K per study segment (recommended) vs one global K for all
   three passes; and whether to expose `--gamma-from-rssnr / --gamma-const`
   as a tool switch for A/B comparisons.
