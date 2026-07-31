# Basal-clutter altitude triplet — PILOT results (2026-07-31)

Tool: `tools/run_basal_clutter.py` (pilot s=30–40 km, 48 sim traces/pass,
coherent SURFACE+BED only, surf-rough ON, att 15 dB/km). Outputs in
`outputs/basal_clutter/pilot/`. Scout: `claude_notes/basal_clutter_scout.md`.
Stopped after the pilot per plan — the 50 km run needs a go-ahead.

## Derived cross-track reaches (nadir-bed delay + 3 µs, both interfaces)

| pass | surface reach | bed reach (refracted) | ct used | facet spacing | facets/iface | wall |
|---|---|---|---|---|---|---|
| low 442 m | 2493 m | 927 m | ±2493 m | 10.67 m | 1.31 M | 36.4 s |
| mid 9150 m | 6185 m | 2957 m | ±6185 m | 46.26 m | 179 k | 11.3 s |
| high 10684 m | 6643 m | 3181 m | ±6643 m | 49.77 m | 169 k | 4.2 s |

Surface interface always binds (its target delay spans the whole ice
column); no caps applied. The old 6 km default cap would have clipped both
high passes. High passes are ~4.7× coarser-faceted and *cheaper* than the
low pass despite 2.6× the swath (scout quirk 4 confirmed).

## Headline: does the sim reproduce the ~20 dB altitude effect?

Mid-column mean power (surf+1.0 → bed−0.5 µs), dB rel own surface peak:

| pass | measured | sim | sim − meas |
|---|---|---|---|
| low | −54.7 | −71.0 | −16.3 |
| mid | −35.4 | −41.4 | −6.0 |
| high | −35.2 | −41.3 | −6.1 |
| **high − low** | **+19.5** | **+29.7** | +10.2 |

* **YES, the sim shows much more clutter at altitude** (+29.7 dB high−low
  vs +19.5 measured; mid ≈ high in both, as the scout found).
* The sim OVERSHOOTS the trend by ~10 dB, and the overshoot is entirely at
  the LOW pass: at altitude the surface+bed sim comes within ~6 dB of the
  measured mid-column, i.e. high-altitude mid-column clutter is mostly
  surface+bed geometry; at 442 m the sim is 16 dB below measured — the
  measured low-pass mid-column is dominated by englacial returns (internal
  layers/volume scatter, clearly visible in its radargram) that the
  surface+bed sim omits BY DESIGN. So the measured +19.5 dB is compressed
  by an englacial floor the model doesn't carry; the geometric-clutter
  altitude effect itself is larger.
* Measured floors (record-end −12..−8 µs window): −125 / −86 / −76 dB rel
  surface — no pass is noise-limited in the mid-column. (First attempt used
  the pre-surface window: WRONG on the low pass, TX-leakage/img_comb zone
  reads 26 dB ABOVE its mid-column. Record tail last ~4 µs is rolled off
  too — probed 2026-07-31, see FLOOR_TAIL note in the tool.)

## Decomposition (per-interface coherent fields): SURFACE-borne

The mid-column window is **surface-borne at all three altitudes**:
bed-borne mid-column contributions are ≤ −115 dB rel surface (negligible);
sim surface-borne ≈ sim total there. Bed-borne energy dominates only the
bed window (bed−0.5 → bed+1.5 µs), where sim vs measured agree well:
low −50.9/−55.5, mid −47.4/−48.1, high −47.2/−47.7 dB (att 15 dB/km left
the low-pass sim bed ~4.6 dB hot; mid/high within 1 dB). The measured
high-pass "basal" clutter filling the column is off-nadir SURFACE
scattering, not bed scattering — the discriminator the study wanted.

## Sanity

Surface-gate medians 0.39/0.41/0.50 bins (≤5 gate, pass); p90 is 18/27
bins on the high passes purely because the measured high-altitude Surface
pick is ~3.6 bins noisy (scout registration table) — offsets fitted per
pass. Fields all finite; dropped-power fraction ≤ 1.5e-4; bed_clamp 0;
scout-predicted spacings (10.67/47/50.6 m) and reaches (±5.4 km at bed
delay without margin) reproduced by the tool's derivation (tested).

## 50 km projection (s=18–68 km, 240 traces, 5 pilot-sized chunks/pass)

low ~182 s, mid ~57 s, high ~21 s → **~260 s (~4.5 min) total sim time**.
Caveats: per-chunk JAX recompile (chunk shapes differ) could add minutes;
the wider bbox needs new REMA/BedMachine windows (one-time downloads,
few minutes). Realistic end-to-end: **~10–15 min**. No parameter changes
suggested by the pilot; if anything, consider whether to add the firn/
internal-layer stack later to close the low-pass −16 dB englacial gap
(out of scope for this study by explicit design).

## Figure notes

`radargrams.png`: sim reproduces the bright mid-column fill at altitude and
the dark column at 442 m; sim bed arcs are visibly smoother than measured
(BedMachine 500 m texture caveat, expected). `decomposition.png`: orange
(surface-borne) carries the mid-column at all altitudes; green (bed-borne)
switches on at the first off-nadir bed arrival and owns the bed window.

## Full 50 km run (2026-07-31)

All 15 chunks green, 260 s total simulation (36 s/chunk low, 11 mid, 4
high), reaches surface 2.5/6.4/6.9 km. Mid-column mean power rel own
surface peak (meas/sim dB): low -54.6/-72.2, mid -35.5/-45.0, high
-34.8/-44.0 -- sims reproduce the altitude effect (high-low +28.2 dB sim
vs +19.8 measured; mid ~ high in both). Decomposition: surface-borne at
all altitudes; bed-borne is confined to the bed window, where sim bed
arcs reproduce the measured 1-5 km hyperbola trains (s~55-65 km) though
visibly smoother (BedMachine 500 m posting). The measured-over-sim
mid-column residual (~9-18 dB, largest at the low pass) is englacial
scattering excluded by design in this surface+bed study.

## Picked-bed 50 km run (`--picked-bed`, 2026-07-31)

`outputs/basal_clutter/full_pbed/` (vs BedMachine `full/`). Bed =
BedMachine + resid(s), resid(s) = picked_bed(s) - BedMachine at nadir(s) on
the anchor along-track axis; picks from the LOW pass ONLY
(20161105_05_005-007) applied identically to all three sims. Elevations use
the tool's existing convention (surface = Elevation - c*Surface/2,
thickness = (Bottom-Surface)*c/(2*sqrt(3.17)), WGS84-ellipsoidal like
REMA/BedMachine). Nadir matches the picks exactly; BedMachine's cross-track
relief is preserved (a constant cross-track extension of the 1-D profile
would have erased it). Grid/reach/facet spacing unchanged from the
BedMachine run, so the two are directly comparable.

Residual over s=18-68 km: **81.3 m rms, +31.3 m mean, 216 m |max|, gap
fraction 0.0000** (the +31 m mean is the scout's "BedMachine - radar bed
= -30 m"). Along-track bed roughness (rms about a 5 km running mean):
**28.5 -> 60.3 m** (scout: 33.3 BedMachine / 60.5 picks; our 28.5 is the
32 m-grid-resampled BedMachine, slightly smoother than the scout's native
sampling). 264.6 s sim wall (183/62/20 s), 15 chunks, clamp 0.

| dB rel own surface peak | low | mid | high |
|---|---|---|---|
| sim mid-column (bedmachine -> picked) | -72.2 -> -71.3 | -45.0 -> -44.8 | -44.0 -> -43.7 |
| sim bed window | -48.5 -> -50.4 | -47.0 -> -45.2 | -46.2 -> -44.9 |
| measured bed window | -54.3 | -46.0 | -46.1 |
| sim midcol/bed-peak (scout metric) | -56.6 -> -48.3 | -31.3 -> -18.7 | -28.6 -> -18.5 |
| measured midcol/bed-peak | -28.6 | -4.6 | -3.0 |
| **bed-borne** mid-column | -143.7 -> -141.1 | -126.0 -> -58.1 | -124.2 -> -61.4 |

Altitude trend barely moves (high-low +28.2 -> +27.7 dB sim vs +19.8
measured): the mid-column is still surface-borne at all three altitudes.
The real change is at the BED: the bed-borne mid-column contribution jumps
**+63 to +68 dB** at altitude (still 13-16 dB below the surface-borne
term), the specular bed peak drops so the scout contrast metric closes
8-13 dB of its ~25 dB gap to measured, and the low pass's 5.8 dB-hot bed
window drops to 3.9 dB hot. Radargrams: the smooth BedMachine bed band is
replaced by a dense field of overlapping hyperbolae across the whole
segment, the bed envelope now follows the measured bed (by construction),
and the post-bed tail matches measured within a few dB instead of falling
10-20 dB short.

**Caveat, do not tune this away.** The residual is constant along the
cross-track normal, so along-track pick detail becomes cross-track RIDGES
out to +-ct. That is the unavoidable consequence of correcting a 2-D DEM
with a 1-D profile, and it makes the added bed roughness perfectly
correlated cross-track (anisotropic). The +63 dB bed-borne mid-column jump
and the arc density should therefore be read as an UPPER bound on what a
truly 2-D bed of the same rms would give.
