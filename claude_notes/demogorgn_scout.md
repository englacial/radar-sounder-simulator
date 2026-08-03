# DEMOGORGN-Antarctica as a bed source for the basal-clutter study (2026-08-03)

Scout + prototype only. No `src/` or `tools/` changes. Deliverables: this note,
one cached realization window (`outputs/cache/demogorgn_antarctic_seed000_*.tif`
+ `.json`), and `outputs/demogorgn_scout/demogorgn_assessment.png`. Scripts in
the session scratchpad (`fetch_demogorgn.py`, `assess_demogorgn.py`,
`datum_check.py`, `final_table.py`).

**Verdict: it covers our segment, it is CONDITIONED on our flight line, and it
carries the isotropic 2-D texture that both current bed options lack. Adopt it
as the base bed.** Details and the two things that must be got right (geoid
datum, 43 m nadir misfit) below.

## 1. What DEMOGORGN is

DEMOGORGN ("Digital Elevation Models of Geostatistical ORiGiN") is a
**100-member ensemble of geostatistically simulated Antarctic bed topographies**
on the **Bedmap3 500 m grid**, from the Gator Glaciology group (Emma MacKie's
lab, U. Florida) / "Englacial". It is explicitly a *stochastic* product: rather
than one smoothest-fit bed (BedMachine/Bedmap3), it draws many equally-probable
beds that (a) honour the radar measurements exactly where they exist and (b)
carry geologically realistic roughness in between.

Method, per the README and the group's blog:

* **Sequential Gaussian Simulation (SGS)** conditioned on ice-penetrating radar
  bed measurements, using a **Matern variogram (range ~140 km, sill 0.85,
  smoothness 0.66)** with 50 nearest neighbours and a 30 km search radius, after
  a normal-score transform.
* A **geostatistical MCMC mass-conservation bed inversion** is merged in over
  fast-flowing ice (100 MCMC realizations replace the SGS bed inside a
  high-velocity mask), with SGS run over a 5 km buffer to blend the seam.
* Self-described as a **"living data product"** — expect the ensemble to change
  under you. Pin the icechunk snapshot (below).
* README's own caveat, quoted from the access notebook: *"Simulations are not
  postprocessed and have some issues."*

It is the natural successor to MacKie et al. 2020 (JGR Earth Surface,
doi:10.1029/2019JF005420), which pioneered simulated-roughness Antarctic DEMs
for subglacial-lake prediction.

## 2. Access recipe (works, anonymous, no credentials)

The 71 GB netCDF the README points at is **not** what you want — an
**icechunk/zarr store now exists** in the same bucket (the README predates it
and still says "icechunk is still in progress"), and it supports windowed reads.

```python
import icechunk, zarr
st = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="englacial/demogorgn/icechunk/realizations.icechunk",
    region="us-west-2", anonymous=True)          # no credentials needed
repo = icechunk.Repository.open(st)
g = zarr.open_group(store=repo.readonly_session("main").store, mode="r")
# g["realizations"] (100, 13334, 13334) float32, chunks (100, 128, 128)
# g["x"] (13334,)  -3333250 .. +3333250, step +500
# g["y"] (13334,)  +3333250 .. -3333250, step -500   (y DESCENDING)
# g["seed_id"] (100,) int64, 0..99
bed = g["realizations"][0, r0:r1, c0:c1]         # window read
snapshot = repo.lookup_branch("main")            # PIN THIS: "living data product"
```

`icechunk` (2.1.2) and `zarr` (3.3.0) are **already in the project env** — no new
dependencies needed.

Store contents (`https://data.source.coop/englacial/?list-type=2&prefix=demogorgn/`):

| object | size | note |
|---|---|---|
| `demogorgn/realizations.nc` | **71.1 GB** | the monolithic netCDF the README documents; avoid |
| `demogorgn/antarctica_fast_flow_mask_50m.nc` | 3.72 GB | the MCMC / mass-conservation mask |
| `demogorgn/icechunk/realizations.icechunk/` | — | **use this** |
| `demogorgn/icechunk/antarctica_fast_flow_mask_50m.icechunk/` | — | mask, same trick |

**Gotcha:** the mask is named `..._50m` but its grid is **450 m** posting
(12445 x 12445, x -2 800 000 .. +2 799 800). It has `fast_flow_mask` (int 0/1)
plus 2-D `lat`/`lon`. Don't trust the filename.

**Metadata is absent.** The zarr arrays carry **no attrs at all** — no CRS, no
units, no vertical datum, no variable description. Everything in section 3 was
determined empirically.

### Cost

The array chunk is `(100, 128, 128)` — **all 100 seeds live in every spatial
tile**, so reading one realization costs the same as reading all of them (which
is why the conditioning test below was free). Our 60.4 x 65.9 km window touches
**4 zarr chunks**; store objects are ~4.5 MB median, so order **10–20 MB
transferred, 5.1 s wall** on a cold read. The cached GeoTIFF is **49 kB**
(deflate, 134 x 122 cells). The whole 100-member window is 6.5 MB in RAM.

## 3. Facts our pipeline needs

| property | value | how established |
|---|---|---|
| CRS | **EPSG:3031** | x/y span ±3 333 250 on the Bedmap3 grid |
| posting | **500 m** | median `diff(x)` |
| grid registration | cell centres at ...250 / ...750 | **staggered 250 m from BedMachine's** ...000/...500 grid — do NOT assume index alignment |
| **vertical datum** | **EIGEN-6C4 GEOID** (same as BedMachine's *native* `bed`) | see below |
| coverage | whole continent; our window 100 % finite, no nodata | |
| realizations | 100, `seed_id` 0..99 | |
| y axis | **descending** (north-up already) | |

### Vertical datum — the one thing that will silently ruin everything

DEMOGORGN publishes **geoid-referenced** elevations, exactly like BedMachine's
raw `bed` variable, and therefore needs the **same `+ geoid` conversion**
`opr.fetch_bedmachine_window` already applies before it can enter our
WGS84-ellipsoidal scene stack.

Established over a **400 x 400 km box** around the segment (640 000 cells),
median DEMOGORGN − BedMachine:

| BedMachine mask class | n | vs BM **native** (geoid) | vs BM **ellipsoidal** |
|---|---|---|---|
| ocean | 151 669 | **−0.1 m** | +35.1 m |
| floating ice | 103 680 | **+0.2 m** | +36.5 m |
| grounded ice | 384 287 | **+1.7 m** | +36.2 m |
| ALL | 640 000 | **−0.1 m** | +35.4 m |

Over ocean DEMOGORGN inherits BedMachine bathymetry verbatim, so the −0.1 m
agreement in the native datum is conclusive; the +35 m in the ellipsoidal
column is just the EIGEN-6C4 geoid (**−35.3 m** over our segment, range
−36.0 .. −35.0, i.e. effectively constant at our scale). Independent
confirmation from the radar picks: raw DEMOGORGN − picks = **+44.7 m** mean /
+38.8 median, but **+geoid** it becomes **+9.4 mean / +3.4 median** — nearly
unbiased against a direct measurement.

Note the grounded-ice median is +1.7 m regionally but **+44.7 m on our
segment**: DEMOGORGN departs from BedMachine locally *in the direction of the
radar*, because it honours picks that BedMachine smooths away (BedMachine −
picks = −29.1 m here).

## 4. Is it conditioned on OUR line? — Yes, definitively

The 100-member ensemble spread is the test SGS gives you for free: variance goes
to zero at hard data and rises to the sill away from it.

**Ensemble standard deviation vs distance from the anchor flight line:**

| distance from line | median ensemble sd | frac. of cells with sd < 2 m |
|---|---|---|
| **0 – 0.3 km** | **0.5 m** | **1.00** |
| 0.3 – 0.7 km | 14.6 m | 0.42 |
| 0.7 – 1.5 km | 25.2 m | 0.30 |
| 1.5 – 3.0 km | 38.0 m | 0.32 |
| 3 – 8 km | 30–52 m | — |
| window overall | 25.4 m (mean) | 0.28 |

At nadir the ensemble sd is **0.5 m mean / 0.8 m median** against a **25.4 m**
window-wide mean. The line is a conditioning datum. This was checked **inside
and outside the MCMC fast-flow mask separately** (both give sd = 0.5 m and
frac<2m = 1.00 in the 0–0.3 km bin), so it is genuine radar conditioning, not an
artifact of the deterministic mass-conservation patch.

**Consequences, both good:**

* **The seed choice is nearly irrelevant at nadir.** Across all 100 seeds the
  nadir bias spans +44.5 .. +45.3 m, the misfit-to-picks rms 61.5 .. 62.2 m, and
  the along-track roughness 49.9 .. 50.5 m. A single realization is genuinely
  representative — the user's "one realization is fine" is well founded.
  **Seed 0 (`seed_id = 0`, the first) was fetched**, deterministically.
* Seeds differ only **off-nadir**, i.e. exactly where they should, and exactly
  where the study needs plausible variability. A future ensemble study over
  clutter uncertainty is cheap (the chunks already contain all 100).

### But it does NOT reproduce our picks exactly

DEMOGORGN(+geoid) − picks over s = 18–68 km: **mean +9.4 m, median +3.4 m,
sd 42.8 m, rms 43.7 m**. So the ensemble collapses onto *a* bed value at our
line that differs from *our* pick-derived bed by ~43 m rms.

Where the misfit lives (de-biased): **25.6 m at > 5 km wavelengths, 32.4 m at
< 5 km**; the high-pass regression slope of DEMOGORGN on the picks is **0.703**
with **r = 0.844**, i.e. DEMOGORGN carries ~70 % of the picks' fine-scale
amplitude, correlated but not identical. Smoothing the picks doesn't close it
(sd 42.7 → 39.9 at 500 m → 36.5 at 2 km), so it is **not** just the 500 m
band limit.

Most likely cause: the conditioning data are the Bedmap3-ingested version of
this survey — CReSIS's own released L2 thickness with **its** firn correction
and permittivity, possibly a different pick vintage, binned to 500 m cells with
crossing lines averaged — whereas our picks are `(Bottom − Surface)·c/(2√3.15)`
with **no firn correction**. The scout already documented a 6 % spread in
measured thickness across the three passes of this same line (683 / 644 /
643 m); 43 m on a 683 m column is 6.3 %. **Do not tune this away.** It is an
honest disagreement between two thickness conventions, and it is the reason the
hybrid option in section 6 exists.

## 5. The assessment numbers

Segment s = 18–68 km (3365 anchor traces, 14.85 m posting, zero pick gaps).
"Roughness" = rms residual about a 5 km running mean (the study's metric).
Along-/cross-track (AT/CT) columns are averaged over a ±12 km, 250 m-sampled
swath in track coordinates; the nadir column is at the native 14.85 m posting.

| bed source | nadir bias (m) | nadir rms vs picks (m) | AT rough (m) | **CT rough (m)** | 2-D rough (m) | **AT/CT** | slope med (deg) |
|---|---|---|---|---|---|---|---|
| **BedMachine v3** | −29.1 | 80.7 | 26.5 | 30.8 | 39.8 | 0.86 | 2.89 |
| **picked-bed** (BM + 1-D resid) | 0.0 | **0.0** | 58.5 | **30.8** | 65.6 | **1.90** | 5.24 |
| **DEMOGORGN seed 0** | +9.4 | 43.7 | 43.0 | **41.9** | 55.2 | **1.03** | 4.85 |
| DEMOGORGN + 1-D resid (hybrid) | 0.0 | **0.0** | 53.2 | 41.9 | 63.4 | 1.27 | 6.20 |
| radar picks (nadir only) | 0.0 | 0.0 | 60.5 | — | — | — | — |

Nadir along-track roughness at native posting — **the study's headline numbers**:

| | roughness rms | % of the resolvable target |
|---|---|---|
| radar picks (14.85 m posting) | **60.5 m** | — |
| picks band-limited to 500 m (**the honest target for a 500 m product**) | **57.5 m** | 100 % |
| **DEMOGORGN seed 0** | **50.3 m** | **88 %** |
| BedMachine v3 | 28.5 m | 50 % |

(The scout's reference values were 33.3 / 60.5 m from native BedMachine
sampling; 28.5 m is the grid-resampled BedMachine the tool reports. Only 9.2 m
rms of the picks' detail lives below 500 m, so the 500 m band limit is a minor
penalty — the real gap was always BedMachine's interpolation smoothing.)

Other nadir statistics: Pearson r vs picks **+0.960 raw / +0.844 on the 5 km
high-pass** for DEMOGORGN, vs **+0.866 / +0.552** for BedMachine.

### Reading the table — this is the point of the exercise

1. **DEMOGORGN nearly doubles BedMachine's roughness (28.5 → 50.3 m) and
   recovers 88 % of what a 500 m product can possibly recover.** The scout's
   "BedMachine reproduces only 55 % of the radar-pick roughness" complaint is
   essentially resolved.
2. **It is the only option with realistic CROSS-track texture.** CT roughness
   30.8 m (BedMachine) → **41.9 m**, and — decisively — the picked-bed option's
   CT roughness is **exactly BedMachine's 30.8 m**, because adding a constant
   along the cross-track normal cannot change the cross-track residual at all.
   The `--picked-bed` run buys nothing off-nadir; its extra roughness is
   perfectly correlated cross-track. This is visible as the identical
   blue/dashed-green curves in the bottom panel of the figure.
3. **DEMOGORGN is isotropic (AT/CT = 1.03); picked-bed is anisotropic
   (1.90).** The pilot findings flagged the picked-bed arcs as "an UPPER bound
   on what a truly 2-D bed of the same rms would give" — DEMOGORGN *is* that
   truly 2-D bed, so it should settle the question.
4. **Median bed slope 2.89° → 4.85°**, the quantity that actually drives
   off-nadir specular returns. Expect materially more bed-borne clutter than the
   BedMachine run, and a more honest (probably lower and more diffuse) version of
   the picked-bed run's +63 dB bed-borne mid-column jump.
5. Sanity: DEMOGORGN(+geoid) never approaches the REMA surface anywhere in the
   ±12 km window — **min clearance 232 m, zero clamp cells**, median implied
   thickness 881 m. `bed_clamp` should stay at 0.

## 6. Integration plan for a `--demogorgn-bed` option

Slots in next to `--picked-bed` with no change to the reach/facet/gamma
machinery — it only swaps the DEM that `prep_pass` resamples onto the 32 m
scene grid.

**(a) `src/soundersim/opr.py` — new `fetch_demogorgn_window(bounds, pad_m,
seed=0, snapshot=None)`**, a near-clone of `fetch_bedmachine_window`:

* Same signature and return `(bed, transform, crs, meta)`; same GeoTIFF +
  JSON-sidecar cache under `outputs/cache/`, named
  `demogorgn_antarctic_seed{NNN}_{key}.tif` (the prototype already wrote one in
  this layout — reuse or delete it).
* Hash the cache key over `(bucket, prefix, snapshot, seed, rounded proj
  bounds)` so a store update or a seed change forces a refetch.
* Window-select on `g["x"]`/`g["y"]` exactly as the BedMachine path does
  (`>= x0 - step`, `<= x1 + step`); y is already descending, so
  `rasterio.transform.from_origin(x[c0] - step/2, y[r0] + step/2, 500, 500)`.
* **Record the snapshot id** in the sidecar (`repo.lookup_branch("main")`); this
  is a "living data product" and a silent update would invalidate cached runs.
  Optionally accept a pinned snapshot and open that instead of `main` — the
  RSSNR feature already set the precedent for pinning an icechunk snapshot
  (`3YH47013745B2T5ZZR50` in `claude_notes/required_snr_dataset.md`).

**(b) The geoid — the one genuine plumbing change.** DEMOGORGN ships no geoid,
and `fetch_bedmachine_window` currently caches only the *sum* `bed + geoid`, so
the geoid field isn't recoverable from cache. Cheapest honest fix: **write the
geoid as band 2 of the existing BedMachine GeoTIFF** (it is already read into
memory at line ~251; bump `count=1` to `count=2`, add
`"geoid_band": 2` to the sidecar) and have the DEMOGORGN path call
`fetch_bedmachine_window` for the same bounds and add band 2. That keeps one
EIGEN-6C4 source for the whole repo and preserves the documented datum note. It
does invalidate the 20 cached BedMachine tifs (they'd refetch once, or gate on
`src.count`). A constant −35.3 m would be accurate to ±0.5 m over this segment
but is not defensible elsewhere and shouldn't be committed.

**(c) `tools/run_basal_clutter.py`** (another agent is in this file — this is a
proposal, nothing was edited):

* `--demogorgn-bed [--demogorgn-seed N]`, default seed 0. Cache/output suffix
  `_dgn`, composed like the existing tags in `case_tag()` so BedMachine and
  picked-bed runs stay cached and byte-comparable.
* In `prep_pass`, swap the bed source before `resample_to_grid` — the existing
  bilinear resample handles the 250 m grid stagger correctly (it is a proper
  map-referenced warp, not an index copy). Facet spacing, reaches, fast-time
  grid and `gamma_maps` are all untouched, so a `_dgn` run is directly
  comparable to `full/` and `full_pbed/`.
* **It should compose with `--picked-bed`**, and that combination is worth
  running: the residual to apply drops from **81.3 m rms to 43.7 m rms**, so the
  cross-track ridge artifact is roughly halved in amplitude while the underlying
  texture is isotropic (hybrid row: AT/CT 1.27 vs picked-bed's 1.90). That gives
  a bed that matches the picks exactly at nadir *and* has plausible off-nadir
  structure — the "best of both" the task was after. `PICKED_BED_NOTE` would
  need a sentence saying the base is DEMOGORGN and quoting the smaller residual.
* Recommended run set: `--demogorgn-bed` alone (clean, isotropic, physically
  self-consistent, nadir off by 43 m) and `--demogorgn-bed --picked-bed`
  (nadir-exact hybrid). Comparing the two brackets the nadir-registration
  question the way BedMachine-vs-picked-bed brackets the roughness question.

**(d) Provenance/config** to record per run: store URL, icechunk snapshot,
`seed_id`, posting, the geoid conversion, the "living data product" warning, the
absent license, and the 43.7 m nadir misfit with its likely cause.

## 7. License and citation — UNRESOLVED, must ask before publishing

* **No license anywhere.** No `LICENSE` file in the GitHub repo, GitHub API
  reports `license: null`, and the source.coop product metadata carries
  `"data_mode": "open"` and `"tags": []` with **no license field**. The data are
  publicly readable anonymously, but "public" is not "licensed".
* **No citation guidance and no paper.** The README points only at the blog
  (https://www.gatorglaciology.com/demogorgn) "for information on DEMOGORGN
  updates". No DOI, no preprint found. The closest citable methodological
  ancestor is MacKie et al. 2020, doi:10.1029/2019JF005420, but that is not this
  dataset.
* **Action:** contact the Gator Glaciology group / Englacial
  (https://source.coop/englacial) for a license and a preferred citation before
  any of this leaves the repo. Fine for internal engineering work meanwhile.

## 8. Open questions

1. **Is the MCMC mass-conservation patch applied in the published ensemble over
   our window?** 67.5 % of the window and **79 % of our nadir traces** fall
   inside the fast-flow mask. Inside it, 54 % of cells have ensemble sd < 2 m
   (vs 14 % outside) — consistent with a largely *deterministic* MCMC bed there
   rather than a stochastic one. If so, the "ensemble" in our study area is
   partly a single inversion, and the off-nadir texture over the fast-flow
   fraction is MCMC-derived, not SGS-derived. Worth confirming with the authors.
   (It does not change the conditioning result — that was verified inside and
   outside the mask separately.)
2. **What exactly is the 43.7 m nadir misfit?** Resolvable by comparing our
   pick-derived bed against the Bedmap3 point data for `20161105_05` directly —
   if DEMOGORGN reproduces Bedmap3's version of this line, the misfit is purely
   a thickness-convention difference and the hybrid option is clearly the right
   default. Worth an hour.
3. **Which one is the study's "best model"?** DEMOGORGN alone is physically
   self-consistent but 43 m off at nadir; the hybrid is nadir-exact but
   reintroduces a (halved) anisotropic artifact. The RSSNR gamma work assumed a
   nadir-exact bed; check whether the acceptance correlations (r = 0.76–0.84)
   survive on a DEMOGORGN-alone bed.
4. **"Living data product"** — the store may change without notice. Pin the
   snapshot (`WG801625MG778C4DS6Y0` at the time of writing) in any committed
   config.
5. **500 m is still 500 m.** DEMOGORGN fixes the *amplitude* of the roughness
   and its *2-D isotropy*, not the resolution. At the 47–82 m facet spacings the
   high and 30 km passes want, sub-500 m bed texture remains unresolved and
   interpolated. The scout's caveat is reduced, not removed.
6. The README's own "simulations are not postprocessed and have some issues" is
   unexplained. Nothing pathological turned up in our window (100 % finite, no
   clamp, sane slopes), but it warrants a wider sanity sweep if the product gets
   used beyond this segment.

## 9. What was cached

| path | size | contents |
|---|---|---|
| `outputs/cache/demogorgn_antarctic_seed000_b6126fb573a5.tif` | 49 kB | seed 0, 134 x 122 cells @ 500 m, EPSG:3031, **raw geoid-referenced values** (nodata −9999) |
| `outputs/cache/demogorgn_antarctic_seed000_b6126fb573a5.json` | 957 B | store URL, snapshot `WG801625MG778C4DS6Y0`, seed, posting, bounds, datum warning |
| `outputs/demogorgn_scout/demogorgn_assessment.png` | 578 kB | map comparison + nadir profiles + ensemble spread + along/cross-track roughness |

Window: EPSG:3031 x −1 497 198 .. −1 436 810, y −846 326 .. −780 460
(**60.4 x 65.9 km**) — the s = 18–68 km segment with a 5 km along-track margin
and ±15 km cross-track (the 30 km pass's ±11.24 km reach plus 3 km).

**The cached GeoTIFF holds RAW (geoid-referenced) values.** Add the geoid before
use. This deliberately mirrors BedMachine's raw netCDF rather than
`fetch_bedmachine_window`'s converted output, so the conversion stays explicit
and in one place; if `fetch_demogorgn_window` is implemented per section 6 it
should convert on read and say so in the sidecar, as the BedMachine path does.
