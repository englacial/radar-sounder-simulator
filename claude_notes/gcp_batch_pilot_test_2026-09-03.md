# Cloud Batch pilot test (2026-09-03)

Implements option (a) of claude_notes/gcp_compute_options_2026-09-03.md: a
Cloud Batch task array, one VM per pass, using the chunk cache in `runs/` as
the interface. Branch `worktree-agent-ac931ca9a50a12e0a` (on top of
`antenna-directivity-processing` 234167d). Status: IN PROGRESS -- this note
is updated as the run proceeds so a session restart loses nothing.

## Spend tally (running; $20 budget, $15 stop line)

| item | estimate |
|---|---|
| GCS staging (~5 GB, us-central1 standard, ingress free) | ~$0.10/month |
| (jobs appended below as launched) | |
| **total so far** | **$0.10** |

Launched jobs (delete when done: `gcloud batch jobs delete JOB --location us-central1`):
- `soundersim-sim-pin-20260903` 18:57Z: PIN, 6 pass tasks, n2-highmem-8 Spot
  (~$0.19/h est.), memory 56000 MiB/task (one task per VM), maxRunDuration
  90 min -> worst case 6 x 1.5 h x $0.19 = $1.7; expected ~$0.5.
  FAILED at 18:59Z after ~2 min/task (ModuleNotFoundError simc: the runner
  imports the dev-group git dependency at import time; `uv sync --no-dev`
  was wrong) -- deleted 19:00Z. Spend ~6 VM x 3 min = $0.06.
- `soundersim-sim-pin-20260903b` 19:00Z: same job with full `uv sync` +
  git installed on the VM (commit 360145e). VMs up 60 s after submit, tasks
  started 19:01:31Z, env ready 19:03:40Z (uv install + full sync ~2 min),
  then FileNotFoundError on `bedmachine_*.json` (the DEM/BedMachine
  sidecars are read with Path.read_text, which the recorder did not wrap).
  Deleted 19:04Z. Spend ~6 VM x 4 min = $0.08.
- (main checkout meanwhile added a CHUNK_TRACE_SAMPLES = 7e6 memory guard
  to chunk_rows; ported verbatim in 198c8e3 so keys match. PIN/david/getz
  keep 3 chunks per pass; PIS 0 km passes -> 5, geikie -> 4, westcoast
  p3_2016 -> 7.)
- `soundersim-sim-pin-20260903c` 19:05Z: PIN again with sidecars staged,
  n2-highmem-8 Spot, 6 tasks. Tasks started 19:05:54Z (50 s after submit);
  per task env 8 s, data 3 s, upload 2 s; HAPS chunk 63.5 s vs 31.9 s local
  (2.0x slower per VM: 4 Ice Lake cores, shared DDR4).
- 19:12Z rebased onto 8143bb7 (CHUNK_TRACE_FACETS = 1.1e9: chunks split
  until traces x facets/interface fits; PIN stays 3/pass so job c stays
  valid; getz ~4, david ~7, westcoast ~6). Coordinator's memory model:
  ~50 B per trace-facet pair, peak late in the chunk. Local getz chunk
  observed at 83 GB under the new rule (19:13Z) => 128 GB VMs
  (n2-highmem-16) for the other lines, one chunk per VM.
- **BUG in 8143bb7 (needs the coordinator's attention):** its
  `_chunk_facets_estimate` counts 32 m DEM cells, but the kernel lays facets
  at the pass spacing (rac._n_facets: (32/7.47)^2 = 18 facets per cell), so
  the guard never split the low-altitude passes: my --dry-run under 8143bb7
  gave david/getz 3 chunks/pass, and the local getz run was OOM-killed again
  at 19:14Z (exit 137, 568 s, 83 GB seen). Fixed in 1effcbb (crop extent /
  spacing^2): PIN stays 18 chunks (18/18 cached, rids unchanged); getz 17,
  david 19, pineisland_south 31, geikie 29, westcoast 25 chunks. The cloud
  runs of those lines use 1effcbb; local runs must adopt the same rule (or
  an identical n_chunks) for the keys to match.
- 19:18Z rebased onto fc47ed6 (CHUNK_TRACE_FACETS 4e8; getz at ~1.2e9
  actual pairs still hit 99 GB => >= 80 B/pair). With my estimate fix the
  expected counts are the coordinator's (PIN 0 km 6, getz 0 km 9, david
  low ~16); WITHOUT it, fc47ed6 alone still gives 3 chunks (DEM-cell
  count) and will OOM. PIN job c (3-chunk rids) stays valid against the
  existing local PIN results; the new-chunking PIN run is a second target.

## Environment facts found

- gcloud user login had expired (org reauth); user re-ran `gcloud auth login`.
- Quotas, all US regions: `PREEMPTIBLE_CPUS` 0 (Spot VMs then draw on the
  regular family quota, which is how the July psc-* Spot jobs ran), no
  `C3D_CPUS` entry at all (C3D unusable), `C3_CPUS` 24, `N2_CPUS` 200,
  `C2D_CPUS` 100, `CPUS` 200. => c3d-standard-8 Spot (the study's pick) is
  not available in this project; N2 Spot is the fallback with real
  parallelism (200 vCPU).
- Memory (coordinator, 2026-09-03): greenland_westcoast p3_2016 chunk at
  posting_div 8 OOM-killed locally at 110 GB RSS (7989 samples x ~1780
  traces/chunk). westcoast is EXCLUDED from this test; VMs are sized by
  memory (highmem), one task per VM.
- Pilot pass inventory (`--list-passes`): david 5, getz 5, PIN 6, PIS 6,
  geikie 4, westcoast 5 = 31 passes; `--dry-run` chunk counts: 15/15/18/18/
  12/15 = 93 chunks (3 per pass).

## What was built

- `tools/run_basal_clutter.py --config SPEC --line L`
  - `--simulate-only PASS[:c0,c1] ...`: prep + simulate those chunks into
    runs/ under the run's exact rid + meta, exit before processing.
  - `--dry-run`: prep every pass, print each chunk's rid + cache state
    (`CHUNKS n/N cached`), simulate nothing.
  - `--list-passes`: the resolved pass order, no data touched.
  - `run_altitude_comparison.chunk_cached()` = the cache-hit rule, shared by
    `run_level` and the manifest.
- `tools/gcp/stage_bundle.py`: records the files the prep opens
  (xarray/rasterio/numpy readers wrapped) and uploads them + a per-line
  manifest, no-clobber.
- `tools/gcp/batch_task.sh`: the VM-side task (uv install, repo tarball,
  `uv sync -p 3.13` (dev group too: the runner imports simc), line inputs from the GCS mount, copy-in of
  earlier results, run, mirror new outputs/ files back, timing json).
- `tools/gcp/batch_launch.py`: builds tasks.txt (one pass per task or one
  line per task in `--mode process`), uploads `git archive HEAD`, submits
  the job JSON (GCS volume mount, Spot, maxRunDuration, retries), `--wait`.
- `tools/gcp/batch_sync.sh JOB [DEST]`: rsync results into a local outputs/.

## Cache-hit check (local baseline)

The 18 local PIN chunks (main checkout) copied into this worktree's
`outputs/antarctica_pineisland_north/pilot/runs/` -> `--dry-run` reports
`CHUNKS 18/18 cached`: the meta key carries no host paths, so a chunk
simulated elsewhere hits `[skip-exists]`.

- `soundersim-sim-4lines-20260903` 19:25Z: getz+david+PIS+geikie at
  fc47ed6 + estimate fix, 178 chunk tasks (21+32+62+63), n2-highmem-8
  Spot, memory 56000 MiB (one chunk per VM), parallelism 22, maxRunDuration
  60 min. Projection ~9 VM-h x ~$0.28 = ~$2.6. Progress: 21/178 at
  19:49Z (~0.9 tasks/min at the 4-VM level Batch actually reached).
- `soundersim-sim-pinwc-20260903` 19:50Z: PIN at the 6-chunk rule (27
  chunk tasks) + westcoast (49), same VM policy. Projection ~3.7 VM-h
  ~$1.0. Shares the 8-IP cap with the job above.

Tally at 19:50Z: spent ~$0.6 (two failed PIN launches $0.15, PIN job c
$0.4, storage); committed/projected ~$3.6 more => ~$4.2 total, under the
$15 stop line.
- 20:19Z: user created the Cloud NAT (soundersim-nat on
  soundersim-nat-router, us-central1) + Private Google Access. Deleted the
  two IP-capped jobs (4lines at 56/178 done after 54 min = ~3.6 VM-h ~$1.0;
  pinwc had not started) and relaunched with `--no-external-ip`:
  `soundersim-sim-4lines-b-20260903` (178 tasks, 16 VMs, RESULTS_FROM the
  old job so its 56 chunks are copied in and skipped) and
  `soundersim-sim-pinwc-b-20260903` (67 tasks, 8 VMs). Both launchers
  wait and run `nat.py down` in a finally (the last one to finish deletes).
  NAT cost ~$0.044/h while up.
- getz verification (local DONE 20:1xZ, exit 0, 2506 s): the 7 cloud chunks
  available before the relaunch (dc8_2016_0km c00-c02 of 9, dc8_2016_11km
  c00-c02, haps_14km_lambda c02) are meta== and max|diff| = 0 on
  field/nadir/twtt; cloud/local wall 186-262 / 80-140 s (2.0-2.5x).
- 20:25Z: the no-external-IP jobs run through the NAT (env 72-82 s incl.
  uv install + full sync via NAT; earlier chunks copied in and skipped,
  run 9-11 s) but are STILL 4 VMs: Batch now reports `Quota
  'CPUS_ALL_REGIONS' exceeded. Limit: 32.0 globally` -- a project-wide cap
  (4 x 8 vCPU) that the regional N2_CPUS=200 sits under. The IP cap was
  never the binding one for 8-vCPU VMs. Raising CPUS_ALL_REGIONS (console
  quota request) is the only way to more parallelism; the running jobs
  pick a raised quota up automatically. Continuing at 4 VMs (cost is the
  same; ETA ~23:10Z for everything).
- 20:30Z: user raised CPUS_ALL_REGIONS to 256; the b jobs scaled to 12 VMs
  and hit the next cap, `SSD_TOTAL_GB 500/region` (40 GB pd-balanced boot
  disks). Launcher now defaults to 30 GB pd-standard boot disks. Stopped
  the b launchers (so their finally could not delete the NAT), deleted
  the b jobs (4lines-b at 32/178 + 5 running after 22 min; pinwc-b at
  12/67 + 4 running; ~4 VM-h ~$1.1) and relaunched at 20:42Z:
  `soundersim-sim-4lines-c-20260903` (24 VMs) and
  `soundersim-sim-pinwc-c-20260903` (8 VMs), RESULTS_FROM all earlier
  jobs so finished chunks are skipped. 32 VMs = 256 vCPU.
- 20:44Z: 24 VMs running; next cap `INSTANCES 24/region`, so pinwc-c
  waited (re-queued by Batch) until the four-line job drained.
- **`soundersim-sim-4lines-c-20260903` SUCCEEDED 21:06Z: 178/178 in 1449 s
  (24 min) wall from submit**, 22-24 concurrent VMs; the launcher's
  teardown correctly left the NAT up (pinwc-c still active). Cloud
  chunk sets complete: getz 21/21, david 32/32, PIS 62/62, geikie 63/63.
- getz full verification (local DONE 20:1xZ): **21/21 chunks meta== and
  max|diff| = 0** on field/nadir/twtt (claude_notes/logs/gcp_compare_getz.log).
- 21:08Z: `soundersim-proc-4lines-20260903` (process mode, one task per
  line, 4 VMs) launched for getz/david/PIS/geikie; `soundersim-sim-pinwc-c`
  running on 8 VMs. Process tasks hit the cache on every chunk
  (`[skip-exists]` x all, `[ok]` x 0, no errors).
- david cloud processing done 21:19Z (env 38 s, data 7 s, run 449 s,
  upload 7 s); cloud kernel wall summed over its 32 chunks 6512 s
  (basler_2017 4210 s over 16 chunks). Cloud metrics (UNVERIFIED: david
  not yet DONE locally): clutter_*/sim/midcol_rel_surf_db basler_2017
  -55.80, baslermkb_2022 -55.71, baslermkb_2023 -55.72, haps_halflambda
  -48.83, haps_lambda -51.78 dB; bed_visibility halflambda 8.92, lambda
  14.35 dB.
- **`soundersim-sim-pinwc-c-20260903` SUCCEEDED 21:40Z: 67/67 in 3422 s
  (57 min incl. ~25 min queued behind the INSTANCES cap)**; cloud chunk
  sets now complete for all six lines: PIN 27/27 (6-chunk rule),
  westcoast 49/49. NAT again correctly left up (proc job active).
- **getz end-to-end in the cloud**: the process task's metrics.json vs the
  local run: 377 shared scalar keys, 0 differ (e.g. clutter_dc8_2016_0km
  midcol -67.14 / -67.14, dc8_2016_9km -43.06 / -43.06, dc8_2016_11km
  -41.73 / -41.73, haps halflambda -47.11 / -47.11, lambda -47.89 /
  -47.89 dB; bed_visibility halflambda 5.79 / 5.79, lambda 34.84 / 34.84
  dB); radargrams.png byte-identical (md5 60dbaecbbfbccf60c562c12b4ae74fe5).
- 21:44Z: `soundersim-proc-pinwc-20260903` (process PIN + westcoast, 2
  tasks) launched.

## PIN job c (3-chunk rids) -- SUCCEEDED 19:25:38Z, 20.5 min wall

Timing (task json + chunk wall_s): submit 19:05:0xZ -> tasks running
19:05:54Z; only 4 Spot VMs were granted so tasks 3-5 ran sequentially on one
VM. Per task: env 0-13 s (uv install + full sync; 0 when the VM already had
it), data 1-3 s, upload 1-2 s. Simulation per chunk, cloud / local:
dc8 0 km 363-386 / 200-236 s (1.8x), dc8 9 km 72-74 / 29-30 s (2.4x), HAPS
60-64 / 31-32 s (2.0x). Per-pass task wall: 0 km 1125-1149 s, 9 km 222 s,
HAPS 190-202 s. Spend: 4 VMs x ~20 min = 1.3 VM-h x ~$0.28 = ~$0.4.

Verification vs the local run (main checkout outputs/antarctica_pineisland_north/
pilot/runs, tools/gcp/compare_runs.py, log claude_notes/logs/gcp_compare_pin_c.log):
all 18 chunks `meta==` (cloud meta_key == local meta_key, i.e. the runner's
cache-hit rule holds) and **max |diff| = 0.0 on field, nadir_twtt and twtt**
-- bit-identical between the Zen 5 box and Ice Lake N2 VMs (same jaxlib
0.10.2 CPU wheel, XLA:CPU deterministic here).

Rids compared (all `<pass>_pilot_pbed_dgn_rssnr_proc_c0{0,1,2}_srough_sr0.0655_1.67_exp_att13.03_brough0.1_0.886_pdiv8_fs0.5_s03_n1_gfx0.05_<inst>[_wchirp]`):
dc8_2014_0km, dc8_2016_0km, dc8_2018_0km (`_ia8034a068`), dc8_2012_9km
(`_iac8e5ee53`), haps_14km_halflambda (`_ia62093eda_wchirp`),
haps_14km_lambda (`_iac6d87c48_wchirp`); 3 chunks each = 18.

Runner cache-hit on the copied cloud chunks: the normal runner (worktree,
3-chunk budget forced via claude_notes/logs/process_pin_3chunk.py, cloud
npz+json dropped into outputs/antarctica_pineisland_north/pilot/runs/)
printed `[skip-exists]` for every chunk and `[ok]` (re-simulation) for none;
it went straight to focusing/analysis/figures (log
claude_notes/logs/gcp_process_pin_local.log).

Metrics (cloud chunks -> local processing step, vs the main checkout's
metrics.json): 449 shared scalar keys, **0 differ** (wall-time keys
excluded). Spot checks, cloud / local:

| key | cloud | local |
|---|---|---|
| clutter_dc8_2014_0km/sim/midcol_rel_surf_db | -64.01 | -64.01 |
| clutter_dc8_2016_0km/sim/midcol_rel_surf_db | -64.66 | -64.66 |
| clutter_dc8_2018_0km/sim/midcol_rel_surf_db | -64.38 | -64.38 |
| clutter_dc8_2012_9km/sim/midcol_rel_surf_db | -22.54 | -22.54 |
| clutter_haps_14km_halflambda/sim/midcol_rel_surf_db | -44.60 | -44.60 |
| clutter_haps_14km_lambda/sim/midcol_rel_surf_db | -45.67 | -45.67 |
| haps_14km_halflambda_bed_visibility | -8.10 | -8.10 |
| haps_14km_lambda_bed_visibility | 14.82 | 14.82 |

radargrams.png: byte-identical (md5 2488e5ce35f15dc223ff7aa27205b6e0 both).

## Job soundersim-sim-4lines-20260903 (getz/david/PIS/geikie, 178 chunk tasks)

Launched 19:25Z at the ddfa899-equivalent chunking (getz 9 3 3 3 3, david
16 5 5 3 3, PIS 17 18 18 3 3 3, geikie 33 24 3 3). Batch reported
`CODE_GCE_QUOTA_EXCEEDED: Quota 'IN_USE_ADDRESSES' exceeded. Limit: 8.0 in
region us-central1` -- the fan-out is capped at ~8 VMs by external IPs, not
CPUs. Lifting it needs VMs without external IPs = Private Google Access +
Cloud NAT on the default subnet (launcher now has `--no-external-ip`);
creating those was blocked by the permission classifier in this session and
is left to the user (or an IN_USE_ADDRESSES quota increase). Running at
the cap (option c): ~4 min per heavy chunk, 3-4 concurrent at 19:33Z.
