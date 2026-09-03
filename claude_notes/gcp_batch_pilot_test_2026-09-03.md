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

## Timings / verification

(filled in as jobs complete)
