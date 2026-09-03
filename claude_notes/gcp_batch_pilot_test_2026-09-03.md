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
