# Cloud Batch fan-out for the basal-clutter simulator

The chunk cache in `outputs/<line>/<exp>/runs/` is the interface: a Batch
task simulates one pass's chunks under the run's exact rid + meta, and a later
normal run of the same spec hits `[skip-exists]` on every copied chunk and
only does focusing/analysis/figures. Nothing about the physics or cache keys
changes; `run_level`'s hit rule is a pure meta equality with no host paths.

## Prerequisites

- `gcloud` authenticated on project `ice-infrastructure` (Batch, Compute and
  Storage APIs enabled); the bucket `gs://ice-infrastructure-soundersim`.
- A committed tree: the launcher ships `git archive HEAD`.
- Local caches for the lines you stage (`outputs/cache/*.nc|*.tif`,
  `outputs/<line>/rssnr_anchor.npz`) -- the run reads them cache-first.
- Quotas (2026-09-03): `PREEMPTIBLE_CPUS` 0 => Spot VMs draw on the family
  quota (`N2_CPUS` 200, `C2D_CPUS` 100, `C3_CPUS` 24); no C3D quota. The
  binding limits were **`CPUS_ALL_REGIONS` 32 (global; 4 x 8-vCPU VMs)**
  and `IN_USE_ADDRESSES` 8 per region (external IPs; see the NAT section).
  Raise CPUS_ALL_REGIONS before expecting more than four VMs.

## Workflow

```bash
P=gs://ice-infrastructure-soundersim/batch_2026-09-03     # dated prefix

# 1. stage inputs (once per line; records what the prep opens, no-clobber)
uv run python tools/gcp/stage_bundle.py --config config/experiments/pilot.yaml \
    --lines antarctica_pineisland_north antarctica_david --prefix $P

# 2. simulate: chunk tasks (6 chunks per task: one pass preparation per
#    task), one VM per task, spend guarded at the catalog Spot price
uv run python tools/gcp/batch_launch.py --config config/experiments/pilot.yaml \
    --lines antarctica_pineisland_north antarctica_david --per-chunk \
    --chunks-per-task 6 --no-external-ip --max-vms 25 --budget-usd 50 \
    --job soundersim-sim-20260903 --prefix $P --wait

# 3. process in the cloud: one task per line, pulls the chunks of step 2,
#    hits [skip-exists], writes metrics.json + figures
uv run python tools/gcp/batch_launch.py --config config/experiments/pilot.yaml \
    --lines antarctica_pineisland_north antarctica_david --mode process \
    --results-from soundersim-sim-20260903 --job soundersim-proc-20260903 \
    --prefix $P --wait

# 4. fetch (mirror of outputs/) and compare against a local run
tools/gcp/batch_sync.sh soundersim-sim-20260903 outputs_cloud $P
tools/gcp/batch_sync.sh soundersim-proc-20260903 outputs_cloud $P
uv run python tools/gcp/compare_runs.py outputs_cloud outputs \
    --lines antarctica_pineisland_north --exp pilot

# 5. clean up (VMs are Batch-managed and gone with the job; keep the bucket)
gcloud batch jobs delete soundersim-sim-20260903 --location us-central1
```

Or skip step 3: `batch_sync.sh JOB outputs` into the real `outputs/` and run
`run_basal_clutter.py --config SPEC --line L` locally -- every chunk prints
`[skip-exists]`.

`tools/gcp/batch_watch.sh JOB` prints state changes; each task also writes
`results/<job>/timing/task_N.json` (env, data, run, upload seconds) and the
chunk jsons carry `wall_s`.

## VM count, external IPs and the Cloud NAT

External IPs are capped by `IN_USE_ADDRESSES` (8 per region in this project),
so a job with external IPs runs at most ~8 VMs regardless of CPU quota. Pass
`--no-external-ip` to lift that: VMs then reach GCS/Batch/logging through
Private Google Access on the default subnet (free, enabled once and left on)
and the few internet fetches (uv installer, PyPI wheels, the simc git
dependency) through a Cloud NAT `soundersim-nat` on router
`soundersim-nat-router` in us-central1.

The NAT costs ~$0.044/h while it exists (+ $0.045/GB processed, PyPI traffic
only), so its lifetime is tied to the launcher: `--no-external-ip` runs
`nat.py up` (idempotent) before submitting, forces `--wait`, and runs
`nat.py down` in a `finally` when the wait ends or the launcher is
interrupted. `down` deletes the NAT and router unless another soundersim job
is still QUEUED/SCHEDULED/RUNNING in the region (then it prints a warning and
leaves the gateway for that job). Manual control:

```bash
uv run python tools/gcp/nat.py status   # shows a leftover router/NAT
uv run python tools/gcp/nat.py down     # --force to delete while jobs run
gcloud compute routers list --regions us-central1   # must show no soundersim-*
```

## Runner flags added for this

- `--simulate-only PASS[:c0,c1] ...` simulate those chunks into runs/, exit.
- `--dry-run` prep every pass, print each chunk's rid + cache state.
- `--list-passes` the pass order the spec runs on `--line` (no data).

## Notes

- Tasks are idempotent: a retried (preempted) task copies its job's earlier
  results in first and re-simulates only the missing chunks. A grouped task
  publishes each finished chunk (npz, then json) every 60 s and at exit, so
  a preemption or a late failure keeps the chunks already done.
- Budget accounting bills VM lifetime (running VMs x elapsed, accumulated
  every guard tick) or the task records, whichever is larger, so failed and
  preempted attempts count; a job is reserved in the ledger at submit with
  its projection and replaced by the measured cost at the end, so
  concurrent launchers see each other. A budget kill is only reported when
  the deletion is confirmed (`BUDGET_KILL_FAILED` otherwise).
- Chunks the staged manifest marks cached are skipped only when their
  files exist locally, and those files are uploaded to the job's results
  so the process job does not re-simulate them; an all-cached plan submits
  nothing. Manifests are written per experiment (`<line>__<exp>.json`).
- `nat.py down` keeps the gateway whenever the Batch job listing fails
  (unknown is not empty).
- Per-chunk memory at posting_div 8 scales with traces x samples per chunk
  (greenland_westcoast p3_2016 exceeded 110 GB locally); size the VM by
  memory and set `--memory-mib` near the VM's size so Batch never packs two
  tasks on one VM.
- `--per-chunk` builds tasks from the staged `--dry-run` manifest;
  `--chunks-per-task N` (default 6) groups uncached chunks so each task pays
  the pass preparation (frames, DEMs, picks, bed synthesis) once. On the
  2026-09-04 full campaign one-chunk tasks spent 10 min preparing getz for
  every 6 min chunk; grouping by 6 cuts that overhead ~6x.
- Spend: `tools/gcp/pricing.py MACHINE` reads the Cloud Billing catalog
  (cached a week in outputs/gcp/pricing.json, dated fallback when offline).
  The launcher prints a projection (past task records when present, else
  defaults) and, with `--budget-usd CAP`, refuses to submit when the ledger
  total plus the projection exceeds the cap, tallies the running job every
  10 min from its task records and deletes it if the cap is crossed, and
  appends each finished job to outputs/gcp/spend_ledger.json. Estimation
  lessons: claude_notes/gcp_cost_estimation_notes_2026-09-04.md.
- gcloud tokens expire (~12 h under the org's reauth policy): a launcher
  that loses auth mid-wait prints `describe failed (auth?)` and keeps
  retrying, but the guard and sync are blind until `gcloud auth login` is
  run again in a terminal. Run long campaigns right after re-authenticating.
- The older GPU trial files (`vm_bootstrap.sh`, `launch_benchmark_vm.sh`,
  `batch_benchmark_job.json`) are unrelated to this workflow.
