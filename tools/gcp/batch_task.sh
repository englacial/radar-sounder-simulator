#!/usr/bin/env bash
# One Cloud Batch task of the basal-clutter simulator; runs ON the VM.
# Launched by tools/gcp/batch_launch.py, which sets the environment:
#   GCS_MOUNT  bucket mount point (Batch gcs volume)      PREFIX  dated prefix
#   JOB        this job's name                            CONFIG  spec path
#   RESULTS_FROM  (process mode) space-separated earlier job names whose
#                 results/<job>/outputs/ chunks are copied in first
# Task BATCH_TASK_INDEX reads line N+1 of jobs/$JOB/tasks.txt:
#   simulate LINE PASS[:CHUNK,..] OUTDIR   |   process LINE - OUTDIR
# Everything the run writes under outputs/ (except cache/) is mirrored to
# results/$JOB/outputs/, plus a timing json per task.
set -euo pipefail
M="$GCS_MOUNT/$PREFIX"
J="$M/jobs/$JOB"; R="$M/results/$JOB"
t0=$(date +%s)
read -r mode line pass outdir < <(sed -n "$((BATCH_TASK_INDEX + 1))p" "$J/tasks.txt")
echo "task $BATCH_TASK_INDEX: $mode $line $pass -> $outdir ($(hostname), $(nproc) cpu)"

export HOME=/root PATH="/root/.local/bin:$PATH" UV_CACHE_DIR=/root/uv-cache
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
W=/root/soundersim; mkdir -p "$W"; cd "$W"
# the job's own repo snapshot (a VM may serve several tasks; extract once/job)
if [ "$(cat .job 2>/dev/null || true)" != "$JOB" ]; then
  rm -rf src tools config tests docs; tar xzf "$J/repo.tar.gz"; echo "$JOB" > .job
fi
uv sync -q --no-dev -p 3.13
t1=$(date +%s)

# the line's inputs (manifest from tools/gcp/stage_bundle.py), cache-first
while read -r rel; do
  [ -n "$rel" ] && [ ! -f "$rel" ] && { mkdir -p "$(dirname "$rel")"; cp "$M/data/$rel" "$rel"; }
done < "$M/data/lines/$line.txt"
# chunks already produced for this line (retry after preemption; process
# mode after simulate jobs): drop them into outputs/ so the run skips them
for job in $JOB ${RESULTS_FROM:-}; do
  src="$M/results/$job/$outdir"
  [ -d "$src" ] && { mkdir -p "$outdir"; cp -rn "$src/." "$outdir/"; } || true
done
t2=$(date +%s)

touch .marker
case "$mode" in
  simulate) uv run --no-sync python tools/run_basal_clutter.py --config "$CONFIG" \
              --line "$line" --simulate-only "$pass" ;;
  process)  uv run --no-sync python tools/run_basal_clutter.py --config "$CONFIG" \
              --line "$line" ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac
t3=$(date +%s)

find outputs -type f -newer .marker ! -path 'outputs/cache/*' | while read -r f; do
  mkdir -p "$R/$(dirname "$f")"; cp "$f" "$R/$f"
done
t4=$(date +%s)
mkdir -p "$R/timing"
printf '{"task": %d, "mode": "%s", "line": "%s", "pass": "%s", "host": "%s", "env_s": %d, "data_s": %d, "run_s": %d, "upload_s": %d}\n' \
  "$BATCH_TASK_INDEX" "$mode" "$line" "$pass" "$(hostname)" \
  $((t1 - t0)) $((t2 - t1)) $((t3 - t2)) $((t4 - t3)) > "$R/timing/task_$BATCH_TASK_INDEX.json"
echo "TASK_DONE env $((t1 - t0)) data $((t2 - t1)) run $((t3 - t2)) upload $((t4 - t3)) s"
