#!/usr/bin/env bash
# Pull a Batch job's results (mirror of outputs/) into a local outputs/ tree.
#   tools/gcp/batch_sync.sh JOB [DEST_OUTPUTS_DIR] [PREFIX]
# After a simulate job, DEST/<line>/<exp>/runs/ holds the chunks and a normal
# `run_basal_clutter.py --config SPEC --line L` hits [skip-exists] on each.
set -euo pipefail
JOB=${1:?job name}
DEST=${2:-outputs}
PREFIX=${3:-gs://ice-infrastructure-soundersim/batch_2026-09-03}
mkdir -p "$DEST"
gcloud storage rsync -r "$PREFIX/results/$JOB/outputs" "$DEST"
gcloud storage cp -r "$PREFIX/results/$JOB/timing" "$DEST/gcp/$JOB/" 2>/dev/null || true
