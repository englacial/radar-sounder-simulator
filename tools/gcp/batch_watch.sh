#!/usr/bin/env bash
# Print a Batch job's state + task counts whenever they change; exit on a
# terminal state.   tools/gcp/batch_watch.sh JOB [POLL_S]
JOB=${1:?job}; POLL=${2:-30}; prev=""
while true; do
  s=$(gcloud batch jobs describe "$JOB" --location=us-central1 \
        --format="value(status.state,status.taskGroups.group0.counts)" 2>/dev/null \
      || echo describe-failed)
  [ "$s" != "$prev" ] && { echo "$(date -u +%H:%M:%S) $JOB $s"; prev=$s; }
  case "$s" in SUCCEEDED*|FAILED*|CANCELLED*|DELETION*) exit 0;; esac
  sleep "$POLL"
done
