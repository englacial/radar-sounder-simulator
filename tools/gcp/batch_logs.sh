#!/usr/bin/env bash
# Recent task log lines of a Batch job (Cloud Logging), oldest first.
#   tools/gcp/batch_logs.sh JOB [LIMIT] [FRESHNESS] [GREP_REGEX]
JOB=${1:?job}; LIMIT=${2:-100}; FRESH=${3:-2h}; RE=${4:-.}
gcloud logging read "logName:\"batch_task_logs\" AND labels.task_group_name:\"/jobs/$JOB/\"" \
  --limit="$LIMIT" --freshness="$FRESH" --format="value(timestamp,textPayload)" \
  2>/dev/null | tac | grep -E "$RE"
