#!/bin/bash
# B1 validation pilots (sequential: the box is DRAM-bound, fan-out is useless).
# usage: bash claude_notes/roughness_b1/run_pilots.sh <spec> <line...>
cd "$(dirname "$0")/../.."
spec=$1; shift
for line in "$@"; do
  log=claude_notes/roughness_b1/logs/${spec}_${line}.log
  echo "== $spec $line $(date -u +%FT%TZ)" | tee -a claude_notes/roughness_b1/logs/driver.log
  /usr/bin/time -f "wall %e s, maxrss %M kB" \
    uv run python tools/run_basal_clutter.py --config config/experiments/$spec.yaml --line $line \
    > "$log" 2>&1
  echo "   exit $? $(tail -1 "$log")" | tee -a claude_notes/roughness_b1/logs/driver.log
done
