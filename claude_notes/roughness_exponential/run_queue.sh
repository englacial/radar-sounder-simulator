#!/bin/bash
# Exponential-ACF validation runs (sequential: the box is DRAM-bound).
# usage: bash claude_notes/roughness_exponential/run_queue.sh
cd "$(dirname "$0")/../.."
L=claude_notes/roughness_exponential/logs
mkdir -p $L
run() {  # name, command...
  local name=$1; shift
  echo "== $name $(date -u +%FT%TZ)" | tee -a $L/driver.log
  /usr/bin/time -f "wall %e s, maxrss %M kB" "$@" > "$L/$name.log" 2>&1
  echo "   exit $? $(tail -1 "$L/$name.log")" | tee -a $L/driver.log
}
run pilot_smoke_exp_greenland_geikie01_transit uv run python tools/run_basal_clutter.py \
    --config config/experiments/pilot_smoke_exp.yaml --line greenland_geikie01_transit
run pilot_smoke_exp_greenland_westcoast uv run python tools/run_basal_clutter.py \
    --config config/experiments/pilot_smoke_exp.yaml --line greenland_westcoast
run b26_rough_exp uv run python tools/run_b26_comparison.py \
    --rough-runs 40:mcords,40:ar,20:mcords:exponential,20:mcords:gaussian:1 \
    --only firn_N20_rough_mcords_exp,firn_N20_rough_mcords_gfx
