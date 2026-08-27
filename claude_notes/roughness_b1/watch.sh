#!/bin/bash
# emit driver.log exit lines as they appear; stop when $1 exits have accumulated
cd "$(dirname "$0")/../.."
want=${1:-4}
n0=$(grep -c "exit" claude_notes/roughness_b1/logs/driver.log)
while true; do
  n=$(grep -c "exit" claude_notes/roughness_b1/logs/driver.log)
  if [ "$n" -gt "$n0" ]; then tail -n 2 claude_notes/roughness_b1/logs/driver.log; n0=$n; fi
  grep -l "Traceback" claude_notes/roughness_b1/logs/*.log 2>/dev/null
  [ "$n" -ge "$want" ] && break
  sleep 20
done
