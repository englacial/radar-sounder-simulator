#!/bin/bash
# Detached re-launch of the B26 Gaussian+grazing-fix twin run (+ report rebuild).
cd "$(dirname "$0")/../.."
L=claude_notes/roughness_exponential/logs
setsid nohup bash -c '/usr/bin/time -f "wall %e s, maxrss %M kB" uv run python tools/run_b26_comparison.py --rough-runs 40:mcords,40:ar,20:mcords:exponential,20:mcords:gaussian:1 --only firn_N20_rough_mcords_exp,firn_N20_rough_mcords_gfx > '"$L"'/b26_rough_gfx.log 2>&1; echo "   b26_gfx exit $? $(tail -1 '"$L"'/b26_rough_gfx.log)" >> '"$L"'/driver.log' > /dev/null 2>&1 < /dev/null &
echo launched
