#!/bin/bash
# Attenuation sweep (session artifact): att = 20 and 26, configured
# IDENTICALLY to the existing endpoints hypothesis_tests/baseline (att 15)
# and t2_att31 (att 31) -- full 50 km, all four passes, DEMOGORGN bed,
# RSSNR gamma, matched processing, UNSPLIT (no T5 flags).
set -u
cd /home/thomasteisberg/Documents/coherent-radar-simulator
H=outputs/basal_clutter/hypothesis_tests
COMMON="--segment full --demogorgn-bed --gamma-from-rssnr --processing standard --add-30km --no-companion --out $H"
for a in 20 26; do
  uv run python tools/run_basal_clutter.py $COMMON --att $a \
      --out-name att$a > $H/att$a.log 2>&1
  echo "att$a done"
done
