#!/bin/bash
# Level-anchored family at the lower attenuations (session artifact).
# D is CONTAMINATION-AWARE: per pass D_clean solves
# bed*10^(D/10) + surface = measured on the recorded bed-window
# decomposition, then the median over the three real passes is used.
# att20: D_clean = -1.28 / +4.11 / +3.56 -> median +3.56
# att26: D_clean = +6.39 / +11.63 / +11.03 -> median +11.03
set -u
cd /home/thomasteisberg/Documents/coherent-radar-simulator
H=outputs/basal_clutter/hypothesis_tests
COMMON="--segment full --demogorgn-bed --gamma-from-rssnr --processing standard --add-30km --no-companion --anchor level --out $H"
uv run python tools/run_basal_clutter.py $COMMON --att 20 --level-deficit-db 3.56 \
    --out-name att20_klevel > $H/att20_klevel.log 2>&1
echo "att20_klevel done"
uv run python tools/run_basal_clutter.py $COMMON --att 26 --level-deficit-db 11.03 \
    --out-name att26_klevel > $H/att26_klevel.log 2>&1
echo "att26_klevel done"
