#!/bin/bash
# T5 specular/diffuse pilot scan, revision B (session artifact).
# Mean-normalized tilt weight (double-count guard). s0 = 3 deg primary
# (between the user's 1 deg and the bed DEM's own 6.6 deg median tilt);
# f_s scan, then two s0 trials at the best f_s. f_s = 0 has no specular
# channel at all, so it is s0-independent and serves every s0.
set -u
cd /home/thomasteisberg/Documents/coherent-radar-simulator
H=outputs/basal_clutter/hypothesis_tests
COMMON="--segment pilot --demogorgn-bed --gamma-from-rssnr --processing standard --no-companion --passes low mid high --out $H"
run () {  # name fs s0
  uv run python tools/run_basal_clutter.py $COMMON \
      --specular-fraction $2 --spec-tilt-deg $3 --diffuse-exponent 1 \
      --out-name $1 > $H/$1.log 2>&1
  echo "$1 done (fs=$2 s0=$3)"
}
run t5p_fs1.0_s3   1.0 3.0
run t5p_fs0.9_s3   0.9 3.0
run t5p_fs0.5_s3   0.5 3.0
run t5p_fs0.0      0.0 3.0
run t5p_fs0.9_s1   0.9 1.0
run t5p_fs0.9_s6.6 0.9 6.6
