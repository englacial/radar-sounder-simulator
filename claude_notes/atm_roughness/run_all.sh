#!/bin/bash
cd /home/thomasteisberg/Documents/coherent-radar-simulator
L=claude_notes/atm_roughness/logs
run() { uv run claude_notes/atm_roughness/atm_roughness.py --line $1 --date $2 > $L/$1_$2.log 2>&1; echo "done $1 $2"; }
run greenland_westcoast 2017-05-10 &
run greenland_westcoast 2019-05-14 &
run greenland_westcoast 2016-05-11 &
wait
run greenland_geikie01_transit 2014-04-21 &
run antarctica_getz 2016-11-05 &
run antarctica_david 2013-11-19 &
run antarctica_david 2013-11-20 &
wait
echo ALL DONE
