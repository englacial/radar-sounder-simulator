#!/bin/bash
cd /home/thomasteisberg/Documents/coherent-radar-simulator
uv run python tools/run_basal_clutter.py --config claude_notes/pilot_exponential/pilot_exponential.yaml --line greenland_westcoast --force > claude_notes/pilot_exponential/logs/westcoast_rerun_oneline.log 2>&1
uv run python claude_notes/pilot_exponential/compare.py > claude_notes/pilot_exponential/logs/compare_rerun.log 2>&1
echo done > claude_notes/pilot_exponential/logs/wc_rerun.done
