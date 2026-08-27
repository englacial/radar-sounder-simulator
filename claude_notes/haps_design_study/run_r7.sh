#!/bin/bash
cd /home/thomasteisberg/Documents/coherent-radar-simulator/.claude/worktrees/agent-a70c6d1cabd5d33f1
uv run python claude_notes/haps_design_study/run_rough_sens.py 1.0 0.05 config/experiments/wc_hd_r7p8.yaml > claude_notes/logs/wc_hd_r7p8.log 2>&1
echo done > claude_notes/logs/wc_hd_r7.done
