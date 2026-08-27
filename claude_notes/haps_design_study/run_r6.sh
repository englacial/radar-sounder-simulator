#!/bin/bash
cd /home/thomasteisberg/Documents/coherent-radar-simulator/.claude/worktrees/agent-a70c6d1cabd5d33f1
for pd in 4 8; do
  uv run python claude_notes/haps_design_study/run_rough_sens.py 1.0 0.05 config/experiments/wc_hd_r6p$pd.yaml > claude_notes/logs/wc_hd_r6p$pd.log 2>&1
done
echo R6DONE > claude_notes/logs/wc_hd_r6.done
