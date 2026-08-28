#!/bin/bash
cd /home/thomasteisberg/Documents/coherent-radar-simulator/claude_notes/atm_regional/tier2
for k in 0 1 2; do
  uv run python batch.py --shard $k --nshard 3 --phases 1 3 --budget-gb 60 > logs/batch_p13_$k.log 2>&1 &
done
wait
echo PHASE13 DONE >> logs/batch_p13_0.log
