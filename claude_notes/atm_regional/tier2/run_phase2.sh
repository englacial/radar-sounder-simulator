#!/bin/bash
cd /home/thomasteisberg/Documents/coherent-radar-simulator/claude_notes/atm_regional/tier2
until grep -q "PHASE13 DONE" logs/batch_p13_0.log; do sleep 60; done
for k in 0 1 2; do
  uv run python batch.py --shard $k --nshard 3 --phases 2 --budget-gb 65 > logs/batch_p2_$k.log 2>&1 &
done
wait
echo PHASE2 DONE >> logs/batch_p2_0.log
