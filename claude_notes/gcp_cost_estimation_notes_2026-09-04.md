# Estimating Cloud Batch cost for a simulation campaign (agent notes)

Written after the 2026-09-04 full-window campaign overran a $50 cap
(~$56 at the catalog Spot rate). What went wrong and the rules that
follow. Tooling that encodes them: tools/gcp/pricing.py (catalog price,
task-time estimates, projection), batch_launch.py (`--chunks-per-task`,
`--budget-usd`, spend ledger).

## Where the 2026-09-04 estimate broke

| Item | Assumed | Measured | Effect |
|---|---|---|---|
| n2-highmem-8 Spot, us-central1 | $0.15/VM-h (guess) | $0.3144/VM-h (catalog) | x2.1 on everything |
| 195 MHz low-altitude chunk | 5.5 min | 5.7 min (PIN) | ok |
| 60 MHz Basler chunk | 3.3 min ("half of 195") | 6.1 min | x1.8 |
| HAPS chunk, full window | 1.2 min | 5 min (david) | x4 |
| 9/11 km chunk on getz | 1.2 min | 16 min | x13 |
| chunks per line | pilot count x (full/pilot traces) | staging manifest | 1.3-3x low (david 60 MHz: 60 est, 186 actual) |
| guard | live | blind 5 h (token expired) | david + westcoast finished unguarded |

The getz numbers are the key lesson: a per-chunk task re-runs the whole
pass preparation (frames, picked bed, DEMOGORGN synthesis, RSSNR gamma
maps) before simulating one chunk. On a 148 km line that preparation is
~10 min, longer than the chunk. Task time is NOT proportional to chunk
size; it is prep + chunks x per-chunk time.

## Rules

1. **Price from the catalog, never from memory.** Spot discounts move;
   `pricing.py n2-highmem-8` gives today's $/VM-h. On 2026-09-04 the Spot
   discount on N2 was only 40 %.
2. **Stage first, then count.** `stage_bundle.py` writes the exact chunk
   manifest per line (`outputs/gcp/chunks/<line>.json`). Ratio estimates
   from pilot chunk counts were 1.3-3x low because the chunk budget rule
   splits by traces x facets, not by track length.
3. **Task time = prep + N x chunk.** Measure both from past jobs: chunk
   wall_s is in each `runs/*.json`; task run_s is in
   `outputs/gcp/<job>/timing/task_*.json`; prep = run_s - sum(wall_s).
   `pricing.estimate()` does this per line. Without records use 300 s prep
   and 300 s per heavy chunk (120 s light) at posting_div 8, then multiply
   by 1.1 for boot/idle/tail.
4. **Group chunks per task** (`--chunks-per-task 6`) so prep is paid once
   per group. Pass-level tasks are the limit of this; per-chunk tasks make
   sense only for preemption resilience on very long passes.
5. **Rates for HAPS and high-altitude passes are not "light".** Their
   facets are coarse but the along-track window at 14 km/9 km reach is
   wide and the trace count per chunk is high; on full windows they cost
   about as much per chunk as the low passes.
6. **Add a fixed line for the NAT and egress** (~$0.05/h + ~$1) and keep
   ~20 % margin under the cap: the projection's own error is that big.
7. **Guard the spend from the launcher, not from a side monitor**, and
   re-authenticate gcloud right before launching (tokens expired ~12 h in
   on 2026-09-04, blinding the guard). The launcher now tallies the
   running job every 10 min and deletes it past `--budget-usd`, with a
   persistent ledger across launches.
8. **The processing step is a separate budget.** Full-window HAPS passes
   take ~85 min each locally (1350-trace apertures); one cloud task per
   line exceeds a 5 h limit. Fan out per pass or process locally.

## Sanity anchors (posting_div 8, n2-highmem-8)

- PIN 0 km pass, full window: 111 chunks, 340 s per chunk-task = 10.5 VM-h.
- David full window: 676 chunks, 74 VM-h; westcoast 476 chunks, 33 VM-h;
  getz (per-chunk tasks, prep-dominated) 189 chunks, 48 VM-h.
- Pilot (10 km windows), six lines, ~250 chunks total: ~26 VM-h.
