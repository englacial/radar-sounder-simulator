# GPU cloud benchmark: B26 runs on GCP V100 vs local 9900X (2026-07-28)

Goal: replicate the local B26 firn simulations on a cloud GPU, measure time
and cost. Budget $20; actual spend ~$5.5 (see costs).

## Setup that worked

Spot `n1-standard-8` + 1x V100 (us-central1-a, ~$0.82/h est.), plain
`ubuntu-2204-lts` image + `nvidia-driver-550-server` via startup script,
`uv sync` then `uv pip install 'jax[cuda12]==<locked jax version>'` and ALL
invocations via `uv run --no-sync`. Scripts: tools/gcp/. A2 (A100) quota was
0 everywhere; global GPUS_ALL_REGIONS 0->1 auto-approved in ~4 min; per-type
regional quota allows 1x V100 -- the only strong-f64 GPU available to us.

## Results (identical config, 60 traces, 4 chunks; first-call incl. compile)

| run | 9900X (24t) | V100 | speedup |
|---|---|---|---|
| firn_N10 | 312.6 s | 290.7 s | 1.08x |
| firn_N20 | 1290.7 s | 838.5 s | 1.54x |
| firn_N40 | 5331.1 s | 3258.7 s | **1.64x** |
| firn_N10, joint block 16384 | — | 326.0 s | worse |
| firn_N10, joint block 32768 | — | 343.2 s | worse |

Numerical consistency GPU vs CPU (firn_N40): field max rel diff 4.1e-4
(f32 reduction-order), twtt bit-exact, nadir_twtt 2.8e-14. Interchangeable.

## Interpretation

The multilayer kernel is SERIALIZATION-bound on GPU, not width-bound: the
joint refraction path is a scan-of-scans (Newton iterations x interface
sweep x Thomas forward/back), each step small; widening blocks 4096->32k
made it monotonically slower (recompile + memory pressure, no occupancy
win). nvidia-smi shows 100% "utilization" while throughput is CPU-class --
the classic low-occupancy signature. Speedup grows with N (more work per
launch), so N=160+ runs would fare somewhat better, but O(2x) at ~$0.75/run
(vs a free local box) is not a win.

GPU becomes worth revisiting only after kernel restructuring: fuse/batch the
sequential scan steps, mostly-f32 solve with f64 polish + f64 path
accumulation, larger per-launch work. The f32 surface kernel would fly
(untimed here -- it was cache-skipped on the GPU pass), but it is never the
bottleneck.

**The actual cloud win for this workload is parallelism ACROSS runs, not
within one**: runs are independent, so a Cloud Batch task-array of Spot CPU
VMs (e.g. c2d-standard-32, ~$0.5/h Spot each) rebuilds a full 9-run cache in
~90 min wall for a few dollars -- that is the recommended "real jobs" path
(tools/gcp/batch_benchmark_job.json has the GPU variant; swap the instance
policy for CPU and fan out taskCount).

## Traps hit (so nobody hits them again)

1. **DLVM images ship driver 580 with OPEN kernel modules** -- open modules
   support Turing+ only, so the V100 probe fails even though proprietary 580
   still supports Volta (580 is the LAST Volta branch). All pre-580 DLVM
   image families are retired. Fix: plain Ubuntu + nvidia-driver-550-server
   (proprietary, DKMS builds clean on the stock kernel).
2. **`uv run` re-syncs to uv.lock**, silently reverting `uv pip install
   jax[cuda12]` -- the first "GPU" pass ran on 8 vCPUs with a
   plugin/jaxlib version-skew warning as the only clue. Fix: pin the CUDA
   install to the locked jax version and use `uv run --no-sync` everywhere.
3. **Run-cache npz do not skip across machines** (meta check fails for
   GCS-copied caches; same-machine caches skip fine). Batch jobs must either
   regenerate caches or the meta check needs a portability mode.
4. **Idle burn**: the first VM (driver dead end) idled ~4 h ($3.3) while
   attention was elsewhere. The 6 h max-run-duration cap bounded it, but the
   lesson stands: delete broken VMs immediately, debug from the next one.

## Costs (estimates at Spot list prices; check billing for exact)

| item | duration | est. cost |
|---|---|---|
| VM1 (DLVM driver dead end, mostly idle) | 4.85 h | ~$4.0 |
| VM2 (working benchmark + experiments) | ~1.8 h | ~$1.5 |
| GCS bundle (30-day lifecycle) + egress | — | <$0.10 |
| **total** | | **~$5.5 / $20** |

Cost per firn_N40-class run on V100 Spot: ~$0.74 at 1.64x local speed.
