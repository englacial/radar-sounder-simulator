# GCP compute options for the basal-clutter simulations (2026-09-03)

Read-only investigation (no benchmarks run; pilots were running locally). Prices
are us-central1 list prices as mirrored by gcloud-compute.com (SKU data dated
2026-08-30) because cloud.google.com/compute/vm-instance-pricing and
/compute/gpus-pricing render client-side and could not be scraped; treat them
as ±5 % and re-check in the console before committing. Cloud Batch itself is
free (cloud.google.com/batch/pricing); internet egress $0.12/GB (first TB).

## Bottom line

| option | six pilots @ pdiv 8 (local 4.0 h) | one PIN pilot (local 40 min) | full campaign @ pdiv 8 (local ~72 h est.) | effort |
|---|---|---|---|---|
| (a) Batch fan-out, c3d-standard-8 **Spot**, one task per pass (26) | **~20 min chunks + local post-proc; $0.5** (OD $2.6) | ~18 min; $0.10 (OD $0.5) | ~2 h with ~100 VMs (chunk tasks); **$11** (OD $56) + $1.5 egress | 2–3 d |
| (a') same, one task per chunk (78) | ~10 min; $0.8 (OD $4) | ~10 min; $0.16 (OD $0.8) | same as above | same |
| (a) with c3-standard-8 Spot | ~20 min; $0.9 (OD $2.9) | $0.17 (OD $0.5) | $20 (OD $62) | same |
| (c) one c3d-standard-360 Spot, 6 line processes, no tool change | ~65 min; $3.7 (OD $18) | ~65 min (no gain) | ~20 h (bound by longest line) | 0.5 d |
| (c') same VM + chunk-level `xargs -P 20` | ~35 min; $2 (OD $11) | ~20 min; $1.1 | ~5 h; $17 (OD $84); preemption risk | 1.5 d |
| (b) one A100-40GB (a2-highgpu-1g) Spot, **assumed 5x/chunk (2–20x)** | 26–134 min; $0.9–4.7 (OD $1.6–8) | ~17 min; $0.6 | 3.6–36 h; $8–76 | 3–4 d + benchmark |
| (b) one L4 (g2-standard-4) Spot, assumed 2.5x (1–8x) | ~110 min; $0.8 (OD $1.3) | ~25 min; $0.2 | ~30 h; $13 | same |

Arithmetic and assumptions in §4. Cost numbers exclude ~$0.05 of disk/GCS
and assume Spot capacity is available (c3d Spot in us-central1 is normally
plentiful; A2 Spot was quota 0 in July, see claude_notes/gpu_benchmark_findings.md).

**Recommendation:** build (a) — a Cloud Batch (or plain `gcloud` loop) fan-out
over Spot c3d-standard-8 VMs at pass granularity, using the existing
cache-first chunk store as the interface. It is the cheapest option by 3–10x,
turns the 4 h pilot set into ~20 min and the ~72 h full campaign into ~2 h,
is immune to preemption (chunks re-run), and most of the plumbing exists in
tools/gcp/. Option (c) is the low-effort stopgap if something is needed this
week. Do not build on GPUs until a $3 one-chunk benchmark has replaced the
guessed speedup (§3b).

## 1. Workload

- **Software:** jax 0.10.2 / jaxlib 0.10.2, CPU wheels only in uv.lock (no
  jax-cuda plugin). Kernels (`src/soundersim/kernels/{coherent,multilayer}.py`)
  are plain `jax.numpy` + `lax.scan`/`vmap`/`segment_sum`, with the bed path
  under a scoped `jax.enable_x64()`. They ran unchanged on a V100 in July
  (`uv pip install 'jax[cuda12]==<locked>'` + `uv run --no-sync`; fields agree
  to 4e-4 rel). So GPU is a packaging change, not a code change.
- **Cost model:** O(traces x facets); bed (f64 sequential-Snell) ~85 % of kernel
  time; kernel ~97 % of wall. Measured today on PIN dc8_2014_0km at pdiv 8:
  1803 traces x 563k facets/interface per chunk, 206 s/chunk (first chunk
  236 s => **~30 s JIT compile per process**; persistent cache at
  ~/.cache/soundersim/jax, 620 MB, keyed on CPU target so it will not carry
  to a different CPU). HAPS 14 km chunks (70 m facets) ~30 s.
- **Memory:** the running chunk process is 2.1 GB RSS at ~4.7 cores busy.
  Full-campaign chunks reach 3.3M facets/interface (geikie low); expect 2–6 GB.
  8 vCPU / 16–32 GB per worker is ample; nothing needs >8 GB.
- **Local box:** Ryzen 9 9900X 12C/24T (Zen 5), 123 GB RAM, no GPU.
  claude_notes/runtime_reduction_proposals_2026-08-24.md measured 29.4 GB/s
  DRAM stream, saturated by one chunk process (~7.8 KB traffic per
  facet-trace pair): 2–4 concurrent chunks give only 1.15–1.22x aggregate.
  More cores here do nothing; the win must come from more memory channels
  (many VMs, a 2-socket VM, or GPU HBM).
- **Parallel units:** chunk (78 in the pilot set, 30 s–4 min each, ~16 MB
  npz + json; `run_level()` in tools/run_altitude_comparison.py:823 hits the
  cache purely on `json.dumps(meta, sort_keys=True)` equality — no host paths
  in `chunk_meta`, so a chunk simulated elsewhere and dropped into `runs/`
  should skip; the July "GCS caches don't skip" finding was with the B26 tool
  and must be re-verified with one chunk), pass (26; one compile per pass is
  the natural job), line (6; scene prep + focusing + analysis stay per-line).
  The tool has no "simulate one chunk and exit" entry point yet.

## 2. Data a worker needs (all cache-first, all under outputs/cache/)

| input | how loaded | size to stage |
|---|---|---|
| OPR frames `frame_<season>_<frame>_CSARP_standard.nc` + `layers_*.nc` picks | `opr.load_frame`, xopr on miss (anonymous STAC/S3, slow) | 65–200 MB each; 6–14 frames per line => ~0.5–1.5 GB/line, ~5 GB all six (5.4 GB of .nc cached now; the 6.1 GB `xopr/` raw cache is not needed) |
| REMA/ArcticDEM windows `dem_*.tif` | rasterio on the cached tif; STAC on miss | 1.2 GB total (258 files, most stale keys; the live ones are ~100 MB) |
| BedMachine `bedmachine_*.tif` (Greenland lines + geoid band) | cached tif; on miss needs Earthdata netrc | 53 MB |
| DEMOGORGN `demogorgn_antarctic_seed000_*.tif` | cached tif; on miss anonymous icechunk S3 | 3 MB |
| RSSNR anchor `outputs/<line>/rssnr_anchor.npz` | cache-first; on miss anonymous icechunk S3 | 7–11 KB each |
| ATM roughness | `config/roughness/*.yaml` in the repo; `outputs/cache/atm{,2}` (67 GB) **not read** at run time | 0 |
| `covariates/` (1 GB) | not referenced by run_basal_clutter | 0 |

So a **~2–6 GB bundle** (repo tarball + the six lines' frames, tifs, anchors)
uploaded once to GCS (ingress free, $0.02/GB-month) covers everything; no
S3/Earthdata credentials are required if every cache file ships, and the two
icechunk stores are anonymous anyway. Results: pilot 78 x ~16 MB ≈ 1.2 GB
($0.15 egress); full campaign ~800 chunks x ~16 MB ≈ 13 GB ($1.5). Sync
back with `gcloud storage rsync gs://.../runs/ outputs/<line>/pilot/runs/`
and re-run the line locally: every chunk should print `[skip-exists]` and
only focusing/analysis/figures run (a few minutes per line).

## 3. Options and prices (us-central1, hourly, 2026-08-30 data)

| machine | vCPU / GB | on-demand | Spot | notes |
|---|---|---|---|---|
| c3d-standard-8 (Genoa, 12x DDR5 ch/socket) | 8 / 32 | $0.3632 | **$0.0728** | cheapest Spot per vCPU ($0.0091) |
| c3d-standard-16 | 16 / 64 | $0.7264 | $0.1457 | |
| c3-standard-8 (Sapphire Rapids, 8x DDR5) | 8 / 32 | $0.4032 | $0.1319 | |
| c3-standard-22 | 22 / 88 | $1.1088 | $0.3626 | |
| n2-standard-8 (Ice Lake, DDR4) | 8 / 32 | $0.3885 | $0.233 | less bandwidth, 3x pricier Spot: skip |
| c3-standard-176 / c3-highmem-176 (2 sockets) | 176 / 704 or 1408 | $8.87 / $11.64 | $2.90 / $3.81 | |
| c3d-standard-360 (2 sockets, 180 cores) | 360 / 1440 | $16.34 | $3.28 | best bandwidth per $ for (c) |
| m3-ultramem-32 (Ice Lake) | 32 / 976 | $6.09 | $3.65 | memory size is not our constraint: skip |
| g2-standard-4, 1x L4 | 4 / 16 | $0.7068 | $0.424 | one third-party page lists L4 as on-demand only; verify |
| a2-highgpu-1g, 1x A100 40 GB | 12 / 85 | $3.67 | $2.12 | A2 quota was 0 in July; request needed |
| a2-ultragpu-1g, 1x A100 80 GB | 12 / 170 | $5.07 | $2.93 | |
| a3-highgpu-1g, 1x H100 80 GB | 26 / 234 | $11.06 | $6.64 | |

**(a) CPU fan-out.** Per-chunk speed on a 4-core (8 vCPU) slice is the key
unknown: fair-share DRAM bandwidth is ~20–25 GB/s (c3: ~307 GB/s per socket
/ 56 cores; c3d: ~460 GB/s / 90 cores), i.e. about what one process gets
locally, but bandwidth is not partitioned so an uncontended host gives more,
while 4 physical cores vs ~4.7 used locally costs a little. Assumed
**1.3x slower than local per chunk (range 0.9–1.6x)**; a c3d-standard-16 at
2x the price would remove the core limit if the 8-vCPU trial disappoints.
c3d Spot wins on price; c3 is the fallback if c3d Spot capacity is short.
Batch runs several tasks per VM sequentially, so boot/pull amortise; the
compile does not unless one process handles a whole pass.

**(b) GPU.** Would run without code changes (see §1). Speedup for THIS
kernel shape is unmeasured. Evidence: the July V100 run gave 1.08–1.64x vs
the local box, but on the 10–40-interface joint-Newton firn stack
(scan-of-scans, serialisation-bound) with 60 traces; the pilot bed kernel has
one crossing (sequential Snell) and 1800 traces/chunk, giving ~0.9M lanes per
scan step (auto_block_size -> 512 facets x 1803 traces) — far better
occupancy. Theoretical bounds are enormous (an A100's 9.7 TFLOPS f64 would do
the ~1e9-pair chunk in seconds), the practical limit is how well XLA:GPU fuses
the `fori_loop` Newton + `segment_sum` scatter; ~1 % of peak is typical for
this kind of code, hence **A100 5x (2–20x), L4 2.5x (1–8x; f64 at 1/64 rate
and 300 GB/s), H100 7x (3–30x)** per chunk vs local. A GPU only competes if
the realised speedup is >=10x, which is exactly what a one-chunk benchmark
would settle for ~$3 (one PIN dc8 0 km chunk on a2-highgpu-1g Spot, ~1 h
including driver install; tools/gcp/vm_bootstrap.sh already handles the
CUDA-jax install traps).

**(c) One big VM.** A 2-socket c3d-standard-360 has ~20–30x the local memory
bandwidth (~920 GB/s aggregate) and 180 cores, so ~20 chunk processes should
run at near-solo speed (assume 1.3x slowdown each, **effective 15x**). With
zero tool changes (six `run_basal_clutter.py --line` processes) the gain is
capped at 6 processes and the longest line (PIN, ~52 min); a chunk-level
launcher plus `xargs -P 20` is needed to reach the 15x. Spot preemption
kills all in-flight chunks at once (cache makes it resumable). Setup
~15 min (boot, rsync ~5 GB, `uv sync`, one compile per process).

## 4. Arithmetic (S = 1.3 cloud/local per-chunk factor; job overhead = 1.5 min boot + 1.5 min `uv sync`/image pull + 0.5 min staging + 1 min compile = 4.5 min; local kernel work: pilot set 240 min, PIN 40 min, full ~72 h = 8 h x 9 for pdiv 8)

- (a) pilot set, 26 pass tasks: 240 x 1.3 + 26 x 4.5 = 429 VM-min = 7.15 VM-h
  -> c3d Spot $0.52 / OD $2.60; c3 Spot $0.94 / OD $2.88; n2 Spot $1.67.
  Wall = 4.5 + longest pass (3 chunks x 3.5 x 1.3 = 13.7) ≈ 18–20 min, then
  local per-line post-processing (~5 min/line, can overlap). Needs ~208 Spot
  vCPUs of C3D quota (a fresh project may default lower; raise in advance).
- (a') 78 chunk tasks: 312 + 78 x 4.5 = 663 VM-min = 11 VM-h -> c3d Spot
  $0.80 / OD $4.0; wall ≈ 4.5 + 4.6 ≈ 10 min (overhead-dominated).
- (a) PIN alone, 6 pass tasks: 52 + 27 = 79 VM-min -> c3d Spot $0.10 / OD
  $0.48; wall ~18 min. 18 chunk tasks: 133 VM-min -> $0.16 / $0.81; ~10 min.
- (a) full campaign, ~800 chunk tasks, Batch parallelism 100 (800 vCPU
  quota): 72 x 60 x 1.3 + 800 x 4.5 = 9216 VM-min = 154 VM-h -> c3d Spot
  $11.2 / OD $56; c3 Spot $20 / OD $62; wall ≈ 92 min + tail ≈ 2 h;
  egress 13 GB $1.5. Pass-level tasks would instead be bound by the longest
  pass (geikie low ≈ 47 chunks x 144 s x 9 ≈ 17 h), so chunk granularity is
  required for the full campaign.
- (c) c3d-standard-360 Spot $3.28/h: six line processes: 15 + 40 x 1.3 =
  67 min -> $3.7 (OD $18). Chunk-level, effective 15x: 15 + 312/15 ≈ 36 min
  -> $2.0 (OD $11). PIN alone (18 chunks at once): 15 + 4.6 ≈ 20 min ->
  $1.1. Full: 72 h / 15 ≈ 4.8 h + 0.3 -> $17 (OD $84). c3-highmem-176 is
  similar with fewer cores at $3.81 Spot.
- (b) A100 Spot $2.12/h, sequential in one process per line (6 compiles ≈
  6 min, 8 min driver/CUDA setup): 5x -> 48 + 14 = 62 min, $2.2 (OD $3.8);
  20x -> 26 min, $0.9; 2x -> 134 min, $4.7. PIN at 5x: 17 min, $0.6.
  Full at 5x: 14.4 h, $31 (OD $53); at 20x: 3.6 h, $8. L4 Spot $0.424/h at
  2.5x: 96 + 14 = 110 min, $0.78; at 1x it is slower than local. H100 Spot
  $6.64/h at 7x: 48 min, $5.3.

## 5. Engineering effort, least to most

1. **(c) big Spot VM, line-level (0.5 day):** `gcloud compute instances
   create` with `--provisioning-model=SPOT --max-run-duration` (pattern in
   tools/gcp/launch_benchmark_vm.sh), rsync repo + cache subset, `uv sync`,
   run six lines with `&`, rsync `runs/` back. Nothing in the tool changes.
2. **Chunk-level entry point (+1 day, shared by every other option):** a
   `--simulate-only [--pass KEY] [--chunk N]` mode in tools/run_basal_clutter.py
   that runs scene prep, simulates the requested chunk(s) into `runs/` and
   exits before focusing; plus a dry-run that enumerates (line, pass, chunk,
   rid) so a task array can be built. Then (c') is `xargs -P 20` on the big VM.
3. **(a) Batch fan-out (+1–2 days on top of 2):** GCS bucket with the
   bundle; a script runnable (no container needed — the existing
   batch_benchmark_job.json fetches a bootstrap script and `uv sync`s, ~2–3
   min; a prebuilt image in Artifact Registry would shave ~1.5 min per VM
   later); Batch job JSON with `taskCount` = passes or chunks and
   `BATCH_TASK_INDEX` -> (line, pass, chunk); a sync-back + local
   `[skip-exists]` verification; C3D Spot quota request. Credentials: none
   beyond `gcloud auth` if all caches ship (icechunk stores are anonymous).
4. **(b) GPU (+1–2 days on top of 3, after a benchmark):** CUDA base image or
   `installGpuDrivers`, `jax[cuda12]==0.10.2` pinned with `uv run --no-sync`,
   A2/G2 Spot quota, and a kernel block-size knob (`block_size` is already an
   argument) tuned for GPU occupancy.

## 6. Recommendation

Pursue (a): Cloud Batch over c3d-standard-8 Spot with the chunk-level entry
point (item 2 then 3). It is the only option that scales to the pdiv-8 full
campaign (~2 h, ~$12) and it makes the six-pilot loop a ~20-minute, sub-dollar
operation; preemption costs nothing because chunks are the cache unit and
`run_level` re-simulates only what is missing. First step is a $0.10
calibration: one c3d-standard-8 and one c3d-standard-16 Spot VM each
simulating a single PIN dc8 0 km chunk, to replace the assumed 1.3x factor
and to confirm the copied chunk hits `[skip-exists]` locally. If a run is
needed before that is built, (c) with six line processes on a
c3d-standard-360 Spot VM is half a day of work for a ~4x pilot-set speedup at
~$4. Park the GPU until a one-chunk A100/L4 benchmark (~$3) shows >=10x; at
the assumed 5x it is neither cheaper nor faster than the CPU fan-out.

Sources: gcloud-compute.com machine pages (c3-standard-8, c3d-standard-8,
c3d-standard-16, c3-standard-22, n2-standard-8, c3-standard-176,
c3-highmem-176, c3d-standard-360, m3-ultramem-32, g2-standard-4,
a2-highgpu-1g, a2-ultragpu-1g, a3-highgpu-1g; "Last Update Sun Aug 30 2026");
docs.cloud.google.com/compute/docs/general-purpose-machines (C3 = Sapphire
Rapids, 8 DDR5 channels; C3D = Genoa; N2 = Ice Lake); cloud.google.com/batch/pricing
(no Batch surcharge); egresscost.com/gcp and cloud.google.com/vpc/pricing-announce
($0.12/GB internet egress first TB, 2026); claude_notes/gpu_benchmark_findings.md
(2026-07-28 V100 result) and claude_notes/runtime_reduction_proposals_2026-08-24.md
(DRAM-bound finding).
