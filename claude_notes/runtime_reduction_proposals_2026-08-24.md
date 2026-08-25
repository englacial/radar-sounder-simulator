# Runtime reduction study (2026-08-24) — evidence + proposals

Goal: full re-run of all four lines (last: 30.6 h wall, sequential) in < 12 h,
ideally on the local 9900X (12C/24T, 128 GB, no GPU); cloud allowed at
<= $5/run. Nothing implemented yet; this note records the measurements
behind the proposal list.

## Where the 30.6 h goes (reconstructed from runs/*.json wall_s + mtimes)

| line / experiment | wall | kernel | dominant pass |
|---|---|---|---|
| getz full_line | 5.4 h | 4.96 h | real_low 3.1 h (49 chunks x 226 s) |
| david david_full | 9.2 h | 9.08 h | basler195 5.1 h (41 x 454 s, 2.5M facets/iface) |
| geikie std_benchmark | 4.3 h | 4.39 h | low 2.5 h incl. companion |
| westcoast std_benchmark | 11.6 h | 11.60 h | 3 x P-3 (32 x 250 s + 32 x 160 s companion) |

- Kernel time is > 97 % of wall; orchestration (prep, focusing, analysis,
  figures) ~35 min total. Chunks run in a serial Python loop, one process.
- `companion: true` (gl_std_benchmark only) re-simulates every real pass at
  constant gamma solely for the bed-brightness-correlation metric:
  **6.2 h of the 30.6** (geikie 1.9 h, westcoast 4.3 h).
- Bed layer (`kernels/multilayer.py`, sequential refraction, f64) is
  ~85-88 % of kernel time; surface (`kernels/coherent.py`, f32) the rest.
- Cost is O(traces x facets): 370-490 ns per facet-trace pair (both
  layers) in production records, reproduced at 235-366 ns on a synthetic
  2-interface scene (scratchpad bench_ml.py).
- Every trace processes every facet of the chunk window
  ((3 km + 2 ct) x 2 ct), but only a disk of radius ct is inside the time
  window: **~62 % of facet-trace work is out-of-window** at 3 km chunks
  (dropped_power ~1e-8 confirms it carries nothing).

## Why the local box does not parallelise (clean measurements, idle CPU)

Synthetic chunk: 200 traces, 250k facets/iface, roughness both layers,
grazing fix, 7-el array. Solo: 26 s wall, 142 cpu-s (5.5 cores busy).

| arrangement | per-job wall | aggregate throughput |
|---|---|---|
| 1 proc, unpinned | 26 s | 1.00x |
| 1 proc pinned to 6 physical cores | 27 s | 1.00x (a process cannot use more) |
| 2 procs unpinned | 45 s | 1.15x |
| 2 procs pinned 0-5 / 6-11 (one per CCD) | 44-45 s | 1.18x |
| 4 procs unpinned | 87 s | 1.20x |
| 4 procs pinned 3 cores each | 83-86 s | 1.22x |
| 1 proc, pmap over 8 virtual CPU devices | 24.5 s | 1.07x (bed); 4.7x on surface-only |
| facet block 65536 -> 16384 / 4096 / 2048 / 1024 | 26-33 s | ~1.0x (bit-identical), also at 2.56M facets |
| trace batch 200 -> 50 -> 12 | 277-353 ns/pair | ~1.0x |

Machine DRAM bandwidth (numpy stream): **29.4 GB/s total**, one thread
saturates it; 4 concurrent get 7.4 GB/s each. 260 ns/pair x 30 GB/s =
~7.8 KB of traffic per facet-trace pair, matching the compiled bed HLO
(~340 loop fusions per scan step, each materialising a (T, B) f64 array).
=> the bed kernel is **DRAM-bandwidth-bound**; more cores/processes on
this box cannot help. Only fewer bytes per pair (fusion, f32, culling,
fewer pairs) or more memory channels (cloud VMs, GPU) do.

Side notes: `--xla_cpu_multi_thread_eigen=false` is bit-identical and
26 s vs 35 s on the very first (cold) timing only — no effect once warm.
Root filesystem 89 % full with heavy I/O pressure (PSI full ~80 %) during
the study; chunk npz writes (np.savez_compressed) could stall on it.

## Proposals — see the chat summary / handoff for the categorised list

## Implemented 2026-08-24 (1a + 1b + companion off) — kernel era "2026-08-24-cull"

Working tree, uncommitted. Files: src/soundersim/kernels/{geometry,coherent,
multilayer,__init__}.py, tools/run_basal_clutter.py (chunk_meta "kernel" key
-> every chunk cache re-simulates), config/experiments/gl_std_benchmark.yaml
(companion: false). Harness + evidence: claude_notes/runtime_opt/
(kernel_regression.py, exactness_{surface,bed}.py, surface_f64_truth.py,
bench_{bed,surf}.py, compare_pilot.py; old_tree/ = git HEAD source for A/B).

1a  Per-trace along-track facet windowing (geometry.py window_reach_m /
    along_track_order / block_windows). Facets sorted along the chunk track
    axis; each trace scans only the block range within R = c t_end / 2 of
    its own projection (any facet farther away has twtt >= t_end on every
    path, refracted included). Verified: new(cull) == new(no cull)
    bit-for-bit on every field/out array (both kernels, split sides,
    roughness, per-facet gamma, diffuse, incoherent); only `dropped` shrinks.
    Old kernel vs new: bed (f64) <= 1.3e-7 of peak (complex64 output ulp);
    surface (f32) <= 3e-5 of peak on the low scenes, up to 1.8e-3 of peak /
    ~1 dB at samples 60 dB below the peak on the mid/HAPS scenes = f32
    summation-order noise. f64 truth check (surface_f64_truth.py): old kernel
    error rms 6.2e-4 / max 3.8e-2 of peak, new 4.7e-4 / 3.3e-2, identical dB
    error profile -> differences are inside the existing kernel's own noise.
1b  Bed path: component-form 3-vector math (no (T,B,3) temporaries, one
    interleaved centre+normal gather), Newton iterations as lax.fori_loop
    with single-consumer expensive ops (XLA:CPU fuses sqrt/rsqrt/divide only
    into a single consumer; the unrolled chain either materialised ~7 (T,B)
    f64 temporaries per iteration or, once single-consumer, got duplicated
    across iterations -> 8x slower), adaptive facet block size
    (~256k f64 lanes / ~512k f32 lanes per scan step, geometry.auto_block_size;
    at 200 traces block 1024 is 1.65x faster than 4096 on the bed).
    Bed microbench (64 traces x 320k facets, full feature set): 165 -> 126
    ns/pair; XLA fusions per step 348 -> 125. Remaining: Newton 47 ns
    (1.3 ns/iteration, loop-carry bandwidth), radiometric features 33 ns,
    scatter 5 ns, base geometry ~43 ns. Newton budget 10+25 -> 3+8 would
    give 126 -> ~100 ns (proposal 2b, NOT applied).
    441 unit tests pass; ruff clean.

Real chunks (pilot_smoke, outputs/_runtime_bench, vs pilot_fixed records):
getz real_low 2.93x, real_9km 2.77x, real_10km 2.49x; david basler195 2.74x
(see compare_pilot.py output in the handoff). getz pilot metrics identical to
pilot_fixed (residuals, clutter, alignment, tails) except simulation_wall_s
515.6 -> 183.0 s.

## All four pilots at the committed kernels (8ab38b7, 2026-08-24 evening)

Driver wall (outputs/sim_runs_2026-08-24/pilots_driver.log): getz 3 min (was
15), david 5 (23), geikie 11 (56), westcoast 8 (44). Per-chunk kernel wall vs
the previous records (getz/david: pilot_fixed, same physics; Greenland:
08-22 pilot_smoke, pre-grazing-fix, same facets/geometry):
getz 2.5-2.9x, david 2.7-2.8x, geikie high 4.8x / low 4.9x, westcoast 5.6x;
all pilot kernel time 124 -> 28 min (4.5x). Full-run projection with these
per-line factors: getz ~2.0 h, david ~3.4 h, geikie ~0.65 h, westcoast
~1.5 h, + ~0.6 h orchestration => ~8 h (was 30.6 h).

## Full campaign at the merged kernels (main af9917f, 2026-08-24 21:35 -> 08-25 03:04)

outputs/sim_runs_2026-08-24/full_all.log, all rc=0, every chunk re-simulated:
getz 82 min (was 321), david 127 (551), geikie 35 (264), westcoast 84 (700)
=> 328 min = 5.5 h wall (was 30.6 h, 5.6x). Kernel hours: getz 0.96 (4.96),
david 2.02 (9.08), geikie 0.58 (2.76 + 1.9 companion), westcoast see
metrics.json (7.29 + 4.3 companion).
