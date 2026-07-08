# JIT compile-time fix: cached jitted kernels + persistent XLA cache (2026-07-07)

## What changed

- kernels/{multilayer,incoherent,coherent}.py: the jitted callable is now built
  once per static configuration by a module-level `functools.lru_cache` factory
  (`_refracted_fn`, `_incoherent_fn`, `_coherent_fn`) instead of
  `jax.jit(jax.vmap(one_trace))` inline per call. Cache keys (true statics):
  - multilayer: `(coherent, split_sides, n_samples, n_crossed)`
  - incoherent/coherent: `(split_sides, n_samples)`
  Everything numeric that varies between runs (t0/dt/c, gamma, k0/k, per-leg
  eps/index/attenuation arrays, interface lookup constants, facet blocks,
  positions) is a traced argument (vmap in_axes=None broadcast); shape changes
  retrace via jit's own cache. Multilayer tracing still happens under the
  caller's `jax.enable_x64()` scope, so the f64 path stays f64.
- simulate.py: `_configure_jax_cache()` (called on first `simulate()`) enables
  jax's persistent compilation cache at `~/.cache/soundersim/jax/`
  (`jax_persistent_cache_min_compile_time_secs = 1.0`). Override dir with
  `SOUNDERSIM_JAX_CACHE_DIR=/path`, disable with `SOUNDERSIM_JAX_CACHE_DIR=0`.

## Regression (claude_notes/jit_regression_check.py vs jit_baseline_before.npz)

All incoherent outputs, dropped power, nadir twtt: bit-identical. Coherent
fields: max rel-to-peak diff 5e-7..1e-6 (complex64 ulp level) — k/pi and
k/2pi are now computed in traced float32 instead of being constant-folded
through a float64 NumPy scalar before the f32 multiply.

## firn_bench (coherent multilayer, 3 traces), N = layer count

Before (orchestrator baseline): every call paid full compile — N=5: 4.7 s,
N=10: 17 s, N=20: 73 s per call, second call no cheaper.

After, process 1 (cold persistent cache):

| N  | first (compile+run) | second | third |
|----|---------------------|--------|-------|
| 5  | 4.8 s               | 0.2 s  | 0.2 s |
| 10 | 15.1 s              | 0.4 s  | 0.5 s |
| 20 | 66.1 s              | 1.8 s  | 1.9 s |
| 40 | 307.1 s             | 7.0 s  | 7.0 s |

Process 2 (warm persistent cache), first call: N=5 4.8 s, N=10 9.0 s,
N=20 38.6 s, N=40 180.3 s (second/third unchanged). The persistent cache
saves the XLA backend compile (~40-45% at N>=10); the remainder is
trace+lowering, which jax's persistent cache cannot store (and modules
compiling <1 s are not persisted — all of N=5).

## Test suites

- `uv run pytest`: 141 passed, 10.57 s -> 9.6 s (most CI compiles are
  sub-second and per-test shapes differ, so the gain is modest reuse).
- `uv run pytest -m integration`: 18 passed (292 s).
