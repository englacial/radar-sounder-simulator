# Future option: joint multi-interface refraction solve ("D+")

Status: IMPLEMENTED 2026-07-09 — solver in `src/soundersim/refraction_joint.py` (validated by `tests/test_refraction_joint.py`), integrated into `kernels/multilayer.py` behind `SimConfig.refraction` (default `"joint"` for ≥2 crossings; single-crossing targets keep the exact sequential two-point path), kernel regression in `tests/test_multilayer_joint.py`, benchmarks + correctness in the report case `refraction_joint` (`tools/run_refraction_joint_bench.py`, group "Radar equation comparison"). Docs: `docs/refraction.md`. Originally written up 2026-07-08 after the firn compile-cost investigation; the firn chirp sweep met trigger condition 1. (The simpler alternative "D2" — scanning the existing sequential chain over stacked uniform interfaces — was considered and rejected: it only helps compile time, requires shape-uniform interface grids, and adds masked-padding complexity for no physics gain. The implemented kernel integration does reuse a masked-padding idea, but per-target-layer on the joint solve — power-of-two no-op buckets so target layers share executables — where it is an exact pass-through, not a physics compromise.)

## What we do today (M16)

For a target facet under j interfaces, the kernel chains j independent **two-point** Snell solves: solve against interface 1's local facet plane, step to that crossing, solve against interface 2, and so on. Documented approximation: each two-point solve is exact for its own flat plane, but the *sequence* is not the true stationary path of the full stack — the error vanishes with layer contrast (fine for firn) and was measured on rough interfaces in the M17 twomedia_field case. Compile cost: the chain and the Newton iterations inside each crossing are unrolled into the XLA graph → graph size ∝ N²·n_iter (mitigated by the 2026-07 kernel-caching work, but still paid once per layer count).

## The joint formulation

Solve the full refracted path in one shot: unknowns are the N crossing offsets x₁…x_N (2-D offsets in each interface's local plane, or 1-D after reducing to the vertical plane through source/target); objective is total optical path `Σ nᵢ·|segment i|`; stationarity gives Snell's law at every interface simultaneously.

Key structural fact: the Hessian/Jacobian of the stationarity system is **block-tridiagonal** — crossing i couples only to crossings i−1 and i+1 (adjacent segments). So one Newton step costs O(N) via the Thomas algorithm (block-tridiagonal solve), fully vectorizable over (facets × traces) as batched small solves, and expressible as a fixed-size `lax.scan` (forward elimination + back-substitution) — one graph regardless of N.

## Why it's strictly better (when it's worth building)

1. **Removes the sequential-chaining approximation** — the solution is the true stationary path of the whole stack (still within the local-plane-per-interface model). Accuracy improves precisely where M17 measured degradation (rough interfaces, large contrasts).
2. **Faster asymptotically**: one converging Newton on N unknowns instead of N chained 2-point solves each with its own iteration budget; graph size O(1) in N and in iteration count.
3. **Unlocks large N**: with compile flat and runtime ∝ N per facet (not N² over targets — note each *target layer* still needs its own path, so total work over all layers remains ∝ N² but with a much smaller constant), firn stacks of 150–300 explicit layers become routine — plausibly enough to capture fine-layer statistics directly and retire the "needs a 1-D hybrid" caveat for the upper firn.

## What it costs

- New physics code on the numerically delicate core → needs its own M15-style validation campaign: brute-force Fermat referee on multi-interface stacks (extend `compare/fermat.py` to N interfaces), analytic two/three-layer cases, reciprocity, convergence-budget characterization.
- Initialization and robustness: the joint Newton needs a good starting path — the existing sequential chain is the natural (and already-implemented) initializer; safeguarding (line search / trust region) needed near grazing/TIR. TIR masking must be rethought for the joint system (a path is invalid if ANY crossing goes evanescent).
- Heterogeneous interface grids are fine (unlike D2): the solve operates per target facet on local planes gathered per interface; grid shapes never enter the scan.
- Estimate: a focused milestone (referee extension + solver + validation + kernel integration + regression vs the sequential path), comparable in size to M15+half of M16.

## Trigger conditions for scheduling it

- The firn plateau investigation shows the phenomenon emerging with layer count (evidence that N ≳ 100 matters), or
- multilayer accuracy on rough interfaces (M17's measured chaining degradation) starts limiting a real use case, or
- runtime of the sequential chain becomes the bottleneck on production bed-clutter runs.
