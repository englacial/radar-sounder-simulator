"""Runtime benchmark for the firn plateau redo: compile vs run cost per layer count.

Usage: uv run python claude_notes/firn_bench.py N [N ...]
Times, per N: first simulate() call (XLA compile + run) and second/third calls
with different layer depths but identical shapes (jit cache hit -> run-only).
Run twice in separate processes to see the persistent-cache effect on the
first call (~/.cache/soundersim/jax; SOUNDERSIM_JAX_CACHE_DIR=0 disables).
"""

import importlib.util
import sys
import time

import numpy as np

spec = importlib.util.spec_from_file_location("tfp", "tests/test_firn_plateau.py")
tfp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tfp)

from soundersim.config import (FlatInterface, Medium, OffsetInterface,  # noqa: E402
                               RadarConfig, SimConfig, FacetConfig, DemInterface)
from soundersim import synthetic as syn  # noqa: E402
from soundersim.simulate import simulate  # noqa: E402

Z, RHO = tfp.load_b26()
EPS = tfp.eps_kovacs(RHO)


def point_eps(depth):
    """Closest-sample permittivity at a given depth below the surface."""
    return float(EPS[np.argmin(np.abs(Z - depth))])


def build_cfg(depths):
    media = [Medium(name="air", eps_r=1.0)]
    interfaces = [DemInterface(name="surface")]
    for i, d in enumerate(depths):
        media.append(Medium(name=f"firn{i}", eps_r=point_eps(d)))
        interfaces.append(OffsetInterface(name=f"layer{i}",
                                          reference="surface", offset=-float(d)))
    media.append(Medium(name=f"firn{len(depths)}", eps_r=point_eps(depths[-1] + 1)))
    return SimConfig(mode="coherent", media=media, interfaces=interfaces,
                     radar=RadarConfig(dt=tfp.DT, n_samples=tfp.NSAMP,
                                       t0=tfp.T0, f0=tfp.F0),
                     facets=FacetConfig(spacing=4.0))


def run(n):
    scene = syn.flat_scene(elevation=tfp.ELEV, altitude=tfp.H,
                           extent=tfp.EXTENT, n_traces=3, posting=50.0)
    d1 = np.linspace(5.0, 100.0, n)
    d2 = d1 + 1.7  # same shapes, different data -> should hit the jit cache
    d3 = d1 + 3.1
    t0 = time.perf_counter()
    simulate(scene, build_cfg(d1))
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    simulate(scene, build_cfg(d2))
    t_second = time.perf_counter() - t0
    t0 = time.perf_counter()
    simulate(scene, build_cfg(d3))
    t_third = time.perf_counter() - t0
    print(f"N={n:3d}  first(compile+run)={t_first:7.1f} s   "
          f"second(run-only)={t_second:6.1f} s   third={t_third:6.1f} s   "
          f"compile~={t_first - t_second:7.1f} s", flush=True)


if __name__ == "__main__":
    for n in [int(a) for a in sys.argv[1:]]:
        run(n)
