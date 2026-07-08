"""M19 probe: compile+run time of simulate() vs firn-layer count (temporary).

Tiny flat scene, offset-interface stack. Measures the O(N^2) Newton-unroll
compile cost in kernels/multilayer.py before committing to a layer count.
"""
import time
import warnings

import numpy as np

import soundersim
from soundersim.config import (DemInterface, FacetConfig, Medium,
                               OffsetInterface, RadarConfig, SimConfig)
from soundersim import synthetic as syn

C = 299792458.0


def run(n_layers, mode="coherent"):
    scene = syn.flat_scene(elevation=500.0, altitude=500.0, extent=240.0,
                           posting=24.0, n_traces=2)
    depths = np.linspace(2.0, 100.0, n_layers)
    interfaces = [DemInterface(name="surface")] + [
        OffsetInterface(name=f"l{i}", reference="surface", offset=-float(d))
        for i, d in enumerate(depths)]
    eps = [1.0] + list(np.linspace(1.6, 2.9, n_layers + 1))
    media = [Medium(name=f"m{i}", eps_r=float(e)) for i, e in enumerate(eps)]
    rc = RadarConfig(dt=5e-9, n_samples=500, t0=2.0 * 480.0 / C, f0=195e6)
    cfg = SimConfig(mode=mode, radar=rc, facets=FacetConfig(spacing=12.0),
                    media=media, interfaces=interfaces)
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = soundersim.simulate(scene, cfg)
    return time.perf_counter() - t, ds


for n in (5, 10, 20, 30):
    dt, ds = run(n)
    print(f"n_layers={n:3d}  total={dt:7.2f} s", flush=True)
