"""Surface-kernel microbench. usage: bench_surf.py T BLK"""
import os, sys, time, resource
os.environ["SOUNDERSIM_JAX_CACHE_DIR"] = "0"
import numpy as np
from soundersim.kernels.coherent import coherent_cluttergram
from soundersim.antenna import pattern_args
from soundersim.config import AntennaConfig
T, B = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(3)
xs = np.arange(-6000, 6000, 15.0); ys = np.arange(-3000, 3000, 15.0)
X, Y = np.meshgrid(xs, ys, indexing="ij"); Z = 20 * np.sin(X / 700.0) + 15 * np.cos(Y / 500.0)
centers = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1); F = len(centers)
n = np.stack([-20 / 700 * np.cos(X / 700.0), 15 / 500 * np.sin(Y / 500.0), np.ones_like(X)], -1).reshape(-1, 3)
normals = n / np.linalg.norm(n, axis=-1, keepdims=True)
e1 = np.tile([15.0, 0, 0], (F, 1)); e2 = np.tile([0, 15.0, 0], (F, 1)); areas = np.full(F, 225.0)
pos = np.stack([np.linspace(-1500, 1500, T), np.zeros(T), np.full(T, 700.0)], -1)
u_at = np.tile([1.0, 0, 0], (T, 1)); u_ct = np.tile([0, 1.0, 0], (T, 1))
c = 299792458.0; k = 2 * np.pi * 195e6 / c
ph = (rng.standard_normal(F) + 1j * rng.standard_normal(F)).astype(np.complex64) / np.sqrt(2)
pat = pattern_args(AntennaConfig(kind="array", n_elements=7, spacing_lam=0.5), u_at, u_ct)
kw = dict(k=k, gamma=0.3, t0=2 * 600 / c, dt=5e-9, n_samples=4000, c=c, pattern=pat,
          roughness=(0.05, 3.0, ph, 10), taper_s=0.05, d_phi_area=True, block_size=B, window_cull=False)
coherent_cluttergram(pos, u_ct, centers, normals, areas, e1, e2, **kw)
r0 = resource.getrusage(resource.RUSAGE_SELF); t = time.perf_counter()
coherent_cluttergram(pos, u_ct, centers, normals, areas, e1, e2, **kw)
w = time.perf_counter() - t; r1 = resource.getrusage(resource.RUSAGE_SELF)
cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
print(f"T={T} B={B} F={F} wall={w:.2f}s util={cpu/w:.1f} ns/pair={w/(T*F)*1e9:.1f}")
