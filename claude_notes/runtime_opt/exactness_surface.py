"""Bit-exactness of the 1a surface kernel: old kernel on pre-sorted facets
(same block partition) vs new kernel with / without window culling.
usage: exactness_surface.py old|new_cull|new_nocull out.npz"""
import os, sys
os.environ["SOUNDERSIM_JAX_CACHE_DIR"] = "0"
mode, out = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
if mode == "old":
    sys.path.insert(0, os.path.join(HERE, "old_tree", "src"))
import numpy as np, soundersim
from soundersim.kernels.coherent import coherent_cluttergram
from soundersim.antenna import pattern_args
from soundersim.config import AntennaConfig
print("using", os.path.dirname(soundersim.__file__))
rng = np.random.default_rng(3)
T, B = 30, 4096
xs = np.arange(-6000, 6000, 12.0); ys = np.arange(-3000, 3000, 12.0)
X, Y = np.meshgrid(xs, ys, indexing="ij")
Z = 20 * np.sin(X / 700.0) + 15 * np.cos(Y / 500.0)
centers = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
F = len(centers)
n = np.stack([-20 / 700 * np.cos(X / 700.0), 15 / 500 * np.sin(Y / 500.0), np.ones_like(X)], -1).reshape(-1, 3)
normals = n / np.linalg.norm(n, axis=-1, keepdims=True)
e1 = np.tile([12.0, 0, 0], (F, 1)) + rng.normal(0, 0.1, (F, 3)); e2 = np.tile([0, 12.0, 0], (F, 1)) + rng.normal(0, 0.1, (F, 3))
areas = np.linalg.norm(np.cross(e1, e2), axis=-1)
pos = np.stack([np.linspace(-1500, 1500, T) + rng.normal(0, 1, T), rng.normal(0, 2, T), 700.0 + rng.normal(0, 3, T)], -1)
u_at = np.tile([1.0, 0, 0], (T, 1)); u_ct = np.tile([0, 1.0, 0], (T, 1))
c = 299792458.0; k = 2 * np.pi * 195e6 / c
ph = (rng.standard_normal(F) + 1j * rng.standard_normal(F)).astype(np.complex64) / np.sqrt(2)
gam = (-0.3 - 0.2 * rng.random(F)).astype(np.float32)
pat = pattern_args(AntennaConfig(kind="array", n_elements=7, spacing_lam=0.5), u_at, u_ct)
t0 = 2 * 600 / c; dt = 5e-9; ns = 1200
# pre-sort along +x (the track axis first->last position) for the old kernel
if mode == "old":
    order = np.argsort(centers[:, :2] @ np.array([1.0, 0.0]), kind="stable")
    sel = lambda a: a[order]
    kw = dict(block_size=B)
else:
    sel = lambda a: a
    kw = dict(block_size=B, window_cull=(mode == "new_cull"))
res = {}
for split in (False, True):
    for rough, gf, interp in ((None, 0.3, False), ((0.05, 3.0, sel(ph), 10), sel(gam), True)):
        f, d = coherent_cluttergram(pos, u_ct, sel(centers), sel(normals), sel(areas), sel(e1), sel(e2),
                                    k=k, gamma=gf, t0=t0, dt=dt, n_samples=ns, c=c, split_sides=split,
                                    interp_bins=interp, pattern=pat, roughness=rough, taper_s=0.05,
                                    d_phi_area=True, **kw)
        key = f"split{int(split)}_rough{int(rough is not None)}"
        res[key + "/field"] = f; res[key + "/dropped"] = d
np.savez(out, **res); print("saved", out)
