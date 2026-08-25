"""Bit-exactness of the 1a bed kernel: old kernel on pre-sorted facets (same
block partition) vs new kernel with / without culling. usage: exactness_bed.py old|new_cull|new_nocull out.npz"""
import os, sys
os.environ["SOUNDERSIM_JAX_CACHE_DIR"] = "0"
mode, out = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
if mode == "old":
    sys.path.insert(0, os.path.join(HERE, "old_tree", "src"))
import numpy as np, soundersim
from soundersim.kernels.multilayer import refracted_cluttergram
from soundersim.scene import Facets
from soundersim.antenna import pattern_args
from soundersim.config import AntennaConfig
print("using", os.path.dirname(soundersim.__file__))
rng = np.random.default_rng(5)
T, B = 24, 4096

def lattice(x0, x1, y0, y1, sp, zf, nf):
    xs = np.arange(x0, x1, sp); ys = np.arange(y0, y1, sp)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = zf(X, Y); c = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
    n = nf(X, Y).reshape(-1, 3); n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    F = len(c)
    e1 = np.tile([sp, 0, 0], (F, 1)) + rng.normal(0, 0.05, (F, 3)); e2 = np.tile([0, sp, 0], (F, 1)) + rng.normal(0, 0.05, (F, 3))
    a = np.linalg.norm(np.cross(e1, e2), axis=-1)
    cell = np.stack(np.meshgrid(np.arange(len(xs)), np.arange(len(ys)), indexing="ij"), -1).reshape(-1, 2)[:, ::-1]
    return Facets(centers=c, normals=n, areas=a, e1=e1, e2=e2, cell=cell, grid_shape=(len(ys), len(xs)))

# NOTE Facets.cell is (row, col); grid index row = y index, col = x index but we reshape (ny, nx) via grid_shape;
# centers.reshape(ny, nx, 3) must match the cell layout: build arrays in (y, x) order instead.
def lattice_yx(x0, x1, y0, y1, sp, zf, nf):
    xs = np.arange(x0, x1, sp); ys = np.arange(y0, y1, sp)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    Z = zf(X, Y); c = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
    n = nf(X, Y).reshape(-1, 3); n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    F = len(c)
    e1 = np.tile([sp, 0, 0], (F, 1)) + rng.normal(0, 0.05, (F, 3)); e2 = np.tile([0, sp, 0], (F, 1)) + rng.normal(0, 0.05, (F, 3))
    a = np.linalg.norm(np.cross(e1, e2), axis=-1)
    cell = np.stack(np.meshgrid(np.arange(len(ys)), np.arange(len(xs)), indexing="ij"), -1).reshape(-1, 2)
    return Facets(centers=c, normals=n, areas=a, e1=e1, e2=e2, cell=cell, grid_shape=(len(ys), len(xs)))

surf = lattice_yx(-6000, 6000, -3000, 3000, 15.0, lambda X, Y: 20 * np.sin(X / 700.0) + 10 * np.cos(Y / 400.0),
                  lambda X, Y: np.stack([-20 / 700 * np.cos(X / 700.0), 10 / 400 * np.sin(Y / 400.0), np.ones_like(X)], -1))
bed = lattice_yx(-6000, 6000, -3000, 3000, 15.0, lambda X, Y: -1200 + 40 * np.sin(Y / 500.0) + 0.05 * X,
                 lambda X, Y: np.stack([-0.05 * np.ones_like(X), -40 / 500 * np.cos(Y / 500.0), np.ones_like(X)], -1))
F = len(bed.centers)
pos = np.stack([np.linspace(-1200, 1200, T) + rng.normal(0, 1, T), rng.normal(0, 2, T), 700.0 + rng.normal(0, 3, T)], -1)
u_at = np.tile([1.0, 0, 0], (T, 1)); u_ct = np.tile([0, 1.0, 0], (T, 1))
c = 299792458.0; k0 = 2 * np.pi * 195e6 / c
ph = (rng.standard_normal(F) + 1j * rng.standard_normal(F)).astype(np.complex64) / np.sqrt(2)
dph = (rng.standard_normal(F) + 1j * rng.standard_normal(F)).astype(np.complex64) / np.sqrt(2)
gam = (-0.3 - 0.2 * rng.random(F)); damp = 0.1 + 0.2 * rng.random(F)
pat = pattern_args(AntennaConfig(kind="array", n_elements=7, spacing_lam=0.5), u_at, u_ct)
t0 = 2 * 600 / c; dt = 5e-9; ns = 3200
if mode == "old":
    order = np.argsort(bed.centers[:, :2] @ np.array([1.0, 0.0]), kind="stable")
    sel = lambda a: a[order]
    bed_s = Facets(centers=bed.centers[order], normals=bed.normals[order], areas=bed.areas[order], e1=bed.e1[order],
                   e2=bed.e2[order], cell=bed.cell[order], grid_shape=bed.grid_shape)
    kw = dict(block_size=B)
else:
    sel = lambda a: a; bed_s = bed
    kw = dict(block_size=B, window_cull=(mode == "new_cull"))
res = {}
cases = {
    "smooth_iso": dict(gamma=0.3, pattern=None),
    "full": dict(gamma=sel(gam), pattern=pat, roughness=(0.10, 0.886, sel(ph), 12), crossed_sigma=[0.05],
                 diffuse=(sel(damp), sel(dph), 1.0), taper_s=0.05, d_phi_area=True),
}
for name, extra in cases.items():
    for split in (False, True):
        o, d = refracted_cluttergram(pos, u_ct, bed_s, [surf], [1.0, 3.17], [0.0, 18.0], mode="coherent", t0=t0, dt=dt,
                                     n_samples=ns, c=c, k0=k0, split_sides=split, **extra, **kw)
        res[f"{name}_split{int(split)}/out"] = o; res[f"{name}_split{int(split)}/dropped"] = d
o, d = refracted_cluttergram(pos, u_ct, bed_s, [surf], [1.0, 3.17], [0.0, 18.0], mode="incoherent", t0=t0, dt=dt,
                             n_samples=ns, c=c, **kw)
res["incoherent/out"] = o; res["incoherent/dropped"] = d
np.savez(out, **res); print("saved", out)
