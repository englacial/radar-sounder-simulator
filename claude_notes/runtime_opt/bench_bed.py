"""Bed-kernel microbench (production feature set): ns per facet-trace pair and
compiled-HLO fusion count. usage: bench_bed.py [T] [spacing_m] [label]"""
import os, sys, time, resource, re
os.environ["SOUNDERSIM_JAX_CACHE_DIR"] = "0"
import numpy as np, jax
VAR = os.environ.get("VAR", "")
_jit = jax.jit
def _cjit(f, **k):
    j = _jit(f, **k)
    def w(*a, **kw):
        if not hasattr(w, "hlo"):
            txt = j.lower(*a, **kw).compile().as_text()
            w.hlo = txt
            print(f"HLO: {len(txt.splitlines())} lines, {txt.count(' fusion(')} fusion calls, "
                  f"{txt.count('fused_computation')} fused computations, loop_fusion {txt.count('kind=kLoop')}, gathers {txt.count(' gather(')}")
        return j(*a, **kw)
    return w
jax.jit = _cjit
import soundersim
from soundersim.kernels import multilayer as ML
from soundersim.kernels.multilayer import refracted_cluttergram
from soundersim.scene import Facets
from soundersim.antenna import pattern_args
from soundersim.config import AntennaConfig
T = int(sys.argv[1]) if len(sys.argv) > 1 else 64
SP = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
label = sys.argv[3] if len(sys.argv) > 3 else ""
rng = np.random.default_rng(5)

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

SSP = float(os.environ.get("SURF_SP", SP))
surf = lattice_yx(-6000, 6000, -3000, 3000, SSP, lambda X, Y: 20 * np.sin(X / 700.0) + 10 * np.cos(Y / 400.0),
                  lambda X, Y: np.stack([-20 / 700 * np.cos(X / 700.0), 10 / 400 * np.sin(Y / 400.0), np.ones_like(X)], -1))
bed = lattice_yx(-6000, 6000, -3000, 3000, SP, lambda X, Y: -1200 + 40 * np.sin(Y / 500.0) + 0.05 * X,
                 lambda X, Y: np.stack([-0.05 * np.ones_like(X), -40 / 500 * np.cos(Y / 500.0), np.ones_like(X)], -1))
F = len(bed.centers)
pos = np.stack([np.linspace(-1200, 1200, T), np.zeros(T), np.full(T, 700.0)], -1)
u_at = np.tile([1.0, 0, 0], (T, 1)); u_ct = np.tile([0, 1.0, 0], (T, 1))
c = 299792458.0; k0 = 2 * np.pi * 195e6 / c
ph = (rng.standard_normal(F) + 1j * rng.standard_normal(F)).astype(np.complex64) / np.sqrt(2)
dph = (rng.standard_normal(F) + 1j * rng.standard_normal(F)).astype(np.complex64) / np.sqrt(2)
gam = (-0.3 - 0.2 * rng.random(F)); damp = 0.1 + 0.2 * rng.random(F)
pat = pattern_args(AntennaConfig(kind="array", n_elements=7, spacing_lam=0.5), u_at, u_ct)
t0 = 2 * 600 / c; dt = 5e-9; ns = 4000
kw = dict(mode="coherent", t0=t0, dt=dt, n_samples=ns, c=c, k0=k0, gamma=gam, pattern=pat,
          roughness=(0.10, 0.886, ph, 12), crossed_sigma=[0.05], diffuse=(damp, dph, 1.0), taper_s=0.05, d_phi_area=True)
if "norough" in VAR: kw["roughness"] = None
if "nodiff" in VAR: kw["diffuse"] = None
if "nopat" in VAR: kw["pattern"] = None
if "nocross" in VAR: kw["crossed_sigma"] = None
if "notaper" in VAR: kw["taper_s"] = None
if "scalargam" in VAR: kw["gamma"] = 0.3
if "noscatter" in VAR:
    import jax.numpy as _jnp
    ML.jax.ops.segment_sum = lambda data, seg, num_segments: _jnp.zeros(num_segments, data.dtype).at[0].set(_jnp.sum(data))
if "newton" in VAR:
    n1, n2 = [int(x) for x in VAR.split("newton")[1].split("_")[:2]]
    _orig = ML._snell_c
    ML._snell_c = lambda p, q, o, nrm, a, b, n_iter: _orig(p, q, o, nrm, a, b, n1 if n_iter == 10 else n2)
if os.environ.get("BLK"): kw["block_size"] = int(os.environ["BLK"])
kw.update({k: v for k, v in (("window_cull", False),) if "window_cull" in refracted_cluttergram.__code__.co_varnames})
refracted_cluttergram(pos, u_ct, bed, [surf], [1.0, 3.17], [0.0, 18.0], **kw)  # compile + warm
r0 = resource.getrusage(resource.RUSAGE_SELF); t = time.perf_counter()
out, drop = refracted_cluttergram(pos, u_ct, bed, [surf], [1.0, 3.17], [0.0, 18.0], **kw)
w = time.perf_counter() - t; r1 = resource.getrusage(resource.RUSAGE_SELF)
cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
# fusion count of the cached jitted kernel (last built)
nfus = "?"
try:
    fn = list(ML._refracted_fn.cache_info() and [])  # placeholder
except Exception:
    pass
print(f"{label:10s} T={T} F={F} wall={w:.2f}s cpu={cpu:.1f}s util={cpu/w:.1f} ns/pair={w/(T*F)*1e9:.0f}")
if len(sys.argv) > 4:
    np.save(sys.argv[4], out)
