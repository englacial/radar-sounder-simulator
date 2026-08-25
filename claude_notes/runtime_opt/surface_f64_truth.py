"""Surface kernel: error of old vs new f32 kernel against an f64 NumPy truth
(lpa_contributions xp=np, binned with bincount). usage: surface_f64_truth.py old|new"""
import os, sys
os.environ["SOUNDERSIM_JAX_CACHE_DIR"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
if sys.argv[1] == "old": sys.path.insert(0, os.path.join(HERE, "old_tree", "src"))
import numpy as np, soundersim
from soundersim.kernels.coherent import coherent_cluttergram, lpa_contributions
print("using", os.path.dirname(soundersim.__file__))
rng = np.random.default_rng(11)
T = 6
xs = np.arange(-4000, 4000, 30.0); ys = np.arange(-3000, 3000, 30.0)
X, Y = np.meshgrid(xs, ys, indexing="ij"); Z = 30 * np.sin(X / 900.0) + 20 * np.cos(Y / 600.0)
centers = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1); F = len(centers)
n = np.stack([-30 / 900 * np.cos(X / 900.0), 20 / 600 * np.sin(Y / 600.0), np.ones_like(X)], -1).reshape(-1, 3)
normals = n / np.linalg.norm(n, axis=-1, keepdims=True)
e1 = np.tile([30.0, 0, 0], (F, 1)); e2 = np.tile([0, 30.0, 0], (F, 1)); areas = np.full(F, 900.0)
pos = np.stack([np.linspace(-600, 600, T), np.zeros(T), np.full(T, 9500.0)], -1)
u_ct = np.tile([0, 1.0, 0], (T, 1))
c = 299792458.0; k = 2 * np.pi * 195e6 / c; t0 = 2 * 9400 / c; dt = 5e-9; ns = 3000
f, _ = coherent_cluttergram(pos, u_ct, centers, normals, areas, e1, e2, k=k, gamma=0.3, t0=t0, dt=dt,
                            n_samples=ns, c=c, taper_s=0.05)
truth = np.zeros((T, ns), np.complex128)
for i in range(T):
    contrib, r = lpa_contributions(pos[i], centers, normals, areas, e1, e2, k, 0.3, r_ref=0.0, xp=np, taper_s=0.05)
    b = np.floor((2 * r / c - t0) / dt).astype(int); ok = (b >= 0) & (b < ns)
    truth[i] = np.bincount(b[ok], weights=contrib[ok].real, minlength=ns) + 1j * np.bincount(b[ok], weights=contrib[ok].imag, minlength=ns)
pk = np.abs(truth).max(); d = np.abs(f - truth)
m = np.abs(truth) > 1e-3 * pk
print(f"{sys.argv[1]}: max|err|/peak {d.max()/pk:.2e}  rms|err|/peak {np.sqrt((d**2).mean())/pk:.2e}  "
      f"max |dB err| (>-60 dB) {np.abs(20*np.log10(np.abs(f[m])/np.abs(truth[m]))).max():.2e}  "
      f"median |dB err| (>-60 dB) {np.median(np.abs(20*np.log10(np.abs(f[m])/np.abs(truth[m])))):.2e}")
