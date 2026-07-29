"""Reflectivity statistics: full-resolution B26 profile vs N-point-sampled stack.

Question: how much per-range-cell reflection power (dB rel surface) does
point-sampling the Kovacs permittivity at N depths discard, over the MCoRDS
band, in the 20-70 m band?

Method (mirrors C&S 2020 Sec IV-C / ~/Documents/clutter transfer_matrix.py):
  - full-res: bin the 1 mm profile into range-resolution windows, transfer
    matrix per bin -> effective |r|^2 per range cell.
  - N-sampled: interfaces at equal depths Z_TOP..ZMAX with point-sampled
    0.1 m-smoothed eps (exactly tools/run_b26_comparison.py layered_cfg);
    per-interface Fresnel gamma; per-range-cell power = sum of gammas in cell.
  - segment-aggregated (the PROPOSED fix): N interfaces, each carrying the
    transfer-matrix |r| of the full-res profile between segment midpoints.
All normalized to the air->firn surface power reflectivity.
"""
import numpy as np
from pathlib import Path

FIX = Path("/home/thomasteisberg/Documents/coherent-radar-simulator/tests/fixtures/firn")
C = 299792458.0
F0, B = 195e6, 30e6
LAM = C / F0
Z_TOP, N_LIST = 1.0, (10, 20, 40, 80)
BAND = (20.0, 70.0)

# ---- load ----
lines = (FIX / "ngt37C95.2_density.tab").read_text().splitlines()
hdr = next(i for i, l in enumerate(lines) if l.startswith("Depth ice/snow"))
data = np.loadtxt(FIX / "ngt37C95.2_density.tab", delimiter="\t", skiprows=hdr + 1)
z, rho = data[:, 0], data[:, 1]
dz = np.median(np.diff(z))

def smooth(x, meters):
    k = int(round(meters / dz)) | 1
    box = np.ones(k)
    return np.convolve(x, box, "same") / np.convolve(np.ones_like(x), box, "same")

def eps_kovacs(r):  # C&S Eq 4
    return (1.0 + 0.845 * r / 1000.0) ** 2

rho_s = smooth(rho, 0.1)             # the sim pipeline's profile
eps_full = eps_kovacs(rho)           # raw 1 mm
eps_s = eps_kovacs(rho_s)            # 0.1 m smoothed
n_full, n_s = np.sqrt(eps_full), np.sqrt(eps_s)
ZMAX = z.max()
eps_mean = eps_full.mean()
res_firn = 1.44 * C / (2 * B) / np.sqrt(eps_mean)   # hann-broadened, in firn
print(f"grid: {len(z)} pts, dz={dz*1000:.1f} mm, ZMAX={ZMAX:.2f} m")
print(f"eps_mean={eps_mean:.3f}  in-firn res={res_firn:.2f} m  lam_firn={LAM/np.sqrt(eps_mean):.2f} m")

gamma_surf = ((1 - n_s[np.argmin(np.abs(z - 0.2))]) / (1 + n_s[np.argmin(np.abs(z - 0.2))])) ** 2
print(f"surface gamma (air->firn@0.2m, smoothed): {10*np.log10(gamma_surf):.2f} dB")

# ---- transfer matrix (Yeh; same as clutter repo) ----
def tmm_r(n_stack, d, lam):
    kx = 2 * np.pi / lam * n_stack
    phi = kx[1:-1] * d
    M = np.eye(2, dtype=complex)
    for m in range(len(d)):
        ratio = kx[m + 1] / kx[m]
        D = 0.5 * np.array([[1 + ratio, 1 - ratio], [1 - ratio, 1 + ratio]], dtype=complex)
        P = np.diag([np.exp(-1j * phi[m]), np.exp(1j * phi[m])])
        M = M @ D @ P
    ratio = kx[-1] / kx[-2]
    M = M @ (0.5 * np.array([[1 + ratio, 1 - ratio], [1 - ratio, 1 + ratio]], dtype=complex))
    return M[1, 0] / M[0, 0]

def fullres_bins(nprof, res):
    edges = np.arange(z[0], ZMAX, res)
    idx = np.searchsorted(z, edges)
    out_z, out_p = [], []
    for i in range(len(edges)):
        s = idx[i]
        e = idx[i + 1] if i + 1 < len(edges) else len(z)
        if e - s < 1:
            continue
        n_stack = np.concatenate(([nprof[s - 1] if s else 1.0], nprof[s:e],
                                  [nprof[e] if e < len(nprof) else nprof[-1]]))
        r = tmm_r(n_stack, np.full(e - s, dz), LAM)
        out_z.append(edges[i] + res / 2)
        out_p.append(abs(r) ** 2)
    return np.array(out_z), np.array(out_p)

def band_db(zc, p, band=BAND):
    m = (zc >= band[0]) & (zc <= band[1])
    return 10 * np.log10(p[m].mean() / gamma_surf)

# full-res, raw 1 mm and 0.1 m-smoothed inputs
for name, nprof in [("raw 1mm", n_full), ("0.1m-smoothed", n_s)]:
    zc, p = fullres_bins(nprof, res_firn)
    print(f"full-res TMM [{name}]: 20-70 m band = {band_db(zc, p):+.2f} dB rel surface")

# sanity vs C&S Fig 9b digitized (their sim, MCoRDS3, 6 m bins)
zc6, p6 = fullres_bins(n_full, 6.0)
fig9 = np.loadtxt(FIX / "fig09b_digitized.csv", delimiter=",", skiprows=4)
m9 = (fig9[:, 0] >= BAND[0]) & (fig9[:, 0] <= BAND[1])
print(f"full-res TMM [raw, 6.0 m bins like C&S]: band = {band_db(zc6, p6):+.2f} dB; "
      f"C&S fig9b digitized band mean = {fig9[m9, 1].mean():+.2f} dB (their normalization)")

# ---- N-point-sampled stacks (current sim inputs) ----
def point_n(depth):
    return n_s[np.argmin(np.abs(z - depth))]

print("\nN-point-sampled equal-placement stacks (current sim inputs):")
for N in N_LIST:
    depths = np.linspace(Z_TOP, ZMAX, N)
    nn = np.array([point_n(d) for d in depths] + [point_n(depths[-1] + 1.0)])
    gam = ((nn[:-1] - nn[1:]) / (nn[:-1] + nn[1:])) ** 2   # interfaces at depths[1:] + substrate
    iface_z = np.concatenate([depths[1:], [depths[-1] + 0.01]])
    m = (iface_z >= BAND[0]) & (iface_z <= BAND[1])
    # mean power per range cell in band = (band gamma sum) * res / bandwidth_m
    per_cell = gam[m].sum() * res_firn / (BAND[1] - BAND[0])
    print(f"  N={N:3d}: interfaces in band={m.sum():3d}, "
          f"median iface gamma={10*np.log10(np.median(gam[m])):+.1f} dB, "
          f"band level={10*np.log10(per_cell / gamma_surf):+.2f} dB rel surface")

# ---- segment-aggregated N=40 (proposed fix): TMM r per inter-layer segment ----
print("\nSegment-aggregated stacks (proposed fix -- full-res TMM per segment):")
for N in (20, 40, 80):
    depths = np.linspace(Z_TOP, ZMAX, N)
    mids = np.concatenate([[z[0]], (depths[:-1] + depths[1:]) / 2, [ZMAX]])
    seg_p = []
    for a, b in zip(mids[:-1], mids[1:]):
        s, e = np.searchsorted(z, a), np.searchsorted(z, b)
        if e - s < 2:
            continue
        n_stack = np.concatenate(([n_full[s - 1] if s else 1.0], n_full[s:e],
                                  [n_full[min(e, len(n_full) - 1)]]))
        seg_p.append(abs(tmm_r(n_stack, np.full(e - s, dz), LAM)) ** 2)
    seg_p = np.array(seg_p)
    m = (depths >= BAND[0]) & (depths <= BAND[1])
    per_cell = seg_p[m].sum() * res_firn / (BAND[1] - BAND[0])
    print(f"  N={N:3d}: band level={10*np.log10(per_cell / gamma_surf):+.2f} dB rel surface")

# ---- transmission loss check (does higher contrast eat itself?) ----
g_full = ((n_full[:-1] - n_full[1:]) / (n_full[:-1] + n_full[1:])) ** 2
m45 = z[:-1] <= 45.0
loss_full = -10 * np.log10(np.prod(1 - g_full[m45]) ** 2)
depths = np.linspace(Z_TOP, ZMAX, 40)
nn = np.array([point_n(d) for d in depths])
g40 = ((nn[:-1] - nn[1:]) / (nn[:-1] + nn[1:])) ** 2
loss_40 = -10 * np.log10(np.prod(1 - g40[depths[1:] <= 45.0]) ** 2)
print(f"\ntwo-way transmission loss to 45 m: full-res {loss_full:.3f} dB, N=40 {loss_40:.4f} dB")

# ---- where does the contrast live? variance vs smoothing scale ----
print("\nBragg-scale content: band-level of full-res TMM vs pre-smoothing of the profile:")
for sm in (0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0):
    nprof = n_full if sm == 0 else np.sqrt(eps_kovacs(smooth(rho, sm)))
    zc, p = fullres_bins(nprof, res_firn)
    print(f"  smooth {sm:4.2f} m: band = {band_db(zc, p):+.2f} dB rel surface")
