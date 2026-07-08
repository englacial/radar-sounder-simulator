"""M17 prototype: twomedia_field kernel-vs-referee sub-cases (scratch)."""
import time
import numpy as np

from soundersim.compare.brute_force_layered import (
    fermat_crossing_batch, surface_facets, two_media_trace)
from soundersim.compare.fermat import fermat_crossing
from soundersim.kernels.multilayer import refracted_cluttergram
from soundersim.physics import C, fresnel_normal
from soundersim.refraction import snell_crossing

EPS_ICE, EPS_BED = 3.17, 8.0
N_ICE = float(np.sqrt(EPS_ICE))
F0 = 195e6
LAM = C / F0
LAM_ICE = LAM / N_ICE
K0 = 2.0 * np.pi / LAM
GAM_B = float(fresnel_normal(EPS_ICE, EPS_BED))
H, D = 500.0, 60.0
P = np.array([[0.0, 0.0, H]])
UCT = np.array([[0.0, -1.0, 0.0]])
T0, DT, NSAMP = 3.9e-6, 1e-8, 80


def run_case(name, surf_fn, bed_fn, extent, fs, fb, rough_surface=False):
    t = time.time()
    surf = surface_facets(extent, fs, surf_fn)
    bed = surface_facets(extent, fb, bed_fn, z0=-D)
    kern, kdrop = refracted_cluttergram(
        P, UCT, bed, [surf], [1.0, EPS_ICE], [0.0, 0.0], mode="coherent",
        t0=T0, dt=DT, n_samples=NSAMP, c=C, gamma=GAM_B, k0=K0)
    t_kern = time.time() - t

    t = time.time()
    fine = surface_facets(extent, LAM_ICE / 8.0, bed_fn, z0=-D)
    if rough_surface:
        r0 = snell_crossing(P[0], fine.centers, np.zeros(3),
                            np.array([0.0, 0.0, 1.0]), 1.0, N_ICE, xp=np)
        x, s1, s2, opl = fermat_crossing_batch(
            P[0], fine.centers, surf_fn, 1.0, N_ICE, x0=r0.x[:, :2], half0=6.0)
        # true analytic normal at the crossing
        eps = 1e-4
        gx = (np.asarray(surf_fn(x[:, 0] + eps, x[:, 1]))
              - np.asarray(surf_fn(x[:, 0] - eps, x[:, 1]))) / (2 * eps)
        gy = (np.asarray(surf_fn(x[:, 0], x[:, 1] + eps))
              - np.asarray(surf_fn(x[:, 0], x[:, 1] - eps))) / (2 * eps)
        nrm = np.column_stack([-gx, -gy, np.ones(len(x))])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        # spot-check the batch solver against the scalar referee
        rng = np.random.default_rng(1)
        idx = rng.integers(0, len(fine.centers), 5)
        worst = 0.0
        for i in idx:
            fc = fermat_crossing(P[0], fine.centers[i], surf_fn, 1.0, N_ICE)
            worst = max(worst, abs(fc.opl - opl[i]))
        print(f"   batch-vs-scalar opl err (5 spots): {worst:.2e} m")
    else:
        r = snell_crossing(P[0], fine.centers, np.zeros(3),
                           np.array([0.0, 0.0, 1.0]), 1.0, N_ICE, xp=np)
        assert r.valid.all()
        x = r.x
        nrm = np.array([0.0, 0.0, 1.0])
    ref, ref_tot = two_media_trace(P[0], fine, x, nrm, 1.0, EPS_ICE, GAM_B,
                                   K0, T0, DT, NSAMP, C)
    t_ref = time.time() - t

    k = kern[0]
    ratio = k.sum() / ref.sum()
    peak_k = np.abs(k).argmax()
    peak_r = np.abs(ref).argmax()
    # facet-scale profile comparison
    for agg in (2, 4):
        n = (NSAMP // agg) * agg
        pk = (np.abs(k[:n]) ** 2).reshape(-1, agg).sum(1)
        pr = (np.abs(ref[:n]) ** 2).reshape(-1, agg).sum(1)
        m = pr > pr.max() * 1e-2  # above -20 dB
        dbmax = np.abs(10 * np.log10(pk[m] / pr[m])).max()
        print(f"   agg={agg}: max |dB diff| above -20 dB: {dbmax:.3f} "
              f"({m.sum()} bins)")
    print(f"{name}: |ratio|={abs(ratio):.4f} phase={np.degrees(np.angle(ratio)):+.2f} deg "
          f"peak {peak_k} vs {peak_r}, kdrop={float(kdrop[0]):.2e}, "
          f"n_fine={len(fine.centers)}, t_kern={t_kern:.1f}s t_ref={t_ref:.1f}s")


flat = lambda x, y: 0.0 * x
run_case("flat_slab", flat, flat, 80.0, 5.0, 2.5)
rough_bed = lambda x, y: 0.15 * np.sin(2 * np.pi * y / 30.0)
run_case("rough_bed", flat, rough_bed, 80.0, 5.0, 2.0)
for a_s in (0.15, 0.25):
    rough_surf = lambda x, y: a_s * np.sin(2 * np.pi * y / 40.0)
    run_case(f"rough_surface A={a_s}", rough_surf, flat, 60.0, 2.0, 2.5,
             rough_surface=True)

# ---- same-facet chaining degradation sweep (kernel replica vs exact Fermat)
import time as _t
from soundersim.compare.brute_force_layered import local_plane_opl

t = _t.time()
print("\nchaining degradation sweep (Lambda=40 m, facet 2 m, extent 60 m):")
bed_flat = surface_facets(60.0, 2.5, flat, z0=-D)
for a_s in (0.25, 0.5, 1.0, 2.0, 4.0):
    sf = lambda x, y: a_s * np.sin(2 * np.pi * y / 40.0)
    surf = surface_facets(60.0, 2.0, sf)
    xk, opl_k, idx = local_plane_opl(P[0], bed_flat.centers, surf, 1.0, N_ICE)
    r0 = snell_crossing(P[0], bed_flat.centers, np.zeros(3),
                        np.array([0.0, 0.0, 1.0]), 1.0, N_ICE, xp=np)
    xe, s1, s2, opl_e = fermat_crossing_batch(
        P[0], bed_flat.centers, sf, 1.0, N_ICE, x0=r0.x[:, :2],
        half0=max(6.0, 4 * a_s))
    d = np.abs(opl_k - opl_e)
    print(f"  A={a_s:4.2f}: opl err max={d.max():.4e} m rms={np.sqrt((d**2).mean()):.4e} m "
          f"(2-way phase max {2*K0*d.max():.4f} rad)")
print(f"sweep wall: {_t.time()-t:.1f}s")
