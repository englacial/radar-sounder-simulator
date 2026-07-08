"""M19 dev run (temporary): full-scale firn-plateau pipeline, prints the
measurements used to set the test gates. Mirrors the planned test exactly."""
import time

import numpy as np

import soundersim
from soundersim.config import (DemInterface, FacetConfig, Medium,
                               OffsetInterface, RadarConfig, SimConfig)
from soundersim.physics import C, fresnel_normal
from soundersim import synthetic as syn

FIX = "tests/fixtures/firn"


def load_b26(path=f"{FIX}/ngt37C95.2_density.tab", smooth_m=0.1):
    lines = open(path).read().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Depth ice/snow"))
    data = np.loadtxt(path, delimiter="\t", skiprows=hdr + 1)
    z, rho = data[:, 0], data[:, 1]
    k = int(round(smooth_m / np.median(np.diff(z)))) | 1
    rho = np.convolve(rho, np.ones(k) / k, mode="same")
    return z, rho


def eps_kovacs(rho_kgm3):
    return (1.0 + 0.845 * np.asarray(rho_kgm3) / 1000.0) ** 2


def stack(depths, z, rho):
    """Media (air + per-slab MEAN-density firn + substrate) + interfaces."""
    edges = np.concatenate([[0.0], depths, [depths[-1] + 10.0]])
    rho_slab = [rho[(z >= a) & (z < b)].mean() if ((z >= a) & (z < b)).any()
                else float(np.interp(0.5 * (a + b), z, rho))
                for a, b in zip(edges[:-1], edges[1:])]
    eps = [1.0] + [float(e) for e in eps_kovacs(np.array(rho_slab))]
    media = [Medium(name="air", eps_r=1.0)] + [
        Medium(name=f"firn_{i}", eps_r=e) for i, e in enumerate(eps[1:])]
    ifaces = [DemInterface(name="surface")] + [
        OffsetInterface(name=f"L{d:g}", reference="surface", offset=-float(d))
        for d in depths]
    return media, ifaces, np.array(eps)


DEPTHS_C = np.arange(5.0, 100.1, 5.0)
DEPTHS_I = np.arange(10.0, 100.1, 10.0)
H, ELEV, EXTENT = 500.0, 500.0, 600.0
RC = dict(dt=5e-9, n_samples=512, t0=2.0 * (H - 10.0) / C, f0=195e6)


def run(mode, depths, z, rho):
    media, ifaces, eps = stack(depths, z, rho)
    scene = syn.flat_scene(elevation=ELEV, altitude=H, extent=EXTENT,
                           posting=4.0, n_traces=3)
    cfg = SimConfig(mode=mode, radar=RadarConfig(**RC),
                    facets=FacetConfig(spacing=4.0), media=media,
                    interfaces=ifaces)
    t = time.perf_counter()
    ds = soundersim.simulate(scene, cfg)
    print(f"{mode} {len(depths)} layers: {time.perf_counter() - t:.1f} s",
          flush=True)
    return ds, eps


def r_eff(depths, eps):
    """Nadir effective range h + sum(dz_i / n_i) per interface (surface first)."""
    edges = np.concatenate([[0.0], depths])
    dz = np.diff(edges)
    n = np.sqrt(eps[1:1 + len(dz)])
    return H + np.concatenate([[0.0], np.cumsum(dz / n)])


z, rho = load_b26()
ds_c, eps_c = run("coherent", DEPTHS_C, z, rho)
ds_i, eps_i = run("incoherent", DEPTHS_I, z, rho)

# per-layer coherent: total field over the window (haynes convention)
f = ds_c.field.sum("twtt").values                # (traces, layers)
p_c = (np.abs(f) ** 2).mean(axis=0)
comp_c = (r_eff(DEPTHS_C, eps_c) / H) ** 2
r_c = 10 * np.log10(p_c * comp_c / (p_c[0] * comp_c[0]))
tot_c = ds_c.power.sum("twtt").values.mean(axis=0)
drop_c = ds_c.dropped_power.values.mean(axis=0)
print("coherent dropped/total per layer max:",
      float((drop_c / np.maximum(tot_c, 1e-300)).max()))

p_i = ds_i.power.sum("twtt").values.mean(axis=0)
comp_i = (r_eff(DEPTHS_I, eps_i) / H) ** 2
r_i = 10 * np.log10(p_i * comp_i / (p_i[0] * comp_i[0]))

gam = fresnel_normal(eps_c[:-1], eps_c[1:])      # per interface
print("\n z(m)   R_coh(dB)   gamma(dB)")
for d, rc_, g in zip(np.r_[0.0, DEPTHS_C], r_c, 20 * np.log10(np.abs(gam))):
    print(f"{d:6.1f}  {rc_:8.2f}   {g:8.1f}")
print("\n z(m)   R_inc(dB)  [incoherent nodes]")
for d, ri_ in zip(np.r_[0.0, DEPTHS_I], r_i):
    print(f"{d:6.1f}  {ri_:8.2f}")

zc = np.r_[0.0, DEPTHS_C]
lay = zc > 0
plat = (zc >= 2.0) & (zc <= 40.0)
deep = zc >= 70.0
print("\nplateau band mean (2-40 m):", r_c[plat].mean())
print("plateau band max/min:", r_c[plat].max(), r_c[plat].min())
print("deep band mean (70-100 m):", r_c[deep].mean())
print("rolloff (plateau mean - deep mean):", r_c[plat].mean() - r_c[deep].mean())
zi = np.r_[0.0, DEPTHS_I]
print("incoherent total decay (first layer - last):", r_i[1] - r_i[-1])
print("incoherent band means 2-40 / 70-100:",
      r_i[(zi >= 2) & (zi <= 40)].mean(), r_i[zi >= 70].mean())

np.savez("claude_notes/m19_dev_profile.npz", zc=zc, r_c=r_c, zi=zi, r_i=r_i,
         gam=gam, p_c=p_c, p_i=p_i)
soundersim.save(ds_c, "claude_notes/m19_dev_coh.nc")
soundersim.save(ds_i, "claude_notes/m19_dev_inc.nc")
