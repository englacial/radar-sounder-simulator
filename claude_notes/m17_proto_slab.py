"""M17 prototype: slab_absolute single run vs closed form (scratch)."""
import time
import numpy as np

import soundersim
from soundersim.config import DemInterface, FacetConfig, Medium, RadarConfig, SimConfig
from soundersim.physics import C, fresnel_normal
from soundersim import synthetic as syn

F0 = 195e6
LAM = C / F0
K0 = 2.0 * np.pi * LAM ** -1


def slab_run(h, d, eps_ice, att=0.0, eps_bed=8.0, u=45.0, fac=0.09):
    n = np.sqrt(eps_ice)
    r_eff = h + d / n
    ext = u * np.sqrt(LAM * r_eff)
    spacing = fac * np.sqrt((LAM / n) * r_eff)
    dt = 20e-9
    t0 = 2.0 * (h - 5.0) / C
    opl_max = np.sqrt(h * h + 2.0 * (ext / 2.0) ** 2) + n * d + 10.0
    n_samples = int(np.ceil((2.0 * opl_max / C - t0) / dt)) + 4
    scene = syn.slab_scene(surface=500.0, depth=d, extent=ext, posting=ext / 64.0,
                           n_traces=2, altitude=h)
    media = [Medium(name="air", eps_r=1.0),
             Medium(name="ice", eps_r=eps_ice, attenuation_db_per_km=att),
             Medium(name="bed", eps_r=eps_bed)]
    cfg = SimConfig(mode="coherent",
                    radar=RadarConfig(dt=dt, n_samples=n_samples, t0=t0, f0=F0),
                    facets=FacetConfig(spacing=spacing), media=media,
                    interfaces=[DemInterface(name="surface"), DemInterface(name="bed")])
    t = time.time()
    ds = soundersim.simulate(scene, cfg)
    wall = time.time() - t
    nfac = (ext / spacing) ** 2
    return ds, wall, nfac, media


def check(ds, eps_ice, eps_bed, att, d_nom):
    n = np.sqrt(eps_ice)
    gam_s = fresnel_normal(1.0, eps_ice)
    gam_b = fresnel_normal(eps_ice, eps_bed)
    tau2 = 1.0 - gam_s ** 2
    out = []
    for tr in range(2):
        opl = C * float(ds.nadir_twtt.sel(layer="bed")[tr]) / 2.0
        h = C * float(ds.nadir_twtt.sel(layer="surface")[tr]) / 2.0
        d = (opl - h) / n
        r_eff = h + d / n
        f = complex(np.asarray(ds.field.sel(layer="bed")[tr].values).sum())
        loss_db = d / 1000.0 * att
        ref = tau2 * gam_b * 10 ** (-loss_db / 10.0) * np.exp(-2j * K0 * opl) / (2.0 * r_eff)
        out.append((abs(f) / abs(ref), np.degrees(np.angle(f / ref))))
        # bin exactness
        bed = np.abs(ds.field.sel(layer="bed")[tr].values).ravel()
        first = np.nonzero(bed)[0][0]
        expect_bin = int(np.floor((2 * opl / C - float(ds.twtt[0])) / (float(ds.twtt[1]) - float(ds.twtt[0]))))
        out[-1] += (first == expect_bin, float(ds.dropped_power.sel(layer="bed")[tr]))
    return out


for (h, d, eps, att) in [(1000.0, 300.0, 3.17, 0.0),
                         (1000.0, 50.0, 3.17, 0.0),
                         (500.0, 300.0, 1.5, 0.0),
                         (1000.0, 300.0, 3.17, 10.0)]:
    ds, wall, nfac, media = slab_run(h, d, eps, att=att)
    res = check(ds, eps, 8.0, att, d)
    print(f"h={h:6.0f} d={d:5.0f} eps={eps:4.2f} att={att:4.1f} "
          f"facets/iface={nfac:9.0f} wall={wall:5.1f}s")
    for tr, (mag, ph, binok, drop) in enumerate(res):
        print(f"   trace {tr}: mag_ratio={mag:.4f} phase_err={ph:+.2f} deg "
              f"bin_exact={binok} bed_dropped={drop:.3e}")
