"""M17 prototype: bed_falloff sweeps (scratch). Reuses slab helper."""
import time
import numpy as np

import importlib.util
spec = importlib.util.spec_from_file_location(
    "proto_slab", "claude_notes/m17_proto_slab.py")
# avoid re-running the proto script body; inline the helper instead
import soundersim
from soundersim.config import DemInterface, FacetConfig, Medium, RadarConfig, SimConfig
from soundersim.physics import C, fresnel_normal
from soundersim import synthetic as syn

F0 = 195e6
LAM = C / F0
K0 = 2.0 * np.pi / LAM
EPS_ICE, EPS_BED = 3.17, 8.0
N = np.sqrt(EPS_ICE)


def slab_run(h, d, eps_ice=EPS_ICE, att=0.0, eps_bed=EPS_BED, u=45.0, fac=0.09):
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
    return soundersim.simulate(scene, cfg)


def point(ds):
    """(r_eff, measured bed power, predicted power) mean over traces."""
    gam_b = fresnel_normal(EPS_ICE, EPS_BED)
    tau2 = 1.0 - fresnel_normal(1.0, EPS_ICE) ** 2
    p, r = [], []
    for tr in range(2):
        opl = C * float(ds.nadir_twtt.sel(layer="bed")[tr]) / 2.0
        h = C * float(ds.nadir_twtt.sel(layer="surface")[tr]) / 2.0
        d = (opl - h) / N
        r.append(h + d / N)
        f = complex(np.asarray(ds.field.sel(layer="bed")[tr].values).sum())
        p.append(abs(f) ** 2)
    r_eff = float(np.mean(r))
    return r_eff, float(np.mean(p)), (tau2 * gam_b / (2 * r_eff)) ** 2


t = time.time()
for label, pts in (("altitude sweep (d=300)",
                    [(h, 300.0) for h in (500.0, 1000.0, 2000.0, 4000.0)]),
                   ("depth sweep (h=800)",
                    [(800.0, d) for d in (100.0, 500.0, 1000.0, 2000.0)])):
    rr, pp, pred = [], [], []
    for h, d in pts:
        ds = slab_run(h, d)
        a, b, c = point(ds)
        rr.append(a), pp.append(b), pred.append(c)
    slope = np.polyfit(np.log(rr), np.log(pp), 1)[0]
    lev = np.abs(np.array(pp) / np.array(pred) - 1.0).max()
    print(f"{label}: slope={slope:+.4f}  max|level ratio-1|={lev:.4f} "
          f"r_eff={np.round(rr,1).tolist()}")
print(f"wall: {time.time()-t:.1f}s")
