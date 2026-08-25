"""Kernel regression harness for the 2026-08-24 runtime work (1a/1b).

  uv run python claude_notes/runtime_opt/kernel_regression.py save  <tag>
  uv run python claude_notes/runtime_opt/kernel_regression.py compare <tag_ref> <tag_new>

Simulates a set of synthetic 2-interface scenes through simulate() with the
production feature set (both roughnesses, grazing fix, array antenna,
per-facet bed gamma + diffuse maps, chirp waveform) and stores field /
dropped / nadir per scene under claude_notes/runtime_opt/ref_<tag>.npz.
"""
import os, sys, time
os.environ.setdefault("SOUNDERSIM_JAX_CACHE_DIR", "0")
import numpy as np
from affine import Affine
from soundersim import synthetic as syn
from soundersim.config import (SimConfig, RadarConfig, FacetConfig, Medium, DemInterface,
                               RoughnessConfig, GrazingFixConfig, AntennaConfig, WaveformConfig)
from soundersim.simulate import simulate

C = 299792458.0
HERE = os.path.dirname(os.path.abspath(__file__))


def radar(alt, depth, f0=200e6, dt=5e-9):
    t_nadir = 2 * (alt + depth * np.sqrt(3.17)) / C
    t0 = 2 * alt / C - 0.8e-6
    ns = int(np.ceil((t_nadir + 3.5e-6 - t0) / dt))
    return RadarConfig(dt=dt, n_samples=ns, t0=t0, f0=f0,
                       waveform=WaveformConfig(kind="chirp", bandwidth=30e6, pulse_length=1e-6),
                       antenna=AntennaConfig(kind="array", n_elements=7, spacing_lam=0.5))


def cfg(rc, spacing, split=False, bed_rough=True, gfix=True, surf_rough=True, ant=None):
    return SimConfig(
        mode="coherent", split_sides=split,
        grazing_fix=GrazingFixConfig(s_eff=0.05) if gfix else None,
        diffuse_exponent=1.0,
        radar=rc if ant is None else rc.model_copy(update={"antenna": ant}),
        facets=FacetConfig(spacing=spacing),
        media=[Medium(name="air", eps_r=1.0), Medium(name="ice", eps_r=3.17, attenuation_db_per_km=18.0),
               Medium(name="bed", eps_r=8.0)],
        interfaces=[DemInterface(name="surface", roughness=RoughnessConfig(sigma_m=0.05, corr_length_m=3.0) if surf_rough else None),
                    DemInterface(name="bed", roughness=RoughnessConfig(sigma_m=0.10, corr_length_m=0.886) if bed_rough else None)])


def attach_maps(scene, seed=0):
    rng = np.random.default_rng(seed)
    shp = scene.dems[0].shape
    g = (-0.3 - 0.2 * rng.random(shp)).astype(np.float64)
    d = (0.1 + 0.2 * rng.random(shp)).astype(np.float64)
    scene.gamma_maps = {"bed": (g, scene.transform, scene.crs)}
    scene.diffuse_maps = {"bed": (d, scene.transform, scene.crs)}
    return scene


def scenes():
    """name -> (scene, SimConfig). Chunk-like: ~40 traces at 7.5 m posting."""
    out = {}
    s = syn.rough_bed_scene(surface=500.0, depth=1500.0, amplitude=30.0, n_traces=40, altitude=700.0,
                            extent=12000.0, posting=32.0, spacing=7.5)
    out["low_flat_maps"] = (attach_maps(s), cfg(radar(700, 1500), 15.0))
    s = syn.tilted_bed_scene(surface=500.0, depth=1200.0, slope_deg=8.0, n_traces=40, altitude=700.0,
                             extent=12000.0, posting=32.0, spacing=7.5)
    out["low_tilted_split"] = (s, cfg(radar(700, 1200), 15.0, split=True))
    s = syn.rough_surface_scene(surface=500.0, depth=800.0, amplitude=60.0, wavelength=1500.0, n_traces=40,
                                altitude=700.0, extent=12000.0, posting=32.0, spacing=7.5)
    out["low_roughsurf_nogfix"] = (attach_maps(s, 1), cfg(radar(700, 800), 15.0, gfix=False, bed_rough=False))
    s = syn.rough_bed_scene(surface=500.0, depth=2000.0, amplitude=80.0, n_traces=24, altitude=14000.0,
                            extent=8000.0, posting=32.0, spacing=50.0)
    out["haps_iso"] = (s, cfg(radar(14000, 2000, f0=60e6, dt=10e-9), 60.0, ant=AntennaConfig(kind="isotropic")))
    s = syn.rough_bed_scene(surface=500.0, depth=1000.0, amplitude=30.0, n_traces=40, altitude=9500.0,
                            extent=6000.0, posting=32.0, spacing=30.0)
    out["mid_smooth"] = (attach_maps(s, 2), cfg(radar(9500, 1000), 35.0, surf_rough=False, bed_rough=False))
    return out


def run(tag):
    res = {}
    for name, (sc, c) in scenes().items():
        t = time.perf_counter(); ds = simulate(sc, c); w = time.perf_counter() - t
        res[f"{name}/field"] = np.asarray(ds.field.values, np.complex64)
        res[f"{name}/dropped"] = np.asarray(ds.dropped_power.values, np.float64)
        res[f"{name}/nadir"] = np.asarray(ds.nadir_twtt.values, np.float64)
        res[f"{name}/wall"] = np.float64(w)
        print(f"{name:24s} {w:6.1f} s  field {res[f'{name}/field'].shape}  dropped {res[f'{name}/dropped'].sum(0)}", flush=True)
    np.savez_compressed(os.path.join(HERE, f"ref_{tag}.npz"), **res)


def compare(a, b):
    A = np.load(os.path.join(HERE, f"ref_{a}.npz")); B = np.load(os.path.join(HERE, f"ref_{b}.npz"))
    for k in A.files:
        if k.endswith("/wall"):
            print(f"{k:32s} {float(A[k]):7.1f} -> {float(B[k]):7.1f} s  ({float(A[k])/max(float(B[k]),1e-9):.2f}x)"); continue
        x, y = A[k], B[k]
        if np.array_equal(x, y):
            print(f"{k:32s} bit-identical"); continue
        if k.endswith("/field"):
            for l in range(x.shape[-1]):
                xl, yl = x[..., l], y[..., l]
                pk = np.abs(xl).max()
                d = np.abs(xl - yl)
                # power-domain check where the signal is not negligible
                m = np.abs(xl) > 1e-3 * pk
                pdb = np.abs(20*np.log10(np.abs(yl[m])/np.abs(xl[m]))) if m.any() else np.zeros(1)
                print(f"{k:32s} layer {l}: max|d|/peak {d.max()/pk:.2e}  rms|d|/peak {np.sqrt((d**2).mean())/pk:.2e}  "
                      f"max |dB| (>-60 dB samples) {pdb.max():.2e}")
        else:
            print(f"{k:32s} max rel diff {np.abs(x-y).max()/max(np.abs(x).max(),1e-300):.2e}")


if __name__ == "__main__":
    if sys.argv[1] == "save": run(sys.argv[2])
    else: compare(sys.argv[2], sys.argv[3])
