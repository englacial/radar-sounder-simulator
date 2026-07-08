"""Regression capture/compare for the jit-caching refactor (session artifact).

Usage:
  uv run python claude_notes/jit_regression_check.py save <path.npz>
  uv run python claude_notes/jit_regression_check.py compare <path.npz>

Runs representative configs (multilayer slab coherent+incoherent, a 3-interface
offset stack coherent with split_sides, single-interface coherent + incoherent)
and saves/compares every output array.
"""

import sys

import numpy as np

import soundersim
from soundersim import synthetic as syn
from soundersim.config import (DemInterface, FacetConfig, Medium,
                               OffsetInterface, RadarConfig, SimConfig)


def _media(*eps, att=None):
    att = att or [0.0] * len(eps)
    return [Medium(name=f"m{i}", eps_r=e, attenuation_db_per_km=a)
            for i, (e, a) in enumerate(zip(eps, att))]


def _slab_cfg(mode, media, *, f0=None, spacing=None, split_sides=False):
    return SimConfig(
        mode=mode, split_sides=split_sides,
        radar=RadarConfig(dt=1e-8, n_samples=1250, t0=0.0, f0=f0),
        facets=FacetConfig(spacing=spacing), media=media,
        interfaces=[DemInterface(name="surface"), DemInterface(name="bed")])


def _stack_cfg(mode, *, f0=None, spacing=15.0, split_sides=False):
    media = _media(1.0, 1.8, 2.4, att=[0.0, 8.0, 12.0])
    interfaces = [DemInterface(name="surface"),
                  OffsetInterface(name="l1", reference="surface", offset=-20.0),
                  OffsetInterface(name="l2", reference="surface", offset=-55.0)]
    media.append(Medium(name="m3", eps_r=3.0, attenuation_db_per_km=15.0))
    return SimConfig(mode=mode, split_sides=split_sides,
                     radar=RadarConfig(dt=1e-8, n_samples=1250, t0=0.0, f0=f0),
                     facets=FacetConfig(spacing=spacing), media=media,
                     interfaces=interfaces)


def run_all():
    out = {}
    slab = syn.slab_scene(surface=500.0, depth=300.0, extent=2000.0,
                          n_traces=3, altitude=1000.0)
    flat = syn.flat_scene(elevation=500.0, extent=2000.0, n_traces=3,
                          altitude=1000.0, posting=50.0)

    ds = soundersim.simulate(
        slab, _slab_cfg("coherent", _media(1.0, 3.17, 6.0, att=[0.0, 10.0, 0.0]),
                        f0=195e6, spacing=15.0))
    out["ml_coh_field"] = ds.field.values
    out["ml_coh_dropped"] = ds.dropped_power.values
    out["ml_coh_nadir"] = ds.nadir_twtt.values

    ds = soundersim.simulate(
        slab, _slab_cfg("incoherent",
                        _media(1.0, 3.17, 6.0, att=[0.0, 10.0, 0.0])))
    out["ml_inc_power"] = ds.power.values
    out["ml_inc_dropped"] = ds.dropped_power.values

    ds = soundersim.simulate(flat, _stack_cfg("coherent", f0=195e6,
                                              split_sides=True))
    out["stack_coh_field"] = ds.field.values
    out["stack_coh_dropped"] = ds.dropped_power.values
    out["stack_coh_nadir"] = ds.nadir_twtt.values

    ds = soundersim.simulate(
        flat, SimConfig(mode="coherent",
                        radar=RadarConfig(dt=1e-8, n_samples=1250, t0=0.0,
                                          f0=195e6),
                        facets=FacetConfig(spacing=15.0),
                        media=_media(1.0, 3.17)))
    out["si_coh_field"] = ds.field.values
    out["si_coh_dropped"] = ds.dropped_power.values

    ds = soundersim.simulate(
        flat, SimConfig(mode="incoherent",
                        radar=RadarConfig(dt=1e-8, n_samples=1250, t0=0.0),
                        facets=FacetConfig(), media=_media(1.0, 3.17),
                        split_sides=True))
    out["si_inc_power"] = ds.power.values
    out["si_inc_dropped"] = ds.dropped_power.values
    return out


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    res = run_all()
    if cmd == "save":
        np.savez(path, **res)
        print(f"saved {len(res)} arrays to {path}")
    else:
        ref = np.load(path)
        worst = 0.0
        for k in ref.files:
            a, b = ref[k], res[k]
            assert a.shape == b.shape, k
            if np.array_equal(a, b):
                print(f"{k:20s} bit-identical")
                continue
            scale = max(np.abs(a).max(), 1e-30)
            d = np.abs(a - b).max() / scale
            worst = max(worst, d)
            print(f"{k:20s} max rel-to-peak diff {d:.3e}")
        print(f"worst relative diff: {worst:.3e}")
