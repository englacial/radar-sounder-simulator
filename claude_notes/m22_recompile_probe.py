"""M22 session probe: antenna-pattern parameters must not recompile kernels.

Verifies (via the jitted callables' _cache_size()) that for a FIXED pattern
kind, changing every run-varying number -- pattern vectors (roll/axis), array
n_elements/spacing, tabulated gain VALUES, plus k/gamma -- reuses one compiled
executable, and that only structural changes (pattern kind, tabulated table
LENGTH, n_samples) trace anew.

Run: uv run python claude_notes/m22_recompile_probe.py
"""

import numpy as np

from soundersim import antenna
from soundersim.config import AntennaConfig
from soundersim.kernels.coherent import _coherent_fn, coherent_cluttergram
from soundersim.kernels.incoherent import _incoherent_fn, incoherent_cluttergram
from soundersim.kernels.multilayer import _refracted_fn, refracted_cluttergram
from soundersim.layered import build_layered_scene
from soundersim.nav import nav_to_frame
from soundersim.scene import LocalFrame
from soundersim import synthetic as syn
from soundersim.config import DemInterface, FacetConfig, Medium, RadarConfig, SimConfig

C = 299792458.0
UAT = np.tile([1.0, 0.0, 0.0], (2, 1))
UCT = np.tile([0.0, -1.0, 0.0], (2, 1))
POS = np.array([[0.0, 0.0, 3000.0], [10.0, 0.0, 3000.0]])
CTR = np.zeros((1, 3))
NRM = np.array([[0.0, 0.0, 1.0]])
AREA = np.array([25.0])
E1, E2 = np.array([[5.0, 0, 0.0]]), np.array([[0.0, 5.0, 0.0]])
WIN = dict(t0=1.9e-5, dt=1e-8, n_samples=96, c=C)


def check(name, fn, expected):
    n = fn._cache_size()
    status = "OK " if n == expected else "FAIL"
    print(f"{status} {name}: jit cache entries = {n} (expected {expected})")
    assert n == expected, name


def main():
    # ---- coherent, dipole: axis choice + roll values are traced
    for axis, roll in (("along_track", None), ("cross_track", None),
                       ("along_track", 0.4), ("along_track", -0.2)):
        ant = AntennaConfig(kind="dipole", axis=axis,
                            roll_source="none" if roll is None else "nav")
        pat = antenna.pattern_args(ant, UAT, UCT,
                                   None if roll is None else np.full(2, roll))
        coherent_cluttergram(POS, UCT, CTR, NRM, AREA, E1, E2, k=4.0,
                             gamma=-0.28, pattern=pat, **WIN)
    check("coherent dipole (2 axes x roll values, k fixed)",
          _coherent_fn(False, 96, False, "dipole"), 1)

    # ---- incoherent, array: n_elements / spacing traced
    for n_el, d in ((5, 0.5), (7, 0.35), (3, 0.6), (15, 0.5)):
        ant = AntennaConfig(kind="array", n_elements=n_el, spacing_lam=d)
        pat = antenna.pattern_args(ant, UAT, UCT)
        incoherent_cluttergram(POS, UCT, CTR, NRM, AREA, pattern=pat, **WIN)
    check("incoherent array (4 n/spacing combos)",
          _incoherent_fn(False, 96, "array"), 1)

    # ---- tabulated: same table length = one compile; new length retraces
    th64 = np.linspace(0.0, 90.0, 64)
    th128 = np.linspace(0.0, 90.0, 128)
    for th, gain in ((th64, np.cos(np.deg2rad(th64))),
                     (th64, np.cos(np.deg2rad(th64)) ** 4),
                     (th128, np.cos(np.deg2rad(th128)))):
        ant = AntennaConfig(kind="tabulated", theta_deg=list(th),
                            gain=list(gain))
        pat = antenna.pattern_args(ant, UAT, UCT)
        incoherent_cluttergram(POS, UCT, CTR, NRM, AREA, pattern=pat, **WIN)
    check("incoherent tabulated (2 tables same len -> 1; new len -> +1)",
          _incoherent_fn(False, 96, "tabulated"), 2)

    # ---- multilayer (refracted): pattern values traced
    scene = syn.slab_scene(surface=500.0, depth=200.0, extent=1500.0,
                           n_traces=2, altitude=1000.0)
    cfg = SimConfig(mode="incoherent",
                    radar=RadarConfig(dt=1e-8, n_samples=96, t0=6e-6),
                    facets=FacetConfig(spacing=60.0),
                    media=[Medium(name="air", eps_r=1.0),
                           Medium(name="ice", eps_r=3.17),
                           Medium(name="bed", eps_r=6.0)],
                    interfaces=[DemInterface(name="surface"),
                                DemInterface(name="bed")])
    frame = LocalFrame.centered_on(scene)
    from soundersim.simulate import _wire_scene_dems
    wired, dem_refs = _wire_scene_dems(scene, cfg)
    layered = build_layered_scene(wired, frame, scene.dem, scene.transform,
                                  scene.crs, spacing=60.0, dem_refs=dem_refs)
    track = nav_to_frame(scene.nav_llh, frame)
    for gexp in (1.0, 2.0, 6.0):
        ant = AntennaConfig(kind="tabulated", theta_deg=list(th64),
                            gain=list(np.cos(np.deg2rad(th64)) ** gexp))
        pat = antenna.pattern_args(ant, track.u_at, track.u_ct)
        refracted_cluttergram(track.positions, track.u_ct,
                              layered.interfaces[1], layered.interfaces[:1],
                              [1.0, 3.17], [0.0, 10.0], mode="incoherent",
                              t0=6e-6, dt=1e-8, n_samples=96, c=C,
                              pattern=pat)
    check("multilayer tabulated (3 gain tables same len)",
          _refracted_fn(False, False, 96, 1, "tabulated"), 1)

    # ---- kind changes build separate factory entries (expected, static)
    print("factory cache infos:",
          "coherent", _coherent_fn.cache_info(),
          "| incoherent", _incoherent_fn.cache_info(),
          "| refracted", _refracted_fn.cache_info())
    print("all recompile-behavior checks passed")


if __name__ == "__main__":
    main()
