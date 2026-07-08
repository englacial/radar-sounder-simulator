"""High-level simulation entry point.

Single-interface configs run the stage-2 path unchanged. Multilayer configs
(len(sim_config.interfaces) > 1) build a ``LayeredScene`` and add the
refracted-path contributions of every subsurface interface via
``kernels.multilayer``; the output Dataset gains the ``layer`` dimension
(docs/output.md) with per-layer nadir_twtt and dropped_power.

Reflection-coefficient convention: the TARGET interface uses its
normal-incidence coefficient (``fresnel_normal``) in both the single-interface
and multilayer paths -- the stage-2 convention, kept so that the surface layer
is bit-compatible with stage 2 and the eps->1 multilayer reduction to a
single-interface run is exact. Interface CROSSINGS use the angle-dependent TE
coefficients (``physics.fresnel_te``) where the angle dependence is
first-order (transmission tapering, TIR-adjacent grazing); upgrading the
target reflection to Gamma(theta) is a one-line change in
kernels/multilayer.py left for when a validation case demands it.

XLA compilation cache: the first ``simulate()`` call enables jax's persistent
compilation cache so kernel compiles (seconds to minutes for deep multilayer
stacks) survive across processes. Default location
``~/.cache/soundersim/jax/``; override with the environment variable
``SOUNDERSIM_JAX_CACHE_DIR=/some/dir``, or disable with
``SOUNDERSIM_JAX_CACHE_DIR=0``. Only compiles taking >= 1 s are persisted.
"""

import os
from pathlib import Path

import numpy as np

from .config import DemInterface, SimConfig
from .kernels.coherent import coherent_cluttergram
from .kernels.incoherent import incoherent_cluttergram
from .kernels.multilayer import refracted_cluttergram
from .layered import build_layered_scene
from .nav import nav_to_frame
from .output import build_dataset
from .physics import fresnel_normal
from .scene import LocalFrame, build_facets, check_facet_size
from .synthetic import SyntheticScene

_jax_cache_configured = False


def _configure_jax_cache():
    """Enable jax's persistent (on-disk) compilation cache, once per process.

    See the module docstring: SOUNDERSIM_JAX_CACHE_DIR overrides the default
    ``~/.cache/soundersim/jax/``; the value ``0`` (or empty) disables.
    """
    global _jax_cache_configured
    if _jax_cache_configured:
        return
    _jax_cache_configured = True
    loc = os.environ.get("SOUNDERSIM_JAX_CACHE_DIR")
    if loc is not None and loc.strip() in ("", "0"):
        return
    path = Path(loc) if loc else Path.home() / ".cache" / "soundersim" / "jax"
    path.mkdir(parents=True, exist_ok=True)
    import jax

    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def simulate(scene: SyntheticScene, sim_config: SimConfig):
    """Run a simulation on a scene; returns the output xarray Dataset.

    Builds the local frame (centered on the DEM), facets, and nav track
    internally, runs the kernel for ``sim_config.mode``, and assembles the
    Dataset per docs/output.md. Multilayer configs need a scene providing the
    interface DEMs (e.g. ``synthetic.MultilayerScene``): a bare
    ``DemInterface`` at position i takes the scene's i-th DEM.
    """
    _configure_jax_cache()
    if len(sim_config.interfaces) > 1:
        return _simulate_multilayer(scene, sim_config)
    frame = LocalFrame.centered_on(scene)
    facets = build_facets(scene.dem, scene.transform, scene.crs, frame,
                          spacing=sim_config.facets.spacing)
    track = nav_to_frame(scene.nav_llh, frame)
    rc = sim_config.radar
    window = dict(t0=rc.t0, dt=rc.dt, n_samples=rc.n_samples, c=rc.c,
                  split_sides=sim_config.split_sides)

    if sim_config.mode == "incoherent":
        power, dropped = incoherent_cluttergram(
            track.positions, track.u_ct, facets.centers, facets.normals,
            facets.areas, **window)
        return build_dataset(power, dropped, scene=scene, frame=frame,
                             facets=facets, track=track, sim_config=sim_config)

    # coherent
    if rc.f0 is None:
        raise ValueError("coherent mode requires radar.f0 to be set")
    lam = rc.wavelength
    media = sim_config.media
    gamma = fresnel_normal(media[0].eps_r, media[1].eps_r)
    # Conservative minimum slant range (vertical clearance to the highest
    # facet) for the Fresnel-zone facet-size check; warns when LPA is at risk.
    min_range = max(float(track.positions[:, 2].min()
                          - facets.centers[:, 2].max()), lam)
    check_facet_size(facets, lam, min_range)
    field, dropped = coherent_cluttergram(
        track.positions, track.u_ct, facets.centers, facets.normals,
        facets.areas, facets.e1, facets.e2, k=2.0 * np.pi / lam, gamma=gamma,
        **window)
    return build_dataset(np.abs(field) ** 2, dropped, field=field, scene=scene,
                         frame=frame, facets=facets, track=track,
                         sim_config=sim_config)


def _wire_scene_dems(scene, sim_config):
    """Map bare DemInterfaces (no path/ref) onto the scene's DEM stack.

    Interface i takes ``scene.dems[i]`` (MultilayerScene); index 0 already
    defaults to the reference DEM inside ``build_layered_scene``. Returns
    (config for the scene builder, dem_refs).
    """
    dem_refs = {}
    interfaces = list(sim_config.interfaces)
    dems = getattr(scene, "dems", None)
    for i, ic in enumerate(interfaces):
        if (i > 0 and isinstance(ic, DemInterface) and ic.path is None
                and ic.ref is None):
            if dems is None or i >= len(dems):
                raise ValueError(
                    f"interface {i} ({ic.name or 'unnamed'}) needs a DEM: the "
                    "scene provides none (give the interface a path/ref or use "
                    "a MultilayerScene)")
            key = f"_scene_dem_{i}"
            dem_refs[key] = (dems[i], scene.transform, scene.crs)
            interfaces[i] = ic.model_copy(update={"ref": key})
    return sim_config.model_copy(update={"interfaces": interfaces}), dem_refs


def _simulate_multilayer(scene, sim_config):
    """Multilayer run: surface via the stage-2 kernels (bit-compatible),
    deeper interfaces via the refracted-path kernel; one layer per interface."""
    frame = LocalFrame.centered_on(scene)
    wired, dem_refs = _wire_scene_dems(scene, sim_config)
    layered = build_layered_scene(wired, frame, scene.dem, scene.transform,
                                  scene.crs, spacing=sim_config.facets.spacing,
                                  dem_refs=dem_refs)
    track = nav_to_frame(scene.nav_llh, frame)
    rc = sim_config.radar
    window = dict(t0=rc.t0, dt=rc.dt, n_samples=rc.n_samples, c=rc.c,
                  split_sides=sim_config.split_sides)
    eps = [m.eps_r for m in sim_config.media]
    att = [m.attenuation_db_per_km for m in sim_config.media]
    coherent = sim_config.mode == "coherent"
    surf = layered.interfaces[0]

    if coherent:
        if rc.f0 is None:
            raise ValueError("coherent mode requires radar.f0 to be set")
        lam = rc.wavelength
        k0 = 2.0 * np.pi / lam
        for j, f in enumerate(layered.interfaces):
            lam_j = lam / np.sqrt(eps[j])  # in-medium wavelength at interface j
            min_range = max(float(track.positions[:, 2].min()
                                  - f.centers[:, 2].max()), lam_j)
            check_facet_size(f, lam_j, min_range)

    outs, drops = [], []
    for j, target in enumerate(layered.interfaces):
        gamma_j = fresnel_normal(eps[j], eps[j + 1])
        if j == 0:
            if coherent:
                out, drop = coherent_cluttergram(
                    track.positions, track.u_ct, surf.centers, surf.normals,
                    surf.areas, surf.e1, surf.e2, k=k0, gamma=gamma_j,
                    **window)
            else:
                out, drop = incoherent_cluttergram(
                    track.positions, track.u_ct, surf.centers, surf.normals,
                    surf.areas, **window)
        else:
            out, drop = refracted_cluttergram(
                track.positions, track.u_ct, target, layered.interfaces[:j],
                eps[:j + 1], att[:j + 1], mode=sim_config.mode, gamma=gamma_j,
                k0=k0 if coherent else None, **window)
        outs.append(out)
        drops.append(drop)

    stacked = np.stack(outs, axis=-1)
    dropped = np.stack(drops, axis=-1)
    nadir = _nadir_twtt_layers(track.positions, layered.interfaces, eps, rc.c)
    kw = dict(scene=scene, frame=frame, facets=surf, track=track,
              sim_config=sim_config, layers=layered.names, nadir_twtt=nadir)
    if coherent:
        return build_dataset(np.abs(stacked) ** 2, dropped, field=stacked,
                             **kw)
    return build_dataset(stacked, dropped, **kw)


def _nadir_twtt_layers(pos, interfaces, eps, c):
    """Per-layer nadir twtt: vertical two-way time through the stack with
    in-medium speeds, via the horizontally nearest facet center per interface
    (the single-interface convention, see output.py)."""
    n_traces = pos.shape[0]
    nadir = np.empty((n_traces, len(interfaces)))
    prev_z = pos[:, 2]
    acc = np.zeros(n_traces)
    for j, f in enumerate(interfaces):
        d2 = ((f.centers[None, :, :2] - pos[:, None, :2]) ** 2).sum(-1)
        z = f.centers[d2.argmin(axis=1), 2]
        acc = acc + np.sqrt(eps[j]) * (prev_z - z)
        nadir[:, j] = 2.0 * acc / c
        prev_z = z
    return nadir
