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
import warnings
from pathlib import Path

import numpy as np

from .antenna import pattern_args
from .config import DemInterface, SimConfig
from .kernels.coherent import coherent_cluttergram
from .kernels.incoherent import incoherent_cluttergram
from .kernels.multilayer import refracted_cluttergram
from .layered import build_layered_scene
from .nav import nav_to_frame
from .output import build_dataset
from .physics import fresnel_normal
from .roughness import n_terms_for, speckle_phasors
from .scene import LocalFrame, build_facets, check_facet_size
from .synthetic import SyntheticScene
from .waveform import apply_waveform

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


def _warn_quantization_alias(rc):
    """Warn when a chirped coherent run leaves the envelope-quantization
    artifact in band (M21 measurement, claude_notes/m20_m21_findings.md).

    The binned trace carries the carrier phase exactly but quantizes envelope
    delays to dt, which plants quantization noise at the aliased carrier
    frequency f_a = f0 - round(f0*dt)/dt. If |f_a| < B/2 the compressed
    pulse passes it (measured on a smooth surface at 195 MHz / dt = 5 ns:
    off-nadir floor up to ~ -18 dB of the surface peak with plain binning,
    ~ -23..-32 dB with interp_bins); if |f_a| > B/2 the convolution rejects
    it entirely -- choosing dt with the alias out of band beats any binning
    fix.
    """
    wf = rc.waveform
    if wf.kind != "chirp" or wf.interp_bins:
        return
    f_alias = abs(rc.f0 - round(rc.f0 * rc.dt) / rc.dt)
    if f_alias < wf.bandwidth / 2.0:
        warnings.warn(
            f"chirped run with the envelope-quantization alias in band "
            f"(|f0 - round(f0*dt)/dt| = {f_alias / 1e6:.1f} MHz < B/2 = "
            f"{wf.bandwidth / 2e6:.1f} MHz): expect a nonphysical off-nadir "
            "floor (measured ~ -18 dB of the surface peak on smooth scenes "
            "at 195 MHz / 5 ns). Set waveform.interp_bins=true (measured "
            "8-16 dB lower) or choose dt so the alias falls out of band")


def _antenna_pattern(rc, scene, track):
    """Kernel pattern args for ``rc.antenna`` (None when isotropic).

    ``roll_source="nav"`` takes per-trace roll (radians, positive = right wing
    down) from ``scene.nav_roll``; scenes without roll data (all synthetic
    scenes) use roll = 0.
    """
    ant = rc.antenna
    roll = None
    if ant.roll_source == "nav":
        roll = getattr(scene, "nav_roll", None)
        if roll is None:
            roll = np.zeros(len(track.positions))
    return pattern_args(ant, track.u_at, track.u_ct, roll)


def _gamma_map_values(gm, facets, frame):
    """Per-facet FIELD reflection coefficients from a map-referenced grid.

    ``gm`` is ``(grid, transform, crs)`` -- a 2-D array of signed field
    coefficients on an affine grid (any CRS). Facet centers are converted
    local frame -> llh -> map CRS and the grid is sampled bilinearly
    (edge-clamped). Scenes attach these as ``scene.gamma_maps`` = {interface
    name: (grid, transform, crs)} (the ``nav_roll`` pattern); consumed by the
    multilayer coherent path only.
    """
    from pyproj import Transformer

    grid, transform, crs = gm
    grid = np.asarray(grid, np.float64)
    llh = frame.local_to_llh(facets.centers)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        llh[:, 1], llh[:, 0])
    cols, rows = (~transform) * (np.asarray(x), np.asarray(y))
    r = np.clip(rows - 0.5, 0.0, grid.shape[0] - 1.0)
    c = np.clip(cols - 0.5, 0.0, grid.shape[1] - 1.0)
    r0 = np.clip(np.floor(r).astype(int), 0, grid.shape[0] - 2)
    c0 = np.clip(np.floor(c).astype(int), 0, grid.shape[1] - 2)
    fr, fc = r - r0, c - c0
    return (grid[r0, c0] * (1 - fr) * (1 - fc)
            + grid[r0, c0 + 1] * (1 - fr) * fc
            + grid[r0 + 1, c0] * fr * (1 - fc)
            + grid[r0 + 1, c0 + 1] * fr * fc)


def _roughness_args(iface, facets, k_local, seed):
    """Kernel roughness tuple for one interface, or None when smooth.

    ``k_local`` is the wavenumber in the medium ABOVE the interface (the
    facets' local medium); the series length covers the nadir worst case
    sigma^2 K^2 = (2 k_local sigma)^2. Warns when the correlation length
    exceeds the facet size (docs/roughness.md validity limit: such roughness
    is facet tilt and belongs in the DEM).
    """
    rc = iface.roughness
    if rc is None or rc.sigma_m == 0.0:
        return None
    edge = max(float(np.linalg.norm(facets.e1, axis=1).max()),
               float(np.linalg.norm(facets.e2, axis=1).max()))
    if rc.corr_length_m > edge:
        warnings.warn(
            f"interface {iface.name or '?'}: roughness corr_length_m = "
            f"{rc.corr_length_m:g} m exceeds the facet size ({edge:.1f} m); "
            "roughness at scales above the facet is really facet tilt -- "
            "put it in the DEM (docs/roughness.md)")
    n_terms = n_terms_for((2.0 * k_local * rc.sigma_m) ** 2)
    phasors = speckle_phasors(len(facets.centers), seed)
    if rc.acf != "gaussian":
        return (rc.sigma_m, rc.corr_length_m, phasors, n_terms, rc.acf)
    return (rc.sigma_m, rc.corr_length_m, phasors, n_terms)


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
                  split_sides=sim_config.split_sides,
                  pattern=_antenna_pattern(rc, scene, track))

    if sim_config.mode == "incoherent":
        power, dropped = incoherent_cluttergram(
            track.positions, track.u_ct, facets.centers, facets.normals,
            facets.areas, **window)
        power = apply_waveform(power, rc, "incoherent")
        return build_dataset(power, dropped, scene=scene, frame=frame,
                             facets=facets, track=track, sim_config=sim_config)

    # coherent
    if rc.f0 is None:
        raise ValueError("coherent mode requires radar.f0 to be set")
    _warn_quantization_alias(rc)
    lam = rc.wavelength
    media = sim_config.media
    gamma = fresnel_normal(media[0].eps_r, media[1].eps_r)
    # Conservative minimum slant range (vertical clearance to the highest
    # facet) for the Fresnel-zone facet-size check; warns when LPA is at risk.
    min_range = max(float(track.positions[:, 2].min()
                          - facets.centers[:, 2].max()), lam)
    check_facet_size(facets, lam, min_range)
    rough = _roughness_args(sim_config.interfaces[0], facets,
                            2.0 * np.pi / lam, (sim_config.roughness_seed, 0))
    gfx = sim_config.grazing_fix
    field, dropped = coherent_cluttergram(
        track.positions, track.u_ct, facets.centers, facets.normals,
        facets.areas, facets.e1, facets.e2, k=2.0 * np.pi / lam, gamma=gamma,
        interp_bins=rc.waveform.interp_bins, roughness=rough,
        taper_s=gfx.s_eff if gfx else None, d_phi_area=gfx is not None,
        **window)
    field = apply_waveform(field, rc, "coherent")
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


def _joint_pad_to(j, n_max):
    """Power-of-two padding bucket for a joint-path target under ``j``
    interfaces (capped at ``n_max``, the run's deepest stack): log2(N_max)
    executables serve every target layer, with <= 2x padded no-op work."""
    b = 1
    while b < j:
        b *= 2
    return min(b, n_max)


def _simulate_multilayer(scene, sim_config):
    """Multilayer run: surface via the stage-2 kernels (bit-compatible),
    deeper interfaces via the refracted-path kernel; one layer per interface.

    ``sim_config.refraction`` picks the crossing solver for targets under
    >= 2 interfaces; single-crossing targets (e.g. the bed of a two-media
    run) always use the sequential path -- the two-point solve is exact
    there and the joint solver reproduces it to ~2e-11 m (config.py).
    Joint-path calls are padded to power-of-two interface counts so target
    layers share compiled executables (kernels/multilayer.py docstring)."""
    if sim_config.radar.waveform.interp_bins:
        raise ValueError(
            "waveform.interp_bins is not yet supported for multilayer runs "
            "(the refracted-path kernel bins without sub-bin splitting)")
    frame = LocalFrame.centered_on(scene)
    wired, dem_refs = _wire_scene_dems(scene, sim_config)
    layered = build_layered_scene(wired, frame, scene.dem, scene.transform,
                                  scene.crs, spacing=sim_config.facets.spacing,
                                  dem_refs=dem_refs)
    track = nav_to_frame(scene.nav_llh, frame)
    rc = sim_config.radar
    window = dict(t0=rc.t0, dt=rc.dt, n_samples=rc.n_samples, c=rc.c,
                  split_sides=sim_config.split_sides,
                  pattern=_antenna_pattern(rc, scene, track))
    eps = [m.eps_r for m in sim_config.media]
    att = [m.attenuation_db_per_km for m in sim_config.media]
    coherent = sim_config.mode == "coherent"
    surf = layered.interfaces[0]

    if coherent:
        if rc.f0 is None:
            raise ValueError("coherent mode requires radar.f0 to be set")
        _warn_quantization_alias(rc)
        lam = rc.wavelength
        k0 = 2.0 * np.pi / lam
        for j, f in enumerate(layered.interfaces):
            lam_j = lam / np.sqrt(eps[j])  # in-medium wavelength at interface j
            min_range = max(float(track.positions[:, 2].min()
                                  - f.centers[:, 2].max()), lam_j)
            check_facet_size(f, lam_j, min_range)

    ifaces = sim_config.interfaces
    gfx = sim_config.grazing_fix  # None (legacy) or the grazing-fix pair
    gfx_kw = dict(taper_s=gfx.s_eff if gfx else None,
                  d_phi_area=gfx is not None)
    sig_all = np.array([(ic.roughness.sigma_m if ic.roughness else 0.0)
                        for ic in ifaces])

    # Per-facet reflectivity maps (scene.gamma_maps: {interface name ->
    # (grid, transform, crs)}, see _gamma_map_values). Coherent only: the
    # incoherent path books no target reflectivity by convention.
    gmaps = getattr(scene, "gamma_maps", None) or {}
    if gmaps:
        if not coherent:
            raise ValueError("scene.gamma_maps requires coherent mode")
        unknown = set(gmaps) - set(layered.names)
        if unknown:
            raise ValueError(f"gamma_maps for unknown interfaces: {unknown}")
    # Per-facet DIFFUSE amplitudes (scene.diffuse_maps, same grid convention
    # as gamma_maps): the incoherent companion channel of a specular/diffuse
    # reflectivity split (kernels/multilayer.py). Coherent + refracted only.
    dmaps = getattr(scene, "diffuse_maps", None) or {}
    if dmaps:
        if not coherent:
            raise ValueError("scene.diffuse_maps requires coherent mode")
        unknown = set(dmaps) - set(layered.names)
        if unknown:
            raise ValueError(f"diffuse_maps for unknown interfaces: {unknown}")
        if layered.names[0] in dmaps:
            raise ValueError("diffuse_maps on the top interface is not "
                             "wired (the surface uses coherent.py)")

    outs, drops = [], []
    for j, target in enumerate(layered.interfaces):
        gamma_j = fresnel_normal(eps[j], eps[j + 1])
        if layered.names[j] in gmaps:
            gamma_j = _gamma_map_values(gmaps[layered.names[j]], target, frame)
        rough = (_roughness_args(ifaces[j], layered.interfaces[j],
                                 k0 * np.sqrt(eps[j]),
                                 (sim_config.roughness_seed, j))
                 if coherent else None)
        if j == 0:
            if coherent:
                out, drop = coherent_cluttergram(
                    track.positions, track.u_ct, surf.centers, surf.normals,
                    surf.areas, surf.e1, surf.e2, k=k0, gamma=gamma_j,
                    roughness=rough, **gfx_kw, **window)
            else:
                out, drop = incoherent_cluttergram(
                    track.positions, track.u_ct, surf.centers, surf.normals,
                    surf.areas, **window)
        else:
            refr = sim_config.refraction if j > 1 else "sequential"
            n_max = len(layered.interfaces) - 1
            crossed_sig = (sig_all[:j] if coherent and sig_all[:j].any()
                           else None)
            dif = None
            if layered.names[j] in dmaps:
                dif = (_gamma_map_values(dmaps[layered.names[j]], target,
                                         frame),
                       speckle_phasors(len(target.centers),
                                       (sim_config.roughness_seed, 1000 + j)),
                       float(sim_config.diffuse_exponent))
            out, drop = refracted_cluttergram(
                track.positions, track.u_ct, target, layered.interfaces[:j],
                eps[:j + 1], att[:j + 1], mode=sim_config.mode, gamma=gamma_j,
                k0=k0 if coherent else None, refraction=refr,
                pad_to=_joint_pad_to(j, n_max) if refr == "joint" else None,
                roughness=rough, crossed_sigma=crossed_sig, diffuse=dif,
                **(gfx_kw if coherent else {}), **window)
        outs.append(out)
        drops.append(drop)

    stacked = np.stack(outs, axis=-1)
    # Convolution is linear, so per-layer application == application to the
    # layer sum; the layer dim rides along as a trailing axis.
    stacked = apply_waveform(stacked, rc,
                             "coherent" if coherent else "incoherent")
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
