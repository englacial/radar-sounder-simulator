"""High-level simulation entry point."""

import numpy as np

from .config import SimConfig
from .kernels.coherent import coherent_cluttergram
from .kernels.incoherent import incoherent_cluttergram
from .nav import nav_to_frame
from .output import build_dataset
from .physics import fresnel_normal
from .scene import LocalFrame, build_facets, check_facet_size
from .synthetic import SyntheticScene


def simulate(scene: SyntheticScene, sim_config: SimConfig):
    """Run a simulation on a scene; returns the output xarray Dataset.

    Builds the local frame (centered on the DEM), facets, and nav track
    internally, runs the kernel for ``sim_config.mode``, and assembles the
    Dataset per docs/output.md.
    """
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
