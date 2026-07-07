"""High-level simulation entry point."""

from .config import SimConfig
from .kernels.incoherent import incoherent_cluttergram
from .nav import nav_to_frame
from .output import build_dataset
from .scene import LocalFrame, build_facets
from .synthetic import SyntheticScene


def simulate(scene: SyntheticScene, sim_config: SimConfig):
    """Run a simulation on a scene; returns the output xarray Dataset.

    Builds the local frame (centered on the DEM), facets, and nav track
    internally, runs the kernel for ``sim_config.mode``, and assembles the
    Dataset per docs/output.md.
    """
    if sim_config.mode != "incoherent":
        raise NotImplementedError(f"mode {sim_config.mode!r} not implemented yet")
    frame = LocalFrame.centered_on(scene)
    facets = build_facets(scene.dem, scene.transform, scene.crs, frame,
                          spacing=sim_config.facets.spacing)
    track = nav_to_frame(scene.nav_llh, frame)
    rc = sim_config.radar
    power, dropped = incoherent_cluttergram(
        track.positions, track.u_ct, facets.centers, facets.normals,
        facets.areas, t0=rc.t0, dt=rc.dt, n_samples=rc.n_samples, c=rc.c,
        split_sides=sim_config.split_sides)
    return build_dataset(power, dropped, scene=scene, frame=frame,
                         facets=facets, track=track, sim_config=sim_config)
