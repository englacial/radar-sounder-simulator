"""Layered scene container: one Facets set per interface (CPU, NumPy, float64).

An ordered stack of interfaces (surface, layer_1, ..., bed) each resolved to
rectangular facets in a shared ``LocalFrame`` via ``build_facets``. Interfaces
come from their own DEM, a flat constant-elevation synthesis over a reference
footprint, or a constant vertical offset of another interface. The refraction
solve and kernels (M15/M16) consume ``LayeredScene`` -- this module only builds
geometry, it does not simulate.
"""

from dataclasses import dataclass

import numpy as np
import rasterio

from .config import DemInterface, FlatInterface, OffsetInterface, SimConfig
from .scene import Facets, build_facets


@dataclass
class LayeredScene:
    """Per-interface facets plus the media stack, top-down.

    ``interfaces[k]`` is the Facets for interface k; ``media`` has one more entry
    than ``interfaces`` (medium k is above interface k, the last is the
    substrate). ``names`` labels interfaces for output/reference resolution.
    """

    frame: object                 # LocalFrame
    interfaces: list              # list[Facets], top-down
    media: list                   # list[Medium], len == len(interfaces) + 1
    names: list                   # list[str] interface labels

    def __len__(self):
        return len(self.interfaces)


def offset_facets(facets: Facets, frame, offset: float) -> Facets:
    """Shift every facet center by ``offset`` metres along local ellipsoidal up.

    Fast path for a constant-elevation offset interface: to first order a
    constant vertical offset translates each facet along local up, leaving
    normals, areas and edge vectors unchanged (verified in the tests against a
    full DEM->ECEF rebuild). Returns a new Facets sharing the source's
    normals/areas/e1/e2/cell metadata.
    """
    up = frame.up_at(facets.centers)
    up = up / np.linalg.norm(up, axis=1, keepdims=True)
    centers = facets.centers + offset * up
    return Facets(centers, facets.normals.copy(), facets.areas.copy(),
                  facets.e1.copy(), facets.e2.copy(), facets.cell.copy(),
                  facets.grid_shape,
                  None if facets.phase_keys is None else facets.phase_keys.copy())


def _load_dem(iface: DemInterface, ref_dem, ref_transform, ref_crs, dem_refs,
              grid_origin):
    """Resolve a DEM interface to (dem, transform, crs)."""
    if iface.path is not None:
        with rasterio.open(iface.path) as src:
            return (src.read(1).astype(np.float64), src.transform,
                    src.crs.to_string(), (0, 0))
    if iface.ref is not None:
        ref = dem_refs[iface.ref]
        dem, transform, crs = ref[:3]
        origin = ref[3] if len(ref) > 3 else grid_origin
        return np.asarray(dem, np.float64), transform, crs, origin
    # Neither: the scene's primary surface DEM (backward-compatible default).
    return np.asarray(ref_dem, np.float64), ref_transform, ref_crs, grid_origin


def build_layered_scene(sim_config: SimConfig, frame, dem, transform, crs,
                        *, spacing=None, dem_refs=None, grid_origin=(0, 0)):
    """Build a LayeredScene from a config and a reference (surface) DEM.

    ``dem``/``transform``/``crs`` describe the reference footprint used for the
    primary surface, flat-interface synthesis, and the grid every offset copy
    inherits. ``dem_refs`` maps in-memory DEM keys to (dem, transform, crs).
    """
    dem_refs = dem_refs or {}
    ref_dem = np.asarray(dem, np.float64)
    n = len(sim_config.interfaces)
    facets: list = [None] * n
    names = [ic.name or f"interface_{i}" for i, ic in
             enumerate(sim_config.interfaces)]

    def resolve_index(ref):
        return ref if isinstance(ref, int) else names.index(ref)

    # First pass: DEM and flat interfaces (offsets need their reference built).
    for i, ic in enumerate(sim_config.interfaces):
        if isinstance(ic, DemInterface):
            d, t, c, origin = _load_dem(
                ic, ref_dem, transform, crs, dem_refs, grid_origin)
            facets[i] = build_facets(d, t, c, frame, spacing=spacing,
                                     grid_origin=origin)
        elif isinstance(ic, FlatInterface):
            flat = np.full_like(ref_dem, ic.elevation)
            facets[i] = build_facets(flat, transform, crs, frame,
                                     spacing=spacing,
                                     grid_origin=grid_origin)

    # Remaining passes: resolve offsets (may chain through other offsets).
    pending = [i for i, ic in enumerate(sim_config.interfaces)
               if isinstance(ic, OffsetInterface)]
    while pending:
        progressed = False
        for i in list(pending):
            ic = sim_config.interfaces[i]
            j = resolve_index(ic.reference)
            if facets[j] is not None:
                facets[i] = offset_facets(facets[j], frame, ic.offset)
                pending.remove(i)
                progressed = True
        if not progressed:
            raise ValueError("cyclic offset interface references")

    return LayeredScene(frame, facets, list(sim_config.media), names)
