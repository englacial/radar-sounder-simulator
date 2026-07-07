"""Pydantic configuration models for a simulation run (JSON round-trippable)."""

from typing import Literal, Optional

from pydantic import BaseModel


class RadarConfig(BaseModel):
    """Fast-time sampling of the two-way-travel-time (twtt) window."""

    dt: float  # sample spacing (s)
    n_samples: int  # samples per trace
    t0: float  # window start, twtt (s)
    c: float = 299792458.0  # speed of light (m/s)


class FacetConfig(BaseModel):
    """Facet tessellation controls."""

    spacing: Optional[float] = None  # facet spacing (m); None -> use DEM posting


class SimConfig(BaseModel):
    """Top-level simulation configuration."""

    mode: Literal["incoherent", "coherent"]
    split_sides: bool = False
    radar: RadarConfig
    facets: FacetConfig
