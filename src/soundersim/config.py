"""Pydantic configuration models for a simulation run (JSON round-trippable)."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RadarConfig(BaseModel):
    """Fast-time sampling of the two-way-travel-time (twtt) window."""

    dt: float  # sample spacing (s)
    n_samples: int  # samples per trace
    t0: float  # window start, twtt (s)
    c: float = 299792458.0  # speed of light (m/s)
    f0: Optional[float] = None  # carrier frequency (Hz); needed for coherent mode

    @property
    def wavelength(self) -> float:
        """Carrier wavelength c / f0 (m). Raises if f0 is unset."""
        if self.f0 is None:
            raise ValueError("wavelength requires radar.f0 to be set")
        return self.c / self.f0


class Medium(BaseModel):
    """A dielectric medium: relative permittivity for scalar Fresnel physics."""

    name: str
    eps_r: float  # relative permittivity


class FacetConfig(BaseModel):
    """Facet tessellation controls."""

    spacing: Optional[float] = None  # facet spacing (m); None -> use DEM posting


def _default_media():
    """Stage-2 defaults, ordered top-down (stage 3 will extend the list)."""
    return [Medium(name="air", eps_r=1.0), Medium(name="ice", eps_r=3.17)]


class SimConfig(BaseModel):
    """Top-level simulation configuration."""

    mode: Literal["incoherent", "coherent"]
    split_sides: bool = False
    radar: RadarConfig
    facets: FacetConfig
    media: list[Medium] = Field(default_factory=_default_media)
