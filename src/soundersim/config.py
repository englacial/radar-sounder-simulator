"""Pydantic configuration models for a simulation run (JSON round-trippable)."""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


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
    attenuation_db_per_km: float = 0.0  # one-way power attenuation (dB/km)


class FacetConfig(BaseModel):
    """Facet tessellation controls."""

    spacing: Optional[float] = None  # facet spacing (m); None -> use DEM posting


class DemInterface(BaseModel):
    """An interface given by its own DEM.

    ``path`` points to a GeoTIFF; ``ref`` names an in-memory DEM (looked up by
    the scene builder). Neither set means "the scene's primary surface DEM" (the
    backward-compatible stage-2 default). At most one of path/ref may be set.
    """

    kind: Literal["dem"] = "dem"
    name: Optional[str] = None
    path: Optional[str] = None  # GeoTIFF path
    ref: Optional[str] = None   # in-memory DEM key

    @model_validator(mode="after")
    def _one_source(self):
        if self.path is not None and self.ref is not None:
            raise ValueError("dem interface: set at most one of path/ref")
        return self


class FlatInterface(BaseModel):
    """A flat interface at a constant ellipsoidal elevation (m)."""

    kind: Literal["flat"] = "flat"
    name: Optional[str] = None
    elevation: float  # constant ellipsoidal height (m)


class OffsetInterface(BaseModel):
    """A constant vertical offset of another interface (e.g. surface - 2 m).

    ``reference`` is the index (int) or name (str) of another interface;
    ``offset`` is the vertical shift in metres (negative = below the reference).
    """

    kind: Literal["offset"] = "offset"
    name: Optional[str] = None
    reference: Union[int, str]
    offset: float  # vertical offset (m)


InterfaceConfig = Annotated[
    Union[DemInterface, FlatInterface, OffsetInterface],
    Field(discriminator="kind"),
]


def _default_media():
    """Stage-2 defaults, ordered top-down (stage 3 will extend the list)."""
    return [Medium(name="air", eps_r=1.0), Medium(name="ice", eps_r=3.17)]


def _default_interfaces():
    """Single surface interface backed by the scene's primary DEM (stage-2)."""
    return [DemInterface(name="surface")]


class SimConfig(BaseModel):
    """Top-level simulation configuration.

    ``media`` are ordered top-down (air, ice, ..., substrate) and ``interfaces``
    are the boundaries between them (surface, layer_1, ..., bed), so there is
    always exactly one more medium than interface.
    """

    mode: Literal["incoherent", "coherent"]
    split_sides: bool = False
    radar: RadarConfig
    facets: FacetConfig
    media: list[Medium] = Field(default_factory=_default_media)
    interfaces: list[InterfaceConfig] = Field(default_factory=_default_interfaces)

    @model_validator(mode="after")
    def _check_stack(self):
        if len(self.media) != len(self.interfaces) + 1:
            raise ValueError(
                f"need len(media) == len(interfaces) + 1; got "
                f"{len(self.media)} media, {len(self.interfaces)} interfaces")
        names = [i.name for i in self.interfaces]
        for i, iface in enumerate(self.interfaces):
            if isinstance(iface, OffsetInterface):
                ref = iface.reference
                if isinstance(ref, int):
                    if not 0 <= ref < len(self.interfaces) or ref == i:
                        raise ValueError(
                            f"offset interface {i}: bad reference index {ref}")
                elif ref not in names or names.index(ref) == i:
                    raise ValueError(
                        f"offset interface {i}: unknown reference name {ref!r}")
        return self
