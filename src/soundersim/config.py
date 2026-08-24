"""Pydantic configuration models for a simulation run (JSON round-trippable)."""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class WaveformConfig(BaseModel):
    """Transmit waveform / pulse-compression model (stage 4, M20).

    ``kind="delta"`` (default) keeps the kernels' native delta-at-carrier
    response -- exact pre-stage-4 behavior. ``kind="chirp"`` convolves the
    coherent trace with the pulse-compressed LFM response (see waveform.py:
    peak-normalized windowed-sinc, applied post-kernel along fast time).

    ``interp_bins`` splits each facet's contribution linearly between the two
    adjacent fast-time bins (kernel-level), suppressing the dt quantization of
    the envelope delay; ``incoherent_envelope`` opts the incoherent mode into
    a |p|^2 power-envelope convolution (default OFF -- simc parity, plan D4-4).
    """

    kind: Literal["delta", "chirp"] = "delta"
    bandwidth: Optional[float] = None  # chirp bandwidth B (Hz)
    pulse_length: Optional[float] = None  # uncompressed pulse length T (s)
    window: Literal["none", "hann", "hamming"] = "hann"
    interp_bins: bool = False  # sub-bin linear splitting in the kernel binning
    incoherent_envelope: bool = False  # opt-in power-envelope conv (incoherent)

    @model_validator(mode="after")
    def _chirp_params(self):
        if self.kind == "chirp":
            if not self.bandwidth or self.bandwidth <= 0:
                raise ValueError("chirp waveform requires bandwidth > 0")
            if not self.pulse_length or self.pulse_length <= 0:
                raise ValueError("chirp waveform requires pulse_length > 0")
        return self


class AntennaConfig(BaseModel):
    """Antenna gain pattern (stage 4, M22). All gains are ONE-WAY FIELD gains
    g (see antenna.py): monostatic two-way weighting is g**2 on fields
    (coherent kernels) and g**4 on power (incoherent kernel).

    Kinds:

    - ``isotropic`` (default): g = 1, exact pre-stage-4 behavior.
    - ``dipole``: half-wave dipole, g = cos((pi/2) cos(psi)) / sin(psi) with
      psi the angle from the dipole ``axis`` (null along the axis, 1 at
      broadside). ``axis`` is along-track (default) or cross-track.
    - ``array``: uniform unsteered linear array of ``n_elements`` isotropic
      elements along the CROSS-TRACK axis with boresight at nadir (the
      MCoRDS-like case); element spacing ``spacing_lam`` is in CARRIER
      WAVELENGTHS (dimensionless). g = |array factor|
      = sin(N x)/(N sin x), x = pi * spacing_lam * sin(theta_ct).
    - ``tabulated``: g(theta) linearly interpolated from ``theta_deg`` /
      ``gain`` samples, rotationally symmetric about the nadir boresight;
      theta is the angle from boresight in degrees, ascending (clamped at the
      table ends).

    ``roll_source="nav"`` rotates the pattern frame about the along-track
    axis by the nav frame's per-trace roll (radians, positive = right wing
    down; scenes without roll data use 0).
    """

    kind: Literal["isotropic", "dipole", "array", "tabulated"] = "isotropic"
    axis: Literal["along_track", "cross_track"] = "along_track"  # dipole only
    n_elements: int = 5          # array only
    spacing_lam: float = 0.5     # array element spacing (carrier wavelengths)
    theta_deg: Optional[list[float]] = None  # tabulated: angle from boresight
    gain: Optional[list[float]] = None       # tabulated: one-way FIELD gain
    roll_source: Literal["none", "nav"] = "none"

    @model_validator(mode="after")
    def _pattern_params(self):
        if self.kind == "array":
            if self.n_elements < 2:
                raise ValueError("array antenna requires n_elements >= 2")
            if self.spacing_lam <= 0:
                raise ValueError("array antenna requires spacing_lam > 0")
        if self.kind == "tabulated":
            if not self.theta_deg or not self.gain:
                raise ValueError(
                    "tabulated antenna requires theta_deg and gain samples")
            if len(self.theta_deg) != len(self.gain) or len(self.gain) < 2:
                raise ValueError(
                    "tabulated antenna needs >= 2 matching theta_deg/gain")
            t = self.theta_deg
            if any(b <= a for a, b in zip(t, t[1:])):
                raise ValueError("tabulated theta_deg must be ascending")
            if any(g < 0 for g in self.gain):
                raise ValueError("tabulated gain samples must be >= 0 "
                                 "(one-way FIELD gain)")
        return self


class RadarConfig(BaseModel):
    """Fast-time sampling of the two-way-travel-time (twtt) window."""

    dt: float  # sample spacing (s)
    n_samples: int  # samples per trace
    t0: float  # window start, twtt (s)
    c: float = 299792458.0  # speed of light (m/s)
    f0: Optional[float] = None  # carrier frequency (Hz); needed for coherent mode
    waveform: WaveformConfig = Field(default_factory=WaveformConfig)
    antenna: AntennaConfig = Field(default_factory=AntennaConfig)

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


class RoughnessConfig(BaseModel):
    """Sub-facet Gaussian roughness of an interface (Gerekos et al. 2023;
    docs/roughness.md). ``sigma_m`` is the RMS height along the facet normal,
    ``corr_length_m`` the isotropic Gaussian correlation length; coherent
    mode only. Validity: corr_length_m should not exceed the facet size
    (larger-scale roughness belongs in the DEM as facet tilt)."""

    sigma_m: float          # RMS height (m), >= 0
    corr_length_m: float    # Gaussian correlation length (m), > 0

    @model_validator(mode="after")
    def _positive(self):
        if self.sigma_m < 0:
            raise ValueError("roughness sigma_m must be >= 0")
        if self.corr_length_m <= 0:
            raise ValueError("roughness corr_length_m must be > 0")
        return self


class GrazingFixConfig(BaseModel):
    """Grazing-angle facet-lattice fix (coherent mode only; opt-in -- the
    field's absence traces exactly the legacy kernels, so every existing
    cache and regression gate is untouched).

    ONE switch enables two coupled changes, because they are two faces of
    the same artifact -- facet-grid spatial aliasing once the LPA phase ramp
    across a facet exceeds pi (2kL sin(theta) >> pi at grazing) -- and the
    acceptance criterion (facet-size-invariant effective sigma0) needs both:

    - COHERENT off-specular taper: each facet's smooth (sinc*sinc specular)
      FIELD is multiplied by T(alpha) = exp(-tan^2(alpha)/(2 s_eff^2)),
      alpha the arrival angle off the facet normal. Physical basis: the
      sub-facet slope distribution -- a facet only mirrors power back within
      its slope spread. Near-specular returns (alpha << s_eff, e.g. glinting
      valley walls) keep T ~ 1; the non-converging sinc/grid-lobe tails at
      grazing go to 0. The removed power is grid aliasing, not physical
      power, and is dropped rather than re-booked (the physical off-specular
      return is the D_Phi channel below, plus the optional spec/diffuse
      split).
    - AREA-TERM-ONLY D_Phi: the sub-facet-roughness incoherent variance
      keeps only its facet-area-scaling term (the per-facet infinite-surface
      PO law, Gerekos et al. 2023 Appendix C; roughness.d_phi area_only),
      dropping the facet-edge remainder whose sigma0 goes as 1/L^2 * O(1)
      (facet-size dependent, +30 dB unphysical at grazing).

    ``s_eff``: effective rms slope of the taper, in tan(alpha) units.
    Physical scale ~ sqrt(2) * sigma/l of the sub-facet roughness (0.02-0.16
    for the campaign interfaces); the default 0.05 (~3 deg) is a mildly
    conservative single value for all interfaces.
    """

    s_eff: float = 0.05

    @model_validator(mode="after")
    def _positive(self):
        if self.s_eff <= 0:
            raise ValueError("grazing_fix s_eff must be > 0")
        return self


class _InterfaceBase(BaseModel):
    """Fields shared by every interface kind."""

    name: Optional[str] = None
    roughness: Optional[RoughnessConfig] = None  # None -> smooth (exact)


class DemInterface(_InterfaceBase):
    """An interface given by its own DEM.

    ``path`` points to a GeoTIFF; ``ref`` names an in-memory DEM (looked up by
    the scene builder). Neither set means "the scene's primary surface DEM" (the
    backward-compatible stage-2 default). At most one of path/ref may be set.
    """

    kind: Literal["dem"] = "dem"
    path: Optional[str] = None  # GeoTIFF path
    ref: Optional[str] = None   # in-memory DEM key

    @model_validator(mode="after")
    def _one_source(self):
        if self.path is not None and self.ref is not None:
            raise ValueError("dem interface: set at most one of path/ref")
        return self


class FlatInterface(_InterfaceBase):
    """A flat interface at a constant ellipsoidal elevation (m)."""

    kind: Literal["flat"] = "flat"
    elevation: float  # constant ellipsoidal height (m)


class OffsetInterface(_InterfaceBase):
    """A constant vertical offset of another interface (e.g. surface - 2 m).

    ``reference`` is the index (int) or name (str) of another interface;
    ``offset`` is the vertical shift in metres (negative = below the reference).
    """

    kind: Literal["offset"] = "offset"
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

    ``refraction`` selects the multilayer refracted-path solver for targets
    under two or more interfaces (kernels/multilayer.py): ``"joint"``
    (default) solves all crossings of the stack at once (the true stationary
    path of the anchored local planes; O(1)-in-layer-count compiled graph),
    ``"sequential"`` chains per-interface two-point solves (the stage-3
    approximation, exact only for one crossing). Targets under a SINGLE
    interface always use the sequential path: with one crossing the two-point
    solve is already exact (no chaining approximation exists) and the joint
    solver reproduces it to ~2e-11 m, so the switch only affects deeper
    stacks. See docs/refraction.md.

    ``roughness_seed`` seeds the deterministic per-facet speckle phasors of
    any interface with a ``roughness`` config (docs/roughness.md); runs with
    the same seed are reproducible, different seeds draw independent speckle
    realizations. It also seeds the DIFFUSE channel's phasors (independent
    stream).

    ``grazing_fix`` (default None = off, the legacy program) enables the
    grazing-angle facet-lattice fix -- coherent off-specular taper +
    area-term-only D_Phi -- see ``GrazingFixConfig``.

    ``diffuse_exponent`` is the exponent n of the cos^n(theta_incidence)
    angular law of the diffuse channel, used only when a scene attaches
    ``diffuse_maps`` (per-facet diffuse FIELD amplitudes; see
    kernels/multilayer.py). n = 0 is the flat law; 1-2 spans the usual
    near-Lambert range.
    """

    mode: Literal["incoherent", "coherent"]
    split_sides: bool = False
    refraction: Literal["sequential", "joint"] = "joint"
    grazing_fix: Optional[GrazingFixConfig] = None
    roughness_seed: int = 0
    diffuse_exponent: float = 1.0
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
        if self.mode != "coherent" and any(i.roughness is not None
                                           for i in self.interfaces):
            raise ValueError("interface roughness requires coherent mode "
                             "(the incoherent kernel has no phase to perturb)")
        if self.mode != "coherent" and self.grazing_fix is not None:
            raise ValueError("grazing_fix requires coherent mode (it tapers "
                             "the coherent facet response)")
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
