"""Declarative run spec for tools/run_basal_clutter.py (YAML <-> RunSpec).

ONE schema, TWO front doors: a YAML experiment file and the argparse CLI both
build the same ``RunSpec``, and ``RunSpec.to_run_kwargs()`` produces exactly
the keyword dict ``run_basal_clutter.run()`` already accepts. ``run()`` is
untouched, so chunk cache keys (chunk_rid / chunk_meta) cannot move.

The YAML groups the inputs so that the ONE group with a different invalidation
cost is visually separate: everything under ``figures:`` is cache-replay safe
and never re-simulates, while ``bed:`` / ``reflectivity:`` / ``physics:`` /
``processing:`` edits cost a full run.

The bed is split between data and method: the LINE declares which DEM
exists for its grounded ice (``identity.bed_dem``), the experiment declares
what is done on top of it (``bed.nadir``, ``bed.floating``) so one spec
means the same thing on every line. The hybrid grounded/floating bed is
inferred by ``run()`` from the segment's ``crosses_gl``.

See config/README.md for the file layout and the meta block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ConfigDict, model_validator

SCHEMA_VERSION = 1


class _Base(BaseModel):
    # A typo in a config file must fail loudly, not be silently ignored.
    model_config = ConfigDict(extra="forbid")


class ExtraPass(_Base):
    """An observation this experiment invents: a carrier pass's geometry
    re-flown at a new altitude, optionally with a different radar.

    This is the second swap axis. The line declares what was really flown;
    an experiment can ask "the same line, but a stratospheric box at 14 km"
    without editing the line definition. The literal carrier "reference"
    resolves to each line's reference pass at run time, so one multi-line
    spec can invent the same observation on lines whose pass names differ."""

    carrier: str                       # real pass supplying line + picks
    altitude_m: float
    instrument: str | None = None
    facet_spacing_scale: float | None = None
    why: str | None = None


class Meta(_Base):
    """Experiment identity: ``name`` is the file stem and the output
    directory under outputs/<line case_prefix>/."""

    name: str
    description: str | None = None


class Bed(_Base):
    """Bed construction METHOD (the DEM itself is the line's bed_dem).

    nadir: 'picked' pins the grounded bed at nadir to the reference pass's
    radar picks (DEM + along-track residual); 'dem' uses the DEM as is.
    floating: on a crosses_gl segment the floating bed is the reference
    pass's picks (the ice-ocean interface; DEMs report the seafloor)."""

    nadir: Literal["picked", "dem"] = "picked"
    floating: Literal["picked"] = "picked"
    demogorgn_seed: int = 0           # used when the line's bed_dem is demogorgn


class BedRoughness(_Base):
    sigma_m: float
    corr_length_m: float
    extra_db: float = 0.0             # pilot-measured incoherent residual


class SurfaceRoughnessPair(_Base):
    """Explicit (sigma, l) on the surface interface (the fixture is
    sigma 0.049474 m, l 2.982179 m, Gaussian); ``acf`` selects the
    correlation function (docs/roughness.md; exponential needs the grazing
    fix, which is on by default)."""
    sigma_m: float
    corr_length_m: float
    acf: Literal["gaussian", "exponential"] = "gaussian"


class SurfaceRoughnessSource(_Base):
    """Path B1: per line and per pass carrier, the effective Gaussian pair
    tangent to the measured (OIB ATM) surface PSD at the Bragg wavenumber of
    theta_c (config/roughness/atm_b1.yaml; tools/surface_roughness_b1.py).
    ``atm_exponential``: the table's exponential-ACF (sigma, l) entry used
    DIRECTLY with acf: exponential (power-law entries are refused)."""
    source: Literal["atm_b1", "atm_exponential"]
    theta_c_deg: float | None = None      # None -> the table's default (30)


class SpecularDiffuse(_Base):
    specular_fraction: float
    tilt_s0_deg: float = 1.0
    diffuse_exponent: float = 1.0


class Reflectivity(_Base):
    """Bed reflectivity model. With gamma_from_rssnr the mapping is
    anchoring-free: |Gamma_bed|^2 = 2 A H - RSSNR + (gamma_surface - T^2),
    with gamma_surface and A from the LINE's calibration block. The old
    anchor/level-deficit machinery (K, D) is gone -- D was the residual of
    solving K, and gamma_surface IS that constant, named physically."""

    gamma_from_rssnr: bool = False
    specular_diffuse: SpecularDiffuse | None = None

    @model_validator(mode="after")
    def _spec_needs_rssnr(self):
        if self.specular_diffuse and not self.gamma_from_rssnr:
            raise ValueError("specular_diffuse splits the RSSNR-mapped bed "
                             "reflectivity: needs gamma_from_rssnr: true")
        return self


class Physics(_Base):
    # REQUIRED, no default. Either a stated number (exploratory sweeps) or
    # the literal "solve" -- resolve from the LINE's calibration block
    # (manual value with provenance, or the RSSNR-vs-2H regression). No
    # attenuation rate lives in any spec or in code.
    att_db_per_km: float | Literal["solve"]
    # bool (fixture on/off), an explicit {sigma_m, corr_length_m}, or
    # {source: atm_b1[, theta_c_deg]} resolved per pass at run time
    surface_roughness: (bool | SurfaceRoughnessPair
                        | SurfaceRoughnessSource) = True
    antenna: Literal["array", "isotropic", "array8"] = "array"
    bed_roughness: BedRoughness | None = None
    # grazing-angle facet-lattice fix (coherent off-specular taper +
    # area-only D_Phi). A BUG FIX, ON by default: omitted/null = analysis.yaml
    # grazing_fix.s_eff; a number = override s_eff; false = the legacy
    # artifact path (debug/A-B only). s_eff is part of the chunk cache key.
    grazing_fix: float | Literal[False] | None = None


class Processing(_Base):
    chain: Literal["none", "standard"] = "none"
    proc_cache: bool = False
    posting_div: int = 1
    focus_aperture: Literal[
        "alias_limited", "product_resolution", "first_fresnel", "fixed_angle"
    ] = "alias_limited"
    # fixed_angle: Doppler half-angle of the focusing band; the result is
    # multilooked down to the product_resolution azimuth resolution
    focus_half_angle_deg: float = Field(default=5.0, gt=0.0, le=45.0)
    # Whether to run the constant-gamma comparison arm the RSSNR acceptance
    # analysis scores against. It runs INSIDE this experiment, in this
    # experiment's own cache directory, so there is no companion run to name
    # and no other experiment to depend on.
    companion: bool = True

    @model_validator(mode="after")
    def _posting_div_needs_chain(self):
        if self.posting_div > 1 and self.chain != "standard":
            raise ValueError("posting_div refines the product-posting sim "
                             "grid: use it with processing.chain 'standard'")
        if self.focus_aperture != "alias_limited" and self.chain != "standard":
            raise ValueError("focus_aperture requires processing.chain "
                             "'standard'")
        return self


class Figures(_Base):
    """Cache-replay safe: nothing here re-simulates or invalidates a cache."""

    trace_decomp_s_km: list[float] | None = None
    per_pass: bool = False
    plot_s_max_km: float | None = None
    width_scale: float = 1.0
    bed_overlay: bool = True
    report: bool = True


def known_lines():
    """Line names defined under config/lines/.

    Read from the definition directory rather than from run_basal_clutter, so
    validating a spec costs a directory listing instead of importing
    matplotlib and jax."""
    try:
        from clutter_lines import known_line_names
        return known_line_names()
    except Exception:
        return None


class Run(_Base):
    # ONE line, or a LIST of lines the experiment is valid on -- a benchmark
    # protocol that runs identically on several lines. With `lines`, the
    # line is chosen at run time (--line); outputs cannot collide because
    # each line's case_prefix gives the same experiment name its own
    # directory and cache. Validated against config/lines/, so a typo fails
    # at load rather than after the scene prep.
    line: str | None = None
    lines: list[str] | None = None
    segment: str
    out_root: str | None = None
    n_traces: int | None = None
    passes: list[str] | None = None
    # Swap axis 1: keep the geometry, change the radar. Maps a pass name to
    # an instrument in config/instruments/; anything unnamed keeps the
    # instrument the line pins for that pass.
    instruments: dict[str, str] = {}
    # Swap axis 2: observations this experiment invents (new altitude, and
    # optionally a new radar too).
    extra_passes: dict[str, ExtraPass] = {}
    bed: Bed = Bed()
    reflectivity: Reflectivity = Reflectivity()
    physics: Physics
    processing: Processing = Processing()
    figures: Figures = Figures()

    @model_validator(mode="after")
    def _line_is_registered(self):
        if (self.line is None) == (self.lines is None):
            raise ValueError("state exactly one of run.line (single) or "
                             "run.lines (a multi-line protocol)")
        have = known_lines()
        for name in ([self.line] if self.line else self.lines):
            if have is not None and name not in have:
                raise ValueError(f"unknown line {name!r}; registered lines "
                                 f"are {sorted(have)}")
        return self

    @model_validator(mode="after")
    def _swaps_are_coherent(self):
        named = set(self.passes or [])
        for key in self.extra_passes:
            if self.passes is not None and key not in named:
                raise ValueError(
                    f"extra_passes defines {key!r} but passes: does not list "
                    "it, so it would never be simulated")
        for key in self.instruments:
            known = named | set(self.extra_passes)
            if self.passes is not None and key not in known:
                raise ValueError(f"instruments names pass {key!r}, which this "
                                 "run does not simulate")
        return self


class RunSpec(_Base):
    schema_version: int = SCHEMA_VERSION
    meta: Meta
    run: Run

    @model_validator(mode="after")
    def _known_schema(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version {self.schema_version} != "
                             f"{SCHEMA_VERSION} (this loader)")
        return self

    # ---------------------------------------------------------------- kwargs
    def to_run_kwargs(self):
        """The keyword dict run_basal_clutter.run() accepts, so a spec and
        the equivalent CLI invocation are indistinguishable to run() -- and
        therefore to the chunk cache."""
        r, ref, phy, proc, fig = (self.run, self.run.reflectivity,
                                  self.run.physics, self.run.processing,
                                  self.run.figures)
        kw = {
            "line": r.line,          # None for a multi-line protocol:
                                     # main_config resolves --line into it
            "segment": r.segment,
            "out_name": self.meta.name,
            "out_root": r.out_root,
            "n_traces": r.n_traces,
            "passes": list(r.passes) if r.passes else None,
            "instruments": dict(r.instruments) or None,
            "extra_passes": ({k: v.model_dump(exclude_none=True)
                              for k, v in r.extra_passes.items()}
                             or None),
            "picked_bed": r.bed.nadir == "picked",
            # None -> the line's bed_dem decides (demogorgn or bedmachine)
            "demogorgn_bed": None,
            "demogorgn_seed": r.bed.demogorgn_seed,
            "gamma_rssnr": ref.gamma_from_rssnr,
            "spec": (None if ref.specular_diffuse is None else
                     (ref.specular_diffuse.specular_fraction,
                      ref.specular_diffuse.tilt_s0_deg,
                      ref.specular_diffuse.diffuse_exponent)),
            "att": phy.att_db_per_km,
            "surf_rough": (
                phy.surface_roughness
                if isinstance(phy.surface_roughness, bool)
                else [phy.surface_roughness.sigma_m,
                      phy.surface_roughness.corr_length_m]
                + ([] if phy.surface_roughness.acf == "gaussian"
                   else [phy.surface_roughness.acf])
                if isinstance(phy.surface_roughness, SurfaceRoughnessPair)
                else phy.surface_roughness.model_dump(exclude_none=True)),
            "antenna": phy.antenna,
            "bed_rough": (None if phy.bed_roughness is None else
                          (phy.bed_roughness.sigma_m,
                           phy.bed_roughness.corr_length_m)),
            "bed_rough_extra_db": (0.0 if phy.bed_roughness is None
                                   else phy.bed_roughness.extra_db),
            "grazing_fix": phy.grazing_fix,
            "processing": proc.chain,
            "proc_cache": proc.proc_cache,
            "posting_div": proc.posting_div,
            "focus_aperture": proc.focus_aperture,
            "focus_half_angle_deg": proc.focus_half_angle_deg,
            "companion": bool(proc.companion),
            "trace_decomp_s_km": fig.trace_decomp_s_km,
            "per_pass_figs": fig.per_pass,
            "plot_s_max_km": fig.plot_s_max_km,
            "make_report": fig.report,
        }
        return kw


def load_spec(path):
    """Parse one experiment YAML into a validated RunSpec."""
    path = Path(path)
    with path.open() as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return RunSpec.model_validate(doc)
