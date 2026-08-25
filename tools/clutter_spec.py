"""Declarative run spec for tools/run_basal_clutter.py (YAML <-> RunSpec).

ONE schema, TWO front doors: a YAML experiment file and the argparse CLI both
build the same ``RunSpec``, and ``RunSpec.to_run_kwargs()`` produces exactly
the keyword dict ``run_basal_clutter.run()`` already accepts. ``run()`` is
untouched, so chunk cache keys (chunk_rid / chunk_meta) cannot move.

The YAML groups the inputs so that the ONE group with a different invalidation
cost is visually separate: everything under ``figures:`` is cache-replay safe
and never re-simulates, while ``bed:`` / ``reflectivity:`` / ``physics:`` /
``processing:`` edits cost a full run.

Numbers that were DERIVED from another run (the level-anchor deficit D) carry
their provenance inline as {value, from, how} rather than sitting in a comment
detached from the run they came from. A bare float is accepted and coerced.

Bed topography is an ENUM, not the historical pair of mutually exclusive
booleans; ``hybrid`` must be stated explicitly (``run()`` still infers it from
the segment, so the spec asserts the two agree rather than silently differing).

See config/README.md for the file layout and the meta block.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

SCHEMA_VERSION = 1

# Naming a synthetic pass in ``passes:`` is the only way to request it: the
# LINE definition (config/lines/*.yaml) declares which synthetics exist.

class BedSource(StrEnum):
    """Bed topography source (replaces the picked_bed/demogorgn_bed pair)."""

    BEDMACHINE = "bedmachine"
    PICKED = "picked"
    DEMOGORGN = "demogorgn"
    HYBRID = "hybrid"


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
    """Experiment identity. Only ``name`` travels into the run outputs: the
    rest describes the EXPERIMENT (which supersedes which, what claim it
    backs) and is maintained in config/experiments/, not in a stale output dir."""

    name: str
    status: str = "exploratory"       # adopted | superseded-by:<name> | ...
    # A benchmark is re-run to answer "did a simulator change help or hurt".
    # Its `expected` block holds the acceptance numbers a fidelity comparison
    # scores against.
    role: Literal["study", "benchmark"] = "study"
    expected: dict = {}
    requires: list[str] = []
    backs: str | None = None
    runtime: str | None = None
    note: str | None = None


class Bed(_Base):
    source: BedSource = BedSource.BEDMACHINE
    demogorgn_seed: int = 0
    ablation: bool = False            # --bed-ablation (adds bed-source rows)


class BedRoughness(_Base):
    sigma_m: float
    corr_length_m: float
    extra_db: float = 0.0             # pilot-measured incoherent residual


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
    surface_roughness: bool = True
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
    out_name: str | None = None
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

    @model_validator(mode="after")
    def _hybrid_is_explicit(self):
        """run() infers the hybrid bed from the segment's declared
        crosses_gl. The spec states it, so the two must agree on every
        named line -- otherwise a file could read 'demogorgn' while the
        run silently built a hybrid (or vice versa)."""
        try:
            from clutter_lines import load_all
            specs = load_all()
        except Exception:
            return self                    # registry unreadable: run() guards
        hyb = self.bed.source is BedSource.HYBRID
        for name in ([self.line] if self.line else self.lines):
            seg = specs.get(name) and specs[name].segments.get(self.segment)
            if seg is None:
                continue                   # unknown segment: run() rejects it
            if hyb != seg.crosses_gl:
                raise ValueError(
                    f"line {name!r} segment {self.segment!r} "
                    f"{'crosses' if seg.crosses_gl else 'does not cross'} "
                    "the grounding line, so bed.source "
                    f"{'must' if seg.crosses_gl else 'must not'} be "
                    f"'hybrid' (got {self.bed.source!r})")
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
        if self.run.out_name and self.run.out_name != self.meta.name:
            raise ValueError(f"meta.name {self.meta.name!r} must equal "
                             f"run.out_name {self.run.out_name!r}")
        return self

    # ---------------------------------------------------------------- kwargs
    def to_run_kwargs(self):
        """The EXACT keyword dict run_basal_clutter.run() accepts today.

        Nothing here is new behaviour: the enum is expanded back into the two
        historical booleans and the synthetic pass keys back into --add-<N>km,
        so a spec and the equivalent CLI invocation are indistinguishable to
        run() -- and therefore to the chunk cache."""
        r, ref, phy, proc, fig = (self.run, self.run.reflectivity,
                                  self.run.physics, self.run.processing,
                                  self.run.figures)
        src = r.bed.source
        kw = {
            "line": r.line,          # None for a multi-line protocol:
                                     # main_config resolves --line into it
            "segment": r.segment,
            "out_name": r.out_name or self.meta.name,
            "out_root": r.out_root,
            "n_traces": r.n_traces,
            "passes": list(r.passes) if r.passes else None,
            "instruments": dict(r.instruments) or None,
            "extra_passes": ({k: v.model_dump(exclude_none=True)
                              for k, v in r.extra_passes.items()}
                             or None),
            "picked_bed": src is BedSource.PICKED,
            # the hybrid bed is a DEMOGORGN grounded side blended into the
            # floating picks, so it enters run() through the same flag
            "demogorgn_bed": src in (BedSource.DEMOGORGN, BedSource.HYBRID),
            "demogorgn_seed": r.bed.demogorgn_seed,
            "bed_ablation": r.bed.ablation,
            "gamma_rssnr": ref.gamma_from_rssnr,
            "spec": (None if ref.specular_diffuse is None else
                     (ref.specular_diffuse.specular_fraction,
                      ref.specular_diffuse.tilt_s0_deg,
                      ref.specular_diffuse.diffuse_exponent)),
            "att": phy.att_db_per_km,
            "surf_rough": phy.surface_roughness,
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
