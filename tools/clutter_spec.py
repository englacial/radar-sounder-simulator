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
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class Derived(_Base):
    """A number computed from another run, carrying its own provenance."""

    value: float
    from_run: str | None = Field(default=None, alias="from")
    how: str | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ExtraPass(_Base):
    """An observation this experiment invents: a carrier pass's geometry
    re-flown at a new altitude, optionally with a different radar.

    This is the second swap axis. The line declares what was really flown;
    an experiment can ask "the same line, but a stratospheric box at 14 km"
    without editing the line definition."""

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
    gamma_from_rssnr: bool = False
    anchor: Literal["median", "level"] = "median"
    level_deficit_db: Derived | float | None = None
    specular_diffuse: SpecularDiffuse | None = None

    @model_validator(mode="after")
    def _level_needs_rssnr(self):
        if self.anchor == "level" and not self.gamma_from_rssnr:
            raise ValueError("anchor 'level' requires gamma_from_rssnr: true")
        if self.anchor == "level" and self.level_deficit_db is None:
            raise ValueError(
                "anchor 'level' requires an explicit level_deficit_db: D is "
                "solved against a particular run at a particular "
                "attenuation, so there is no line-level default")
        if self.specular_diffuse and not self.gamma_from_rssnr:
            raise ValueError("specular_diffuse splits the RSSNR-mapped bed "
                             "reflectivity: needs gamma_from_rssnr: true")
        return self

    @property
    def deficit_db(self):
        d = self.level_deficit_db
        return d.value if isinstance(d, Derived) else d

    @property
    def deficit_note(self):
        """Where D came from, recorded beside it in the run config so the
        number and its derivation cannot drift apart."""
        d = self.level_deficit_db
        if not isinstance(d, Derived):
            return None
        src = f"derived from {d.from_run}" if d.from_run else "supplied"
        return f"{src}: {d.how}" if d.how else src


class Physics(_Base):
    # REQUIRED, no default: a silent default is how a run reproduced a
    # REJECTED attenuation (see claude_notes/foundations_review_2026-08-17.md
    # section A2). Every spec states the number that defines its result.
    att_db_per_km: float
    surface_roughness: bool = True
    antenna: Literal["array", "isotropic", "array8"] = "array"
    bed_roughness: BedRoughness | None = None


class Processing(_Base):
    chain: Literal["none", "standard"] = "none"
    proc_cache: bool = False
    posting_div: int = 1
    # true -> derived sibling directory; a string names it explicitly;
    # false -> skip the constant-gamma companion acceptance analysis.
    companion: bool | str = True

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
    # validated against the LINES registry, not just typed as str: the whole
    # point of a schema is that a typo fails at load rather than after the
    # scene prep
    line: str
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
        have = known_lines()
        if have is not None and self.line not in have:
            raise ValueError(f"unknown line {self.line!r}; registered lines "
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
        """run() infers the hybrid bed from segment == 'full_line'. The spec
        states it, so the two must agree -- otherwise a file could read
        'demogorgn' while the run silently built a hybrid."""
        hyb = self.bed.source is BedSource.HYBRID
        if hyb != (self.segment == "full_line"):
            raise ValueError(
                "bed.source 'hybrid' and segment 'full_line' imply each other "
                f"(got source={self.bed.source!r}, segment={self.segment!r})")
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
            "line": r.line,
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
            "anchor": ref.anchor,
            "level_deficit_db": ref.deficit_db,
            "level_deficit_note": ref.deficit_note,
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
            "processing": proc.chain,
            "proc_cache": proc.proc_cache,
            "posting_div": proc.posting_div,
            "companion": bool(proc.companion),
            "companion_name": (proc.companion
                               if isinstance(proc.companion, str) else None),
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
