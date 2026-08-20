"""Study-line definitions as data (config/lines/*.yaml).

A "line" is a flight line the clutter tool can be pointed at: its CRS, its
per-pass frame slices, its segments, its figure framing, its RSSNR store pin
and the provenance prose recorded into every run built on it. It used to be
~250 lines of Python dict literal inside tools/run_basal_clutter.py; it is
now a validated YAML document per line, and ``LineSpec.to_globals()`` returns
exactly the mapping ``activate_line`` rebinds. Nothing else about the
mechanism changed, so the ~4,000 lines of analysis code are untouched.

Three things are DERIVED rather than stated, because stating them twice is
how they drift:
  * ``lam_ice_m``   from ``identity.fc_hz``
  * ``SEGMENTS`` / ``SYNTHETIC_KEYS``  from the mapping keys
  * ``OUT_DEFAULT`` / ``RSSNR_CACHE``  from ``identity.case_prefix``
  * ``REF_SEASON``  from the reference pass's own season

And one thing is deliberately ABSENT: the level-anchor deficit D. It is not
a property of a line -- it is solved against a particular run at a particular
attenuation -- so it lives in the experiment spec that uses it, with its
provenance attached. A line-level default is exactly how the Antarctic 14.8
went stale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

LINE_SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
LINES_DIR = ROOT / "config" / "lines"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class FramePart(_Base):
    """One frame and the half-open slow_time slice taken from it."""

    frame: str
    slice: tuple[int, int]

    @model_validator(mode="after")
    def _ordered(self):
        if self.slice[1] <= self.slice[0]:
            raise ValueError(f"{self.frame}: empty slice {self.slice}")
        return self


class PassSpec(_Base):
    agl_med_m: float | None = None
    season: str | None = None          # falls back to identity.season
    param_frame: str                   # OPR frame whose params were read
    # The radar that actually flew this pass. A DEFAULT: an experiment may
    # swap it (that is the point of the line/instrument split) while keeping
    # the geometry fixed.
    instrument: str
    reversed: bool = False
    segments: dict[str, list[FramePart]]


class SyntheticPass(_Base):
    """A constant-altitude pass flown over a real pass's line geometry."""

    altitude_m: float
    carrier: str                       # real pass whose geometry it rides
    facet_spacing_scale: float = 1.0
    why: str | None = None


class SegmentSpec(_Base):
    s0_km: float                       # display origin (anchor along-track)
    n_traces: int                      # simulated traces
    decomp_s_km: list[float]           # single-trace decomposition location(s)
    # This window spans the line's grounding line. Declared, not inferred
    # from a segment NAME (the old trigger was `segment == "full_line"`, a
    # getz-ism): a crossing segment gets the hybrid bed machinery -- grounded
    # DEMOGORGN blended into the floating radar picks -- and the zone-split
    # metrics.
    crosses_gl: bool = False


class ManualValue(_Base):
    """A manually set calibration parameter. The why is mandatory: a pinned
    number without its provenance is how stale values survive."""

    value: float
    why: str


class Calibration(_Base):
    """The line's physical calibration: gamma_surface and A.

    The reflectivity mapping is |Gamma_bed|^2 = 2 A H - RSSNR + (gamma_surface
    - T^2), with T^2 the two-way Fresnel transmission. gamma_surface is the
    EFFECTIVE surface power reflectivity of the RSSNR reference. Either
    manual {value, why}, or 'solve' (THE DEFAULT choice on the study lines):
    set to zero the qualifying-median bed-level residual against the
    measured data -- exact in one evaluation plus a verification, since the
    received level shifts dB-for-dB with the constant. This is legitimate
    residual-fitting of a NAMED physical parameter, recorded per run; what
    was retired (2026-08-20) was the opaque K/D bookkeeping, not the fit.
    It cannot come from the RSSNR regression intercept (degenerate with the
    mean bed reflectivity), which is why the solve needs a simulation. A is
    manual or 'solve' (Theil-Sen regression of RSSNR on 2H; dataset-only;
    grounded samples when the line has a grounding line and gl_aware, the
    default)."""

    gamma_surface_db: ManualValue | Literal["solve"]
    att_db_per_km: ManualValue | Literal["solve"]
    gl_aware: bool = True


class Identity(_Base):
    case_prefix: str                   # -> outputs/<case_prefix>, cache paths
    season: str                        # line default; passes may override
    crs: str
    fc_hz: float
    grounding_line_s_km: float | None = None


class Reference(_Base):
    """The pass whose picks define the anchor axis and drive the RSSNR fetch."""

    pass_key: str = Field(alias="pass")
    frames: list[str]


class RadargramFraming(_Base):
    y_us: tuple[float, float]
    db: tuple[float, float]
    scale: Literal["shared", "per_panel"] = "shared"


class ProfileFraming(_Base):
    rel_us: tuple[float, float]        # DATA extent, not just an axis limit
    x_us: tuple[float, float]
    db: tuple[float, float]

    @model_validator(mode="after")
    def _plot_within_data(self):
        if self.x_us[1] > self.rel_us[1]:
            raise ValueError("profile.x_us plots beyond profile.rel_us, which "
                             "is the DATA window -- the extra range is never "
                             "computed, only cropped")
        return self


class Framing(_Base):
    radargram: RadargramFraming
    profile: ProfileFraming


class RssnrStore(_Base):
    bucket: str
    prefix: str
    region: str


class Rssnr(_Base):
    snapshot: str
    store: RssnrStore


class Provenance(_Base):
    measured_caveats: str
    real_chain: dict


class LineSpec(_Base):
    schema_version: int = LINE_SCHEMA_VERSION
    name: str
    identity: Identity
    reference: Reference
    order: list[str]                   # real passes, altitude order
    passes: dict[str, PassSpec]
    synthetic_passes: dict[str, SyntheticPass] = {}
    segments: dict[str, SegmentSpec]
    figures: Framing
    calibration: Calibration
    rssnr: Rssnr | None = None
    unsupported: list[str] = []
    provenance: Provenance
    # line-level overrides of config/analysis.yaml (Phase B). Never
    # settable per experiment: these define what the metrics MEAN.
    analysis: dict = {}

    # ------------------------------------------------------------ validate
    @model_validator(mode="after")
    def _coherent(self):
        if self.schema_version != LINE_SCHEMA_VERSION:
            raise ValueError(f"line schema_version {self.schema_version} != "
                             f"{LINE_SCHEMA_VERSION}")
        unknown = [k for k in self.order if k not in self.passes]
        if unknown:
            raise ValueError(f"order names undefined passes {unknown}")
        if self.reference.pass_key not in self.passes:
            raise ValueError(f"reference.pass {self.reference.pass_key!r} is "
                             "not a defined pass")
        for key, syn in self.synthetic_passes.items():
            if syn.carrier not in self.passes:
                raise ValueError(f"{key}: carrier {syn.carrier!r} is not a "
                                 "defined pass")
        segs = set(self.segments)
        # A segment is a WINDOW on the line. Not every flight reaches every
        # window -- on a multi-year repeat line the flights start and stop in
        # different places -- so a pass may omit segments, but may never
        # invent one.
        for key, ps in self.passes.items():
            stray = set(ps.segments) - segs
            if stray:
                raise ValueError(f"pass {key!r} defines segment(s) "
                                 f"{sorted(stray)} absent from segments:")
        ref = self.passes[self.reference.pass_key]
        for name in segs:
            covering = [k for k, ps in self.passes.items()
                        if name in ps.segments]
            if len(covering) < 2:
                raise ValueError(f"segment {name!r} is covered by "
                                 f"{covering or 'no pass'}: a window needs at "
                                 "least two passes to compare")
            if name not in ref.segments:
                raise ValueError(
                    f"segment {name!r} is not covered by the reference pass "
                    f"{self.reference.pass_key!r}, so there is no axis to "
                    "project the others onto")
        if self.rssnr is None and "gamma_rssnr" not in self.unsupported:
            raise ValueError(
                f"line {self.name!r} configures no rssnr store, so it must "
                "list 'gamma_rssnr' in unsupported: -- otherwise a run could "
                "ask for a reflectivity mapping there is no data for")
        for name, seg in self.segments.items():
            if seg.crosses_gl and self.identity.grounding_line_s_km is None:
                raise ValueError(
                    f"segment {name!r} declares crosses_gl but the line has "
                    "no grounding_line_s_km")
        return self

    # ------------------------------------------------------------- globals
    def _pass_table(self):
        """The flat PASSES mapping the tool consumes: segment names are
        top-level keys of each pass entry (``spec[segment]``), and a synthetic
        pass copies its carrier's slices and system params."""
        out = {}
        for key in list(self.order) + [k for k in self.passes
                                       if k not in self.order]:
            ps = self.passes[key]
            entry = {"agl_med_m": ps.agl_med_m, "rev": ps.reversed,
                     "param_frame": ps.param_frame,
                     "instrument": ps.instrument}
            if ps.season is not None:
                entry["season"] = ps.season
            for seg, parts in ps.segments.items():
                entry[seg] = [(p.frame, tuple(p.slice)) for p in parts]
            entry["_segments"] = tuple(ps.segments)
            out[key] = entry
        for key, syn in self.synthetic_passes.items():
            carrier = out[syn.carrier]
            entry = {"agl_med_m": None, "rev": carrier["rev"],
                     "param_frame": carrier["param_frame"],
                     "instrument": carrier["instrument"]}
            if "season" in carrier:
                entry["season"] = carrier["season"]
            for seg in carrier.get("_segments", ()):
                entry[seg] = carrier[seg]
            entry["_segments"] = carrier.get("_segments", ())
            entry["synthetic_msl_m"] = syn.altitude_m
            if syn.facet_spacing_scale != 1.0:
                entry["facet_spacing_scale"] = syn.facet_spacing_scale
            out[key] = entry
        return out

    def to_globals(self, root=None):
        """Exactly the mapping ``run_basal_clutter.activate_line`` rebinds."""
        import numpy as np
        root = Path(root or ROOT)
        c = 299792458.0
        idn = self.identity
        out_default = root / "outputs" / idn.case_prefix
        ref_season = (self.passes[self.reference.pass_key].season
                      or idn.season)
        g = {
            "LINE": self.name,
            "SEASON": idn.season,
            "CRS": idn.crs,
            "CASE_PREFIX": idn.case_prefix,
            "OUT_DEFAULT": out_default,
            "FC_HZ": idn.fc_hz,
            "LAM_ICE_M": c / (idn.fc_hz * float(np.sqrt(3.17))),
            "PASSES": self._pass_table(),
            "ORDER": list(self.order),
            "SEGMENTS": tuple(self.segments),
            "S0_KM": {k: v.s0_km for k, v in self.segments.items()},
            "DECOMP_S_KM": {k: (tuple(v.decomp_s_km)
                                if len(v.decomp_s_km) > 1
                                else v.decomp_s_km[0])
                            for k, v in self.segments.items()},
            "N_TRACES_BY_SEGMENT": {k: v.n_traces
                                    for k, v in self.segments.items()},
            "CALIBRATION": self.calibration.model_dump(),
            "SEGMENTS_CROSSING_GL": tuple(
                k for k, v in self.segments.items() if v.crosses_gl),
            "REF_PASS": self.reference.pass_key,
            "REF_SEASON": ref_season,
            "REF_FRAMES": tuple(self.reference.frames),
            "GL_S_KM": idn.grounding_line_s_km,
            "SYNTHETIC_KEYS": tuple(self.synthetic_passes),
            "MEASURED_CAVEATS": self.provenance.measured_caveats,
            "UNSUPPORTED": tuple(self.unsupported),
            "REAL_CHAIN": dict(self.provenance.real_chain),
            "RADARGRAM_Y_US": tuple(self.figures.radargram.y_us),
            "RADARGRAM_DB": tuple(self.figures.radargram.db),
            "RADARGRAM_SCALE": self.figures.radargram.scale,
            "PROFILE_REL_US": tuple(self.figures.profile.rel_us),
            "PROFILE_X_US": tuple(self.figures.profile.x_us),
            "PROFILE_DB": tuple(self.figures.profile.db),
        }
        # Every line binds every name -- activation is total -- so a line
        # with no required-surface-SNR store still supplies the keys, empty.
        # Such a line must declare gamma_rssnr unsupported (validated below),
        # so the emptiness can never be reached by a run.
        g["RSSNR_SNAPSHOT"] = self.rssnr.snapshot if self.rssnr else ""
        g["RSSNR_STORE"] = (self.rssnr.store.model_dump() if self.rssnr
                            else {})
        g["RSSNR_CACHE"] = out_default / "rssnr_anchor.npz"
        return g


def load_line(path):
    with Path(path).open() as fh:
        return LineSpec.model_validate(yaml.safe_load(fh))


def load_all(lines_dir=None):
    """{name: LineSpec} over config/lines/*.yaml."""
    d = Path(lines_dir or LINES_DIR)
    out = {}
    for fp in sorted(d.glob("*.yaml")):
        spec = load_line(fp)
        if spec.name in out:
            raise ValueError(f"duplicate line name {spec.name!r} in {fp}")
        if fp.stem != spec.name:
            raise ValueError(f"{fp.name}: filename must match name "
                             f"{spec.name!r}")
        out[spec.name] = spec
    if not out:
        raise RuntimeError(f"no line definitions found in {d}")
    return out


def known_line_names(lines_dir=None):
    d = Path(lines_dir or LINES_DIR)
    return {fp.stem for fp in d.glob("*.yaml")}
