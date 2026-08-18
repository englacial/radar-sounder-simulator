"""Radar system definitions (config/instruments/*.yaml).

An instrument is the radar box: carrier, bandwidth, pulse, compression window
and antenna. It is deliberately SEPARATE from the line, so the two axes that
matter for mission design vary independently -- swap the box at a fixed
altitude, or fly the same box higher.

Two kinds:

``source: {kind: opr_frame}``
    A REAL system: every simulated parameter is read from the OPR frame the
    pass was actually flown on (``param_frame`` in the line definition), which
    is what keeps a measured-vs-simulated comparison honest. ``simulated:``
    entries left null defer to that read; a stated value overrides it and is
    flagged in the run config as a deviation from the recorded system.

``source: {kind: stated}``
    A SYNTHETIC system: the values are the design, and no OPR frame is
    consulted.

FIELD NAMES follow the mission design tool
(radar_return_statistics_postprocessing/mission_design_tool) so one config
can describe a system to both tools. This simulator is CLUTTER-limited: it
has no receiver-noise model and no link budget, so it consumes only
frequency / bandwidth / pulse length / window / antenna. The link-budget
fields (tx power, gains, losses, noise figure) live under ``recorded:`` --
carried into the run config as provenance, consumed by nothing here yet.
Platform altitude and velocity are NOT instrument fields here: altitude is a
property of the observation, which is what makes the swap axes independent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import (BaseModel, ConfigDict, field_validator,
                      model_validator)

INSTRUMENT_SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
INSTRUMENTS_DIR = ROOT / "config" / "instruments"

# What the simulator actually reads. Everything else is provenance.
SIMULATED_FIELDS = ("frequency_Hz", "bandwidth_Hz", "pulse_length_s", "window")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Source(_Base):
    kind: Literal["opr_frame", "stated"]


class Antenna(_Base):
    kind: Literal["isotropic", "dipole", "array", "tabulated"] = "array"
    n_elements: int = 7
    spacing_lam: float = 0.5
    roll_source: Literal["none", "nav"] = "nav"


class Simulated(_Base):
    """Null = defer to the OPR frame (only legal when source is opr_frame)."""

    frequency_Hz: float | None = None
    bandwidth_Hz: float | None = None
    pulse_length_s: float | None = None
    window: str | None = None
    antenna: Antenna = Antenna()


def segment_of(frame_id):
    """Segment key of an OPR/CReSIS frame id: YYYYMMDD_SS_FFF -> YYYYMMDD_SS.

    Data from one segment is one instrument -- the radar is not reconfigured
    mid-segment -- so a segment is the natural unit an instrument covers."""
    return str(frame_id).rsplit("_", 1)[0]


class InstrumentSpec(_Base):
    schema_version: int = INSTRUMENT_SCHEMA_VERSION
    name: str
    description: str | None = None
    source: Source
    # Segments (YYYYMMDD_SS) this system flew. Data from one segment is one
    # instrument, so this is the unit of coverage: a pass whose param_frame
    # falls outside its instrument's segments is a mis-pinned config, and
    # validate_line_instruments() refuses it before any run starts.
    segments: list[str] = []

    @field_validator("segments", mode="before")
    @classmethod
    def _must_be_quoted(cls, v):
        # YAML 1.1 reads underscores as digit separators, so an unquoted
        # 20161105_05 silently becomes the integer 2016110505. Say so.
        if isinstance(v, list):
            bad = [x for x in v if not isinstance(x, str)]
            if bad:
                raise ValueError(
                    f"segment ids must be QUOTED in YAML -- {bad} parsed as "
                    "numbers because YAML 1.1 treats '_' as a digit "
                    "separator. Write '20161105_05', not 20161105_05.")
        return v
    simulated: Simulated = Simulated()
    # mission-design fields with no consumer in this simulator; recorded so a
    # config is portable and the eventual link budget needs no re-authoring
    recorded: dict = {}
    provenance: str | None = None

    @model_validator(mode="after")
    def _stated_is_complete(self):
        if self.schema_version != INSTRUMENT_SCHEMA_VERSION:
            raise ValueError(f"instrument schema_version "
                             f"{self.schema_version} != "
                             f"{INSTRUMENT_SCHEMA_VERSION}")
        if self.source.kind == "stated" and self.segments:
            raise ValueError(f"instrument {self.name!r} is 'stated' but "
                             "lists segments: a synthetic system flew none")
        if self.source.kind == "opr_frame" and not self.segments:
            raise ValueError(f"instrument {self.name!r} reads from OPR data "
                             "but lists no segments it covers")
        if self.source.kind == "stated":
            missing = [f for f in SIMULATED_FIELDS
                       if getattr(self.simulated, f) is None]
            if missing:
                raise ValueError(
                    f"instrument {self.name!r} is 'stated' but leaves "
                    f"{missing} null: there is no OPR frame to defer to")
        return self

    # ------------------------------------------------------------- resolve
    def resolve(self, frame_params=None):
        """(waveform dict, antenna, deviations) for one pass.

        ``frame_params`` is ``rac.mcords_params(...)['waveform']`` for the
        pass's own param frame, or None for a stated instrument. The returned
        waveform dict uses the tool's existing key names, so nothing
        downstream -- including the chunk cache key -- changes shape."""
        want = {f: getattr(self.simulated, f) for f in SIMULATED_FIELDS}
        if self.source.kind == "opr_frame":
            if frame_params is None:
                raise ValueError(f"instrument {self.name!r} reads from the "
                                 "OPR frame but no frame params were supplied")
            base = {
                "frequency_Hz": frame_params["center_frequency_Hz"],
                "bandwidth_Hz": frame_params["bandwidth_Hz"],
                "pulse_length_s": frame_params["bed_waveform_pulse_length_s"],
                "window": frame_params["pulse_compression_freq_window"],
            }
        else:
            base = dict(want)
        deviations = {}
        out = {}
        for f in SIMULATED_FIELDS:
            if want[f] is None or want[f] == base[f]:
                out[f] = base[f]
            else:
                out[f] = want[f]
                deviations[f] = {"recorded_system": base[f],
                                 "used": want[f]}
        wf = {"center_frequency_Hz": out["frequency_Hz"],
              "bandwidth_Hz": out["bandwidth_Hz"],
              "bed_waveform_pulse_length_s": out["pulse_length_s"],
              "pulse_compression_freq_window": out["window"]}
        return wf, self.simulated.antenna, deviations

    def covers(self, frame_id):
        return segment_of(frame_id) in set(self.segments)

    def provenance_block(self, deviations=None):
        b = {"instrument": self.name, "source": self.source.kind,
             "antenna": self.simulated.antenna.model_dump()}
        if self.description:
            b["description"] = self.description
        if self.provenance:
            b["provenance"] = self.provenance
        if self.segments:
            b["segments"] = list(self.segments)
        if self.recorded:
            b["recorded_not_simulated"] = {
                **self.recorded,
                "note": "mission-design fields with no consumer in this "
                        "clutter-limited simulator (no receiver-noise model, "
                        "no link budget); carried as provenance"}
        if deviations:
            b["deviations_from_recorded_system"] = deviations
        return b


def load_instrument(path):
    with Path(path).open() as fh:
        return InstrumentSpec.model_validate(yaml.safe_load(fh))


def load_all(d=None):
    d = Path(d or INSTRUMENTS_DIR)
    out = {}
    for fp in sorted(d.glob("*.yaml")):
        spec = load_instrument(fp)
        if fp.stem != spec.name:
            raise ValueError(f"{fp.name}: filename must match name "
                             f"{spec.name!r}")
        out[spec.name] = spec
    return out


def known_instrument_names(d=None):
    return {fp.stem for fp in Path(d or INSTRUMENTS_DIR).glob("*.yaml")}


def validate_line_instruments(lines, instruments):
    """Every pass must name a known instrument that covers its own segment.

    Catches a mis-pinned instrument at import, not 40 minutes into a run.
    A synthetic instrument covers no segments, so pinning one as a line
    DEFAULT is refused -- swapping one in is an experiment's job."""
    problems = []
    for line_name, spec in lines.items():
        for key, ps in spec.passes.items():
            inst = instruments.get(ps.instrument)
            if inst is None:
                problems.append(f"{line_name}/{key}: unknown instrument "
                                f"{ps.instrument!r}")
                continue
            if inst.source.kind == "stated":
                problems.append(
                    f"{line_name}/{key}: {inst.name!r} is a SYNTHETIC "
                    "instrument and cannot be a line default -- swap it in "
                    "from an experiment instead")
            elif not inst.covers(ps.param_frame):
                problems.append(
                    f"{line_name}/{key}: param_frame {ps.param_frame} is in "
                    f"segment {segment_of(ps.param_frame)}, which "
                    f"{inst.name!r} does not cover {inst.segments}")
    if problems:
        raise ValueError("line/instrument mismatch:\n  "
                         + "\n  ".join(problems))
    return True
