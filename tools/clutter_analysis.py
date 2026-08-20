"""Study-wide measurement conventions (config/analysis.yaml).

These define what the metrics MEAN -- where the mid-column window starts,
which delays the bed tail is fitted over, what counts as a trustworthy noise
floor. They are deliberately NOT settable per experiment: a per-run window is
an invitation to move the bed window until the residual looks good, which is
metric shopping rather than measurement.

A LINE may override a subset, because some of these are genuinely properties
of the data rather than of the study -- the Greenland high pass records only
~7.9 us of post-bed tail, so its floor window cannot be the one that suits a
21 us tail. An override is merged over the defaults and RECORDED in the run
config, so a line that measures differently says so out loud.

``to_globals()`` returns the module-level names run_basal_clutter binds, so
the analysis code reads plain constants exactly as it always has.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

ANALYSIS_SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = ROOT / "config" / "analysis.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Coverage(_Base):
    """What the simulation must cover. Drives the cross-track reach and the
    fast-time window, so these are science-critical, not cosmetic."""

    clutter_margin_us: float
    post_bed_window_us: float
    pre_surface_us: float


class MidColumn(_Base):
    after_surface_us: float
    before_bed_us: float


class BedWindow(_Base):
    before_us: float
    after_us: float


class ScoutContrast(_Base):
    lo_us: float
    hi_us: float
    peak_halfwidth_us: float


class Windows(_Base):
    surface_peak_halfwidth_us: float
    midcolumn: MidColumn
    bed: BedWindow
    scout_contrast: ScoutContrast


class NoiseFloor(_Base):
    record_end_window_us: tuple[float, float]
    bed_guard_us: float
    rolloff_us: float
    min_width_us: float


class BedTail(_Base):
    profile_us: tuple[float, float]
    fit_us: tuple[float, float]
    excess_delays_us: tuple[float, ...]
    guard_db: float
    floor_margin_db: float


class GammaSurfaceSolve(_Base):
    seed_db: float
    tolerance_db: float
    min_bed_over_surface_db: float


class AttenuationRegression(_Base):
    min_samples: int
    min_thickness_span_m: float


class Smoothing(_Base):
    profile_m: float
    roughness_detrend_m: float


class HybridBed(_Base):
    gl_ramp_km: float
    seawater_eps: float


class Processing(_Base):
    n_looks: int


class FigureScaling(_Base):
    radargram_percentiles: tuple[float, float]


class Compute(_Base):
    """Tuning, not science -- but chunk_m sets the chunk count, which is part
    of the cache key, so changing it re-simulates."""

    chunk_m: float
    chunk_m_fine_posting: float


class AnalysisSpec(_Base):
    schema_version: int = ANALYSIS_SCHEMA_VERSION
    coverage: Coverage
    windows: Windows
    noise_floor: NoiseFloor
    bed_tail: BedTail
    gamma_surface_solve: GammaSurfaceSolve
    attenuation_regression: AttenuationRegression
    smoothing: Smoothing
    hybrid_bed: HybridBed
    processing: Processing
    figures: FigureScaling
    compute: Compute

    def to_globals(self):
        c, w, nf = self.coverage, self.windows, self.noise_floor
        bt, sm, hb = self.bed_tail, self.smoothing, self.hybrid_bed
        return {
            "MARGIN_US": c.clutter_margin_us,
            "POST_BED_US": c.post_bed_window_us,
            "PRE_SURF_US": c.pre_surface_us,
            "SURF_WIN_US": w.surface_peak_halfwidth_us,
            "MID_LO_US": w.midcolumn.after_surface_us,
            "MID_HI_US": w.midcolumn.before_bed_us,
            "BED_LO_US": w.bed.before_us,
            "BED_HI_US": w.bed.after_us,
            "SCOUT_LO_US": w.scout_contrast.lo_us,
            "SCOUT_HI_US": w.scout_contrast.hi_us,
            "SCOUT_PK_US": w.scout_contrast.peak_halfwidth_us,
            "FLOOR_TAIL_LO_US": nf.record_end_window_us[0],
            "FLOOR_TAIL_HI_US": nf.record_end_window_us[1],
            "FLOOR_BED_GUARD_US": nf.bed_guard_us,
            "FLOOR_ROLLOFF_US": nf.rolloff_us,
            "FLOOR_MIN_WIDTH_US": nf.min_width_us,
            "TAIL_PROF_US": tuple(bt.profile_us),
            "TAIL_FIT_US": tuple(bt.fit_us),
            "TAIL_EXCESS_US": tuple(bt.excess_delays_us),
            "TAIL_GUARD_DB": bt.guard_db,
            "TAIL_FLOOR_MARGIN_DB": bt.floor_margin_db,
            "GAMMA_SURFACE_SOLVE": self.gamma_surface_solve.model_dump(),
            "ATTENUATION_REGRESSION":
                self.attenuation_regression.model_dump(),
            "CORR_WIN_M": sm.profile_m,
            "ROUGH_WIN_M": sm.roughness_detrend_m,
            "GL_RAMP_KM": hb.gl_ramp_km,
            "EPS_SEAWATER": hb.seawater_eps,
            "N_LOOKS_SIM": self.processing.n_looks,
            "RADARGRAM_PCT": tuple(self.figures.radargram_percentiles),
            "CHUNK_M": self.compute.chunk_m,
            "CHUNK_M_PROC": self.compute.chunk_m_fine_posting,
        }

    def merged(self, override):
        """Deep-merge a LINE's partial override over these defaults."""
        if not override:
            return self, {}

        def deep(base, over, path=""):
            out, changed = dict(base), {}
            for k, v in over.items():
                if k not in base:
                    raise ValueError(f"analysis override sets unknown key "
                                     f"{path}{k}")
                if isinstance(v, dict) and isinstance(base[k], dict):
                    out[k], sub = deep(base[k], v, f"{path}{k}.")
                    changed.update(sub)
                else:
                    out[k] = v
                    changed[f"{path}{k}"] = {"default": base[k], "line": v}
            return out, changed

        merged, changed = deep(self.model_dump(), override)
        return AnalysisSpec.model_validate(merged), changed


def load_analysis(path=None):
    with Path(path or ANALYSIS_PATH).open() as fh:
        return AnalysisSpec.model_validate(yaml.safe_load(fh))
