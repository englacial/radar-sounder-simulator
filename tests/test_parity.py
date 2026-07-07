"""Parity vs cached simc fixtures: flat scene in CI, all five in integration.

Metric conventions and threshold adjustments (docs/incoherent_simulation.md
allows loosening only with written justification -- given here with numbers):

1. Shape metrics (peak alignment, Pearson, dB residual) are evaluated on
   fast-time profiles aggregated to the facet scale (posting / (c*dt/2) = 33
   raw bins of 1.5 m), with the documented thresholds. At raw 1.5 m bins a bin
   holds O(1-10) 50 m facets and simc's per-trace track-aligned regrid (50 m
   ECEF steps ~ 49.3 m projected) is a *different tessellation of the same
   surface* than our fixed 50 m projected grid, so raw per-bin power is facet-
   placement shot noise: on the flat scene raw per-bin ratios scatter 0.3-3x
   (raw Pearson 0.77, raw peak argmax scatter up to 17 bins among statistically
   equal leading-edge bins) while facet-scale profiles agree to Pearson
   >= 0.993, peak alignment 0, and <= 0.7 dB RMS on every scene. The raw-bin
   values are recorded in the metric dict for tracking.

2. First-return bin tolerance is +-1 (documented) for flat/hill but +-3 for the
   sloped scenes (tilted, sinusoid, crater) in the integration suite. Checked
   against a densely upsampled surface, *our* first return is within +-1 bin of
   ground truth on all five scenes; simc's is off by up to 3 bins on sloped
   scenes because its int-truncation DEM sampling places heights up to one
   pixel away from their true position (known simc divergence, see
   docs/incoherent_simulation.md "Explicit divergences").

3. The absolute power ratio ours/simc is ~1.028 on every scene: our facets are
   50 m in *projected* EPSG:3413 meters = 50/k true meters (scale factor
   k = 0.98666 at 75N), simc's are 50 true meters; total incoherent power over
   a fixed ground area scales with facet area, (1/k)^2 = 1.027. The gate is the
   documented one: per-trace ratio constancy (CV <= 3%; observed <= 2e-4).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from soundersim import simulate
from soundersim.compare import plots
from soundersim.compare.metrics import compare_to_simc, load_fixture
from soundersim.config import FacetConfig, RadarConfig, SimConfig
from soundersim.synthetic import ALL_SCENES

FIXTURES = Path(__file__).parent / "fixtures"
OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "verification"
SCENES = {f.__name__.removesuffix("_scene"): f for f in ALL_SCENES}
FRET_TOL = {"tilted": 3, "sinusoid": 3, "crater": 3}  # see module docstring


def simulate_scene(name):
    """Load a fixture, run the matching scene, return (ds, fixture, radar cfg)."""
    fixture = load_fixture(name, FIXTURES)
    meta = fixture["meta"]
    scene = SCENES[name](**meta["scene"]["params"])
    rc = RadarConfig(**meta["radar_config"])
    cfg = SimConfig(mode="incoherent", split_sides=True, radar=rc,
                    facets=FacetConfig(spacing=None))
    return simulate(scene, cfg), fixture, rc


def run_parity(name):
    ds, fixture, _ = simulate_scene(name)
    return compare_to_simc(ds, fixture, fret_tol=FRET_TOL.get(name, 1))


def assert_all_pass(name, metrics):
    failed = {k: v for k, v in metrics.items() if not v["pass"]}
    assert not failed, f"{name}: failed metrics: {json.dumps(failed, indent=1)}"


def write_artifacts(name, ds, fixture, rc, metrics):
    """Emit metrics.json + the two shape figures per the artifact convention."""
    meta = fixture["meta"]
    agg = int(round(meta["scene"]["params"]["posting"] / (rc.c * rc.dt / 2)))
    ours = np.asarray(ds.power.sum("side"), np.float64)      # (T, n_samples)
    simc = np.asarray(fixture["cluttergram"], np.float64).T  # (T, n_samples)
    twtt = np.asarray(ds.twtt.values, np.float64)
    d = OUTDIR / name
    plots.write_metrics(
        d / "metrics.json", name, metrics, group="simc comparison",
        notes=f"simc parity, facet-scale agg={agg} (~{agg * rc.c * rc.dt / 2:.0f} m); "
              "shape metrics on facet-scale profiles, fret/power on raw bins.")
    plots.three_panel_db(d / "cluttergram_db.png", ours, simc, agg, twtt=twtt,
                         title=f"{name}: soundersim vs simc (facet scale)")
    plots.profile_overlays(d / "profiles_db.png", ours, simc, agg, twtt=twtt,
                           title=f"{name}: per-trace power profiles")


def test_flat_parity():
    """CI-fast: full metric set against the flat fixture, strict thresholds."""
    metrics = run_parity("flat")
    assert_all_pass("flat", metrics)
    assert metrics["first_return_bin"]["threshold"] == 1


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(SCENES))
def test_scene_parity(name):
    ds, fixture, rc = simulate_scene(name)
    metrics = compare_to_simc(ds, fixture, fret_tol=FRET_TOL.get(name, 1))
    write_artifacts(name, ds, fixture, rc, metrics)
    assert_all_pass(name, metrics)
