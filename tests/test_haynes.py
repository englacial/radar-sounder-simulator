"""Haynes et al. 2018 geometric fall-off check (integration, simc-independent).

Flat scene simulated across a sweep of altitudes:

- Per-facet: the nadir-most facet's power (A cosθ)^2/r^4 goes as r^-4 (a single
  specular plate; r ~ h at nadir). Near-exact.
- Aggregate: leading-edge power -- the power summed over one range-resolution
  window (agg raw bins = the facet scale) starting at the first-return bin --
  goes as r^-3. In the pulse-limited/rough regime the annulus that falls in the
  leading range cell has area ~ pi(2 h Δρ) so its facet count grows ∝ r, and
  r * r^-4 = r^-3 (docs/incoherent_simulation.md, Haynes Table I rough row).

Aggregation alignment note: the window is anchored to each altitude's
first-return bin (raw[fret : fret+agg]); the fixed facet-scale aggregation grid
(_aggregate) is NOT aligned to it, so indexing a fixed aggregated bin splits the
leading annulus across two cells and corrupts the slope (observed -2.8 / -3.5).
The annulus depth is exactly one range resolution by construction, so the
anchored window captures it whole. At 8 km the annulus radius sqrt(2 h Δρ) ~
0.9 km sits well inside the 8x8 km scene and no power is dropped.
"""

from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer

from soundersim import simulate
from soundersim.compare import plots
from soundersim.config import FacetConfig, RadarConfig, SimConfig
from soundersim.nav import nav_to_frame
from soundersim.scene import LocalFrame, build_facets
from soundersim.synthetic import flat_scene, nav_ecef

OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "verification"
C = 299792458.0
DT = 10e-9
ALTITUDES = [500.0, 1000.0, 2000.0, 4000.0, 8000.0]


def _n_samples_for(scene, dt, c):
    """Bins covering [0, max twtt]: max ECEF range from any nav pos to any DEM
    node (facet centroids are convex combinations of nodes, so bounded by it)."""
    ny, nx = scene.dem.shape
    a = scene.transform
    xs = a.c + (np.arange(nx) + 0.5) * a.a
    ys = a.f + (np.arange(ny) + 0.5) * a.e
    X, Y = np.meshgrid(xs, ys)
    lon, lat = Transformer.from_crs(scene.crs, "EPSG:4326", always_xy=True).transform(X, Y)
    ex, ey, ez = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True).transform(
        lon, lat, scene.dem.astype(np.float64))
    nodes = np.column_stack([ex.ravel(), ey.ravel(), ez.ravel()])
    max_r = max(np.linalg.norm(nodes - p, axis=1).max() for p in nav_ecef(scene))
    return int(np.ceil(2 * max_r / c / dt)) + 4


def _nadir_facet_power(scene):
    """(A cosθ)^2/r^4 of the horizontally nearest facet, and its range r."""
    frame = LocalFrame.centered_on(scene)
    facets = build_facets(scene.dem, scene.transform, scene.crs, frame, spacing=None)
    track = nav_to_frame(scene.nav_llh, frame)
    p = track.positions[len(track.positions) // 2]
    i = int(((facets.centers[:, :2] - p[:2]) ** 2).sum(1).argmin())
    rv = p - facets.centers[i]
    r = float(np.linalg.norm(rv))
    cos = abs(float(rv @ facets.normals[i]) / r)
    return (facets.areas[i] * cos) ** 2 / r ** 4, r


@pytest.mark.integration
def test_haynes_altitude_sweep():
    r, per_facet, lead_edge, max_dropped = [], [], [], 0.0
    for h in ALTITUDES:
        scene = flat_scene(altitude=h, n_traces=3)
        posting = scene.params["posting"]
        agg = int(round(posting / (C * DT / 2)))
        rc = RadarConfig(dt=DT, n_samples=_n_samples_for(scene, DT, C), t0=0.0)
        cfg = SimConfig(mode="incoherent", split_sides=True, radar=rc,
                        facets=FacetConfig(spacing=None))
        ds = simulate(scene, cfg)

        power = np.asarray(ds.power.sum("side"), np.float64)
        total = power.sum(axis=1)
        dropped = np.asarray(ds.dropped_power.values, np.float64)
        max_dropped = max(max_dropped, float((dropped / total).max()))
        # Window covers the whole scene, so essentially nothing is dropped.
        assert (dropped / total).max() < 1e-6

        # Leading-edge power: one range-resolution window anchored at first return.
        fret = np.floor((ds.first_return_twtt.values - rc.t0) / rc.dt).astype(int)
        annulus_r = np.sqrt(2 * h * agg * C * DT / 2)
        assert annulus_r < scene.params["extent"] / 2  # annulus inside the scene
        lead = np.array([power[t, fret[t]:fret[t] + agg].sum()
                         for t in range(power.shape[0])])
        # The window must actually hold the leading returns.
        assert np.all(lead > 0)

        pf, rr = _nadir_facet_power(scene)
        r.append(rr)
        per_facet.append(pf)
        lead_edge.append(float(lead.mean()))

    r = np.array(r)
    pf_slope = float(np.polyfit(np.log(r), np.log(per_facet), 1)[0])
    le_slope = float(np.polyfit(np.log(r), np.log(lead_edge), 1)[0])

    metrics = {
        "per_facet_r4_slope": {"value": pf_slope, "threshold": 0.05,
                               "target": -4.0, "pass": abs(pf_slope + 4.0) <= 0.05,
                               "tolerance": "+-0.05"},
        "leading_edge_r3_slope": {"value": le_slope, "threshold": 0.1,
                                  "target": -3.0, "pass": abs(le_slope + 3.0) <= 0.1,
                                  "tolerance": "+-0.1"},
        "max_dropped_power_frac": {"value": max_dropped, "threshold": 1e-6,
                                   "pass": max_dropped < 1e-6},
    }
    plots.write_metrics(
        OUTDIR / "haynes" / "metrics.json", "haynes", metrics,
        group="Radar equation comparison",
        notes=f"Flat scene, altitudes {ALTITUDES} m. Per-facet nadir power fits "
              f"r^-4 ({pf_slope:+.3f}); leading-edge power fits r^-3 "
              f"({le_slope:+.3f}, Haynes 2018 rough/pulse-limited regime).")
    plots.haynes_loglog(OUTDIR / "haynes" / "haynes_loglog.png", r, per_facet,
                        lead_edge, per_facet_slope=pf_slope, lead_slope=le_slope)

    assert abs(pf_slope + 4.0) <= 0.05, f"per-facet slope {pf_slope}"
    assert abs(le_slope + 3.0) <= 0.1, f"leading-edge slope {le_slope}"
