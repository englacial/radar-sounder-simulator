"""Regenerate cached simc reference fixtures for the five synthetic scenes.

Writes tests/fixtures/<scene>.npz + <scene>.json. Run: uv run python tools/make_fixtures.py

Conventions (recorded in each json): datum = t0 = 0 for every trace, so bin k
spans twtt [k*dt, (k+1)*dt). n_samples is sized from scene geometry to contain
every return, so simc's mod-tracesamples wrap never triggers.
"""

import datetime
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

from soundersim.compare.simc_harness import run_simc
from soundersim.config import RadarConfig
from soundersim.synthetic import ALL_SCENES, nav_ecef

SIMC_SHA = "bac8b97a66bc8bc327ff48a793031a6d1fcd6915"  # matches pyproject dev dep
DT = 10e-9
AT_DIST = CT_DIST = 4000.0  # covers the full 8x8 km DEM extent
AT_STEP = CT_STEP = 50.0  # = DEM posting

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def n_samples_for(scene, dt, c):
    """Bins to cover [0, max twtt]: max ECEF distance nav -> any DEM node, +margin.

    Facet centroids are averages of DEM surface nodes, so their ranges are
    bounded by the max node range (distance to a point is convex).
    """
    ny, nx = scene.dem.shape
    a = scene.transform
    xs = a.c + (np.arange(nx) + 0.5) * a.a
    ys = a.f + (np.arange(ny) + 0.5) * a.e
    X, Y = np.meshgrid(xs, ys)
    lon, lat = Transformer.from_crs(scene.crs, "EPSG:4326", always_xy=True).transform(X, Y)
    ex, ey, ez = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True).transform(
        lon, lat, scene.dem.astype(np.float64))
    nodes = np.column_stack([ex.ravel(), ey.ravel(), ez.ravel()])
    nav = nav_ecef(scene)
    max_r = max(np.linalg.norm(nodes - p, axis=1).max() for p in nav)
    return int(np.ceil(2 * max_r / c / dt)) + 4


def main():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    created = datetime.date.today().isoformat()
    for make_scene in ALL_SCENES:
        scene = make_scene()
        rc = RadarConfig(dt=DT, n_samples=n_samples_for(scene, DT, 299792458.0), t0=0.0)
        res = run_simc(scene, rc, AT_DIST, CT_DIST, AT_STEP, CT_STEP)

        npz_path = FIXTURE_DIR / f"{scene.name}.npz"
        np.savez_compressed(
            npz_path,
            cluttergram=res["cluttergram"].astype(np.float32),
            left=res["left"].astype(np.float32),
            right=res["right"].astype(np.float32),
            fret_xyz=res["fret_xyz"],
            fret_twtt=res["fret_twtt"],
            fret_bin=res["fret_bin"],
            nav_ecef=res["nav_ecef"],
        )
        meta = {
            "simc_sha": SIMC_SHA,
            "simc_url": "https://github.com/lpl-tapir/simc",
            "created": created,
            "scene": {"name": scene.name, "crs": scene.crs, "params": scene.params},
            "radar_config": rc.model_dump(),
            "confDict": res["confDict"],
            "binning": "bin k spans twtt [t0 + k*dt, t0 + (k+1)*dt); datum = t0 = 0 "
                       "for all traces; window contains all returns (no wrap)",
        }
        json_path = FIXTURE_DIR / f"{scene.name}.json"
        json_path.write_text(json.dumps(meta, indent=1) + "\n")
        kb = npz_path.stat().st_size / 1024
        print(f"{scene.name}: n_samples={rc.n_samples}, {kb:.0f} kB")


if __name__ == "__main__":
    main()
