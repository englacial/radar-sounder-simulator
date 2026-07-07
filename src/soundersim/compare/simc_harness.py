"""Drive the simc clutter simulator programmatically to produce parity fixtures.

simc is imported from the pinned git commit (dev dependency). We bypass its CLI,
ini parsing, and nav loaders entirely: the confDict is built here with the
lowercased key names its code actually reads, and nav is supplied as the pandas
DataFrame (x, y, z ECEF + datum) that simc.prep.prep expects. prep computes the
uv/ul track unit vectors and the duplicate-removal inverse index.

Binning convention: simc bins at int((twtt - datum) / dt) with a mod-tracesamples
wrap. We set datum = radar_config.t0 for every trace and require n_samples large
enough that no return reaches the wrap (callers should size the window from the
scene geometry; see tools/make_fixtures.py).
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
from simc import output as simc_output
from simc import prep as simc_prep
from simc import sim as simc_sim

from ..config import RadarConfig
from ..synthetic import SyntheticScene, nav_ecef, write_dem_geotiff

ECEF = "+proj=geocent +ellps=WGS84 +datum=WGS84 +no_defs"
LLE = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"


def build_confdict(radar_config: RadarConfig, at_dist, ct_dist, at_step, ct_step) -> dict:
    """simc confDict with exactly the (lowercased) keys sim/prep/output read."""
    outputs = {k: False for k in (
        "shownadir", "showfret", "combinedadj", "combinedcolored", "binary",
        "echomap", "echomapadj", "echomapcolored", "echomapgeoref", "nadir",
        "exportfacetsarray", "exportfacetsarrayh5")}
    outputs.update(combined=True, left=True, right=True, fret=True)
    return {
        "simParams": {
            "speedlight": radar_config.c,
            "dt": radar_config.dt,
            "tracesamples": radar_config.n_samples,
            "dembump": False,
            "deminterp": False,
            "centerplane": False,
            "antenna_pattern": "none",
            "coherent": False,
            "body": "earth",
        },
        "facetParams": {
            "atdist": float(at_dist), "ctdist": float(ct_dist),
            "atstep": float(at_step), "ctstep": float(ct_step),
        },
        "outputs": outputs,
        "navigation": {"xyzsys": ECEF, "llesys": LLE},
    }


def run_simc(scene: SyntheticScene, radar_config: RadarConfig,
             at_dist, ct_dist, at_step, ct_step) -> dict:
    """Run simc on a synthetic scene; return cluttergrams, first-return info, confDict."""
    confDict = build_confdict(radar_config, at_dist, ct_dist, at_step, ct_step)

    xyz = nav_ecef(scene)
    nav = pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                        "datum": np.full(len(xyz), radar_config.t0)})

    with tempfile.TemporaryDirectory() as tmp:
        dem_path = Path(tmp) / f"{scene.name}.tif"
        write_dem_geotiff(scene, dem_path)
        with rio.open(dem_path) as dem:
            xform = pyproj.Transformer.from_crs(ECEF, dem.crs, always_xy=True)
            nav, oDict, inv = simc_prep.prep(confDict, dem, nav)
            win = rio.windows.Window(0, 0, dem.width, dem.height)
            demData = dem.read(1, window=win)
            for i in range(nav.shape[0]):
                fcalc = simc_sim.sim(confDict, dem, nav, xform, demData, win, i)
                if fcalc.shape[0] == 0:
                    continue
                oi = np.where(inv == i)[0]
                simc_output.build(confDict, oDict, fcalc, dem, win, xform, nav, i, oi)

    # First-return info (as output.save computes it, without its file side effects):
    # fret = ECEF position of the min-adjusted-twtt facet; bin via int truncation.
    fret_xyz = oDict["fret"][:, 0:3].copy()
    x, y, z = (nav[k].to_numpy() for k in "xyz")
    fr = np.sqrt((x - fret_xyz[:, 0]) ** 2 + (y - fret_xyz[:, 1]) ** 2
                 + (z - fret_xyz[:, 2]) ** 2)
    fret_twtt = 2 * fr / radar_config.c
    fret_bin = ((fret_twtt - nav["datum"].to_numpy()) / radar_config.dt).astype(np.int32)

    return {
        "cluttergram": oDict["combined"],  # (n_samples, n_traces) float64
        "left": oDict["left"],
        "right": oDict["right"],
        "fret_xyz": fret_xyz,  # (n_traces, 3) ECEF
        "fret_twtt": fret_twtt,
        "fret_bin": fret_bin,
        "nav_ecef": xyz,
        "confDict": confDict,
    }
