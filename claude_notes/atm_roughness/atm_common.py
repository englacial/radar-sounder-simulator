"""Shared helpers: study-line nav, anchor axis, ATM flight-day lookup.

The anchor axis is the one the radar products use (tools/line_geometry):
arc length over the UNSLICED reference-pass frames, s = 0 at the first trace
(the YAML slice indices x ~14.85 m posting reproduce the segment s0_km).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import line_geometry as lg  # noqa: E402

CACHE = ROOT / "outputs" / "cache"
ATM_CACHE = CACHE / "atm"

# radar flight days with ATM on the same aircraft (task brief); frames listed
# are the pass's full-segment frames on that day
ATM_DAYS = {
    "greenland_westcoast": {
        "2016-05-11": ("2016_Greenland_P3", ["20160511_03_033", "20160511_03_034", "20160511_03_035"]),
        "2017-05-10": ("2017_Greenland_P3", ["20170510_03_013", "20170510_03_014"]),
        "2019-05-14": ("2019_Greenland_P3", ["20190514_01_039", "20190514_01_040"]),
    },
    "greenland_geikie01_transit": {
        "2014-04-21": ("2014_Greenland_P3", ["20140421_01_069", "20140421_01_070", "20140421_01_071"]),
        "2017-04-24": ("2017_Greenland_P3", ["20170424_01_067", "20170424_01_068", "20170424_01_069"]),
    },
    "antarctica_getz": {
        "2016-10-28": ("2016_Antarctica_DC8", ["20161028_05_004", "20161028_05_005", "20161028_05_006", "20161028_05_007"]),
        "2016-10-31": ("2016_Antarctica_DC8", ["20161031_07_002", "20161031_07_003", "20161031_07_004", "20161031_07_005"]),
        "2016-11-05": ("2016_Antarctica_DC8", ["20161105_05_005", "20161105_05_006", "20161105_05_007"]),
    },
    "antarctica_pineisland_south": {
        "2014-10-29": ("2014_Antarctica_DC8", ["20141029_05_024", "20141029_05_025", "20141029_05_026"]),
        "2016-11-04": ("2016_Antarctica_DC8", ["20161104_05_019", "20161104_05_020", "20161104_05_021"]),
        "2018-11-07": ("2018_Antarctica_DC8", ["20181107_01_022", "20181107_01_023", "20181107_01_024"]),
    },
    "antarctica_pineisland_north": {
        "2014-10-29": ("2014_Antarctica_DC8", ["20141029_05_012", "20141029_05_013", "20141029_05_014"]),
        "2016-11-04": ("2016_Antarctica_DC8", ["20161104_05_008", "20161104_05_009", "20161104_05_010"]),
        "2018-11-07": ("2018_Antarctica_DC8", ["20181107_01_011", "20181107_01_012", "20181107_01_013"]),
    },
    # david: Basler/MKB only -- no ATM; handled by bbox search across years
    "antarctica_david": {},
}


def load_line(name):
    return yaml.safe_load((ROOT / "config" / "lines" / f"{name}.yaml").read_text())


def open_frame(season, fid):
    return xr.open_dataset(CACHE / f"frame_{season}_{fid}_CSARP_standard.nc", engine="h5netcdf")


def frames_nav(season, fids):
    """lat, lon, time (datetime64) concatenated over frames."""
    lat, lon, t = [], [], []
    for fid in fids:
        f = open_frame(season, fid)
        lat.append(f.Latitude.values); lon.append(f.Longitude.values); t.append(f.slow_time.values)
    return np.concatenate(lat), np.concatenate(lon), np.concatenate(t)


def anchor_axis(line):
    """(xy_ref, s_ref, crs) of the line's reference pass, unsliced frames."""
    spec = load_line(line)
    crs = spec["identity"]["crs"]
    ref = spec["passes"][spec["reference"]["pass"]]
    season = ref.get("season", spec["identity"]["season"])
    lat, lon, _ = frames_nav(season, spec["reference"]["frames"])
    xy = lg.to_crs(lat, lon, crs)
    return xy, lg.arc_length(xy), crs


def bbox_deg(lat, lon, pad_m=2000.0):
    dlat = pad_m / 111.2e3
    dlon = pad_m / (111.2e3 * np.cos(np.radians(np.mean(lat))))
    return (float(lon.min() - dlon), float(lat.min() - dlat), float(lon.max() + dlon), float(lat.max() + dlat))
