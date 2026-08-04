"""DEMOGORGN bed source: the geoid band-2 BedMachine cache change, the
fetch_demogorgn_window cache/datum path, and the tool flag plumbing. All
offline: caches are pre-seeded, network entry points are stubbed.
"""

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_basal_clutter as rbc  # noqa: E402

from soundersim import opr  # noqa: E402

BOUNDS = (-104.0, -75.2, -103.5, -75.0)   # lon/lat bbox in the DGN domain


def _key(parts):
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:12]


def _write_tif(path, bands, transform, crs="EPSG:3031"):
    with rasterio.open(path, "w", driver="GTiff", height=bands[0].shape[0],
                       width=bands[0].shape[1], count=len(bands),
                       dtype="float32", crs=crs, transform=transform,
                       nodata=opr.NODATA) as dst:
        for i, b in enumerate(bands):
            dst.write(b.astype(np.float32), i + 1)


def _grid(pb, step, value, extra=0.0):
    """(array, transform) covering padded proj bounds pb at ``step``."""
    x0, y0, x1, y1 = pb
    nx = int(np.ceil((x1 - x0) / step)) + 3
    ny = int(np.ceil((y1 - y0) / step)) + 3
    arr = np.full((ny, nx), value, np.float32) + extra
    tr = rasterio.transform.from_origin(x0 - step, y1 + step, step, step)
    return arr, tr


def _seed_bedmachine(cache, pad, bed=100.0, geoid=-35.0, bands=2):
    prod = opr.BEDMACHINE["antarctic"]
    pb = opr._padded_proj_bounds(BOUNDS, prod["crs"], pad)
    key = _key([prod["url"], [round(b, 1) for b in pb]])
    tif = cache / f"bedmachine_antarctic_{key}.tif"
    a, tr = _grid(pb, prod["posting"], bed)
    g, _ = _grid(pb, prod["posting"], geoid)
    _write_tif(tif, [a, g][:bands], tr)
    tif.with_suffix(".json").write_text(json.dumps(
        {"product": prod["product"], "geoid_band": 2}) + "\n")
    return tif


# --------------------------------------------- BedMachine geoid band 2 gate

def test_two_band_cache_reads_offline(tmp_path):
    _seed_bedmachine(tmp_path, pad=0.0, bed=123.0, geoid=-35.0)
    bed, tr, crs, meta = opr.fetch_bedmachine_window(
        BOUNDS, "antarctic", pad_m=0.0, cache_dir=tmp_path)
    assert np.allclose(bed, 123.0)          # band 1 only: already ellipsoidal
    g, trg, crsg = opr.bedmachine_geoid_window(BOUNDS, "antarctic",
                                               pad_m=0.0, cache_dir=tmp_path)
    assert np.allclose(g, -35.0)
    assert crsg == "EPSG:3031"


def test_single_band_cache_treated_as_miss(tmp_path, monkeypatch):
    """A pre-geoid-band (count==1) cache must REFETCH, never silently serve
    without the geoid band -- proven by the stubbed earthaccess being hit."""
    _seed_bedmachine(tmp_path, pad=0.0, bands=1)

    class Sentinel(RuntimeError):
        pass

    stub = types.ModuleType("earthaccess")

    def _boom(*a, **k):
        raise Sentinel("refetch attempted")

    stub.login = _boom
    monkeypatch.setitem(sys.modules, "earthaccess", stub)
    with pytest.raises(Sentinel):
        opr.fetch_bedmachine_window(BOUNDS, "antarctic", pad_m=0.0,
                                    cache_dir=tmp_path)


# ------------------------------------------------- fetch_demogorgn_window

def test_demogorgn_cache_datum_and_meta(tmp_path):
    """Cached RAW realization + BedMachine band-2 geoid -> ellipsoidal on
    read; meta carries the pinned snapshot, seed and datum notes."""
    pad = 0.0
    _seed_bedmachine(tmp_path, pad=pad, bed=0.0, geoid=-35.0)
    pb = opr._padded_proj_bounds(BOUNDS, opr.DEMOGORGN["crs"], pad)
    key = _key([opr.DEMOGORGN["bucket"], opr.DEMOGORGN["prefix"],
                opr.DEMOGORGN_SNAPSHOT, 0,
                [round(b, 1) for b in pb]])
    tif = tmp_path / f"demogorgn_antarctic_seed000_{key}.tif"
    raw, tr = _grid(pb, 500.0, -400.0)      # raw geoid-referenced bed
    # stagger the DGN grid 250 m from the BedMachine grid (the real layout)
    tr = tr * rasterio.Affine.translation(0.5, 0.5)
    _write_tif(tif, [raw], tr)
    tif.with_suffix(".json").write_text(json.dumps({
        "snapshot_id": opr.DEMOGORGN_SNAPSHOT, "seed_id": 0,
        "posting_m": 500.0, "vertical_datum": "raw geoid-referenced"}) + "\n")

    bed, tr2, crs, meta = opr.fetch_demogorgn_window(
        BOUNDS, pad_m=pad, seed=0, cache_dir=tmp_path)
    inner = bed[1:-1, 1:-1]                 # geoid resample edge-clamps
    assert np.allclose(inner[np.isfinite(inner)], -435.0)  # -400 + (-35)
    assert meta["snapshot_id"] == opr.DEMOGORGN_SNAPSHOT
    assert meta["seed_id"] == 0
    assert "ellipsoid" in meta["returned_datum"]


def test_demogorgn_key_pins_snapshot_and_seed(tmp_path):
    """A different snapshot or seed changes the cache key: nothing cached ->
    the store path is entered (stubbed icechunk import proves it; a silent
    stale-cache serve would not raise)."""
    _seed_bedmachine(tmp_path, pad=0.0)

    class Sentinel(RuntimeError):
        pass

    stub = types.ModuleType("icechunk")

    def _boom(*a, **k):
        raise Sentinel("store fetch attempted")

    stub.s3_storage = _boom
    import unittest.mock as um
    with um.patch.dict(sys.modules, {"icechunk": stub}):
        with pytest.raises(Sentinel):
            opr.fetch_demogorgn_window(BOUNDS, pad_m=0.0, seed=7,
                                       cache_dir=tmp_path)
        with pytest.raises(Sentinel):
            opr.fetch_demogorgn_window(BOUNDS, pad_m=0.0, seed=0,
                                       snapshot="DIFFERENT",
                                       cache_dir=tmp_path)


# ------------------------------------------------------------ tool plumbing

def test_case_tag_with_dgn():
    assert rbc.case_tag(False, True, True, True) == "_dgn_rssnr_proc"
    assert rbc.case_tag(False, False, False, True) == "_dgn"
    # existing cases unchanged
    assert rbc.case_tag(True, True, True) == "_pbed_rssnr_proc"
    assert rbc.case_tag(True) == "_pbed"


def test_dgn_plus_picked_bed_raises():
    with pytest.raises(ValueError, match="follow-up"):
        rbc.run(segment="full", picked_bed=True, demogorgn_bed=True)


def test_nadir_bed_offset_math():
    # bot_sim = the Bottom pick ON THE SIM TRACES (prep_pass contract:
    # bot[idx] at the product posting, the refined grid at --posting-div>1)
    p = {"bot": np.array([10e-6, 10e-6, 10e-6, 99.0]),
         "idx": np.array([0, 1, 2]),
         "bot_sim": np.array([10e-6, 10e-6, 10e-6])}
    sim = {"nadir": np.column_stack([np.full(3, 5e-6),
                                     np.full(3, 10.5e-6)])}
    off = rbc.nadir_bed_offset(p, sim)
    assert off["med_us"] == pytest.approx(0.5, abs=1e-9)
    # 0.5 us two-way in ice: c/(2 sqrt(eps)) * 0.5e-6 ~ 42.1 m
    assert off["med_m_ice"] == pytest.approx(
        0.5e-6 * 299792458.0 / (2 * np.sqrt(rbc.rac.EPS_ICE)), abs=0.1)
