"""Tier 2 shared: paths, site list, ILATM1B readers (HDF5 v2 and Qfit v1 binary)."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "atm_regional" / "tier2"
CACHE = ROOT / "outputs" / "cache" / "atm" / "tier2"
ATM2 = ROOT / "outputs" / "cache" / "atm2"
LOGS = Path(__file__).resolve().parent / "logs"
for p in (OUT, CACHE, LOGS): p.mkdir(parents=True, exist_ok=True)
CRS = {"gl": "EPSG:3413", "aa": "EPSG:3031"}
SITE_RADIUS_M = 2500.0     # 5 km site


def sites():
    return pd.read_csv(OUT / "sites.csv")


def _hhmmss_to_s(v):
    v = np.asarray(v, dtype=np.float64)
    hr = np.floor(v / 1e4); mn = np.floor((v - hr * 1e4) / 100); sc = v - hr * 1e4 - mn * 100
    return hr * 3600 + mn * 60 + sc


def read_h5(path):
    import h5py
    with h5py.File(path) as h:
        return dict(lat=h["latitude"][:].astype(np.float64), lon=h["longitude"][:].astype(np.float64),
                    h=h["elevation"][:].astype(np.float64), t=_hhmmss_to_s(h["instrument_parameters/time_hhmmss"][:]),
                    rcv=h["instrument_parameters/rcv_sigstr"][:].astype(np.float32),
                    pw=h["instrument_parameters/pulse_width"][:].astype(np.float32))


def read_qi(path):
    """Qfit binary (ILATM1B v1, 2009-2012): little-endian int32 records of 12 or 14 words.
    First word of the file = record length in bytes; negative first words = header records.
    12-word: t(ms) lat*1e6 lon*1e6 elev(mm) xmt rcv az pitch roll pdop*10 pw gps(hhmmssss)
    14-word: t lat lon elev xmt rcv az pitch roll passive_sig p_lat p_lon p_elev gps"""
    raw = np.fromfile(path, dtype="<i4")
    nbytes = int(raw[0])
    if nbytes not in (48, 56):
        raw = raw.byteswap(); nbytes = int(raw[0])
    nw = nbytes // 4
    rec = raw[: len(raw) // nw * nw].reshape(-1, nw)
    data = rec[(rec[:, 0] > 0) & (rec[:, 1] != 0)]
    data = data[1:] if data[0, 0] == nbytes else data
    gps = data[:, -1].astype(np.float64)   # hhmmssss (ms)
    t = _hhmmss_to_s(gps / 1000.0)
    pw = data[:, 10].astype(np.float32) if nw == 12 else np.full(len(data), np.nan, np.float32)
    return dict(lat=data[:, 1] / 1e6, lon=((data[:, 2] / 1e6 + 180) % 360) - 180, h=data[:, 3] / 1e3,
                t=t, rcv=data[:, 5].astype(np.float32), pw=pw)


def read_granule(path):
    path = Path(path)
    return read_h5(path) if path.suffix == ".h5" else read_qi(path)
