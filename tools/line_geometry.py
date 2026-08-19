"""Along-track projection shared by the line report and slice proposal.

One reference pass defines the ANCHOR AXIS: a polyline in the line's CRS with
a cumulative along-track coordinate s. Every other pass is projected onto it,
which turns "these flights overlap" into two numbers per trace -- where it
sits along the line (s) and how far off it flew (lateral offset).

Discovery can run on STAC geometry, which is a coarse decimation of the
track; MEASUREMENT must not. A frame's STAC LineString can sit hundreds of
metres from where the aircraft actually was, so every offset reported to a
user comes from the frame's own nav.
"""

from __future__ import annotations

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

C = 299792458.0


def to_crs(lat, lon, crs):
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tr.transform(np.asarray(lon), np.asarray(lat))
    return np.column_stack([np.asarray(x), np.asarray(y)])


def arc_length(xy):
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])


def project(xy_q, xy_ref, s_ref):
    """(s, signed lateral offset) of query points on the reference polyline.

    Nearest reference sample plus its tangential offset -- exact for a
    straight track and accurate for the smooth, densely sampled tracks here.
    The lateral sign is left of track positive."""
    ux, uy = np.gradient(xy_ref[:, 0]), np.gradient(xy_ref[:, 1])
    n = np.hypot(ux, uy)
    ux, uy = ux / np.where(n == 0, 1, n), uy / np.where(n == 0, 1, n)
    _, i = cKDTree(xy_ref).query(xy_q)
    dx = xy_q[:, 0] - xy_ref[i, 0]
    dy = xy_q[:, 1] - xy_ref[i, 1]
    return s_ref[i] + dx * ux[i] + dy * uy[i], -dx * uy[i] + dy * ux[i]


def frame_nav(frame, crs):
    """(xy, lat, lon) for one loaded frame, in the line's CRS."""
    lat = np.asarray(frame.Latitude.values, np.float64)
    lon = np.asarray(frame.Longitude.values, np.float64)
    return to_crs(lat, lon, crs), lat, lon


def agl_m(frame):
    """Nadir air range from the frame's own surface pick."""
    return np.asarray(frame.Surface.values, np.float64) * C / 2.0


def slice_to_span(s, lo, hi):
    """Half-open trace index range whose along-track s lies in [lo, hi].

    Returns None when the pass never enters the span. The range is
    CONTIGUOUS by construction (first..last in-span index), so a track that
    leaves and re-enters is reported whole with its coverage fraction, rather
    than silently split."""
    m = (s >= lo) & (s <= hi)
    if not m.any():
        return None, 0.0
    idx = np.where(m)[0]
    a, b = int(idx[0]), int(idx[-1]) + 1
    return (a, b), float(m[a:b].mean())
