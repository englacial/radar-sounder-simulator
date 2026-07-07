"""Scene geometry (CPU, NumPy, float64).

A ``LocalFrame`` is an ENU frame anchored on the WGS84 ellipsoid. ``build_facets``
turns a projected DEM window into a triangle tessellation (centers/normals/areas)
expressed in that frame. Curvature matters over km scales, so projected x/y are
never treated as local metres: every vertex goes DEM -> projected -> geodetic ->
ECEF -> local ENU.
"""

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

_LLH = "EPSG:4979"  # geographic 3D (lon, lat, h) with always_xy
_ECEF = "EPSG:4978"


def _atleast2d(a):
    a = np.asarray(a, dtype=np.float64)
    return a[None, :] if a.ndim == 1 else a


class LocalFrame:
    """East-North-Up frame anchored at (lat0, lon0, h0) on the WGS84 ellipsoid."""

    def __init__(self, lat0, lon0, h0=0.0):
        self.lat0, self.lon0, self.h0 = float(lat0), float(lon0), float(h0)
        self._llh2ecef = Transformer.from_crs(_LLH, _ECEF, always_xy=True)
        self._ecef2llh = Transformer.from_crs(_ECEF, _LLH, always_xy=True)
        self.origin_ecef = np.array(
            self._llh2ecef.transform(self.lon0, self.lat0, self.h0), dtype=np.float64)
        lat, lon = np.deg2rad(self.lat0), np.deg2rad(self.lon0)
        sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
        # Rows are the E, N, U basis vectors in ECEF; R @ (xyz-origin) = enu.
        self.R = np.array([
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ], dtype=np.float64)

    @classmethod
    def centered_on(cls, scene=None, *, lat=None, lon=None, h0=0.0):
        """Anchor on a scene's DEM center, or on an explicit (lat, lon)."""
        if scene is not None:
            ny, nx = scene.dem.shape
            cx, cy = scene.transform * (nx / 2.0, ny / 2.0)
            lon0, lat0 = Transformer.from_crs(
                scene.crs, "EPSG:4326", always_xy=True).transform(cx, cy)
            return cls(lat0, lon0, h0)
        return cls(lat, lon, h0)

    def ecef_to_local(self, xyz):
        return (_atleast2d(xyz) - self.origin_ecef) @ self.R.T

    def local_to_ecef(self, enu):
        return _atleast2d(enu) @ self.R + self.origin_ecef

    def llh_to_local(self, llh):
        llh = _atleast2d(llh)
        x, y, z = self._llh2ecef.transform(llh[:, 1], llh[:, 0], llh[:, 2])
        return self.ecef_to_local(np.column_stack([x, y, z]))

    def local_to_llh(self, enu):
        xyz = self.local_to_ecef(enu)
        lon, lat, h = self._ecef2llh.transform(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        return np.column_stack([lat, lon, h])

    def up_at(self, enu):
        """Unit ellipsoidal-up vectors (local frame) at the given local points."""
        llh = self.local_to_llh(enu)
        lat, lon = np.deg2rad(llh[:, 0]), np.deg2rad(llh[:, 1])
        up_ecef = np.column_stack([
            np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        return up_ecef @ self.R.T


@dataclass
class Facets:
    """Triangle facets in a LocalFrame.

    Two triangles per rectangular DEM cell, sharing the (v00, v11) diagonal.
    Flat facet index of cell (i, j), triangle t in {0, 1} is
    ``(i * (nx-1) + j) * 2 + t`` -- see ``index_map``.
    """

    centers: np.ndarray   # (N, 3) float64 centroids in local ENU
    normals: np.ndarray   # (N, 3) float64 unit normals, +z (up) oriented
    areas: np.ndarray     # (N,)   float64 true ground areas
    cell: np.ndarray      # (N, 2) int: (row i, col j) source grid cell
    tri: np.ndarray       # (N,)   int in {0, 1}
    grid_shape: tuple     # (ny-1, nx-1) number of cells

    def index_map(self):
        """(ny-1, nx-1, 2) -> flat facet index."""
        n = self.grid_shape[0] * self.grid_shape[1] * 2
        return np.arange(n).reshape(*self.grid_shape, 2)


def build_facets(dem, transform, crs, frame, spacing=None):
    """Tessellate a projected DEM window into upward-oriented triangle facets.

    dem values are ellipsoidal heights (m). If ``spacing`` is given, pixel centers
    are subsampled (nearest) to roughly that posting before tessellation.
    """
    dem = np.asarray(dem, dtype=np.float64)
    ny, nx = dem.shape
    if spacing is not None:
        px = 0.5 * (abs(transform.a) + abs(transform.e))
        step = max(1, int(round(spacing / px)))
        dem = dem[::step, ::step]
        rows = np.arange(0, ny, step)
        cols = np.arange(0, nx, step)
    else:
        rows = np.arange(ny)
        cols = np.arange(nx)

    # Pixel centers -> projected (E, N), carrying ellipsoidal height.
    J, I = np.meshgrid(cols + 0.5, rows + 0.5)
    E, N = transform * (J, I)
    ecef = np.column_stack(
        Transformer.from_crs(crs, _ECEF, always_xy=True).transform(
            E.ravel(), N.ravel(), dem.ravel()))
    V = frame.ecef_to_local(ecef).reshape(*dem.shape, 3)  # (H, W, 3)

    v00, v01 = V[:-1, :-1], V[:-1, 1:]
    v10, v11 = V[1:, :-1], V[1:, 1:]
    # Triangle 0: (v00, v01, v11); Triangle 1: (v00, v11, v10). Shared diagonal v00-v11.
    c0 = (v00 + v01 + v11) / 3.0
    n0 = np.cross(v01 - v00, v11 - v00)
    c1 = (v00 + v11 + v10) / 3.0
    n1 = np.cross(v11 - v00, v10 - v00)

    centers = np.stack([c0, c1], axis=2).reshape(-1, 3)
    raw_n = np.stack([n0, n1], axis=2).reshape(-1, 3)
    mag = np.linalg.norm(raw_n, axis=1)
    areas = 0.5 * mag
    normals = raw_n / mag[:, None]
    normals *= np.where(normals[:, 2] < 0.0, -1.0, 1.0)[:, None]  # orient +up

    hc, wc = dem.shape[0] - 1, dem.shape[1] - 1
    ci, cj = np.meshgrid(np.arange(hc), np.arange(wc), indexing="ij")
    cell = np.stack([np.repeat(ci.ravel(), 2), np.repeat(cj.ravel(), 2)], axis=1)
    tri = np.tile([0, 1], hc * wc)
    return Facets(centers, normals, areas, cell, tri, (hc, wc))
