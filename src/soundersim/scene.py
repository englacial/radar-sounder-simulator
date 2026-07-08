"""Scene geometry (CPU, NumPy, float64).

A ``LocalFrame`` is an ENU frame anchored on the WGS84 ellipsoid. ``build_facets``
turns a projected DEM window into one rectangular mean-plane facet per DEM cell
(centers/normals/areas plus the two edge vectors) expressed in that frame.
Curvature matters over km scales, so projected x/y are never treated as local
metres: every vertex goes DEM -> projected -> geodetic -> ECEF -> local ENU.
"""

import warnings
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
    """Rectangular mean-plane facets in a LocalFrame, one per DEM cell.

    Per cell with corners v00 (row i, col j), v01, v10, v11: the mean edge
    vectors ``e1`` (along +column) and ``e2`` (along +row) span a parallelogram
    whose area and normal are stored. Flat facet index of cell (i, j) is
    ``i * (nx-1) + j`` -- see ``index_map``.
    """

    centers: np.ndarray   # (N, 3) float64 centroids (mean of 4 corners) in local ENU
    normals: np.ndarray   # (N, 3) float64 unit normals, +z (up) oriented
    areas: np.ndarray     # (N,)   float64 mean-plane parallelogram areas (m^2)
    e1: np.ndarray        # (N, 3) float64 mean edge vector along +column (m)
    e2: np.ndarray        # (N, 3) float64 mean edge vector along +row (m)
    cell: np.ndarray      # (N, 2) int: (row i, col j) source grid cell
    grid_shape: tuple     # (ny-1, nx-1) number of cells

    def index_map(self):
        """(ny-1, nx-1) -> flat facet index."""
        n = self.grid_shape[0] * self.grid_shape[1]
        return np.arange(n).reshape(self.grid_shape)


def _bilinear(dem, rows, cols):
    """Sample ``dem`` on the (rows x cols) mesh of fractional pixel indices."""
    r0 = np.clip(np.floor(rows).astype(int), 0, dem.shape[0] - 2)
    c0 = np.clip(np.floor(cols).astype(int), 0, dem.shape[1] - 2)
    fr, fc = (rows - r0)[:, None], (cols - c0)[None, :]
    d00, d01 = dem[np.ix_(r0, c0)], dem[np.ix_(r0, c0 + 1)]
    d10, d11 = dem[np.ix_(r0 + 1, c0)], dem[np.ix_(r0 + 1, c0 + 1)]
    return (d00 * (1 - fr) * (1 - fc) + d01 * (1 - fr) * fc
            + d10 * fr * (1 - fc) + d11 * fr * fc)


def build_facets(dem, transform, crs, frame, spacing=None):
    """Tessellate a projected DEM window into upward-oriented rectangular facets.

    dem values are ellipsoidal heights (m). ``spacing`` (m) sets the target facet
    size: coarser than the DEM posting subsamples by stride; finer bilinearly
    upsamples the DEM (in the projected grid, before the ECEF pipeline) to
    roughly that size; ``None`` keeps the native posting.
    """
    dem = np.asarray(dem, dtype=np.float64)
    ny, nx = dem.shape
    px = 0.5 * (abs(transform.a) + abs(transform.e))
    if spacing is None or abs(spacing - px) < 1e-9:
        rows, cols = np.arange(ny, dtype=float), np.arange(nx, dtype=float)
    elif spacing > px:  # coarser: stride subsample
        step = max(1, int(round(spacing / px)))
        rows = np.arange(0, ny, step, dtype=float)
        cols = np.arange(0, nx, step, dtype=float)
    else:  # finer: bilinear subdivision toward the target facet size
        f = px / spacing
        rows = np.linspace(0, ny - 1, max(2, int(round((ny - 1) * f)) + 1))
        cols = np.linspace(0, nx - 1, max(2, int(round((nx - 1) * f)) + 1))
    dem = _bilinear(dem, rows, cols)

    # Pixel centers -> projected (E, N), carrying ellipsoidal height.
    J, I = np.meshgrid(cols + 0.5, rows + 0.5)
    E, N = transform * (J, I)
    ecef = np.column_stack(
        Transformer.from_crs(crs, _ECEF, always_xy=True).transform(
            E.ravel(), N.ravel(), dem.ravel()))
    V = frame.ecef_to_local(ecef).reshape(*dem.shape, 3)  # (H, W, 3)

    v00, v01 = V[:-1, :-1], V[:-1, 1:]
    v10, v11 = V[1:, :-1], V[1:, 1:]
    e1 = ((v01 + v11) - (v00 + v10)) / 2.0  # mean edge along +column
    e2 = ((v10 + v11) - (v00 + v01)) / 2.0  # mean edge along +row
    centers = ((v00 + v01 + v10 + v11) / 4.0).reshape(-1, 3)
    raw_n = np.cross(e1, e2).reshape(-1, 3)  # |e1 x e2| = planar-quad area
    mag = np.linalg.norm(raw_n, axis=1)
    areas = mag
    normals = raw_n / mag[:, None]
    normals *= np.where(normals[:, 2] < 0.0, -1.0, 1.0)[:, None]  # orient +up

    hc, wc = dem.shape[0] - 1, dem.shape[1] - 1
    ci, cj = np.meshgrid(np.arange(hc), np.arange(wc), indexing="ij")
    cell = np.stack([ci.ravel(), cj.ravel()], axis=1)
    return Facets(centers, normals, areas, e1.reshape(-1, 3), e2.reshape(-1, 3),
                  cell, (hc, wc))


def check_facet_size(facets, wavelength, min_range, beta=0.5):
    """Fresnel-zone validity check for the linear-phase approximation.

    Warns when the largest facet edge exceeds ``beta * sqrt(wavelength * min_range)``
    (half a Fresnel-zone radius by default); returns the max edge/limit ratio.
    Not wired into the incoherent path -- it is for the coherent kernel.
    """
    edge = np.maximum(np.linalg.norm(facets.e1, axis=1),
                      np.linalg.norm(facets.e2, axis=1))
    limit = beta * np.sqrt(wavelength * min_range)
    ratio = float(edge.max() / limit)
    if ratio > 1.0:
        warnings.warn(
            f"max facet edge {edge.max():.1f} m exceeds Fresnel-zone limit "
            f"{limit:.1f} m (ratio {ratio:.2f}); linear-phase approximation "
            "may be invalid -- subdivide the scene")
    return ratio
