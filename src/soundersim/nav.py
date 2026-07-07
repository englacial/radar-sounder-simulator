"""Platform navigation in a scene's LocalFrame (CPU, NumPy, float64)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class NavTrack:
    """Per-trace platform geometry in a LocalFrame."""

    positions: np.ndarray  # (T, 3) local ENU
    u_at: np.ndarray       # (T, 3) unit along-track (direction of travel)
    u_ct: np.ndarray       # (T, 3) unit cross-track, RIGHT of travel, horizontal


def _unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def nav_to_frame(nav_llh, frame):
    """Convert (T, 3) lat/lon/height nav into a NavTrack in ``frame``."""
    pos = frame.llh_to_local(np.asarray(nav_llh, dtype=np.float64))
    # Along-track from central differences (one-sided at the endpoints).
    u_at = _unit(np.gradient(pos, axis=0))
    # Cross-track right of travel: u_at x up, which is (a_y, -a_x, 0), horizontal
    # and perpendicular to u_at. Right of +E travel is -N (south), as required.
    z = np.array([0.0, 0.0, 1.0])
    u_ct = _unit(np.cross(u_at, z))
    return NavTrack(pos, u_at, u_ct)
