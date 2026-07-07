"""Geometry-layer tests for nav_to_frame (CPU, float64)."""

import numpy as np

from soundersim import synthetic as syn
from soundersim.scene import LocalFrame
from soundersim.nav import nav_to_frame


def test_unit_vectors_and_positions():
    scene = syn.flat_scene()
    frame = LocalFrame.centered_on(scene)
    nav = nav_to_frame(scene.nav_llh, frame)
    T = scene.params["n_traces"]
    assert nav.positions.shape == (T, 3)
    assert np.allclose(np.linalg.norm(nav.u_at, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(nav.u_ct, axis=1), 1.0)
    assert nav.positions.dtype == np.float64


def test_u_at_matches_track_direction():
    """The nav rides the projected easting axis; u_at aligns with the scene's
    chord direction and points mostly +E. The track is straight in projected
    space but curves gently in local ENU (Earth curvature tilts u_at by up to
    ~extent/2R ~ 1e-4 rad end-to-end), so alignment is asserted within 0.05 deg."""
    scene = syn.flat_scene()
    frame = LocalFrame.centered_on(scene)
    nav = nav_to_frame(scene.nav_llh, frame)
    track_dir = nav.positions[-1] - nav.positions[0]
    track_dir /= np.linalg.norm(track_dir)
    ang = np.degrees(np.arccos(np.clip(nav.u_at @ track_dir, -1.0, 1.0)))
    assert ang.max() < 0.05  # aligned to chord within curvature
    assert track_dir[0] > 0.99  # dominantly local east


def test_u_ct_right_of_travel_and_horizontal():
    """u_ct is horizontal, perpendicular to u_at, and to the RIGHT of travel:
    heading (near) +E, right is -N (south); (u_at x u_ct) points down."""
    scene = syn.flat_scene()
    frame = LocalFrame.centered_on(scene)
    nav = nav_to_frame(scene.nav_llh, frame)
    assert np.allclose(nav.u_ct[:, 2], 0.0, atol=1e-12)          # horizontal
    assert np.allclose(np.sum(nav.u_at * nav.u_ct, axis=1), 0.0, atol=1e-12)
    assert np.all(nav.u_ct[:, 1] < 0.0)                          # points south
    handed = np.cross(nav.u_at, nav.u_ct)
    assert np.all(handed[:, 2] < 0.0)                            # right-handed
