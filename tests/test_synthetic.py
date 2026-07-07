import numpy as np
import pytest
import rasterio

from soundersim import synthetic as syn


def test_geotiff_round_trip(tmp_path):
    scene = syn.flat_scene()
    path = tmp_path / "dem.tif"
    syn.write_dem_geotiff(scene, path)
    with rasterio.open(path) as src:
        arr = src.read(1)
        assert np.array_equal(arr, scene.dem)
        assert src.transform == scene.transform
        assert src.crs.to_string() == scene.crs
        assert src.nodata == -9999.0


def test_flat_constant():
    scene = syn.flat_scene(elevation=500.0)
    assert np.allclose(scene.dem, 500.0)


def test_tilted_slope_recovery():
    slope = 4.0
    scene = syn.tilted_scene(slope_deg=slope)
    # Cross-track is northing (rows). Fit slope of DEM vs. northing.
    ny, nx = scene.dem.shape
    rows = np.arange(ny)
    n_per_row = scene.transform.e  # negative posting (dN per row)
    fit = np.polyfit(rows * n_per_row, scene.dem.mean(axis=1), 1)[0]
    assert np.isclose(fit, np.tan(np.deg2rad(slope)), rtol=1e-3)


def test_hill_peak():
    height, sigma, offset, elev = 200.0, 800.0, 1500.0, 500.0
    scene = syn.hill_scene(height=height, sigma=sigma, offset=offset,
                           elevation=elev)
    assert np.isclose(scene.dem.max(), elev + height, rtol=1e-3)
    # Peak location: expected at (cx, cy + offset) in projected coords.
    i, j = np.unravel_index(np.argmax(scene.dem), scene.dem.shape)
    x, y = scene.transform * (j + 0.5, i + 0.5)
    cx, cy = syn._to_proj.transform(-40.0, 75.0)
    assert abs(x - cx) <= scene.params["posting"]
    assert abs(y - (cy + offset)) <= scene.params["posting"]


def test_crater_minimum():
    scene = syn.crater_scene(depth=200.0, elevation=500.0)
    assert scene.dem.min() < 500.0
    # Minimum well below the scene edges (surrounding surface).
    edge = np.concatenate([scene.dem[0], scene.dem[-1],
                           scene.dem[:, 0], scene.dem[:, -1]])
    assert scene.dem.min() < edge.mean() - 100.0


def test_nav_ecef_magnitude_and_spacing():
    scene = syn.flat_scene()
    ecef = syn.nav_ecef(scene)
    r = np.linalg.norm(ecef, axis=1)
    assert np.all(r > 6.3e6) and np.all(r < 6.4e6 + 1e4)
    d = np.linalg.norm(np.diff(ecef, axis=0), axis=1)
    assert np.allclose(d, scene.params["spacing"], rtol=0.01)


@pytest.mark.parametrize("factory", syn.ALL_SCENES)
def test_scene_shapes(factory):
    scene = factory()
    ny, nx = scene.dem.shape
    assert scene.dem.dtype == np.float32
    assert scene.nav_llh.shape == (scene.params["n_traces"], 3)
    assert ny == nx == int(scene.params["extent"] / scene.params["posting"])
