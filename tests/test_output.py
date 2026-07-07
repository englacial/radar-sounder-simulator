"""Output Dataset structure per docs/output.md, save round-trip, combine."""

import json

import numpy as np
import pytest
import xarray as xr

import soundersim
from soundersim.config import FacetConfig, RadarConfig, SimConfig
from soundersim.synthetic import flat_scene

RC = RadarConfig(dt=1e-8, n_samples=1250, t0=0.0)


@pytest.fixture(scope="module")
def ds():
    scene = flat_scene(extent=2000.0, n_traces=4, altitude=1000.0)
    cfg = SimConfig(mode="incoherent", split_sides=True, radar=RC,
                    facets=FacetConfig())
    return soundersim.simulate(scene, cfg)


def test_dataset_structure(ds):
    assert ds.power.dims == ("slow_time", "twtt", "side")
    assert ds.power.dtype == np.float32
    assert list(ds.side.values) == ["left", "right"]
    assert ds.sizes == {"slow_time": 4, "twtt": RC.n_samples, "side": 2}
    np.testing.assert_allclose(np.diff(ds.twtt), RC.dt)
    assert ds.twtt.values[0] == RC.t0
    for coord in ("trace", "lat", "lon", "elevation", "x", "y", "z"):
        assert ds[coord].dims == ("slow_time",)
    for var in ("nadir_twtt", "first_return_twtt", "first_return_lat",
                "first_return_lon", "dropped_power"):
        assert ds[var].dims == ("slow_time",)
    assert ds.attrs["mode"] == "incoherent"
    cfg = json.loads(ds.attrs["config"])  # full pydantic config round-trips
    assert cfg["radar"]["n_samples"] == RC.n_samples
    assert ds.attrs["soundersim_version"] == soundersim.__version__
    assert "created" in ds.attrs and "scene_frame" in ds.attrs


def test_physics_sanity(ds):
    # flat scene at 1000 m altitude: nadir and first return at 2h/c
    expected = 2 * 1000.0 / RC.c
    np.testing.assert_allclose(ds.nadir_twtt, expected, rtol=1e-3)
    np.testing.assert_allclose(ds.first_return_twtt, expected, rtol=1e-3)
    assert (ds.dropped_power >= 0).all()


def test_combine_rule(ds):
    combined = soundersim.combine(ds, "side")
    np.testing.assert_allclose(combined, ds.power.sum("side"))


def test_save_roundtrip(ds, tmp_path):
    path = soundersim.save(ds, tmp_path / "out.nc")
    back = xr.open_dataset(path)
    np.testing.assert_array_equal(back.power, ds.power)
    assert back.attrs["mode"] == "incoherent"
    back.close()
