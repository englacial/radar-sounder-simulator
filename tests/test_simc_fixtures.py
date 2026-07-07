"""Sanity checks on the cached simc fixtures (no simc import; CI-fast)."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from soundersim.synthetic import ALL_SCENES

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SCENE_NAMES = [f.__name__.removesuffix("_scene") for f in ALL_SCENES]


@pytest.fixture(params=SCENE_NAMES)
def fixture_pair(request):
    name = request.param
    data = np.load(FIXTURE_DIR / f"{name}.npz")
    meta = json.loads((FIXTURE_DIR / f"{name}.json").read_text())
    return name, data, meta


def test_fixture_loads_and_is_sane(fixture_pair):
    name, data, meta = fixture_pair
    rc = meta["radar_config"]
    n_traces = meta["scene"]["params"]["n_traces"]
    cg = data["cluttergram"]
    assert cg.shape == (rc["n_samples"], n_traces)
    for key in ("cluttergram", "left", "right"):
        arr = data[key]
        assert np.isfinite(arr).all()
        assert (arr >= 0).all()
        assert arr.sum() > 0
    assert np.allclose(data["left"] + data["right"], cg, rtol=1e-6)
    assert data["nav_ecef"].shape == (n_traces, 3)
    assert data["fret_bin"].shape == (n_traces,)
    # No wrap: nothing lands in the last bin and first returns are in-window
    assert cg[-1].sum() == 0
    assert (data["fret_bin"] >= 0).all() and (data["fret_bin"] < rc["n_samples"]).all()


def test_json_sidecar(fixture_pair):
    name, _, meta = fixture_pair
    assert re.fullmatch(r"[0-9a-f]{40}", meta["simc_sha"])
    assert meta["scene"]["name"] == name
    assert meta["confDict"]["simParams"]["tracesamples"] == meta["radar_config"]["n_samples"]
    assert meta["confDict"]["simParams"]["dt"] == meta["radar_config"]["dt"]
    assert meta["radar_config"]["t0"] == 0.0


def test_flat_leading_edge():
    data = np.load(FIXTURE_DIR / "flat.npz")
    meta = json.loads((FIXTURE_DIR / "flat.json").read_text())
    rc = meta["radar_config"]
    h = meta["scene"]["params"]["altitude"]  # platform height above the flat surface
    expected = int(np.floor(2 * h / rc["c"] / rc["dt"]))
    first_nonzero = np.array([np.nonzero(col)[0][0] for col in data["cluttergram"].T])
    assert np.all(np.abs(first_nonzero - expected) <= 1)
    assert np.all(np.abs(data["fret_bin"] - expected) <= 1)
