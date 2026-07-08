"""Integrity-only test for the firn plateau investigation (tools/run_firn_investigation).

Exploratory experiment (not a physics gate): this only checks the machinery --
a tiny sweep (N=10 equal + one N=10 random seed) produces finite depth-power
profiles, writes diagnostics json + npz per run, and the HTML report builds.
The full sweep is run out-of-band by the tool.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "run_firn_investigation", ROOT / "tools" / "run_firn_investigation.py")
rfi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rfi)


@pytest.mark.integration
def test_firn_investigation_integrity(tmp_path, monkeypatch):
    monkeypatch.setattr(rfi, "OUTDIR", tmp_path)
    skips = rfi.run_sweep(layer_counts=(10,), n_seeds=1)
    assert skips == []

    runs_dir = tmp_path / "runs"
    for rid in ("reference", "equal_N10", "random_N10_s0"):
        jpath = runs_dir / f"{rid}.json"
        npath = runs_dir / f"{rid}.npz"
        assert jpath.exists(), f"missing diagnostics for {rid}"
        assert npath.exists(), f"missing profile npz for {rid}"
        arr = np.load(npath)
        assert arr["depth"].size > 0
        assert np.isfinite(arr["depth"]).all()
        assert np.isfinite(arr["prof_db"]).all()

    # layered runs carry the plateau/secondary-max diagnostics
    import json
    diag = json.loads((runs_dir / "equal_N10.json").read_text())
    assert diag["grad_interval_primary"]["length_m"] >= 0.0
    assert np.isfinite(diag["realized_gamma_db"]["median"])

    report = rfi.build_report()
    assert report.exists()
    assert report.stat().st_size > 1000
