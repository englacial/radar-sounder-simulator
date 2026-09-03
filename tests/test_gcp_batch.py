"""Pure-Python pieces of the Cloud Batch fan-out (tools/gcp) and the chunk
cache-hit rule they rely on."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "gcp"))

import run_altitude_comparison as rac  # noqa: E402
import batch_launch  # noqa: E402
import compare_runs  # noqa: E402


def _write_chunk(runs, rid, meta, field):
    (runs / f"{rid}.json").write_text(json.dumps(
        {"rid": rid, "wall_s": 1.0, "meta": meta,
         "meta_key": json.dumps(meta, sort_keys=True)}))
    np.savez(runs / f"{rid}.npz", field=field, twtt=np.arange(3.0),
             nadir_twtt=np.zeros((2, 2)))


def test_chunk_cached_is_pure_meta_equality(tmp_path):
    meta = {"pass": "p", "chunk": 0, "kernel": "k"}
    _write_chunk(tmp_path, "rid", meta, np.zeros((2, 3, 2), np.complex64))
    # key order irrelevant, values decisive, both files required
    assert rac.chunk_cached("rid", dict(reversed(list(meta.items()))),
                            tmp_path)["rid"] == "rid"
    assert rac.chunk_cached("rid", {**meta, "kernel": "other"},
                            tmp_path) is None
    (tmp_path / "rid.npz").unlink()
    assert rac.chunk_cached("rid", meta, tmp_path) is None
    assert rac.chunk_cached("absent", meta, tmp_path) is None


def test_split_prefix():
    assert batch_launch.split_prefix("gs://b/p/q/") == ("b", "p/q")
    assert batch_launch.split_prefix("gs://b") == ("b", "")


def test_job_spec_one_task_per_vm_and_spot():
    a = batch_launch.argparse.Namespace(
        config="c.yaml", results_from=["j0"], max_vms=4, cpu_milli=8000,
        memory_mib=56000, max_run_min=90, retries=2,
        machine_type="n2-highmem-8", provisioning="SPOT")
    s = batch_launch.job_spec(a, 6, "bucket", "prefix", "job")
    tg = s["taskGroups"][0]
    assert tg["taskCount"] == 6 and tg["parallelism"] == 4
    ts = tg["taskSpec"]
    assert ts["maxRunDuration"] == "5400s"
    assert ts["computeResource"]["memoryMib"] == 56000
    assert ts["volumes"][0]["gcs"]["remotePath"] == "bucket"
    env = ts["environment"]["variables"]
    assert env["PREFIX"] == "prefix" and env["JOB"] == "job"
    assert env["RESULTS_FROM"] == "j0"
    pol = s["allocationPolicy"]["instances"][0]["policy"]
    assert pol["provisioningModel"] == "SPOT"
    assert pol["machineType"] == "n2-highmem-8"


def test_compare_chunks_reports_meta_and_array_diffs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    meta = {"pass": "p", "chunk": 0}
    f = np.ones((2, 3, 2), np.complex64)
    _write_chunk(a, "r0", meta, f)
    _write_chunk(b, "r0", meta, f * (1 + 1e-6))
    _write_chunk(a, "r1", meta, f)
    _write_chunk(b, "r1", {**meta, "chunk": 1}, f)
    rows = {r[0]: r for r in compare_runs.cmp_chunks(a, b)}
    assert rows["r0"][1] == "meta=="
    d, rel = rows["r0"][2]["field"]
    assert 0 < rel < 2e-6 and rows["r0"][2]["twtt"] == (0.0, 0.0)
    assert rows["r1"][1] == "META DIFFERS"


def test_compare_metrics_flattens_and_filters(tmp_path):
    ma = {"metrics": {"clutter_x": {"sim": {"midcol_rel_surf_db": -30.0}},
                      "haps_bed_visibility": 3.0, "wall_s": 5.0}}
    mb = {"metrics": {"clutter_x": {"sim": {"midcol_rel_surf_db": -30.5}},
                      "haps_bed_visibility": 3.0, "wall_s": 9.0}}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(ma)), pb.write_text(json.dumps(mb))
    rows, n = compare_runs.cmp_metrics(pa, pb, ["midcol", "bed_visibility"])
    assert n == 3
    assert rows == [("clutter_x/sim/midcol_rel_surf_db", -30.0, -30.5, 0.5)]
