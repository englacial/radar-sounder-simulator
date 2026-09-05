"""Pure-Python pieces of the Cloud Batch fan-out (tools/gcp) and the chunk
cache-hit rule they rely on."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "gcp"))

import run_altitude_comparison as rac  # noqa: E402
import batch_launch  # noqa: E402
import compare_runs  # noqa: E402
import nat  # noqa: E402


class _Run:
    """Fake gcloud: records calls, answers describe/list from a table."""
    def __init__(self, exists, jobs):
        self.exists, self.jobs, self.calls = exists, jobs, []

    def __call__(self, *args, check=True):
        self.calls.append(args)
        r = type("R", (), {"returncode": 0, "stdout": ""})()
        if "describe" in args:
            name = args[args.index("describe") + 1]
            r.returncode = 0 if name in self.exists else 1
            r.stdout = "True\n" if name == "default" else ""
        if args[:3] == ("batch", "jobs", "list"):
            r.stdout = json.dumps([{"name": f"projects/p/jobs/{n}",
                                    "status": {"state": s}}
                                   for n, s in self.jobs])
        return r


def test_nat_down_refuses_while_jobs_run(monkeypatch):
    fake = _Run(exists={"default", "soundersim-nat-router", "soundersim-nat"},
                jobs=[("soundersim-sim-x", "RUNNING"), ("psc-1", "RUNNING"),
                      ("soundersim-old", "SUCCEEDED")])
    monkeypatch.setattr(nat, "_run", fake)
    assert nat.active_jobs() == ["soundersim-sim-x"]
    assert nat.down() is False
    assert not any("delete" in c for c in fake.calls)
    assert nat.down(force=True) is True
    deleted = [c[c.index("delete") + 1] for c in fake.calls if "delete" in c]
    assert deleted == ["soundersim-nat", "soundersim-nat-router"]


def test_nat_up_is_idempotent(monkeypatch):
    fake = _Run(exists={"default", "soundersim-nat-router", "soundersim-nat"},
                jobs=[])
    monkeypatch.setattr(nat, "_run", fake)
    nat.up()
    assert not any(("create" in c) or ("update" in c) for c in fake.calls)
    fake2 = _Run(exists={"default"}, jobs=[])
    monkeypatch.setattr(nat, "_run", fake2)
    nat.up()
    assert [c[c.index("create") + 1] for c in fake2.calls if "create" in c] \
        == ["soundersim-nat-router", "soundersim-nat"]


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
        machine_type="n2-highmem-8", provisioning="SPOT",
        no_external_ip=True, boot_disk_type="pd-standard", boot_disk_gb=30)
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
    nic = s["allocationPolicy"]["network"]["networkInterfaces"][0]
    assert nic["noExternalIpAddress"] is True
    a.no_external_ip = False
    assert "network" not in batch_launch.job_spec(a, 6, "b", "p", "j")[
        "allocationPolicy"]


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
    rows, n, problems = compare_runs.cmp_metrics(
        pa, pb, ["midcol", "bed_visibility"])
    assert n == 3 and problems == []
    assert rows == [("clutter_x/sim/midcol_rel_surf_db", -30.0, -30.5, 0.5)]


# ------------------------------------------- chunk grouping + spend guard
import pricing  # noqa: E402


def test_group_chunks_packs_uncached_indices():
    assert batch_launch.group_chunks([0, 1, 2, 3, 4, 5, 6], 3) == \
        ["0,1,2", "3,4,5", "6"]
    assert batch_launch.group_chunks([4, 9], 1) == ["4", "9"]
    assert batch_launch.group_chunks([], 6) == []


def test_over_budget_and_ledger(tmp_path):
    led = tmp_path / "ledger.json"
    assert batch_launch.ledger_total(led) == 0.0
    batch_launch.ledger_add({"job": "a", "usd": 6.4}, led)
    batch_launch.ledger_add({"job": "b", "usd": 23.3}, led)
    assert batch_launch.ledger_total(led) == pytest.approx(29.7)
    assert not batch_launch.over_budget(29.7, 10.0, None)
    assert not batch_launch.over_budget(29.7, 10.0, 50.0)
    assert batch_launch.over_budget(29.7, 25.0, 50.0)


def test_pricing_machine_type_and_catalog_rate():
    assert pricing.parse_machine_type("n2-highmem-8") == ("N2", 8, 64.0)
    assert pricing.parse_machine_type("c2d-standard-16") == ("C2D", 16, 64.0)

    def sku(desc, usd, regions=("us-central1",)):
        units, nanos = divmod(round(usd * 1e9), 10 ** 9)
        return {"description": desc, "serviceRegions": list(regions),
                "pricingInfo": [{"pricingExpression": {"tieredRates": [
                    {"unitPrice": {"units": str(units), "nanos": nanos}}]}}]}
    skus = [sku("Spot Preemptible N2 Instance Core running in Americas",
                0.01896),
            sku("Spot Preemptible N2 Instance Ram running in Americas",
                0.002542),
            sku("Spot Preemptible N2 Instance Core running in Americas",
                0.5, regions=("europe-west1",))]
    rate = pricing.rate_from_skus(skus, "N2", 8, 64.0, "SPOT", "us-central1")
    assert rate == pytest.approx(8 * 0.01896 + 64 * 0.002542, rel=1e-6)
    assert pricing.rate_from_skus(skus, "N2", 8, 64.0, "STANDARD",
                                  "us-central1") is None


def test_projection_defaults_and_pass_classes():
    assert pricing.pass_class("dc8_2014_0km") == "heavy"
    assert pricing.pass_class("dc8_2012_9km") == "light"
    assert pricing.pass_class("haps_14km_lambda") == "light"
    est = pricing.estimate("no_such_line", "full",
                           ["dc8_2014_0km", "haps_14km_lambda"], 6)
    assert est["dc8_2014_0km"] == (pricing.DEFAULT_PREP_S,
                                   pricing.DEFAULT_CHUNK_S["heavy"])
    vmh, usd = pricing.project({"dc8_2014_0km": 10}, est, 6, 0.3144)
    secs = 10 * (pricing.DEFAULT_PREP_S + 6 * pricing.DEFAULT_CHUNK_S["heavy"])
    assert vmh == pytest.approx(secs / 3600 * (1 + pricing.OVERHEAD))
    assert usd == pytest.approx(vmh * 0.3144)


# -------------------------------------------------- PR #2 review findings
from types import SimpleNamespace  # noqa: E402


def test_budget_counts_elapsed_running_vm_time(monkeypatch):
    """Finding 2: a VM that runs for an hour is billed for an hour even
    when no task record has been written yet."""
    clock = SimpleNamespace(now=0.0)
    deleted = []

    def gs(*args, **kwargs):
        if args[:3] == ("batch", "jobs", "delete"):
            deleted.append(args)
            return ""
        state = "RUNNING" if clock.now < 3600 else "FAILED"
        return json.dumps({"status": {"state": state}})
    monkeypatch.setattr(batch_launch, "gs", gs)
    monkeypatch.setattr(batch_launch.time, "time", lambda: clock.now)
    monkeypatch.setattr(batch_launch.time, "sleep",
                        lambda s: setattr(clock, "now", clock.now + s))
    monkeypatch.setattr(batch_launch, "job_task_seconds", lambda *a: 0.0)
    monkeypatch.setattr(batch_launch, "running_vms", lambda *a: 1)
    state, vmh = batch_launch.wait("job", poll_s=600, rate=1.0,
                                   prefix="unused", budget=0.5,
                                   guard_every_s=600)
    assert state == "BUDGET_KILLED" and deleted
    assert vmh > 0.5


def test_failed_deletion_is_not_reported_as_a_kill(monkeypatch):
    """Finding 3: an unconfirmed delete returns an explicit failure state."""
    def gs(*args, **kwargs):
        if args[:3] == ("batch", "jobs", "describe"):
            return json.dumps({"status": {"state": "RUNNING"}})
        return None    # delete failed
    monkeypatch.setattr(batch_launch, "gs", gs)
    monkeypatch.setattr(batch_launch.time, "sleep", lambda s: None)
    monkeypatch.setattr(batch_launch, "job_task_seconds", lambda *a: 3600.0)
    monkeypatch.setattr(batch_launch, "running_vms", lambda *a: 1)
    state, _ = batch_launch.wait("job", rate=1.0, prefix="unused", budget=0.5)
    assert state == "BUDGET_KILL_FAILED"


def test_nat_preserved_when_job_listing_fails(monkeypatch):
    """Finding 4: an unknown job state never justifies deleting the NAT."""
    deleted = []

    def run(*args, **kwargs):
        if args[:3] == ("batch", "jobs", "list"):
            return SimpleNamespace(returncode=1, stdout="", stderr="API down")
        if "delete" in args:
            deleted.append(args)
        return SimpleNamespace(returncode=0, stdout="True\n", stderr="")
    monkeypatch.setattr(nat, "_run", run)
    assert nat.active_jobs() is None
    assert nat.down() is False
    assert not deleted


def test_ledger_reservation_is_replaced_by_the_measured_cost(tmp_path):
    led = tmp_path / "ledger.json"
    batch_launch.ledger_set("j1", {"usd": 20.0, "state": "RESERVED"}, led)
    batch_launch.ledger_set("j2", {"usd": 5.0, "state": "SUCCEEDED"}, led)
    assert batch_launch.ledger_total(led) == pytest.approx(25.0)
    batch_launch.ledger_set("j1", {"usd": 12.5, "state": "SUCCEEDED"}, led)
    assert batch_launch.ledger_total(led) == pytest.approx(17.5)
    assert [e["job"] for e in json.loads(led.read_text())] == ["j2", "j1"]


def test_projection_counts_each_tasks_chunks_and_process_tasks(monkeypatch):
    """Finding 7: a 2-chunk task is two chunks, a process task is not free."""
    a = argparse.Namespace(config=str(ROOT / "config/experiments/pilot.yaml"),
                           chunks_per_task=6, no_external_ip=False,
                           force_budget=False, max_run_min=120)
    two = [("simulate", "no_such_line", "dc8_2014_0km:3,4", "o")]
    six = [("simulate", "no_such_line", "dc8_2014_0km:0,1,2,3,4,5", "o")]
    v2, _ = batch_launch.projection(a, two, 1.0)
    v6, _ = batch_launch.projection(a, six, 1.0)
    prep, per = pricing.DEFAULT_PREP_S, pricing.DEFAULT_CHUNK_S["heavy"]
    assert v2 == pytest.approx((prep + 2 * per) / 3600 * (1 + pricing.OVERHEAD))
    assert v6 == pytest.approx((prep + 6 * per) / 3600 * (1 + pricing.OVERHEAD))
    vp, up = batch_launch.projection(
        a, [("process", "no_such_line", "-", "o")], 1.0)
    assert vp > 0 and up > 0
    with pytest.raises(SystemExit):   # pass-level task, no manifest
        batch_launch.projection(
            a, [("simulate", "no_such_line", "dc8_2014_0km", "o")], 1.0)


def test_locally_cached_chunk_is_skipped_only_if_its_files_exist(
        monkeypatch, tmp_path):
    """Finding 6: a manifest 'cached' flag without local files is a task;
    with files it is skipped and handed back for upload."""
    import run_altitude_comparison as rac
    import run_basal_clutter as rbc
    line, outdir = "antarctica_getz", "outputs/antarctica_getz/pilot"
    man = tmp_path / "outputs/gcp/chunks"
    man.mkdir(parents=True)
    (man / f"{line}.json").write_text(json.dumps({
        "pass": {"n_chunks": 2, "cached": [True, True],
                 "rids": ["absent", "present"]}}))
    (tmp_path / outdir / "runs").mkdir(parents=True)
    for ext in ("npz", "json"):
        (tmp_path / outdir / "runs" / f"present.{ext}").write_bytes(b"x")
    monkeypatch.setattr(batch_launch, "ROOT", tmp_path)
    monkeypatch.setattr(rbc, "run", lambda **kw: ["pass"])
    monkeypatch.setattr(rbc, "OUT_DEFAULT", rbc.ROOT / "outputs" / line)
    cached = []
    tasks = batch_launch.pass_tasks(
        str(ROOT / "config/experiments/pilot.yaml"), [line], per_chunk=True,
        cached_out=cached)
    assert [t[2] for t in tasks] == ["pass:0"]
    assert len(cached) == 1 and cached[0][0] == outdir
    assert sorted(f.name for f in cached[0][1]) == ["present.json",
                                                    "present.npz"]


def test_compare_metrics_reports_missing_and_nonfinite(tmp_path):
    ma = {"metrics": {"clutter_x": {"sim": {"midcol_rel_surf_db": -30.0}},
                      "clutter_y": {"sim": {"midcol_rel_surf_db": float("nan")}},
                      "only_cloud_bed_visibility": 1.0}}
    mb = {"metrics": {"clutter_x": {"sim": {"midcol_rel_surf_db": -30.0}},
                      "clutter_y": {"sim": {"midcol_rel_surf_db": -31.0}}}}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(ma)), pb.write_text(json.dumps(mb))
    rows, n, problems = compare_runs.cmp_metrics(
        pa, pb, ["midcol", "bed_visibility"])
    assert rows == [] and n == 2
    assert sorted(p[0] for p in problems) == [
        "clutter_y/sim/midcol_rel_surf_db", "only_cloud_bed_visibility"]
