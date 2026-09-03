#!/usr/bin/env python3
"""Launch a Cloud Batch task array for one experiment spec.

Fan-out unit = the chunk cache in runs/: every ``simulate`` task runs
``run_basal_clutter.py --config SPEC --line L --simulate-only PASS`` (one pass
per task, or one chunk with ``--per-chunk``) on its own VM; a later
``process`` job (one task per line) copies those chunks in, hits
[skip-exists] on all of them and produces metrics.json + figures.

  uv run python tools/gcp/batch_launch.py --config config/experiments/pilot.yaml \
      --lines antarctica_pineisland_north --machine-type n2-highmem-8 --wait
  uv run python tools/gcp/batch_launch.py --config ... --mode process \
      --results-from soundersim-simulate-20260903-1200 --wait

Needs: the data bundle staged by tools/gcp/stage_bundle.py under --prefix,
`gcloud` authenticated on the project, and a committed tree (the repo
snapshot is `git archive HEAD`). Prints the job name; results land in
<prefix>/results/<job>/outputs/ (mirror of outputs/) -- fetch them with
tools/gcp/batch_sync.sh.
"""
import argparse
import datetime
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

DEFAULT_PREFIX = "gs://ice-infrastructure-soundersim/batch_2026-09-03"
REGION = "us-central1"


def gs(*args, capture=False, check=True):
    r = subprocess.run(["gcloud", *args], text=True, check=check,
                       capture_output=capture)
    return r.stdout if capture else None


def split_prefix(prefix):
    """gs://bucket/path -> (bucket, path)"""
    b, _, p = prefix.removeprefix("gs://").partition("/")
    return b, p.rstrip("/")


def pass_tasks(config, lines, per_chunk):
    """[(mode, line, pass[:chunks], outdir)] for the simulate job."""
    from clutter_spec import load_spec
    import run_basal_clutter as rbc
    spec = load_spec(config)
    tasks = []
    for line in lines:
        kw = spec.to_run_kwargs()
        kw["line"] = line
        order = rbc.run(**kw, list_passes=True)
        outdir = str(rbc.OUT_DEFAULT.relative_to(rbc.ROOT)
                     / (kw["out_name"] or kw["segment"]))
        if per_chunk:
            # chunk counts need the prepped pass: stage_bundle.py saves the
            # --dry-run manifest {pass: {n_chunks, cached, ...}}
            mp = ROOT / "outputs" / "gcp" / "chunks" / f"{line}.json"
            if not mp.exists():
                raise SystemExit(f"--per-chunk needs {mp} (run "
                                 "tools/gcp/stage_bundle.py first)")
            man = json.loads(mp.read_text())
            for key in order:
                for ci in range(man[key]["n_chunks"]):
                    if not man[key]["cached"][ci]:
                        tasks.append(("simulate", line, f"{key}:{ci}",
                                      outdir))
            continue
        for key in order:
            tasks.append(("simulate", line, key, outdir))
    return tasks


def process_tasks(config, lines):
    from clutter_spec import load_spec
    import run_basal_clutter as rbc
    spec = load_spec(config)
    tasks = []
    for line in lines:
        kw = spec.to_run_kwargs()
        kw["line"] = line
        rbc.run(**kw, list_passes=True)   # activates the line
        outdir = str(rbc.OUT_DEFAULT.relative_to(rbc.ROOT)
                     / (kw["out_name"] or kw["segment"]))
        tasks.append(("process", line, "-", outdir))
    return tasks


def job_spec(a, n_tasks, bucket, path, job):
    env = {"GCS_MOUNT": "/mnt/gcs", "PREFIX": path, "JOB": job,
           "CONFIG": a.config, "RESULTS_FROM": " ".join(a.results_from)}
    return {
        "taskGroups": [{
            "taskCount": n_tasks,
            "parallelism": min(n_tasks, a.max_vms),
            "taskSpec": {
                "runnables": [{"script": {
                    "text": "bash $GCS_MOUNT/$PREFIX/jobs/$JOB/batch_task.sh"}}],
                "environment": {"variables": env},
                "computeResource": {"cpuMilli": a.cpu_milli,
                                    "memoryMib": a.memory_mib},
                "maxRunDuration": f"{a.max_run_min * 60}s",
                "maxRetryCount": a.retries,
                "volumes": [{"gcs": {"remotePath": bucket},
                             "mountPath": "/mnt/gcs"}]}}],
        "allocationPolicy": {
            "instances": [{"policy": {
                "machineType": a.machine_type,
                "provisioningModel": a.provisioning,
                "bootDisk": {"sizeGb": 40, "type": "pd-balanced"}}}],
            # external IPs are capped by IN_USE_ADDRESSES (8/region here):
            # --no-external-ip needs Private Google Access + a Cloud NAT on
            # the subnet (uv/PyPI/GitHub) to lift the VM count to the quota
            **({"network": {"networkInterfaces": [{
                "network": "global/networks/default",
                "subnetwork": f"regions/{REGION}/subnetworks/default",
                "noExternalIpAddress": True}]}} if a.no_external_ip else {}),
            "location": {"allowedLocations": [f"regions/{REGION}"]}},
        "logsPolicy": {"destination": "CLOUD_LOGGING"}}


def wait(job, poll_s=30):
    t0 = time.time()
    while True:
        out = gs("batch", "jobs", "describe", job, f"--location={REGION}",
                 "--format=json", capture=True)
        d = json.loads(out)
        state = d["status"]["state"]
        counts = d["status"].get("taskGroups", {}).get("group0", {}).get(
            "counts", {})
        print(f"  {time.time() - t0:6.0f} s  {state}  {counts}", flush=True)
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return state
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--lines", nargs="+", required=True)
    ap.add_argument("--mode", choices=["simulate", "process"],
                    default="simulate")
    ap.add_argument("--per-chunk", action="store_true")
    ap.add_argument("--results-from", nargs="*", default=[],
                    help="process mode: simulate job names to pull chunks from")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--job", default=None, help="job name (default: dated)")
    ap.add_argument("--machine-type", default="n2-highmem-8")
    ap.add_argument("--provisioning", choices=["SPOT", "STANDARD"],
                    default="SPOT")
    ap.add_argument("--cpu-milli", type=int, default=8000)
    ap.add_argument("--memory-mib", type=int, default=60000,
                    help="per task; set near the VM's memory so Batch never "
                    "packs two chunk simulations onto one VM")
    ap.add_argument("--max-vms", type=int, default=24)
    ap.add_argument("--no-external-ip", action="store_true",
                    help="VMs without external IPs (needs Private Google "
                    "Access + Cloud NAT on the default subnet)")
    ap.add_argument("--max-run-min", type=int, default=120)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--dry", action="store_true",
                    help="print tasks + job json, submit nothing")
    a = ap.parse_args()

    tasks = (pass_tasks(a.config, a.lines, a.per_chunk) if a.mode == "simulate"
             else process_tasks(a.config, a.lines))
    if a.mode == "process" and not a.results_from:
        ap.error("--mode process needs --results-from <simulate job> ...")
    job = a.job or ("soundersim-" + a.mode + "-"
                    + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    bucket, path = split_prefix(a.prefix)
    spec = job_spec(a, len(tasks), bucket, path, job)
    for t in tasks:
        print("  ", *t)
    print(f"{len(tasks)} tasks, {a.machine_type} {a.provisioning}, "
          f"parallelism {spec['taskGroups'][0]['parallelism']}, job {job}")
    if a.dry:
        print(json.dumps(spec, indent=1))
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "tasks.txt").write_text(
            "".join(" ".join(t) + "\n" for t in tasks))
        (td / "job.json").write_text(json.dumps(spec, indent=1))
        subprocess.run(["git", "archive", "--format=tar.gz", "-o",
                        str(td / "repo.tar.gz"), "HEAD"], cwd=ROOT,
                       check=True)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              text=True, capture_output=True).stdout.strip()
        (td / "provenance.json").write_text(json.dumps(
            {"git_head": head, "args": vars(a), "tasks": tasks}, indent=1))
        dest = f"{a.prefix}/jobs/{job}/"
        gs("storage", "cp", "-q", str(td / "tasks.txt"), str(td / "job.json"),
           str(td / "repo.tar.gz"), str(td / "provenance.json"),
           str(ROOT / "tools/gcp/batch_task.sh"), dest)
        gs("batch", "jobs", "submit", job, f"--location={REGION}",
           f"--config={td / 'job.json'}")
    print(f"submitted {job}; results -> {a.prefix}/results/{job}/outputs/")
    if a.wait:
        print("final state:", wait(job))


if __name__ == "__main__":
    main()
