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


def group_chunks(indices, per_task):
    """Split uncached chunk indices into runs of ``per_task`` -> 'c0,c1,..'
    strings. One task then pays the pass preparation once for the group;
    the 2026-09-04 campaign paid it per chunk (10 min on getz, more than
    the chunk itself)."""
    per_task = max(1, int(per_task))
    return [",".join(str(c) for c in indices[i:i + per_task])
            for i in range(0, len(indices), per_task)]


def pass_tasks(config, lines, per_chunk, chunks_per_task=1):
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
                todo = [ci for ci in range(man[key]["n_chunks"])
                        if not man[key]["cached"][ci]]
                for grp in group_chunks(todo, chunks_per_task):
                    tasks.append(("simulate", line, f"{key}:{grp}", outdir))
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
                "bootDisk": {"sizeGb": a.boot_disk_gb,
                             "type": a.boot_disk_type}}}],
            # external IPs are capped by IN_USE_ADDRESSES (8/region here):
            # --no-external-ip needs Private Google Access + a Cloud NAT on
            # the subnet (uv/PyPI/GitHub) to lift the VM count to the quota
            **({"network": {"networkInterfaces": [{
                "network": "global/networks/default",
                "subnetwork": f"regions/{REGION}/subnetworks/default",
                "noExternalIpAddress": True}]}} if a.no_external_ip else {}),
            "location": {"allowedLocations": [f"regions/{REGION}"]}},
        "logsPolicy": {"destination": "CLOUD_LOGGING"}}


LEDGER = ROOT / "outputs" / "gcp" / "spend_ledger.json"


def ledger_total(path=LEDGER):
    """USD recorded for finished jobs (the campaign-wide tally)."""
    if not Path(path).exists():
        return 0.0
    return sum(e.get("usd", 0.0) for e in json.loads(Path(path).read_text()))


def ledger_add(entry, path=LEDGER):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(Path(path).read_text()) if Path(path).exists() else []
    doc.append(entry)
    Path(path).write_text(json.dumps(doc, indent=1))


def job_task_seconds(job, prefix):
    """Sum of per-task seconds recorded so far by ``job`` (rsyncs the
    results/<job>/timing records into outputs/gcp/<job>/timing)."""
    dest = ROOT / "outputs" / "gcp" / job / "timing"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcloud", "storage", "rsync", f"{prefix}/results/{job}/timing",
                    str(dest)], capture_output=True, check=False)
    secs = 0.0
    for f in dest.glob("task_*.json"):
        d = json.loads(f.read_text())
        secs += sum(d.get(k, 0) for k in ("env_s", "data_s", "run_s",
                                         "upload_s"))
    return secs


def running_vms(job):
    out = gs("compute", "instances", "list", "--format=value(name)",
             capture=True, check=False) or ""
    return sum(1 for n in out.split() if n.startswith(job[:20]))


def over_budget(spent_before, job_usd, budget):
    return budget is not None and spent_before + job_usd > budget


def wait(job, poll_s=30, rate=None, prefix=None, budget=None,
         spent_before=0.0, guard_every_s=600):
    """Poll to a terminal state; with ``rate`` also tally this job's spend
    from its task records (+ running VMs in flight) and DELETE the job when
    ``spent_before`` + this job would exceed ``budget``. Returns
    (state, job_vm_hours)."""
    t0 = time.time()
    last_guard, vmh = -1e9, 0.0
    while True:
        out = gs("batch", "jobs", "describe", job, f"--location={REGION}",
                 "--format=json", capture=True, check=False)
        try:
            d = json.loads(out)
            state = d["status"]["state"]
        except (TypeError, ValueError, KeyError):
            print("  describe failed (auth?), retrying", flush=True)
            time.sleep(poll_s)
            continue
        counts = d["status"].get("taskGroups", {}).get("group0", {}).get(
            "counts", {})
        line = f"  {time.time() - t0:6.0f} s  {state}  {counts}"
        if rate is not None and time.time() - last_guard >= guard_every_s:
            last_guard = time.time()
            secs = job_task_seconds(job, prefix)
            vmh = (secs + running_vms(job) * guard_every_s * 0.5) / 3600.0
            usd = vmh * rate
            line += f"  spend ~${usd:.2f} (+${spent_before:.2f} before)"
            if over_budget(spent_before, usd, budget):
                print(line, flush=True)
                print(f"BUDGET: ${spent_before + usd:.2f} > ${budget:.2f}: "
                      f"deleting {job}", flush=True)
                gs("batch", "jobs", "delete", job, f"--location={REGION}",
                   "--quiet", check=False)
                return "BUDGET_KILLED", vmh
        print(line, flush=True)
        if state in ("SUCCEEDED", "FAILED", "CANCELLED",
                     "DELETION_IN_PROGRESS"):
            if rate is not None:
                vmh = job_task_seconds(job, prefix) / 3600.0
            return state, vmh
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--lines", nargs="+", required=True)
    ap.add_argument("--mode", choices=["simulate", "process"],
                    default="simulate")
    ap.add_argument("--per-chunk", action="store_true")
    ap.add_argument("--chunks-per-task", type=int, default=6,
                    help="--per-chunk: chunks simulated per task (one pass "
                    "preparation per task)")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="campaign cap: refuse to submit if the ledger total "
                    "plus this job's projection exceeds it, and delete the "
                    "job while waiting if the tally does")
    ap.add_argument("--rate-usd-per-vm-h", type=float, default=None,
                    help="override the catalog VM price")
    ap.add_argument("--force-budget", action="store_true",
                    help="submit even if the projection exceeds --budget-usd")
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
    # pd-standard: SSD_TOTAL_GB (500/region) capped pd-balanced at 12 VMs
    ap.add_argument("--boot-disk-type", default="pd-standard")
    ap.add_argument("--boot-disk-gb", type=int, default=30)
    ap.add_argument("--no-external-ip", action="store_true",
                    help="VMs without external IPs (needs Private Google "
                    "Access + Cloud NAT on the default subnet)")
    ap.add_argument("--max-run-min", type=int, default=120)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--dry", action="store_true",
                    help="print tasks + job json, submit nothing")
    a = ap.parse_args()

    tasks = (pass_tasks(a.config, a.lines, a.per_chunk, a.chunks_per_task)
             if a.mode == "simulate" else process_tasks(a.config, a.lines))
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
    import pricing
    rate, src = ((a.rate_usd_per_vm_h, "override") if a.rate_usd_per_vm_h
                 else pricing.vm_hour_price(a.machine_type, a.provisioning))
    spent = ledger_total()
    proj_vmh, proj_usd = projection(a, tasks, rate)
    print(f"rate ${rate:.4f}/VM-h ({src}); projected {proj_vmh:.1f} VM-h "
          f"~${proj_usd:.2f}; ledger so far ${spent:.2f}"
          + (f"; budget ${a.budget_usd:.2f}" if a.budget_usd else ""),
          flush=True)
    if over_budget(spent, proj_usd, a.budget_usd) and not a.force_budget:
        raise SystemExit(f"projection ${spent + proj_usd:.2f} exceeds "
                         f"--budget-usd {a.budget_usd:.2f}; not submitting "
                         "(--force-budget overrides)")
    if a.dry:
        print(json.dumps(spec, indent=1))
        return
    if a.no_external_ip:
        # the NAT costs while it exists: bring it up here and ALWAYS tear it
        # down (finally) once this job is done -- so the launcher waits
        import nat
        nat.up()
        a.wait = True
    if a.budget_usd is not None:
        a.wait = True   # the guard lives in wait()
    try:
        submit(a, tasks, spec, job)
        if a.wait:
            state, vmh = wait(job, rate=rate, prefix=a.prefix,
                              budget=a.budget_usd, spent_before=spent)
            usd = vmh * rate
            ledger_add({"job": job, "date": datetime.date.today().isoformat(),
                        "machine_type": a.machine_type, "rate": rate,
                        "vm_hours": round(vmh, 2), "usd": round(usd, 2),
                        "state": state})
            print(f"final state: {state}; {vmh:.1f} VM-h ~${usd:.2f}; "
                  f"ledger total ${ledger_total():.2f}", flush=True)
    finally:
        if a.no_external_ip:
            nat.down()   # skipped with a warning while other jobs run


def projection(a, tasks, rate):
    """Projected (VM-hours, USD) for the simulate tasks from past records
    (pricing.estimate) -- process tasks are not projected (0)."""
    import pricing
    from clutter_spec import load_spec
    kw = load_spec(a.config).to_run_kwargs()
    exp = kw.get("out_name") or kw["segment"]
    counts, est = {}, {}
    for mode, line, pk, _ in tasks:
        if mode != "simulate":
            continue
        key, _, cs = pk.partition(":")
        n = len(cs.split(",")) if cs else a.chunks_per_task
        counts[(line, key)] = counts.get((line, key), 0) + 1
        if (line, key) not in est:
            est[(line, key)] = pricing.estimate(line, exp, [key], n)[key]
    vmh = usd = 0.0
    for (line, key), ntask in counts.items():
        v, u = pricing.project({key: ntask}, {key: est[(line, key)]},
                               a.chunks_per_task, rate)
        vmh += v
        usd += u
    return vmh, usd + (a.max_run_min / 60.0 * 0.044 if a.no_external_ip
                       else 0.0)


def submit(a, tasks, spec, job):
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


if __name__ == "__main__":
    main()
