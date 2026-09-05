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
    """gcloud wrapper. With ``capture`` returns stdout, or None when the
    command failed (so a failed delete/describe is never mistaken for an
    empty success)."""
    r = subprocess.run(["gcloud", *args], text=True, check=check,
                       capture_output=capture)
    if not capture:
        return None
    return r.stdout if r.returncode == 0 else None


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


def manifest_path(line, config):
    """The staged chunk manifest, bound to the experiment when available."""
    d = ROOT / "outputs" / "gcp" / "chunks"
    exp = Path(config).stem
    for name in (f"{line}__{exp}.json", f"{line}.json"):
        if (d / name).exists():
            return d / name
    return d / f"{line}__{exp}.json"


def local_chunk_files(outdir, rid):
    """(npz, json) paths of a locally cached chunk, or None if incomplete."""
    fs = [ROOT / outdir / "runs" / f"{rid}.{ext}" for ext in ("npz", "json")]
    return fs if all(f.exists() for f in fs) else None


def pass_tasks(config, lines, per_chunk, chunks_per_task=1, cached_out=None):
    """[(mode, line, pass[:chunks], outdir)] for the simulate job. A chunk
    the manifest marks cached is skipped ONLY if its files exist locally;
    those files are listed in ``cached_out`` (if given) so the launcher can
    upload them where the workers and the process job will look."""
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
            mp = manifest_path(line, config)
            if not mp.exists():
                raise SystemExit(f"--per-chunk needs {mp} (run "
                                 "tools/gcp/stage_bundle.py first)")
            man = json.loads(mp.read_text())
            for key in order:
                todo = []
                for ci in range(man[key]["n_chunks"]):
                    files = (local_chunk_files(outdir, man[key]["rids"][ci])
                             if man[key]["cached"][ci] else None)
                    if files is None:
                        todo.append(ci)
                    elif cached_out is not None:
                        cached_out.append((outdir, files))
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


def ledger_set(job, entry, path=LEDGER):
    """Replace (or add) the ledger entry for ``job``: a RESERVED projection
    at submit becomes the measured cost at the end, so concurrent launchers
    see each other's commitments in ledger_total()."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(Path(path).read_text()) if Path(path).exists() else []
    doc = [e for e in doc if e.get("job") != job] + [dict(entry, job=job)]
    Path(path).write_text(json.dumps(doc, indent=1))


def delete_job(job, attempts=3, pause_s=20):
    """Delete ``job`` and confirm it: True only when the delete command
    succeeded or a describe shows it gone / being deleted."""
    for i in range(attempts):
        out = gs("batch", "jobs", "delete", job, f"--location={REGION}",
                 "--quiet", capture=True, check=False)
        if out is not None:
            return True
        d = gs("batch", "jobs", "describe", job, f"--location={REGION}",
               "--format=json", capture=True, check=False)
        if d is None:
            return True     # not found any more
        try:
            if json.loads(d)["status"]["state"] in ("DELETION_IN_PROGRESS",
                                                   "CANCELLED", "FAILED",
                                                   "SUCCEEDED"):
                return True
        except (ValueError, KeyError):
            pass
        if i < attempts - 1:
            time.sleep(pause_s)
    return False


def wait(job, poll_s=30, rate=None, prefix=None, budget=None,
         spent_before=0.0, guard_every_s=600):
    """Poll to a terminal state; with ``rate`` also tally this job's spend
    from its task records (+ running VMs in flight) and DELETE the job when
    ``spent_before`` + this job would exceed ``budget``. Returns
    (state, job_vm_hours)."""
    t0 = time.time()
    last_guard, vmh = None, 0.0
    vm_secs_seen = 0.0    # running VMs x elapsed, accumulated every guard tick
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
        now = time.time()
        if rate is not None and (last_guard is None
                                 or now - last_guard >= guard_every_s):
            # VM lifetime is what is billed: accumulate running VMs x elapsed
            # (covers failed / preempted attempts that never write a timing
            # record) and take the larger of that and the task records
            nvm = running_vms(job)
            vm_secs_seen += nvm * (now - (last_guard if last_guard is not None
                                          else t0))
            last_guard = now
            secs = max(job_task_seconds(job, prefix), vm_secs_seen)
            vmh = secs / 3600.0
            usd = vmh * rate
            line += (f"  spend ~${usd:.2f} (+${spent_before:.2f} before; "
                     f"{nvm} VMs)")
            if over_budget(spent_before, usd, budget):
                print(line, flush=True)
                print(f"BUDGET: ${spent_before + usd:.2f} > ${budget:.2f}: "
                      f"deleting {job}", flush=True)
                if delete_job(job):
                    return "BUDGET_KILLED", vmh
                print(f"BUDGET: could not confirm deletion of {job}; "
                      "delete it by hand (gcloud batch jobs delete)",
                      flush=True)
                return "BUDGET_KILL_FAILED", vmh
        print(line, flush=True)
        if state in ("SUCCEEDED", "FAILED", "CANCELLED",
                     "DELETION_IN_PROGRESS"):
            if rate is not None:
                vm_secs_seen += running_vms(job) * (time.time() - (
                    last_guard if last_guard is not None else t0))
                vmh = max(job_task_seconds(job, prefix), vm_secs_seen) / 3600.0
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

    cached = []
    tasks = (pass_tasks(a.config, a.lines, a.per_chunk, a.chunks_per_task,
                        cached_out=cached)
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
    if not tasks:
        print(f"nothing to do: every chunk is cached locally "
              f"({len(cached)} files); no job submitted", flush=True)
        return
    if cached:
        upload_cached(cached, a.prefix, job)
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
        # reserve the projection so a concurrent launcher counts it
        ledger_set(job, {"date": datetime.date.today().isoformat(),
                         "machine_type": a.machine_type, "rate": rate,
                         "vm_hours": round(proj_vmh, 2),
                         "usd": round(proj_usd, 2), "state": "RESERVED"})
        if a.wait:
            state, vmh = wait(job, rate=rate, prefix=a.prefix,
                              budget=a.budget_usd, spent_before=spent)
            usd = vmh * rate
            ledger_set(job, {"date": datetime.date.today().isoformat(),
                             "machine_type": a.machine_type, "rate": rate,
                             "vm_hours": round(vmh, 2), "usd": round(usd, 2),
                             "state": state})
            print(f"final state: {state}; {vmh:.1f} VM-h ~${usd:.2f}; "
                  f"ledger total ${ledger_total():.2f}", flush=True)
    finally:
        if a.no_external_ip:
            nat.down()   # skipped with a warning while other jobs run


def task_chunk_count(line, key, cs, config):
    """Chunks a task simulates: the listed indices, else the whole pass
    from the staged manifest, else None (unknown)."""
    if cs:
        return len(cs.split(","))
    mp = manifest_path(line, config)
    if mp.exists():
        man = json.loads(mp.read_text())
        if key in man:
            return man[key]["n_chunks"] - sum(man[key]["cached"])
    return None


def projection(a, tasks, rate):
    """Projected (VM-hours, USD): each simulate task costs its pass
    preparation plus ITS chunk count x the per-chunk estimate; each process
    task costs the line's past process-task time (else a labelled 1 h
    default). Raises when a pass-level task's chunk count is unknown and
    --force-budget is not set."""
    import pricing
    from clutter_spec import load_spec
    kw = load_spec(a.config).to_run_kwargs()
    exp = kw.get("out_name") or kw["segment"]
    secs, notes = 0.0, []
    for mode, line, pk, _ in tasks:
        if mode == "process":
            past = [r.get("run_s", 0) for r in pricing.timing_records(
                line, mode="process")]
            t = (sorted(past)[len(past) // 2] if past
                 else pricing.DEFAULT_PROCESS_S)
            if not past:
                notes.append(f"process {line}: no records, default "
                             f"{pricing.DEFAULT_PROCESS_S / 3600:.1f} h")
            secs += t
            continue
        key, _, cs = pk.partition(":")
        n = task_chunk_count(line, key, cs, a.config)
        if n is None:
            if not getattr(a, "force_budget", False):
                raise SystemExit(f"cannot project {line} {key}: no staged "
                                 "manifest for a pass-level task (stage "
                                 "first, or --force-budget)")
            n, _ = a.chunks_per_task, notes.append(
                f"{line} {key}: chunk count unknown, assumed "
                f"{a.chunks_per_task}")
        prep, per = pricing.estimate(line, exp, [key], n)[key]
        secs += prep + n * per
    for m in notes:
        print(f"  projection note: {m}", flush=True)
    vmh = secs / 3600.0 * (1.0 + pricing.OVERHEAD)
    nat = (a.max_run_min / 60.0 * 0.044 if getattr(a, "no_external_ip", False)
           else 0.0)
    return vmh, vmh * rate + nat


def upload_cached(cached, prefix, job):
    """Put locally cached chunks where the workers and the process job copy
    chunks from (results/<job>/<outdir>/runs/), so skipping them in the
    task plan does not make the process job re-simulate them."""
    by_dir = {}
    for outdir, files in cached:
        by_dir.setdefault(outdir, []).extend(str(f) for f in files)
    for outdir, paths in by_dir.items():
        gs("storage", "cp", "-n", "-q", *paths,
           f"{prefix}/results/{job}/{outdir}/runs/")
    print(f"uploaded {sum(len(v) for v in by_dir.values())} locally cached "
          f"chunk files for the workers", flush=True)


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
