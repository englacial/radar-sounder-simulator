#!/usr/bin/env python3
"""VM prices from the Cloud Billing catalog and task-time estimates from
past runs, for projecting and guarding Batch spend.

  uv run python tools/gcp/pricing.py n2-highmem-8 [--provisioning SPOT]

Rates are cached in outputs/gcp/pricing.json for 7 days. When the catalog
is unreachable the 2026-09-04 us-central1 values below are used and
flagged. Estimation lessons: claude_notes/gcp_cost_estimation_notes_2026-09-04.md
"""
import argparse
import datetime
import glob
import json
import re
import statistics
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "gcp" / "pricing.json"
COMPUTE_SERVICE = "6F81-5844-456A"
# $/VM-hour observed in the catalog on 2026-09-04 (us-central1)
FALLBACK_RATES = {("n2-highmem-8", "SPOT"): 0.3144,
                  ("n2-highmem-8", "STANDARD"): 0.5241}
RAM_PER_CORE_GIB = {"highmem": 8.0, "standard": 4.0, "highcpu": 2.0,
                    "megamem": 14.0, "ultramem": 28.0}
# per-task seconds when no records exist: pass preparation (frames, DEMs,
# picks, bed synthesis -- paid once per task) + one chunk of simulation
DEFAULT_PREP_S = 300.0
DEFAULT_CHUNK_S = {"heavy": 300.0, "light": 120.0}
OVERHEAD = 0.10        # VM boot / idle / tail not covered by task records


def parse_machine_type(mt):
    """'n2-highmem-8' -> (family 'N2', cores 8, ram GiB 64)."""
    m = re.fullmatch(r"([a-z]\d[a-z]?)-(\w+?)-(\d+)", mt)
    if not m:
        raise ValueError(f"cannot parse machine type {mt!r}")
    fam, kind, cores = m.group(1).upper(), m.group(2), int(m.group(3))
    return fam, cores, cores * RAM_PER_CORE_GIB[kind]


def rate_from_skus(skus, family, cores, ram_gib, provisioning, region):
    """$/VM-hour from catalog SKUs (core + RAM SKUs of the family)."""
    prefix = "Spot Preemptible " if provisioning == "SPOT" else ""
    want = {f"{prefix}{family} Instance Core running in Americas": cores,
            f"{prefix}{family} Instance Ram running in Americas": ram_gib}
    total, found = 0.0, 0
    for s in skus:
        d = s.get("description")
        if d in want and region in s.get("serviceRegions", []):
            r = s["pricingInfo"][0]["pricingExpression"]["tieredRates"][-1][
                "unitPrice"]
            total += (float(r.get("units", 0) or 0) + r.get("nanos", 0) / 1e9
                      ) * want[d]
            found += 1
    return total if found == 2 else None


def fetch_skus():
    tok = subprocess.run(["gcloud", "auth", "print-access-token"], text=True,
                         capture_output=True, check=True).stdout.strip()
    skus, token = [], ""
    for _ in range(12):
        url = (f"https://cloudbilling.googleapis.com/v1/services/"
               f"{COMPUTE_SERVICE}/skus?pageSize=5000"
               + (f"&pageToken={token}" if token else ""))
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
        skus += d.get("skus", [])
        token = d.get("nextPageToken", "")
        if not token:
            break
    return skus


def vm_hour_price(machine_type, provisioning="SPOT", region="us-central1",
                  max_age_days=7):
    """($/VM-hour, source): catalog (cached) or the dated fallback."""
    key = f"{machine_type}|{provisioning}|{region}"
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    ent = cache.get(key)
    if ent and (datetime.date.today()
                - datetime.date.fromisoformat(ent["date"])).days <= max_age_days:
        return ent["usd_per_h"], f"catalog cached {ent['date']}"
    try:
        fam, cores, ram = parse_machine_type(machine_type)
        rate = rate_from_skus(fetch_skus(), fam, cores, ram, provisioning,
                              region)
    except Exception as e:  # noqa: BLE001 -- any failure -> fallback
        rate = None
        err = str(e)[:80]
    if rate is None:
        fb = FALLBACK_RATES.get((machine_type, provisioning))
        if fb is None:
            raise SystemExit(f"no catalog price and no fallback for "
                             f"{machine_type} {provisioning}")
        return fb, "FALLBACK 2026-09-04 (catalog unavailable)"
    cache[key] = {"usd_per_h": rate, "date": datetime.date.today().isoformat()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1))
    return rate, "catalog"


# ------------------------------------------------------------ task times
def pass_class(key):
    """'heavy' = low-altitude real pass (7.5 m facets); 'light' otherwise."""
    if key.startswith("haps") or any(t in key for t in
                                     ("9km", "10km", "11km", "high")):
        return "light"
    return "heavy"


def timing_records(line):
    """Per-task timing json records of past jobs for ``line``."""
    out = []
    for f in glob.glob(str(ROOT / "outputs" / "gcp" / "*" / "timing"
                           / "*.json")):
        d = json.load(open(f))
        if d.get("line") == line and d.get("mode") == "simulate":
            out.append(d)
    return out


def chunk_wall_s(line, exp):
    """Median simulated wall_s per chunk by pass, from the runs/ jsons."""
    by = {}
    for f in glob.glob(str(ROOT / "outputs" / line / exp / "runs" / "*.json")):
        d = json.load(open(f))
        key = d.get("meta", {}).get("pass") or Path(f).name.split(
            f"_{exp}_")[0]
        if "wall_s" in d:
            by.setdefault(key, []).append(d["wall_s"])
    return {k: statistics.median(v) for k, v in by.items()}


def estimate(line, exp, passes, chunks_per_task):
    """{pass: (prep_s, chunk_s)} from records when available, else defaults.

    prep_s is the per-task fixed cost (median task run_s minus the chunk
    wall_s it simulated); chunk_s the median simulated seconds per chunk."""
    walls = chunk_wall_s(line, exp)
    recs = timing_records(line)
    prep = None
    if recs and walls:
        diffs = []
        for r in recs:
            key, _, cs = r["pass"].partition(":")
            n = len(cs.split(",")) if cs else 1
            if key in walls and r.get("run_s", 0) > n * walls[key]:
                diffs.append(r["run_s"] - n * walls[key])
        if diffs:
            prep = statistics.median(diffs)
    out = {}
    for key in passes:
        cs = walls.get(key) or DEFAULT_CHUNK_S[pass_class(key)]
        out[key] = (prep if prep is not None else DEFAULT_PREP_S, cs)
    return out


def project(task_counts, est, chunks_per_task, rate, nat_hours=0.0):
    """VM-hours and USD for {pass: n_tasks} at ``chunks_per_task``."""
    secs = sum(n * (est[k][0] + chunks_per_task * est[k][1])
               for k, n in task_counts.items())
    vmh = secs / 3600.0 * (1.0 + OVERHEAD)
    return vmh, vmh * rate + nat_hours * 0.044


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("machine_type")
    ap.add_argument("--provisioning", default="SPOT")
    ap.add_argument("--region", default="us-central1")
    a = ap.parse_args()
    rate, src = vm_hour_price(a.machine_type, a.provisioning, a.region)
    print(f"{a.machine_type} {a.provisioning} {a.region}: ${rate:.4f}/VM-h "
          f"({src})")


if __name__ == "__main__":
    main()
