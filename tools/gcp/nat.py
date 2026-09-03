#!/usr/bin/env python3
"""Cloud NAT lifecycle for no-external-IP Batch VMs.

External IPs are capped by IN_USE_ADDRESSES (8 per region here), so a
fan-out beyond ~8 VMs needs VMs without external IPs: Private Google Access
on the subnet (free; GCS/Batch/logging traffic) plus a Cloud NAT for the
few internet fetches (uv installer, PyPI wheels, the simc git dependency).
The NAT gateway costs ~$0.044/h while it exists (+ $0.045/GB processed), so
it is created on demand and deleted when the run finishes.

  uv run python tools/gcp/nat.py status | up | down [--force]

`down` refuses (prints a warning) while any other soundersim Batch job is
QUEUED/SCHEDULED/RUNNING in the region, so concurrent launches do not pull
the gateway out from under each other; `--force` overrides.
"""
import argparse
import json
import subprocess
import sys

REGION = "us-central1"
NETWORK, SUBNET = "default", "default"
ROUTER, NAT = "soundersim-nat-router", "soundersim-nat"
JOB_PREFIX = "soundersim"


def _run(*args, check=True):
    return subprocess.run(["gcloud", *args], text=True, capture_output=True,
                          check=check)


def _exists(*args):
    return _run(*args, "--format=value(name)", check=False).returncode == 0


def status(region=REGION):
    st = {"private_google_access": _run(
        "compute", "networks", "subnets", "describe", SUBNET,
        f"--region={region}", "--format=value(privateIpGoogleAccess)",
        check=False).stdout.strip(),
        "router": _exists("compute", "routers", "describe", ROUTER,
                          f"--region={region}"),
        "nat": _exists("compute", "routers", "nats", "describe", NAT,
                       f"--router={ROUTER}", f"--region={region}")}
    print(f"{region}: private google access {st['private_google_access']}, "
          f"router {ROUTER} {'EXISTS' if st['router'] else 'absent'}, "
          f"NAT {NAT} {'EXISTS (~$0.044/h)' if st['nat'] else 'absent'}",
          flush=True)
    return st


def up(region=REGION):
    """Idempotent: PGA on (left on permanently), router + NAT if absent."""
    st = status(region)
    if st["private_google_access"] != "True":
        _run("compute", "networks", "subnets", "update", SUBNET,
             f"--region={region}", "--enable-private-ip-google-access")
        print("enabled Private Google Access", flush=True)
    if not st["router"]:
        _run("compute", "routers", "create", ROUTER, f"--network={NETWORK}",
             f"--region={region}")
        print(f"created router {ROUTER}", flush=True)
    if not st["nat"]:
        _run("compute", "routers", "nats", "create", NAT, f"--router={ROUTER}",
             f"--region={region}", "--auto-allocate-nat-external-ips",
             "--nat-all-subnet-ip-ranges")
        print(f"created NAT {NAT}", flush=True)
    return status(region)


def active_jobs(region=REGION, prefix=JOB_PREFIX):
    """Names of soundersim Batch jobs still queued/scheduled/running."""
    out = _run("batch", "jobs", "list", f"--location={region}",
               "--format=json", check=False).stdout or "[]"
    return [j["name"].rsplit("/", 1)[-1] for j in json.loads(out)
            if prefix in j["name"].rsplit("/", 1)[-1]
            and j.get("status", {}).get("state") in
            ("QUEUED", "SCHEDULED", "RUNNING")]


def down(region=REGION, force=False):
    """Delete NAT + router unless another soundersim job still needs them."""
    st = status(region)
    if not (st["router"] or st["nat"]):
        return True
    busy = active_jobs(region)
    if busy and not force:
        print(f"WARNING: NAT left up: jobs still active {busy} -- run "
              "`tools/gcp/nat.py down` when they finish", flush=True)
        return False
    if st["nat"]:
        _run("compute", "routers", "nats", "delete", NAT, f"--router={ROUTER}",
             f"--region={region}", "--quiet")
        print(f"deleted NAT {NAT}", flush=True)
    if st["router"]:
        _run("compute", "routers", "delete", ROUTER, f"--region={region}",
             "--quiet")
        print(f"deleted router {ROUTER}", flush=True)
    status(region)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["status", "up", "down"])
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--force", action="store_true",
                    help="down: delete even while jobs are active")
    a = ap.parse_args()
    if a.action == "status":
        status(a.region)
    elif a.action == "up":
        up(a.region)
    else:
        sys.exit(0 if down(a.region, a.force) else 1)


if __name__ == "__main__":
    main()
