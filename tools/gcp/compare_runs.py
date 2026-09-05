#!/usr/bin/env python3
"""Compare two outputs/ trees produced by the same spec: chunk arrays per
rid, and metrics.json scalars per pass key. Cloud-vs-local check.

  uv run python tools/gcp/compare_runs.py outputs_cloud outputs_local \
      --lines antarctica_pineisland_north --exp pilot
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def cmp_chunks(a, b):
    """Per rid: max |diff| and max |diff| / max |ref| of every array."""
    rows = []
    for ja in sorted(a.glob("*.json")):
        rid = ja.stem
        jb = b / ja.name
        if not jb.exists():
            rows.append((rid, "MISSING in ref", None, None, None))
            continue
        ma, mb = json.loads(ja.read_text()), json.loads(jb.read_text())
        same_meta = ma["meta_key"] == mb["meta_key"]
        za, zb = np.load(ja.with_suffix(".npz")), np.load(jb.with_suffix(".npz"))
        worst = {}
        for k in za.files:
            x, y = za[k], zb[k]
            if x.shape != y.shape:
                worst[k] = ("shape", x.shape, y.shape)
                continue
            d = np.abs(x - y).max()
            ref = np.abs(y).max()
            worst[k] = (float(d), float(d / ref) if ref else 0.0)
        rows.append((rid, "meta==" if same_meta else "META DIFFERS",
                     worst, ma["wall_s"], mb["wall_s"]))
    return rows


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def cmp_metrics(a, b, keys):
    """(differing rows, n shared keys, problems): a non-finite value on
    either side or a filtered key present on one side only is a problem,
    never silently skipped."""
    ma = flat(json.loads(a.read_text()).get("metrics", {}))
    mb = flat(json.loads(b.read_text()).get("metrics", {}))
    sel = lambda k: not keys or any(s in k for s in keys)  # noqa: E731
    skip = lambda k: "wall" in k or k.endswith("_s")       # noqa: E731
    rows, problems = [], []
    for k in sorted((set(ma) ^ set(mb))):
        if sel(k) and not skip(k):
            problems.append((k, "only in cloud" if k in ma else "only in local"))
    for k in sorted(set(ma) & set(mb)):
        if not sel(k) or skip(k):
            continue
        if not (np.isfinite(ma[k]) and np.isfinite(mb[k])):
            if not (np.isnan(ma[k]) and np.isnan(mb[k])):
                problems.append((k, f"non-finite: cloud {ma[k]} local {mb[k]}"))
            continue
        if ma[k] != mb[k]:
            rows.append((k, ma[k], mb[k], ma[k] - mb[k]))
    return rows, len(set(ma) & set(mb)), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cloud")
    ap.add_argument("local")
    ap.add_argument("--lines", nargs="+", required=True)
    ap.add_argument("--exp", default="pilot")
    ap.add_argument("--metric-keys", nargs="*",
                    default=["midcol_rel_surf_db", "bed_visibility",
                             "clutter_", "rssnr_level"])
    a = ap.parse_args()
    bad = 0
    for line in a.lines:
        ca, cb = (Path(a.cloud) / line / a.exp, Path(a.local) / line / a.exp)
        print(f"\n=== {line}/{a.exp}")
        for rid, st, worst, wa, wb in cmp_chunks(ca / "runs", cb / "runs"):
            if worst is None:
                print(f"  {rid}: {st}")
                bad += 1
                continue
            if st != "meta==" or any(v[0] == "shape" or v[0] > 0
                                     for v in worst.values()):
                bad += 1
            f = worst.get("field")
            if f[0] == "shape":
                print(f"  {rid[:40]}..: {st}  field shape {f[1]} vs {f[2]} "
                      f"(same rid, different chunking)")
                continue
            print(f"  {rid[:40]}..: {st}  field max|d| {f[0]:.3e} "
                  f"rel {f[1]:.2e}  nadir {worst['nadir_twtt'][0]:.2e} s  "
                  f"twtt {worst['twtt'][0]:.2e} s  wall cloud/local "
                  f"{wa:.0f}/{wb:.0f} s")
        if (ca / "metrics.json").exists() and (cb / "metrics.json").exists():
            rows, n, problems = cmp_metrics(
                ca / "metrics.json", cb / "metrics.json", a.metric_keys)
            print(f"  metrics: {n} shared scalar keys, {len(rows)} differ, "
                  f"{len(problems)} missing/non-finite (filtered to "
                  f"{a.metric_keys}):")
            for k, x, y, d in rows:
                print(f"    {k}: cloud {x:.4f} local {y:.4f} diff {d:+.4f}")
            for k, why in problems:
                print(f"    {k}: {why}")
            bad += len(rows) + len(problems)
        else:
            print("  metrics: missing on one side")
            bad += 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
