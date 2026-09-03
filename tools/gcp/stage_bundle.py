#!/usr/bin/env python3
"""Stage the inputs a Batch worker needs for each line of a spec to GCS.

Runs the spec's prep on each line locally with ``--dry-run`` (frames, DEM
windows, picks, RSSNR anchor -- no simulation) while recording every file
opened under outputs/, then uploads those files to <prefix>/data/<relpath>
(no-clobber) and one manifest per line to <prefix>/data/lines/<line>.txt.
Only the files a run actually reads ship: ~0.5-1.5 GB per line, none of
the 67 GB ATM/covariate caches.

  uv run python tools/gcp/stage_bundle.py --config config/experiments/pilot.yaml \
      --lines antarctica_pineisland_north [--no-upload]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

DEFAULT_PREFIX = "gs://ice-infrastructure-soundersim/batch_2026-09-03"


def record_opens(seen):
    """Wrap the readers the tool uses on cached inputs -- netCDF via xarray,
    GeoTIFF via rasterio, npz via numpy (all C-level opens the audit hook
    cannot see) plus Path.read_text/open for the json sidecars -- so their
    paths land in ``seen`` as outputs/-relative strings."""
    import builtins
    import pathlib

    import numpy as np
    import rasterio
    import xarray as xr

    def wrap(mod, name):
        orig = getattr(mod, name)

        def rec(path, *a, **k):
            try:
                # absolute, NOT resolved: cache files may be symlinks into
                # another checkout and must stay outputs/-relative here
                p = Path(os.fspath(path)).absolute()
                out = ROOT / "outputs"
                if out in p.parents:
                    seen.add(str(Path("outputs") / p.relative_to(out)))
            except (TypeError, ValueError):
                pass
            return orig(path, *a, **k)
        setattr(mod, name, rec)
    wrap(xr, "open_dataset")
    wrap(rasterio, "open")
    wrap(np, "load")
    wrap(pathlib.Path, "read_text")
    wrap(builtins, "open")


def line_inputs(config, line):
    import run_basal_clutter as rbc
    seen = set()
    record_opens(seen)
    sys.argv = ["run_basal_clutter", "--config", config, "--line", line,
                "--dry-run"]
    rbc.main_config()
    # the run's own chunk cache is output, not input
    return sorted(f for f in seen if "/runs/" not in f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--lines", nargs="+", required=True)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    import subprocess
    mdir = ROOT / "outputs" / "gcp" / "lines"
    mdir.mkdir(parents=True, exist_ok=True)
    for line in a.lines:
        files = line_inputs(a.config, line)
        size = sum((ROOT / f).stat().st_size for f in files)
        (mdir / f"{line}.txt").write_text("".join(f + "\n" for f in files))
        print(f"{line}: {len(files)} files, {size / 1e9:.2f} GB", flush=True)
        if a.no_upload:
            continue
        # symlinked caches (a read-only view of another checkout) upload as
        # their targets; one cp per destination directory, no-clobber
        by_dir = {}
        for f in files:
            by_dir.setdefault(str(Path(f).parent), []).append(
                str((ROOT / f).resolve()))
        for d, srcs in by_dir.items():
            subprocess.run(["gcloud", "storage", "cp", "-n", "-q", *srcs,
                            f"{a.prefix}/data/{d}/"], check=True)
        subprocess.run(["gcloud", "storage", "cp", "-q",
                        str(mdir / f"{line}.txt"),
                        f"{a.prefix}/data/lines/"], check=True)


if __name__ == "__main__":
    main()
