"""Pull OIB ATM L1B (ILATM1B) granules for the study lines' radar flight days.

Spatial: line nav envelope +- 2 km. Temporal: frame UTC span +- 3 min.
  uv run claude_notes/atm_roughness/atm_pull.py --dry-run
  uv run claude_notes/atm_roughness/atm_pull.py --lines greenland_westcoast
  uv run claude_notes/atm_roughness/atm_pull.py --david-search   # all years over the David bbox
Logs granule ids/sizes to outputs/cache/atm/<line>/<date>/granules.json.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
import atm_common as ac  # noqa: E402

MAX_GB_PER_LINE = 5.0


def search(short_name, bbox, temporal):
    import earthaccess
    res = earthaccess.search_data(short_name=short_name, bounding_box=bbox, temporal=temporal)
    return [(g["umm"]["GranuleUR"], float(g.size()), g) for g in res]


def pull_day(line, date, season, fids, dry, extra_pad_min=3):
    import earthaccess
    lat, lon, t = ac.frames_nav(season, fids)
    bbox = ac.bbox_deg(lat, lon)
    t0 = (t.min() - np.timedelta64(extra_pad_min, "m")).astype("datetime64[s]")
    t1 = (t.max() + np.timedelta64(extra_pad_min, "m")).astype("datetime64[s]")
    out = ac.ATM_CACHE / line / date
    out.mkdir(parents=True, exist_ok=True)
    log = {"line": line, "date": date, "season": season, "frames": fids, "bbox": bbox,
           "temporal": [str(t0), str(t1)], "collections": {}}
    total = 0.0
    for sn in ("ILATM1B", "ILNSA1B"):
        gr = search(sn, bbox, (str(t0), str(t1)))
        log["collections"][sn] = [{"id": i, "size_mb": s} for i, s, _ in gr]
        mb = sum(s for _, s, _ in gr)
        total += mb
        print(f"  {line} {date} {sn}: {len(gr)} granules, {mb:.0f} MB  bbox={np.round(bbox, 3).tolist()} t={t0}..{t1}")
        if not dry and gr:
            if total / 1e3 > MAX_GB_PER_LINE:
                print(f"  !! exceeds {MAX_GB_PER_LINE} GB, skipping download"); log["skipped"] = "size"; break
            todo = [g for i, _, g in gr if not (out / i).exists()]
            if todo:
                earthaccess.download(todo, str(out), threads=4)
    log["total_mb"] = total
    (out / "granules.json").write_text(json.dumps(log, indent=1))
    return total


def david_search(dry):
    """No ATM on the David passes: find OIB ATM tracks over the David bbox, any year."""
    import earthaccess
    xy, s, crs = ac.anchor_axis("antarctica_david")
    spec = ac.load_line("antarctica_david")
    ref = spec["passes"][spec["reference"]["pass"]]
    lat, lon, _ = ac.frames_nav(ref["season"], spec["reference"]["frames"])
    bbox = ac.bbox_deg(lat, lon, pad_m=5000.0)
    print("david bbox", np.round(bbox, 3).tolist())
    found = {}
    for yr in range(2009, 2020):
        for sn in ("ILATM1B", "ILNSA1B"):
            gr = search(sn, bbox, (f"{yr}-01-01", f"{yr}-12-31"))
            if gr:
                print(f"  {yr} {sn}: {len(gr)} granules {sum(s for _, s, _ in gr):.0f} MB: "
                      + ", ".join(i for i, _, _ in gr[:8]) + (" ..." if len(gr) > 8 else ""))
                found.setdefault(sn, []).extend(gr)
    out = ac.ATM_CACHE / "antarctica_david"
    out.mkdir(parents=True, exist_ok=True)
    (out / "search_all_years.json").write_text(json.dumps(
        {sn: [{"id": i, "size_mb": s} for i, s, _ in v] for sn, v in found.items()}, indent=1))
    if not dry:
        import earthaccess
        for sn, v in found.items():
            for i, _, g in v:
                ds = i.split("_")[1]; d = out / f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"
                d.mkdir(exist_ok=True)
                if not (d / i).exists():
                    earthaccess.download([g], str(d))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", nargs="*", default=list(ac.ATM_DAYS))
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--david-search", action="store_true")
    a = ap.parse_args()
    import earthaccess
    earthaccess.login(strategy="netrc")
    if a.david_search:
        david_search(a.dry_run); return
    for line in a.lines:
        tot = 0.0
        for date, (season, fids) in ac.ATM_DAYS[line].items():
            if a.dates and date not in a.dates:
                continue
            tot += pull_day(line, date, season, fids, a.dry_run)
        print(f"{line}: total {tot / 1e3:.2f} GB")


if __name__ == "__main__":
    main()
