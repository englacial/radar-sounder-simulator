"""ILATM1B (+ ILNSA1B) granule search/download per visit via earthaccess.
Cache: outputs/cache/atm/tier2/<site>/<date>/<granule>; granules.json per site-date."""
from __future__ import annotations
import json, time
import numpy as np
from common import CACHE

_logged = False


def login():
    global _logged
    import earthaccess
    if not _logged:
        earthaccess.login(strategy="netrc"); _logged = True


def search_visit(lat, lon, date, t0, t1, pad_s=90, pad_km=3.0):
    """Granules covering the site on that date/time window. Returns list of (id, MB, granule)."""
    import earthaccess
    login()
    dlat = pad_km / 111.2; dlon = pad_km / (111.2 * np.cos(np.radians(lat)))
    bbox = (float(lon - dlon), float(lat - dlat), float(lon + dlon), float(lat + dlat))
    d = str(date)[:10]
    a, b = max(0, t0 - pad_s), min(86399, t1 + pad_s)
    ta = f"{d}T{int(a // 3600):02d}:{int(a % 3600 // 60):02d}:{int(a % 60):02d}"
    tb = f"{d}T{int(b // 3600):02d}:{int(b % 3600 // 60):02d}:{int(b % 60):02d}"
    out = []
    for sn in ("ILATM1B", "ILNSA1B"):
        for attempt in range(3):
            try:
                res = earthaccess.search_data(short_name=sn, bounding_box=bbox, temporal=(ta, tb)); break
            except Exception as e:  # noqa: BLE001
                res = []; time.sleep(5 * (attempt + 1)); err = str(e)[:80]
        for g in res:
            out.append((g["umm"]["GranuleUR"], float(g.size()), sn, g))
    return out


def download(granules, site, date, threads=4):
    """Download missing granules to the site/date dir; returns (paths, MB new, ok)."""
    import earthaccess
    d = CACHE / site / str(date)[:10]; d.mkdir(parents=True, exist_ok=True)
    todo = [g for gid, _, _, g in granules if not (d / gid).exists()]
    mb_new = sum(mb for gid, mb, _, _ in granules if not (d / gid).exists())
    for attempt in range(3):
        if not todo: break
        try:
            earthaccess.download(todo, str(d), threads=threads)
        except Exception as e:  # noqa: BLE001
            print(f"    download error {site} {date} attempt {attempt}: {str(e)[:100]}", flush=True); time.sleep(10)
        todo = [g for gid, _, _, g in granules if not (d / gid).exists()]
    paths = [d / gid for gid, _, _, _ in granules if (d / gid).exists()]
    (d / "granules.json").write_text(json.dumps([dict(id=gid, mb=mb, coll=sn) for gid, mb, sn, _ in granules], indent=1))
    return paths, mb_new, len(todo) == 0
