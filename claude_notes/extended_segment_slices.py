"""Derive + verify the EXTENDED segment slices (anchor s = 0 -> 69.7 km).

Session artifact. Projects every candidate frame's nav onto the anchor
polyline (the same machinery prep_pass/ref_bed_picks use), finds the traces
inside the extended window, and prints the per-pass parts in increasing-s
order together with the coverage/offset/contiguity checks.

    uv run python claude_notes/extended_segment_slices.py
"""
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_altitude_comparison as rac      # noqa: E402
import run_basal_clutter as rbc            # noqa: E402
from soundersim.opr import load_bottom_pick, load_frame   # noqa: E402

S_LO, S_HI = 0.0, 69_700.0
CANDIDATES = {
    "low": ["20161105_05_005", "20161105_05_006", "20161105_05_007"],
    "mid": ["20161028_05_004", "20161028_05_005", "20161028_05_006",
            "20161028_05_007"],
    "high": ["20161031_07_002", "20161031_07_003", "20161031_07_004",
             "20161031_07_005", "20161031_07_006"],
}


def main():
    axis = rbc.ref_bed_picks()
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    tw_ref = {}
    for key, frames in CANDIDATES.items():
        rev = rbc.PASSES[key]["rev"]
        parts, rows = [], []
        for fid in frames:
            try:
                fr = load_frame(rbc.SEASON, fid)
            except Exception as e:                       # noqa: BLE001
                print(f"  {fid}: NOT AVAILABLE ({type(e).__name__})")
                continue
            lat, lon = rac._lonlat(fr)
            x, y = tr.transform(lon, lat)
            s = rbc.project_to_track(x, y, axis["x"], axis["y"], axis["s"])
            # perpendicular offset to the anchor line (nearest anchor sample
            # at its native 14.85 m posting, tangential component removed)
            _, i = cKDTree(np.column_stack([axis["x"], axis["y"]])).query(
                np.column_stack([x, y]))
            d = np.abs(np.hypot(x - axis["x"][i], y - axis["y"][i]))
            n = len(s)
            m = (s >= S_LO) & (s <= S_HI)
            bot = load_bottom_pick(fr)
            tw_ref.setdefault(key, np.asarray(fr.twtt.values, np.float64))
            same_tw = bool(np.allclose(np.asarray(fr.twtt.values, np.float64),
                                       tw_ref[key]))
            if not m.any():
                print(f"  {fid}: n={n}, s {s.min()/1e3:.2f}..{s.max()/1e3:.2f}"
                      " km -- OUTSIDE the window")
                continue
            a, b = int(np.argmax(m)), int(n - np.argmax(m[::-1]))
            if not m[a:b].all():
                raise RuntimeError(f"{fid}: non-contiguous window selection")
            parts.append((fid, (a, b)))
            rows.append(dict(fid=fid, n=n, a=a, b=b, s_a=s[a], s_b=s[b - 1],
                             off_med=float(np.median(d[a:b])),
                             off_max=float(d[a:b].max()),
                             gap=float(np.mean(~np.isfinite(bot[a:b]))),
                             same_tw=same_tw))
        # increasing-s order after per-part reversal
        order = np.argsort([min(r["s_a"], r["s_b"]) for r in rows])
        rows = [rows[i] for i in order]
        parts = [parts[i] for i in order]
        print(f"\n== {key} (rev={rev}) ==")
        for r in rows:
            print(f"  {r['fid']}: slice({r['a']}, {r['b']}) "
                  f"n_frame={r['n']} n_sel={r['b'] - r['a']} "
                  f"s {r['s_a']/1e3:+.2f} -> {r['s_b']/1e3:+.2f} km  "
                  f"offset med/max {r['off_med']:.0f}/{r['off_max']:.0f} m  "
                  f"bottom-pick gaps {r['gap']:.4f}  twtt_match={r['same_tw']}")
        s_all = []
        for r in rows:
            ss = [r["s_a"], r["s_b"]]
            s_all += (ss[::-1] if rev else ss)
        cov = (min(s_all), max(s_all))
        n_tot = sum(r["b"] - r["a"] for r in rows)
        print(f"  parts = {parts}")
        print(f"  coverage {cov[0]/1e3:.2f} -> {cov[1]/1e3:.2f} km, "
              f"{n_tot} traces, {(cov[1]-cov[0])/max(n_tot-1,1):.2f} m/trace")
        # contiguity between consecutive parts (increasing s)
        for r0, r1 in zip(rows[:-1], rows[1:]):
            j0 = max(r0["s_a"], r0["s_b"])
            j1 = min(r1["s_a"], r1["s_b"])
            print(f"  join {r0['fid']} -> {r1['fid']}: ds = {j1 - j0:+.1f} m")


if __name__ == "__main__":
    main()
