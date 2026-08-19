"""Survey one study line: what data it uses, and how well the passes repeat.

    uv run python tools/line_report.py config/lines/<name>.yaml [--segment S]

Produces, under outputs/line_reports/<line>/:

  map.png         the flight tracks in the line's own CRS, with the study
                  segment highlighted and each pass labelled by year
  radargrams.png  every pass over the SAME along-track span, trimmed to it
                  and aligned on its own surface pick, on a shared
                  depth-below-surface axis
  metrics.json    how close the passes actually are -- lateral offset, along
                  -track coverage, surface and thickness agreement, and
                  whether they were even flown by comparable instruments
  report.html     the three together

WHY IT EXISTS: a "repeat pass" is a claim, and the claim has a size. Two
flights 400 m apart over rough bed are not sampling the same scene, and two
flights on different fast-time lattices are not one instrument. This turns
both into numbers before a study is built on them.

Offsets are measured from each frame's OWN NAV, never from the STAC
geometry: STAC carries a coarse decimation of the track and can misplace it
by hundreds of metres, which is fine for discovery and useless as a metric.
"""

import argparse
import base64
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import line_geometry as lg  # noqa: E402
from clutter_lines import load_line  # noqa: E402
from soundersim.opr import load_bottom_pick, load_frame  # noqa: E402

C = 299792458.0
OUT_ROOT = ROOT / "outputs" / "line_reports"
REL_US = (-1.0, 36.0)        # depth-below-surface window for the radargrams
REL_DT_NS = 10.0             # shared fast-time step the passes are put on
N_ALONG = 1400               # shared along-track samples across the segment


# ------------------------------------------------------------------ load
def load_pass(spec, key, segment):
    """Concatenated frames, nav and picks for one pass over one segment."""
    ps = spec.passes[key] if key in spec.passes else None
    syn = spec.synthetic_passes.get(key)
    if ps is None and syn is not None:
        return None                     # synthetic: no measured data exists
    season = ps.season or spec.identity.season
    parts = ps.segments[segment]
    frames, bots, full_ll = [], [], []
    for part in parts:
        f = load_frame(season, part.frame)
        a, b = part.slice
        # unsliced nav, kept for the map: the segment is a WINDOW on these
        # frames, and a map that shows only the window cannot show that.
        full_ll.append((np.asarray(f.Latitude.values, np.float64),
                        np.asarray(f.Longitude.values, np.float64)))
        frames.append(f.isel(slow_time=slice(a, b)))
        try:
            bots.append(load_bottom_pick(f)[a:b])
        except Exception:
            bots.append(np.full(b - a, np.nan))
    import xarray as xr
    fs = (frames[0] if len(frames) == 1
          else xr.concat(frames, dim="slow_time", combine_attrs="override"))
    if ps.reversed:
        fs = fs.isel(slow_time=slice(None, None, -1))
        bots = [b[::-1] for b in bots][::-1]
    return {"key": key, "season": season, "frame": fs,
            "full_lat": np.concatenate([a for a, _ in full_ll]),
            "full_lon": np.concatenate([b for _, b in full_ll]),
            "bot": np.concatenate(bots),
            "frames": [p.frame for p in parts],
            "instrument": ps.instrument,
            "agl_declared": ps.agl_med_m}


def measure(spec, segment):
    """Project every real pass onto the reference pass's axis."""
    ref_key = spec.reference.pass_key
    crs = spec.identity.crs
    # a pass may not reach every window; those simply are not in this survey
    keys = [k for k in spec.order
            if k in spec.passes and segment in spec.passes[k].segments]
    absent = [k for k in spec.order if k in spec.passes
              and segment not in spec.passes[k].segments]
    if absent:
        print(f"  segment {segment!r}: {absent} do not reach this window",
              flush=True)
    data = {}
    for k in keys:
        d = load_pass(spec, k, segment)
        if d is None:
            continue
        xy, lat, lon = lg.frame_nav(d["frame"], crs)
        d.update(xy=xy, lat=lat, lon=lon, agl=lg.agl_m(d["frame"]),
                 xy_full=lg.to_crs(d["full_lat"], d["full_lon"], crs),
                 surf=np.asarray(d["frame"].Surface.values, np.float64))
        data[k] = d
    if ref_key not in data:
        raise SystemExit(f"reference pass {ref_key!r} has no data")
    ref = data[ref_key]
    ref["s_ref"] = lg.arc_length(ref["xy"])
    for k, d in data.items():
        d["s"], d["lat_off"] = lg.project(d["xy"], ref["xy"], ref["s_ref"])
    return data, ref_key


def common_span(data):
    """The along-track range every pass actually covers."""
    lo = max(float(np.nanmin(d["s"])) for d in data.values())
    hi = min(float(np.nanmax(d["s"])) for d in data.values())
    if hi <= lo:
        raise SystemExit("the passes share no along-track span")
    return lo, hi


# --------------------------------------------------------------- metrics
def pass_metrics(d, lo, hi, ref_key, ref):
    m = (d["s"] >= lo) & (d["s"] <= hi)
    n = int(m.sum())
    off = np.abs(d["lat_off"][m])
    tw = np.asarray(d["frame"].twtt.values, np.float64)
    dt = float((tw[-1] - tw[0]) / (len(tw) - 1))
    sp = np.diff(d["s"][m])
    out = {
        "frames": d["frames"], "season": d["season"],
        "instrument": d["instrument"],
        "n_traces_in_span": n,
        "coverage_frac_of_span": round(
            float((d["s"][m].max() - d["s"][m].min()) / (hi - lo)), 4)
        if n > 1 else 0.0,
        "trace_spacing_m": round(float(np.median(np.abs(sp))), 2)
        if n > 1 else None,
        "agl_m": {"med": round(float(np.nanmedian(d["agl"][m])), 0),
                  "p5": round(float(np.nanpercentile(d["agl"][m], 5)), 0),
                  "p95": round(float(np.nanpercentile(d["agl"][m], 95)), 0)},
        "product_dt_ns": round(dt * 1e9, 4),
        "product_t0_us": round(float(tw[0]) * 1e6, 4),
    }
    if d["key"] != ref_key:
        out["lateral_offset_m"] = {
            "med": round(float(np.median(off)), 1),
            "p95": round(float(np.percentile(off, 95)), 1),
            "max": round(float(off.max()), 1),
            "note": "distance from the reference pass's track, measured from "
                    "each frame's OWN nav"}
        # surface and thickness agreement on a shared along-track grid
        grid = np.linspace(lo, hi, 400)
        def on(dd, v):
            mm = (dd["s"] >= lo) & (dd["s"] <= hi) & np.isfinite(v)
            if mm.sum() < 2:
                return np.full_like(grid, np.nan)
            o = np.argsort(dd["s"][mm])
            return np.interp(grid, dd["s"][mm][o], v[mm][o])
        z_ref = on(ref, np.asarray(ref["frame"].Elevation.values, np.float64)
                   - ref["surf"] * C / 2.0)
        z_d = on(d, np.asarray(d["frame"].Elevation.values, np.float64)
                 - d["surf"] * C / 2.0)
        th_ref = on(ref, (ref["bot"] - ref["surf"]) * C / (2 * np.sqrt(3.17)))
        th_d = on(d, (d["bot"] - d["surf"]) * C / (2 * np.sqrt(3.17)))
        def stat(a, b, unit):
            r = b - a
            ok = np.isfinite(r)
            if ok.sum() < 2:
                return {"note": "no overlap in picks"}
            return {"med": round(float(np.median(r[ok])), 1),
                    "rms": round(float(np.sqrt(np.mean(r[ok] ** 2))), 1),
                    "p95_abs": round(float(np.percentile(np.abs(r[ok]), 95)), 1),
                    "unit": unit, "n": int(ok.sum())}
        out["surface_elevation_minus_reference"] = stat(z_ref, z_d, "m")
        out["ice_thickness_minus_reference"] = stat(th_ref, th_d, "m")
    return out


def instrument_parity(mets):
    """Are these even the same instrument? Different fast-time lattices mean
    different products, whatever the carrier says."""
    dts = {k: v["product_dt_ns"] for k, v in mets.items()}
    uniq = sorted(set(dts.values()))
    return {"product_dt_ns_per_pass": dts,
            "distinct_lattices": len(uniq),
            "identical_lattice": len(uniq) == 1,
            "instrument_per_pass": {k: v["instrument"]
                                    for k, v in mets.items()},
            "note": "a shared fast-time lattice is the cheapest test that two "
                    "passes are one product. Differing dt does not invalidate "
                    "a comparison, but every depth/delay comparison must then "
                    "be made in metres or microseconds on each pass's own "
                    "lattice, never by bin index"}


# --------------------------------------------------------------- figures
def fig_map(out, spec, data, ref_key, lo, hi):
    """Where the data is: full tracks muted, the shared span highlighted."""
    fig, ax = plt.subplots(figsize=(9.6, 8.4))
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(data)))
    for (k, d), col in zip(data.items(), cols):
        ax.plot(*(d["xy_full"] / 1e3).T, color=col, lw=0.7, alpha=0.35)
        xy = d["xy"] / 1e3
        m = (d["s"] >= lo) & (d["s"] <= hi)
        yr = d["season"].split("_")[0]
        ax.plot(xy[m, 0], xy[m, 1], color=col, lw=2.0,
                label=f"{k}  {yr}  {np.nanmedian(d['agl'][m]):.0f} m AGL"
                      + ("  (reference)" if k == ref_key else ""))
    r = data[ref_key]
    for s_mark, txt in ((lo, "s0"), (hi, "s1")):
        i = int(np.argmin(np.abs(r["s_ref"] - s_mark)))
        ax.plot(*(r["xy"][i] / 1e3), "k.", ms=9)
        ax.annotate(f" {txt} ({s_mark/1e3:.1f} km)", r["xy"][i] / 1e3,
                    fontsize=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel(f"easting (km, {spec.identity.crs})")
    ax.set_ylabel("northing (km)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0)
    ax.set_title(f"{spec.name}: flight data used\n"
                 f"bold = the {(hi-lo)/1e3:.1f} km span every pass shares; "
                 "faint = the whole frames it is cut from", fontsize=10)
    fig.tight_layout()
    fp = out / "map.png"
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


def _aligned(d, lo, hi, s_grid, rel_us):
    """One pass resampled onto (shared along-track s) x (twtt below its OWN
    surface pick). Both axes are physical, so passes on different fast-time
    lattices land on the same picture without any bin-index assumption."""
    m = (d["s"] >= lo) & (d["s"] <= hi)
    if m.sum() < 2:
        return None
    data = np.asarray(d["frame"].Data.values, np.float64)[m]
    tw = np.asarray(d["frame"].twtt.values, np.float64)
    surf = d["surf"][m]
    out = np.full((m.sum(), rel_us.size), np.nan)
    for i in range(data.shape[0]):
        if not np.isfinite(surf[i]):
            continue
        out[i] = np.interp(surf[i] + rel_us * 1e-6, tw, data[i],
                           left=np.nan, right=np.nan)
    # Normalise each trace by its OWN surface peak. Absolute product scaling
    # is not comparable across six seasons and four radars, and an unnormalised
    # panel is dominated by the surface return -- which crushes the bed to
    # black on the wideband products. dB rel own surface peak is the
    # convention the rest of this repo measures in.
    w = np.abs(rel_us) <= 0.5
    pk = np.nanmax(np.where(w, out, np.nan), axis=1)
    out = out / np.where(np.isfinite(pk) & (pk > 0), pk, np.nan)[:, None]
    s = d["s"][m]
    o = np.argsort(s)
    cols = np.empty((s_grid.size, rel_us.size))
    for j in range(rel_us.size):
        cols[:, j] = np.interp(s_grid, s[o], out[o, j])
    return cols


def fig_radargrams(out, spec, data, ref_key, lo, hi):
    rel_us = np.arange(REL_US[0], REL_US[1], REL_DT_NS * 1e-3)
    s_grid = np.linspace(lo, hi, N_ALONG)
    keys = list(data)
    fig, axs = plt.subplots(len(keys), 1, figsize=(11.0, 2.8 * len(keys)),
                            sharex=True, sharey=True, squeeze=False)
    for ax, k in zip(axs[:, 0], keys):
        d = data[k]
        img = _aligned(d, lo, hi, s_grid, rel_us)
        if img is None:
            ax.set_axis_off()
            continue
        db = 10.0 * np.log10(np.maximum(img, 1e-30))
        fin = db[np.isfinite(db)]
        if spec.figures.radargram.scale == "shared":
            vmin, vmax = spec.figures.radargram.db
        else:
            vmin, vmax = (np.percentile(fin, [2, 99.9]) if fin.size
                          else (-70.0, 5.0))
        ax.imshow(db.T, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax,
                  extent=[s_grid[0] / 1e3, s_grid[-1] / 1e3,
                          rel_us[-1], rel_us[0]])
        ax.axhline(0.0, color="tab:cyan", lw=0.7, ls=":", alpha=0.8)
        yr = d["season"].split("_")[0]
        tw = np.asarray(d["frame"].twtt.values, np.float64)
        dt = (tw[-1] - tw[0]) / (len(tw) - 1) * 1e9
        m = (d["s"] >= lo) & (d["s"] <= hi)
        off = ("reference" if k == ref_key else
               f"lat {np.median(np.abs(d['lat_off'][m])):.0f} m")
        ax.set_ylabel("us below surface", fontsize=8)
        ax.set_title(f"{k} | {yr} {'/'.join(d['frames'])} | "
                     f"{np.nanmedian(d['agl'][m]):.0f} m AGL | dt {dt:.2f} ns "
                     f"| {off}", fontsize=8, loc="left")
    axs[-1, 0].set_xlabel(f"along-track s on the {ref_key} axis (km)")
    fig.suptitle(f"{spec.name}: passes trimmed to the shared span and "
                 "aligned on each pass's own surface pick\n"
                 "dB rel each trace's own surface peak; common depth axis in "
                 "us, NOT bin index, since the lattices differ", fontsize=10)
    fig.tight_layout()
    fp = out / "radargrams.png"
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    return fp


def write_report(out, spec, segment, doc, figs):
    css = ("body{font-family:system-ui,Arial,sans-serif;margin:2rem;"
           "max-width:1200px}img{max-width:100%;border:1px solid #ddd}"
           "pre{background:#f6f6f6;padding:1rem;overflow-x:auto;"
           "font-size:.8rem}")
    imgs = "".join(
        f"<h2>{html.escape(Path(f).stem)}</h2>"
        f"<img src='data:image/png;base64,"
        f"{base64.b64encode(Path(f).read_bytes()).decode()}'>" for f in figs)
    body = (f"<h1>{html.escape(spec.name)} &mdash; line survey "
            f"({html.escape(segment)})</h1>{imgs}"
            f"<h2>metrics</h2><pre>{html.escape(json.dumps(doc, indent=1))}"
            "</pre>")
    fp = out / "report.html"
    fp.write_text(f"<!doctype html><html><head><meta charset='utf-8'>"
                  f"<title>{html.escape(spec.name)}</title>"
                  f"<style>{css}</style></head><body>{body}</body></html>")
    return fp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("line_yaml")
    ap.add_argument("--segment", default=None,
                    help="which segment to survey (default: the longest)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = load_line(args.line_yaml)
    segment = args.segment or max(spec.segments,
                                  key=lambda s: spec.segments[s].n_traces)
    if segment not in spec.segments:
        raise SystemExit(f"{spec.name} has no {segment!r} segment; have "
                         f"{list(spec.segments)}")
    out = Path(args.out or (OUT_ROOT / spec.name / segment))
    out.mkdir(parents=True, exist_ok=True)

    print(f"{spec.name}: surveying segment {segment!r}", flush=True)
    data, ref_key = measure(spec, segment)
    lo, hi = common_span(data)
    print(f"  reference pass {ref_key}; shared span "
          f"{lo/1e3:.2f}-{hi/1e3:.2f} km ({(hi-lo)/1e3:.2f} km)", flush=True)

    mets = {k: pass_metrics(d, lo, hi, ref_key, data[ref_key])
            for k, d in data.items()}
    doc = {"line": spec.name, "segment": segment, "reference_pass": ref_key,
           "crs": spec.identity.crs,
           "shared_span_km": [round(lo / 1e3, 3), round(hi / 1e3, 3)],
           "shared_span_length_km": round((hi - lo) / 1e3, 3),
           "passes": mets,
           "instrument_parity": instrument_parity(mets),
           "note": "lateral offsets and pick agreement are measured from each "
                   "frame's OWN nav over the shared span. Synthetic passes are "
                   "absent by construction: no measured data exists for them."}
    (out / "metrics.json").write_text(json.dumps(doc, indent=1) + "\n")

    figs = [fig_map(out, spec, data, ref_key, lo, hi),
            fig_radargrams(out, spec, data, ref_key, lo, hi)]
    fp = write_report(out, spec, segment, doc, figs)
    for k, v in mets.items():
        off = v.get("lateral_offset_m")
        print(f"  {k:12s} {v['n_traces_in_span']:5d} traces  "
              f"AGL {v['agl_m']['med']:5.0f} m  dt {v['product_dt_ns']:7.4f} ns"
              + (f"  lat med/p95 {off['med']}/{off['p95']} m"
                 if off else "  (reference)"), flush=True)
    print(f"wrote {fp}", flush=True)


if __name__ == "__main__":
    main()
