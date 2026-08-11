"""Four-panel single-trace decomposition figures on the full_line campaign.

Session artifact; ASSEMBLY ONLY (att20_klevel settings, K = +7.92 reused).
Panels low / mid / syn14km / syn300km (NOT high) at one along-track
location: per-interface single-trace curves ("surface returns" / "bed
returns", measured where it exists), the trace's ACTUAL bed marked (sim
bed-layer nadir twtt; measured Bottom pick too when it differs visibly),
NO window shading/annotations.

Uses the tool's processed-stack cache (`load_proc_pass` /
`process_standard_cached`): with a warm cache a figure is seconds; a cold
or stale pass falls back to the full prep+replay+focus path and populates
the cache.

    uv run python claude_notes/trace4_fig.py --s-km 35.0 \
        --out decomposition_trace4_s35.png [--verify-pass mid]
    uv run python claude_notes/trace4_fig.py --populate-only --passes high
"""
import argparse
import gc
import shutil
import sys
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_basal_clutter as rbc  # noqa: E402

KEYS = ["low", "mid", "syn14km", "syn300km"]
ATT, D = 20.0, 3.56
OUT = ROOT / "outputs" / "basal_clutter" / "full_line"
VER = ROOT / "outputs" / "verification" / "basal_clutter_full_line"
_STATE = {}


def _gmap_axis():
    if "gmap" not in _STATE:
        axis = rbc.ref_bed_picks()
        gmap = rbc.build_rssnr_gamma(axis, "full_line", ATT, anchor="level",
                                     level_deficit_db=D,
                                     k_anchor_segment="full")
        assert gmap["k_db"] == 7.92, gmap["k_db"]
        _STATE["gmap"], _STATE["axis"] = gmap, axis
    return _STATE["gmap"], _STATE["axis"]


def get_pass(key):
    """(p, sim, proc): proc-cache fast path, else full path (populates)."""
    t0 = time.perf_counter()
    loaded = rbc.load_proc_pass(key, OUT)
    if loaded is None:
        gmap, axis = _gmap_axis()
        p = rbc.prep_pass(key, "full_line", None, gmap=gmap, axis=axis,
                          fine_posting=True, dgn_seed=0, hybrid=True)
        sim = rbc.simulate_pass(p, OUT / "runs", ATT, True, False)
        proc = rbc.process_standard_cached(p, sim, OUT, ATT, True)
        loaded = (p, sim, proc)
    print(f"  [{key}] ready in {time.perf_counter() - t0:.1f} s", flush=True)
    return loaded


def collect(key, s_km):
    p, sim, proc = get_pass(key)
    a = rbc.analyze_pass(p, sim, proc=proc, trace_s_km=s_km)
    ti = a["trace_info"]
    keep = {"tinfo": ti, "tprofs": dict(a["trace_profs"]),
            "src": rbc.source_label(key, p), "h_med": p["h_med"],
            "meas_bed_below_surf_us": None}
    if "measured_trace_index" in ti:
        j = ti["measured_trace_index"]
        keep["meas_bed_below_surf_us"] = float(
            (p["bot"][j] - p["surf"][j]) * 1e6)
    print(f"  {key}: sim trace {ti['sim_trace_index']} (s = "
          f"{ti['sim_s_km']:.3f} km)"
          + (f", measured trace {ti['measured_trace_index']}"
             if "measured_trace_index" in ti else "")
          + f"; bed at {ti['bed_below_surface_us']:.2f} us; bed-window "
          f"bed - surface returns "
          f"{ti['bed_window_bed_minus_surface_returns_db']:+.1f} dB",
          flush=True)
    del p, sim, proc, a
    gc.collect()
    return keep


def verify_bitexact(key, s_km):
    """Cached path vs a DIRECT full recompute: the extracted trace curves
    and the full processed power stack must be bit-identical."""
    print(f"== VERIFY {key}: cached vs direct recompute ==", flush=True)
    loaded = rbc.load_proc_pass(key, OUT)
    assert loaded is not None, "cache missing for verification"
    pc, simc, procc = loaded
    ac = rbc.analyze_pass(pc, simc, proc=procc, trace_s_km=s_km)
    gmap, axis = _gmap_axis()
    pd_ = rbc.prep_pass(key, "full_line", None, gmap=gmap, axis=axis,
                        fine_posting=True, dgn_seed=0, hybrid=True)
    simd = rbc.simulate_pass(pd_, OUT / "runs", ATT, True, False)
    procd = rbc.process_standard(pd_, simd)          # UNCACHED direct
    ad = rbc.analyze_pass(pd_, simd, proc=procd, trace_s_km=s_km)
    ok_p = bool(np.array_equal(procc["P"], procd["P"]))
    ok_tr = all(np.array_equal(ac["trace_profs"][k][1],
                               ad["trace_profs"][k][1])
                and np.array_equal(ac["trace_profs"][k][0],
                                   ad["trace_profs"][k][0])
                for k in ad["trace_profs"])
    print(f"VERIFY {key}: P bit-identical={ok_p}, trace curves "
          f"bit-identical={ok_tr} "
          f"({'PASS' if ok_p and ok_tr else 'FAIL'})", flush=True)
    if not (ok_p and ok_tr):
        raise SystemExit(1)


def make_fig(s_km, fname):
    panels = {}
    for key in KEYS:
        print(f"== {key} ==", flush=True)
        panels[key] = collect(key, s_km)
    series = [("measured", "measured", dict(color="black", lw=1.2)),
              ("sim_surface", "sim surface returns",
               dict(color="tab:orange", lw=1.0, ls="--")),
              ("sim_bed", "sim bed returns",
               dict(color="tab:green", lw=1.1, ls="-."))]
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 9.2), sharex=True,
                            sharey=True)
    for k, key in enumerate(KEYS):
        ax, pn = axs.flat[k], panels[key]
        ti = pn["tinfo"]
        for pk, label, st in series:
            if pk in pn["tprofs"]:
                ax.plot(*pn["tprofs"][pk], label=label, **st)
        tb = ti["bed_below_surface_us"]
        ax.axvline(tb, color="tab:red", lw=1.2)
        ax.text(tb + 0.12, 2.5, "bed", color="tab:red", fontsize=9,
                va="top")
        tbm = pn["meas_bed_below_surf_us"]
        if tbm is not None and abs(tbm - tb) > 0.1:
            ax.axvline(tbm, color="0.35", lw=1.0, ls=":")
            ax.text(tbm - 0.12, -100, "bed (measured pick)", color="0.35",
                    fontsize=8, ha="right")
        ax.set_xlim(-1.0, 13.5)
        ax.set_ylim(-110, 5)
        ax.grid(alpha=0.3)
        src = "\n".join(textwrap.wrap(pn["src"], 58))
        ax.set_title(f"{key} ({pn['h_med']:.0f} m AGL) -- trace "
                     f"{ti['sim_trace_index']}\n{src}", fontsize=8.5)
        if k % 2 == 0:
            ax.set_ylabel("dB rel own surface-return peak\n(single trace)")
        if k >= 2:
            ax.set_xlabel("twtt below surface returns (us)")
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"SINGLE-TRACE decomposition at anchor s = {s_km:.1f} km "
                 "(one sounding per panel; red line = that trace's bed)",
                 fontsize=11)
    fig.tight_layout()
    fp = OUT / fname
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    VER.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fp, VER / fp.name)
    print(f"wrote {fp} (+ mirror {VER / fp.name})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s-km", type=float, default=35.0)
    ap.add_argument("--out", default=None, help="figure file name")
    ap.add_argument("--passes", nargs="+", default=KEYS,
                    help="passes to touch (with --populate-only: just warm "
                    "their proc caches)")
    ap.add_argument("--populate-only", action="store_true")
    ap.add_argument("--verify-pass", default=None,
                    help="bit-compare this pass's cached stack + trace "
                    "curves against a direct recompute")
    args = ap.parse_args()
    if args.populate_only:
        for key in args.passes:
            print(f"== populate {key} ==", flush=True)
            get_pass(key)
            gc.collect()
        return
    make_fig(args.s_km, args.out or f"decomposition_trace4_s{args.s_km:g}.png")
    if args.verify_pass:
        verify_bitexact(args.verify_pass, args.s_km)


if __name__ == "__main__":
    main()
