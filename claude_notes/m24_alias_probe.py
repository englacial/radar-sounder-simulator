"""M24 alias-dt decision probe (see implementation_plan_stage4.md M24).

The OPR frames sample at dt = 33.333 ns (fs = 30 MHz), so f0*dt = 6.5 and the
envelope-quantization alias lands at |f_a| = 15 MHz = B/2 EXACTLY -- the hann
band edge (a fragile boundary: float rounding of the frame-derived dt decides
which side of round() 6.5 falls on, and simulate()'s in-band warning uses a
strict <). This probe measures, on the real Greenland frame (fast test config:
25 traces / 1200 m reach / 64 m facets), the chirped coherent multilayer
output at

  A) native dt = 33.333 ns  (alias at the band edge)
  B) dt/4     =  8.333 ns  (f0*dt = 1.625 -> alias at 45 MHz = 3*B/2, safely
                            out of band), decimated [::4] back onto the frame
                            twtt grid (t0 = 0 -> exact bin alignment)

and reports the quiet-band floor (between surface and bed) relative to the
surface peak for both, plus which warnings fired. Decision recorded in
claude_notes/m24_findings.md.

Run: uv run python claude_notes/m24_alias_probe.py
"""

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

_spec = importlib.util.spec_from_file_location(
    "run_opr_comparison", ROOT / "tools" / "run_opr_comparison.py")
roc = importlib.util.module_from_spec(_spec)
sys.modules["run_opr_comparison"] = roc
_spec.loader.exec_module(roc)
_spec2 = importlib.util.spec_from_file_location(
    "run_opr_coherent_bed", ROOT / "tools" / "run_opr_coherent_bed.py")
rocb = importlib.util.module_from_spec(_spec2)
sys.modules["run_opr_coherent_bed"] = rocb
_spec2.loader.exec_module(rocb)

from soundersim.config import (DemInterface, FacetConfig, RadarConfig,
                               SimConfig, WaveformConfig)
from soundersim.opr import load_bottom_pick, load_frame
from soundersim.simulate import simulate

B = 30e6
T_PULSE = 10e-6
case = roc.CASES[1]  # Greenland (faster frame)
frame = load_frame(case["season"], case["frame_id"])
mscene, aux = rocb._bed_scene(frame, 25, 1200.0)
rc0, idx = aux["rc"], aux["idx"]
dt, t0, n = rc0.dt, rc0.t0, rc0.n_samples
print(f"native dt = {dt*1e9:.6f} ns, n = {n}, t0 = {t0}")
for name, d in (("native", dt), ("dt/4", dt / 4)):
    fa = abs(195e6 - round(195e6 * d) / d)
    print(f"  {name}: f0*dt = {195e6*d:.9f}, |f_a| = {fa/1e6:.3f} MHz "
          f"(B/2 = {B/2e6:.0f})")

surf_pick = frame.Surface.values[idx]
bot_pick = load_bottom_pick(frame)[idx]


def run(dt_run, n_run, tag):
    wf = WaveformConfig(kind="chirp", bandwidth=B, pulse_length=T_PULSE,
                        window="hann")
    rc = RadarConfig(dt=dt_run, n_samples=n_run, t0=t0, f0=195e6, waveform=wf)
    cfg = SimConfig(mode="coherent", split_sides=False, radar=rc,
                    facets=FacetConfig(spacing=64.0), media=aux["media"],
                    interfaces=[DemInterface(name="surface"),
                                DemInterface(name="bed")])
    t_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        ds = simulate(mscene, cfg)
    wall = time.perf_counter() - t_start
    fired = [str(w.message)[:100] for w in wl]
    alias_warned = any("alias" in m for m in fired)
    print(f"[{tag}] wall {wall:.1f} s; alias warning fired: {alias_warned}")
    for m in fired:
        print(f"    warn: {m}")
    return ds


def quiet_floor(comb_frame_grid, tag):
    """Median combined power in the quiet band (surface+1.5..3.5 us, clipped
    1 us above the bed pick), dB rel the per-trace surface peak."""
    p = np.asarray(comb_frame_grid, np.float64)
    vals = []
    for t in range(p.shape[0]):
        if not (np.isfinite(surf_pick[t]) and np.isfinite(bot_pick[t])):
            continue
        lo = surf_pick[t] + 1.5e-6
        hi = min(surf_pick[t] + 3.5e-6, bot_pick[t] - 1.0e-6)
        if hi <= lo:
            continue
        b0, b1 = int((lo - t0) / dt), int((hi - t0) / dt)
        pk = p[t, : int((surf_pick[t] - t0) / dt) + 40].max()
        band = p[t, b0:b1]
        band = band[band > 0]
        if band.size and pk > 0:
            vals.append(10 * np.log10(np.median(band) / pk))
    v = np.array(vals)
    print(f"[{tag}] quiet-band floor rel surface peak: median "
          f"{np.median(v):.1f} dB (p10 {np.percentile(v,10):.1f}, "
          f"p90 {np.percentile(v,90):.1f}) over {len(v)} traces")
    return v


from soundersim.output import combine  # noqa: E402

ds_a = run(dt, n, "A native dt")
va = quiet_floor(np.asarray(combine(ds_a, "layer"), np.float64), "A native dt")

n_b = 4 * (n - 1) + 1
ds_b = run(dt / 4, n_b, "B dt/4")
comb_b = np.asarray(combine(ds_b, "layer"), np.float64)[:, ::4]
assert comb_b.shape[1] == n
vb = quiet_floor(comb_b, "B dt/4 (decimated)")

print(f"\nfloor difference (A - B) median: "
      f"{np.median(va) - np.median(vb):+.1f} dB")
