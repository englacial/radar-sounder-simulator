"""Firn power-plateau investigation, MCoRDS-matched chirp redo (M19 follow-up).

This is an honest-observation experiment, NOT a pass/fail gate. It builds flat
multi-layer coherent scenes from the B26 firn density core and records the
Culberg & Schroeder (2020) plateau DIAGNOSTICS (gradient intervals, secondary
maximum), never gating on them. This second sweep repeats the DELTA-pulse
sweep (now archived in ``outputs/firn_investigation/delta_runs/``) with the
MCoRDS 2017 P-3 waveform (claude_notes/m24_findings.md provenance):

  * waveform: chirp, B = 30 MHz, hann compression window, pulse 3 us
    (compressed shape is B+window-determined; Tpd only sets the truncation).
    Chirped range resolution: hann 1.44*c/(2B) = 7.2 m in air,
    7.2/sqrt(eps_mean) ~ 4.4 m in firn -- N >= 40 stacks (mean spacing
    <= 3 m) are now FULLY UNRESOLVED, the paper's regime direction.
  * alias-free fast-time grid (THE ALIAS RULE, claude_notes/m20_m21_findings):
    the binned trace quantizes envelope delays to dt, planting quantization
    noise at the aliased carrier f_a = f0 - round(f0*dt)/dt; require
    |f_a| > B/2. The old firn dt = 5 ns puts f_a = -5 MHz IN band; dt = 4 ns
    gives f0*dt = 0.78 -> round to 1 -> f_a = 195 - 250 = -55 MHz, |f_a| =
    55 MHz > 15 MHz. n_samples 512 -> 640 keeps the same 2560 ns twtt window.
    Every simulate() call is checked: the in-band-alias warning must NOT fire.
    interp_bins stays off (unsupported for multilayer; unnecessary here).
  * surface-subtracted profile: the kernel is single-bounce LINEAR, so the
    total coherent field = surface field + layer fields exactly (convolution
    is linear too). |E_total - E_surface|^2 isolates the layer signal from
    the surface's own off-nadir response, which the delta sweep found
    contaminating the upper ~45 m. E_surface is the run's OWN layer-0 field
    (== the surface-only reference field up to the scalar surface-gamma
    ratio when the top-medium eps differs; the ratio-scaled agreement is
    recorded per run as ``surface_field_check_rel``).
  * complex fields (complex64, 3 traces) are saved per run, incl. the
    surface-only reference.

Scene design (unchanged from the delta sweep):
  * flat surface, 500 m AGL, 195 MHz, 600 m extent, 4 m facets, 3 traces,
    plus N flat offset layers below it.
  * eps convention (POINT sampling): the firn slab immediately below
    interface k (depth d_k) takes the permittivity of the CLOSEST
    0.1 m-smoothed B26 sample to d_k, via Kovacs et al. (1993)
    eps = (1 + 0.845*rho[g/cc])^2. Substrate: eps(d_N + 1 m).
  * placements: equal spacing and uniformly-random (sorted, min sep ~0.25 m,
    seeded, same 3 seeds), spanning ~1 m to the core end (~119.7 m).
  * layer counts N in {10, 20, 40, 80}; surface-only reference run.

Runtime: kernel compile scales ~O(N^2); runs of a given N are batched in one
process so compiled graphs are reused. Resumable: a run whose diagnostics
json exists is skipped (the dt/waveform change means none of the delta-sweep
outputs qualify -- they live in delta_runs/). Cutoff: 30 min wall per
simulate() call (compile-inclusive); on breach the remaining runs of that N
are skipped and the skip is recorded loudly.

Run: uv run python tools/run_firn_investigation.py         # full sweep + report
     uv run python tools/run_firn_investigation.py --report-only
"""

import argparse
import base64
import html
import json
import time
import warnings as _warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import soundersim  # noqa: E402
from soundersim.config import (DemInterface, FacetConfig, Medium,  # noqa: E402
                               OffsetInterface, RadarConfig, SimConfig,
                               WaveformConfig)
from soundersim.physics import C, fresnel_normal  # noqa: E402
from soundersim import synthetic as syn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXDIR = ROOT / "tests" / "fixtures" / "firn"
OUTDIR = ROOT / "outputs" / "firn_investigation"

# --- scene geometry (old test_firn_plateau, verbatim) ----------------------
H, ELEV, EXTENT = 500.0, 500.0, 600.0     # platform AGL, surface elev, scene (m)
F0 = 195e6                                # MCoRDS3 carrier (180-210 MHz chirp)
T0 = 2.0 * (H - 10.0) / C
POSTING, SPACING, NTRACES = 4.0, 4.0, 3

# --- waveform / fast-time grid (MCoRDS-matched, alias-free) -----------------
BANDWIDTH = 30e6                          # param_records.radar.wfs f1-f0
PULSE_LEN = 3e-6                          # ANT bed waveform Tpd
WINDOW = "hann"                           # param_csarp.csarp.ft_wind
DT, NSAMP = 4e-9, 640                     # 512 x 5 ns -> 640 x 4 ns, same window
F_ALIAS = abs(F0 - round(F0 * DT) / DT)   # = 55 MHz > B/2 = 15 MHz (alias rule)
assert F_ALIAS > BANDWIDTH / 2.0, f"in-band alias: {F_ALIAS/1e6:.1f} MHz"

# --- placement / analysis constants -----------------------------------------
Z_TOP = 1.0                               # shallowest layer depth (m)
MIN_SEP = 0.25                            # random min layer separation (m)
N_SEEDS = 3
SMOOTH_M = 5.0                            # depth-power smoothing window (m)
GRAD_THRESHOLDS = (0.0, -0.05, -0.1)      # dB/m; -0.05 is the reported "near-zero"
GRAD_PRIMARY = -0.05
SURF_EXCL_M = 5.0                         # exclude surface lobe from secondary-max
LAYER_COUNTS = (10, 20, 40, 80)
CUTOFF_S = 30 * 60                        # per-simulation wall cutoff (compile-incl.)


# ========================================================================
# B26 density core -> permittivity
# ========================================================================
def load_b26(smooth_m=0.1):
    """B26 density (depth m, rho kg/m^3), lightly 0.1 m-smoothed (PANGAEA tab).

    Edge-normalized boxcar: the moving average is divided by the local window
    overlap rather than the full width. A plain ``mode='same'`` convolution
    zero-pads beyond the core and HALVES the deepest ~0.05 m of density
    (882 -> 446 kg/m^3 -> eps 3.05 -> 1.90), which the point-sampled deepest
    layer lands on and reads as a spurious bright deep reflector. Edge
    normalization keeps the deepest sample physical (eps 3.05) and is identical
    to the plain average in the interior.
    """
    path = FIXDIR / "ngt37C95.2_density.tab"
    lines = path.read_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Depth ice/snow"))
    data = np.loadtxt(path, delimiter="\t", skiprows=hdr + 1)
    z, rho = data[:, 0], data[:, 1]
    k = int(round(smooth_m / np.median(np.diff(z)))) | 1
    box = np.ones(k)
    return z, np.convolve(rho, box, "same") / np.convolve(np.ones_like(rho),
                                                          box, "same")


def eps_kovacs(rho_kgm3):
    """Kovacs et al. (1993) / C&S 2020 Eq. (4): eps = (1 + 0.845*rho[g/cc])^2."""
    return (1.0 + 0.845 * np.asarray(rho_kgm3) / 1000.0) ** 2


Z, RHO = load_b26()
EPS = eps_kovacs(RHO)
ZMAX = float(Z.max())
EPS_MEAN = float(EPS.mean())
RES_AIR_M = 1.44 * C / (2.0 * BANDWIDTH)          # hann-broadened resolution
RES_FIRN_M = RES_AIR_M / np.sqrt(EPS_MEAN)


def point_eps(depth):
    """Closest 0.1 m-smoothed B26 sample permittivity at ``depth`` (point sample)."""
    return float(EPS[np.argmin(np.abs(Z - depth))])


def fullres_adjacent_gamma_db():
    """Context stat: |Fresnel| between adjacent (1 mm) full-res samples, dB."""
    g = np.abs(fresnel_normal(EPS[:-1], EPS[1:]))
    gdb = 20.0 * np.log10(np.maximum(g, 1e-30))
    return dict(median=float(np.median(gdb)), p90=float(np.percentile(gdb, 90)),
                max=float(gdb.max()), spacing_m=float(np.median(np.diff(Z))))


# ========================================================================
# layer placement
# ========================================================================
def equal_depths(n):
    return np.linspace(Z_TOP, ZMAX, n)


def random_depths(n, seed):
    """n sorted depths in [Z_TOP, ZMAX] with min separation MIN_SEP, seeded."""
    rng = np.random.default_rng((int(seed), int(n)))
    slack = (ZMAX - Z_TOP) - (n - 1) * MIN_SEP
    if slack <= 0:
        raise ValueError(f"N={n} too many for min sep {MIN_SEP}")
    u = np.sort(rng.uniform(0.0, slack, n))
    return Z_TOP + u + np.arange(n) * MIN_SEP


# ========================================================================
# scene / config
# ========================================================================
def _scene():
    return syn.flat_scene(elevation=ELEV, altitude=H, extent=EXTENT,
                          posting=POSTING, n_traces=NTRACES)


def _radar():
    return RadarConfig(dt=DT, n_samples=NSAMP, t0=T0, f0=F0,
                       waveform=WaveformConfig(kind="chirp",
                                               bandwidth=BANDWIDTH,
                                               pulse_length=PULSE_LEN,
                                               window=WINDOW))


def layered_cfg(depths):
    """Coherent multi-layer config for the given (sorted) layer depths."""
    media = [Medium(name="air", eps_r=1.0)]
    ifaces = [DemInterface(name="surface")]
    for i, d in enumerate(depths):
        media.append(Medium(name=f"firn{i}", eps_r=point_eps(d)))
        ifaces.append(OffsetInterface(name=f"L{i}", reference="surface",
                                      offset=-float(d)))
    media.append(Medium(name="substrate", eps_r=point_eps(depths[-1] + 1.0)))
    return SimConfig(mode="coherent", media=media, interfaces=ifaces,
                     radar=_radar(), facets=FacetConfig(spacing=SPACING))


def reference_cfg():
    """Surface-only single-interface coherent config (air over eps(d=Z_TOP))."""
    media = [Medium(name="air", eps_r=1.0),
             Medium(name="firn0", eps_r=point_eps(Z_TOP))]
    return SimConfig(mode="coherent", media=media,
                     interfaces=[DemInterface(name="surface")],
                     radar=_radar(), facets=FacetConfig(spacing=SPACING))


def _simulate_checked(scene, cfg):
    """simulate() with the in-band-alias warning asserted SILENT (alias rule)."""
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        ds = soundersim.simulate(scene, cfg)
    for w in caught:
        if "alias" in str(w.message):
            raise RuntimeError(f"in-band-alias warning fired: {w.message}")
        print(f"  [warn] {w.message}", flush=True)
    return ds


# ========================================================================
# observable + diagnostics
# ========================================================================
def _smooth(p):
    """~SMOOTH_M boxcar on linear power; returns (smoothed, effective_m)."""
    bin_depth = C * DT / (2.0 * np.sqrt(EPS_MEAN))          # ~0.37 m/bin
    w = max(int(round(SMOOTH_M / bin_depth)) | 1, 3)
    return np.convolve(p, np.ones(w) / w, mode="same"), float(w * bin_depth)


def _depth_axis(ds, depths):
    """twtt -> depth mapping and the deep window edge (see delta-sweep note)."""
    twtt = ds.twtt.values
    if depths is not None:
        node_twtt = ds.nadir_twtt.mean("slow_time").values  # (layer,)
        node_depth = np.concatenate([[0.0], depths])
        return np.interp(twtt, node_twtt, node_depth), float(depths[-1])
    surf_twtt = float(ds.nadir_twtt.mean())
    depth = np.maximum((twtt - surf_twtt) * C / (2.0 * np.sqrt(EPS_MEAN)), 0.0)
    return depth, ZMAX


def depth_power_profiles(ds, depths):
    """Raw + surface-subtracted trace power vs depth (dB rel. raw surface peak).

    Raw: layer-summed coherent field, |.|^2, trace mean, smoothed. Subtracted:
    |E_total - E_surface|^2 with E_surface the run's own layer-0 field
    (exact: the kernel and the pulse convolution are linear), same smoothing,
    normalized by the SAME raw surface peak so levels are comparable.
    ``depths`` None (surface-only reference) yields sub = None.

    Returns dict(depth, raw_db, sub_db, grad_raw, grad_sub, eff_win) restricted
    to the in-layer window (0, deepest layer).
    """
    fld = ds.field
    if "layer" in fld.dims:
        f = fld.values                                     # (trace, samp, layer)
        tot, sub = f.sum(axis=-1), f[..., 1:].sum(axis=-1)
    else:
        tot, sub = fld.values, None

    ps_raw, eff_win = _smooth((np.abs(tot) ** 2).mean(axis=0))
    peak = ps_raw.max()
    raw_db = 10.0 * np.log10(np.maximum(ps_raw / peak, 1e-12))
    sub_db = None
    if sub is not None:
        ps_sub, _ = _smooth((np.abs(sub) ** 2).mean(axis=0))
        sub_db = 10.0 * np.log10(np.maximum(ps_sub / peak, 1e-12))

    depth, z_deep = _depth_axis(ds, depths)
    m = (depth > 0.0) & (depth < z_deep)
    d = depth[m]
    order = np.argsort(d)
    keep = np.concatenate([[True], np.diff(d[order]) > 1e-6])
    sel = np.flatnonzero(m)[order][keep]
    d = depth[sel]
    out = dict(depth=d, raw_db=raw_db[sel], eff_win=eff_win,
               grad_raw=np.gradient(raw_db[sel], d),
               sub_db=None, grad_sub=None)
    if sub_db is not None:
        out["sub_db"] = sub_db[sel]
        out["grad_sub"] = np.gradient(sub_db[sel], d)
    return out


def longest_nonneg_interval(depth, grad, thresh):
    """Longest contiguous depth span where grad >= thresh; (length_m, z0, z1)."""
    ok = grad >= thresh
    best = (0.0, None, None)
    i = 0
    n = len(ok)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        length = float(depth[j] - depth[i])
        if length > best[0]:
            best = (length, float(depth[i]), float(depth[j]))
        i = j + 1
    return best


def secondary_max(depth, prof_db):
    """Highest smoothed level (dB rel. raw surface) below the surface lobe."""
    m = depth >= SURF_EXCL_M
    if not m.any():
        return None, None
    idx = np.argmax(prof_db[m])
    return float(prof_db[m][idx]), float(depth[m][idx])


def realized_gamma_db(depths):
    """|Fresnel| in dB at each layer interface (excludes surface); median/p90."""
    eps = [1.0] + [point_eps(d) for d in depths] + [point_eps(depths[-1] + 1.0)]
    eps = np.asarray(eps)
    g = np.abs(fresnel_normal(eps[:-1], eps[1:]))
    gdb = 20.0 * np.log10(np.maximum(g, 1e-30))
    surf_db = float(gdb[0])
    layer_db = gdb[1:]                                      # interfaces below surface
    return dict(median=float(np.median(layer_db)),
                p90=float(np.percentile(layer_db, 90)),
                min=float(layer_db.min()), max=float(layer_db.max()),
                surface_db=surf_db)


def _plateau_block(depth, grad):
    """Gradient-interval diagnostics at all thresholds for one profile."""
    return {f"{t:+.2f}": dict(zip(("length_m", "z0_m", "z1_m"),
                                  longest_nonneg_interval(depth, grad, t)))
            for t in GRAD_THRESHOLDS}


def diagnose(ds, depths, placement, n, seed, wall_s, first_per_n, ref_field):
    prof = depth_power_profiles(ds, depths)
    d = prof["depth"]
    gi_raw = _plateau_block(d, prof["grad_raw"])
    gi_sub = _plateau_block(d, prof["grad_sub"])
    sec_db, sec_z = secondary_max(d, prof["raw_db"])
    sec_db_s, sec_z_s = secondary_max(d, prof["sub_db"])

    # cross-check: layer-0 field == surface-only reference field scaled by the
    # surface-gamma ratio (top-medium eps may differ for random placements)
    fld = ds.field.values
    ratio = (fresnel_normal(1.0, point_eps(depths[0]))
             / fresnel_normal(1.0, point_eps(Z_TOP)))
    check = float(np.abs(fld[..., 0] - ratio * ref_field).max()
                  / np.abs(ref_field).max())

    diag = {
        "placement": placement, "n": n, "seed": seed,
        "wall_s": round(wall_s, 2), "first_per_n": first_per_n,
        "waveform": {"kind": "chirp", "bandwidth_hz": BANDWIDTH,
                     "pulse_length_s": PULSE_LEN, "window": WINDOW},
        "dt_s": DT, "n_samples": NSAMP,
        "f_alias_mhz": round(F_ALIAS / 1e6, 3), "alias_warning_fired": False,
        "range_res_air_m": round(RES_AIR_M, 3),
        "range_res_firn_m": round(float(RES_FIRN_M), 3),
        "smooth_window_m": SMOOTH_M,
        "effective_window_m": round(prof["eff_win"], 3),
        "grad_primary_threshold_db_per_m": GRAD_PRIMARY,
        "grad_interval_primary": gi_raw[f"{GRAD_PRIMARY:+.2f}"],
        "grad_interval_sensitivity": gi_raw,
        "grad_interval_primary_sub": gi_sub[f"{GRAD_PRIMARY:+.2f}"],
        "grad_interval_sensitivity_sub": gi_sub,
        "secondary_max_db": sec_db, "secondary_max_depth_m": sec_z,
        "secondary_max_db_sub": sec_db_s, "secondary_max_depth_m_sub": sec_z_s,
        "surface_field_check_rel": check,
        "realized_gamma_db": realized_gamma_db(depths),
        "n_depth_samples": int(len(d)),
        "mean_layer_spacing_m": float(np.mean(np.diff(np.concatenate(
            [[0.0], depths])))),
    }
    return diag, prof


# ========================================================================
# sweep
# ========================================================================
def run_id(placement, n, seed):
    return f"{placement}_N{n}" + (f"_s{seed}" if seed is not None else "")


def _paths(rid):
    return OUTDIR / "runs" / f"{rid}.json", OUTDIR / "runs" / f"{rid}.npz"


def _do_run(scene, cfg, depths, placement, n, seed, first_per_n, ref_field):
    rid = run_id(placement, n, seed)
    jpath, npath = _paths(rid)
    if jpath.exists():
        print(f"  [skip-exists] {rid}", flush=True)
        return json.loads(jpath.read_text()), False
    t = time.perf_counter()
    ds = _simulate_checked(scene, cfg)
    wall = time.perf_counter() - t
    diag, prof = diagnose(ds, depths, placement, n, seed, wall, first_per_n,
                          ref_field)
    for arr in (prof["depth"], prof["raw_db"], prof["sub_db"]):
        if not np.isfinite(arr).all():
            raise ValueError(f"{rid}: non-finite profile")
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(diag, indent=1) + "\n")
    np.savez(npath, depth=prof["depth"], prof_db=prof["raw_db"],
             prof_sub_db=prof["sub_db"], twtt=ds.twtt.values,
             layer_depths=np.asarray(depths),
             field=ds.field.values.astype(np.complex64))
    kind = "first/compile" if first_per_n else "cached"
    print(f"  [ok] {rid}  {wall:6.1f} s ({kind})  "
          f"plateau(-0.05) raw={diag['grad_interval_primary']['length_m']:.1f} "
          f"sub={diag['grad_interval_primary_sub']['length_m']:.1f} m  "
          f"sec_max raw={diag['secondary_max_db']:.1f} "
          f"sub={diag['secondary_max_db_sub']:.1f} dB", flush=True)
    return diag, wall > CUTOFF_S


def _run_reference(scene):
    """Surface-only reference run; returns its complex field (traces, samples)."""
    jpath, npath = _paths("reference")
    if jpath.exists():
        print("  [skip-exists] reference", flush=True)
        return np.load(npath)["field"]
    t = time.perf_counter()
    ds = _simulate_checked(scene, reference_cfg())
    wall = time.perf_counter() - t
    prof = depth_power_profiles(ds, None)
    field = ds.field.values.astype(np.complex64)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(
        {"placement": "reference", "wall_s": round(wall, 2),
         "dt_s": DT, "n_samples": NSAMP,
         "f_alias_mhz": round(F_ALIAS / 1e6, 3), "alias_warning_fired": False,
         "waveform": {"kind": "chirp", "bandwidth_hz": BANDWIDTH,
                      "pulse_length_s": PULSE_LEN, "window": WINDOW},
         "effective_window_m": round(prof["eff_win"], 3)}, indent=1) + "\n")
    np.savez(npath, depth=prof["depth"], prof_db=prof["raw_db"],
             twtt=ds.twtt.values, field=field)
    print(f"  [ok] reference  {wall:5.1f} s", flush=True)
    return field


def run_sweep(layer_counts=LAYER_COUNTS, n_seeds=N_SEEDS):
    (OUTDIR / "runs").mkdir(parents=True, exist_ok=True)
    print(f"chirp B={BANDWIDTH/1e6:.0f} MHz {WINDOW} Tpd={PULSE_LEN*1e6:.0f} us; "
          f"dt={DT*1e9:.0f} ns n={NSAMP}; |f_a|={F_ALIAS/1e6:.0f} MHz "
          f"(> B/2 = {BANDWIDTH/2e6:.0f} MHz, alias-free)", flush=True)
    scene = _scene()
    skips = []
    ref_field = _run_reference(scene)

    for n in sorted(layer_counts):                # ascending: share compiled graphs
        first_done = False
        breached = False
        runs = [("equal", equal_depths(n), None)]
        runs += [("random", random_depths(n, s), s) for s in range(n_seeds)]
        for placement, depths, seed in runs:
            rid = run_id(placement, n, seed)
            if breached:
                print(f"  [SKIP-CUTOFF] {rid}: prior {n}-layer call breached "
                      "the wall cutoff", flush=True)
                skips.append(rid)
                continue
            first = not first_done
            _, over = _do_run(scene, layered_cfg(depths), depths, placement, n,
                              seed, first, ref_field)
            first_done = True
            if over:
                print(f"  [CUTOFF] {rid} exceeded {CUTOFF_S/60:.0f} min; "
                      f"skipping remaining N={n} runs", flush=True)
                breached = True
    if skips:
        (OUTDIR / "skipped.json").write_text(json.dumps(skips, indent=1) + "\n")
    return skips


# ========================================================================
# report
# ========================================================================
FINDINGS_NARRATIVE = """\
<b>Is the plateau there now? Yes -- the paper's operational criterion is met.</b>
With the MCoRDS-matched chirp all 16 layered runs plus the reference completed
(no cutoff skips; the previously-skipped random N=80 seeds ran in ~31 s each on
the batched compiled kernel). The longest contiguous interval with
smoothed-trace gradient &ge; -0.05 dB/m exceeds the paper's 10 m in 6/16 runs
on the RAW profile (up to 18.0 m, random N=80 s1 at 42-60 m depth; also 13.3 m
at 30-43 m for s2 and 10.3-11.4 m elsewhere) and in 11/16 runs on the
SURFACE-SUBTRACTED profile (10.3-18.7 m; every random N=20 and N=40 run, three
of four N=80 runs). In the delta sweep NO run met the criterion (best 9.6 m).
The subtracted intervals at N=20-80 often start essentially at the surface
(e.g. [0.1, 16-18 m]) -- the paper's near-surface plateau -- while the N=80
raw intervals sit mid-firn (30-60 m). Threshold sensitivity (0 / -0.1 dB/m)
changes the long intervals by &lt; 1 m. (Caution: equal-placement raw
intervals at 99-117 m ride the flattened deep tail of a periodic stack;
random placement is the physical case.)

<b>Which change mattered?</b> All three, in different depth bands.
(1) <i>Alias-free chirp (dt 5 -> 4 ns, |f_a| 5 -> 55 MHz):</i> the
surface-only reference floor fell 6-17 dB over 5-45 m apparent depth (5-10 m:
-16.8 -> -23.3 dB; 10-20 m: -17.3 -> -34.3; 20-45 m: -24.7 -> -36.5). Most of
the delta sweep's "upper ~45 m surface contamination" was the in-band
envelope-quantization alias, not physical off-nadir response: with it gone the
layered profiles sit 10-20 dB above the reference beyond ~10 m, so raw and
subtracted profiles agree there. (2) <i>Pulse integration:</i> the 4.4 m
in-firn hann pulse integrates several interfaces per resolution cell; deep
bands rise 2-6 dB over the delta profiles (equal N=80, 100-119 m: -40.4 ->
-34.1 dB), flattening the apparent decay -- part of why gradients now clear
the threshold. (3) <i>Surface-field subtraction:</i> decisive only in the top
~15 m, where the surface main lobe lives: the raw secondary maximum is pinned
at ~5 m depth (-4.0..-8.8 dB, the surface-lobe shoulder) in every run, while
the subtracted profiles remove it exactly (random N=20 s0, 5-10 m band:
-20.2 -> -59.0 dB where that stack has no shallow layers) and expose the
near-surface plateaus. (4) <i>N=80 random (newly completed):</i> supplies the
two longest RAW plateaus -- the criterion is met without subtraction only at
N=80.

<b>Secondary maximum.</b> On the subtracted profiles it lands at
-10.2..-23.0 dB relative to the surface peak (median ~ -17.5 dB) at 5-65 m
depth -- the paper's "typically 10-15 dB below the surface" regime, slightly
weak on average. One outlier: random N=10 s2 has real layers at 3.98/4.34 m
whose merged echo reaches -1.4 dB. The raw secondary maxima are all the ~5 m
surface shoulder and are not layer diagnostics.

<b>Resolved-to-unresolved transition.</b> Mean layer spacing 12.0/6.0/3.0/
1.5 m (N=10/20/40/80) against the 4.4 m in-firn chirped resolution: N=10/20
are resolved (echo trains, visible in the panels), N&ge;40 fully unresolved
(continuous profiles). The criterion tracks it: subtracted pass rate 1/4 at
N=10 vs 10/12 at N&ge;20; raw passes concentrate at N=80. The deep floor is
still rising at N=80 (50-100 m random-mean: about -31 dB at N=10-40 ->
-27 dB at N=80; 100-119 m: -46 (N=20) -> -36 dB) -- band levels move 3-6 dB
per doubling of N, i.e. NOT converged at N=80. Realized per-interface
|gamma| medians (-40.5 to -52.9 dB, N=10 -> 80) remain ~30-40 dB above the
full-resolution 1 mm adjacent-sample statistics: an 80-layer point-sampled
stack is still far from a converged discretisation of the continuous core.

<b>Joint-solve (D+) trigger verdict.</b> The trigger's first condition --
"the firn plateau investigation shows the phenomenon emerging with layer
count" -- is met: plateau length and pass rate grow with N, the raw-profile
criterion is first met at N=80, and nothing has converged by N=80 while the
sequential-chain compile wall (O(N^2): 26.6 min at N=80) blocks the N ~
150-300 runs a convergence study needs. The nuance: the operational plateau
no longer NEEDS larger N -- chirp + subtraction demonstrate it at feasible
layer counts -- so D+ should be scheduled when the CONVERGED absolute levels
and fine-layer statistics (comparison against a 1-D transfer-matrix referee,
C&amp;S-style) matter for instrument work, not for plateau morphology alone.

<b>Rigor notes.</b> Every simulate() call ran with the in-band-alias warning
asserted silent (|f_a| = 55 MHz &gt; B/2 = 15 MHz at dt = 4 ns). The
surface-subtraction identity was cross-checked per run: the run's layer-0
field equals the surface-only reference field scaled by the surface-gamma
ratio to &le; 1.4e-6 of the reference peak (exactly 0 for equal placements,
whose top-medium eps matches the reference). Wall times: first-per-N
(compile-inclusive) 14.0 / 69.8 / 318.6 / 1593.5 s for N=10/20/40/80, cached
same-N calls 0.6 / 2.1 / 8.0 / 31.2 s -- all under the 30 min per-simulation
cutoff."""


def _load_runs(runs_dir=None):
    runs = {}
    for jp in sorted((runs_dir or OUTDIR / "runs").glob("*.json")):
        rid = jp.stem
        npz = jp.with_suffix(".npz")
        if not npz.exists():
            continue
        arr = np.load(npz, allow_pickle=True)
        sub = arr["prof_sub_db"] if "prof_sub_db" in arr.files else None
        if sub is not None and sub.ndim == 0:
            sub = None
        runs[rid] = (json.loads(jp.read_text()), arr["depth"], arr["prof_db"],
                     sub)
    return runs


def _panel(ax, runs, keys, ref, title):
    for i, k in enumerate(keys):
        if k not in runs:
            continue
        diag, d, r, s = runs[k]
        lbl = "equal" if diag.get("seed") is None else f"seed {diag['seed']}"
        ax.plot(d, r, lw=1.0, color=f"C{i}", label=f"{lbl} raw")
        if s is not None:
            ax.plot(d, s, lw=0.9, color=f"C{i}", ls="--", alpha=0.85,
                    label=f"{lbl} −surf")
    if ref is not None:
        ax.plot(ref[1], ref[2], "k:", lw=1.0, alpha=0.7, label="surface only")
    ax.set_xlim(0, 120)
    ax.set_ylim(-70, 3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("depth (m)")
    ax.set_ylabel("power (dB rel. surface)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="lower left", ncol=2)


def _figure_grid(runs, path):
    ref = runs.get("reference")
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    axes = axes.ravel()
    panels = [(p, n) for p in ("equal", "random") for n in LAYER_COUNTS]
    for ax, (placement, n) in zip(axes[:8], panels):
        keys = ([f"equal_N{n}"] if placement == "equal"
                else [f"random_N{n}_s{s}" for s in range(N_SEEDS)])
        _panel(ax, runs, keys, ref, f"{placement}, N={n}")

    ax = axes[8]
    for i, n in enumerate(LAYER_COUNTS):
        k = f"equal_N{n}"
        if k in runs:
            _, d, r, s = runs[k]
            ax.plot(d, r, lw=1.1, color=f"C{i}", label=f"N={n} raw")
            if s is not None:
                ax.plot(d, s, lw=0.9, color=f"C{i}", ls="--", alpha=0.85,
                        label=f"N={n} −surf")
    if ref is not None:
        ax.plot(ref[1], ref[2], "k:", lw=1.0, alpha=0.7, label="surface only")
    ax.set_xlim(0, 120)
    ax.set_ylim(-70, 3)
    ax.set_title("Summary: equal placement across N", fontsize=10)
    ax.set_xlabel("depth (m)")
    ax.set_ylabel("power (dB rel. surface)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="lower left", ncol=2)
    fig.suptitle("Firn power plateau, MCoRDS-matched chirp (30 MHz hann) -- "
                 "B26 core, 195 MHz, 500 m AGL, point-sampled eps; dashed = "
                 "surface-field-subtracted", fontsize=12)
    fig.savefig(path, dpi=95)
    plt.close(fig)


def _figure_delta_vs_chirp(runs, path, key="equal_N40"):
    """Old delta-pulse sweep (dt=5 ns) vs this chirped sweep, same scene."""
    old_dir = OUTDIR / "delta_runs"
    if not (old_dir / f"{key}.npz").exists() or key not in runs:
        return None
    old = np.load(old_dir / f"{key}.npz")
    old_ref = (np.load(old_dir / "reference.npz")
               if (old_dir / "reference.npz").exists() else None)
    _, d, r, s = runs[key]
    ref = runs.get("reference")
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(old["depth"], old["prof_db"], color="C3", lw=1.1,
            label="delta pulse, dt=5 ns (in-band alias) raw")
    if old_ref is not None:
        ax.plot(old_ref["depth"], old_ref["prof_db"], color="C3", ls=":",
                lw=0.9, alpha=0.7, label="delta surface only")
    ax.plot(d, r, color="C0", lw=1.2, label="chirp 30 MHz hann, dt=4 ns raw")
    if s is not None:
        ax.plot(d, s, color="C0", ls="--", lw=1.0,
                label="chirp − surface field")
    if ref is not None:
        ax.plot(ref[1], ref[2], "k:", lw=1.0, alpha=0.7,
                label="chirp surface only")
    ax.set_xlim(0, 120)
    ax.set_ylim(-70, 3)
    ax.set_xlabel("depth (m)")
    ax.set_ylabel("power (dB rel. surface)")
    ax.set_title(f"Delta (archived sweep) vs MCoRDS-matched chirp -- {key}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(path, dpi=95)
    plt.close(fig)
    return path


def _fmt_interval(gi):
    if gi["z0_m"] is None:
        return "-"
    return f"{gi['length_m']:.1f} m @ [{gi['z0_m']:.0f}, {gi['z1_m']:.0f}]"


def _diag_table(runs):
    hdr = ("<tr><th>run</th><th>N</th><th>spacing (m)</th>"
           "<th>plateau raw (-0.05 dB/m)</th><th>plateau &minus;surf</th>"
           "<th>sec. max raw</th><th>sec. max &minus;surf</th>"
           "<th>|&gamma;| med / p90 (dB)</th><th>wall (s)</th></tr>")
    rows = []
    for rid in sorted(k for k in runs if k != "reference"):
        d = runs[rid][0]
        g = d["realized_gamma_db"]
        rows.append(
            f"<tr><td>{html.escape(rid)}</td><td>{d['n']}</td>"
            f"<td>{d['mean_layer_spacing_m']:.2f}</td>"
            f"<td>{_fmt_interval(d['grad_interval_primary'])}</td>"
            f"<td>{_fmt_interval(d['grad_interval_primary_sub'])}</td>"
            f"<td>{d['secondary_max_db']:.1f} dB @ {d['secondary_max_depth_m']:.0f} m</td>"
            f"<td>{d['secondary_max_db_sub']:.1f} dB @ {d['secondary_max_depth_m_sub']:.0f} m</td>"
            f"<td>{g['median']:.1f} / {g['p90']:.1f}</td>"
            f"<td>{d['wall_s']:.1f}</td></tr>")
    return f"<table>{hdr}{''.join(rows)}</table>"


def _img_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def build_report():
    runs = _load_runs()
    if not runs:
        raise SystemExit("no runs found; run the sweep first")
    grid = OUTDIR / "figure_grid.png"
    _figure_grid(runs, grid)
    cmp_png = _figure_delta_vs_chirp(runs, OUTDIR / "figure_delta_vs_chirp.png")
    fullres = fullres_adjacent_gamma_db()
    skips = []
    sp = OUTDIR / "skipped.json"
    if sp.exists():
        skips = json.loads(sp.read_text())

    css = ("body{font-family:system-ui,Arial,sans-serif;margin:2rem;max-width:1200px;"
           "color:#1a1a1a}h1{margin-bottom:.2rem}table{border-collapse:collapse;"
           "margin:1rem 0;font-size:.85rem}th,td{border:1px solid #ccc;padding:.3rem .5rem;"
           "text-align:left}th{background:#f0f0f0}img{max-width:100%;border:1px solid #ddd}"
           ".note{color:#333;background:#f6f6f6;border-left:3px solid #bbb;padding:.6rem 1rem;"
           "margin:1rem 0}.warn{background:#fff3cd;border:1px solid #ffe08a;padding:.5rem;"
           "border-radius:4px}code{background:#eee;padding:0 .2rem}")
    warn = (f'<div class="warn"><b>Runs skipped on wall-cutoff:</b> '
            f'{", ".join(skips)}</div>' if skips else "")
    cmp_html = ""
    if cmp_png:
        cmp_html = (f"<h2>Delta-pulse sweep vs MCoRDS-matched chirp</h2>"
                    f"<img src='data:image/png;base64,{_img_b64(cmp_png)}' "
                    f"alt='delta vs chirp'>"
                    "<p class='note'>The archived delta sweep (in "
                    "<code>delta_runs/</code>) used dt = 5 ns, which places the "
                    "envelope-quantization alias at |f_a| = 5 MHz -- IN the "
                    "30 MHz band; the chirped sweep uses dt = 4 ns "
                    "(|f_a| = 55 MHz &gt; B/2 = 15 MHz), so the compressed pulse "
                    "rejects the quantization noise entirely.</p>")
    narrative = FINDINGS_NARRATIVE.replace("\n\n", "</p><p>")
    body = f"""
<h1>Firn power plateau investigation &mdash; MCoRDS-matched chirp</h1>
<p class="note"><b>Exploratory experiment, not a gate.</b> Records the
Culberg &amp; Schroeder (2020) plateau diagnostics on point-sampled B26 firn
stacks, now with the MCoRDS 2017 P-3 waveform (chirp B = 30 MHz, hann
compression window, Tpd = 3 &mu;s; provenance
<code>outputs/cache/mcords_2017P3_params.json</code>). The paper's plateau
definition is operational: near-zero/nonnegative smoothed-trace gradient over
depth intervals &gt; 10 m, a near-surface secondary maximum typically
10&ndash;15 dB below the surface return, sharp decay below ~60 m. The previous
DELTA-pulse sweep is archived in <code>delta_runs/</code>.</p>
<p class="note"><b>Waveform / grid.</b> dt = {DT*1e9:.0f} ns, n = {NSAMP}
(same 2560 ns twtt window as the delta sweep's 512 &times; 5 ns). Alias rule:
|f0 &minus; round(f0&middot;dt)/dt| = {F_ALIAS/1e6:.0f} MHz &gt; B/2 =
{BANDWIDTH/2e6:.0f} MHz &mdash; the in-band-alias warning is asserted silent on
every run. Chirped range resolution (hann, 1.44&middot;c/2B):
{RES_AIR_M:.1f} m air / {RES_FIRN_M:.1f} m in-firn &mdash; N &ge; 40 stacks
(mean spacing &le; 3 m) are FULLY UNRESOLVED; N = 10/20 (12 / 6 m) remain
resolved.</p>
{warn}
<h2>Depth-power profiles</h2>
<img src="data:image/png;base64,{_img_b64(grid)}" alt="figure grid">
<p class="note"><b>Method.</b> Coherent fields summed over layers, |.|^2,
trace-averaged; twtt mapped to depth through the per-layer in-firn nadir
times; linear power boxcar-smoothed over {SMOOTH_M:.0f} m
(&asymp;{runs[next(iter(runs))][0].get('effective_window_m', 0):.1f} m
effective); dB relative to the raw surface peak. <b>Dashed curves</b> are the
surface-field-subtracted profiles |E_total &minus; E_surface|&sup2; (same
smoothing and normalization): the kernel is single-bounce linear, so
subtracting the surface's own complex field exactly isolates the layer
signal from the surface off-nadir response that contaminates the upper
~45 m. E_surface is the run's own layer-0 field; its ratio-scaled agreement
with the surface-only reference field is recorded per run
(<code>surface_field_check_rel</code>). eps is POINT-sampled from the
0.1 m-smoothed B26 core (Kovacs eps(rho)).</p>
{cmp_html}
<h2>Diagnostics (recorded, never gated)</h2>
{_diag_table(runs)}
<p class="note">Full-resolution B26 <b>adjacent-sample</b> |&gamma;| (1 mm
spacing, {fullres['spacing_m']*1000:.0f} mm) for context: median
{fullres['median']:.1f} dB, p90 {fullres['p90']:.1f} dB, max
{fullres['max']:.1f} dB. Realized per-interface |&gamma;| are far larger
because consecutive layers are meters apart, not mm. Gradient-interval
threshold sensitivity (0.0 / -0.05 / -0.1 dB/m) is stored per run in
<code>runs/*.json</code> for both raw and subtracted profiles.</p>
<h2>Findings</h2>
<p>{narrative}</p>
"""
    out = OUTDIR / "report.html"
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'>"
                   f"<title>Firn plateau investigation (chirp)</title><style>{css}"
                   f"</style></head><body>{body}</body></html>")
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--layer-counts", type=int, nargs="+", default=list(LAYER_COUNTS))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()
    if not args.report_only:
        run_sweep(tuple(args.layer_counts), args.seeds)
    build_report()


if __name__ == "__main__":
    main()
