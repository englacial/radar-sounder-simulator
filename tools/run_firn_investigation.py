"""Firn power-plateau investigation (EXPLORATORY redo of the flawed M19 case).

This is an honest-observation experiment, NOT a pass/fail gate. It builds flat
multi-layer coherent scenes from the B26 firn density core and records the
Culberg & Schroeder (2020) plateau DIAGNOSTICS (gradient intervals, secondary
maximum), never gating on them. See ``claude_notes/firn_investigation_findings.md``.

Design (user-specified):
  * Scene: flat surface, 500 m AGL, 195 MHz, 600 m extent, 4 m facets, 3 traces
    (the old test_firn_plateau geometry), plus N flat offset layers below it.
  * Media/eps convention (POINT sampling -- preserves local contrast, unlike the
    flawed 5 m slab-MEAN decimation that collapsed the stack to the compaction
    trend): the firn slab immediately below interface k (at depth d_k) is
    assigned the permittivity of the CLOSEST 0.1 m-smoothed B26 sample to d_k,
    via Kovacs et al. (1993) eps = (1 + 0.845*rho[g/cc])^2. Interface k's
    reflection is thus Fresnel(eps(d_k), eps(d_{k+1})); the surface reflects air
    against eps(d_1). The substrate below the deepest layer takes eps(d_N + 1 m).
  * Placements: (a) equally spaced, (b) uniformly-random (sorted, min sep
    ~0.25 m, seeded), both spanning ~1 m to the core end (~119.7 m).
  * Layer counts N in {10, 20, 40, 80}; 3 seeds per N for random placement.
  * Reference: the same scene with only the surface interface.

Observable (paper's quantity): combined coherent trace power vs depth. Fields
are summed over layers, |.|^2, trace-averaged; twtt is mapped to depth through
the per-layer in-firn nadir times; the linear power is boxcar-smoothed over
~5 m; reported in dB relative to the surface peak.

Runtime: kernel compile scales ~O(N^2) (one graph per crossed-interface count);
runs of a given N are batched in one process so the compiled graphs are reused
(seconds after the first). Resumable: a run whose diagnostics json exists is
skipped. Cutoffs: 25 min wall for a first-per-N (compile-inclusive) call, 15 min
for subsequent same-N calls; on breach the remaining runs of that N are skipped
and the skip is recorded loudly.

Run: uv run python tools/run_firn_investigation.py         # full sweep + report
     uv run python tools/run_firn_investigation.py --report-only
"""

import argparse
import base64
import html
import json
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import soundersim  # noqa: E402
from soundersim.config import (DemInterface, FacetConfig, Medium,  # noqa: E402
                               OffsetInterface, RadarConfig, SimConfig)
from soundersim.physics import C, fresnel_normal  # noqa: E402
from soundersim import synthetic as syn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXDIR = ROOT / "tests" / "fixtures" / "firn"
OUTDIR = ROOT / "outputs" / "firn_investigation"

# --- scene geometry (old test_firn_plateau, verbatim) ---------------------
H, ELEV, EXTENT = 500.0, 500.0, 600.0     # platform AGL, surface elev, scene (m)
F0, DT, NSAMP = 195e6, 5e-9, 512          # MCoRDS3-like carrier; 5 ns bins
T0 = 2.0 * (H - 10.0) / C
POSTING, SPACING, NTRACES = 4.0, 4.0, 3

# --- placement / analysis constants ---------------------------------------
Z_TOP = 1.0                               # shallowest layer depth (m)
MIN_SEP = 0.25                            # random min layer separation (m)
N_SEEDS = 3
SMOOTH_M = 5.0                            # depth-power smoothing window (m)
GRAD_THRESHOLDS = (0.0, -0.05, -0.1)      # dB/m; -0.05 is the reported "near-zero"
GRAD_PRIMARY = -0.05
SURF_EXCL_M = 5.0                         # exclude surface lobe from secondary-max
LAYER_COUNTS = (10, 20, 40, 80)
CUTOFF_FIRST_S = 25 * 60
CUTOFF_NEXT_S = 15 * 60


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
    return RadarConfig(dt=DT, n_samples=NSAMP, t0=T0, f0=F0)


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


# ========================================================================
# observable + diagnostics
# ========================================================================
def depth_power_profile(ds, depths):
    """Combined coherent trace power vs depth (dB rel. surface peak), smoothed.

    Returns (depth_m, prof_db, grad_db_per_m) restricted to the in-layer window
    (0, deepest layer). ``depths`` is None for the surface-only reference (twtt
    mapped with the mean-firn speed for display).
    """
    fld = ds.field
    tot = fld.sum("layer").values if "layer" in fld.dims else fld.values
    p = (np.abs(tot) ** 2).mean(axis=0)                        # (n_samples,)

    twtt = ds.twtt.values
    if depths is not None:
        node_twtt = ds.nadir_twtt.mean("slow_time").values     # (layer,)
        node_depth = np.concatenate([[0.0], depths])
        depth = np.interp(twtt, node_twtt, node_depth)
        z_deep = float(depths[-1])
    else:
        surf_twtt = float(ds.nadir_twtt.mean())
        depth = np.maximum((twtt - surf_twtt) * C
                           / (2.0 * np.sqrt(EPS_MEAN)), 0.0)
        z_deep = ZMAX

    bin_depth = C * DT / (2.0 * np.sqrt(EPS_MEAN))              # ~0.46 m/bin
    w = max(int(round(SMOOTH_M / bin_depth)) | 1, 3)
    p_s = np.convolve(p, np.ones(w) / w, mode="same")
    prof_db = 10.0 * np.log10(np.maximum(p_s / p_s.max(), 1e-12))

    m = (depth > 0.0) & (depth < z_deep)
    d, r = depth[m], prof_db[m]
    order = np.argsort(d)
    d, r = d[order], r[order]
    keep = np.concatenate([[True], np.diff(d) > 1e-6])
    d, r = d[keep], r[keep]
    grad = np.gradient(r, d)
    return d, r, grad, float(w * bin_depth)


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
    """Highest smoothed level (dB rel. surface) below the surface lobe."""
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
    layer_db = gdb[1:]                                          # interfaces below surface
    return dict(median=float(np.median(layer_db)),
                p90=float(np.percentile(layer_db, 90)),
                min=float(layer_db.min()), max=float(layer_db.max()),
                surface_db=surf_db)


def diagnose(ds, depths, placement, n, seed, wall_s, first_per_n):
    d, r, grad, eff_win = depth_power_profile(ds, depths)
    grad_intervals = {f"{t:+.2f}": dict(zip(("length_m", "z0_m", "z1_m"),
                                            longest_nonneg_interval(d, grad, t)))
                      for t in GRAD_THRESHOLDS}
    sec_db, sec_z = secondary_max(d, r)
    diag = {
        "placement": placement, "n": n, "seed": seed,
        "wall_s": round(wall_s, 2), "first_per_n": first_per_n,
        "smooth_window_m": SMOOTH_M, "effective_window_m": round(eff_win, 3),
        "grad_primary_threshold_db_per_m": GRAD_PRIMARY,
        "grad_interval_primary": grad_intervals[f"{GRAD_PRIMARY:+.2f}"],
        "grad_interval_sensitivity": grad_intervals,
        "secondary_max_db": sec_db, "secondary_max_depth_m": sec_z,
        "realized_gamma_db": realized_gamma_db(depths),
        "n_depth_samples": int(len(d)),
        "mean_layer_spacing_m": float(np.mean(np.diff(np.concatenate(
            [[0.0], depths])))),
    }
    return diag, d, r


# ========================================================================
# sweep
# ========================================================================
def run_id(placement, n, seed):
    return f"{placement}_N{n}" + (f"_s{seed}" if seed is not None else "")


def _paths(rid):
    return OUTDIR / "runs" / f"{rid}.json", OUTDIR / "runs" / f"{rid}.npz"


def _do_run(scene, cfg, depths, placement, n, seed, first_per_n):
    rid = run_id(placement, n, seed)
    jpath, npath = _paths(rid)
    if jpath.exists():
        print(f"  [skip-exists] {rid}", flush=True)
        return json.loads(jpath.read_text()), False
    t = time.perf_counter()
    ds = soundersim.simulate(scene, cfg)
    wall = time.perf_counter() - t
    diag, d, r = diagnose(ds, depths, placement, n, seed, wall, first_per_n)
    if not (np.isfinite(r).all() and np.isfinite(d).all()):
        raise ValueError(f"{rid}: non-finite profile")
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(diag, indent=1) + "\n")
    np.savez(npath, depth=d, prof_db=r)
    kind = "first/compile" if first_per_n else "cached"
    print(f"  [ok] {rid}  {wall:6.1f} s ({kind})  "
          f"plateau(-0.05)={diag['grad_interval_primary']['length_m']:.1f} m  "
          f"sec_max={diag['secondary_max_db']:.1f} dB", flush=True)
    return diag, wall > (CUTOFF_FIRST_S if first_per_n else CUTOFF_NEXT_S)


def run_sweep(layer_counts=LAYER_COUNTS, n_seeds=N_SEEDS):
    (OUTDIR / "runs").mkdir(parents=True, exist_ok=True)
    scene = _scene()
    skips = []

    # reference (surface only) -----------------------------------------
    rid = "reference"
    jpath, npath = _paths(rid)
    if not jpath.exists():
        t = time.perf_counter()
        ds = soundersim.simulate(scene, reference_cfg())
        wall = time.perf_counter() - t
        d, r, grad, eff = depth_power_profile(ds, None)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(
            {"placement": "reference", "wall_s": round(wall, 2),
             "effective_window_m": round(eff, 3)}, indent=1) + "\n")
        np.savez(npath, depth=d, prof_db=r)
        print(f"  [ok] reference  {wall:5.1f} s", flush=True)
    else:
        print("  [skip-exists] reference", flush=True)

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
                              seed, first)
            first_done = True
            if over:
                print(f"  [CUTOFF] {rid} exceeded its wall budget; skipping "
                      f"remaining N={n} runs", flush=True)
                breached = True
    if skips:
        (OUTDIR / "skipped.json").write_text(json.dumps(skips, indent=1) + "\n")
    return skips


# ========================================================================
# report
# ========================================================================
FINDINGS_NARRATIVE = """\
<b>Is a plateau there?</b> Partially -- the morphology, not the paper's
operational criterion. Every layered variant holds power far above the
surface-only reference from ~10 m to the bottom of the core (8-11 dB mean
excess over 40-100 m), with a common shape set by the B26 core itself, not by
layer count or placement: a fast drop off the surface peak to about -15 dB by
10 m, a broad structured shoulder over ~10-50 m (total span 12-16 dB, mean
about -20 dB) containing a pronounced dip at ~25 m and a local maximum at
~40-45 m, then steady decay below ~50-60 m reaching -40 to -50 dB by
100-119 m. The decay-onset depth is qualitatively where Culberg &amp;
Schroeder's Fig. 3 profiles roll off (their "sharp decay below ~60 m").

<b>The paper's operational criterion is NOT met by any run.</b> The longest
contiguous interval with smoothed-trace gradient &ge; -0.05 dB/m is 2.4-9.6 m
across the 13 completed runs -- all short of the paper's "more than 10 m".
The closest is random N=40 seed 1 (9.6 m spanning 26.9-36.5 m). Where the
placement is dense enough for a continuous profile (N&ge;40), the interval
consistently locates in the 27-42 m rising limb (the recovery from the 25 m
dip to the 40-45 m bump); for resolved small-N stacks it jumps between
isolated echoes. Threshold sensitivity is mild (0.0 vs -0.1 dB/m changes
lengths by under ~2.5 m), so the shortfall is not a threshold artifact.

<b>Secondary maximum.</b> In every run it sits at 5.1-5.5 m depth at -6.0 to
-13.7 dB relative to the surface return (median about -11 dB). The paper's
Fig. 8 discussion puts typical near-surface secondary maxima 10-15 dB below
the surface -- ours are the same order, slightly strong on average.

<b>N-dependence: the resolved-to-unresolved transition is visible.</b> Mean
layer spacing runs from 12.0 m (N=10) to 1.5 m (N=80) against a ~3.7 m
in-firn range resolution. At N=10-20 (spacing &gt; resolution) the profile is
a train of isolated boxy echoes riding the surface-clutter rolloff -- visibly
not a plateau. At N=40 (3.0 m, about the resolution) and N=80 (1.5 m) the
echoes merge into a continuous profile and the deep floor rises: the 60-100 m
band mean climbs from about -35 dB (N=10-40) to -31.6 dB (N=80), and the
100-119 m mean from -49 to -40 dB. More interfaces per resolution cell add
power and slow the apparent decay. The trend points toward the paper's
plateau -- produced by mm-scale stratigraphy, i.e. thousands of interfaces per
resolution cell -- emerging further along this N direction than the compile
budget reaches (the N=80 first call took 26.1 min, breaching the 25 min
cutoff; its three random-placement runs were skipped).

<b>Equal vs random placement.</b> At fixed N the band means agree within
~2 dB and the gross morphology is identical. Individual bright isolated
echoes are placement-specific (equal N=10's boxes at ~60 and ~80 m move or
vanish across random seeds, seed-to-seed scatter ~3 dB in the 60-100 m band),
but the 25 m dip and the 40-45 m bump appear in all 13 runs: they are
core-driven features, not sampling accidents.

<b>Realized contrasts vs the continuous profile.</b> The per-interface
|gamma| median falls with N: -40.5 / -46.2 / -50.5 / -52.9 dB for equal
N=10/20/40/80 (p90 -30.7 to -41.7 dB), because closer point samples span
smaller density differences. The full-resolution adjacent-sample statistics
(1 mm spacing) are median -90.7 dB, p90 -79.9 dB: even N=80's distribution is
~30-40 dB above the mm-adjacent level, i.e. the density profile decorrelates
over decimetres-to-metres and an O(80)-layer point-sampled stack is nowhere
near a converged discretisation of the continuous profile. This is the
quantitative sense in which these runs under-sample the physics that the
paper's 1-D transfer-matrix model (every 1 mm sample a layer) captures.

<b>Methodological note (found and fixed during this sweep).</b> The first
pass reused the old case's plain mode='same' boxcar for the 0.1 m density
smoothing; its zero-padding halves the deepest ~5 cm of density (882 to
446 kg/m^3, eps 3.05 to 1.90), and the point-sampled deepest layer landed on
it, planting a spurious -18 dB reflector at the stack bottom (false
"secondary maximum" of -3.7 dB at ~108 m). The loader here normalises by the
window overlap; the artifact is absent in all results shown. The old
test_firn_plateau case (5 m slab means) is largely insensitive to this
particular edge effect, but its framing is under review separately; this
investigation also repaired only its figure plotting (digitized Fig. 9 curves
are now sorted by depth before drawing -- fig09a had 59 depth reversals and
drew as a self-intersecting path)."""


def _load_runs():
    runs = {}
    for jp in sorted((OUTDIR / "runs").glob("*.json")):
        rid = jp.stem
        npz = OUTDIR / "runs" / f"{rid}.npz"
        if not npz.exists():
            continue
        arr = np.load(npz)
        runs[rid] = (json.loads(jp.read_text()), arr["depth"], arr["prof_db"])
    return runs


def _figure_grid(runs, path):
    ref = runs.get("reference")
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    axes = axes.ravel()
    panels = [(p, n) for p in ("equal", "random") for n in LAYER_COUNTS]
    for ax, (placement, n) in zip(axes[:8], panels):
        if placement == "equal":
            keys = [f"equal_N{n}"]
        else:
            keys = [f"random_N{n}_s{s}" for s in range(N_SEEDS)]
        for i, k in enumerate(keys):
            if k not in runs:
                continue
            diag, d, r = runs[k]
            lbl = "equal" if placement == "equal" else f"seed {diag['seed']}"
            ax.plot(d, r, lw=1.0, color=f"C{i}", label=lbl)
        if ref is not None:
            ax.plot(ref[1], ref[2], "k--", lw=0.9, alpha=0.6,
                    label="surface only")
        ax.set_xlim(0, 120)
        ax.set_ylim(-60, 3)
        ax.set_title(f"{placement}, N={n}", fontsize=10)
        ax.set_xlabel("depth (m)")
        ax.set_ylabel("power (dB rel. surface)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")

    ax = axes[8]
    for i, n in enumerate(LAYER_COUNTS):
        k = f"equal_N{n}"
        if k in runs:
            _, d, r = runs[k]
            ax.plot(d, r, lw=1.1, color=f"C{i}", label=f"equal N={n}")
    if ref is not None:
        ax.plot(ref[1], ref[2], "k--", lw=0.9, alpha=0.6, label="surface only")
    ax.set_xlim(0, 120)
    ax.set_ylim(-60, 3)
    ax.set_title("Summary: equal-placement profiles across N", fontsize=10)
    ax.set_xlabel("depth (m)")
    ax.set_ylabel("power (dB rel. surface)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Firn power plateau investigation (C&S 2020 Fig. 3 analog) -- "
                 "B26 core, 195 MHz, 500 m AGL, point-sampled eps", fontsize=13)
    fig.savefig(path, dpi=95)
    plt.close(fig)


def _diag_table(runs):
    hdr = ("<tr><th>run</th><th>N</th><th>placement</th><th>mean spacing (m)</th>"
           "<th>plateau interval (-0.05 dB/m)</th><th>secondary max</th>"
           "<th>realized |&gamma;| median / p90 (dB)</th><th>wall (s)</th></tr>")
    rows = []
    for rid in sorted(k for k in runs if k != "reference"):
        d, _, _ = runs[rid]
        gi = d["grad_interval_primary"]
        loc = (f"{gi['length_m']:.1f} m @ [{gi['z0_m']:.0f}, {gi['z1_m']:.0f}] m"
               if gi["z0_m"] is not None else "-")
        g = d["realized_gamma_db"]
        rows.append(
            f"<tr><td>{html.escape(rid)}</td><td>{d['n']}</td>"
            f"<td>{d['placement']}</td><td>{d['mean_layer_spacing_m']:.2f}</td>"
            f"<td>{loc}</td>"
            f"<td>{d['secondary_max_db']:.1f} dB @ {d['secondary_max_depth_m']:.0f} m</td>"
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
    narrative = FINDINGS_NARRATIVE.replace("\n\n", "</p><p>")
    body = f"""
<h1>Firn power plateau investigation</h1>
<p class="note"><b>Exploratory redo of the flawed M19 case.</b> This records the
Culberg &amp; Schroeder (2020) plateau diagnostics on point-sampled B26 firn
stacks; it is NOT a pass/fail gate. The paper's plateau definition is
operational: a trace shows a plateau if the gradient of the smoothed trace is
near-zero or nonnegative over depth intervals &gt; 10 m (text around Figs 2-3),
with a near-surface secondary maximum typically 10-15 dB below the surface
return (Fig. 8) and sharp decay below ~60 m. Fig. 3 is the qualitative visual
reference (not digitized). The flawed first attempt
(<code>tests/test_firn_plateau.py</code>) used 5 m slab-MEAN decimation, which
collapsed the stack to the smooth compaction trend and killed the dm-scale
variability that sustains the plateau, then gated on a self-defined "plateau".
Its framing is under review separately.</p>
{warn}
<h2>Depth-power profiles</h2>
<img src="data:image/png;base64,{_img_b64(grid)}" alt="figure grid">
<p class="note"><b>Method.</b> Coherent fields summed over layers, |.|^2,
trace-averaged; twtt mapped to depth through the per-layer in-firn nadir times;
linear power boxcar-smoothed over {SMOOTH_M:.0f} m
(&asymp;{runs[next(iter(runs))][0].get('effective_window_m', 0):.1f} m
effective); dB relative to the surface peak. eps is POINT-sampled from the
0.1 m-smoothed B26 core at each interface depth (Kovacs eps(rho)); interface k
reflects eps(d_k) against eps(d_(k+1)).</p>
<h2>Diagnostics (recorded, never gated)</h2>
{_diag_table(runs)}
<p class="note">Full-resolution B26 <b>adjacent-sample</b> |&gamma;| (1 mm
spacing, {fullres['spacing_m']*1000:.0f} mm) for context: median
{fullres['median']:.1f} dB, p90 {fullres['p90']:.1f} dB, max
{fullres['max']:.1f} dB. The realized per-interface |&gamma;| above are far
larger because consecutive layers are meters apart, not mm -- the contrast is
sampled across the actual layer spacing. Gradient-interval sensitivity to the
threshold (0.0 / -0.05 / -0.1 dB/m) is stored per run in
<code>runs/*.json</code>.</p>
<h2>Findings</h2>
<p>{narrative}</p>
"""
    out = OUTDIR / "report.html"
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'>"
                   f"<title>Firn plateau investigation</title><style>{css}"
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
