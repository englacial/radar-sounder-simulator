"""Firn plateau vs bandwidth / centre frequency at the B26 core, airborne (frame
20190418_01_009 as flown) and HAPS (14 km constant ellipsoidal height) geometry.

Culberg & Schroeder 2020 (Sec. VI, Fig. 14): the near-surface firn "power
plateau" is a stack of quasi-specular density-layer reflections; at fixed
bandwidth it grows with centre frequency, and at fixed frequency it falls with
bandwidth (fewer interfaces per range bin). This tool reuses the validated B26
machinery of tools/run_b26_comparison.py (ArcticDEM surface + BedMachine bed
wide run, N-layer effective-contrast firn stack on a narrow strip, field-summed
without the firn run's own surface) but simulates the DELTA-kernel field once
per geometry x carrier and applies every pulse (bandwidth, length, window,
analytic/explicit-chirp construction) in post-processing -- the kernels do not
depend on the waveform (waveform.py: pulse compression is a post-kernel fast-
time convolution), so a bandwidth sweep costs one kernel run.

Cases:
  airborne  nav as flown, 7-el 0.5-lambda MCoRDS array (roll from nav), smooth
            surface/bed/layers, the reference B26 conventions (ct +-3 km wide,
            +-600 m firn strip, beta 0.5 facets), 195 MHz; pulses 10-97 MHz.
  haps      nav z = --alt (14 km) constant, 8-el Hann-tapered 10 m cross-track
            array (design-study antenna), C&S Fig. 11 sub-facet roughness on
            the surface (rac.SURF_ROUGH_*) and every internal firn interface
            ('mcords' inversion), design-study bed roughness, grazing fix on,
            firn stack on the FULL wide DEM (+-ct_wide) so wide-angle layer
            scatter can reach the bed delay; carriers 195 and 300 MHz.

Fast-time grid: dt_sim = 16.667/8 = 2.083 ns (envelope-quantization alias
195 MHz at f0 = 195, 180 MHz at f0 = 300 -> every bandwidth <= 150 MHz is
alias-free by the M21/M24 rule, asserted per pulse). Airborne fields are
decimated [::8] onto the frame grid for the measured comparison.

Also a 1-D reference: the RAW 1 mm B26 density profile (1 cm resampled) as a
normal-incidence transfer-matrix reflection spectrum across each band,
Hann-weighted and inverse-transformed -- the C&S 1-D layered-dielectric model
at this site, i.e. what their Fig. 14 computes.

Run: uv run python tools/run_firn_bandwidth.py airborne
     uv run python tools/run_firn_bandwidth.py haps --alt 14000 --n-traces 24
     uv run python tools/run_firn_bandwidth.py airborne --report-only
Outputs: outputs/firn_bandwidth/<case>/ (runs/ cache, metrics.json, figures).
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from pyproj import Transformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_b26_comparison as rb  # noqa: E402
import run_firn_investigation as rfi  # noqa: E402
from soundersim import firn  # noqa: E402
from soundersim.config import (AntennaConfig, DemInterface, FacetConfig,  # noqa: E402
                               GrazingFixConfig, Medium, RadarConfig,
                               RoughnessConfig, SimConfig, WaveformConfig)
from soundersim.opr import load_bottom_pick, load_frame  # noqa: E402
from soundersim.simulate import simulate  # noqa: E402
from soundersim.waveform import compressed_pulse, convolve_fast_time  # noqa: E402

C = rb.C
OVS = 8                                  # dt_sim = dt_frame / 8 = 2.083 ns
OUT_ROOT = ROOT / "outputs" / "firn_bandwidth"
SURF_ROUGH = (0.049474, 2.982179)        # rac.SURF_ROUGH_* (C&S Fig. 11 mcords, 0 m)
BED_ROUGH = (0.1, 0.886)                 # design-study bed roughness
GFIX_S_EFF = 0.05                        # config/analysis.yaml grazing_fix
HAPS_INSTR = ROOT / "config" / "instruments" / "hd_f300_n8_hann.yaml"
HAPS_SPAN_M = 10.0
PRE_SURF_US, POST_BED_US = 0.8, 3.5      # HAPS window margins (analysis.yaml)
MID_US, BED_US = (1.0, 0.5), (0.5, 1.5)  # analysis.yaml midcolumn / bed windows
BANDS = ((5, 20), (20, 60), (60, 120), (20, 70), (40, 100), (80, 120))
CASES = {
    "airborne": dict(alt=None, ct_wide=3000.0, ct_firn=600.0, f0s=(195e6,),
                     rough=False, n_traces=60),
    "haps": dict(alt=14000.0, ct_wide=12500.0, ct_firn=None, f0s=(195e6, 300e6),
                 rough=True, n_traces=24),
}
# pulses evaluated in post-processing: (f0_MHz, B_MHz, T_s, construction)
PULSES = {
    "airborne": [(195, b, 10e-6, "analytic") for b in (10, 30, 60, 97)]
                + [(195, b, 10e-6, "chirp") for b in (30, 97)],
    "haps": [(195, b, 8e-6, "chirp") for b in (10, 30, 60, 97)]
            + [(300, b, 8e-6, "chirp") for b in (10, 30, 60, 100, 150)],
}
ONED_B_MHZ = (5, 10, 20, 30, 60, 100, 150)


# ========================================================================
# geometry
# ========================================================================
def dem_at_nav(scene):
    """(z_surface, thickness) of the scene's DEM stack under each nav point."""
    tr = Transformer.from_crs("EPSG:4326", scene.crs, always_xy=True)
    px, py = tr.transform(scene.nav_llh[:, 1], scene.nav_llh[:, 0])
    cols, rows = (~scene.transform) * (px, py)
    r = np.clip(np.round(rows).astype(int), 0, scene.dem.shape[0] - 1)
    c = np.clip(np.round(cols).astype(int), 0, scene.dem.shape[1] - 1)
    return scene.dems[0][r, c], (scene.dems[0] - scene.dems[1])[r, c]


def antenna(case, f0):
    if case == "airborne":
        return AntennaConfig(kind="array", n_elements=rb.N_ELEMENTS,
                             spacing_lam=rb.SPACING_LAM, roll_source="nav")
    a = yaml.safe_load(HAPS_INSTR.read_text())["simulated"]["antenna"]
    n = a["n_elements"]
    return AntennaConfig(kind="array_tapered", n_elements=n,
                         spacing_lam=HAPS_SPAN_M / ((n - 1) * C / f0),
                         tx_weights=a["tx_weights"], rx_weights=a["rx_weights"],
                         roll_source="none")


def radar_grid(frame, fsub, bot_sub, wscene, f0, alt, ant):
    """Delta-waveform simulation grid at dt_frame/OVS, t0 anchored on a frame
    bin: the reference B26 pick-based window when flying as flown, else the
    geometry window [min surface - PRE, max bed + POST] at ``alt``."""
    tw = frame.twtt.values
    dt, t0f = float((tw[-1] - tw[0]) / (len(tw) - 1)), float(tw[0])
    if alt is None:
        _, rc_frame, b0 = rb.radar_grids(frame, fsub, bot_sub,
                                         rb.mcords_2019_params())
        nb = rc_frame.n_samples
    else:
        z_s, thick = dem_at_nav(wscene)
        t_s = 2.0 * (alt - z_s) / C
        t_b = t_s + 2.0 * thick * np.sqrt(rb.EPS_ICE) / C
        b0 = int(np.floor((t_s.min() - PRE_SURF_US * 1e-6 - t0f) / dt))
        nb = int(np.ceil((t_b.max() + POST_BED_US * 1e-6 - t0f) / dt)) - b0 + 1
    rc = RadarConfig(dt=dt / OVS, n_samples=OVS * (nb - 1) + 1,
                     t0=t0f + b0 * dt, f0=f0,
                     waveform=WaveformConfig(kind="delta"), antenna=ant)
    return rc, dt


def alias_hz(f0, dt):
    return abs(f0 - round(f0 * dt) / dt)


# ========================================================================
# configs
# ========================================================================
def wide_cfg(rc, spacing, rough):
    rs = RoughnessConfig(sigma_m=SURF_ROUGH[0], corr_length_m=SURF_ROUGH[1]) \
        if rough else None
    rbd = RoughnessConfig(sigma_m=BED_ROUGH[0], corr_length_m=BED_ROUGH[1]) \
        if rough else None
    return SimConfig(
        mode="coherent", split_sides=False, radar=rc,
        facets=FacetConfig(spacing=spacing),
        grazing_fix=GrazingFixConfig(s_eff=GFIX_S_EFF) if rough else None,
        media=[Medium(name="air", eps_r=1.0),
               Medium(name="ice", eps_r=rb.EPS_ICE,
                      attenuation_db_per_km=rb.ATT_DB_PER_KM),
               Medium(name="bed", eps_r=rb.EPS_BED)],
        interfaces=[DemInterface(name="surface", roughness=rs),
                    DemInterface(name="bed", roughness=rbd)])


def firn_cfg(rc, spacing, depths, eps, rough):
    r = rb.layer_roughness(depths, "mcords") if rough else None
    media, ifaces = firn.firn_stack(depths, eps, rb.ATT_DB_PER_KM, roughness=r)
    return SimConfig(
        mode="coherent", split_sides=False, radar=rc,
        facets=FacetConfig(spacing=spacing),
        grazing_fix=GrazingFixConfig(s_eff=GFIX_S_EFF) if rough else None,
        media=media, interfaces=ifaces)


# ========================================================================
# cached delta-kernel runs (full dt_sim per-layer fields)
# ========================================================================
def run_cached(rid, chunks, cfg, meta, runs_dir):
    runs_dir.mkdir(parents=True, exist_ok=True)
    jp, npz = runs_dir / f"{rid}.json", runs_dir / f"{rid}.npz"
    key = json.dumps(meta, sort_keys=True)
    if jp.exists() and npz.exists():
        d = json.loads(jp.read_text())
        if d["meta_key"] == key:
            print(f"  [cache] {rid} ({d['wall_s']:.0f} s recorded)", flush=True)
            return d, dict(np.load(npz))
    n_tr = sum(len(rows) for _, rows in chunks)
    field = nadir = twtt = None
    wall, msgs, facets = 0.0, [], []
    for scene, rows in chunks:
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = simulate(scene, cfg)
        wall += time.perf_counter() - t
        msgs += [str(x.message) for x in w]
        f = np.asarray(ds.field.values, np.complex64)
        if field is None:
            field = np.zeros((n_tr,) + f.shape[1:], np.complex64)
            nadir = np.zeros((n_tr, f.shape[-1]))
            twtt = np.asarray(ds.twtt.values, np.float64)
            layers = [str(x) for x in ds.layer.values]
        field[rows] = f
        nadir[rows] = ds.nadir_twtt.values
        facets.append(rb._n_facets(scene.dem.shape, cfg.facets.spacing))
        print(f"    chunk {rows[0]}-{rows[-1]}: {time.perf_counter() - t:.0f} s",
              flush=True)
    if not np.isfinite(field).all():
        raise RuntimeError(f"{rid}: non-finite field")
    d = {"rid": rid, "wall_s": round(wall, 1), "meta_key": key, "meta": meta,
         "layers": layers, "n_chunks": len(chunks), "facets_per_chunk": facets,
         "warnings": sorted(set(msgs))}
    np.savez_compressed(npz, field=field, twtt=twtt, nadir_twtt=nadir)
    jp.write_text(json.dumps(d, indent=1) + "\n")
    print(f"  [ok] {rid}  {wall:.0f} s", flush=True)
    return d, dict(field=field, twtt=twtt, nadir_twtt=nadir)


# ========================================================================
# analysis
# ========================================================================
def pulse(E, dt, f0, b_hz, t_s, construction, window="hann"):
    """Apply a compressed pulse to a delta-kernel field (traces, n[, ...])."""
    assert alias_hz(f0, dt) > b_hz / 2.0, "envelope-quantization alias in band"
    p, m = compressed_pulse(b_hz, t_s, dt, window, construction)
    return convolve_fast_time(np.asarray(E, np.complex128), p, m)


def _wmean(P, twtt, lo, hi):
    """Per-trace mean power in [lo, hi] (twtt arrays per trace)."""
    out = np.full(P.shape[0], np.nan)
    for t in range(P.shape[0]):
        m = (twtt >= lo[t]) & (twtt < hi[t])
        if m.any():
            out[t] = P[t, m].mean()
    return out


def _med_db(num, den):
    ok = np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    return float(np.median(10 * np.log10(num[ok] / den[ok]))) if ok.any() \
        else float("nan")


def mean_profile(P, twtt, t_guess, dt, zmax=rb.PROFILE_MAX_M):
    """Mean-linear-power depth profile over traces, dB rel own surface peak."""
    acc, dep = [], None
    for t in range(P.shape[0]):
        t_s = rb.surface_peak_twtt(P[[t]], twtt, np.array([t_guess[t]]), dt)[0]
        d, db = rb.profile_vs_depth(np.asarray(P[t], np.float64), twtt, t_s, dt)
        m = (d >= -10) & (d <= zmax)
        if dep is None:
            dep = d[m]
        acc.append(np.interp(dep, d[m], 10 ** (db[m] / 10)))
    return dep, 10 * np.log10(np.mean(acc, 0))


def analyze(Es, Ef, Eb, twtt, dt, t_s_guess, t_b):
    """Plateau bands (mean linear power rel own surface peak, over traces),
    mid-column / bed-window decomposition (medians over traces of per-trace
    ratios to the trace's own surface peak) and the bed-over-clutter number."""
    Pt = np.abs(Es + Ef + Eb) ** 2
    parts = {"surface": np.abs(Es) ** 2, "firn": np.abs(Ef) ** 2,
             "bed": np.abs(Eb) ** 2, "surface+firn": np.abs(Es + Ef) ** 2,
             "total": Pt}
    t_s = rb.surface_peak_twtt(Pt, twtt, t_s_guess, dt)
    spk = np.array([Pt[t, int(np.searchsorted(twtt, t_s[t]))]
                    for t in range(Pt.shape[0])])
    bands = rb.band_levels_meanp(Pt, twtt, t_s_guess, dt, edges=(5, 20, 60, 120),
                                 extra=((20, 70), (40, 100), (80, 120)))
    mid = {k: _med_db(_wmean(P, twtt, t_s + MID_US[0] * 1e-6,
                             t_b - MID_US[1] * 1e-6), spk)
           for k, P in parts.items()}
    bedw = {k: _med_db(_wmean(P, twtt, t_b - BED_US[0] * 1e-6,
                              t_b + BED_US[1] * 1e-6), spk)
            for k, P in parts.items()}
    bw_b = _wmean(parts["bed"], twtt, t_b - BED_US[0] * 1e-6,
                  t_b + BED_US[1] * 1e-6)
    bw_c = _wmean(parts["surface+firn"], twtt, t_b - BED_US[0] * 1e-6,
                  t_b + BED_US[1] * 1e-6)
    bw_s = _wmean(parts["surface"], twtt, t_b - BED_US[0] * 1e-6,
                  t_b + BED_US[1] * 1e-6)
    bw_f = _wmean(parts["firn"], twtt, t_b - BED_US[0] * 1e-6,
                  t_b + BED_US[1] * 1e-6)
    # absolute (sim units) surface peak and 20-70 m band power: separates the
    # firn power's own bandwidth scaling from the surface-peak normalizer
    dep = (twtt[None, :] - t_s[:, None]) * C / (2 * np.sqrt(rfi.EPS_MEAN))
    m2070 = (dep >= 20) & (dep < 70)
    p2070 = np.array([Pt[t, m2070[t]].mean() for t in range(Pt.shape[0])])
    absd = {"surface_peak_db": round(float(np.median(10 * np.log10(spk))), 2),
            "band_20_70_db": round(float(10 * np.log10(np.mean(p2070))), 2)}
    return {"plateau_db_rel_surface": {k: round(v, 2) for k, v in bands.items()},
            "absolute_db": absd,
            "midcol_rel_surf_db": {k: round(v, 2) for k, v in mid.items()},
            "bedwin_rel_surf_db": {k: round(v, 2) for k, v in bedw.items()},
            "bed_over_clutter_db": round(_med_db(bw_b, bw_c), 2),
            "bed_over_surface_only_db": round(_med_db(bw_b, bw_s), 2),
            "firn_over_surface_at_bed_db": round(_med_db(bw_f, bw_s), 2)}


# ========================================================================
# 1-D reference: full-resolution TMM band response (C&S layered model)
# ========================================================================
def tmm_spectrum(n, dz, freqs):
    """Normal-incidence TMM reflection r(f) of slabs ``n`` (thickness dz) under
    air, last slab as the half-space; vectorized over ``freqs`` (Yeh, as
    firn.tmm_reflection)."""
    k = 2 * np.pi * np.asarray(freqs) / C
    stack = np.concatenate([[1.0], n, [n[-1]]])
    kx = k[:, None] * stack[None, :]                   # (Nf, Nz+2)
    M = np.tile(np.eye(2, dtype=complex), (len(k), 1, 1))
    for m in range(len(n)):
        q = kx[:, m + 1] / kx[:, m]
        I = 0.5 * np.stack([np.stack([1 + q, 1 - q], -1),
                            np.stack([1 - q, 1 + q], -1)], -2)
        ph = np.exp(-1j * kx[:, m + 1] * dz)
        M = M @ I
        M[:, :, 0] *= ph[:, None]
        M[:, :, 1] *= np.conj(ph)[:, None]
    q = kx[:, -1] / kx[:, -2]
    I = 0.5 * np.stack([np.stack([1 + q, 1 - q], -1),
                        np.stack([1 - q, 1 + q], -1)], -2)
    M = M @ I
    return M[:, 1, 0] / M[:, 0, 0]


def oned_profile(f0, b_hz, dz=0.01, nf=1024, pad=16):
    """(depth_m, dB rel surface peak) of the raw B26 profile's Hann-weighted
    band response at (f0, B): the C&S 1-D layered-dielectric model."""
    z, n = rb.B26_CORE.raw_index()
    zg = np.arange(z[0], z[-1], dz)
    ng = np.interp(zg + dz / 2, z, n)
    f = f0 + (np.arange(nf) - nf / 2) / nf * b_hz
    r = tmm_spectrum(ng, dz, f)
    W = 0.5 + 0.5 * np.cos(2 * np.pi * (f - f0) / b_hz)
    # TMM phases are exp(+i...) for delay (conjugate to the DFT kernel);
    # zero-pad with the negative offsets at the END of the spectrum
    S = W * np.conj(r)
    Sf = np.zeros(nf * pad, complex)
    Sf[:nf // 2], Sf[-nf // 2:] = S[nf // 2:], S[:nf // 2]
    e = np.fft.ifft(Sf)
    t = np.arange(nf * pad) / (b_hz * pad)              # twtt from the top
    P = np.abs(e) ** 2
    # r(f) is referenced to z[0]; the air-firn surface reflection is at t=0
    keep = t < 3.0e-6
    P, t = P[keep], t[keep]
    w = max(int(round(5.0 / (C / (2 * np.sqrt(rfi.EPS_MEAN)) / (b_hz * pad)))) | 1, 3)
    Ps = np.convolve(P, np.ones(w) / w, "same")
    depth = t * C / (2 * np.sqrt(rfi.EPS_MEAN)) + z[0]
    return depth, 10 * np.log10(np.maximum(Ps / Ps[depth < 5].max(), 1e-15))


def band_means(depth, db, bands=BANDS):
    lin = 10 ** (db / 10)
    return {f"{a}-{b}m": round(float(10 * np.log10(lin[(depth >= a) & (depth < b)].mean())), 2)
            for a, b in bands}


# ========================================================================
# driver
# ========================================================================
def run_case(case, n_traces=None, alt=None, along_m=rb.ALONG_M, n_layers=20,
             report_only=False):
    cs = CASES[case]
    n_traces = n_traces or cs["n_traces"]
    alt = cs["alt"] if alt is None else alt
    out = OUT_ROOT / case
    runs_dir = out / "runs"
    out.mkdir(parents=True, exist_ok=True)

    frame = load_frame(rb.SEASON, rb.FRAME_ID)
    bot_full = load_bottom_pick(frame)
    fsub, sinfo = rb.sub_frame(frame, along_m)
    a, b = sinfo["slice"]
    bot_sub = bot_full[a:b]
    wscene, waux = rb.wide_scene(fsub, n_traces, cs["ct_wide"])
    idx = waux["idx"]
    if alt is not None:
        wscene.nav_llh = wscene.nav_llh.copy()
        wscene.nav_llh[:, 2] = alt
        wscene.nav_roll = None
    z_s, thick = dem_at_nav(wscene)
    r_min = float((wscene.nav_llh[:, 2] - z_s).min())
    depths = rfi.equal_depths(n_layers)
    ct_firn = cs["ct_firn"] or cs["ct_wide"]
    meta0 = {"season": rb.SEASON, "frame_id": rb.FRAME_ID, "along_m": along_m,
             "n_traces": int(len(idx)), "alt_m": alt, "ovs": OVS,
             "rough": cs["rough"], "att": rb.ATT_DB_PER_KM}

    results, profiles, cfgs = {}, {}, {}
    t_all = time.perf_counter()
    for f0 in cs["f0s"]:
        tag = f"f{int(f0 / 1e6)}"
        ant = antenna(case, f0)
        rc, dt_frame = radar_grid(frame, fsub, bot_sub, wscene, f0, alt, ant)
        spacing = rb.facet_spacing(rc, r_min, float(np.median(thick)))
        eps, r_eff = rb.effective_contrast_eps(depths, rc.wavelength)
        chunks = rb.firn_scenes(wscene, ct_firn, spacing, n_chunks=(
            None if case == "airborne" else max(1, len(idx) // 8)))
        m = {**meta0, "f0": f0, "spacing": round(spacing, 3),
             "dt_ns": round(rc.dt * 1e9, 4), "t0_us": round(rc.t0 * 1e6, 4),
             "n_samples": rc.n_samples, "ct_wide": cs["ct_wide"],
             "ct_firn": ct_firn, "antenna": ant.model_dump()}
        cfgs[tag] = {**m, "n_layers": n_layers,
                     "eps_sum": round(float(eps.sum()), 6),
                     "n_facets_wide": rb._n_facets(wscene.dem.shape, spacing),
                     "n_facets_firn_chunk": [rb._n_facets(s.dem.shape, spacing)
                                             for s, _ in chunks],
                     "alias_MHz": alias_hz(f0, rc.dt) / 1e6}
        print(f"[{case} {tag}] spacing {spacing:.2f} m, dt {rc.dt*1e9:.3f} ns, "
              f"n {rc.n_samples}, wide facets {cfgs[tag]['n_facets_wide']}, "
              f"firn chunks {len(chunks)} x ~{np.mean(cfgs[tag]['n_facets_firn_chunk']):.0f}",
              flush=True)
        if report_only:
            dw = json.loads((runs_dir / f"wide_{tag}.json").read_text())
            df = json.loads((runs_dir / f"firn_{tag}.json").read_text())
            W = dict(np.load(runs_dir / f"wide_{tag}.npz"))
            F = dict(np.load(runs_dir / f"firn_{tag}.npz"))
        else:
            dw, W = run_cached(f"wide_{tag}", [(wscene, np.arange(len(idx)))],
                               wide_cfg(rc, spacing, cs["rough"]),
                               {**m, "kind": "surface+bed"}, runs_dir)
            df, F = run_cached(f"firn_{tag}", chunks,
                               firn_cfg(rc, spacing, depths, eps, cs["rough"]),
                               {**m, "kind": f"firn_N{n_layers}_h1eff",
                                "eps_sum": cfgs[tag]["eps_sum"]}, runs_dir)
        cfgs[tag]["wall_s"] = {"wide": dw["wall_s"], "firn": df["wall_s"]}
        cfgs[tag]["warnings"] = {"wide": dw["warnings"], "firn": df["warnings"]}
        twtt = W["twtt"]
        Es, Eb = W["field"][..., 0], W["field"][..., 1]
        Ef = F["field"][..., 1:].sum(-1)              # firn run's surface excluded
        t_s_guess, t_b = W["nadir_twtt"][:, 0], W["nadir_twtt"][:, 1]
        dec = OVS if case == "airborne" else 1
        for (fm, bm, T, con) in PULSES[case]:
            if fm * 1e6 != f0:
                continue
            key = f"f{fm}_b{bm}_{con}"
            e = [pulse(E, rc.dt, f0, bm * 1e6, T, con)[:, ::dec]
                 for E in (Es, Ef, Eb)]
            tw = twtt[::dec]
            res = analyze(*e, tw, rc.dt * dec, t_s_guess, t_b)
            # surface+bed only (no firn) for the plateau floor
            res["plateau_no_firn_db"] = analyze(
                e[0], 0 * e[1], e[2], tw, rc.dt * dec, t_s_guess,
                t_b)["plateau_db_rel_surface"]
            res.update(f0_MHz=fm, B_MHz=bm, T_us=T * 1e6, construction=con)
            results[key] = res
            Pt = np.abs(e[0] + e[1] + e[2]) ** 2
            profiles[key] = mean_profile(Pt, tw, t_s_guess, rc.dt * dec)
            profiles[key + "_arms"] = arms_profile(e, tw, rc.dt * dec,
                                                   t_s_guess)
            print(f"  {key}: abs surf {res['absolute_db']['surface_peak_db']:.1f} "
                  f"band {res['absolute_db']['band_20_70_db']:.1f} | "
                  f"plateau 20-70 {res['plateau_db_rel_surface']['20-70m']:.1f} "
                  f"40-100 {res['plateau_db_rel_surface']['40-100m']:.1f}  "
                  f"bed/clutter {res['bed_over_clutter_db']:.1f} dB  "
                  f"midcol {res['midcol_rel_surf_db']['total']:.1f}", flush=True)
    wall = time.perf_counter() - t_all

    # 1-D reference at both carriers
    oned = {}
    for fm in (195, 300):
        for bm in ONED_B_MHZ:
            d, db = oned_profile(fm * 1e6, bm * 1e6)
            oned[f"f{fm}_b{bm}"] = band_means(d, db)
            profiles[f"oned_f{fm}_b{bm}"] = (d, db)

    # measured (airborne only)
    meas = {}
    if case == "airborne":
        for prod, fr in (("standard", frame), ("qlook", rb.load_qlook_frame())):
            if fr is None:
                continue
            fs, _ = rb.sub_frame(fr, along_m)
            tw = fr.twtt.values
            dtm = float((tw[-1] - tw[0]) / (len(tw) - 1))
            P = np.asarray(fs.Data.values, np.float64)
            meas[prod] = rb.band_levels_meanp(
                P, tw, fs.Surface.values, dtm, edges=(5, 20, 60, 120),
                extra=((20, 70), (40, 100), (80, 120)))
            profiles[f"measured_{prod}"] = mean_profile(P, tw, fs.Surface.values,
                                                        dtm)

    metrics = {"case": case, "config": cfgs, "pulses": results, "oned": oned,
               "measured": meas, "wall_s_total": round(wall, 1),
               "bed_depth_m_median": round(float(np.median(thick)), 1),
               "agl_m_median": round(float(np.median(wscene.nav_llh[:, 2] - z_s)), 1),
               "layer_depths_m": [round(float(x), 2) for x in depths]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    figures(case, out, profiles, results, oned, meas)
    print_tables(metrics)
    return metrics


def arms_profile(e, tw, dt, t_guess, lo_us=-1.0, hi_us=40.0):
    """Mean power of each arm (surface, firn, bed, total) vs twtt below the
    per-trace surface peak, rel own total surface peak."""
    Pt = np.abs(e[0] + e[1] + e[2]) ** 2
    t_s = rb.surface_peak_twtt(Pt, tw, t_guess, dt)
    rel = np.arange(lo_us, hi_us, dt * 1e6) * 1e-6
    out = {}
    for name, E in zip(("surface", "firn", "bed", "total"),
                       (e[0], e[1], e[2], e[0] + e[1] + e[2])):
        P = np.abs(E) ** 2
        acc = []
        for t in range(P.shape[0]):
            spk = Pt[t, int(np.searchsorted(tw, t_s[t]))]
            acc.append(np.interp(t_s[t] + rel, tw, P[t] / spk, left=np.nan,
                                 right=np.nan))
        out[name] = 10 * np.log10(np.maximum(np.nanmean(acc, 0), 1e-30))
    return rel * 1e6, out


def figures(case, out, prof, res, oned, meas):
    keys = [k for k in res if res[k]["construction"] == ("analytic" if case == "airborne" else "chirp")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for k in keys:
        d, db = prof[k]
        ax[0].plot(d, db, label=f"sim {res[k]['f0_MHz']}/{res[k]['B_MHz']} MHz")
    for p in meas:
        d, db = prof[f"measured_{p}"]
        ax[0].plot(d, db, "k", lw=2 if p == "qlook" else 1,
                   ls="-" if p == "qlook" else "--", label=f"measured {p} (195/30)")
    ax[0].set(xlim=(0, 200), ylim=(-70, 2), xlabel="depth below surface (m)",
              ylabel="dB rel own surface peak (mean power over traces)",
              title=f"{case}: 3-D sim, N=20 effective-contrast firn stack")
    ax[0].legend(fontsize=7)
    for fm, ls in ((195, "-"), (300, "--")):
        for bm in (10, 30, 100):
            d, db = prof[f"oned_f{fm}_b{bm}"]
            ax[1].plot(d, db, ls=ls, label=f"1-D TMM {fm}/{bm} MHz")
    ax[1].set(xlim=(0, 200), ylim=(-70, 2), xlabel="depth (m)",
              title="1-D full-res B26 TMM (C&S layered model), Hann band")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "profiles.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for band, mk in (("20-70m", "o"), ("40-100m", "s")):
        for fm, col in ((195, "C0"), (300, "C3")):
            bs = [b for b in ONED_B_MHZ]
            ax.plot(bs, [oned[f"f{fm}_b{b}"][band] for b in bs], col + mk + ":",
                    label=f"1-D {fm} MHz {band}")
            ks = [k for k in keys if res[k]["f0_MHz"] == fm]
            if ks:
                ax.plot([res[k]["B_MHz"] for k in ks],
                        [res[k]["plateau_db_rel_surface"][band] for k in ks],
                        col + mk + "-", ms=8, label=f"3-D sim {fm} MHz {band}")
        for p in meas:
            ax.plot([30], [meas[p][band]], "k" + mk, ms=10, mfc="none",
                    label=f"measured {p} {band}")
    b = np.array([5, 150.0])
    ax.plot(b, -20 - 10 * np.log10(b / 30), "k--", lw=0.8, label="1/B reference")
    ax.set(xscale="log", xlabel="bandwidth (MHz)",
           ylabel="plateau band level, dB rel surface peak",
           title=f"{case}: firn plateau vs bandwidth")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "plateau_vs_bandwidth.png", dpi=130)
    plt.close(fig)

    if case == "haps":
        fig, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 4),
                                sharey=True)
        for a, k in zip(np.atleast_1d(axs), keys):
            rel, arms = prof[k + "_arms"]
            for name, col in (("total", "k"), ("surface", "C0"), ("firn", "C2"),
                              ("bed", "C3")):
                a.plot(rel, arms[name], col, lw=1, label=name)
            a.set(title=f"{res[k]['f0_MHz']}/{res[k]['B_MHz']} MHz", xlabel="us below surface",
                  ylim=(-160, 5))
        np.atleast_1d(axs)[0].set_ylabel("dB rel surface peak (mean over traces)")
        np.atleast_1d(axs)[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "haps_arms.png", dpi=130)
        plt.close(fig)


def print_tables(m):
    print(f"\n== {m['case']}: plateau (dB rel surface, mean power) ==")
    print("pulse                 5-20  20-60  60-120  20-70  40-100  80-120 | no-firn 20-70 | bed/clut  bed/surf  firn/surf@bed | midcol tot/surf/firn")
    for k, r in m["pulses"].items():
        p, q = r["plateau_db_rel_surface"], r["plateau_no_firn_db"]
        mc = r["midcol_rel_surf_db"]
        print(f"{k:22s} {p['5-20m']:5.1f} {p['20-60m']:6.1f} {p['60-120m']:7.1f} "
              f"{p['20-70m']:6.1f} {p['40-100m']:7.1f} {p['80-120m']:7.1f} | "
              f"{q['20-70m']:13.1f} | {r['bed_over_clutter_db']:8.1f} "
              f"{r['bed_over_surface_only_db']:9.1f} {r['firn_over_surface_at_bed_db']:13.1f} | "
              f"{mc['total']:.1f}/{mc['surface']:.1f}/{mc['firn']:.1f}")
    for p, v in m["measured"].items():
        print(f"measured_{p:12s} " + " ".join(f"{v[b]:6.1f}" for b in
              ("5-20m", "20-60m", "60-120m", "20-70m", "40-100m", "80-120m")))
    print("== 1-D full-res TMM ==")
    for k, v in m["oned"].items():
        print(f"{k:12s} " + " ".join(f"{v[b]:6.1f}" for b in v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=list(CASES))
    ap.add_argument("--n-traces", type=int, default=None)
    ap.add_argument("--alt", type=float, default=None,
                    help="constant ellipsoidal platform height (m)")
    ap.add_argument("--along-m", type=float, default=rb.ALONG_M)
    ap.add_argument("--n-layers", type=int, default=20)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    run_case(a.case, a.n_traces, a.alt, a.along_m, a.n_layers, a.report_only)


if __name__ == "__main__":
    main()
