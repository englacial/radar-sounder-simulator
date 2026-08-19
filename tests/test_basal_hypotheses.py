"""Hypothesis-campaign knobs of tools/run_basal_clutter.py (T1 bed
roughness + its RSSNR double-count guard, T2 attenuation re-anchoring, T4
antenna pattern) and the cache-key backward compatibility that lets every
pre-campaign run stay cached. Config level only -- no network, no kernels."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_altitude_comparison as rac  # noqa: E402
import run_basal_clutter as rbc  # noqa: E402
from soundersim.config import AntennaConfig, RadarConfig, WaveformConfig  # noqa: E402

C = 299792458.0

# Pass and segment names come from the line definition, not from literals:
# these tests broke wholesale the last time the passes were renamed.
LINE = rbc.DEFAULT_LINE
rbc.activate_line(LINE)
ORDER = list(rbc.LINES[LINE]["ORDER"])
SYNTH = list(rbc.LINES[LINE]["SYNTHETIC_KEYS"])
SEGS = list(rbc.LINES[LINE]["SEGMENTS"])
CARRIER = rbc.LINE_SPECS[LINE].synthetic_passes[SYNTH[0]].carrier if SYNTH \
    else ORDER[0]
SEG = next(s for s in SEGS if s != "full_line")


def _rc():
    return RadarConfig(
        dt=1e-9, n_samples=64, t0=0.0, f0=rbc.FC_HZ,
        waveform=WaveformConfig(kind="chirp", bandwidth=50e6,
                                pulse_length=1e-5, window="hann"),
        antenna=AntennaConfig(kind="array", n_elements=5, spacing_lam=0.5,
                              roll_source="nav"))


def _p():
    """Minimal prep_pass-shaped dict for the cache-key builders."""
    return {"key": "low", "segment": "full", "picked_bed": False,
            "gamma_rssnr": True, "proc": True, "dgn": True,
            "rev": False, "parts": [("20161105_05_005", (1212, 3333))],
            "spacing": 10.6712, "reach": {"ct_m": 2529.4},
            "window": "hann", "rc_sim": _rc(),
            "aux": {"rssnr_gamma": {"k_db": 11.39},
                    "demogorgn": {"seed_id": 0, "snapshot_id": "WG80"}}}


# ------------------------------------------------------- T4 / T1 sim_cfg

def test_sim_cfg_antenna_and_bed_roughness_wiring():
    rc = _rc()
    base = rbc.sim_cfg(rc, 10.0, 15.0, True)
    assert base.radar.antenna.kind == "array"          # default untouched
    assert base.radar.antenna.n_elements == 5
    assert base.interfaces[1].roughness is None        # bed smooth by default
    assert base.interfaces[0].roughness.sigma_m == rac.SURF_ROUGH_SIGMA_M

    iso = rbc.sim_cfg(rc, 10.0, 15.0, True, antenna="isotropic")
    assert iso.radar.antenna.kind == "isotropic"
    # the pattern swap must not disturb anything else
    assert iso.radar.dt == rc.dt and iso.radar.f0 == rc.f0
    assert iso.interfaces[0].roughness.sigma_m == rac.SURF_ROUGH_SIGMA_M

    # the more-directive bracket: same cross-track array, wider aperture
    a8 = rbc.sim_cfg(rc, 10.0, 15.0, True, antenna="array8")
    assert a8.radar.antenna.kind == "array"
    assert a8.radar.antenna.n_elements == 8
    assert a8.radar.antenna.spacing_lam == rac.SPACING_LAM
    assert a8.radar.antenna.roll_source == "nav"

    rgh = rbc.sim_cfg(rc, 10.0, 15.0, True, bed_rough=(0.22, 0.886))
    assert rgh.interfaces[1].roughness.sigma_m == 0.22
    assert rgh.interfaces[1].roughness.corr_length_m == 0.886
    # bed roughness must not touch the SURFACE interface or the pattern
    assert rgh.interfaces[0].roughness.sigma_m == rac.SURF_ROUGH_SIGMA_M
    assert rgh.radar.antenna.kind == "array"


# --------------------------------------------- cache-key backward compat

def test_chunk_keys_unchanged_for_the_baseline_case():
    """The pre-campaign cache names and meta keys must be reproduced exactly
    when every hypothesis knob is at its default -- otherwise every existing
    run silently re-simulates."""
    p, rows = _p(), np.arange(198)
    rid = rbc.chunk_rid(p, 0, rac.ATT_DB_PER_KM, True)
    assert rid == "low_full_dgn_rssnr_proc_c00_srough"
    meta = rbc.chunk_meta(p, 0, rows, 17, 3365, rac.ATT_DB_PER_KM, True)
    assert "antenna" not in meta and "bed_rough" not in meta
    assert meta["att_db_per_km"] == rac.ATT_DB_PER_KM
    assert meta["demogorgn_seed"] == 0 and meta["rssnr_k_db"] == 11.39


def test_chunk_keys_separate_every_hypothesis_variant():
    p, rows = _p(), np.arange(198)
    base = rbc.chunk_meta(p, 0, rows, 17, 3365, 15.0, True)
    variants = {
        "t2": dict(att=31.0, antenna=rbc.ANT_DEFAULT, bed_rough=None),
        "t4": dict(att=15.0, antenna="isotropic", bed_rough=None),
        "t1": dict(att=15.0, antenna=rbc.ANT_DEFAULT,
                   bed_rough=(0.22, 0.886)),
    }
    rids, metas = set(), []
    for v in variants.values():
        rids.add(rbc.chunk_rid(p, 0, v["att"], True, v["antenna"],
                               v["bed_rough"]))
        metas.append(rbc.chunk_meta(p, 0, rows, 17, 3365, v["att"], True,
                                    v["antenna"], v["bed_rough"]))
    assert len(rids) == 3                       # distinct cache files
    assert all(m != base for m in metas)        # distinct cache keys
    assert rbc.chunk_rid(p, 0, 31.0, True).endswith("_srough_att31")
    assert "_antisotropic" in rbc.chunk_rid(p, 0, 15.0, True, "isotropic")
    assert metas[2]["bed_rough"] == [0.22, 0.886]


# ------------------------------------------- T1 roughness / double-count

def test_bed_rough_nadir_attenuation_matches_gerekos():
    """exp(-sigma^2 K^2) with K = 2 k_ice at nadir (buried facet -> LOCAL
    medium wavenumber), expressed in dB and returned negative."""
    for sigma in (0.0, 0.02, 0.1, 0.22):
        k_ice = 2.0 * np.pi * rbc.FC_HZ * np.sqrt(rac.EPS_ICE) / C
        want = 10.0 * np.log10(np.exp(-(sigma * 2.0 * k_ice) ** 2))
        assert rbc.bed_rough_nadir_db(sigma) == pytest.approx(want, rel=1e-12)
    assert rbc.bed_rough_nadir_db(0.0) == 0.0
    # monotone, and quadratic in sigma (doubling sigma quadruples the dB)
    assert rbc.bed_rough_nadir_db(0.2) == pytest.approx(
        4.0 * rbc.bed_rough_nadir_db(0.1), rel=1e-12)
    # the comfortable ceiling is lambda_ice/4
    assert rbc.LAM_ICE_M / 4.0 == pytest.approx(0.2216, abs=1e-3)


def _mapping_inputs(n=60):
    rng = np.random.default_rng(0)
    s = np.linspace(0.0, 60e3, n)
    thick = 600.0 + 100.0 * np.sin(s / 8e3)
    rssnr = 2.0 * 25.0 * thick / 1e3 + rng.normal(0.0, 1.0, n)
    return s, rssnr, thick, np.ones(n, bool)


def test_gamma_offset_shifts_the_whole_mapping():
    """The T1 guard rides on K: every recorded level moves by exactly the
    offset, the SHAPE is untouched, and the anchored median moves off the
    Fresnel constant by the offset (that is the point -- the kernel takes it
    straight back out as roughness attenuation)."""
    s, rssnr, thick, qc = _mapping_inputs()
    a = rbc.rssnr_gamma_profile(s, rssnr, thick, qc, 15.0, 0.0, 60e3)
    off = 12.5
    b = rbc.rssnr_gamma_profile(s, rssnr, thick, qc, 15.0, 0.0, 60e3,
                                g2_offset_db=off)
    assert b["k_db"] == pytest.approx(a["k_db"] + off, abs=1e-9)
    assert np.allclose(b["g2_db"], a["g2_db"] + off)
    assert b["g2_seg_db"]["med"] == pytest.approx(a["g2_const_db"] + off,
                                                  abs=0.05)
    assert b["censored_floor_db"] == pytest.approx(
        a["censored_floor_db"] + off, abs=0.02)
    # zero offset is the pre-campaign mapping, exactly
    z = rbc.rssnr_gamma_profile(s, rssnr, thick, qc, 15.0, 0.0, 60e3,
                                g2_offset_db=0.0)
    assert np.array_equal(z["g2_db"], a["g2_db"]) and z["k_db"] == a["k_db"]


def test_attenuation_re_anchors_k_but_not_the_median():
    """T2: raising A moves K (it absorbs 2*A*H) while the median-anchored
    level stays pinned; the along-track SPREAD grows because 2*A*H(s) does."""
    s, rssnr, thick, qc = _mapping_inputs()
    lo = rbc.rssnr_gamma_profile(s, rssnr, thick, qc, 15.0, 0.0, 60e3)
    hi = rbc.rssnr_gamma_profile(s, rssnr, thick, qc, 31.0, 0.0, 60e3)
    assert hi["k_db"] < lo["k_db"]
    assert hi["g2_seg_db"]["med"] == pytest.approx(lo["g2_seg_db"]["med"],
                                                   abs=0.05)
    spread = lambda d: d["g2_seg_db"]["p95"] - d["g2_seg_db"]["p5"]  # noqa: E731
    assert spread(hi) > spread(lo)
    # K - K_phys is the diagnostic of what the anchoring absorbed
    assert hi["k_minus_kphys_db"] < lo["k_minus_kphys_db"]


# ------------------------------------------------------ T3 posting_div

def _frame(n=9):
    import xarray as xr
    s = np.arange(n, dtype=np.float64)
    return xr.Dataset(
        {"Data": (("slow_time", "twtt"), np.arange(n * 4.0).reshape(n, 4)),
         "Latitude": ("slow_time", -75.0 + s * 1e-4),
         "Longitude": ("slow_time", -118.0 + s * 3e-4),
         "Elevation": ("slow_time", 900.0 + s),
         "Roll": ("slow_time", 0.01 * s),
         "Surface": ("slow_time", 3e-6 + s * 1e-9)},
        coords={"slow_time": s, "twtt": np.arange(4) * 2e-8})


def test_upsample_fsub_refines_geometry_and_keeps_endpoints():
    fs = _frame()
    bot = 1.1e-5 + np.arange(9) * 2e-9
    up, bot_up = rbc.upsample_fsub(fs, bot, 2)
    n = fs.sizes["slow_time"]
    assert up.sizes["slow_time"] == (n - 1) * 2 + 1 == 17
    assert len(bot_up) == 17
    # original traces are reproduced EXACTLY at the even positions
    for v in ("Latitude", "Longitude", "Elevation", "Roll", "Surface"):
        assert np.allclose(up[v].values[::2], fs[v].values, rtol=0, atol=1e-12)
    assert np.allclose(bot_up[::2], bot)
    # inserted traces are the midpoints (linear interpolation)
    assert np.allclose(up.Elevation.values[1::2],
                       0.5 * (fs.Elevation.values[:-1]
                              + fs.Elevation.values[1:]))
    assert np.allclose(bot_up[1::2], 0.5 * (bot[:-1] + bot[1:]))
    # div=1 is the identity on geometry
    up1, bot1 = rbc.upsample_fsub(fs, bot, 1)
    assert up1.sizes["slow_time"] == n and np.allclose(bot1, bot)


def test_posting_div_halves_the_spacing_and_doubles_the_aperture():
    """The T3 mechanism: refined posting -> alias-limited aperture doubles
    (sin(theta) = lam/(4*ds)), so the along-track resolution follows the
    posting while the range geometry is untouched."""
    lam, r = C / rbc.FC_HZ, 10000.0
    L1, th1 = rbc.alias_limited_aperture(lam, 14.85, r)
    L2, th2 = rbc.alias_limited_aperture(lam, 14.85 / 2, r)
    assert np.sin(np.radians(th2)) == pytest.approx(
        2.0 * np.sin(np.radians(th1)), rel=1e-9)
    assert L2 / L1 == pytest.approx(2.0, rel=0.01)


# -------------------------------------------------------- run() guards

def test_run_rejects_unsupported_hypothesis_combinations():
    # NOTE: the old "--out-name needs --companion-name" guard is gone -- the
    # constant-gamma arm runs inside the experiment now, so there is no
    # sibling directory to locate and nothing for --out-name to break.
    with pytest.raises(ValueError, match="out-name"):
        rbc.run(line=LINE, segment=SEG, out_name="t2", picked_bed=True,
                bed_ablation=True, companion=False)
    with pytest.raises(ValueError, match="bed-rough"):
        rbc.run(line=LINE, segment=SEG, bed_rough=(0.22, 0.886),
                gamma_rssnr=False)


# --------------------------------------- T5 specular/diffuse bed splitting

class _Base:
    """Minimal prep-scene stand-in for apply_rssnr_gamma."""

    crs = "EPSG:3031"

    def __init__(self, bed, transform):
        self.dems = [bed + 500.0, bed]
        self.transform = transform


def _axis_and_gmap(x0, y0, n=40, step=100.0):
    s = np.arange(n) * step
    axis = {"x": x0 + s, "y": np.full(n, y0), "s": s}
    gmap = {"s": s, "g2_db": np.full(n, -12.9)}
    return axis, gmap


def test_bed_tilt_matches_a_known_ramp():
    from affine import Affine
    tr = Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0)
    ny, nx = 12, 14
    slope = 0.1                                   # 10 % grade along +x
    bed = np.arange(nx)[None, :] * 20.0 * slope + np.zeros((ny, 1))
    psi = rbc.bed_tilt_rad(bed, tr)
    assert np.allclose(psi[1:-1, 1:-1], np.arctan(slope), atol=1e-12)
    assert np.allclose(rbc.bed_tilt_rad(np.zeros((ny, nx)), tr), 0.0)


def _apply(spec, slope=0.0):
    from affine import Affine
    tr = Affine(20.0, 0.0, 1e6, 0.0, -20.0, -1e6)
    ny, nx = 30, 140          # >= 100 axis samples inside the scene
    bed = np.arange(nx)[None, :] * 20.0 * slope + np.zeros((ny, 1))
    base = _Base(bed, tr)
    xs, ys = tr * (np.arange(nx) + 0.5, np.full(nx, 0.5))
    axis, gmap = _axis_and_gmap(xs[0], ys[0], n=nx, step=20.0)
    axis["x"], axis["y"] = np.asarray(xs), np.asarray(ys)
    axis["s"] = np.arange(nx) * 20.0
    gmap = {"s": axis["s"], "g2_db": np.full(nx, -12.9),
            "k_db": 0.0, "k_phys_db": 0.0, "k_minus_kphys_db": 0.0,
            "g2_seg_db": {}, "n_censored": 0,
            "provenance": {"snapshot_id": "x", "source": "y"}}
    stats = rbc.apply_rssnr_gamma(base, axis, gmap, spec)
    return base, stats


def test_specular_fraction_one_and_no_tilt_weight_is_the_unsplit_baseline():
    """The bit-identity gate at the tool level: f_s = 1, s0 = 0 must give
    exactly the unsplit gamma grid and attach NO diffuse map (so the kernel
    traces the pre-feature program)."""
    plain, _ = _apply(None)
    split, _ = _apply((1.0, 0.0, 1.0))
    assert np.array_equal(plain.gamma_bed, split.gamma_bed)
    assert getattr(plain, "diffuse_bed", None) is None
    assert getattr(split, "diffuse_bed", None) is None


def test_split_conserves_power_in_the_scene_mean():
    """<f_s * G_n> + (1 - f_s) == 1 for every f_s and s0: the tilt weight is
    mean-normalized (double-count guard), so the split conserves the SCENE
    MEAN bed power exactly -- on a flat bed that is also per-pixel."""
    plain, _ = _apply(None)
    ref = (plain.gamma_bed.astype(np.float64) ** 2).mean()
    for f_s in (0.0, 0.5, 0.9, 0.99):
        for s0, slope in ((1.0, 0.0), (3.0, 0.05), (1.0, 0.05)):
            b, _ = _apply((f_s, s0, 1.0), slope=slope)
            tot = b.gamma_bed.astype(np.float64) ** 2
            if f_s < 1.0:
                tot = tot + b.diffuse_bed.astype(np.float64) ** 2
            pl, _ = _apply(None, slope=slope)
            assert tot.mean() == pytest.approx(
                (pl.gamma_bed.astype(np.float64) ** 2).mean(), rel=1e-6)
    assert ref > 0


def test_tilt_weight_darkens_the_specular_channel_only():
    """On a tilted bed the specular share is cut by G(psi) while the diffuse
    share keeps its full (1-f_s): the 'bright because flat' mechanism."""
    f_s, s0 = 0.9, 1.0
    flat, _ = _apply((f_s, s0, 1.0), slope=0.0)
    tilt, st = _apply((f_s, s0, 1.0), slope=np.tan(np.deg2rad(3.0)))
    i = (slice(2, -2), slice(2, -2))
    # a UNIFORMLY tilted bed normalizes back to <G> = 1, so the specular
    # level is unchanged: the weight is a relative flat-vs-tilted contrast
    ratio = (tilt.gamma_bed[i] ** 2) / (flat.gamma_bed[i] ** 2)
    assert np.allclose(ratio, 1.0, rtol=1e-5)
    assert np.allclose(tilt.diffuse_bed[i], flat.diffuse_bed[i], rtol=1e-6)
    assert st["spec_diffuse"]["specular_fraction"] == f_s
    assert st["spec_diffuse"]["bed_tilt_deg"]["med"] == pytest.approx(3.0,
                                                                      abs=0.1)


def test_tilt_weight_makes_flat_facets_brighter_than_tilted_ones():
    """Within ONE scene the specular weight is the flat-vs-tilted contrast
    exp(-tan^2(psi)/(2 s0^2)) -- the 'bright because flat' mechanism (the
    mean normalization only sets the overall level)."""
    from affine import Affine
    tr = Affine(20.0, 0.0, 1e6, 0.0, -20.0, -1e6)
    ny, nx, s0 = 30, 140, 3.0
    ramp = np.tan(np.deg2rad(6.0)) * 20.0
    z = np.zeros((ny, nx))
    z[:, nx // 2:] = ramp * np.arange(nx - nx // 2)[None, :]   # tilted half
    base = _Base(z, tr)
    xs, ys = tr * (np.arange(nx) + 0.5, np.full(nx, 0.5))
    axis = {"x": np.asarray(xs), "y": np.asarray(ys),
            "s": np.arange(nx) * 20.0}
    gmap = {"s": axis["s"], "g2_db": np.full(nx, -12.9), "k_db": 0.0,
            "k_phys_db": 0.0, "k_minus_kphys_db": 0.0, "g2_seg_db": {},
            "n_censored": 0, "provenance": {"snapshot_id": "x",
                                            "source": "y"}}
    rbc.apply_rssnr_gamma(base, axis, gmap, (1.0, s0, 1.0))
    flat = base.gamma_bed[5, 5] ** 2
    tilted = base.gamma_bed[5, -5] ** 2
    want = np.exp(-np.tan(np.deg2rad(6.0)) ** 2
                  / (2.0 * np.tan(np.deg2rad(s0)) ** 2))
    assert tilted / flat == pytest.approx(want, rel=1e-4)
    assert tilted < flat


def test_spec_diffuse_enters_the_cache_key_only_when_on():
    p, rows = _p(), np.arange(198)
    assert "spec_diffuse" not in rbc.chunk_meta(p, 0, rows, 17, 3365, 31.0,
                                                True)
    m = rbc.chunk_meta(p, 0, rows, 17, 3365, 31.0, True, spec=(0.9, 1.0, 1.0))
    assert m["spec_diffuse"] == [0.9, 1.0, 1.0]
    rid = rbc.chunk_rid(p, 0, 31.0, True, spec=(0.9, 1.0, 1.0))
    assert rid.endswith("_fs0.9_s01_n1")
    assert rbc.chunk_rid(p, 0, 31.0, True) != rid


# ------------------------------------------------ K anchoring (level mode)

def _fake_rssnr(monkeypatch, n=80):
    """Synthetic anchor samples so build_rssnr_gamma runs without network."""
    from pyproj import Transformer
    s = np.linspace(0.0, 60e3, n)
    x, y = -1.2e6 + s, np.full(n, -6.0e5)
    lon, lat = Transformer.from_crs("EPSG:3031", "EPSG:4326",
                                    always_xy=True).transform(x, y)
    thick = 600.0 + 100.0 * np.sin(s / 8e3)
    stw = np.full(n, 3e-6)
    btw = stw + 2.0 * thick * np.sqrt(rac.EPS_ICE) / C
    rng = np.random.default_rng(0)
    d = {"lon": lon, "lat": lat, "stw": stw, "btw": btw,
         "rssnr": 2.0 * 25.0 * thick / 1e3 + rng.normal(0.0, 1.0, n),
         "qc": np.ones(n, bool)}
    monkeypatch.setattr(rbc, "fetch_rssnr_anchor",
                        lambda *a, **k: (d, {"snapshot_id": "S",
                                             "source": "test"}))
    monkeypatch.setattr(rbc, "segment_s_range", lambda *a: (0.0, 60e3))
    return {"x": x, "y": y, "s": s, "eps_ice": rac.EPS_ICE}


def test_level_anchor_raises_k_by_the_recorded_deficit(monkeypatch):
    """--anchor level must move K (and every level statistic) by exactly the
    deficit, leaving the SHAPE of the mapped profile untouched; --anchor
    median must be bit-identical to the pre-feature mapping."""
    axis = _fake_rssnr(monkeypatch)
    med = rbc.build_rssnr_gamma(axis, "full", 31.0)
    # D is stated by the caller: it is solved against a particular run at a
    # particular attenuation, so there is no line-level default to read.
    d = 14.8
    lvl = rbc.build_rssnr_gamma(axis, "full", 31.0, anchor="level",
                                level_deficit_db=d)
    assert med["anchor"] == "median" and "level_anchor" not in med
    assert lvl["k_db"] == pytest.approx(med["k_db"] + d, abs=0.01)
    assert np.allclose(lvl["g2_db"], med["g2_db"] + d, atol=1e-9)
    for k in ("min", "p5", "med", "p95", "max"):
        assert lvl["g2_seg_db"][k] == pytest.approx(
            med["g2_seg_db"][k] + d, abs=0.06)
    la = lvl["level_anchor"]
    assert la["deficit_db"] == pytest.approx(d, abs=0.01)
    assert la["k_level_db"] == lvl["k_db"]
    assert la["k_median_db"] == pytest.approx(med["k_db"], abs=0.01)
    # raising K raises the implied reflectivity: the discriminator diagnostic
    assert lvl["g2_pos_frac_seg"] >= med["g2_pos_frac_seg"]


def test_level_anchor_deficit_override_and_bad_mode(monkeypatch):
    axis = _fake_rssnr(monkeypatch)
    med = rbc.build_rssnr_gamma(axis, "full", 31.0)
    lvl = rbc.build_rssnr_gamma(axis, "full", 31.0, anchor="level",
                                level_deficit_db=6.0)
    assert lvl["k_db"] == pytest.approx(med["k_db"] + 6.0, abs=0.01)
    assert lvl["level_anchor"]["source"] == "supplied"
    with pytest.raises(ValueError, match="anchor"):
        rbc.build_rssnr_gamma(axis, "full", 31.0, anchor="nonsense")


def test_level_anchor_composes_with_the_bed_roughness_guard(monkeypatch):
    """The two K offsets are independent and additive."""
    axis = _fake_rssnr(monkeypatch)
    base = rbc.build_rssnr_gamma(axis, "full", 31.0)
    both = rbc.build_rssnr_gamma(axis, "full", 31.0, anchor="level",
                                 level_deficit_db=14.8,
                                 bed_rough_sigma=0.05, extra_db=-1.0)
    want = (14.8 - rbc.bed_rough_nadir_db(0.05) - 1.0)
    assert both["k_db"] == pytest.approx(base["k_db"] + want, abs=0.02)


# ------------------------------------------------- syn500km orbital pass



def test_syn500km_geometry_scales_as_expected():
    """Derived reach and facet spacing at 500 km: the surface interface
    binds the reach (its target delay spans the whole ice column) and the
    bed's in-ice Fresnel zone sets the spacing."""
    h, dbs, d = 5.0e5, 10.6e-6, 500.0
    r = rbc.derive_reach(h, dbs, d)
    assert r["ct_m"] == pytest.approx(r["surface_reach_m"])
    assert 40e3 < r["ct_m"] < 50e3
    assert r["bed_reach_m"] < r["surface_reach_m"]
    lam = C / rbc.FC_HZ
    sp = rac.facet_spacing(lam, h, 683.0) * 0.7
    assert 200.0 < sp < 260.0
    # aperture at the orbital bed range, product posting
    L, th = rbc.alias_limited_aperture(lam, 14.85, 5.007e5)
    assert 25e3 < L < 28e3
    assert th == pytest.approx(1.522, abs=0.01)


# --------------------------------------------- EXTENDED segment (0-69.7 km)

def test_extended_segment_table_is_a_superset_of_the_full_segment():
    """Every pass gains an 'extended' entry whose parts CONTAIN the full
    segment's parts of the same frame (the study window only grows), in
    increasing-s order, with matching trace counts across the triplet."""
    assert set(("pilot", "full", "extended", "full_line")) <= set(SEGS)
    counts = {}
    for key in ORDER:
        ext = rbc.PASSES[key]["extended"]
        full = dict(rbc.PASSES[key]["full"])
        assert ext, key
        for fid, (a, b) in ext:
            assert 0 <= a < b, (key, fid)
            if fid in full:
                fa, fb = full[fid]
                assert a <= fa and b >= fb, (key, fid)   # only grows
        # the extension adds at most one new frame per pass
        assert len(set(dict(ext)) - set(full)) <= 1
        counts[key] = sum(b - a for _, (a, b) in ext)
    n = np.array(list(counts.values()), float)
    assert (n.max() - n.min()) / n.mean() < 0.002       # 4692/4696/4698
    # the synthetic passes re-fly the LOW pass line on every segment
    for skey in SYNTH:
        for seg in SEGS:
            assert rbc.PASSES[skey][seg] == rbc.PASSES[CARRIER][seg]
    # every segment is fully parameterised
    for seg in SEGS:
        assert seg in rbc.S0_KM and seg in rbc.DECOMP_S_KM
    assert rbc.S0_KM["extended"] == 0.0
    assert rbc.N_TRACES_BY_SEGMENT['extended'] > rbc.N_TRACES_BY_SEGMENT['full']


def test_extended_cache_names_are_distinct_from_the_full_segment():
    """The segment is part of the chunk cache name AND key, so the extended
    run cannot collide with (or silently reuse) the 50 km caches."""
    p_full = _p()
    p_ext = {**_p(), "segment": "extended"}
    n_full = rbc.chunk_rid(p_full, 0, 20.0, True)
    n_ext = rbc.chunk_rid(p_ext, 0, 20.0, True)
    assert n_full != n_ext
    assert "_full_" in n_full and "_extended_" in n_ext
    rows = np.arange(10)
    assert rbc.chunk_meta(p_ext, 0, rows, 1, 10, 20.0, True)["segment"] \
        == "extended"


def test_extended_k_anchor_reuses_the_full_segment_mapping(monkeypatch):
    """The extended run must NOT re-derive K on the longer line: with
    k_anchor_segment='full' the mapped profile is bit-identical to the 50 km
    run's, and the run-segment statistics are recorded separately."""
    axis = _fake_rssnr(monkeypatch)
    monkeypatch.setattr(rbc, "segment_s_range",
                        lambda ref, seg: {"full": (18e3, 68e3),
                                          "extended": (0.0, 69.7e3)}[seg])
    full = rbc.build_rssnr_gamma(axis, "full", 20.0, anchor="level",
                                 level_deficit_db=3.56)
    ext = rbc.build_rssnr_gamma(axis, "extended", 20.0, anchor="level",
                                level_deficit_db=3.56,
                                k_anchor_segment="full")
    assert ext["k_db"] == full["k_db"]
    assert np.array_equal(ext["g2_db"], full["g2_db"])
    assert ext["k_anchor_segment"] == "full"
    assert ext["seg_s_km"] == full["seg_s_km"]
    # the run segment's own statistics are recorded, not used for K
    assert ext["g2_run_seg_db"]["seg_s_km"] == [0.0, 69.7]
    assert ext["g2_run_seg_db"]["n_seg"] > full["n_seg"]
    assert "g2_run_seg_db" not in full
    # ... and re-deriving on the extended segment WOULD move K (why we don't)
    naive = rbc.build_rssnr_gamma(axis, "extended", 20.0, anchor="level",
                                  level_deficit_db=3.56)
    assert naive["k_anchor_segment"] == "extended"
    assert naive["k_db"] != full["k_db"]
    assert rbc.K_ANCHOR_SEGMENT["extended"] == "full"


# ------------------------------------------- single-trace decomposition

def _synthetic_pass(n_tr=6, n_s=900, t_s_us=3.0, dbs_us=9.0):
    """Minimal synthetic-pass (p, sim) pair: clean surface and bed impulses
    on separate layers, no measured side (p['synthetic'] set)."""
    dt, t0 = 20.202e-9, 0.0
    tw = t0 + np.arange(n_s) * dt
    t_s = np.full(n_tr, t_s_us * 1e-6)
    t_b = t_s + dbs_us * 1e-6
    field = np.zeros((n_tr, n_s, 2), np.complex64)
    for j, w in enumerate([0.05, 0.3, 1.0, 0.5, 0.2]):
        field[:, int(round(t_s_us * 1e-6 / dt)) - 2 + j, 0] = w
        field[:, int(round((t_s_us + dbs_us) * 1e-6 / dt)) - 2 + j, 1] = \
            0.02 * w
    sim = {"field": field, "twtt": tw,
           "nadir": np.column_stack([t_s, t_b])}
    rc = RadarConfig(dt=dt, n_samples=n_s, t0=t0, f0=rbc.FC_HZ,
                     waveform=WaveformConfig(kind="chirp", bandwidth=50e6,
                                             pulse_length=1e-5,
                                             window="hann"),
                     antenna=AntennaConfig(kind="array", n_elements=5,
                                           spacing_lam=0.5,
                                           roll_source="nav"))
    s = np.linspace(0.0, 5000.0, n_tr)
    p = {"key": SYNTH[0], "segment": "extended", "rc_frame": rc,
         "spacing": 10.0, "surf_sim": t_s, "s_sim": s, "s_m": s,
         "surf": t_s, "bot": t_b, "tw_m": tw, "dt": dt,
         "synthetic": {"agl_med_m": 30000.0}, "h_med": 30000.0,
         "thick_med": 700.0}
    return p, sim


def test_single_trace_decomposition_records_a_parameterised_location():
    p, sim = _synthetic_pass()
    a = rbc.analyze_pass(p, sim, trace_s_km=3.0)
    ti, tp = a["trace_info"], a["trace_profs"]
    assert ti["requested_s_km"] == 3.0
    # nearest sim trace to 3.0 km on a 0..5 km, 6-trace grid = index 3
    assert ti["sim_trace_index"] == 3
    assert ti["sim_s_km"] == pytest.approx(3.0, abs=0.6)
    assert "measured" not in tp                  # synthetic: no measured data
    assert ti["bed_below_surface_us"] == pytest.approx(9.0, abs=0.05)
    # the bed window is pure bed returns here -> a very large guard margin
    assert ti["bed_window_bed_minus_surface_returns_db"] > 10.0
    # the curves are the per-interface fields at that one trace
    for k in ("sim_total", "sim_surface", "sim_bed"):
        rel, db = tp[k]
        assert rel[0] == pytest.approx(-1.5, abs=0.05)
        assert len(rel) == len(db)
    rel, sdb = tp["sim_surface"]
    _, bdb = tp["sim_bed"]
    assert rel[int(np.argmax(sdb))] == pytest.approx(0.0, abs=0.05)
    assert rel[int(np.argmax(bdb))] == pytest.approx(9.0, abs=0.05)
    # the location is a parameter: moving it moves the trace
    a2 = rbc.analyze_pass(p, sim, trace_s_km=0.0)
    assert a2["trace_info"]["sim_trace_index"] == 0
    # ... and omitting it costs nothing
    a3 = rbc.analyze_pass(p, sim)
    assert a3["trace_info"] is None and a3["trace_profs"] is None
    assert a3["sim"]["bed_rel_surf_db"] == a["sim"]["bed_rel_surf_db"]


def test_fig_decomposition_trace_renders_and_names_the_location(tmp_path):
    p, sim = _synthetic_pass()
    a = rbc.analyze_pass(p, sim, trace_s_km=3.0)
    fp = rbc.fig_decomposition_trace(tmp_path, {SYNTH[0]: p},
                                     {SYNTH[0]: a}, keys=[SYNTH[0]])
    assert fp is not None and fp.exists() and fp.name \
        == "decomposition_trace.png"
    # passes without the single-trace analysis are skipped, not crashed on
    a0 = rbc.analyze_pass(p, sim)
    assert rbc.fig_decomposition_trace(tmp_path, {SYNTH[0]: p},
                                       {SYNTH[0]: a0},
                                       keys=[SYNTH[0]]) is None


# ---------------------------- FULL_LINE segment + hybrid bed (0-148.45 km)

def test_full_line_segment_table_spans_the_whole_line():
    """Every pass gains a 'full_line' entry that CONTAINS its 'extended'
    parts (the window only grows through the GL), in increasing-s order
    after reversal, with trace counts matching across the triplet."""
    counts = {}
    for key in ORDER:
        line = rbc.PASSES[key]["full_line"]
        ext = dict(rbc.PASSES[key]["extended"])
        for fid, (a, b) in line:
            assert 0 <= a < b, (key, fid)
            if fid in ext:
                ea, eb = ext[fid]
                assert a <= ea and b >= eb, (key, fid)   # only grows
        counts[key] = sum(b - a for _, (a, b) in line)
    n = np.array(list(counts.values()), float)
    assert 9990 < n.min() and n.max() < 10010          # 9993/10004/10006
    assert (n.max() - n.min()) / n.mean() < 0.002
    for skey in SYNTH:
        assert rbc.PASSES[skey]["full_line"] == rbc.PASSES[CARRIER]["full_line"]
    assert rbc.S0_KM["full_line"] == 0.0
    assert rbc.K_ANCHOR_SEGMENT["full_line"] == "full"
    assert rbc.N_TRACES_BY_SEGMENT['full_line'] > rbc.N_TRACES_BY_SEGMENT['extended']
    # the default single-trace decomposition pair: one grounded, one floating
    lo, hi = rbc.DECOMP_S_KM["full_line"]
    assert lo < rbc.GL_S_KM < hi
    # blend geometry: the ramp starts AT the GL (grounded side pure DEMOGORGN)
    assert 2.0 <= rbc.GL_RAMP_KM <= 5.0
    assert rbc.GL_S_KM == 69.7


def test_full_line_cache_names_and_keys_are_distinct():
    """Segment name AND the hybrid marker separate the full_line caches from
    every earlier run; the baseline names stay byte-identical (p without a
    'hybrid' key -> no suffix, no meta key)."""
    p_base = _p()
    p_line = {**_p(), "segment": "full_line", "hybrid": True}
    rid_b = rbc.chunk_rid(p_base, 0, 20.0, True)
    rid_l = rbc.chunk_rid(p_line, 0, 20.0, True)
    assert rid_b != rid_l
    assert "_full_line_" in rid_l and "_hyb" in rid_l
    assert "_hyb" not in rid_b
    rows = np.arange(10)
    m_b = rbc.chunk_meta(p_base, 0, rows, 1, 10, 20.0, True)
    m_l = rbc.chunk_meta(p_line, 0, rows, 1, 10, 20.0, True)
    assert "hybrid_bed" not in m_b
    assert m_l["hybrid_bed"]["gl_s_km"] == rbc.GL_S_KM
    assert m_l["hybrid_bed"]["ramp_km"] == rbc.GL_RAMP_KM
    assert m_l["segment"] == "full_line"


def test_full_line_k_anchor_reuses_the_full_mapping_with_zone_stats(
        monkeypatch):
    """K stays pinned to the 50 km 'full' segment (bit-identical profile);
    the zone-aware physicality block judges grounded samples vs 0 dB and
    floating samples vs the ice-seawater Fresnel ceiling."""
    axis = _fake_rssnr(monkeypatch)
    monkeypatch.setattr(rbc, "segment_s_range",
                        lambda ref, seg: {"full": (18e3, 68e3),
                                          "full_line": (0.0, 148.45e3)}[seg])
    full = rbc.build_rssnr_gamma(axis, "full", 20.0, anchor="level",
                                 level_deficit_db=3.56)
    line = rbc.build_rssnr_gamma(axis, "full_line", 20.0, anchor="level",
                                 level_deficit_db=3.56,
                                 k_anchor_segment="full",
                                 zone_gl_km=rbc.GL_S_KM)
    assert line["k_db"] == full["k_db"]
    assert np.array_equal(line["g2_db"], full["g2_db"])
    assert line["k_anchor_segment"] == "full"
    z = line["g2_zones_db"]
    from soundersim.physics import fresnel_normal
    want_ceil = 20.0 * np.log10(abs(
        fresnel_normal(rac.EPS_ICE, rbc.EPS_SEAWATER)))
    assert z["floating_ceiling_db"] == pytest.approx(want_ceil, abs=0.01)
    assert z["grounded_fresnel_anchor_db"] == pytest.approx(-12.86, abs=0.05)
    assert z["grounded"]["n"] >= 3
    assert 0.0 <= z["grounded"]["frac_above_0db"] <= 1.0
    assert (z["grounded"]["frac_above_seawater_ceiling"]
            >= z["grounded"]["frac_above_0db"])
    # the fake dataset stops at 60 km: the floating zone must degrade
    # gracefully to a counted too-few-samples record, never crash
    assert z["floating"]["n"] == 0 and "note" in z["floating"]
    assert "g2_zones_db" not in full


def test_picks_bed_nn_skips_gaps_and_clamps_edges():
    axis = {"s": np.array([0.0, 10.0, 20.0, 30.0]),
            "bed": np.array([1.0, np.nan, 3.0, 4.0])}
    got = rbc.picks_bed_nn(axis, np.array([-5.0, 4.0, 9.0, 11.0, 26.0, 99.0]))
    # finite picks at s = 0/20/30 -> nearest-FINITE selection, edge-clamped
    assert np.array_equal(got, [1.0, 1.0, 1.0, 3.0, 4.0, 4.0])


def test_zone_g2_stats_judges_each_zone_against_its_own_ceiling():
    n = 200
    s = np.linspace(0.0, 148.45e3, n)
    g2 = np.where(s < rbc.GL_S_KM * 1e3, -10.0, -2.0)   # floating brighter
    gmap = {"s": s, "g2_db": g2, "ok": np.ones(n, bool),
            "g2_const_db": -12.86}
    z = rbc.zone_g2_stats(gmap, 0.0, 148.45e3)
    assert z["grounded"]["med"] == pytest.approx(-10.0)
    assert z["floating"]["med"] == pytest.approx(-2.0)
    assert z["grounded"]["frac_above_0db"] == 0.0
    # -2 dB beats the ~-3.5 dB seawater ceiling but not 0 dB
    assert z["floating"]["frac_above_seawater_ceiling"] == 1.0
    assert z["floating"]["frac_above_0db"] == 0.0
    assert z["floating"]["med_minus_zone_ceiling_db"] == pytest.approx(
        1.5, abs=0.1)
    assert z["grounded"]["n_total"] == z["grounded"]["n"]
    assert z["grounded"]["qc_pass_frac"] == 1.0


def test_apply_hybrid_bed_blends_demogorgn_into_the_floating_picks(
        monkeypatch):
    """Synthetic straight-line scene: pure DEMOGORGN before the GL, pure
    picks past GL + ramp, linear in between; clearance/clamp and the
    blend-zone step recorded; the DEMOGORGN fetch is restricted to the
    grounded(+ramp) track."""
    from types import SimpleNamespace

    from affine import Affine
    from pyproj import Transformer

    gl_m = rbc.GL_S_KM * 1e3
    ramp_m = rbc.GL_RAMP_KM * 1e3
    x0, y0 = -1.45e6, -5.6e5              # Amundsen-ish EPSG:3031 corner
    step = 500.0
    nx, ny = 300, 5                       # 150 km x 2.5 km scene
    tr = Affine(step, 0.0, x0, 0.0, -step, y0)
    xs = x0 + (np.arange(nx) + 0.5) * step
    yc = y0 - (ny // 2 + 0.5) * step
    s_ax = xs - xs[0]
    axis = {"x": xs, "y": np.full(nx, yc), "s": s_ax,
            "bed": np.full(nx, -600.0)}
    axis["bed"][nx - 3] = np.nan          # one floating pick gap
    surface = np.zeros((ny, nx), np.float32) + 100.0
    base = SimpleNamespace(dems=[surface, np.zeros((ny, nx), np.float32)],
                           transform=tr, crs="EPSG:3031", params={})
    inv = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(xs, np.full(nx, yc))
    import xarray as xr
    fsub = xr.Dataset({"Latitude": ("slow_time", lat),
                       "Longitude": ("slow_time", lon)})
    seen = {}

    def fake_fetch(bounds, pad_m=0.0, seed=0, **kw):
        seen["bounds"] = bounds
        seen["pad_m"] = pad_m
        dgn = np.full((ny, nx), -500.0, np.float32)
        return dgn, tr, "EPSG:3031", {"posting_m": 500.0,
                                      "returned_datum": "test"}

    import soundersim.opr as opr
    monkeypatch.setattr(opr, "fetch_demogorgn_window", fake_fetch)
    st = rbc.apply_hybrid_bed(base, fsub, 600.0, 0, axis)
    bed = np.asarray(base.dems[1], np.float64)
    s_pix = np.broadcast_to(xs - xs[0], (ny, nx))
    assert np.allclose(bed[s_pix < gl_m - step], -500.0, atol=1e-3)
    assert np.allclose(bed[s_pix > gl_m + ramp_m + step], -600.0, atol=1e-3)
    mid = (s_pix > gl_m + 0.3 * ramp_m) & (s_pix < gl_m + 0.7 * ramp_m)
    assert (bed[mid] > -600.0).all() and (bed[mid] < -500.0).all()
    # recorded stats
    h = st["hybrid"]
    assert h["gl_s_km"] == rbc.GL_S_KM and h["ramp_km"] == rbc.GL_RAMP_KM
    assert h["blend_zone_dgn_minus_picks_m"]["med"] == pytest.approx(
        100.0, abs=1.0)
    assert h["clearance_m"]["min"] > 0 and st["bed_clamp_frac"] == 0.0
    # floating zone INCLUDES the ramp, whose start still sits near the
    # DEMOGORGN level: min clearance ~600 m (ramp start), median ~700 (picks)
    assert 595.0 <= h["clearance_m"]["floating_min"] <= 700.0
    assert h["clearance_m"]["floating_med"] == pytest.approx(700.0, abs=5.0)
    assert h["floating_picks"]["gap_frac"] > 0
    assert st["seed_id"] == 0
    # the DEMOGORGN fetch saw only the grounded(+ramp+2km) track
    fx, _ = Transformer.from_crs("EPSG:4326", "EPSG:3031",
                                 always_xy=True).transform(
        [seen["bounds"][0], seen["bounds"][2]],
        [seen["bounds"][1], seen["bounds"][3]])
    assert max(fx) - xs[0] <= gl_m + ramp_m + 2000.0 + step
    assert h["demogorgn_fetch_track_max_s_km"] <= rbc.GL_S_KM \
        + rbc.GL_RAMP_KM + 2.1


def test_analyze_pass_accepts_multiple_trace_locations():
    p, sim = _synthetic_pass()
    a = rbc.analyze_pass(p, sim, trace_s_km=[1.0, 4.0])
    assert len(a["trace_info_list"]) == 2
    assert a["trace_info"] == a["trace_info_list"][0]
    assert a["trace_profs"] is a["trace_profs_list"][0]
    assert a["trace_info_list"][0]["requested_s_km"] == 1.0
    assert a["trace_info_list"][1]["requested_s_km"] == 4.0
    assert a["trace_info_list"][0]["sim_trace_index"] \
        != a["trace_info_list"][1]["sim_trace_index"]
    # scalar stays a one-element list (backward-compatible figure/config)
    a1 = rbc.analyze_pass(p, sim, trace_s_km=3.0)
    assert len(a1["trace_info_list"]) == 1


def test_fig_decomposition_trace_fans_out_multi_locations(tmp_path):
    p, sim = _synthetic_pass()
    a = rbc.analyze_pass(p, sim, trace_s_km=[1.0, 4.0])
    fp = rbc.fig_decomposition_trace(tmp_path, {SYNTH[0]: p},
                                     {SYNTH[0]: a}, keys=[SYNTH[0]])
    assert fp is not None and fp.exists()


def test_run_rejects_full_line_without_the_hybrid_bed():
    with pytest.raises(ValueError, match="HYBRID"):
        rbc.run(segment="full_line")
    with pytest.raises(ValueError, match="ablation"):
        rbc.run(segment="full_line", demogorgn_bed=True, bed_ablation=True)


# ---------------- altitude-campaign synthetics (syn14km / syn300km) + figs



def test_new_synthetic_altitude_geometry_scales():
    """Derived geometry at the two new altitudes: reach and alias-limited
    aperture bracket between the airborne high pass and syn500km."""
    lam = C / rbc.FC_HZ
    r14 = rbc.derive_reach(14.0e3, 12.4e-6, 421.0)
    assert 8e3 < r14["ct_m"] < 11e3               # vs high's ~7.4 km
    r300 = rbc.derive_reach(300.0e3, 12.4e-6, 421.0)
    assert 33e3 < r300["ct_m"] < 42e3             # vs syn500km's ~45-49 km
    assert r300["ct_m"] == pytest.approx(r300["surface_reach_m"])
    L14, th14 = rbc.alias_limited_aperture(lam, 14.85, 14.7e3)
    L300, th300 = rbc.alias_limited_aperture(lam, 14.85, 3.007e5)
    assert th14 == pytest.approx(1.522, abs=0.01)
    assert th14 == pytest.approx(th300, abs=1e-6)  # posting-set half-angle
    assert 700.0 < L14 < 900.0
    assert 15e3 < L300 < 17e3


def test_frame_span_and_source_label():
    assert rbc.frame_span([("20161031_07_005", (0, 1)),
                           ("20161031_07_004", (0, 1)),
                           ("20161031_07_003", (0, 1)),
                           ("20161031_07_002", (0, 1))]) \
        == "20161031_07_002-005"
    assert rbc.frame_span([("20161105_05_005", (0, 1))]) == "20161105_05_005"
    assert rbc.frame_span([("20161105_05_005", (0, 1)),
                           ("20161028_05_006", (0, 1))]) \
        == "20161105_05_005/20161028_05_006"
    hi_key = ORDER[-1]
    p_meas = {"parts": rbc.PASSES[hi_key]["full_line"], "h_med": 10763.0}
    lbl = rbc.source_label(hi_key, p_meas)
    assert lbl == ("2016_Antarctica_DC8 - measured 20161031_07_002-005 "
                   "(10.8 km AGL)")
    p_syn = {"parts": rbc.PASSES[CARRIER]["full_line"],
             "synthetic": {"synthetic_msl_m": 14000.0}}
    lbl_s = rbc.source_label(SYNTH[0], p_syn)
    assert "SYNTHETIC 14 km" in lbl_s
    assert "20161105_05_005-007" in lbl_s and "no measured data" in lbl_s


def test_emit_pass_figs_writes_suffixed_labeled_files(tmp_path):
    """The staged-delivery figure set: separate per-pass files, written from
    one pass's analysis alone (no other pass needed)."""
    p, sim = _synthetic_pass()
    p["parts"] = rbc.PASSES[CARRIER]["full_line"]
    p["reach"] = {"ct_m": 2500.0}
    a = rbc.analyze_pass(p, sim, trace_s_km=3.0)
    figs = rbc.emit_pass_figs(tmp_path, SYNTH[0], p, a, None, "extended",
                              None, None, "demogorgn")
    names = sorted(f.name for f in figs)
    k = SYNTH[0]
    assert names == sorted([f"bed_tail_{k}.png", f"decomposition_{k}.png",
                            f"decomposition_trace_{k}.png",
                            f"radargrams_{k}.png"])
    assert all(f.exists() for f in figs)
    # the crop knob is plot-only and accepted end-to-end
    fp = rbc.fig_radargrams(tmp_path, {SYNTH[0]: p}, {SYNTH[0]: a},
                            "extended", keys=[SYNTH[0]], plot_s_max_km=3.0,
                            fname="radargrams_crop.png", src="SRC LINE")
    assert fp.exists() and fp.name == "radargrams_crop.png"


# ----------------------------------- processed-stack cache (--proc-cache)

def test_proc_cache_ids_digests_and_staleness(tmp_path):
    import json as _json
    p = {**_p(), "segment": "full_line", "key": "high", "hybrid": True}
    assert rbc.proc_rid(p) == "high_full_line_dgn_rssnr_proc_hyb"
    p_plain = _p()
    assert rbc.proc_rid(p_plain) == "low_full_dgn_rssnr_proc"
    runs = tmp_path / "runs"
    runs.mkdir()
    for ci in range(2):
        rid = rbc.chunk_rid(p, ci, 20.0, True)
        (runs / f"{rid}.json").write_text(
            _json.dumps({"meta_key": f"meta-{ci}"}))
    d = rbc.chunk_digests(p, runs, 2, 20.0, True)
    assert len(d) == 2 and all(len(v) == 16 for v in d.values())
    # unchanged chunk cache -> digests reproduce
    assert rbc._digests_current(d, runs) == d
    # a re-simulated chunk (different meta_key) flips its digest
    rid0 = rbc.chunk_rid(p, 0, 20.0, True)
    (runs / f"{rid0}.json").write_text(_json.dumps({"meta_key": "CHANGED"}))
    cur = rbc._digests_current(d, runs)
    assert cur[rid0] != d[rid0]
    # a missing chunk cache reads as None (stale, never silently served)
    (runs / f"{rid0}.json").unlink()
    assert rbc._digests_current(d, runs)[rid0] is None
    # no cache written yet -> the fast path declines
    assert rbc.load_proc_pass("high", tmp_path) is None


def test_proc_from_stacks_matches_the_processing_tail():
    """Rebuilding P/Ps/Pb from stored complex64 stacks must equal the
    process_standard tail exactly (same look filter on the same bytes)."""
    from scipy import ndimage as ndi
    rng = np.random.default_rng(7)
    Fs = (rng.normal(size=(24, 16)) + 1j * rng.normal(size=(24, 16))
          ).astype(np.complex64)
    Fb = (rng.normal(size=(24, 16)) + 1j * rng.normal(size=(24, 16))
          ).astype(np.complex64)
    tw = np.arange(16) * 2e-8
    pr = rbc._proc_from_stacks(Fs, Fb, tw, {"real_chain": "x"})
    want = ndi.uniform_filter1d(np.abs(Fs + Fb) ** 2, rbc.N_LOOKS_SIM,
                                axis=0, mode="nearest")
    assert np.array_equal(pr["P"], want)
    assert np.array_equal(pr["Ps"], ndi.uniform_filter1d(
        np.abs(Fs) ** 2, rbc.N_LOOKS_SIM, axis=0, mode="nearest"))
    assert pr["Fs"] is Fs and pr["Fb"] is Fb and pr["chain"] == {
        "real_chain": "x"}


def test_level_anchor_without_a_deficit_is_refused(monkeypatch):
    """There is no line-level D any more. Silently falling back to one is
    how the Antarctic 14.8 (a rejected att-31 measurement) stayed wired in
    after the adopted config moved to 3.56."""
    axis = _fake_rssnr(monkeypatch)
    with pytest.raises(ValueError, match="explicit level_deficit_db"):
        rbc.build_rssnr_gamma(axis, "full", 31.0, anchor="level")
