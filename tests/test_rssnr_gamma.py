"""RSSNR-driven bed reflectivity: mapping math, NaN/censoring handling,
snapshot pinning, and the simulate() gamma_maps plumbing.

Mapping (anchoring-free, 2026-08-20): |Gamma_bed|^2(s) dB = 2*A*H(s) -
RSSNR(s) + (gamma_surface - T^2), gamma_surface per the line
calibration (manual or residual-solved). Tool
functions imported from tools/run_basal_clutter.py (pure math only -- no
network, no frames).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_basal_clutter as rbc  # noqa: E402

from soundersim.physics import fresnel_normal  # noqa: E402

G2_CONST = 20.0 * np.log10(abs(fresnel_normal(3.17, 8.0)))  # ~ -12.86 dB


# ------------------------------------------------------------- mapping math

def test_constant_inputs_map_to_the_direct_constant():
    """Uniform RSSNR + thickness: G2 is EXACTLY 2AH - RSSNR + (gamma - T2)
    everywhere -- no anchoring of any kind."""
    s = np.linspace(0.0, 50e3, 40)
    gamma = -11.03
    prof = rbc.rssnr_gamma_profile(s, np.full(40, 37.0), np.full(40, 700.0),
                                   np.ones(40, bool), 15.0, gamma,
                                   0.0, 50e3)
    want = 2 * 15.0 * 0.700 - 37.0 + (gamma - rbc.t2_db())
    np.testing.assert_allclose(prof["g2_db"], want, atol=1e-9)
    assert prof["k_db"] == pytest.approx(gamma - rbc.t2_db(), abs=0.01)
    assert prof["n_censored"] == 0


def test_two_zone_step_and_k_roundtrip():
    """A 10 dB RSSNR step maps to a -10 dB G2 step (2AH fixed); K
    round-trips: RSSNR = 2*A*H - (G2 - K) recovers the input exactly."""
    n, att, h = 60, 15.0, 650.0
    s = np.linspace(0.0, 60e3, n)
    rssnr = np.where(s < 30e3, 30.0, 40.0)
    prof = rbc.rssnr_gamma_profile(s, rssnr, np.full(n, h),
                                   np.ones(n, bool), att, -11.03, 0.0, 60e3)
    g2 = prof["g2_db"]
    step = np.median(g2[s < 30e3]) - np.median(g2[s >= 30e3])
    assert step == pytest.approx(10.0, abs=1e-9)
    back = 2.0 * att * h / 1e3 - (g2 - prof["k_db"])
    np.testing.assert_allclose(back, rssnr, atol=0.005)  # k_db rounded 2dp


def test_thickness_term_uses_att():
    """G2 rises by 2*A*dH for extra thickness at fixed RSSNR (attenuation
    held fixed at the tool's --att; H from the dataset twtts by design)."""
    s = np.linspace(0.0, 10e3, 20)
    thick = np.where(s < 5e3, 600.0, 850.0)
    prof = rbc.rssnr_gamma_profile(s, np.full(20, 35.0), thick,
                                   np.ones(20, bool), 20.0, -11.03,
                                   0.0, 10e3)
    g2 = prof["g2_db"]
    d = np.median(g2[s >= 5e3]) - np.median(g2[s < 5e3])
    assert d == pytest.approx(2.0 * 20.0 * 0.250, abs=1e-9)


def test_k_phys_value_and_surface_anomaly():
    """K_phys = gamma_Fresnel - T2 (-10.32 dB at eps 3.17); the profile
    records the surface anomaly = gamma_surface - gamma_Fresnel, which
    equals K - K_phys exactly (T2 cancels)."""
    assert rbc.k_phys_db(3.17) == pytest.approx(-10.32, abs=0.01)
    s = np.linspace(0.0, 10e3, 20)
    gamma = -3.69
    prof = rbc.rssnr_gamma_profile(s, np.full(20, 30.0), np.full(20, 700.0),
                                   np.ones(20, bool), 15.0, gamma,
                                   0.0, 10e3)
    assert prof["k_minus_kphys_db"] == pytest.approx(
        prof["k_db"] - prof["k_phys_db"], abs=0.011)
    assert prof["surface_anomaly_db"] == pytest.approx(
        gamma - rbc.gamma_surface_fresnel_db(), abs=0.011)
    assert prof["k_minus_kphys_db"] == pytest.approx(
        prof["surface_anomaly_db"], abs=0.02)


# ----------------------------------------------------- censoring / NaN floor

def test_censored_samples_get_floor_not_interpolation():
    """qc-fail / NaN-RSSNR samples take the segment's MINIMUM mapped G2 (a
    brightness floor: their RSSNR is a lower bound, the bed was too dim to
    pick) -- never a value interpolated across the gap."""
    n = 40
    s = np.linspace(0.0, 40e3, n)
    rssnr = np.full(n, 30.0)
    rssnr[10:15] = 20.0            # a BRIGHT zone adjacent to the gap
    qc = np.ones(n, bool)
    rssnr[15:20] = np.nan          # censored gap next to the bright zone
    qc[20] = False                 # qc-fail is censored too
    prof = rbc.rssnr_gamma_profile(s, rssnr, np.full(n, 700.0), qc, 15.0,
                                   -11.03, 0.0, 40e3)
    ok = prof["ok"]
    assert prof["n_censored"] == 6
    floor = float(prof["g2_db"][ok].min())
    assert prof["censored_floor_db"] == pytest.approx(floor, abs=0.01)
    # censored entries sit AT the floor -- 10 dB below their bright neighbor
    np.testing.assert_allclose(prof["g2_db"][~ok], floor, atol=1e-9)
    assert float(prof["g2_db"][ok].max()) - floor == pytest.approx(10.0,
                                                                   abs=1e-6)


def test_all_censored_segment_raises():
    s = np.linspace(0.0, 10e3, 10)
    with pytest.raises(RuntimeError, match="usable RSSNR"):
        rbc.rssnr_gamma_profile(s, np.full(10, np.nan), np.full(10, 700.0),
                                np.ones(10, bool), 15.0, -11.03, 0.0, 10e3)


# ------------------------------------------------------- snapshot pin / cache

def test_snapshot_pinned_and_cache_provenance(tmp_path):
    """The tool pins the completed pre-rebuild snapshot; the cache path
    round-trips arrays + provenance and REJECTS a cache built from a
    different snapshot."""
    assert rbc.RSSNR_SNAPSHOT == "3YH47013745B2T5ZZR50"
    cache = tmp_path / "c.npz"
    prov = {"snapshot_id": rbc.RSSNR_SNAPSHOT, "store": rbc.RSSNR_STORE,
            "frames": list(rbc.REF_FRAMES), "n_traces": 3}
    d = dict(lat=np.array([-75.0, -75.1, -75.2]),
             lon=np.array([-105.0, -105.1, -105.2]),
             rssnr=np.array([30.0, 40.0, np.nan]),
             qc=np.array([1, 1, 0], np.uint8),
             stw=np.array([3e-6] * 3), btw=np.array([11e-6] * 3))
    np.savez(cache, provenance=json.dumps(prov), **d)
    arrs, p = rbc.fetch_rssnr_anchor(cache_path=cache)
    assert p["snapshot_id"] == rbc.RSSNR_SNAPSHOT
    assert p["source"].startswith("cache:")
    np.testing.assert_array_equal(arrs["rssnr"], d["rssnr"])
    # wrong snapshot in the cache -> hard error (never silently mixed)
    np.savez(cache, provenance=json.dumps({**prov, "snapshot_id": "WRONG"}),
             **d)
    with pytest.raises(RuntimeError, match="snapshot"):
        rbc.fetch_rssnr_anchor(cache_path=cache)


def test_segment_s_range():
    ref = {"frames": ["A", "B"], "frame_len": [4, 4],
           "s": np.arange(8, dtype=float) * 1000.0}
    old = rbc.PASSES
    try:
        # keyed on the REFERENCE pass, not a hardcoded 'low': the pick axis
        # is what defines a segment's along-track range, and the pass names
        # are line data
        rbc.PASSES = {rbc.REF_PASS: {"seg": [("A", (1, 4)), ("B", (0, 2))]}}
        lo, hi = rbc.segment_s_range(ref, "seg")
    finally:
        rbc.PASSES = old
    assert (lo, hi) == (1000.0, 5000.0)


# ------------------------------------------- simulate() gamma_maps plumbing

def test_gamma_maps_constant_grid_bitwise_and_incoherent_raises():
    """scene.gamma_maps with a CONSTANT grid equal to the Fresnel value is
    bit-identical to the plain run (grid -> per-facet sampling -> kernel);
    incoherent mode rejects gamma_maps."""
    import soundersim
    from soundersim import synthetic as syn
    from soundersim.config import (
        DemInterface,
        FacetConfig,
        Medium,
        RadarConfig,
        SimConfig,
    )

    def cfg(mode):
        return SimConfig(
            mode=mode, radar=RadarConfig(dt=1e-8, n_samples=1000, t0=0.0,
                                         f0=195e6 if mode == "coherent"
                                         else None),
            facets=FacetConfig(spacing=20.0),
            media=[Medium(name="air", eps_r=1.0),
                   Medium(name="ice", eps_r=3.17),
                   Medium(name="bed", eps_r=8.0)],
            interfaces=[DemInterface(name="surface"),
                        DemInterface(name="bed")])

    scene = syn.slab_scene(surface=500.0, depth=300.0, extent=1500.0,
                           n_traces=2, altitude=1000.0)
    ds0 = soundersim.simulate(scene, cfg("coherent"))
    g = float(fresnel_normal(3.17, 8.0))
    scene.gamma_maps = {"bed": (np.full(scene.dems[1].shape, g),
                                scene.transform, scene.crs)}
    ds1 = soundersim.simulate(scene, cfg("coherent"))
    assert np.array_equal(ds0.field.values, ds1.field.values)
    with pytest.raises(ValueError, match="coherent"):
        soundersim.simulate(scene, cfg("incoherent"))
    scene.gamma_maps = {"nope": (np.full(scene.dems[1].shape, g),
                                 scene.transform, scene.crs)}
    with pytest.raises(ValueError, match="unknown"):
        soundersim.simulate(scene, cfg("coherent"))


# ------------------------------------------------- gamma_surface: solve
def test_qualifying_median_keeps_only_bed_dominated_passes():
    """The solve's residual comes ONLY from passes whose sim bed window is
    a bed measurement (bed returns >= margin above surface returns); a
    clutter-dominated high pass must not drag the solved gamma."""
    res = {"low": -2.0, "mid": -3.0, "high": +9.0}
    marg = {"low": 18.0, "mid": 12.5, "high": -4.0}
    qual, qmed = rbc.gamma_solve_qualifying_median(res, marg, 10.0)
    assert qual == ["low", "mid"]
    assert qmed == pytest.approx(-2.5)


def test_qualifying_median_is_nan_when_nothing_qualifies():
    """No qualifying pass -> NaN, so the driver refuses instead of solving
    gamma against surface clutter."""
    qual, qmed = rbc.gamma_solve_qualifying_median(
        {"a": 1.0}, {"a": 3.0}, 10.0)
    assert qual == [] and np.isnan(qmed)


def test_calibration_gamma_accepts_solve_and_manual():
    from clutter_lines import Calibration
    c = Calibration(gamma_surface_db="solve", att_db_per_km="solve")
    assert c.gamma_surface_db == "solve"
    c2 = Calibration(gamma_surface_db={"value": -10.0, "why": "test"},
                     att_db_per_km="solve")
    assert c2.gamma_surface_db.value == -10.0
    with pytest.raises(ValueError):
        Calibration(gamma_surface_db="sovle", att_db_per_km="solve")
    with pytest.raises(ValueError):        # manual still requires a why
        Calibration(gamma_surface_db={"value": -10.0},
                    att_db_per_km="solve")


def test_resolve_calibration_returns_the_solve_marker(monkeypatch):
    """A solve line resolves gamma to the literal 'solve' plus the solver
    settings -- main_config owns the loop; nothing here invents a number.
    The regression fetch is stubbed out (network-free)."""
    monkeypatch.setattr(rbc, "fetch_rssnr_anchor",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("offline test")))
    for k in rbc.LINE_GLOBALS:            # resolve_calibration activates
        monkeypatch.setattr(rbc, k, getattr(rbc, k))
    line = next(n for n in sorted(rbc.LINES)
                if rbc.LINES[n]["CALIBRATION"]["gamma_surface_db"] == "solve"
                and rbc.LINES[n]["CALIBRATION"]["att_db_per_km"] != "solve")
    gamma, att, rec = rbc.resolve_calibration(line)
    assert gamma == "solve"
    assert rec["gamma_surface_db"] == "solve"
    assert rec["gamma_surface_solve_settings"] == rbc.GAMMA_SURFACE_SOLVE
    assert "error" in rec["regression"]      # diagnostic recorded, not fatal
    assert att == rbc.LINES[line]["CALIBRATION"]["att_db_per_km"]["value"]


def test_bare_run_refuses_an_unresolved_solve_gamma():
    """run()'s calibration fallback (manual_gamma_surface_db) must fail
    loudly on a solve line, never index 'solve' like a manual dict."""
    line = next(n for n in sorted(rbc.LINES)
                if rbc.LINES[n]["CALIBRATION"]["gamma_surface_db"] == "solve")
    saved = {k: getattr(rbc, k) for k in rbc.LINE_GLOBALS}
    try:
        rbc.activate_line(line)
        with pytest.raises(ValueError, match="config driver"):
            rbc.manual_gamma_surface_db()
    finally:
        for k, v in saved.items():
            setattr(rbc, k, v)
