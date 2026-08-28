"""--processing standard and --add-30km config-level pieces of
tools/run_basal_clutter.py: the alias-limited aperture math, the recorded
processing chain (real-chain provenance + honest gap list), and the 30 km
synthetic-pass geometry construction. No network, no kernels."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_basal_clutter as rbc  # noqa: E402

C = 299792458.0


# --------------------------------------------------- alias-limited aperture

def test_alias_limited_aperture_math():
    """sin(theta) = lam/(4*ds); L = 2 r tan(theta). At the alias limit the
    azimuth resolution lam/(4 sin theta) equals the posting."""
    lam, ds, r = C / 190e6, 14.85, 10000.0
    L, th = rbc.alias_limited_aperture(lam, ds, r)
    st = lam / (4.0 * ds)
    assert th == pytest.approx(np.degrees(np.arcsin(st)), rel=1e-9)
    assert L == pytest.approx(2.0 * r * np.tan(np.arcsin(st)), rel=1e-9)
    assert lam / (4.0 * np.sin(np.radians(th))) == pytest.approx(ds, rel=1e-9)
    # aperture grows linearly with range, resolution stays at the posting
    L2, th2 = rbc.alias_limited_aperture(lam, ds, 3.0 * r)
    assert th2 == th and L2 == pytest.approx(3.0 * L, rel=1e-9)


def test_processing_chain_recorded(monkeypatch):
    """process_standard records the real chain, our chain and the gap list
    (g1-g6) -- run on a tiny synthetic pass dict with a flat two-layer
    field."""
    T, nb = 24, 16
    ds_m = 14.85
    s = np.arange(T) * ds_m
    lam = C / 190e6
    rng = np.random.default_rng(0)
    F = (rng.normal(size=(T, nb, 2)) + 1j * rng.normal(size=(T, nb, 2))
         ).astype(np.complex64)
    twtt = 6e-6 + np.arange(nb) * 2.0202e-8
    nav = np.column_stack([np.full(T, -75.0), np.full(T, -105.0) + s / 111e3,
                           np.full(T, 500.0)])

    class Base:
        nav_llh = nav

    p = {"lam": lam, "s_m": s, "idx": np.arange(T), "base": Base(),
         "bot": np.full(T, 8e-6), "surf": np.full(T, 6e-6),
         # sim-trace views (prep_pass contract; == <measured>[idx] at the
         # product posting, the refined grid's own values at --posting-div>1)
         "s_sim": s, "bot_sim": np.full(T, 8e-6),
         "surf_sim": np.full(T, 6e-6)}
    # the chain records the ACTIVE line's own product provenance; assert on
    # the getz chain by activating it, since '11 looks' is a getz fact
    rbc.activate_line("antarctica_getz")
    out = rbc.process_standard(p, {"field": F, "twtt": twtt})
    ch = out["chain"]
    assert ch["real_chain"] == rbc.REAL_CHAIN
    assert "11 looks" in ch["real_chain"]["combine"]
    assert "f-k" in ch["real_chain"]["sar"]
    for g in ("g1", "g2", "g3", "g4", "g5", "g6"):
        assert g in ch["gaps"]
    assert ch["sim_posting_m"] == pytest.approx(ds_m, abs=0.01)
    L, th = rbc.alias_limited_aperture(lam, ds_m, C * 8e-6 / 2.0)
    assert ch["aperture_m"] == pytest.approx(L, abs=0.1)
    assert ch["n_looks_sim"] == rbc.N_LOOKS_SIM
    assert "straight_track_check" in ch and "mocomp" in ch
    # shape/lattice preserved: powers on the same (T, nb) grid
    for k in ("P", "Ps", "Pb"):
        assert out[k].shape == (T, nb)
        assert np.isfinite(out[k]).all()
    np.testing.assert_array_equal(out["twtt"], twtt)


# ------------------------------------------------------- 30 km construction

def _fake_fsub(T=8, elev=600.0, agl=440.0):
    z_surf = elev - agl  # 160 m surface elevation
    surf = np.full(T, 2.0 * agl / C)
    return xr.Dataset({
        "Elevation": ("slow_time", np.full(T, elev)),
        "Surface": ("slow_time", surf),
        "Roll": ("slow_time", np.full(T, 0.02)),
    }, coords={"slow_time": np.arange(T)}), z_surf


def test_synth_altitude_construction():
    """Platform z constant at alt_m (ellipsoidal), roll zeroed, surface twtt
    = 2*(alt - z_surf)/c, bed delay below surface preserved exactly."""
    fsub, z_surf = _fake_fsub()
    dbs = 8.5e-6
    bot = np.asarray(fsub.Surface.values) + dbs
    out, bot2, note = rbc.synth_altitude_fsub(fsub, bot, 30000.0)
    np.testing.assert_allclose(out.Elevation.values, 30000.0)
    np.testing.assert_allclose(out.Roll.values, 0.0)
    np.testing.assert_allclose(out.Surface.values,
                               2.0 * (30000.0 - z_surf) / C, rtol=1e-12)
    np.testing.assert_allclose(bot2 - out.Surface.values, dbs, rtol=1e-12)
    assert note["synthetic_msl_m"] == 30000.0
    assert note["agl_med_m"] == pytest.approx(30000.0 - z_surf, abs=1.0)


def test_synth_altitude_below_surface_raises():
    fsub, z_surf = _fake_fsub(elev=600.0, agl=440.0)  # surface at 160 m
    bot = np.asarray(fsub.Surface.values) + 8e-6
    with pytest.raises(ValueError, match="above the surface"):
        rbc.synth_altitude_fsub(fsub, bot, 100.0)


def test_synthetic_pass_spec_and_tags():
    """A synthetic pass rides its carrier's frames; the proc tag composes
    into the coordinator's output-dir name. Names come from the line
    definition rather than literals -- the passes get renamed."""
    # shipped lines declare no synthetics (experiments add them via
    # extra_passes), so build one on a copy of the default line
    line = rbc.DEFAULT_LINE
    carrier = rbc.LINE_SPECS[line].reference.pass_key
    ls = rbc.LINE_SPECS[line].model_copy(update={"synthetic_passes": {
        "syn_14km": rbc.clutter_lines.SyntheticPass(
            altitude_m=14000.0, carrier=carrier)}})
    table = ls._pass_table()
    spec = table["syn_14km"]
    seg = next(s for s in rbc.LINES[line]["SEGMENTS"])
    assert spec[seg] == table[carrier][seg]
    assert spec["synthetic_msl_m"] == 14000.0
    assert rbc.case_tag(True, True, True) == "_pbed_rssnr_proc"
    assert rbc.case_tag(True, True, False) == "_pbed_rssnr"  # existing runs


def test_chunk_digests_forwards_the_hypothesis_knobs(tmp_path):
    """--specular-fraction / --bed-rough with --proc-cache crashed
    (2026-08-21): the digest step rebuilt chunk rids WITHOUT the knob
    suffixes chunk_rid appends, then looked for chunk files that never
    existed. The digests must be computed with the same knobs the chunks
    were simulated with."""
    p = {"key": "k", "segment": "pilot", "picked_bed": False,
         "gamma_rssnr": True, "proc": True, "dgn": True}
    spec, brough = (0.5, 3.0, 1.0), (0.22, 0.886)
    rid = rbc.chunk_rid(p, 0, 18.61, True, bed_rough=brough, spec=spec)
    assert "_fs0.5" in rid and "_brough0.22" in rid
    (tmp_path / f"{rid}.json").write_text(json.dumps({"meta_key": "m"}))
    d = rbc.chunk_digests(p, tmp_path, 1, 18.61, True,
                          bed_rough=brough, spec=spec)
    assert set(d) == {rid}
    with pytest.raises(FileNotFoundError):    # knobs dropped = old bug
        rbc.chunk_digests(p, tmp_path, 1, 18.61, True)
