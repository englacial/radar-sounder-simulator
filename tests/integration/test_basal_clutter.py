"""Config-level integration cover for tools/run_basal_clutter.py (NO
simulation, NO network): the pass table must match the basal-clutter scout
note exactly, and the derived cross-track reaches must reproduce the scout's
worked geometry (surface reach closed form, refracted bed reach limits)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("run_opr_comparison", "tools/run_opr_comparison.py")
_load("run_opr_coherent_bed", "tools/run_opr_coherent_bed.py")
_load("run_altitude_comparison", "tools/run_altitude_comparison.py")
rbc = _load("run_basal_clutter", "tools/run_basal_clutter.py")


def test_pass_table_matches_scout():
    """Frames, slices, direction and altitudes from claude_notes/
    basal_clutter_scout.md (pilot s=30-40 km; full segment s=18-68 km)."""
    rbc.activate_line("antarctica_getz")
    assert rbc.SEASON == "2016_Antarctica_DC8"
    # the three REAL passes (ORDER) plus the deliberate synthetic 30 km
    # entry (--add-30km), which rides the LOW pass's frames
    assert rbc.ORDER == ["real_low", "real_9km", "real_10km"]
    assert list(rbc.PASSES) == rbc.ORDER + list(rbc.SYNTHETIC_KEYS)
    # the synthetic set is a line-definition choice and gets revised; assert
    # the CONSTRUCTION (constant ellipsoidal height, rides a real carrier)
    # rather than a particular altitude
    for k in rbc.SYNTHETIC_KEYS:
        assert rbc.PASSES[k]["synthetic_msl_m"] > 0
        assert rbc.PASSES[k]["agl_med_m"] is None

    p = rbc.PASSES["real_low"]
    assert not p["rev"]
    assert p["pilot"] == [("20161105_05_005", (2020, 2693))]
    assert p["full"] == [("20161105_05_005", (1212, 3333)),
                         ("20161105_05_006", (0, 1244))]
    p = rbc.PASSES["real_9km"]
    assert p["rev"]
    assert p["pilot"] == [("20161028_05_006", (858, 1532))]
    assert p["full"] == [("20161028_05_006", (0, 2341)),
                         ("20161028_05_005", (2308, 3337))]
    p = rbc.PASSES["real_10km"]
    assert p["rev"]
    assert p["pilot"] == [("20161031_07_005", (337, 1011))]
    assert p["full"] == [("20161031_07_005", (0, 1820)),
                         ("20161031_07_004", (1786, 3336))]

    # scout AGL medians and pilot trace counts (673/674/674)
    assert [rbc.PASSES[k]["agl_med_m"] for k in rbc.ORDER] == \
        [442.0, 9150.0, 10684.0]
    counts = [sum(b - a for _, (a, b) in rbc.PASSES[k]["pilot"])
              for k in rbc.ORDER]
    assert counts == [673, 674, 674]
    # full-segment trace counts match the scout (3365 / 3370 / 3370)
    fcounts = [sum(b - a for _, (a, b) in rbc.PASSES[k]["full"])
               for k in rbc.ORDER]
    assert fcounts == [3365, 3370, 3370]
    # each pass reads its params from a frame it actually flies
    for k in rbc.ORDER:
        fid = rbc.PASSES[k]["param_frame"]
        assert any(fid == part[0] for part in rbc.PASSES[k]["pilot"])
    # cached param provenance, when the data cache has been primed. Skipped
    # rather than failed on a cold cache: outputs/cache is rebuildable and is
    # deliberately emptied at times, so its absence is not a config error.
    from soundersim.opr import CACHE_DIR
    want = [CACHE_DIR / f"mcords_params_{rbc.SEASON}_"
            f"{rbc.PASSES[k]['param_frame']}.json" for k in rbc.ORDER]
    if not any(f.exists() for f in want):
        pytest.skip("param cache not primed (outputs/cache is rebuildable "
                    "and is deliberately emptied at times)")
    for f in want:
        assert f.exists(), f


def test_surface_reach_matches_scout_numbers():
    """Scout quirk 3 worked examples: surface clutter reaching the nadir-bed
    delay of 720 m ice (8.53 us) needs +-1.66 km at 442 m and +-5.40 km at
    10.8 km AGL."""
    assert abs(rbc.surface_reach(442.0, 8.53e-6) - 1663.0) < 25.0
    assert abs(rbc.surface_reach(10684.0, 8.53e-6) - 5383.0) < 40.0
    # monotone in altitude and in delay
    assert rbc.surface_reach(9150.0, 8.53e-6) < rbc.surface_reach(
        10684.0, 8.53e-6)
    assert rbc.surface_reach(442.0, 8.53e-6) < rbc.surface_reach(
        442.0, 11.53e-6)


def test_bed_reach_refraction_limits():
    """The refracted bed reach must (a) reduce to the homogeneous closed
    form when n_ice = 1, (b) shrink when the medium slows (n > 1), and (c)
    never bind: the surface interface's reach target includes the whole ice
    column, so ct == surface reach."""
    h, d, m = 10684.0, 468.0, 3.0e-6
    homog = rbc.surface_reach(h + d, m)     # n=1: one uniform medium
    assert abs(rbc.bed_reach(h, d, 1.0, m) - homog) / homog < 0.01
    n_ice = float(np.sqrt(3.17))
    assert rbc.bed_reach(h, d, n_ice, m) < homog
    doc = rbc.derive_reach(h_max=10826.0, dbs_max=10.58e-6, d_min=468.0)
    assert doc["ct_m"] == doc["surface_reach_m"] > doc["bed_reach_m"]
    assert not doc["capped"]
    # the derived high-pass reach: nadir-bed delay + 3 us at max AGL ~6.9 km
    assert 6500.0 < doc["ct_m"] < 7500.0
    lo = rbc.derive_reach(h_max=547.0, dbs_max=10.58e-6, d_min=468.0)
    assert 2200.0 < lo["ct_m"] < 2800.0     # low pass ~2.5 km
    assert lo["ct_m"] < doc["ct_m"]


def test_picked_bed_reference_is_the_low_pass():
    """--picked-bed takes its picks from ONE reference pass -- the LOW pass
    (cleanest bed of the triplet) -- applied to all three simulations, and
    tags its outputs/cache keys so the BedMachine runs stay cached."""
    assert rbc.REF_PASS == "real_low"
    assert rbc.REF_FRAMES == ("20161105_05_005", "20161105_05_006",
                              "20161105_05_007")
    # every frame the low pass simulates is covered by the reference frames
    for seg in ("pilot", "full"):
        assert all(fid in rbc.REF_FRAMES
                   for fid, _ in rbc.PASSES["real_low"][seg])
    assert rbc.case_tag(True) == rbc.PBED_TAG == "_pbed"
    assert rbc.case_tag(False) == ""
    # offline reference: frame + bottom-pick caches exist for all ref frames
    from soundersim.opr import CACHE_DIR
    for fid in rbc.REF_FRAMES:
        assert (CACHE_DIR /
                f"frame_{rbc.SEASON}_{fid}_CSARP_standard.nc").exists()
        assert (CACHE_DIR / f"layers_{rbc.SEASON}_{fid}_bottom.nc").exists()


def test_project_to_track():
    """Along-track projection: cross-track offsets do not change s, and the
    tangential refinement recovers s between samples / past the end."""
    s = np.arange(0.0, 1000.0, 10.0)
    ang = np.deg2rad(35.0)
    tx, ty = 1e5 + s * np.cos(ang), -2e5 + s * np.sin(ang)
    # points pushed 3 km along the cross-track normal keep their s
    nx, ny = -np.sin(ang), np.cos(ang)
    got = rbc.project_to_track(tx + 3000.0 * nx, ty + 3000.0 * ny, tx, ty, s)
    assert np.abs(got - s).max() < 1e-6
    # between samples, and beyond the end (linear extrapolation)
    mid = rbc.project_to_track(np.array([tx[0] + 7.0 * np.cos(ang)]),
                               np.array([ty[0] + 7.0 * np.sin(ang)]),
                               tx, ty, s)
    assert abs(float(mid[0]) - 7.0) < 1e-6
    beyond = rbc.project_to_track(np.array([tx[-1] + 50.0 * np.cos(ang)]),
                                  np.array([ty[-1] + 50.0 * np.sin(ang)]),
                                  tx, ty, s)
    assert abs(float(beyond[0]) - (s[-1] + 50.0)) < 1e-6


def test_roughness_rms():
    """The scout's roughness metric: rms about a running mean of ROUGH_WIN_M.
    Short-wavelength relief is measured (A/sqrt(2)); wavelengths much longer
    than the window are detrended away."""
    s = np.arange(0.0, 50000.0, 14.85)
    assert abs(rbc.roughness_rms(s, 50.0 * np.sin(2 * np.pi * s / 300.0))
               - 50.0 / np.sqrt(2.0)) < 1.0
    assert rbc.roughness_rms(s, 50.0 * np.sin(2 * np.pi * s / 50000.0)) < 5.0
    # NaN gaps are interpolated, not propagated
    z = 50.0 * np.sin(2 * np.pi * s / 300.0)
    z[10:14] = np.nan
    assert np.isfinite(rbc.roughness_rms(s, z))


def _toy_scene(pick_fn, gap=None):
    """Synthetic MultilayerScene (EPSG:3031, 32 m) with a straight west-east
    track, a sloping bed carrying a 1 km cross-track sinusoid, and a
    reference pick profile -- no network, no simulation."""
    from affine import Affine
    from pyproj import Transformer

    from soundersim.config import Medium
    from soundersim.synthetic import MultilayerScene

    x0, y0, n = -1.4e6, -5.0e5, 200
    tf = Affine.translation(x0, y0) * Affine.scale(32.0, -32.0)
    cols, rows = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    X, Y = tf * (cols, rows)
    bed = (-500.0 + 0.002 * (X - x0) + 40.0 * np.sin(
        2 * np.pi * (Y - y0) / 1000.0)).astype(np.float32)
    surf = (bed + 800.0).astype(np.float32)
    yc = y0 - 32.0 * n / 2.0
    sx = np.arange(x0 - 500.0, x0 + 32.0 * n + 500.0, 14.85)
    sy = np.full_like(sx, yc)
    s = sx - sx[0]
    bed_track = -500.0 + 0.002 * (sx - x0) + 40.0 * np.sin(
        2 * np.pi * (yc - y0) / 1000.0)
    pick = bed_track + pick_fn(s)
    if gap is not None:
        pick[gap(s)] = np.nan
    inv = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(sx[::200], sy[::200])
    nav = np.column_stack([lat, lon, np.full(len(lat), 1000.0)])
    sc = MultilayerScene("toy", [surf, bed.copy()], tf, "EPSG:3031", nav,
                         [Medium(name="air", eps_r=1.0),
                          Medium(name="ice", eps_r=3.17),
                          Medium(name="bed", eps_r=8.0)], {})
    ref = {"pass": "low", "frames": list(rbc.REF_FRAMES), "eps_ice": 3.17,
           "x": sx, "y": sy, "s": s, "bed": pick}
    return sc, ref, bed, np.column_stack([sx, sy])


def test_picked_bed_matches_picks_at_nadir_and_keeps_cross_track():
    """The residual construction: the nadir line ends up ON the picks, while
    BedMachine's cross-track structure is preserved exactly (the correction
    is a function of along-track s only) -- NOT a constant cross-track
    extension of the 1-D profile."""
    def bump(s):
        return 60.0 * np.sin(2 * np.pi * s / 900.0)

    sc, ref, bed0, tpts = _toy_scene(bump)
    stats = rbc.apply_picked_bed(sc, ref)
    bed1 = np.asarray(sc.dems[1], np.float64)

    # nadir: corrected bed == picks (bilinear sampling error only)
    inside = ((tpts[:, 0] > sc.transform.c + 100.0)
              & (tpts[:, 0] < sc.transform.c + 32.0 * 200 - 100.0))
    got = rbc.sample_dem(bed1, sc.transform, tpts[inside, 0], tpts[inside, 1])
    assert np.abs(got - ref["bed"][inside]).max() < 1.0
    # cross-track: the added field depends on s (here x) only, so every
    # column shifts rigidly and all cross-track relief survives untouched
    d = bed1 - bed0
    assert float(np.abs(d - d[:1, :]).max()) < 1e-3
    assert float(np.abs(np.diff(bed1, axis=0)
                        - np.diff(bed0, axis=0)).max()) < 1e-3
    # a constant cross-track extension would have flattened it
    assert np.ptp(bed1[:, 100]) > 50.0
    assert stats["reference_pass"] == "low"
    assert stats["gap_frac_segment"] == 0.0
    assert abs(stats["residual_rms_m"] - 60.0 / np.sqrt(2.0)) < 1.0
    assert abs(stats["residual_absmax_m"] - 60.0) < 1.0
    assert stats["bed_roughness_rms_m"]["picked"] > \
        stats["bed_roughness_rms_m"]["bedmachine"]


def test_picked_bed_gaps_fall_back_to_bedmachine():
    """Pick gaps take zero residual: those columns stay pure BedMachine."""
    def flat(s):
        return np.full_like(s, 25.0)

    sc, ref, bed0, _ = _toy_scene(
        flat, gap=lambda s: np.abs(s - s.mean()) < 900.0)
    stats = rbc.apply_picked_bed(sc, ref)
    d = np.asarray(sc.dems[1], np.float64) - bed0
    assert abs(float(d.max()) - 25.0) < 0.5
    assert abs(float(d.min())) < 0.5              # the gap band: no shift
    assert 0.0 < stats["gap_frac_segment"] < 1.0


def test_windows_and_chunking():
    """The fast-time window must extend past the clutter margin, and the
    chunking must make the 50 km segment ~5 pilot-sized chunks (linear
    wall-time projection)."""
    assert rbc.POST_BED_US >= rbc.MARGIN_US + 0.4
    assert rbc.MID_LO_US == 1.0 and rbc.MID_HI_US == 0.5
    assert max(1, int(round(10000.0 / rbc.CHUNK_M))) == 1
    assert max(1, int(round(50000.0 / rbc.CHUNK_M))) == 5
