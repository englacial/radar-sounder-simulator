"""M16 multilayer kernel + simulate() tests (CI-fast, tiny scenes).

Covers: the eps->1 reduction of the refracted path to the single-interface
kernels (both modes), exact slab nadir delay through simulate(), a smoke
flat-slab coherent field check against a two-media brute-force referee
(per-sample exact flat-interface Fermat solve + direct summation -- M17 does
this at scale), energy bookkeeping with the per-layer dropped channel
(invalid/shadowed paths from steep geometry dropped, not NaN), attenuation
scaling, sequential chaining through >1 interface (offset firn stack), and
the layer-dimension output structure per docs/output.md.
"""

import numpy as np
import pytest

import soundersim
from soundersim.config import (DemInterface, FacetConfig, Medium,
                               OffsetInterface, RadarConfig, SimConfig)
from soundersim.physics import C, fresnel_normal, fresnel_te
from soundersim.refraction import snell_crossing
from soundersim.scene import LocalFrame, build_facets
from soundersim.nav import nav_to_frame
from soundersim import synthetic as syn

N_ICE = float(np.sqrt(3.17))


def _media(*eps, att=None):
    att = att or [0.0] * len(eps)
    return [Medium(name=f"m{i}", eps_r=e, attenuation_db_per_km=a)
            for i, (e, a) in enumerate(zip(eps, att))]


def _slab_cfg(mode, media, *, n_samples=1250, dt=1e-8, t0=0.0, f0=None,
              spacing=None, split_sides=False):
    return SimConfig(
        mode=mode, split_sides=split_sides,
        radar=RadarConfig(dt=dt, n_samples=n_samples, t0=t0, f0=f0),
        facets=FacetConfig(spacing=spacing), media=media,
        interfaces=[DemInterface(name="surface"), DemInterface(name="bed")])


# ---------------------------------------------------------------- eps -> 1

@pytest.fixture(scope="module")
def eps1_scene():
    return syn.slab_scene(surface=500.0, depth=300.0, extent=2000.0,
                          n_traces=3, altitude=1000.0)


@pytest.fixture(scope="module")
def eps1_bed_scene(eps1_scene):
    """The bed DEM re-cast as a surface-only scene (same grid, nav, frame)."""
    s = eps1_scene
    return syn.SyntheticScene("bed_as_surface", s.dems[1], s.transform, s.crs,
                              s.nav_llh, dict(s.params))


def test_eps_to_one_reduction_incoherent(eps1_scene, eps1_bed_scene):
    """With eps_r(ice) = 1 and zero attenuation the bed layer equals a
    surface-only run on the bed DEM (transmission, spreading and flux factors
    all reduce to the single-medium 1/r^4)."""
    ds = soundersim.simulate(eps1_scene,
                             _slab_cfg("incoherent", _media(1.0, 1.0, 6.0)))
    ref = soundersim.simulate(
        eps1_bed_scene,
        SimConfig(mode="incoherent",
                  radar=RadarConfig(dt=1e-8, n_samples=1250, t0=0.0),
                  facets=FacetConfig(), media=_media(1.0, 6.0)))
    bed = ds.power.sel(layer="bed").values
    peak = ref.power.values.max()
    np.testing.assert_allclose(bed, ref.power.values, rtol=2e-4,
                               atol=1e-6 * peak)
    np.testing.assert_allclose(ds.dropped_power.sel(layer="bed").values,
                               ref.dropped_power.values, rtol=1e-4)
    np.testing.assert_allclose(ds.nadir_twtt.sel(layer="bed").values,
                               ref.nadir_twtt.values, rtol=1e-12)


def test_eps_to_one_reduction_coherent(eps1_scene, eps1_bed_scene):
    """Coherent bed-layer field reduces to the single-interface kernel (same
    normal-incidence target gamma convention makes this exact up to float32
    facet/phase quantization)."""
    ds = soundersim.simulate(
        eps1_scene, _slab_cfg("coherent", _media(1.0, 1.0, 6.0), f0=195e6,
                              spacing=15.0))
    ref = soundersim.simulate(
        eps1_bed_scene,
        SimConfig(mode="coherent",
                  radar=RadarConfig(dt=1e-8, n_samples=1250, t0=0.0, f0=195e6),
                  facets=FacetConfig(spacing=15.0), media=_media(1.0, 6.0)))
    bed = ds.field.sel(layer="bed").values
    peak = np.abs(ref.field.values).max()
    assert np.abs(bed - ref.field.values).max() < 5e-3 * peak
    # index-matched surface reflects nothing in the field convention
    assert np.abs(ds.field.sel(layer="surface").values).max() == 0.0


# ------------------------------------------------------------- slab delays

@pytest.fixture(scope="module")
def ds_slab_inc():
    scene = syn.slab_scene(surface=500.0, depth=300.0, extent=2000.0,
                           n_traces=3, altitude=1000.0)
    return soundersim.simulate(scene,
                               _slab_cfg("incoherent", _media(1.0, 3.17, 6.0)))


def test_slab_nadir_delay_exact_bin(ds_slab_inc):
    """Bed nadir delay 2h/c + 2d*sqrt(eps)/c: per-layer nadir_twtt matches and
    the earliest bed return lands in exactly that fast-time bin."""
    ds = ds_slab_inc
    h = float(ds.z[1]) - 500.0  # platform height above the surface, local frame
    expected = 2.0 * h / C + 2.0 * 300.0 * N_ICE / C
    assert float(ds.nadir_twtt.sel(layer="bed")[1]) == pytest.approx(
        expected, rel=1e-9)
    assert float(ds.nadir_twtt.sel(layer="surface")[1]) == pytest.approx(
        2.0 * h / C, rel=1e-9)
    bed = ds.power.sel(layer="bed")[1].values
    assert np.nonzero(bed)[0][0] == int(np.floor(expected / 1e-8))


def test_multilayer_dataset_structure(ds_slab_inc):
    ds = ds_slab_inc
    assert ds.power.dims == ("slow_time", "twtt", "layer")
    assert list(ds.layer.values) == ["surface", "bed"]
    assert ds.nadir_twtt.dims == ("slow_time", "layer")
    assert ds.dropped_power.dims == ("slow_time", "layer")
    assert (ds.dropped_power >= 0).all()
    assert np.isfinite(ds.power).all()
    # combine over layer: incoherent power is additive
    np.testing.assert_allclose(soundersim.combine(ds, "layer"),
                               ds.power.sum("layer"))


# ---------------------------------------------- attenuation through the slab

def test_attenuation_scales_bed_power(ds_slab_inc):
    """10 dB/km one-way in 300 m of ice: bed power drops by 2*10*0.3 = 6 dB;
    checked in the nadir bed bin (path within a bin of vertical)."""
    scene = syn.slab_scene(surface=500.0, depth=300.0, extent=2000.0,
                           n_traces=3, altitude=1000.0)
    ds_att = soundersim.simulate(
        scene, _slab_cfg("incoherent", _media(1.0, 3.17, 6.0,
                                              att=[0.0, 10.0, 0.0])))
    b = int(np.floor(float(ds_slab_inc.nadir_twtt.sel(layer="bed")[1]) / 1e-8))
    p0 = float(ds_slab_inc.power.sel(layer="bed")[1][b])
    p1 = float(ds_att.power.sel(layer="bed")[1][b])
    assert p0 > 0
    assert p1 / p0 == pytest.approx(10.0 ** (-0.6), rel=0.01)
    # surface layer is untouched by ice attenuation
    np.testing.assert_array_equal(ds_att.power.sel(layer="surface").values,
                                  ds_slab_inc.power.sel(layer="surface").values)


# ------------------------------------- coherent flat-slab vs referee (smoke)

def test_flat_slab_coherent_vs_referee():
    """Smoke version of the M17 two-media brute-force check: kernel bed-layer
    field vs an exact per-sample flat-interface Fermat solve + direct
    summation over a sub-wavelength bed sampling (same hard-edged aperture).

    Compared on the window-integrated complex field: the per-bin split is
    binning quantization, not field accuracy -- an LPA facet books its whole
    contribution at its center delay, while the fine referee spreads it over
    the ~3 m-of-optical-path bins (verified: an f64 LPA referee on the same
    5 m facets matches the kernel per-bin to 4 digits). Envelope within a few
    %, phase within a few degrees; peak bin index must agree."""
    eps_ice, eps_bed = 3.17, 8.0
    n = np.sqrt(eps_ice)
    f0 = 195e6
    k0 = 2.0 * np.pi * f0 / C
    t0, dt, n_samples = 3.9e-6, 2e-8, 30
    scene = syn.slab_scene(surface=500.0, depth=60.0, extent=80.0, posting=10.0,
                           altitude=500.0, n_traces=2)
    cfg = SimConfig(
        mode="coherent",
        radar=RadarConfig(dt=dt, n_samples=n_samples, t0=t0, f0=f0),
        facets=FacetConfig(spacing=5.0), media=_media(1.0, eps_ice, eps_bed),
        interfaces=[DemInterface(name="surface"), DemInterface(name="bed")])
    ds = soundersim.simulate(scene, cfg)
    kern = ds.field.sel(layer="bed")[0].values

    # Referee: same frame/track; bed sampled at lambda_ice/8 over the same
    # footprint; exact f64 Snell solve against the surface mean plane.
    frame = LocalFrame.centered_on(scene)
    track = nav_to_frame(scene.nav_llh, frame)
    p = track.positions[0]
    lam_ice = (C / f0) / n
    fine = build_facets(scene.dems[1], scene.transform, scene.crs, frame,
                        spacing=lam_ice / 8.0)
    surf = build_facets(scene.dems[0], scene.transform, scene.crs, frame)
    w = (surf.areas / surf.areas.sum())[:, None]
    mp = (surf.centers * w).sum(0)
    mn = (surf.normals * w).sum(0)
    mn /= np.linalg.norm(mn)
    r = snell_crossing(p, fine.centers, mp, mn, 1.0, n, xp=np)
    assert r.valid.all()
    c1, c2 = np.cos(r.theta1), np.cos(r.theta2)
    tau2 = 1.0 - fresnel_te(1.0, eps_ice, c1).gamma ** 2
    l_par = r.s1 + r.s2 * c1 ** 2 / (n * c2 ** 2)
    l_perp = r.s1 + r.s2 / n
    flux = c1 / (n * c2)
    d = r.x - fine.centers
    s2v = np.linalg.norm(d, axis=1)
    rhat = d / s2v[:, None]
    cos_t = np.sum(rhat * fine.normals, axis=1)
    opl = r.s1 + n * r.s2
    contrib = (1j * (k0 * n / (2.0 * np.pi)) * fresnel_normal(eps_ice, eps_bed)
               * cos_t * fine.areas * tau2 * flux / (l_par * l_perp)
               * np.exp(-2j * k0 * opl))
    bins = np.floor((2.0 * opl / C - t0) / dt).astype(int)
    ok = (bins >= 0) & (bins < n_samples)
    ref = np.zeros(n_samples, complex)
    np.add.at(ref, bins[ok], contrib[ok])

    assert np.abs(kern).argmax() == np.abs(ref).argmax()
    ratio = kern.sum() / ref.sum()
    assert abs(abs(ratio) - 1.0) < 0.05
    assert abs(np.angle(ratio, deg=True)) < 5.0


# ------------------------------- dropped channel: shadow/invalid, bookkeeping

def test_steep_bed_invalid_paths_dropped_not_nan():
    """A bed poking above the surface plane makes same-side (shadowed) solve
    geometry: those contributions land in the per-layer dropped channel and
    the outputs stay finite. In-window + dropped energy is window-invariant."""
    scene = syn.tilted_bed_scene(surface=500.0, depth=100.0, slope_deg=8.0,
                                 extent=4000.0, n_traces=3, altitude=1000.0)
    assert scene.dems[1].max() > scene.dems[0].max()  # bed above surface
    media = _media(1.0, 3.17, 6.0)
    wide = soundersim.simulate(scene, _slab_cfg(
        "incoherent", media, n_samples=1250, dt=1e-8, t0=0.0))
    narrow = soundersim.simulate(scene, _slab_cfg(
        "incoherent", media, n_samples=150, dt=1e-8, t0=6.5e-6))
    for ds in (wide, narrow):
        assert np.isfinite(ds.power).all()
        assert np.isfinite(ds.dropped_power).all()
        assert (ds.dropped_power.sel(layer="bed") > 0).all()
    # energy bookkeeping: binned + dropped is the same total in any window
    for layer in ("surface", "bed"):
        tot_w = (wide.power.sel(layer=layer).sum("twtt")
                 + wide.dropped_power.sel(layer=layer)).values
        tot_n = (narrow.power.sel(layer=layer).sum("twtt")
                 + narrow.dropped_power.sel(layer=layer)).values
        np.testing.assert_allclose(tot_w, tot_n, rtol=1e-4)
    assert (narrow.dropped_power.sel(layer="bed")
            > wide.dropped_power.sel(layer="bed")).all()


# --------------------------------------- >1 crossing: sequential firn chain

def test_offset_stack_chained_delays():
    """Three-interface offset stack (surface + 2 firn layers): nadir delays
    accumulate 2*d_i*sqrt(eps_i)/c per leg exactly (the sequential chain is
    exact on the vertical path), for every layer."""
    scene = syn.offset_stack_scene(surface=500.0, spacings=(3.0, 4.0),
                                   extent=2000.0, n_traces=2, altitude=1000.0)
    eps = [1.0, 1.8, 2.5, 3.5]
    cfg = SimConfig(
        mode="incoherent",
        radar=RadarConfig(dt=5e-9, n_samples=1450, t0=0.0),
        facets=FacetConfig(), media=_media(*eps),
        interfaces=[DemInterface(name="surface"),
                    OffsetInterface(name="l1", reference="surface", offset=-3.0),
                    OffsetInterface(name="l2", reference="surface", offset=-7.0)])
    ds = soundersim.simulate(scene, cfg)
    assert list(ds.layer.values) == ["surface", "l1", "l2"]
    # Base on the dataset's own surface nadir (the local-frame platform height
    # picks up ~0.3 mm of ellipsoid curvature at the off-center trace).
    expected = float(ds.nadir_twtt.sel(layer="surface")[0])
    depths = [0.0, 3.0, 4.0]
    for j, name in enumerate(["surface", "l1", "l2"]):
        expected += 2.0 * depths[j] * np.sqrt(eps[j]) / C if j else 0.0
        assert float(ds.nadir_twtt.sel(layer=name)[0]) == pytest.approx(
            expected, rel=1e-12)
        trace = ds.power.sel(layer=name)[0].values
        assert np.nonzero(trace)[0][0] == int(np.floor(expected / 5e-9))
    assert np.isfinite(ds.power).all()


# ------------------------------------------------- coherent output structure

def test_coherent_multilayer_output_roundtrip(tmp_path):
    """Layer + side dims per docs/output.md; power = |field|^2; combine sums
    FIELDS over layer in coherent mode; save/load round-trips."""
    scene = syn.slab_scene(surface=500.0, depth=300.0, extent=1200.0,
                           n_traces=2, altitude=1000.0)
    cfg = _slab_cfg("coherent", _media(1.0, 3.17, 6.0), n_samples=250,
                    dt=2e-8, t0=6e-6, f0=195e6, spacing=15.0,
                    split_sides=True)
    ds = soundersim.simulate(scene, cfg)
    assert ds.field.dims == ("slow_time", "twtt", "side", "layer")
    assert ds.field.dtype == np.complex64
    assert list(ds.layer.values) == ["surface", "bed"]
    np.testing.assert_array_equal(ds.power.values,
                                  np.abs(ds.field.values) ** 2)
    np.testing.assert_allclose(soundersim.combine(ds, "layer"),
                               np.abs(ds.field.sum("layer")) ** 2)
    np.testing.assert_allclose(soundersim.combine(ds, "side"),
                               np.abs(ds.field.sum("side")) ** 2)
    assert ds.dropped_power.dims == ("slow_time", "layer")
    path = soundersim.save(ds, tmp_path / "multi.nc")
    back = soundersim.load(path)
    assert list(back.layer.values) == ["surface", "bed"]
    np.testing.assert_array_equal(back.field.values, ds.field.values)
    np.testing.assert_array_equal(back.nadir_twtt.values, ds.nadir_twtt.values)
