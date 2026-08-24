"""Antenna pattern tests (M22): config round-trip, closed-form gains through
the kernels (dipole / uniform array / tabulated), pattern-weighted flat
surface vs g**4, roll rotation, multilayer departure-direction gain, and
isotropic-default equivalence.

Convention under test (antenna.py / physics.py): g = ONE-WAY FIELD gain;
coherent kernels weight the field by g**2, the incoherent kernel weights
power by g**4.
"""

import numpy as np
import pytest

from soundersim import antenna, simulate
from soundersim import synthetic as syn
from soundersim.config import (AntennaConfig, DemInterface, FacetConfig,
                               Medium, RadarConfig, SimConfig)
from soundersim.kernels.coherent import coherent_cluttergram
from soundersim.kernels.incoherent import incoherent_cluttergram
from soundersim.physics import C

F0 = 195e6
LAM = C / F0
K = 2.0 * np.pi / LAM
U_AT = np.array([1.0, 0.0, 0.0])
U_CT = np.array([0.0, -1.0, 0.0])  # right of +E travel


def dipole_gain(cos_psi):
    """Closed-form half-wave dipole one-way FIELD gain (simc's formula)."""
    return np.cos(np.pi / 2.0 * cos_psi) / np.sqrt(1.0 - cos_psi ** 2)


def array_factor(u, n, d_lam):
    """|sin(N x)/(N sin x)|, x = pi d u -- uniform unsteered linear array."""
    x = np.pi * d_lam * np.asarray(u, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        af = np.sin(n * x) / (n * np.sin(x))
    return np.abs(np.where(np.abs(np.sin(x)) < 1e-12, 1.0, af))


# ---------------------------------------------------------------- config

def test_config_round_trip_all_kinds():
    for ant in (
        AntennaConfig(),
        AntennaConfig(kind="dipole", axis="cross_track", roll_source="nav"),
        AntennaConfig(kind="array", n_elements=5, spacing_lam=0.5),
        AntennaConfig(kind="tabulated", theta_deg=[0.0, 45.0, 90.0],
                      gain=[1.0, 0.7, 0.1]),
    ):
        cfg = SimConfig(mode="incoherent",
                        radar=RadarConfig(dt=1e-8, n_samples=64, t0=0.0,
                                          antenna=ant),
                        facets=FacetConfig())
        assert SimConfig.model_validate_json(cfg.model_dump_json()) == cfg
    # default is isotropic + no roll
    rc = RadarConfig(dt=1e-8, n_samples=64, t0=0.0)
    assert rc.antenna.kind == "isotropic"
    assert rc.antenna.roll_source == "none"


def test_config_validation():
    with pytest.raises(ValueError):
        AntennaConfig(kind="array", n_elements=1)
    with pytest.raises(ValueError):
        AntennaConfig(kind="array", spacing_lam=0.0)
    with pytest.raises(ValueError):
        AntennaConfig(kind="tabulated")  # no samples
    with pytest.raises(ValueError):
        AntennaConfig(kind="tabulated", theta_deg=[0, 10], gain=[1.0])
    with pytest.raises(ValueError):
        AntennaConfig(kind="tabulated", theta_deg=[10.0, 0.0], gain=[1, 1])
    with pytest.raises(ValueError):
        AntennaConfig(kind="tabulated", theta_deg=[0.0, 10.0], gain=[1, -1])


# ------------------------------------------------- single-facet kernel ratio

def _facet_ratio(dhats, ant, r=3000.0):
    """Per-trace two-way pattern weight through both kernels.

    One facet at the origin (normal +z); trace t sits at -r*dhat[t], so the
    departure direction is exactly dhats[t]. Returns (field_ratio,
    power_ratio): pattern run / isotropic run at each trace's single occupied
    bin (expected g**2 in field, g**4 in power).
    """
    dhats = np.asarray(dhats, np.float64)
    T = len(dhats)
    pos = -r * dhats
    uat = np.tile(U_AT, (T, 1))
    uct = np.tile(U_CT, (T, 1))
    ctr = np.zeros((1, 3))
    nrm = np.array([[0.0, 0.0, 1.0]])
    area = np.array([25.0])
    e1, e2 = np.array([[5.0, 0, 0.0]]), np.array([[0.0, 5.0, 0.0]])
    win = dict(t0=2.0 * (r - 50.0) / C, dt=1e-8, n_samples=128, c=C)
    pat = antenna.pattern_args(ant, uat, uct)

    args = (pos, uct, ctr, nrm, area)
    f_iso, _ = coherent_cluttergram(*args, e1, e2, k=K, gamma=-0.28, **win)
    f_pat, _ = coherent_cluttergram(*args, e1, e2, k=K, gamma=-0.28,
                                    pattern=pat, **win)
    p_iso, _ = incoherent_cluttergram(*args, **win)
    p_pat, _ = incoherent_cluttergram(*args, pattern=pat, **win)

    b = np.abs(f_iso).argmax(axis=1)
    t = np.arange(T)
    return (np.abs(f_pat[t, b]) / np.abs(f_iso[t, b]),
            p_pat[t, b] / p_iso[t, b])


def test_dipole_gain_closed_form():
    """Kernel two-way weight vs cos((pi/2) cos psi)/sin(psi), specific angles.

    Dipole axis along-track (+x). Departure directions at psi = 90 (broadside,
    g = 1), 60 (g = 0.8164966), 45 (g = 0.6279332) degrees from the axis.
    """
    cos_psi = np.array([0.0, 0.5, np.cos(np.pi / 4)])
    dhats = np.column_stack([cos_psi, np.zeros(3),
                             -np.sqrt(1.0 - cos_psi ** 2)])
    g = dipole_gain(np.array([1e-12, 0.5, np.cos(np.pi / 4)]))
    g[0] = 1.0  # exact broadside
    np.testing.assert_allclose(g, [1.0, 0.8164966, 0.6279332], atol=1e-6)

    ant = AntennaConfig(kind="dipole", axis="along_track")
    fr, pr = _facet_ratio(dhats, ant)
    np.testing.assert_allclose(fr, g ** 2, rtol=2e-4)
    np.testing.assert_allclose(pr, g ** 4, rtol=5e-4)


def test_dipole_null_along_axis():
    """g -> 0 along the dipole axis, in the gain functions and the kernel."""
    ant = AntennaConfig(kind="dipole", axis="along_track")
    # NumPy reference at and near the axis
    g_axis = antenna.field_gain(ant, np.array([1.0, 0.0, 0.0]), U_AT, U_CT)
    assert abs(float(g_axis)) < 1e-6
    th = np.deg2rad([0.5, 2.0, 8.0])  # angle FROM the axis
    d_near = np.column_stack([np.cos(th), np.zeros(3), -np.sin(th)])
    g_near = antenna.field_gain(ant, d_near, U_AT, U_CT)
    assert np.all(np.diff(g_near) > 0) and g_near[-1] < 0.25
    # jnp kernel gain matches the reference
    import jax.numpy as jnp
    gj = antenna.gain_fn("dipole")(jnp.asarray(d_near), jnp.asarray(U_AT),
                                   0.0, 0.0)
    np.testing.assert_allclose(np.asarray(gj), g_near, rtol=1e-3, atol=1e-7)
    # (rtol: jnp evaluates in float32 here; f64 belongs to the multilayer path)
    # kernel-level: facet along the axis (tilted normal so cos(theta) != 0)
    r = 3000.0
    pos = np.array([[-r, 0.0, 0.0]])
    nrm = np.array([[-1.0, 0.0, 0.0]])
    win = dict(t0=2.0 * (r - 50.0) / C, dt=1e-8, n_samples=128, c=C)
    pat = antenna.pattern_args(ant, U_AT[None], U_CT[None])
    a = (pos, U_CT[None], np.zeros((1, 3)), nrm, np.array([25.0]))
    p_iso, _ = incoherent_cluttergram(*a, **win)
    p_pat, _ = incoherent_cluttergram(*a, pattern=pat, **win)
    assert p_iso.max() > 0
    # f32 null depth: cos(pi/2) rounds to ~ -4.4e-8 and the sin(psi) clamp is
    # 1e-6, so the kernel's on-axis g is ~4e-2-ish at f32 -> g**4 <= ~4e-6
    # (>= 54 dB down in power); the exact null is checked in f64 above.
    assert p_pat.max() < 1e-5 * p_iso.max()


def test_array_factor_closed_form():
    """5-element cross-track array, d = 0.5 lam: mainlobe/null/sidelobe.

    u = sin(theta_ct); nulls at u = 2m/5 (theta = 23.578, 53.13 deg);
    AF(0.2) = 0.647214, first sidelobe AF(0.6) = 0.247214 (analytic).
    """
    n, d = 5, 0.5
    us = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    af = array_factor(us, n, d)
    np.testing.assert_allclose(
        af, [1.0, 0.6472136, 0.0, 0.2472136, 0.0], atol=1e-7)
    # null positions in angle: arcsin(0.4) = 23.578 deg from nadir
    assert np.rad2deg(np.arcsin(0.4)) == pytest.approx(23.578, abs=1e-3)

    # departure directions in the cross-track plane: dhat . u_ct = u
    dhats = np.column_stack([np.zeros(5), -us, -np.sqrt(1.0 - us ** 2)])
    ant = AntennaConfig(kind="array", n_elements=n, spacing_lam=d)
    fr, pr = _facet_ratio(dhats, ant)
    np.testing.assert_allclose(fr, af ** 2, rtol=3e-4, atol=1e-8)
    np.testing.assert_allclose(pr, af ** 4, rtol=6e-4, atol=1e-10)
    # main-lobe -3 dB half-width: |AF|^2 = 0.5 at u = 0.18032 (N=5, d=0.5;
    # exact root of sin(5x) = sqrt(1/2)*5 sin(x), theta_3dB = 10.39 deg --
    # close to the large-N approximation 0.886/(N d)/2 = 0.1772)
    u3 = 0.18032
    g3 = array_factor(u3, n, d)
    assert g3 ** 2 == pytest.approx(0.5, abs=0.001)
    fr3, _ = _facet_ratio(
        np.array([[0.0, -u3, -np.sqrt(1.0 - u3 ** 2)]]),
        AntennaConfig(kind="array", n_elements=n, spacing_lam=d))
    assert fr3[0] == pytest.approx(0.5, abs=0.002)  # g**2 = half power


def test_array_params_are_traced_no_recompile():
    """Same pattern kind, different n/spacing/vector values: the jitted
    callable is reused (one jit cache entry), per the M19 no-recompile rule."""
    from soundersim.kernels.incoherent import _incoherent_fn

    ctr = np.zeros((1, 3))
    nrm = np.array([[0.0, 0.0, 1.0]])
    area = np.array([25.0])
    pos = np.array([[0.0, 0.0, 3000.0]])
    win = dict(t0=1.9e-5, dt=1e-8, n_samples=64, c=C)
    for n_el, d in ((5, 0.5), (7, 0.35), (3, 0.6)):
        ant = AntennaConfig(kind="array", n_elements=n_el, spacing_lam=d)
        pat = antenna.pattern_args(ant, U_AT[None], U_CT[None])
        incoherent_cluttergram(pos, U_CT[None], ctr, nrm, area, pattern=pat,
                               **win)
    fn = _incoherent_fn(False, 64, "array")
    assert fn._cache_size() == 1


def test_tabulated_round_trips_dipole():
    """A finely sampled dipole pattern fed back as `tabulated` reproduces the
    analytic dipole through the kernel (directions in the vertical
    along-track plane, where the dipole IS a function of theta-from-nadir)."""
    th_tab = np.linspace(0.0, 89.0, 357)  # 0.25 deg sampling
    g_tab = dipole_gain(np.sin(np.deg2rad(th_tab)) + 1e-12)
    tab = AntennaConfig(kind="tabulated", theta_deg=list(th_tab),
                        gain=list(g_tab))
    dip = AntennaConfig(kind="dipole", axis="along_track")

    th = np.deg2rad([0.0, 15.0, 30.0, 47.5, 70.0])
    dhats = np.column_stack([np.sin(th), np.zeros(5), -np.cos(th)])
    fr_tab, pr_tab = _facet_ratio(dhats, tab)
    fr_dip, pr_dip = _facet_ratio(dhats, dip)
    np.testing.assert_allclose(fr_tab, fr_dip, rtol=1e-3)
    np.testing.assert_allclose(pr_tab, pr_dip, rtol=2e-3)


# ------------------------------------------------------ flat-surface weighting

def test_flat_surface_power_ratio_matches_g4():
    """Incoherent flat surface with an axisymmetric tabulated pattern
    g(theta) = cos(theta): per-bin P_pattern/P_isotropic equals the
    facet-exact g**4 average, which is cos**4(theta_bin); two-bin ratio
    equals g**4(theta1)/g**4(theta2) with geometry canceled."""
    th_tab = np.linspace(0.0, 90.0, 181)
    ant = AntennaConfig(kind="tabulated", theta_deg=list(th_tab),
                        gain=list(np.cos(np.deg2rad(th_tab))))
    scene = syn.flat_scene(elevation=500.0, altitude=1000.0, extent=6000.0,
                           n_traces=3)

    def cfg(a):
        return SimConfig(mode="incoherent",
                         radar=RadarConfig(dt=1e-8, n_samples=1250, t0=0.0,
                                           antenna=a),
                         facets=FacetConfig())

    ds_iso = simulate(scene, cfg(AntennaConfig()))
    ds_pat = simulate(scene, cfg(ant))
    p_iso = ds_iso.power.values[1]
    p_pat = ds_pat.power.values[1]

    h = 1000.0
    t0_surf = 2.0 * h / C
    twtt = ds_iso.twtt.values
    occupied = np.nonzero(p_iso > 0)[0]
    # bins at ~30 and ~55 deg incidence: r = h/cos(theta)
    picks = []
    for th_deg in (30.0, 55.0):
        t_target = 2.0 * h / np.cos(np.deg2rad(th_deg)) / C
        picks.append(occupied[np.abs(twtt[occupied] - t_target).argmin()])
    b1, b2 = picks

    ratios = []
    for b in (b1, b2):
        ratio = p_pat[b] / p_iso[b]
        # bin-center incidence from the bin's mid twtt
        r_mid = C * (twtt[b] + 0.5e-8) / 2.0
        g4 = (h / r_mid) ** 4  # cos**4(theta)
        assert ratio == pytest.approx(g4, rel=0.03), (b, ratio, g4)
        ratios.append(ratio)
    # two-angle ratio: geometry factors cancel exactly in the iso-normalized
    # ratios, leaving g**4(theta1)/g**4(theta2)
    r1_mid = C * (twtt[b1] + 0.5e-8) / 2.0
    r2_mid = C * (twtt[b2] + 0.5e-8) / 2.0
    expect = (r2_mid / r1_mid) ** 4
    assert ratios[0] / ratios[1] == pytest.approx(expect, rel=0.05)


# ----------------------------------------------------------------- multilayer

def test_multilayer_departure_gain():
    """Refracted-path kernel applies g at the AIR-leg departure direction:
    with unity-contrast media the path is straight and the bed layer's
    per-bin pattern/isotropic power ratio is cos**4(theta) exactly (same
    axisymmetric cos-pattern as the flat-surface test)."""
    th_tab = np.linspace(0.0, 90.0, 181)
    ant = AntennaConfig(kind="tabulated", theta_deg=list(th_tab),
                        gain=list(np.cos(np.deg2rad(th_tab))))
    media = [Medium(name="air", eps_r=1.0), Medium(name="air2", eps_r=1.0),
             Medium(name="bed", eps_r=6.0)]
    scene = syn.slab_scene(surface=500.0, depth=200.0, media=media,
                           extent=4000.0, n_traces=3, altitude=1000.0)

    def cfg(a):
        return SimConfig(
            mode="incoherent",
            radar=RadarConfig(dt=1e-8, n_samples=1250, t0=0.0, antenna=a),
            facets=FacetConfig(spacing=50.0), media=media,
            interfaces=[DemInterface(name="surface"),
                        DemInterface(name="bed")])

    ds_iso = simulate(scene, cfg(AntennaConfig()))
    ds_pat = simulate(scene, cfg(ant))
    p_iso = ds_iso.power.values[1, :, 1]  # mid trace, bed layer
    p_pat = ds_pat.power.values[1, :, 1]
    twtt = ds_iso.twtt.values

    h = 1200.0  # platform 1000 m above the 500 m surface; bed at 300 m
    occupied = np.nonzero(p_iso > 0)[0]
    for th_deg in (20.0, 40.0):
        t_target = 2.0 * h / np.cos(np.deg2rad(th_deg)) / C
        b = occupied[np.abs(twtt[occupied] - t_target).argmin()]
        r_mid = C * (twtt[b] + 0.5e-8) / 2.0
        g4 = (h / r_mid) ** 4
        assert p_pat[b] / p_iso[b] == pytest.approx(g4, rel=0.03)


# ----------------------------------------------------------------------- roll

def test_roll_rotates_pattern_frame():
    """+90 deg roll (right wing down) tilts the nadir boresight to the LEFT
    of travel (right-handed rotation about u_at in ENU); dipole along-track
    axis is roll-invariant."""
    ant = AntennaConfig(kind="tabulated", theta_deg=[0.0, 90.0],
                        gain=[1.0, 0.0], roll_source="nav")
    v = antenna.frame_vectors(ant, U_AT[None], U_CT[None],
                              roll=np.array([np.pi / 2]))
    np.testing.assert_allclose(v[0], [0.0, 1.0, 0.0], atol=1e-12)  # left (N)
    dip = AntennaConfig(kind="dipole", axis="along_track", roll_source="nav")
    v = antenna.frame_vectors(dip, U_AT[None], U_CT[None],
                              roll=np.array([0.7]))
    np.testing.assert_allclose(v[0], U_AT, atol=1e-12)
    # cross-track dipole rolls: cos(phi) u_ct + sin(phi) (u_at x u_ct)
    dip_ct = AntennaConfig(kind="dipole", axis="cross_track",
                           roll_source="nav")
    phi = 0.3
    v = antenna.frame_vectors(dip_ct, U_AT[None], U_CT[None],
                              roll=np.array([phi]))
    expect = np.cos(phi) * U_CT + np.sin(phi) * np.cross(U_AT, U_CT)
    np.testing.assert_allclose(v[0], expect, atol=1e-12)


def test_roll_shifts_power_across_track():
    """simulate() with roll_source="nav" + scene.nav_roll: a rolled narrow
    axisymmetric beam moves flat-surface power to the left side (split_sides).

    Assertions compare rolled vs un-rolled runs side-by-side (the raw flat
    scene is a few percent L/R asymmetric by itself: the discrete near-nadir
    facets -- the r**-4 peak -- land on one side of the split, so absolute
    L == R symmetry is not the right invariant).
    """
    th_tab = np.linspace(0.0, 90.0, 91)
    gain = list(np.cos(np.deg2rad(th_tab)) ** 6)  # narrow-ish beam
    scene = syn.flat_scene(elevation=500.0, altitude=1000.0, extent=6000.0,
                           n_traces=3)
    scene.nav_roll = np.full(3, np.deg2rad(25.0))  # right wing down

    def run(ant):
        cfg = SimConfig(mode="incoherent", split_sides=True,
                        radar=RadarConfig(dt=1e-8, n_samples=1250, t0=0.0,
                                          antenna=ant),
                        facets=FacetConfig())
        p = simulate(scene, cfg).power.values[1]
        return p[..., 0].sum(), p[..., 1].sum()  # (left, right)

    l0, r0 = run(AntennaConfig(kind="tabulated", theta_deg=list(th_tab),
                               gain=gain, roll_source="none"))
    l1, r1 = run(AntennaConfig(kind="tabulated", theta_deg=list(th_tab),
                               gain=gain, roll_source="nav"))
    # +25 deg roll (right wing down) tilts the boresight LEFT: the rolled/
    # unrolled ratio per side cancels the scene geometry exactly (same scene,
    # same pattern, only the roll differs). Measured: 1.62 / 0.035 / 46.
    assert l1 / l0 > 1.3
    assert r1 / r0 < 0.1
    assert (l1 / r1) / (l0 / r0) > 10.0


# ----------------------------------------------------------- isotropic default

def test_isotropic_explicit_equals_default():
    """An explicit isotropic AntennaConfig is bit-identical to the default
    (same compiled program; pattern args are None in both cases)."""
    scene = syn.flat_scene(elevation=500.0, altitude=1000.0, extent=2000.0,
                           n_traces=3)
    rc = dict(dt=1e-8, n_samples=1250, t0=0.0, f0=F0)
    a = simulate(scene, SimConfig(
        mode="coherent", radar=RadarConfig(**rc), facets=FacetConfig()))
    b = simulate(scene, SimConfig(
        mode="coherent",
        radar=RadarConfig(**rc, antenna=AntennaConfig(kind="isotropic",
                                                      roll_source="nav")),
        facets=FacetConfig()))
    assert np.array_equal(a.field.values, b.field.values)
    assert np.array_equal(a.dropped_power.values, b.dropped_power.values)


# --------------------------------------- realistic david-line models (M-ant)

TX8 = [39.4, 72.6, 125.8, 177.6, 156.0, 120.7, 92.9, 44.8]  # 2017 Basler TX
HANN8 = [float(np.sin(np.pi * k / 9.0) ** 2) for k in range(1, 9)]
D8 = 0.304  # 0.468 m at 195 MHz, in carrier wavelengths


def tapered_af(u, w, d_lam):
    """|sum_m w_m exp(i 2 pi d m u)| / sum w, element index m centered."""
    w = np.asarray(w, np.float64)
    m = np.arange(len(w)) - (len(w) - 1) / 2.0
    ph = 2.0 * np.pi * d_lam * np.asarray(u, np.float64)[..., None] * m
    return np.hypot(np.cos(ph) @ w, np.sin(ph) @ w) / w.sum()


def _ct_dhats(theta_deg):
    """Departure directions theta off-nadir in the cross-track plane."""
    th = np.deg2rad(np.asarray(theta_deg, np.float64))
    return np.column_stack([np.zeros(len(th)), -np.sin(th), -np.cos(th)])


def test_array_tapered_closed_form_angles():
    """g = sqrt(AF_tx * AF_rx) at nadir/30/60/75/82 deg; the David clutter
    band sits at the two-way (40 log10 g) sidelobe floor near -60 dB."""
    ant = AntennaConfig(kind="array_tapered", spacing_lam=D8,
                        tx_weights=TX8, rx_weights=HANN8)
    angles = [0.0, 30.0, 60.0, 75.0, 82.0]
    g = antenna.field_gain(ant, _ct_dhats(angles), U_AT, U_CT)
    u = np.sin(np.deg2rad(angles))
    expect = np.sqrt(tapered_af(u, TX8, D8) * tapered_af(u, HANN8, D8))
    np.testing.assert_allclose(g, expect, rtol=1e-12)
    assert g[0] == pytest.approx(1.0, abs=1e-12)  # peak-normalized at nadir
    tw = 40.0 * np.log10(g[1:])                   # two-way POWER dB
    assert -30.0 < tw[0] < -27.0                  # 30 deg: -28.5
    assert -59.0 < tw[1] < -56.5                  # 60 deg: -57.8
    assert -60.5 < tw[2] < -57.5                  # 75 deg: -59.0
    assert -62.0 < tw[3] < -59.0                  # 82 deg: -60.6


def test_array_tapered_uniform_matches_array_kind():
    """Uniform tx = rx weights collapse to the plain array factor."""
    n, d = 8, 0.304
    us = np.array([0.0, 0.1, 0.3, 0.6, 0.9])
    dhats = np.column_stack([np.zeros(5), -us, -np.sqrt(1.0 - us ** 2)])
    g_t = antenna.field_gain(
        AntennaConfig(kind="array_tapered", spacing_lam=d,
                      tx_weights=[1.0] * n, rx_weights=[1.0] * n),
        dhats, U_AT, U_CT)
    np.testing.assert_allclose(g_t, array_factor(us, n, d), atol=1e-12)


def test_array_tapered_kernel_two_way():
    """Kernels weight field by g**2 and power by g**4 for array_tapered."""
    ant = AntennaConfig(kind="array_tapered", spacing_lam=D8,
                        tx_weights=TX8, rx_weights=HANN8)
    dhats = _ct_dhats([0.0, 20.0, 45.0, 75.0])
    g = antenna.field_gain(ant, dhats, U_AT, U_CT)
    fr, pr = _facet_ratio(dhats, ant)
    np.testing.assert_allclose(fr, g ** 2, rtol=2e-3, atol=1e-8)
    np.testing.assert_allclose(pr, g ** 4, rtol=4e-3, atol=1e-12)


def finite_dipole_gain(cos_psi, L):
    """Closed-form finite-length thin dipole, peak-normalized at broadside."""
    kh = np.pi * L
    s = np.sqrt(1.0 - np.asarray(cos_psi, np.float64) ** 2)
    return np.abs(np.cos(kh * cos_psi) - np.cos(kh)) / (s * (1 - np.cos(kh)))


def test_finite_dipole_closed_form_angles():
    """Cross-track finite dipole (MARFA wing plate, L = 0.4 lam at 60 MHz):
    two-way values at the David clutter angles vs the -16/-36 dB bracket."""
    ant = AntennaConfig(kind="finite_dipole", axis="cross_track",
                        length_lam=0.4)
    angles = [0.0, 30.0, 60.0, 75.0, 82.0]
    g = antenna.field_gain(ant, _ct_dhats(angles), U_AT, U_CT)
    # psi from the cross-track axis: cos(psi) = -sin(theta) * (u_ct . -y)
    cos_psi = np.sin(np.deg2rad(angles))
    np.testing.assert_allclose(g, finite_dipole_gain(cos_psi + 1e-300, 0.4),
                               rtol=1e-9)
    assert g[0] == pytest.approx(1.0, abs=1e-12)  # nadir is broadside
    tw = 40.0 * np.log10(g[1:])
    assert -3.5 < tw[0] < -2.8    # 30 deg: -3.1
    assert -14.5 < tw[1] < -13.4  # 60 deg: -13.9
    assert -26.5 < tw[2] < -25.0  # 75 deg: -25.8
    assert -37.5 < tw[3] < -36.0  # 82 deg: -36.7 (inside the -16/-36 bracket)


def test_finite_dipole_limits():
    """L = 0.5 reproduces the half-wave `dipole` kind exactly; L -> 0 tends
    to the short-dipole sin(psi)."""
    dhats = _ct_dhats([5.0, 25.0, 50.0, 75.0, 88.0])
    g_half = antenna.field_gain(
        AntennaConfig(kind="finite_dipole", axis="cross_track",
                      length_lam=0.5), dhats, U_AT, U_CT)
    g_dip = antenna.field_gain(
        AntennaConfig(kind="dipole", axis="cross_track"), dhats, U_AT, U_CT)
    np.testing.assert_allclose(g_half, g_dip, rtol=1e-12)
    g_short = antenna.field_gain(
        AntennaConfig(kind="finite_dipole", axis="cross_track",
                      length_lam=0.01), dhats, U_AT, U_CT)
    cos_psi = np.sin(np.deg2rad([5.0, 25.0, 50.0, 75.0, 88.0]))
    np.testing.assert_allclose(g_short, np.sqrt(1.0 - cos_psi ** 2),
                               rtol=1e-3)


def test_finite_dipole_kernel_two_way():
    ant = AntennaConfig(kind="finite_dipole", axis="cross_track",
                        length_lam=0.4)
    dhats = _ct_dhats([0.0, 30.0, 60.0, 80.0])
    g = antenna.field_gain(ant, dhats, U_AT, U_CT)
    fr, pr = _facet_ratio(dhats, ant)
    np.testing.assert_allclose(fr, g ** 2, rtol=2e-3, atol=1e-8)
    np.testing.assert_allclose(pr, g ** 4, rtol=4e-3, atol=1e-12)


def test_new_kind_config_validation():
    with pytest.raises(ValueError):
        AntennaConfig(kind="array_tapered", spacing_lam=0.3)  # no weights
    with pytest.raises(ValueError):
        AntennaConfig(kind="array_tapered", spacing_lam=0.3,
                      tx_weights=[1, 2, 3], rx_weights=[1, 2])  # mismatch
    with pytest.raises(ValueError):
        AntennaConfig(kind="array_tapered", spacing_lam=0.3,
                      tx_weights=[1, -1], rx_weights=[1, 1])  # negative
    with pytest.raises(ValueError):
        AntennaConfig(kind="array_tapered", spacing_lam=0.0,
                      tx_weights=[1, 1], rx_weights=[1, 1])
    with pytest.raises(ValueError):
        AntennaConfig(kind="finite_dipole", length_lam=0.0)
    with pytest.raises(ValueError):
        AntennaConfig(kind="finite_dipole", length_lam=1.5)
    # round-trip
    ant = AntennaConfig(kind="array_tapered", spacing_lam=D8,
                        tx_weights=TX8, rx_weights=HANN8, roll_source="nav")
    assert AntennaConfig.model_validate_json(ant.model_dump_json()) == ant
