"""Analytic validation of the brute-force coherent reference (M9).

Anchors (Haynes et al. 2018, "Geometric Power Fall-Off in Radar Sounding"):

- Infinite flat plate / image method (Eq. 19-21): field -> gamma*exp(-2jkh)/(2h).
  This *defines* the normalization convention (see brute_force module docstring
  and claude_notes/coherent_normalization.md).
- Fresnel-zone oscillation of a hard-edged disk (Eq. 14-15):
  |I(a)|^2 oscillates as (1 - cos(k a^2 / h)); in our normalization
  |field(a)|^2 ~= (gamma^2 / (2 h^2)) * (1 - cos(k a^2 / h)) for h >> a.
- First Fresnel zone (Eq. 16-17 territory): a disk of radius
  r_f = sqrt(lambda*h/2) returns 4x the infinite-plate power.

Edge-taper choice (documented per plan): a hard-edged finite plate does NOT
converge pointwise to the image-method value -- the rim contributes a
non-decaying Fresnel oscillation (exactly the Eq. 15 physics). The plate and
convergence tests therefore use a raised-cosine area taper spanning many
Fresnel zones (weights folded into dA), which suppresses the rim contribution
to second order and leaves only the stationary-phase (image) term.

Scenes are sized to keep each test well under ~1 s.
"""

import numpy as np

from soundersim.compare.brute_force import (
    _contributions,
    brute_force_field,
    brute_force_trace,
    flat_disk_samples,
    flat_plate_field,
    flat_rectangle_samples,
)

LAM = 1.0                  # work in units of wavelength
K = 2.0 * np.pi / LAM
GAMMA = -0.281             # air->ice normal-incidence Fresnel coefficient
C = 299792458.0
TWO_PI = 2.0 * np.pi


class _DiskSweep:
    """Cumulative-field lookup for a hard-edged disk of growing radius.

    Sorts per-sample contributions by rho and cumulative-sums, so field(a) for
    any a <= a_max is a searchsorted lookup after a single O(N) pass.
    """

    def __init__(self, h, a_max, spacing):
        pts, nrm, dA = flat_disk_samples(a_max, spacing)
        rho = np.hypot(pts[:, 0], pts[:, 1])
        order = np.argsort(rho)
        self.rho = rho[order]
        contrib, _ = _contributions(np.array([0.0, 0.0, h]), pts[order],
                                    nrm[order], dA[order], K, GAMMA)
        self.csum = np.cumsum(contrib)

    def field(self, a):
        i = np.searchsorted(self.rho, a, side="right")
        return self.csum[i - 1]

    def power(self, a):
        return abs(self.field(a)) ** 2


def test_point_target():
    """Single sample: exact magnitude and phase; trace bins it correctly."""
    rng = np.random.default_rng(7)
    for _ in range(5):
        p = rng.uniform(-20, 20, 3) + np.array([0.0, 0.0, 60.0])
        pt = rng.uniform(-5, 5, 3)
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        dA = 0.0123
        d = p - pt
        r = np.linalg.norm(d)
        cos = float(d @ n) / r
        if cos < 0:  # keep the sample front-facing for the phase check
            n, cos = -n, -cos

        f = brute_force_field(p, pt[None, :], n[None, :], np.array([dA]), K, GAMMA)

        # magnitude: (k*dA*cos/(2*pi))/r^2 times |gamma|, exact
        expected_mag = (K * dA * cos / TWO_PI) / r ** 2 * abs(GAMMA)
        np.testing.assert_allclose(abs(f), expected_mag, rtol=1e-14)

        # phase: -2kr (mod 2pi) after removing the j prefactor and gamma sign
        phase = np.angle(f / (1j * np.sign(GAMMA)))
        dphi = (phase + 2.0 * K * r) % TWO_PI
        assert min(dphi, TWO_PI - dphi) < 1e-12

        # binned trace: everything in bin floor((2r/c - t0)/dt), nothing else
        t0, dt, n_samples = 0.0, 10e-9, 32
        # place the target so its delay lands inside the window
        scale = (n_samples * dt * C / 2) / (2 * r)
        p2 = pt + d * scale * rng.uniform(0.2, 0.9)
        r2 = np.linalg.norm(p2 - pt)
        trace = brute_force_trace(p2, pt[None, :], n[None, :], np.array([dA]),
                                  K, GAMMA, t0, dt, n_samples, C)
        b = int(np.floor((2 * r2 / C - t0) / dt))
        assert 0 <= b < n_samples
        f2 = brute_force_field(p2, pt[None, :], n[None, :], np.array([dA]), K, GAMMA)
        np.testing.assert_allclose(trace[b], f2, rtol=1e-14)
        mask = np.ones(n_samples, bool)
        mask[b] = False
        assert np.all(trace[mask] == 0)


def test_flat_plate_image_method():
    """Tapered large disk at nadir -> gamma*exp(-2jkh)/(2h) within 1% / 1 deg."""
    h = 200.0 * LAM
    a1 = np.sqrt(6.0 * LAM * h)    # taper starts after 6 full-wave Fresnel zones
    a2 = np.sqrt(22.0 * LAM * h)   # ... and spans 16 more (raised cosine)
    pts, nrm, dA = flat_disk_samples(a2, LAM / 10.0, taper_start=a1)
    f = brute_force_field(np.array([0.0, 0.0, h]), pts, nrm, dA, K, GAMMA)
    ref = flat_plate_field(K, h, GAMMA)
    assert abs(abs(f) / abs(ref) - 1.0) < 0.01          # measured ~7e-4
    assert abs(np.degrees(np.angle(f / ref))) < 1.0     # measured ~0.06 deg


def test_fresnel_zone_oscillation():
    """Hard-edged disk: |field(a)|^2 oscillates as (1 - cos(k a^2/h)) (Eq. 15).

    Sweeps a over ~2.4 oscillation periods (a^2 up to 2.4*lambda*h) and checks
    the positions of the first two zeros and first two maxima against the
    exact two-way-phase predictions sqrt((h + n*lam/2)^2 - h^2) (zeros) and
    the half-integer equivalents (maxima).
    """
    h = 200.0 * LAM
    sweep = _DiskSweep(h, np.sqrt(2.4 * LAM * h), LAM / 10.0)
    plate = abs(flat_plate_field(K, h, GAMMA)) ** 2

    def extremum(a_pred, kind):
        a = np.linspace(a_pred - 1.5 * LAM, a_pred + 1.5 * LAM, 601)
        pwr = np.array([sweep.power(ai) for ai in a])
        i = pwr.argmax() if kind == "max" else pwr.argmin()
        return a[i], pwr[i]

    for n in (1, 2):  # zeros: path difference sqrt(h^2+a^2) - h = n*lam/2
        a_pred = np.sqrt((h + n * LAM / 2.0) ** 2 - h ** 2)
        a_meas, p_min = extremum(a_pred, "min")
        assert abs(a_meas - a_pred) < 0.05 * LAM        # measured <= 0.005
        assert p_min < 0.01 * plate                     # near-null

    for n in (0, 1):  # maxima: path difference = (n + 1/2)*lam/2
        a_pred = np.sqrt((h + (n + 0.5) * LAM / 2.0) ** 2 - h ** 2)
        a_meas, p_max = extremum(a_pred, "max")
        assert abs(a_meas - a_pred) < 0.05 * LAM        # measured <= 0.005
        assert abs(p_max / (4.0 * plate) - 1.0) < 0.05  # peaks ~ 4x plate

    # overall shape vs Eq. 15 (h >> a form), relative to the oscillation peak
    a_grid = np.linspace(2.0 * LAM, np.sqrt(2.3 * LAM * h), 150)
    meas = np.array([sweep.power(a) for a in a_grid])
    pred = (GAMMA ** 2 / (2.0 * h ** 2)) * (1.0 - np.cos(K * a_grid ** 2 / h))
    assert np.max(np.abs(meas - pred)) / pred.max() < 0.05  # measured ~0.027


def test_first_fresnel_zone_power():
    """Disk of radius r_f = sqrt(lam*h/2) -> 4x the infinite-plate power."""
    h = 200.0 * LAM
    r_f = np.sqrt(LAM * h / 2.0)
    pts, nrm, dA = flat_disk_samples(r_f, LAM / 10.0)
    f = brute_force_field(np.array([0.0, 0.0, h]), pts, nrm, dA, K, GAMMA)
    plate = abs(flat_plate_field(K, h, GAMMA)) ** 2
    assert abs(abs(f) ** 2 / (4.0 * plate) - 1.0) < 0.03  # measured ~0.0025


def test_convergence_halving_spacing():
    """Halving sample spacing changes the tapered-plate field by < 0.1%."""
    h = 60.0 * LAM
    a1, a2 = np.sqrt(6.0 * LAM * h), np.sqrt(22.0 * LAM * h)
    p = np.array([0.0, 0.0, h])
    fields = {}
    for sp in (LAM / 10.0, LAM / 20.0):
        pts, nrm, dA = flat_disk_samples(a2, sp, taper_start=a1)
        fields[sp] = brute_force_field(p, pts, nrm, dA, K, GAMMA)
    rel = abs(fields[LAM / 20.0] - fields[LAM / 10.0]) / abs(fields[LAM / 20.0])
    assert rel < 1e-3  # measured ~2e-8 (cell-centered midpoint rule)


def test_rectangle_helper_area_and_trace():
    """Rectangle helper: exact total area; trace field sums equal total field."""
    lx, ly, sp = 3.2, 1.7, 0.11
    pts, nrm, dA = flat_rectangle_samples(lx, ly, sp)
    np.testing.assert_allclose(dA.sum(), lx * ly, rtol=1e-12)
    assert np.all(nrm[:, 2] == 1.0)

    p = np.array([0.4, -0.2, 25.0])
    f = brute_force_field(p, pts, nrm, dA, K, GAMMA)
    t0 = 2 * 24.0 / C
    trace = brute_force_trace(p, pts, nrm, dA, K, GAMMA, t0, 1e-9, 64, C)
    np.testing.assert_allclose(trace.sum(), f, rtol=1e-12)
