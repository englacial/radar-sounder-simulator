"""Cross-kernel ensemble consistency (M11): ensemble-averaged coherent
|field|^2 per fast-time bin converges to the incoherent kernel's power.

Deterministic scale factor: per facet the coherent kernel's |field|^2 is
(k*G/2pi)^2 * sinc^2(k rhat.e1) * sinc^2(k rhat.e2) * (A cos)^2 / r^4, while
the incoherent kernel bins (A cos)^2 / r^4 with NO Gamma or k/2pi factors. The
test therefore compares against C0 = (k*G/2pi)^2 times the incoherent power,
with the (deterministic, near-1 here) sinc^2 factor computed exactly from the
same facets via the float64 LPA reference.

Ensemble construction (why it looks the way it does -- both properties were
established by measurement):

- Facet phases must be iid for E|sum|^2 = sum|.|^2. Realizations are flat
  horizontal facets with iid Gaussian vertical offsets (correlation length <
  facet size). Height fields with vertex-level correlation (e.g. smoothed
  Gaussian surfaces) leave neighbouring facets' amplitude+phase coupled and a
  ~2-3x coherent excess that no ensemble size removes.
- The kernel bins by range, so within one bin the phase 2k*r spans exactly
  4pi*w/lam radians (w = bin range width): the within-bin phase resultant is
  sinc(2k*w/2), which vanishes only when w is an integer multiple of lam/2.
  The test uses w = lam/2 (dt = lam/c) AND sigma_h = 2*lam so the within-bin
  range density is smooth on the bin scale (sigma_h ~ lam leaves a ~2x excess
  from the density slope; non-multiple w leaves ~15x).

The 30 realizations run as 30 traces of one kernel call: 30 disjoint tiles
spaced 80*lam apart, platform above each tile's center; neighbouring tiles'
returns arrive >20 bins after the analysis window ends.

Measured (seeds fixed): per-bin ratio mean 1.060 over 19 bins, min/max
0.746/1.440 (per-bin speckle s.e. 1/sqrt(30) ~ 0.18); total ratio 1.012;
incoherent-weighted mean sinc^2 per bin 0.966-0.973; total coherent power vs
C0 * incoherent kernel power = 0.982.
"""

import numpy as np

from soundersim import antenna
from soundersim.config import AntennaConfig
from soundersim.kernels.coherent import coherent_cluttergram, lpa_contributions
from soundersim.kernels.incoherent import incoherent_cluttergram

LAM = 1.0
K = 2.0 * np.pi / LAM
GAMMA = -0.281
C = 299792458.0
C0 = (K * abs(GAMMA) / (2.0 * np.pi)) ** 2

N_REAL = 30
NCELL = 48          # facets per tile side (d = lam/2 -> 24 lam tiles)
D = 0.5 * LAM
SIGMA_H = 2.0 * LAM
H = 100.0 * LAM
TILE_DX = 80.0 * LAM


def _tiles():
    """All realizations' facets in one array, plus per-trace positions."""
    ax = (np.arange(NCELL) - (NCELL - 1) / 2) * D
    X, Y = np.meshgrid(ax, ax)
    base = np.column_stack([X.ravel(), Y.ravel(), np.zeros(NCELL ** 2)])
    n = len(base)
    centers = []
    for t in range(N_REAL):
        rng = np.random.default_rng(100 + t)
        tile = base + [t * TILE_DX, 0.0, 0.0]
        tile[:, 2] += SIGMA_H * rng.normal(size=n)
        centers.append(tile)
    centers = np.concatenate(centers)
    m = len(centers)
    normals = np.tile([0.0, 0.0, 1.0], (m, 1))
    areas = np.full(m, D * D)
    e1 = np.tile([D, 0.0, 0.0], (m, 1))
    e2 = np.tile([0.0, D, 0.0], (m, 1))
    positions = np.column_stack([np.arange(N_REAL) * TILE_DX,
                                 np.zeros(N_REAL), np.full(N_REAL, H)])
    return centers, normals, areas, e1, e2, positions


def test_ensemble_coherent_power_converges_to_incoherent():
    centers, normals, areas, e1, e2, positions = _tiles()
    uct = np.tile([0.0, -1.0, 0.0], (N_REAL, 1))
    dt = 2.0 * (LAM / 2.0) / C     # bin range width = lam/2 exactly
    t0 = 2.0 * (H - 7.0 * LAM) / C
    n_samples = 28                 # window ends >20 bins before neighbour tiles

    field, _ = coherent_cluttergram(positions, uct, centers, normals, areas,
                                    e1, e2, k=K, gamma=GAMMA, t0=t0, dt=dt,
                                    n_samples=n_samples, c=C)
    power, _ = incoherent_cluttergram(positions, uct, centers, normals, areas,
                                      t0=t0, dt=dt, n_samples=n_samples, c=C)
    coh = (np.abs(field) ** 2).mean(axis=0)          # ensemble mean per bin
    inc = power.mean(axis=0).astype(np.float64)

    # float64 per-facet |LPA field|^2 (the sinc^2-weighted incoherent sum)
    ref = np.zeros(n_samples)
    n_tile = NCELL ** 2
    for t in range(N_REAL):
        sl = slice(t * n_tile, (t + 1) * n_tile)
        contrib, r = lpa_contributions(positions[t], centers[sl], normals[sl],
                                       areas[sl], e1[sl], e2[sl], K, GAMMA,
                                       xp=np)
        b = np.floor((2.0 * r / C - t0) / dt).astype(int)
        ok = (b >= 0) & (b < n_samples)
        np.add.at(ref, b[ok], np.abs(contrib[ok]) ** 2 / N_REAL)

    big = ref > 0.05 * ref.max()
    assert big.sum() >= 15
    ratio = coh[big] / ref[big]
    # speckle convergence: per-bin s.e. ~ 1/sqrt(N_REAL) ~ 0.18
    assert abs(ratio.mean() - 1.0) < 0.12       # measured 1.060
    assert ratio.min() > 0.55 and ratio.max() < 1.6  # measured 0.746 / 1.440
    assert abs(coh[big].sum() / ref[big].sum() - 1.0) < 0.10  # measured 1.012

    # deterministic cross-kernel constant: ref = C0 * mean(sinc^2) * inc,
    # with mean(sinc^2) in (0.95, 1) for these near-nadir lam/2 facets
    mean_sinc2 = ref[big] / (C0 * inc[big])
    assert np.all((mean_sinc2 > 0.9) & (mean_sinc2 <= 1.0))  # measured .966-.973
    # end-to-end: coherent ensemble power vs C0 * incoherent kernel power
    assert abs(coh[big].sum() / (C0 * inc[big].sum()) - 1.0) < 0.12  # 0.982


def test_ensemble_convergence_with_antenna_pattern():
    """M22 cross-kernel gain-convention consistency: with a STEEP non-isotropic
    pattern (tabulated, g dropping to ~0.3 at the tile edge angle ~7 deg, so
    per-bin g**4 spans >an order of magnitude), the ensemble-averaged coherent
    |field|^2 (fields weighted g**2) still converges to the incoherent kernel's
    power (weighted g**4): any field-vs-power convention mismatch would show as
    a systematic per-bin ratio drift with angle, far outside the speckle s.e.
    """
    centers, normals, areas, e1, e2, positions = _tiles()
    uct = np.tile([0.0, -1.0, 0.0], (N_REAL, 1))
    uat = np.tile([1.0, 0.0, 0.0], (N_REAL, 1))
    dt = 2.0 * (LAM / 2.0) / C
    t0 = 2.0 * (H - 7.0 * LAM) / C
    n_samples = 28

    th = np.linspace(0.0, 90.0, 181)
    ant = AntennaConfig(kind="tabulated", theta_deg=list(th),
                        gain=list(1.0 / (1.0 + (th / 4.0) ** 2)))
    pat = antenna.pattern_args(ant, uat, uct)

    field, _ = coherent_cluttergram(positions, uct, centers, normals, areas,
                                    e1, e2, k=K, gamma=GAMMA, t0=t0, dt=dt,
                                    n_samples=n_samples, c=C, pattern=pat)
    power, _ = incoherent_cluttergram(positions, uct, centers, normals, areas,
                                      t0=t0, dt=dt, n_samples=n_samples, c=C,
                                      pattern=pat)
    coh = (np.abs(field) ** 2).mean(axis=0)
    inc = power.mean(axis=0).astype(np.float64)

    # float64 per-facet |g^2 * LPA field|^2 reference
    ref = np.zeros(n_samples)
    n_tile = NCELL ** 2
    for t in range(N_REAL):
        sl = slice(t * n_tile, (t + 1) * n_tile)
        contrib, r = lpa_contributions(positions[t], centers[sl], normals[sl],
                                       areas[sl], e1[sl], e2[sl], K, GAMMA,
                                       xp=np)
        dhat = (centers[sl] - positions[t]) / r[:, None]
        g = antenna.field_gain(ant, dhat, uat[t], uct[t])
        contrib = contrib * g ** 2
        b = np.floor((2.0 * r / C - t0) / dt).astype(int)
        ok = (b >= 0) & (b < n_samples)
        np.add.at(ref, b[ok], np.abs(contrib[ok]) ** 2 / N_REAL)

    big = ref > 0.05 * ref.max()
    assert big.sum() >= 15
    ratio = coh[big] / ref[big]
    # measured (seeds fixed): mean 1.045, min/max 0.779/1.397; per-bin g**4
    # spans a factor ~445 across the occupied bins
    assert abs(ratio.mean() - 1.0) < 0.12
    assert ratio.min() > 0.55 and ratio.max() < 1.6
    # end-to-end vs the pattern-weighted incoherent kernel (g**4 on power):
    # C0 * mean(sinc^2) as in the isotropic test; measured sinc^2 .991-.994
    # (the pattern suppresses the wide-angle facets with the lower sinc^2),
    # total ratio 1.011
    mean_sinc2 = ref[big] / (C0 * inc[big])
    assert np.all((mean_sinc2 > 0.9) & (mean_sinc2 <= 1.0))
    assert abs(coh[big].sum() / (C0 * inc[big].sum()) - 1.0) < 0.12
