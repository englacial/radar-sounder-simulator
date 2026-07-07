"""Parity metrics against cached simc fixtures (docs/incoherent_simulation.md).

The comparison is statistical: simc regrids the DEM per trace along its own
track axes (50 m steps in ECEF vs our 50 m projected grid, i.e. different
tessellations of the same surface), so facets never match one-for-one. At the
fixture sampling (dt = 1e-8 s, 1.5 m range bins) a raw fast-time bin holds only
O(1-10) facets and per-bin power is dominated by facet-placement shot noise
(observed raw per-bin ratios scatter 0.3-3x while the facet-scale-smoothed
ratio is a clean constant). The shape metrics (peak alignment, Pearson, dB
residual) are therefore evaluated on profiles aggregated (power-summed) along
fast time to the facet scale (``agg`` bins, default posting / (c*dt/2), ~33);
raw-bin values are recorded alongside. First-return and total-power metrics are
aggregation-independent and use raw bins.

Fixture layout: cluttergram is (n_samples, n_traces), bins from t0 per the
json sidecar, trace order = nav order (same as ours).
"""

import json
from pathlib import Path

import numpy as np
from pyproj import Transformer


def load_fixture(name, fixture_dir):
    """Load a cached simc fixture (npz arrays + json sidecar as 'meta')."""
    fixture_dir = Path(fixture_dir)
    data = np.load(fixture_dir / f"{name}.npz")
    out = {k: data[k] for k in data.files}
    out["meta"] = json.loads((fixture_dir / f"{name}.json").read_text())
    return out


def _aggregate(a, agg):
    n = (a.shape[1] // agg) * agg
    return a[:, :n].reshape(a.shape[0], -1, agg).sum(axis=2)


def _rms_db_residual(ours, simc):
    """RMS dB residual over bins above -40 dB rel per-trace simc peak, after
    removing the constant dB offset. Returns (rms, offset)."""
    res = []
    for o, s in zip(ours, simc):
        mask = (s > s.max() * 1e-4) & (o > 0)
        res.append(10 * np.log10(o[mask] / s[mask]))
    res = np.concatenate(res)
    return float(np.sqrt(np.mean((res - res.mean()) ** 2))), float(res.mean())


def compare_to_simc(ds, fixture, *, agg=None, fret_tol=1):
    """Compute the five parity metrics; returns {metric: {value, threshold, pass}}.

    agg: fast-time aggregation factor for the shape metrics; None derives the
    facet scale, posting / (c*dt/2). fret_tol: first-return bin tolerance
    (documented default 1; loosening requires written justification where used).
    """
    meta = fixture["meta"]
    rc = meta["radar_config"]
    if agg is None:
        agg = int(round(meta["scene"]["params"]["posting"] / (rc["c"] * rc["dt"] / 2)))
    ours = ds.power
    if "side" in ours.dims:
        ours = ours.sum("side")
    ours = np.asarray(ours, dtype=np.float64)                # (T, n_samples)
    simc = np.asarray(fixture["cluttergram"], np.float64).T  # (T, n_samples)
    assert ours.shape == simc.shape, (ours.shape, simc.shape)
    oa, sa = _aggregate(ours, agg), _aggregate(simc, agg)

    m = {}

    # 1. Peak alignment: per-trace argmax within ±1 facet-scale bin.
    pk = int(np.abs(oa.argmax(axis=1) - sa.argmax(axis=1)).max())
    pk_raw = int(np.abs(ours.argmax(axis=1) - simc.argmax(axis=1)).max())
    m["peak_alignment"] = _entry(pk, 1, pk <= 1, agg_bins=agg, raw_bin_diff=pk_raw)

    # 2. First-return raw bin vs simc fret_bin; ground location recorded (m).
    our_bin = np.floor((ds.first_return_twtt.values - rc["t0"]) / rc["dt"]).astype(int)
    fret = int(np.abs(our_bin - fixture["fret_bin"]).max())
    to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    our_xyz = np.column_stack(to_ecef.transform(
        ds.first_return_lon.values, ds.first_return_lat.values,
        np.zeros(ours.shape[0])))
    simc_xyz = fixture["fret_xyz"]
    # Horizontal ground separation via angular distance (heights differ).
    loc_m = np.linalg.norm(
        our_xyz / np.linalg.norm(our_xyz, axis=1, keepdims=True)
        - simc_xyz / np.linalg.norm(simc_xyz, axis=1, keepdims=True),
        axis=1) * np.linalg.norm(simc_xyz, axis=1)
    m["first_return_bin"] = _entry(fret, fret_tol, fret <= fret_tol,
                                   max_location_error_m=float(loc_m.max()))

    # 3. Profile shape: min per-trace Pearson r (linear power) >= 0.99.
    r = min(np.corrcoef(o, s)[0, 1] for o, s in zip(oa, sa))
    r_raw = min(np.corrcoef(o, s)[0, 1] for o, s in zip(ours, simc))
    m["profile_pearson"] = _entry(float(r), 0.99, r >= 0.99, op=">=",
                                  raw_pearson=float(r_raw))

    # 4. Power ratio ours/simc, per-trace totals: gate on the coefficient of
    # variation (a constant scale factor between two relative-power tools is
    # acceptable); record the median absolute ratio.
    ratio = ours.sum(axis=1) / simc.sum(axis=1)
    cv = float(ratio.std() / ratio.mean())
    m["power_ratio_cv"] = _entry(cv, 0.03, cv <= 0.03,
                                 median_ratio=float(np.median(ratio)))

    # 5. RMS dB residual <= 1 dB (constant offset removed, bins above -40 dB).
    rms, off = _rms_db_residual(oa, sa)
    rms_raw, _ = _rms_db_residual(ours, simc)
    m["db_residual_rms"] = _entry(rms, 1.0, rms <= 1.0,
                                  db_offset=off, raw_rms_db=rms_raw)
    return m


def _entry(value, threshold, ok, op="<=", **extra):
    """op: how value relates to threshold when passing ("<=", ">="); entries
    with a "target" extra are tolerance-style (value within threshold of target)."""
    return {"value": value, "threshold": threshold, "pass": bool(ok), "op": op,
            **extra}
