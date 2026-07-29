"""Integration: B26 firn-core measured-vs-simulated comparison (light).

Runs tools/run_b26_comparison.py with a tiny configuration (few traces, N=10
firn layers only, coarse 64 m facets) from the outputs/cache/ frame + DEM +
BedMachine caches (the along-track window and cross-track reaches match the
main run so the cached windows are reused; network is touched only to
populate a missing cache -- if that fails the test skips).

Gates: the artifact schema (metrics.json case/group, figures, report) and the
surface-pick alignment sanity gate (median, offset-removed, <= 5 frame bins).
Everything else is recorded, not gated (real-frame convention). The alias
warning must be silent (asserted inside the tool on every simulate() call and
recorded in the metrics).

The rough-layer runs (firn_N40_rough_*) and the H1 effective-contrast run
(firn_N40_h1eff) are far too expensive for the tiny run, so they are switched
off there and covered CONFIG-LEVEL instead: the measured C&S 2020 Fig. 11
roughness must land on every internal layer interface and nowhere else, and
the synthetic effective-contrast permittivities must reproduce the
full-resolution per-segment reflectivities.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

_spec = importlib.util.spec_from_file_location(
    "run_b26_comparison", ROOT / "tools" / "run_b26_comparison.py")
rb = importlib.util.module_from_spec(_spec)
sys.modules["run_b26_comparison"] = rb
_spec.loader.exec_module(rb)

from soundersim.opr import CACHE_DIR  # noqa: E402
from soundersim.config import RadarConfig  # noqa: E402


def test_rough_run_config():
    """Config-level cover of the rough runs (no simulation): the digitized
    C&S 2020 Fig. 11 profiles interpolate + clamp onto the layer depths, and
    firn_cfg attaches them to every INTERNAL layer interface only -- the
    air-firn surface must stay smooth (it carries the seam check and the
    profile normalization)."""
    import run_firn_investigation as rfi

    depths = rfi.equal_depths(40)
    rc = RadarConfig(dt=4.1667e-9, n_samples=64, t0=0.0, f0=195e6)
    for src, s_rng, l_rng in (("mcords", (0.02, 0.06), (2.0, 3.6)),
                              ("ar", (0.01, 0.031), (0.9, 3.0))):
        sig, cl = rb.layer_roughness(depths, src)
        assert sig.shape == cl.shape == depths.shape
        assert s_rng[0] <= sig.min() and sig.max() <= s_rng[1]
        assert l_rng[0] <= cl.min() and cl.max() <= l_rng[1]
        # clamped past the profile's 90 m end (B26 stack runs to 119.7 m)
        deep = depths > 90.0
        assert deep.any() and np.ptp(sig[deep]) == 0 and np.ptp(cl[deep]) == 0

        cfg = rb.firn_cfg(rc, 10.67, depths, (sig, cl))
        assert cfg.interfaces[0].name == "surface"
        assert cfg.interfaces[0].roughness is None       # surface stays smooth
        assert len(cfg.interfaces) == len(depths) + 1
        for i, iface in enumerate(cfg.interfaces[1:]):
            assert iface.roughness is not None
            assert iface.roughness.sigma_m == float(sig[i])
            assert iface.roughness.corr_length_m == float(cl[i])
        # ... and the smooth build is untouched
        assert all(i.roughness is None
                   for i in rb.firn_cfg(rc, 10.67, depths).interfaces)


def test_effective_contrast_run_config():
    """Config-level cover of the H1 effective-contrast run (no simulation): the
    synthetic permittivity sequence must (a) keep firn0 at its point-sampled
    value -- the surface interface, the seam check and the surface-peak
    normalization ride on it -- (b) stay physical and near the Kovacs trend,
    (c) have PLAIN Fresnel interface contrasts equal to the full-resolution
    transfer-matrix segment reflectivities, and (d) lift the 1-D 20-70 m band
    by the ~11 dB the hypothesis predicts. firn_cfg must change nothing but
    the media permittivities."""
    import run_firn_investigation as rfi

    depths = rfi.equal_depths(40)
    rc = RadarConfig(dt=4.1667e-9, n_samples=64, t0=0.0, f0=195e6)
    eps, r = rb.effective_contrast_eps(depths, rc.wavelength)
    trend = np.array([rfi.point_eps(d) for d in depths]
                     + [rfi.point_eps(float(depths[-1]) + 1.0)])

    assert eps.shape == trend.shape == (len(depths) + 1,)
    assert np.isfinite(eps).all() and (eps > 1.0).all() and (eps < 3.3).all()
    assert eps[0] == trend[0]                       # firn0 point-sampled
    assert np.abs(eps - trend).max() < 0.5          # tracks the Kovacs trend
    # segment reflectivities: median ~ -38.5 dB, all far below a Fresnel step
    rdb = 20.0 * np.log10(r)
    assert r.shape == depths.shape and (rdb < -20.0).all()
    assert -40.0 < np.median(rdb) < -37.0

    n = np.sqrt(eps)
    gam = ((n[:-1] - n[1:]) / (n[:-1] + n[1:])) ** 2
    assert np.abs(gam - r ** 2).max() < 1e-15      # contrasts reproduce |r|

    # 1-D mid-band lift over the point-sampled stack (H1's +11 dB prediction)
    nt = np.sqrt(trend)
    gam_pt = ((nt[:-1] - nt[1:]) / (nt[:-1] + nt[1:])) ** 2
    band = (depths >= 20.0) & (depths <= 70.0)
    lift = 10.0 * np.log10(gam[band].sum() / gam_pt[band].sum())
    assert 10.0 < lift < 13.0

    cfg = rb.firn_cfg(rc, 10.67, depths, eps=eps)
    ref = rb.firn_cfg(rc, 10.67, depths)
    assert [m.eps_r for m in cfg.media[1:]] == [float(x) for x in eps]
    assert [m.eps_r for m in ref.media[1:]] == [float(x) for x in trend]
    assert cfg.media[0].eps_r == 1.0 and cfg.media[1].eps_r == ref.media[1].eps_r
    # geometry / roughness identical to the smooth point-sampled run
    assert all(i.roughness is None for i in cfg.interfaces)
    assert [(i.name, getattr(i, "offset", None)) for i in cfg.interfaces] == \
        [(i.name, getattr(i, "offset", None)) for i in ref.interfaces]
    with pytest.raises(ValueError):
        rb.firn_cfg(rc, 10.67, depths, eps=eps[:-1])


def _cached(product="CSARP_standard"):
    return (CACHE_DIR /
            f"frame_{rb.SEASON}_{rb.FRAME_ID}_{product}.nc").exists()


@pytest.mark.integration
def test_b26_comparison_tiny(tmp_path):
    try:
        metrics, out = rb.run_all(
            out_root=tmp_path, n_traces=8, layer_counts=(10,), spacing=64.0,
            rough_runs=(), eff_runs=(), do_pilot=False)
    except Exception as e:
        if not _cached():
            pytest.skip(f"no local cache for {rb.FRAME_ID} and remote access "
                        f"failed: {type(e).__name__}: {e}")
        raise

    # Surface sanity gate: coherent surface-layer leading edge vs the frame's
    # Surface pick, constant offset removed, median <= 5 frame bins.
    sa = metrics["surface_pick_alignment"]
    assert sa["pass"], (f"surface leading edge misaligned: median "
                        f"{sa['value']:.2f} bins > {sa['threshold']}")
    assert abs(sa["offset_bins"]) < 40, "implausible constant twtt offset"

    # Alias rule: warning asserted silent inside the tool, recorded here.
    assert metrics["alias_free_dt"]["alias_warning_fired"] is False

    # Recorded diagnostics present and finite (x == x rejects NaN).
    for k in ("bed_alignment", "lpa_nadir_error", "bed_depth_at_site",
              "firn_seam_check", "closest_approach_m", "profile_correlation"):
        assert k in metrics and metrics[k]["value"] == metrics[k]["value"]

    # The frame passes within tens of meters of the borehole.
    assert metrics["closest_approach_m"]["value"] < 100.0

    # The field-sum seam must be tight where the strip covers all arrivals.
    assert metrics["firn_seam_check"]["value"] < 0.05

    # Artifact schema for the report builder.
    doc = json.loads((out / "metrics.json").read_text())
    assert doc["case"] == "b26_comparison"
    assert doc["group"] == "xOPR clutter"
    assert isinstance(doc["metrics"], dict) and doc["notes"]
    for fig in ("radargrams_full.png", "radargrams_nearsurface.png",
                "depth_profile.png"):
        assert (out / fig).exists()
    assert (out / "report.html").exists()
    assert (out / "run_config.json").exists()

    # --- CSARP_qlook (unfocused) comparison path ---------------------------
    cfgd = json.loads((out / "run_config.json").read_text())
    corr = metrics["profile_correlation"]
    bands = cfgd["band_levels_db_rel_surface"]
    sims = [k for k in cfgd["profile_correlation_r"]
            if not k.startswith("measured")]
    # metric keys spell "surface+bed" as "surface_bed"
    assert all(f"corr_standard_{k.replace('+', '_')}" in corr for k in sims)
    bd = metrics["band_delta_vs_measured"]
    assert bd["bands"] == ["20-70m", "80-120m"]
    assert set(bd["delta_vs_standard"]) >= set(sims)
    assert bd["value"] == bd["value"] and bd["gap_run"] in sims  # finite
    if not _cached("CSARP_qlook") and cfgd["measured_products"]["qlook"] is None:
        pytest.skip("CSARP_qlook not cached and remote access failed")
    # qlook loaded: same fast-time grid, its own profile, and both correlation
    # / band-delta families keyed against it.
    q = cfgd["measured_products"]["qlook"]
    assert q["same_fast_time_grid"] is True
    assert "measured_qlook" in bands and "20-70m" in bands["measured_qlook"]
    assert all(f"corr_qlook_{k.replace('+', '_')}" in corr for k in sims)
    assert bd["measured_qlook_db"] and set(bd["delta_vs_qlook"]) >= set(sims)
    # the two measured products score each other (product-difference rows)
    assert "corr_qlook_measured" in corr and "corr_standard_measured_qlook" in corr
    assert "measured_qlook" in bd["delta_vs_standard"]
    assert "measured" in bd["delta_vs_qlook"]
