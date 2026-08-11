"""Bed-return tail metrics of tools/run_basal_clutter.py: the robust slope
fit, the bed-referenced ensemble profile, the delay -> refracted bed
incidence angle map, and the assembled metric entry (excess at bed+1/2/3 us,
the surface-vs-bed-returns fair-comparison guard, the measured noise-floor
caveat). Synthetic profiles with closed-form answers -- no network, no
kernels, no simulation."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_basal_clutter as rbc  # noqa: E402

C = 299792458.0
DT_US = 0.0202020            # the 2016 product lattice, in us


def _rel(lo=None, hi=None):
    lo = rbc.TAIL_PROF_US[0] if lo is None else lo
    hi = rbc.TAIL_PROF_US[1] if hi is None else hi
    return np.arange(round((hi - lo) / DT_US) + 1) * DT_US + lo


# ------------------------------------------------------------- slope fit

def test_tail_slope_exact_on_linear_profile():
    """Theil-Sen recovers an exactly linear dB tail, and only the fit window
    is used (data outside it is ignored)."""
    rel = _rel()
    for slope in (-8.0, -2.5, 0.0, 1.5):
        db = -40.0 + slope * rel
        assert rbc.tail_slope_db_per_us(rel, db) == pytest.approx(slope,
                                                                  abs=1e-9)
    # a cliff entirely outside [0.5, 3.5] us must not move the slope
    db = -40.0 - 3.0 * rel
    db[rel > rbc.TAIL_FIT_US[1] + 1e-9] -= 60.0
    db[rel < rbc.TAIL_FIT_US[0] - 1e-9] += 60.0
    assert rbc.tail_slope_db_per_us(rel, db) == pytest.approx(-3.0, abs=1e-9)


def test_tail_slope_is_robust_to_bright_arcs():
    """A few bright samples (an off-nadir arc crossing the window) shift a
    least-squares slope but not the Theil-Sen one."""
    rel = _rel()
    truth = -4.0
    db = -40.0 + truth * rel
    m = (rel >= rbc.TAIL_FIT_US[0]) & (rel <= rbc.TAIL_FIT_US[1])
    hit = np.where(m)[0][[5, 6, 7, 8]]
    db[hit] += 35.0
    ls = np.polyfit(rel[m], db[m], 1)[0]
    ts = rbc.tail_slope_db_per_us(rel, db)
    assert ts == pytest.approx(truth, abs=0.2)
    assert abs(ls - truth) > 1.0          # least squares is pulled off


def test_at_us_interpolates_the_profile():
    rel = _rel()
    db = -30.0 - 5.0 * rel
    for t in rbc.TAIL_EXCESS_US:
        assert rbc._at_us(rel, db, t) == pytest.approx(-30.0 - 5.0 * t,
                                                       abs=1e-9)


# -------------------------------------------------- bed-referenced profile

def test_bed_referenced_profile_recovers_a_known_decay():
    """rel_mean_profile referenced to each trace's OWN bed reference (the
    sim bed-layer nadir twtt) aligns traces with different bed delays: an
    identical -6 dB/us tail planted at scattered bed times comes back as a
    -6 dB/us ensemble profile."""
    dt = DT_US * 1e-6
    n, T = 700, 24
    twtt = 5e-6 + np.arange(n) * dt
    rng = np.random.default_rng(0)
    t_b = 8e-6 + rng.integers(0, 120, T) * dt      # bed bin varies per trace
    slope = -6.0
    P = np.full((n,), 1e-12) * np.ones((T, n))
    for t in range(T):
        k = np.arange(n) - int(round((t_b[t] - twtt[0]) / dt))
        tail = 10.0 ** ((slope * k * DT_US) / 10.0)
        P[t] = np.where(k >= 0, tail, 1e-12)
    rel, db = rbc.rel_mean_profile(P, twtt, dt, t_b, np.ones(T),
                                   *rbc.TAIL_PROF_US)
    assert rbc.tail_slope_db_per_us(rel, db) == pytest.approx(slope, abs=0.05)
    assert rbc._at_us(rel, db, 0.0) == pytest.approx(0.0, abs=0.05)


# ------------------------------------------------------------- angle map

def test_bed_incidence_angle_matches_the_snell_geometry():
    """The delay -> in-ice incidence angle map is consistent with bed_reach:
    at the same extra delay both describe the same refracted ray."""
    h, d, n_ice = 9150.0, 700.0, float(np.sqrt(3.17))
    for dt_us in (0.5, 1.0, 2.0, 3.0, 3.5):
        phi = np.radians(float(rbc.bed_incidence_deg(h, d, n_ice,
                                                     dt_us * 1e-6)))
        theta = np.arcsin(n_ice * np.sin(phi))          # Snell, back to air
        y = h * np.tan(theta) + d * np.tan(phi)
        assert y == pytest.approx(rbc.bed_reach(h, d, n_ice, dt_us * 1e-6),
                                  rel=2e-3)
        t = 2.0 * (h / np.cos(theta) + n_ice * d / np.cos(phi)) / C
        assert (t - 2.0 * (h + n_ice * d) / C) * 1e6 == pytest.approx(
            dt_us, rel=2e-3)
    # monotone in delay, and a low platform reaches steeper angles sooner
    a = [float(rbc.bed_incidence_deg(h, d, n_ice, t * 1e-6))
         for t in (0.5, 1.0, 2.0, 3.0)]
    assert np.all(np.diff(a) > 0)
    assert float(rbc.bed_incidence_deg(442.0, d, n_ice, 1e-6)) > a[1]


# ------------------------------------------------- assembled metric entry

def _fake(rel, tot_slope, tot0, guard_db, meas=None, floor_db=-95.0):
    """One analysis dict: linear sim total, sim bed returns == total,
    sim surface returns ``guard_db`` below it, optional measured curve."""
    tot = tot0 + tot_slope * rel
    bp = {"sim_total": (rel, tot), "sim_bed": (rel, tot),
          "sim_surface": (rel, tot - guard_db)}
    cov = {"sim": 0.99, "measured": None}
    if meas is not None:
        m_slope, m0 = meas
        bp["measured"] = (rel, m0 + m_slope * rel)
        cov["measured"] = 1.0
    return {"bed_profs": bp, "tail_cov": cov, "floor_db": floor_db,
            "floor_doc": {"valid": floor_db is not None}}


def test_bed_tail_entry_slopes_excess_and_guard():
    rel = _rel()
    p = {"h_med": 10684.0, "thick_med": 683.0}
    a = _fake(rel, -2.0, -45.0, 14.0, meas=(-9.0, -47.0))
    ab = _fake(rel, -3.0, -44.0, 25.0)
    e = rbc.bed_tail_entry("high", p, a, [("picked_bed", a),
                                          ("demogorgn", ab)])
    sl = e["bed_return_tail_slope_db_per_us"]
    assert sl["measured"] == pytest.approx(-9.0, abs=1e-9)
    assert sl["picked_bed"] == pytest.approx(-2.0, abs=1e-9)
    assert sl["demogorgn"] == pytest.approx(-3.0, abs=1e-9)
    # excess = sim - measured at each delay, closed form
    for t, tag in zip(rbc.TAIL_EXCESS_US, ("+1us", "+2us", "+3us")):
        want = (-45.0 - 2.0 * t) - (-47.0 - 9.0 * t)
        assert e["bed_return_tail_excess_db"]["picked_bed"][tag] == \
            pytest.approx(want, abs=0.01)
    assert e["value"] == e["bed_return_tail_excess_db"]["picked_bed"]["+2us"]
    # guard: 14 dB and 25 dB of bed-over-surface-return margin both pass
    for slug, want in (("picked_bed", 14.0), ("demogorgn", 25.0)):
        g = e["sim"][slug]["guard"]
        assert g["pass"] and g["min_bed_minus_surface_returns_db"] == \
            pytest.approx(want, abs=0.01)
    # angle map is monotone and recorded for every excess delay
    ang = e["bed_return_angle_map_deg"]
    assert [ang["+1us"], ang["+2us"], ang["+3us"]] == sorted(
        [ang["+1us"], ang["+2us"], ang["+3us"]])
    # angular slope has the sign of the delay slope (angle grows with delay),
    # and is recorded for the measured curve too (item: decay vs angle)
    assert e["sim"]["picked_bed"]["slope_db_per_deg"] < 0.0
    sd = e["bed_return_tail_slope_db_per_deg"]
    assert sd["measured"] < sd["picked_bed"] < 0.0   # measured decays faster
    assert set(sd) == set(sl)


def test_bed_tail_entry_guard_fails_when_surface_returns_crowd_the_bed():
    """A window where the sim surface returns are within TAIL_GUARD_DB of the
    sim bed returns is FLAGGED: the total-field tail there is surface
    clutter, and the bed-returns-only slope is the fallback."""
    rel = _rel()
    p = {"h_med": 30000.0, "thick_med": 683.0}
    a = _fake(rel, -1.0, -42.0, 4.0)
    e = rbc.bed_tail_entry("syn30km", p, a, [("picked_bed", a)])
    g = e["sim"]["picked_bed"]["guard"]
    assert not g["pass"]
    assert g["min_bed_minus_surface_returns_db"] == pytest.approx(4.0,
                                                                  abs=0.01)
    assert rbc.TAIL_FIT_US[0] <= g["at_us"] <= rbc.TAIL_FIT_US[1]
    # no measured data -> no excess, and the headline value is the slope
    assert e["measured"] is None and e["bed_return_tail_excess_db"] is None
    assert e["value"] == pytest.approx(-1.0, abs=1e-9)
    assert e["sim"]["picked_bed"]["bed_returns_slope_db_per_us"] == \
        pytest.approx(-1.0, abs=1e-9)


def test_measured_noise_floor_caveat():
    """The caveat metric compares the measured tail at bed+3 us with the
    pass's own floor estimate and flags a floor-limited tail."""
    rel = _rel()
    tag = f"tail_minus_floor_at_+{rbc.TAIL_EXCESS_US[-1]:g}us_db"
    P = {"h_med": 9150.0, "thick_med": 683.0}
    # clear of the floor: -47 - 9*3 = -74 dB vs a -95 dB floor
    a = _fake(rel, -2.0, -45.0, 14.0, meas=(-9.0, -47.0), floor_db=-95.0)
    nf = rbc.meas_tail_stats(P, a)["noise_floor_caveat"]
    assert nf[tag] == pytest.approx(21.0, abs=0.01)
    assert not nf["floor_limited"]
    # floor 1 dB below the tail -> floor-limited
    a = _fake(rel, -2.0, -45.0, 14.0, meas=(-9.0, -47.0), floor_db=-75.0)
    nf = rbc.meas_tail_stats(P, a)["noise_floor_caveat"]
    assert nf[tag] == pytest.approx(1.0, abs=0.01) and nf["floor_limited"]


def test_fig_bed_tail_renders_measured_and_prediction_panels(tmp_path):
    """The figure assembles from the same bed-referenced profiles + the
    assembled metric entries, including a synthetic pass with no measured
    curve and the ablation bed-source rows."""
    import matplotlib
    matplotlib.use("Agg")
    rel = _rel()
    keys = ["low", "syn30km"]
    preps = {"low": {"h_med": 442.0, "thick_med": 683.0},
             "syn30km": {"h_med": 29858.0, "thick_med": 683.0}}
    analyses = {"low": _fake(rel, -2.0, -45.0, 14.0, meas=(-9.0, -47.0)),
                "syn30km": _fake(rel, -1.0, -42.0, 4.0)}
    ab = {"low": _fake(rel, -3.0, -44.0, 20.0),
          "syn30km": _fake(rel, -2.0, -41.0, 8.0)}
    ablation = [({}, ab, "DEMOGORGN bed, seed 0")]
    metrics = {f"bed_return_tail_{k}": rbc.bed_tail_entry(
        k, preps[k], analyses[k],
        [("picked_bed", analyses[k]), ("demogorgn", ab[k])]) for k in keys}
    fp = rbc.fig_bed_tail(tmp_path, preps, analyses, metrics, keys=keys,
                          ablation=ablation)
    assert fp.exists() and fp.stat().st_size > 5000


def test_tail_metric_terminology():
    """User-mandated terminology for the NEW metric: 'surface returns' /
    'bed returns', never 'surface-borne' / 'bed-borne'."""
    rel = _rel()
    e = rbc.bed_tail_entry(
        "low", {"h_med": 442.0, "thick_med": 683.0},
        _fake(rel, -2.0, -45.0, 14.0, meas=(-9.0, -47.0)),
        [("picked_bed", _fake(rel, -2.0, -45.0, 14.0))])

    def _text(o):
        if isinstance(o, dict):
            return " ".join(_text(v) for v in list(o) + list(o.values()))
        return o if isinstance(o, str) else ""

    blob = _text(e)
    assert "-borne" not in blob
    assert "bed returns" in blob and "surface returns" in blob


# ------------------------------------------------- per-pass floor window
def _tw(t0_us, t_end_us, dt_ns=33.3333):
    n = int(round((t_end_us - t0_us) * 1e3 / dt_ns)) + 1
    return (t0_us + np.arange(n) * dt_ns * 1e-3) * 1e-6


def test_floor_window_default_when_the_tail_is_long():
    """A record with a generous post-bed tail keeps the tool's established
    end -[12, 8] us window bit-identically (every 2016 DC-8 pass, and the
    Greenland LOW pass)."""
    tw = _tw(0.0, 68.0)
    bot = np.full(200, 14.2e-6)                 # 2016 low pass: bed at 14.2 us
    lo, hi, doc = rbc.floor_window(tw, bot)
    assert lo == pytest.approx(tw[-1] - 12e-6)
    assert hi == pytest.approx(tw[-1] - 8e-6)
    assert doc["slid_off_the_bed"] is False
    assert doc["valid"] is True


def test_floor_window_slides_off_a_bed_that_reaches_into_it():
    """The Greenland high pass: record ends 55.4 us, deepest bed 47.53 us, so
    end -[12, 8] us = [43.4, 47.4] us IS the bed. The window must slide."""
    tw = _tw(0.0, 55.4)
    bot = np.linspace(42.33e-6, 47.53e-6, 300)
    lo, hi, doc = rbc.floor_window(tw, bot)
    assert doc["slid_off_the_bed"] is True
    assert lo > 47.53e-6                        # clear of the deepest bed
    assert lo == pytest.approx(47.53e-6 + rbc.FLOOR_BED_GUARD_US * 1e-6)
    assert hi == pytest.approx(tw[-1] - rbc.FLOOR_ROLLOFF_US * 1e-6)
    assert doc["margin_past_deepest_bed_us"] == pytest.approx(
        rbc.FLOOR_BED_GUARD_US)
    assert doc["valid"] is True


def test_floor_window_reports_invalid_when_nothing_is_left():
    """No trustworthy floor exists when the bed runs to the record end; the
    caller must report unknown rather than a contaminated number."""
    tw = _tw(0.0, 55.4)
    bot = np.full(50, 53.0e-6)
    _, _, doc = rbc.floor_window(tw, bot)
    assert doc["valid"] is False
    assert doc["width_us"] < rbc.FLOOR_MIN_WIDTH_US


def test_floor_window_never_lands_on_the_bed():
    """Property: across a sweep of bed depths the window always starts after
    the deepest pick whenever it reports itself valid."""
    tw = _tw(0.0, 55.4)
    for b_us in np.arange(20.0, 52.0, 0.5):
        lo, hi, doc = rbc.floor_window(tw, np.full(10, b_us * 1e-6))
        if doc["valid"]:
            assert lo >= b_us * 1e-6, b_us
            assert hi > lo


def test_meas_tail_stats_tolerates_an_invalid_floor():
    """floor_db None (no trustworthy window) must not crash the tail entry;
    the caveat reports nulls instead of a fabricated margin."""
    rel = _rel()
    a = _fake(rel, -2.0, -45.0, 14.0, meas=(-9.0, -47.0), floor_db=None)
    st = rbc.meas_tail_stats({"h_med": 2483.0, "thick_med": 2476.0}, a)
    cav = st["noise_floor_caveat"]
    assert cav["floor_rel_surf_db"] is None
    assert cav["floor_limited"] is None
    assert st["slope_db_per_us"] == pytest.approx(-9.0, abs=1e-9)
