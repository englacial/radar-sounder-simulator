"""Matplotlib (Agg) figures + metrics.json writer for the integration report.

Every integration case writes ``outputs/verification/<case>/`` with a
``metrics.json`` (see ``write_metrics``) and one or more PNG figures. The shape
figures aggregate raw fast-time bins to the facet scale (see metrics.py: raw
per-bin power is facet-placement shot noise); the difference panel is therefore
computed on facet-scale profiles.
"""

import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def aggregate(a, agg):
    """Power-sum raw fast-time bins into groups of ``agg`` (facet scale)."""
    n = (a.shape[1] // agg) * agg
    return a[:, :n].reshape(a.shape[0], -1, agg).sum(axis=2)


def _db(a, ref):
    out = np.full(a.shape, -np.inf)
    m = a > 0
    out[m] = 10.0 * np.log10(a[m] / ref)
    return out


def write_metrics(path, case, metrics, notes=None, group=None):
    """Write a metrics.json per the artifact convention; returns the path.

    group names the report section this case belongs to (e.g. "simc comparison").
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"case": case,
           "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "metrics": metrics}
    if notes:
        doc["notes"] = notes
    if group:
        doc["group"] = group
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path


def _agg_twtt_us(twtt, agg, nbins):
    """Facet-scale bin-start times (microseconds) for the aggregated profiles."""
    if twtt is None:
        return np.arange(nbins)
    return np.asarray(twtt)[: nbins * agg : agg] * 1e6


def three_panel_db(path, ours, simc, agg, *, twtt=None, title=""):
    """ours / simc / per-bin dB difference as three facet-scale dB images."""
    oa, sa = aggregate(ours, agg), aggregate(simc, agg)
    ref = max(oa.max(), sa.max())
    x = _agg_twtt_us(twtt, agg, oa.shape[1])
    ext = [float(x[0]), float(x[-1]), oa.shape[0] - 0.5, -0.5]
    diff = np.full(oa.shape, np.nan)
    mask = (oa > 0) & (sa > 0) & (sa > sa.max() * 1e-4)
    diff[mask] = 10.0 * np.log10(oa[mask] / sa[mask])
    diff -= np.nanmean(diff)  # remove the documented constant offset

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, img, ttl, kw in (
        (axes[0], _db(oa, ref), "soundersim (dB)",
         dict(cmap="viridis", vmin=-40, vmax=0)),
        (axes[1], _db(sa, ref), "simc (dB)",
         dict(cmap="viridis", vmin=-40, vmax=0)),
        (axes[2], diff, "ours - simc (dB, offset removed)",
         dict(cmap="RdBu_r", vmin=-3, vmax=3)),
    ):
        im = ax.imshow(img, aspect="auto", extent=ext, interpolation="nearest", **kw)
        ax.set_title(ttl)
        ax.set_xlabel("twtt (us)" if twtt is not None else "facet-scale bin")
        ax.set_ylabel("trace")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(title)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def profile_overlays(path, ours, simc, agg, *, twtt=None, traces=None, title=""):
    """Per-trace power profiles (dB, facet scale): ours vs simc for a few traces."""
    oa, sa = aggregate(ours, agg), aggregate(simc, agg)
    if traces is None:
        n = oa.shape[0]
        traces = sorted({0, n // 2, n - 1})
    x = _agg_twtt_us(twtt, agg, oa.shape[1])
    fig, axes = plt.subplots(len(traces), 1, figsize=(9, 2.4 * len(traces)),
                             squeeze=False, sharex=True)
    for ax, t in zip(axes[:, 0], traces):
        ref = max(oa[t].max(), sa[t].max())
        ax.plot(x, _db(oa[t][None], ref)[0], label="soundersim", lw=1.4)
        ax.plot(x, _db(sa[t][None], ref)[0], label="simc", lw=1.0, ls="--")
        ax.set_ylim(-40, 2)
        ax.set_ylabel(f"trace {t}\ndB")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right")
    axes[-1, 0].set_xlabel("twtt (us)" if twtt is not None else "facet-scale bin")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def coherent_r2_panel(path, r, lead, pred, *, slope,
                      title="Haynes coherent altitude sweep"):
    """Left: log-log coherent nadir-return power vs r with the fitted slope and
    an r^-2 reference; right: measured/predicted absolute ratio per altitude."""
    r, lead, pred = (np.asarray(a, float) for a in (r, lead, pred))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.loglog(r, lead, "o-", label=f"nadir return |field|^2\nfit slope {slope:+.3f}")
    rr = np.array([r.min(), r.max()])
    ax.loglog(rr, lead[0] * (rr / r[0]) ** -2.0, "k--", label="reference r^-2")
    ax.set_xlabel("nadir range r ~ h (m)")
    ax.set_ylabel("relative power")
    ax.legend()
    ax.grid(which="both", alpha=0.3)
    ax = axes[1]
    ax.semilogx(r, lead / pred, "o-")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylim(0.8, 1.2)
    ax.set_xlabel("nadir range r ~ h (m)")
    ax.set_ylabel("measured / closed-form")
    ax.grid(which="both", alpha=0.3)
    fig.suptitle(title)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def constants_panel(path, labels, mag_ratio, phase_deg, *, mag_tol, phase_tol,
                    title="Haynes absolute-constant checks"):
    """|field| ratio and phase error vs the closed form, with tolerance bands."""
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, y, tol, ref, ylabel in (
        (axes[0], mag_ratio, mag_tol, 1.0, "|field| / closed form"),
        (axes[1], phase_deg, phase_tol, 0.0, "phase error (deg)"),
    ):
        ax.axhspan(ref - tol, ref + tol, color="green", alpha=0.12)
        ax.axhline(ref, color="k", ls="--", lw=0.8)
        ax.plot(x, y, "o")
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def coherence_loss_panels(path, sigma_lam, panels,
                          title="Fresnel-zone power vs roughness (Haynes Fig. 5)"):
    """One panel per altitude: ensemble dots vs analytic lines per corr. length.

    panels: list of dicts with keys ``title``, ``curves`` (list of
    (label, measured, analytic) power arrays over sigma_lam), ``l0`` (analytic
    zero-correlation curve) and ``floor`` (scalar noise-floor power).
    """
    db = lambda p: 10.0 * np.log10(np.maximum(np.asarray(p, float), 1e-300))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6),
                             constrained_layout=True, squeeze=False)
    for ax, p in zip(axes[0], panels):
        for i, (lbl, meas, ana) in enumerate(p["curves"]):
            c = f"C{i}"
            ax.plot(sigma_lam, db(ana), "-", color=c, lw=1.3)
            ax.plot(sigma_lam, db(meas), "o", color=c, ms=4, label=lbl)
        ax.plot(sigma_lam, db(p["l0"]), "k--", lw=1.0, label="analytic l = 0")
        ax.axhline(db([p["floor"]])[0], color="k", ls=":", lw=1.0,
                   label="noise floor (Eq. 37)")
        ax.set_title(p["title"])
        ax.set_xlabel("sigma_h (lambda)")
        ax.set_ylabel("power (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def speckle_panels(path, amplitude, power,
                   title="Rough-surface speckle statistics"):
    """Amplitude histogram vs Rayleigh pdf; power histogram vs exponential pdf."""
    amplitude = np.asarray(amplitude, float)
    power = np.asarray(power, float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    ax = axes[0]
    s2 = (amplitude ** 2).mean() / 2.0  # Rayleigh scale^2 (moment estimate)
    a = np.linspace(0, amplitude.max() * 1.05, 300)
    ax.hist(amplitude, bins=40, density=True, alpha=0.6, label="simulated")
    ax.plot(a, a / s2 * np.exp(-a ** 2 / (2 * s2)), "k-", lw=1.5,
            label="Rayleigh pdf")
    ax.set_xlabel("|field|")
    ax.set_ylabel("pdf")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    mu = power.mean()
    p = np.linspace(0, power.max() * 1.05, 300)
    ax.hist(power, bins=40, density=True, alpha=0.6, label="simulated")
    ax.plot(p, np.exp(-p / mu) / mu, "k-", lw=1.5, label="exponential pdf")
    ax.set_yscale("log")
    ax.set_xlabel("|field|^2")
    ax.set_ylabel("pdf")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def haynes_loglog(path, r, per_facet, lead_edge, *,
                  per_facet_slope, lead_slope, title="Haynes altitude sweep"):
    """Log-log plot of per-facet (r^-4) and leading-edge (r^-3) fits vs reference."""
    r = np.asarray(r, float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, y, slope, ref, lbl in (
        (axes[0], per_facet, per_facet_slope, -4.0, "nadir facet power"),
        (axes[1], lead_edge, lead_slope, -3.0, "leading-edge power"),
    ):
        y = np.asarray(y, float)
        ax.loglog(r, y, "o-", label=f"{lbl}\nfit slope {slope:+.3f}")
        rr = np.array([r.min(), r.max()])
        yr = y[0] * (rr / r[0]) ** ref
        ax.loglog(rr, yr, "k--", label=f"reference r^{ref:+.0f}")
        ax.set_xlabel("range to nadir facet r ~ h (m)")
        ax.set_ylabel("relative power")
        ax.legend()
        ax.grid(which="both", alpha=0.3)
    fig.suptitle(title)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path
