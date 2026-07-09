"""M23 report case ``sar_processing`` (group "Radar equation comparison").

Point-target focusing demo: a single scatterer under a straight, level track
(195 MHz, 500 m AGL) processed three ways -- raw (single traces), unfocused
SAR (coherent moving sum, no migration correction), and focused SAR (straight-
track time-domain backprojection through air). The azimuth -3 dB widths are
measured at the target range and compared to theory: focused resolution
lambda*r/(2*L) (rectangular aperture -> 0.886x), unfocused ~ sqrt(lambda*r/2),
raw = the full range-migration footprint. Peak coherent gain of the focused
processor equals the aperture trace count.

Validation-grade processor (plan D4-3): straight-track, surface-referenced,
air-only focusing; no motion compensation. Gate values are set from the first
run (repo convention); measured values are recorded inline.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from soundersim import processing as proc
from soundersim.compare import plots

# reuse the analytic point-target Dataset builder + width estimator
from test_processing import _point_target_ds, _minus3db_width, C, LAM

OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "verification"
GROUP = "Radar equation comparison"

H, L = 500.0, 200.0
SPACING = 0.35            # < lambda/4 (0.38 m): no Doppler aliasing


def _az_profile(ds, b0):
    amp = np.abs(ds.field.values[:, b0])
    return ds.x.values, amp, 20.0 * np.log10(np.maximum(amp, 1e-12) / amp.max())


@pytest.mark.integration
def test_sar_processing():
    ds, s, twtt, _ = _point_target_ds(H=H, aperture=L, spacing=SPACING,
                                      extra=25.0)
    dt = twtt[1] - twtt[0]
    b0 = int(round((2.0 * H / C - twtt[0]) / dt))

    # Unfocused coherent integration is only valid over the aperture where the
    # uncorrected quadratic phase stays < pi/4: L_unf = sqrt(lambda*r/2), which
    # yields azimuth resolution ~ sqrt(lambda*r/2). Focused backprojection has
    # no such limit -- it corrects the migration phase over the full aperture L.
    unf_pred = np.sqrt(LAM * H / 2.0)                # unfocused aperture / res
    focused = proc.focused_sar(ds, aperture_m=L, window="none")
    unfocused = proc.unfocused_sar(ds, aperture_m=unf_pred)

    s_raw, _, db_raw = _az_profile(ds, b0)
    s_foc, amp_foc, db_foc = _az_profile(focused, b0)
    s_unf, _, db_unf = _az_profile(unfocused, b0)

    pred = float(LAM * H / (2.0 * L))                # Rayleigh resolution (m)
    unf_pred = float(unf_pred)
    w_foc = float(_minus3db_width(db_foc, s_foc))
    w_unf = float(_minus3db_width(db_unf, s_unf))
    w_raw = float(_minus3db_width(db_raw, s_raw))

    n_ap = int(np.sum(np.abs(s - s[np.argmax(np.abs(focused.field.values[:, b0]))])
                      <= L / 2))
    peak_gain_ratio = float(amp_foc.max() / n_ap)
    ratio_pred = float(w_foc / pred)
    improvement = float(w_raw / w_foc)

    # first azimuth sidelobe of the focused response (rect ~ -13.3 dB)
    i0 = int(np.argmax(amp_foc))
    outside = np.abs(s_foc - s_foc[i0]) > 2.0 * pred
    sidelobe = float(db_foc[outside].max())

    metrics = {
        "focused_width_vs_prediction": {
            "value": ratio_pred, "target": 1.0, "threshold": 0.15,
            "pass": abs(ratio_pred - 1.0) <= 0.15,
            "measured_width_m": w_foc, "prediction_m": pred,
            "comment": "azimuth -3 dB width / (lambda*r/2L); rect aperture "
                       "theory 0.886"},
        "focused_peak_gain_ratio": {
            "value": peak_gain_ratio, "target": 1.0, "threshold": 0.1,
            "pass": abs(peak_gain_ratio - 1.0) <= 0.1,
            "n_aperture_traces": n_ap,
            "comment": "focused peak amplitude / aperture trace count"},
        "focused_first_sidelobe_db": {
            "value": sidelobe, "threshold": -10.0, "op": "<=",
            "pass": sidelobe <= -10.0,
            "comment": "rect-aperture azimuth sidelobe (theory -13.3 dB)"},
        "azimuth_resolution_improvement": {
            "value": improvement, "threshold": 10.0, "op": ">=",
            "pass": improvement >= 10.0,
            "raw_width_m": w_raw, "focused_width_m": w_foc,
            "comment": "raw (migration-footprint) width / focused width"},
        "unfocused_width_m": {
            "value": w_unf, "target": unf_pred, "threshold": 0.5 * unf_pred,
            "pass": abs(w_unf - unf_pred) <= 0.5 * unf_pred,
            "prediction_m": unf_pred,
            "comment": "unfocused -3 dB width vs sqrt(lambda*r/2) aperture "
                       "limit (order-of-magnitude gate)"},
    }

    outdir = OUTDIR / "sar_processing"
    plots.write_metrics(
        outdir / "metrics.json", "sar_processing", metrics, group=GROUP,
        notes="Point-target SAR focusing (M23): a single scatterer at 500 m "
              "range under a straight level track, 195 MHz, along-track "
              f"spacing {SPACING} m (< lambda/4). Focused straight-track "
              "backprojection (air-only, surface-referenced; validation-grade, "
              "no motion compensation) compresses the azimuth response to a "
              f"-3 dB width of {w_foc:.2f} m = {ratio_pred:.2f} x the "
              f"lambda*r/(2L) = {pred:.2f} m prediction (rect-aperture theory "
              "0.886), with peak coherent gain "
              f"{peak_gain_ratio:.2f} x the {n_ap}-trace aperture and a first "
              f"azimuth sidelobe at {sidelobe:.1f} dB. Unfocused SAR (coherent "
              f"moving sum, no migration correction) gives {w_unf:.1f} m "
              f"(~sqrt(lambda*r/2) = {unf_pred:.1f} m); the raw single-trace "
              f"footprint is {w_raw:.0f} m, a {improvement:.0f}x focusing "
              "gain.")

    _figure(outdir / "sar_focusing.png", s_raw, db_raw, s_unf, db_unf,
            s_foc, db_foc, focused, twtt, b0, pred, w_foc)

    for name, e in metrics.items():
        assert e["pass"], (name, e["value"])


def _figure(path, s_raw, db_raw, s_unf, db_unf, s_foc, db_foc,
            focused, twtt, b0, pred, w_foc):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6),
                             constrained_layout=True)

    ax = axes[0]
    ax.plot(s_raw, db_raw, color="0.6", lw=1.0, label="raw (single traces)")
    ax.plot(s_unf, db_unf, color="C1", lw=1.1, label="unfocused SAR")
    ax.plot(s_foc, db_foc, color="C0", lw=1.4, label="focused SAR")
    ax.axhline(-3.0, color="k", ls=":", lw=0.8)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-40, 3)
    ax.set_xlabel("azimuth (m)")
    ax.set_ylabel("power (dB rel. focused peak)")
    ax.set_title("Azimuth profiles at target range")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    depth = (twtt - 2.0 * 500.0 / C) * C / 2.0        # range past nadir (m)
    img = 20.0 * np.log10(
        np.maximum(np.abs(focused.field.values), 1e-12)
        / np.abs(focused.field.values).max())
    m = ax.pcolormesh(s_foc, depth, img.T, vmin=-40, vmax=0,
                      cmap="viridis", shading="auto")
    ax.set_ylim(30, -30)
    ax.set_xlim(-30, 30)
    ax.set_xlabel("azimuth (m)")
    ax.set_ylabel("range past nadir (m)")
    ax.set_title("Focused point target (dB)")
    fig.colorbar(m, ax=ax, label="dB")

    ax = axes[2]
    i0 = int(np.argmax(db_foc))
    ax.plot(s_foc - s_foc[i0], db_foc, color="C0", lw=1.5, label="focused")
    ax.axhline(-3.0, color="k", ls=":", lw=0.8, label="-3 dB")
    for x in (-pred / 2, pred / 2):
        ax.axvline(x, color="C3", ls="--", lw=1.0)
    ax.axvline(np.nan, color="C3", ls="--", lw=1.0,
               label=f"$\\lambda r/2L$ = {pred:.2f} m")
    ax.set_xlim(-4 * pred, 4 * pred)
    ax.set_ylim(-40, 3)
    ax.set_xlabel("azimuth from peak (m)")
    ax.set_ylabel("power (dB)")
    ax.set_title(f"Main-lobe zoom: -3 dB width {w_foc:.2f} m")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle("SAR post-processing: raw vs unfocused vs focused "
                 "(point target, 195 MHz, 500 m AGL)")
    fig.savefig(path, dpi=90)
    plt.close(fig)
