"""Attenuation-sweep strip (session artifact).

Crops the MID-pass panel out of each value-dir's radargrams.png (all four
share one layout, one colour scale and one twtt axis, so the crops are
directly comparable) and lays them out left to right, with the measured mid
panel first as the reference. Writes
outputs/basal_clutter/hypothesis_tests/att_sweep_strip.png.

    uv run python claude_notes/att_sweep_strip.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "outputs" / "basal_clutter" / "hypothesis_tests"
VALUES = [("baseline", 15), ("att20", 20), ("att26", 26), ("t2_att31", 31)]
COL = 1          # 0-based column of the mid pass in the 4-column figure


def panel_boxes(img):
    """(col_spans, row_spans) of the dark imshow panels in a radargram
    figure: the axes are near-black, the figure background is white."""
    g = img[..., :3].mean(-1) if img.ndim == 3 else img
    dark = g < 0.55 if g.max() <= 1.0 else g < 140

    def runs(mask, min_len):
        out, start = [], None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= min_len:
                    out.append((start, i))
                start = None
        if start is not None and len(mask) - start >= min_len:
            out.append((start, len(mask)))
        return out

    rows = runs(dark.mean(1) > 0.35, img.shape[0] // 20)
    # columns from the BOTTOM (all-simulated) row band: the top row's 4th
    # cell is the "no measured data" placeholder and is not dark
    lo, hi = rows[-1] if rows else (0, img.shape[0])
    cols = runs(dark[lo:hi].mean(0) > 0.35, img.shape[1] // 20)
    return cols, rows


def main():
    imgs, labels = [], []
    ref = plt.imread(HYP / VALUES[0][0] / "radargrams.png")
    cols, rows = panel_boxes(ref)
    if len(cols) < 2 or len(rows) < 2:
        sys.exit(f"unexpected layout: {len(cols)} cols, {len(rows)} rows")
    c0, c1 = cols[COL]
    (mr0, mr1), (sr0, sr1) = rows[0], rows[1]
    imgs.append(ref[mr0:mr1, c0:c1])
    labels.append("MEASURED (mid, 9150 m AGL)")
    for name, att in VALUES:
        p = HYP / name / "radargrams.png"
        if not p.exists():
            print(f"missing {p}")
            continue
        imgs.append(plt.imread(p)[sr0:sr1, c0:c1])
        labels.append(f"sim, att = {att} dB/km")
    n = len(imgs)
    fig, axs = plt.subplots(1, n, figsize=(3.6 * n, 4.2), squeeze=False)
    for ax, im, lab in zip(axs[0], imgs, labels):
        ax.imshow(np.asarray(im), aspect="auto", interpolation="nearest")
        ax.set_title(lab, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0, 0].set_ylabel("twtt below surface (same axis in every panel)",
                         fontsize=8)
    fig.suptitle("attenuation sweep, MID pass (9150 m AGL): measured vs "
                 "simulated bed zone, identical grey scale (dB rel own "
                 "surface peak) and twtt axis", fontsize=11)
    fig.tight_layout()
    fp = HYP / "att_sweep_strip.png"
    fig.savefig(fp, dpi=120)
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
