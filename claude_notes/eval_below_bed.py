"""Below-bed bed-arm energy fraction: the parabola-artifact fidelity metric.

For each case/pass: fraction of bed-arm power arriving LATER than the nadir
bed delay + margin, in the RAW chunk fields and the FOCUSED stack. The
investigation baseline (pilot_smoke @ posting_div 1): raw/focused
0.174/0.178 (low), 0.642/0.563 (9km), 0.702/0.682 (10km).

    uv run python claude_notes/eval_below_bed.py <case> <pass> [...]
"""
import glob
import sys

import numpy as np

MARGIN_US = 0.5
ROOT = "outputs/antarctica_getz"


def frac_below(P, twtt, t_bed):
    m = twtt[None, :] > (t_bed[:, None] + MARGIN_US * 1e-6)
    return float((P * m).sum() / P.sum())


def one(case, key):
    raws = []
    for f in sorted(glob.glob(f"{ROOT}/{case}/runs/{key}_*.npz")):
        z = np.load(f)
        P = np.abs(z["field"][..., 1]) ** 2
        raws.append((frac_below(P, z["twtt"], z["nadir_twtt"][:, 1]),
                     P.sum()))
    raw = (sum(f * w for f, w in raws) / sum(w for _, w in raws)
           if raws else float("nan"))
    foc = float("nan")
    pc = glob.glob(f"{ROOT}/{case}/proc_cache/{key}_*.npz")
    if pc:
        z = np.load(pc[0])
        foc = frac_below(np.abs(z["Fb"]) ** 2, z["twtt"], z["nadir"][:, 1])
    print(f"{case:16s} {key:10s} raw {raw:.3f}  focused {foc:.3f}  "
          f"({len(raws)} chunks)")


if __name__ == "__main__":
    args = sys.argv[1:] or ["pilot_smoke", "real_low"]
    for i in range(0, len(args), 2):
        one(args[i], args[i + 1])
