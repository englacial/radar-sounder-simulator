"""Facet-size convergence metrics, getz real_10km DEMOGORGN arm."""
import glob
import sys

import numpy as np

sys.path.insert(0, "tools")

CASES = [("1.0", "pilot_dgns10"), ("0.7", "pilot_dgnfs07"),
         ("0.5", "pilot_dgnfs050"), ("0.35", "pilot_dgnfs035")]

print(f"{'scale':>5s} {'bed win dB':>10s} {'belowbed':>8s} {'dB-std':>7s} "
      f"{'p95-p50':>8s} {'contrast':>8s}")
for scale, case in CASES:
    hits = glob.glob(f"outputs/antarctica_getz/{case}/proc_cache/"
                     "real_10km_*_proc.npz")
    if not hits:
        print(f"{scale:>5s}  (missing)")
        continue
    z = np.load(hits[0])
    P = np.abs(z["Fs"] + z["Fb"]) ** 2
    Pb = np.abs(z["Fb"]) ** 2
    tw, tb = z["twtt"], z["nadir"][:, 1]
    band = np.abs(tw[None, :] - tb[:, None]) < 1.5e-6
    below = tw[None, :] > (tb[:, None] + 0.5e-6)
    # bed-window level rel per-trace surface peak (scalar norm like the tool)
    ts = z["nadir"][:, 0]
    pk = np.nanmedian([P[i, np.abs(tw - ts[i]) < 0.4e-6].max()
                       for i in range(P.shape[0])])
    v = P[band]
    db = 10 * np.log10(np.maximum(v, 1e-300) / np.median(v))
    print(f"{scale:>5s} {10*np.log10(v.mean()/pk):10.2f} "
          f"{(Pb*below).sum()/Pb.sum():8.3f} {db.std():7.2f} "
          f"{np.percentile(db,95):8.1f} {v.std()/v.mean():8.2f}")
