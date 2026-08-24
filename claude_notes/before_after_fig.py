"""Stack the preserved 'before' radargrams over the fresh 'after' ones."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

name, before, after, out = sys.argv[1:5]
fig, axes = plt.subplots(2, 1, figsize=(16, 13))
for ax, path, label in ((axes[0], before, "BEFORE (smooth-mirror bed, native facets)"),
                        (axes[1], after, "AFTER (spec/diffuse + sigma 0.10, facets <= 0.7)")):
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    ax.set_title(f"{name} -- {label}", fontsize=13)
fig.tight_layout()
fig.savefig(out, dpi=95)
print("wrote", out)
