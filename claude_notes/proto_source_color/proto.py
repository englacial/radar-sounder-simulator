"""Prototype: colour-coded radargrams by energy source (surface / bed /
internal layers). Reads the std_benchmark proc caches (focused per-interface
complex stacks); no simulation. Three display options, one figure each.
Palette: Okabe-Ito (colour-blind safe)."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import ndimage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/prototypes/source_color"
OUT.mkdir(parents=True, exist_ok=True)

COL = {"surface": "#56B4E9", "bed": "#E69F00", "layers": "#009E73"}
CASES = [  # (label, proc_cache npz, y range us, db floor)
    ("geikie low (465 m AGL)",
     "outputs/greenland_geikie/std_benchmark/proc_cache/low_full_pbed_rssnr_proc.npz",
     (-1, 34), -120.0),
    ("westcoast p3_2016 (483 m AGL)",
     "outputs/greenland_westcoast/std_benchmark/proc_cache/p3_2016_full_pbed_rssnr_proc.npz",
     (-1, 18), -110.0),
]
CASE_SETS = {
    "greenland": (CASES, OUT, 4),
    "getz_haps20": ([("getz haps_20km (20 km AGL, 60/15 MHz)",
                      "outputs/antarctica_getz/full_line/proc_cache/haps_20km_full_line_dgn_rssnr_proc_hyb.npz",
                      (-1, 13.5), -90.0)], OUT / "getz_haps20", 8),
}
CASES, OUT, DECIM = CASE_SETS[sys.argv[1] if len(sys.argv) > 1 else "greenland"]
OUT.mkdir(parents=True, exist_ok=True)
N_LOOKS = 3


def load(npz, y_us):
    d = np.load(npz)
    look = lambda P: ndimage.uniform_filter1d(P, N_LOOKS, axis=0, mode="nearest")
    Fs, Fb = np.nan_to_num(d["Fs"]), np.nan_to_num(d["Fb"])  # NaN above bed
    P = {"surface": look(np.abs(Fs) ** 2), "bed": look(np.abs(Fb) ** 2),
         "total": look(np.abs(Fs + Fb) ** 2)}
    rel = (d["twtt"] - np.nanmedian(d["surf_sim"])) * 1e6
    m = (rel >= y_us[0]) & (rel <= y_us[1])
    # ref: median per-trace peak of total near the surface
    ref = np.nanmedian(P["total"].max(axis=1))
    db = {k: (10 * np.log10(np.maximum(v[::DECIM, m], 1e-300) / ref)).T
          for k, v in P.items()}
    s_km = d["s_sim"][::DECIM] / 1e3
    ext = [s_km[0], s_km[-1], rel[m][-1], rel[m][0]]
    return db, ext


GAMMA = 0.5  # brightness stretch (applied identically to the grey reference)


def norm(db, lo, hi=0.0):
    return np.clip((db - lo) / (hi - lo), 0, 1) ** GAMMA


def grey(ax, db, lo, ext, **kw):
    return ax.imshow(norm(db, lo), cmap="gray", vmin=0, vmax=1, extent=ext, **kw)


# ---------------------------------------------------------------- option A
def option_a(db, lo):
    """Additive tint composite: sum of per-source tints, each scaled by its
    own normalised dB."""
    rgb = np.zeros(db["total"].shape + (3,))
    for k in ("surface", "bed"):
        rgb += norm(db[k], lo)[..., None] * np.array(to_rgb(COL[k]))
    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------- option B
def option_b(db, lo, thresh_db=None):
    """Luminance = total power (grey ramp); hue = dominant source.
    thresh_db=None: saturation = dominance (|fraction - 0.5| * 2), evenly
    mixed pixels grey. thresh_db=X: full colour only where one source beats
    the other by > X dB, grey otherwise."""
    L = norm(db["total"], lo)
    ps, pb = 10 ** (db["surface"] / 10), 10 ** (db["bed"] / 10)
    fb = pb / (ps + pb + 1e-300)               # bed fraction of power
    if thresh_db is None:
        sat = np.abs(fb - 0.5) * 2
    else:
        sat = (np.abs(db["bed"] - db["surface"]) > thresh_db).astype(float)
    hue = np.where(fb[..., None] > 0.5, np.array(to_rgb(COL["bed"])),
                   np.array(to_rgb(COL["surface"])))
    grey = np.ones(3)
    tint = (1 - sat[..., None]) * grey + sat[..., None] * hue
    return np.clip(L[..., None] * tint, 0, 1)


# ---------------------------------------------------------------- option C
def ramp(c):
    return LinearSegmentedColormap.from_list("r", ["black", c, "white"])


def legend(ax, extra=()):
    h = [Patch(color=COL[k], label=k) for k in ("surface", "bed")]
    h.append(Patch(color=COL["layers"], label="internal layers (reserved)"))
    ax.legend(handles=h + list(extra), loc="lower left", fontsize=7,
              framealpha=0.85)


def main():
    data = [(lab, *load(ROOT / f, y), lo) for lab, f, y, lo in CASES]
    kw = dict(aspect="auto", interpolation="nearest")

    # A
    fig, axs = plt.subplots(len(data), 2, figsize=(13, 4.5 * len(data)), squeeze=False)
    for r, (lab, db, ext, lo) in enumerate(data):
        grey(axs[r, 0], db["total"], lo, ext, **kw)
        axs[r, 0].set_title(f"{lab}: total (grey reference)", fontsize=10)
        axs[r, 1].imshow(option_a(db, lo), extent=ext, **kw)
        axs[r, 1].set_title("A: additive tint composite (surface+bed tints summed)", fontsize=10)
        legend(axs[r, 1])
        axs[r, 0].set_ylabel("twtt below surface (us)")
    for ax in axs[-1]: ax.set_xlabel("along-track s (km)")
    fig.suptitle(f"Option A (brightness: dB rel surface peak, gamma {GAMMA} stretch, all panels) -- additive tints: each source drawn in its own colour at its own dB, "
                 "summed. Overlap -> mixed hue", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "option_A_additive.png", dpi=130); plt.close(fig)

    # B
    fig, axs = plt.subplots(len(data), 2, figsize=(13, 4.5 * len(data)), squeeze=False)
    for r, (lab, db, ext, lo) in enumerate(data):
        grey(axs[r, 0], db["total"], lo, ext, **kw)
        axs[r, 0].set_title(f"{lab}: total (grey reference)", fontsize=10)
        axs[r, 1].imshow(option_b(db, lo), extent=ext, **kw)
        axs[r, 1].set_title("B: brightness = total power, hue = dominant source, "
                            "saturation = dominance", fontsize=10)
        legend(axs[r, 1], [Patch(color="0.7", label="mixed (~50/50) -> grey")])
        axs[r, 0].set_ylabel("twtt below surface (us)")
    for ax in axs[-1]: ax.set_xlabel("along-track s (km)")
    fig.suptitle(f"Option B (brightness: dB rel surface peak, gamma {GAMMA} stretch, all panels) -- one radargram, dominant-source hue: the grey image is unchanged; "
                 "colour only says WHICH interface supplies the energy", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "option_B_dominant_hue.png", dpi=130); plt.close(fig)

    # B threshold variants
    variants = [(None, "linear saturation (current)"), (3.0, "colour only if one source wins by > 3 dB"),
                (6.0, "colour only if one source wins by > 6 dB")]
    fig, axs = plt.subplots(len(data), 3, figsize=(19.5, 4.5 * len(data)), squeeze=False)
    for r, (lab, db, ext, lo) in enumerate(data):
        for c, (th, name) in enumerate(variants):
            axs[r, c].imshow(option_b(db, lo, th), extent=ext, **kw)
            axs[r, c].set_title(f"{lab}\nB: {name}", fontsize=10)
            legend(axs[r, c], [Patch(color="0.7", label="neither dominates -> grey")])
        axs[r, 0].set_ylabel("twtt below surface (us)")
    for ax in axs[-1]: ax.set_xlabel("along-track s (km)")
    fig.suptitle(f"Option B saturation law variants (brightness: dB rel surface peak, gamma {GAMMA} stretch)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "option_B_threshold_variants.png", dpi=130); plt.close(fig)

    # C
    fig, axs = plt.subplots(len(data), 4, figsize=(22, 4.5 * len(data)), squeeze=False)
    for r, (lab, db, ext, lo) in enumerate(data):
        grey(axs[r, 0], db["total"], lo, ext, **kw)
        axs[r, 0].set_title(f"{lab}: total", fontsize=10)
        for c, k in enumerate(("surface", "bed")):
            im = axs[r, c + 1].imshow(norm(db[k], lo), cmap=ramp(COL[k]), vmin=0, vmax=1, extent=ext, **kw)
            axs[r, c + 1].set_title(f"{k} returns only", fontsize=10, color=COL[k])
            cb = plt.colorbar(im, ax=axs[r, c + 1], fraction=0.03, pad=0.01, label="dB rel surface peak")
            ticks = np.arange(lo, 1, 20.0); cb.set_ticks(norm(ticks, lo)); cb.set_ticklabels([f"{t:.0f}" for t in ticks])
        axs[r, 3].imshow(np.zeros_like(db["total"]), cmap=ramp(COL["layers"]), vmin=0, vmax=1, extent=ext, **kw)
        axs[r, 3].set_title("internal layers (no field yet -- reserved)", fontsize=10, color=COL["layers"])
        axs[r, 0].set_ylabel("twtt below surface (us)")
    for ax in axs[-1]: ax.set_xlabel("along-track s (km)")
    fig.suptitle(f"Option C (brightness: dB rel surface peak, gamma {GAMMA} stretch, all panels) -- tinted small multiples: grey total + one single-hue ramp panel per source "
                 "(same dB range everywhere)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "option_C_small_multiples.png", dpi=110); plt.close(fig)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
