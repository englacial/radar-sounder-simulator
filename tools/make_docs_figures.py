"""Generate the vector diagrams for docs/refraction.md.

Every drawn ray path is computed from an actual Snell solve (not hand-drawn):
the two-point crossings come from ``soundersim.refraction.snell_crossing`` and
the true multi-interface stationary path from a constant-ray-parameter solve
(exact for flat parallel layers). Index contrasts are exaggerated (n = 1, 1.7,
2.4, 3.2) so the sequential-chain deviation is visible on the page; the caption
says so. Run:  uv run python tools/make_docs_figures.py

Outputs docs/figures/refraction_{single,chain,joint}.svg (small, self-plotted).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from soundersim.refraction import snell_crossing

# --- shared style -----------------------------------------------------------
C_TRUE = "#2a9d8f"    # true / joint stationary path
C_CHAIN = "#e76f51"   # sequential-chain path
C_IFACE = "#4a4a4a"   # interfaces
C_NORMAL = "#b0b0b0"  # surface normals (thin dashed)
C_ENDPT = "#264653"   # platform / target markers
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")

plt.rcParams.update({
    "font.size": 11,
    "svg.fonttype": "none",   # keep text as text (tiny SVG, selectable)
    "axes.linewidth": 0.0,
})


def _snell2d(px, pz, qx, qz, z_iface, n1, n2):
    """Two-point crossing in the x-z plane against a horizontal interface."""
    r = snell_crossing(
        np.array([px, 0.0, pz]), np.array([qx, 0.0, qz]),
        np.array([0.0, 0.0, z_iface]), np.array([0.0, 0.0, 1.0]),
        float(n1), float(n2), xp=np,
    )
    x = np.asarray(r.x)
    return x[0], x[2]


def true_flat_path(px, pz, qx, qz, z_ifaces, n):
    """True stationary path through flat parallel layers (constant ray param).

    ``z_ifaces`` top-down interface heights, ``n`` media indices (len = len
    (z_ifaces)+1). sin(theta_i) = p/n_i is the same across layers (Snell); the
    ray parameter p is fixed by closing the total horizontal offset.
    """
    z_nodes = [pz] + list(z_ifaces) + [qz]          # segment endpoints in z
    dz = np.abs(np.diff(z_nodes))                    # per-segment thickness
    X = qx - px                                      # total horizontal offset
    n = np.asarray(n, float)

    def offset(p_ray):
        s = p_ray / n
        return np.sum(dz * s / np.sqrt(1.0 - s * s))

    lo, hi = 0.0, np.min(n) * (1.0 - 1e-9)           # p < min(n): no TIR
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if offset(mid) < X:
            lo = mid
        else:
            hi = mid
    p_ray = 0.5 * (lo + hi)
    s = p_ray / n
    dx = dz * s / np.sqrt(1.0 - s * s)               # horizontal step per seg
    xs = px + np.concatenate([[0.0], np.cumsum(dx)])
    zs = np.array(z_nodes)
    return xs, zs                                     # includes p and q


def chained_flat_path(px, pz, qx, qz, z_ifaces, n):
    """Sequential-chain path: N two-point solves, each toward the target q
    treating the whole stack below its interface as one medium n_{i+1}."""
    cur = (px, pz)
    xs, zs = [px], [pz]
    for zi, (n_above, n_below) in zip(z_ifaces, zip(n[:-1], n[1:])):
        cx, cz = _snell2d(cur[0], cur[1], qx, qz, zi, n_above, n_below)
        xs.append(cx)
        zs.append(cz)
        cur = (cx, cz)
    xs.append(qx)
    zs.append(qz)
    return np.array(xs), np.array(zs)


def _finish(ax, xlim, zlim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal")
    ax.axis("off")


def _endpoints(ax, px, pz, qx, qz):
    ax.plot([px], [pz], "^", color=C_ENDPT, ms=11, zorder=6)
    ax.plot([qx], [qz], "o", color=C_ENDPT, ms=9, zorder=6)
    ax.annotate("p", (px, pz), textcoords="offset points", xytext=(-4, 8),
                ha="right", color=C_ENDPT, fontstyle="italic")
    ax.annotate("q", (qx, qz), textcoords="offset points", xytext=(8, -2),
                ha="left", color=C_ENDPT, fontstyle="italic")


def _ray(ax, xs, zs, color, **kw):
    ax.annotate("", xy=(xs[-1], zs[-1]), xytext=(xs[-2], zs[-2]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                shrinkA=0, shrinkB=6, **kw), zorder=5)
    ax.plot(xs[:-1], zs[:-1], "-", color=color, lw=2.2, zorder=5,
            solid_capstyle="round")


# --- figure 1: two-point solve, single interface -----------------------------
def fig_single():
    px, pz, qx, qz = 0.0, 6.0, 6.5, -5.0
    n1, n2 = 1.0, 2.2
    cx, cz = _snell2d(px, pz, qx, qz, 0.0, n1, n2)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    xl = (-1.5, 8.0)
    # true (slightly irregular) surface + the local plane the solve uses
    xx = np.linspace(*xl, 200)
    ax.plot(xx, 0.35 * np.sin(0.9 * xx + 0.4) - 0.05 * xx, color=C_IFACE,
            lw=1.0, alpha=0.5, zorder=1)
    ax.plot(xl, [0, 0], "--", color=C_IFACE, lw=1.4, zorder=2)
    ax.annotate("local plane", (xl[1], 0), color=C_IFACE, fontsize=9,
                xytext=(-2, 5), textcoords="offset points", ha="right")
    # normal at the crossing
    ax.plot([cx, cx], [cz + 0.0, 3.2], "--", color=C_NORMAL, lw=1.0, zorder=1)
    _ray(ax, [px, cx], [pz, cz], C_TRUE)
    _ray(ax, [cx, qx], [cz, qz], C_TRUE)
    ax.plot([cx], [cz], "o", color=C_TRUE, ms=7, zorder=6)
    ax.annotate("x", (cx, cz), textcoords="offset points", xytext=(6, 6),
                color=C_TRUE, fontstyle="italic")
    _endpoints(ax, px, pz, qx, qz)
    ax.annotate(r"$n_1$", (-1.2, 3.0), color=C_ENDPT)
    ax.annotate(r"$n_2$", (-1.2, -3.0), color=C_ENDPT)
    ax.annotate(r"$\theta_1$", (cx - 0.15, 1.4), ha="right", color=C_TRUE,
                fontsize=10)
    ax.annotate(r"$\theta_2$", (cx + 0.25, -1.6), ha="left", color=C_TRUE,
                fontsize=10)
    _finish(ax, xl, (-5.6, 6.6))
    fig.tight_layout(pad=0.2)
    out = os.path.join(FIGDIR, "refraction_single.svg")
    fig.savefig(out)
    plt.close(fig)
    return out


# --- shared 3-interface flat stack for figs 2 and 3 --------------------------
STACK = dict(
    px=0.0, pz=10.0, qx=15.0, qz=-9.0,
    z_ifaces=[0.0, -3.0, -6.0],
    n=[1.0, 2.0, 3.0, 4.0],
)


def _draw_stack(ax, xl):
    for zi in STACK["z_ifaces"]:
        ax.plot(xl, [zi, zi], "-", color=C_IFACE, lw=1.3, zorder=2)
    labels = [r"$n_1$", r"$n_2$", r"$n_3$", r"$n_4$"]
    z_nodes = [STACK["pz"]] + STACK["z_ifaces"] + [STACK["qz"]]
    for lab, z_hi, z_lo in zip(labels, z_nodes[:-1], z_nodes[1:]):
        ax.annotate(lab, (xl[0] + 0.2, 0.5 * (z_hi + z_lo)), color=C_ENDPT,
                    fontsize=10, va="center")


def fig_chain():
    s = STACK
    xt, zt = true_flat_path(**s)
    xc, zc = chained_flat_path(**s)

    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    xl = (-1.4, 16.5)
    _draw_stack(ax, xl)

    # straight-remainder ASSUMPTION made at the first solve: x1 -> q straight.
    ax.plot([xc[1], s["qx"]], [zc[1], s["qz"]], "--", color=C_CHAIN, lw=1.5,
            alpha=0.9, zorder=3)
    ax.annotate("assumed straight\nremainder (step 1)",
                (0.55 * xc[1] + 0.45 * s["qx"], 0.55 * zc[1] + 0.45 * s["qz"]),
                color=C_CHAIN, fontsize=8.5, ha="left", va="center",
                xytext=(20, 10), textcoords="offset points")

    # true stationary path (one color) then chained path (another)
    for i in range(len(xt) - 1):
        _ray(ax, xt[i:i + 2], zt[i:i + 2], C_TRUE)
    for i in range(len(xc) - 1):
        _ray(ax, xc[i:i + 2], zc[i:i + 2], C_CHAIN)
    ax.plot(xc[1:-1], zc[1:-1], "o", color=C_CHAIN, ms=6, zorder=6)
    ax.plot(xt[1:-1], zt[1:-1], "o", color=C_TRUE, ms=6, zorder=6,
            markerfacecolor="white", markeredgewidth=1.6)
    for k in range(1, len(xt) - 1):
        ax.annotate(fr"$x_{k}$", (xc[k], zc[k]), color=C_CHAIN, fontsize=9.5,
                    ha="right", xytext=(-8, -9), textcoords="offset points")

    _endpoints(ax, s["px"], s["pz"], s["qx"], s["qz"])
    # legend proxies
    ax.plot([], [], "-", color=C_TRUE, lw=2.2, label="true stationary path")
    ax.plot([], [], "-", color=C_CHAIN, lw=2.2, label="sequential chain")
    ax.legend(loc="center left", frameon=False, fontsize=9,
              bbox_to_anchor=(0.02, 0.30))
    _finish(ax, xl, (-9.8, 10.8))
    fig.tight_layout(pad=0.2)
    out = os.path.join(FIGDIR, "refraction_chain.svg")
    fig.savefig(out)
    plt.close(fig)
    return out, (xt, zt), (xc, zc)


def fig_joint():
    s = STACK
    xt, zt = true_flat_path(**s)

    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    xl = (-1.4, 16.5)
    _draw_stack(ax, xl)

    for i in range(len(xt) - 1):
        _ray(ax, xt[i:i + 2], zt[i:i + 2], C_TRUE)

    # crossings as the unknowns; tridiagonal coupling arcs between neighbours
    xc, zc = xt[1:-1], zt[1:-1]
    for k in range(len(xc)):
        ax.plot([xc[k]], [zc[k]], "o", color=C_TRUE, ms=8, zorder=6,
                markerfacecolor="white", markeredgewidth=1.8)
        ax.annotate(fr"$x_{k + 1}$", (xc[k], zc[k]), color=C_TRUE, fontsize=10,
                    xytext=(-15, -3), textcoords="offset points")
    for k in range(len(xc) - 1):
        mid_x = 0.5 * (xc[k] + xc[k + 1]) + 1.6
        mid_z = 0.5 * (zc[k] + zc[k + 1])
        ax.annotate("", xy=(xc[k + 1], zc[k + 1]), xytext=(xc[k], zc[k]),
                    arrowprops=dict(arrowstyle="<->", color=C_CHAIN, lw=1.6,
                                    connectionstyle="arc3,rad=0.5",
                                    shrinkA=9, shrinkB=9), zorder=4)
        ax.annotate(fr"$x_{k+1}\!\leftrightarrow\!x_{k+2}$", (mid_x, mid_z),
                    color=C_CHAIN, fontsize=8.5, ha="left", va="center")

    _endpoints(ax, s["px"], s["pz"], s["qx"], s["qz"])
    ax.annotate("solve all crossings jointly\n(block-tridiagonal coupling)",
                (xl[1] - 0.3, 9.6), color=C_ENDPT, fontsize=9, ha="right",
                va="top")
    _finish(ax, xl, (-9.8, 10.8))
    fig.tight_layout(pad=0.2)
    out = os.path.join(FIGDIR, "refraction_joint.svg")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    o1 = fig_single()
    o2, (xt, zt), (xc, zc) = fig_chain()
    o3 = fig_joint()
    # deviation reported for the doc caption (max horizontal crossing offset)
    dev = np.max(np.abs(xt[1:-1] - xc[1:-1]))
    for o in (o1, o2, o3):
        print(f"wrote {os.path.relpath(o)}  ({os.path.getsize(o)/1024:.1f} kB)")
    print(f"max chain-vs-true crossing deviation: {dev:.3f} m (exaggerated n)")


if __name__ == "__main__":
    main()
