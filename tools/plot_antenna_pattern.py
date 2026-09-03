"""Plot the assumed element and cross-track array power-gain patterns."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soundersim import antenna  # noqa: E402
from soundersim.config import AntennaConfig  # noqa: E402

import clutter_instruments  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", help="name from config/instruments")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    source = clutter_instruments.load_all()[args.instrument].simulated.antenna
    ant = AntennaConfig(**source.model_dump(exclude_none=True))
    if ant.kind not in ("array", "array_tapered"):
        raise SystemExit("instrument antenna must be an array")
    if ant.element_directivity_db is None:
        raise SystemExit("instrument must specify element_directivity_db")

    theta_deg = np.linspace(-90.0, 90.0, 3601)
    theta = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta)
    q = antenna.element_power_exponent(ant.element_directivity_db)
    d_element = 10.0 ** (ant.element_directivity_db / 10.0)
    e_field = np.sqrt(d_element) * antenna.element_field_pattern(cos_theta, q)

    dhat = np.column_stack([np.zeros_like(theta), -np.sin(theta),
                            -cos_theta])
    u_at = np.array([1.0, 0.0, 0.0])
    u_ct = np.array([0.0, -1.0, 0.0])
    array_field = antenna.field_gain(ant, dhat, u_at, u_ct)

    def power_db(field):
        return 20.0 * np.log10(np.maximum(np.abs(field), 1e-6))

    element_db = power_db(e_field)
    array_db = power_db(array_field)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                             constrained_layout=True)
    axes[0].plot(theta_deg, element_db, color="C1")
    axes[0].set(title=(f"One element: {ant.element_directivity_db:g} dBi, "
                       f"power pattern cos(theta)^{q:.3f}"),
                ylabel="Power gain (dBi)")
    axes[1].plot(theta_deg, array_db, color="C0")
    n = ant.n_elements if ant.kind == "array" else len(ant.tx_weights)
    axes[1].set(title=f"Array: {n} elements, spacing {ant.spacing_lam:g} lambda",
                xlabel="Cross-track angle from nadir (deg)",
                ylabel="Power gain (dBi)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-90.0, 90.0)
    axes[0].set_ylim(-60.0, element_db.max() + 2.0)
    axes[1].set_ylim(-60.0, array_db.max() + 2.0)
    fig.suptitle(f"{args.instrument} assumed antenna patterns")

    out = args.out or ROOT / "outputs" / "antenna_patterns" / \
        f"{args.instrument}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
