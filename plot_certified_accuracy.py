"""Plot certified accuracy vs. L2 radius from a certify.py CSV.

Certified accuracy at radius r = fraction of evaluated images that were
both correctly classified AND certified robust at radius >= r (Cohen et
al. 2019, the standard reporting curve for randomized smoothing).

Example:
  python plot_certified_accuracy.py --csv ./results/certify.csv --out ./results/certified_accuracy.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"


def certified_accuracy_curve(rows, max_radius, num_points=200):
    radii = np.linspace(0, max_radius, num_points)
    correct_radii = np.array([r["radius"] if r["correct"] else -1.0 for r in rows])
    acc = [(correct_radii >= r).mean() for r in radii]
    return radii, acc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="./results/certified_accuracy.png")
    p.add_argument("--max-radius", type=float, default=2.0)
    args = p.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["radius"] = float(r["radius"])
        r["correct"] = int(r["correct"])

    radii, acc = certified_accuracy_curve(rows, args.max_radius)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(radii, acc, color=SERIES, linewidth=2, solid_capstyle="round")

    ax.set_xlim(0, args.max_radius)
    ax.set_ylim(0, 1)
    ax.set_xlabel("L2 radius", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Certified accuracy", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        "Certified accuracy vs. L2 radius (randomized smoothing)",
        color=INK_PRIMARY, fontsize=11, loc="left", pad=12,
    )

    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
