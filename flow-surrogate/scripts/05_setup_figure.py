"""Explanatory figure for the README: domain, design space, network inputs.

This is not part of the measurement pipeline -- it draws the problem the rest of the
repo solves, so it needs no dataset and no trained model and runs in a second.

    python scripts/05_setup_figure.py        # -> figs/setup.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from flowsurrogate.data import SDF_SCALE
from flowsurrogate.geometry import (
    COEFF_RANGE,
    Domain,
    area,
    level_set,
    r0_for_area,
    radius,
    sample_shapes,
)

SHOWN = np.array([10.0, 0.11, -0.06, 0.07, 0.04, -0.05, 0.06])
MASK_EPS = 0.7  # cells -- the value used by the shape optimiser


def panel_domain(ax, dom: Domain) -> None:
    ax.add_patch(Rectangle((0, 0), dom.nx, dom.ny, fc="#eaf1fb", ec="k", lw=1.2, zorder=0))
    th = np.linspace(0, 2 * np.pi, 400)
    r = radius(th, SHOWN)
    ax.fill(dom.cx + r * np.cos(th), dom.cy + r * np.sin(th), fc="0.35", ec="k", lw=1.0, zorder=3)

    for y in np.linspace(9, dom.ny - 9, 6):
        ax.annotate("", xy=(19, y), xytext=(1, y), arrowprops=dict(arrowstyle="->", lw=1.0, color="#1f4e9c"))
    for y in (0, dom.ny):
        ax.plot([0, dom.nx], [y, y], color="k", lw=2.0)

    ax.text(1, dom.ny + 6, r"inlet  $u = U_\infty e_x$", fontsize=8, color="#1f4e9c")
    ax.text(dom.nx - 1, dom.ny + 6, r"outlet  $\partial_x f = 0$", fontsize=8, ha="right", color="#1f4e9c")
    ax.text(dom.nx / 2, dom.ny + 6, r"free slip  $u\!\cdot\!n = 0$", fontsize=8, ha="center")
    ax.text(dom.nx / 2, -11, r"free slip  $u\!\cdot\!n = 0$", fontsize=8, ha="center")
    ax.annotate(
        "no slip (bounce-back)\n" r"$\partial\Omega_b(c)$",
        xy=(dom.cx + 8, dom.cy + 8),
        xytext=(dom.cx + 30, dom.ny - 16),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.9),
    )
    d_eq = 2.0 * np.sqrt(area(SHOWN) / np.pi)
    ax.annotate("", xy=(dom.cx - d_eq / 2, 14), xytext=(dom.cx + d_eq / 2, 14), arrowprops=dict(arrowstyle="<->", lw=0.9))
    ax.text(dom.cx, 6, r"$D_{eq}$", fontsize=8, ha="center")
    ax.text(
        4,
        8,
        r"$Re = U_\infty D_{eq}/\nu \in [30, 110]$",
        fontsize=8,
        bbox=dict(fc="w", ec="0.7", boxstyle="round,pad=0.25"),
    )

    ax.set_xlim(-6, dom.nx + 6)
    ax.set_ylim(-18, dom.ny + 16)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"(a)  {dom.nx}x{dom.ny} lattice, D2Q9 BGK", fontsize=9, loc="left")


def panel_design_space(ax, dom: Domain) -> None:
    """Ten members of the family, all at exactly the same enclosed area."""
    target = area(SHOWN)
    rng = np.random.default_rng(3)
    coeffs = sample_shapes(rng, 10, coeff_range=COEFF_RANGE)[:, 1:]
    th = np.linspace(0, 2 * np.pi, 400)
    for c in coeffs:
        p = np.concatenate([[r0_for_area(c, target)], c])
        r = radius(th, p)
        ax.plot(r * np.cos(th), r * np.sin(th), lw=1.0, alpha=0.8)
    r_circ = np.sqrt(target / np.pi)
    ax.plot(r_circ * np.cos(th), r_circ * np.sin(th), "k--", lw=1.4, label="equal-area disc")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(
        r"(b)  $r(\theta)=r_0(c)\,[1+\sum_{k=2,3,4} a_k\cos k\theta + b_k\sin k\theta]$,  $A(c)=A_0$",
        fontsize=9,
        loc="left",
    )


def panel_channels(ax, dom: Domain) -> None:
    phi = level_set(SHOWN, dom)
    im = ax.imshow(np.tanh(phi / SDF_SCALE), origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.contour(phi, levels=[0.0], colors="k", linewidths=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)
    ax.set_title(r"(c)  network input $\tanh(\varphi/L_s)$, $\varphi = \rho - r(\theta)$", fontsize=9, loc="left")


def panel_smoothing(ax, dom: Domain) -> None:
    """Hard occupancy has zero gradient; sigmoid(-phi/eps) is the same image with one."""
    phi = level_set(SHOWN, dom)
    row = int(round(dom.cy))
    x = np.arange(dom.nx)
    hard = (phi[row] < 0).astype(float)
    soft = 1.0 / (1.0 + np.exp(phi[row] / MASK_EPS))
    lo, hi = 40, 76

    ax.step(x[lo:hi], hard[lo:hi], where="mid", color="0.25", lw=1.6, label=r"$\mathbf{1}_{\varphi<0}$  (training)")
    ax.plot(x[lo:hi], soft[lo:hi], color="#c1121f", lw=1.6, label=r"$\sigma(-\varphi/\epsilon)$  (optimising)")
    ax.set_xlabel("lattice cell along $y = c_y$", fontsize=8)
    ax.set_ylabel("occupancy", fontsize=8)
    ax.tick_params(labelsize=7)

    ax2 = ax.twinx()
    dsig = soft * (1.0 - soft) / MASK_EPS
    ax2.fill_between(x[lo:hi], dsig[lo:hi], color="#c1121f", alpha=0.18, lw=0)
    ax2.set_ylabel(r"$|\partial\sigma/\partial\varphi|$", fontsize=8, color="#c1121f")
    ax2.tick_params(labelsize=7, colors="#c1121f")

    ax.legend(fontsize=7, loc="upper center")
    ax.set_title(r"(d)  the shape gradient lives in a $\pm\,\epsilon$ band ($\epsilon = 0.7$ cell)", fontsize=9, loc="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figs/setup.png"))
    args = ap.parse_args()

    dom = Domain()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), constrained_layout=True)
    panel_domain(axes[0, 0], dom)
    panel_design_space(axes[0, 1], dom)
    panel_channels(axes[1, 0], dom)
    panel_smoothing(axes[1, 1], dom)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
