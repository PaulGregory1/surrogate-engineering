"""Explanatory figure for the README: the design, the two meshes, the graph.

Not part of the measurement pipeline -- it meshes and solves a single design so the
README can show what the surrogate actually consumes.  Runs in a few seconds.

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
from matplotlib.patches import Circle, Rectangle

from stressgnn.fem import solve
from stressgnn.geometry import HEIGHT, WIDTH, Design
from stressgnn.mesh import (
    NODE_CLAMPED,
    NODE_FREE,
    NODE_HOLE,
    NODE_INTERIOR,
    NODE_LOADED,
    build_mesh,
)

H_COARSE = 0.045
H_FINE = 0.010
N_HOPS = 10

#: A representative design: the peak sits in the ligament between the top edge and
#: the largest hole, which is where a coarse mesh is worst.
DESIGN = Design(
    holes=np.array([[0.70, 0.74, 0.16], [1.10, 0.30, 0.12], [1.55, 0.62, 0.13]]),
    load_angle=-np.pi / 3,
)

TYPE_STYLE = {
    NODE_INTERIOR: ("#b8c4d4", "interior"),
    NODE_CLAMPED: ("#c1121f", "clamped"),
    NODE_LOADED: ("#1f4e9c", "loaded"),
    NODE_HOLE: ("#0b7a4b", "hole"),
    NODE_FREE: ("#e08c00", "free"),
}


def panel_design(ax) -> None:
    ax.add_patch(Rectangle((0, 0), WIDTH, HEIGHT, fc="#eaf1fb", ec="k", lw=1.3))
    for cx, cy, r in DESIGN.holes:
        ax.add_patch(Circle((cx, cy), r, fc="w", ec="k", lw=1.1))
    ax.annotate(
        r"$(x_j, y_j, r_j)$,  $j = 1 \ldots n_h$",
        xy=(DESIGN.holes[1, 0] + DESIGN.holes[1, 2] * 0.7, DESIGN.holes[1, 1] - DESIGN.holes[1, 2] * 0.7),
        xytext=(0.62, -0.26),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.9),
    )

    for y in np.linspace(0.04, HEIGHT - 0.04, 9):
        ax.plot([-0.06, 0.0], [y + 0.045, y], color="#c1121f", lw=1.1)
    ax.plot([0, 0], [0, HEIGHT], color="#c1121f", lw=2.4)
    ax.text(-0.30, HEIGHT / 2, r"$u = 0$", fontsize=9, color="#c1121f", rotation=90, va="center")

    t = DESIGN.traction
    for y in np.linspace(0.08, HEIGHT - 0.08, 6):
        ax.annotate(
            "",
            xy=(WIDTH + 0.30 * t[0], y + 0.30 * t[1]),
            xytext=(WIDTH, y),
            arrowprops=dict(arrowstyle="->", lw=1.1, color="#1f4e9c"),
        )
    ax.text(WIDTH + 0.10, HEIGHT + 0.08, r"$\sigma n = t(\alpha)$", fontsize=9, color="#1f4e9c")
    ax.text(WIDTH + 0.10, HEIGHT - 0.10, r"$\alpha \in [-\pi/2, 0]$", fontsize=8, color="#1f4e9c")
    ax.text(0.05, HEIGHT + 0.10, r"$\sigma n = 0$  on the holes and the top/bottom edges", fontsize=8)

    ax.set_xlim(-0.42, WIDTH + 0.62)
    ax.set_ylim(-0.38, HEIGHT + 0.30)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a)  design vector: 2-4 holes + load angle", fontsize=9, loc="left")


def panel_mesh(ax, coarse) -> None:
    ax.triplot(coarse.nodes[:, 0], coarse.nodes[:, 1], coarse.tris, color="0.75", lw=0.35)
    for kind, (color, label) in TYPE_STYLE.items():
        sel = coarse.node_type == kind
        ax.plot(coarse.nodes[sel, 0], coarse.nodes[sel, 1], ".", ms=2.6, color=color, label=f"{label} ({sel.sum()})")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.set_title(
        f"(b)  coarse mesh, h = {H_COARSE}: {coarse.n_nodes} nodes, one-hot node type",
        fontsize=9,
        loc="left",
    )


def panel_zoom(ax, coarse, fine, sol_c, sol_f) -> None:
    """Both discretisations around the element carrying the fine-mesh peak."""
    e = int(np.argmax(sol_f.vm_elem))
    cx, cy = fine.nodes[fine.tris[e]].mean(axis=0)
    w, hh = 0.26, 0.26 * 0.62
    cy = float(np.clip(cy, hh, HEIGHT - hh))  # keep the window inside the plate
    ax.triplot(fine.nodes[:, 0], fine.nodes[:, 1], fine.tris, color="#1f4e9c", lw=0.3, alpha=0.7)
    ax.triplot(coarse.nodes[:, 0], coarse.nodes[:, 1], coarse.tris, color="#c1121f", lw=0.9)
    ax.plot([cx], [fine.nodes[fine.tris[e]].mean(axis=0)[1]], "k*", ms=11, zorder=5)
    ax.set_xlim(cx - w, cx + w)
    ax.set_ylim(cy - hh, cy + hh)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([], [], color="#1f4e9c", lw=1.0, label=f"fine h = {H_FINE} ({fine.n_nodes} nodes)")
    ax.plot([], [], color="#c1121f", lw=1.0, label=f"coarse h = {H_COARSE} ({coarse.n_nodes} nodes)")
    ax.legend(fontsize=7, loc="lower left")
    err = 100 * (sol_c.vm_node.max() - sol_f.max_vm) / sol_f.max_vm
    ax.set_title(
        rf"(c)  peak (star) on the hole boundary: fine {sol_f.max_vm:.1f}, coarse {sol_c.vm_node.max():.1f} ({err:+.0f}%)",
        fontsize=9,
        loc="left",
    )


def panel_graph(ax, coarse) -> None:
    """Message passing moves information one edge per block: how far is 10 blocks?"""
    edges = coarse.edges()
    adj: list[list[int]] = [[] for _ in range(coarse.n_nodes)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    start = int(np.argmax(coarse.nodes[:, 0] - 0.4 * np.abs(coarse.nodes[:, 1] - HEIGHT / 2)))
    hop = np.full(coarse.n_nodes, -1)
    hop[start] = 0
    frontier = [start]
    for k in range(1, N_HOPS + 1):
        nxt = [j for i in frontier for j in adj[i] if hop[j] < 0]
        for j in nxt:
            hop[j] = k
        frontier = list(set(nxt))

    seg = coarse.nodes[edges]
    ax.add_collection(
        matplotlib.collections.LineCollection(seg, colors="0.85", linewidths=0.3, zorder=0)
    )
    out = hop < 0
    ax.plot(coarse.nodes[out, 0], coarse.nodes[out, 1], ".", ms=2.0, color="0.72", zorder=1)
    sc = ax.scatter(
        coarse.nodes[~out, 0], coarse.nodes[~out, 1], c=hop[~out], s=5, cmap="viridis_r", vmin=0, vmax=N_HOPS, zorder=2
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("message-passing hops", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    reach = coarse.nodes[~out, 0].min()
    ax.annotate(
        "",
        xy=(reach, -0.10),
        xytext=(WIDTH, -0.10),
        arrowprops=dict(arrowstyle="<->", lw=1.0),
    )
    ax.text((reach + WIDTH) / 2, -0.24, f"{N_HOPS} hops reach {WIDTH - reach:.2f} of {WIDTH}", fontsize=8, ha="center")
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, WIDTH + 0.05)
    ax.set_ylim(-0.34, HEIGHT + 0.05)
    ax.axis("off")
    ax.set_title(f"(d)  {N_HOPS}-hop receptive field from a loaded node", fontsize=9, loc="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figs/setup.png"))
    args = ap.parse_args()

    coarse = build_mesh(DESIGN, H_COARSE)
    fine = build_mesh(DESIGN, H_FINE)
    sol_c = solve(coarse, DESIGN)
    sol_f = solve(fine, DESIGN)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.0), constrained_layout=True)
    panel_design(axes[0, 0])
    panel_mesh(axes[0, 1], coarse)
    panel_zoom(axes[1, 0], coarse, fine, sol_c, sol_f)
    panel_graph(axes[1, 1], coarse)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"wrote {args.out}  (coarse {coarse.n_nodes} nodes, fine {fine.n_nodes} nodes)")


if __name__ == "__main__":
    main()
