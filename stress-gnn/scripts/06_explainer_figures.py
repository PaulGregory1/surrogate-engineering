"""Figures for BACKGROUND.md: what the models are and what limits them.

Teaching figures, not measurements.  Two of the three are computed from the real
dataset (the hop-reach curve on a real coarse mesh, the interpolation-floor histograms
over the 400 training designs); the rest is schematic and says so.

    python scripts/06_explainer_figures.py     # -> figs/graph_explained.png
                                               #    figs/interpolation_floor.png
                                               #    figs/ranking_vs_accuracy.png

`--quick` skips the dataset-backed panels (useful without data/).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.stats import spearmanr

from stressgnn.dataset import GRID_H, GRID_W, rasterise, unpack
from stressgnn.geometry import HEIGHT, WIDTH

INK = "#1f4e9c"
RED = "#c1121f"
GREY = "0.35"

# Measured on the test split (RESULTS.md / results/optimization.json).
FLOOR_ERR_PCT = 10.70
SPEARMAN_COARSE = 0.892
SPEARMAN_GNN = 0.856
COARSE_BIAS = -0.358
GNN_BIAS = 0.016
GNN_P95 = 0.188


# ======================================================================================
# helpers
# ======================================================================================
def _box(ax, x, y, w, h, text, fc="#eaf1fb", ec=INK, fontsize=7.5, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3)


def _arrow(ax, p0, p1, color="k", lw=1.1, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, lw=lw, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=4))


def _adjacency(n_nodes: int, edges: np.ndarray) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _unique_edges(tris: np.ndarray) -> np.ndarray:
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


# ======================================================================================
# 1. mesh -> graph -> message passing -> grid baseline
# ======================================================================================
def _panel_mesh_to_graph(ax, sample) -> None:
    nodes, tris = sample["nodes"], sample["tris"].astype(int)
    edges = _unique_edges(tris)
    cx, cy, r = sample["holes"][0]
    w = 3.2 * r

    seg = nodes[edges]
    ax.add_collection(matplotlib.collections.LineCollection(seg, colors="0.55", linewidths=0.8, zorder=1))
    ax.plot(nodes[:, 0], nodes[:, 1], "o", ms=4.0, mfc="white", mec=INK, mew=1.0, zorder=2)
    hole = sample["node_type"] == 3
    ax.plot(nodes[hole, 0], nodes[hole, 1], "o", ms=4.6, mfc=RED, mec=RED, zorder=3)
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=RED, lw=1.2, zorder=0))

    # one edge, annotated -- picked in the clear area above and left of the hole
    mid = nodes[edges].mean(axis=1)
    target = np.array([cx - 0.55 * w, cy + 0.35 * w])
    a, b = edges[int(np.argmin(np.linalg.norm(mid - target, axis=1)))]
    ax.annotate("", xy=nodes[b], xytext=nodes[a],
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#2b8a3e"))
    ax.plot(nodes[[a, b], 0], nodes[[a, b], 1], "o", ms=5.0, mfc="#2b8a3e", mec="#2b8a3e", zorder=4)
    ax.text(nodes[a, 0], nodes[a, 1] + 0.012, "$i$", fontsize=8, color="#2b8a3e", ha="right", zorder=5)
    ax.text(nodes[b, 0], nodes[b, 1] + 0.012, "$j$", fontsize=8, color="#2b8a3e", ha="left", zorder=5)
    ax.text(0.015, 0.09, "node  $v_i = [\\mathrm{onehot(type)},\\ \\cos\\alpha,\\ \\sin\\alpha,\\ h_i/L,\\ x_i\\ (,\\ u^{c}_i,\\ \\sigma^{c}_i)]$",
            fontsize=7.5, color=INK, transform=ax.transAxes,
            bbox=dict(fc="white", ec="0.8", boxstyle="round,pad=0.25"))
    ax.text(0.015, 0.015, "edge  $e_{ij} = [(x_j - x_i)/L,\\ \\|x_j - x_i\\|/L]$   — relative only",
            fontsize=7.5, color="#2b8a3e", transform=ax.transAxes,
            bbox=dict(fc="white", ec="0.8", boxstyle="round,pad=0.25"))

    ax.set_xlim(cx - w, cx + w)
    ax.set_ylim(cy - 0.62 * w, cy + 0.62 * w)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a)  the mesh *is* the graph — nodes, mesh edges, and their features\n"
                 "red: nodes tagged as hole boundary (that is how the BCs get in)", fontsize=9, loc="left")


def _panel_block(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b)  one message-passing block, repeated $L$ times", fontsize=9, loc="left")

    _box(ax, 0.02, 0.78, 0.19, 0.15, "encode\n$v_i, e_{ij} \\to \\mathbb{R}^{64}$", fc="#f1f3f5", ec=GREY)
    _box(ax, 0.30, 0.78, 0.40, 0.15, "$\\times L$ blocks (below)", fc="#eaf1fb")
    _box(ax, 0.78, 0.78, 0.20, 0.15, "decode\n$\\hat y_i$,  $\\hat s$", fc="#f1f3f5", ec=GREY)
    _arrow(ax, (0.21, 0.855), (0.30, 0.855))
    _arrow(ax, (0.70, 0.855), (0.78, 0.855))

    _box(ax, 0.03, 0.46, 0.42, 0.18,
         "edge update\n$e_{ij} \\leftarrow e_{ij} + \\phi_e([e_{ij}, v_i, v_j])$", fc="#fff3bf", ec="#f08c00")
    _box(ax, 0.03, 0.22, 0.42, 0.18,
         "node update\n$v_i \\leftarrow v_i + \\phi_v([v_i, \\sum_{j \\in \\mathcal{N}(i)} e_{ji}, g])$", fc="#fff3bf", ec="#f08c00")
    _box(ax, 0.03, 0.02, 0.42, 0.14,
         "global update (optional)\n$g \\leftarrow g + \\phi_g([g, \\mathrm{mean}_i\\, v_i])$", fc="#f1f3f5", ec=GREY)
    _arrow(ax, (0.24, 0.46), (0.24, 0.40))
    _arrow(ax, (0.24, 0.22), (0.24, 0.16))

    ax.text(0.50, 0.60,
            "$\\phi_e, \\phi_v, \\phi_g$ are small MLPs,\n"
            "shared by every edge / node / graph.\n\n"
            "Only *relative* positions enter $e_{ij}$,\n"
            "so the same weights run unchanged on a\n"
            "700-node and a 3000-node mesh, and the\n"
            "output is permutation-equivariant:\n"
            "renumber the nodes and the prediction\n"
            "is renumbered with them.\n\n"
            "Cost per block: $\\mathcal{O}(|E|)$, i.e. linear\n"
            "in mesh size — no dense matrix anywhere.",
            fontsize=7.6, va="top")


def _panel_reach(ax, sample) -> None:
    nodes, tris = sample["nodes"], sample["tris"].astype(int)
    edges = _unique_edges(tris)
    adj = _adjacency(len(nodes), edges)
    h_mean = float(np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1).mean())

    start = int(np.argmax(nodes[:, 0] - 0.4 * np.abs(nodes[:, 1] - HEIGHT / 2)))  # a loaded node
    hop = np.full(len(nodes), -1)
    hop[start] = 0
    frontier = [start]
    reach = [0.0]
    for k in range(1, 46):
        nxt = [j for i in frontier for j in adj[i] if hop[j] < 0]
        if not nxt:
            break
        for j in nxt:
            hop[j] = k
        frontier = list(set(nxt))
        reach.append(float(np.linalg.norm(nodes[hop >= 0] - nodes[start], axis=1).max()))

    ks = np.arange(len(reach))
    ax.plot(ks, reach, "-", color=INK, lw=1.8, label="measured on a real coarse mesh")
    ax.plot(ks, ks * h_mean, "--", color=GREY, lw=1.2, label=f"$L \\cdot h$,  $h$ = {h_mean:.3f} (mean edge)")
    ax.axhline(WIDTH, color=RED, ls=":", lw=1.4)
    ax.text(1.0, WIDTH * 1.02, "plate length: what a load on the right edge\nmust cross to be felt at the clamp", fontsize=7.5, color=RED)
    for L, style in [(5, ":"), (10, "-."), (15, "--")]:
        if L < len(reach):
            ax.plot([L, L], [0, reach[L]], color="0.6", ls=style, lw=0.9)
            ax.plot([L], [reach[L]], "o", ms=4, color=INK)
            ax.text(L + 0.4, reach[L] - 0.10, f"$L={L}$: {reach[L]:.2f}", fontsize=7.5)
    ax.set_xlabel("message-passing blocks $L$")
    ax.set_ylabel("distance reached")
    ax.set_xlim(0, min(45, len(reach) - 1))
    ax.set_ylim(0, 2.4)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("(c)  information travels one edge per block:\nthe receptive field is an $L$-hop ball of radius $\\approx L h$", fontsize=9, loc="left")


def _panel_grid(ax, sample) -> None:
    grid, uv = rasterise(sample)
    nodes = sample["nodes"]
    ax.imshow(grid[0], origin="lower", extent=[0, WIDTH, 0, HEIGHT], cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.plot(nodes[:, 0], nodes[:, 1], ".", ms=1.3, color="k", alpha=0.55)
    for cx, cy, r in sample["holes"]:
        ax.add_patch(Circle((cx, cy), r, fill=False, ec="k", lw=0.9))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"(d)  the grid baseline rasterises the part onto a fixed {GRID_H}x{GRID_W} image\n"
                 "instead, runs a U-Net, and samples back at the mesh nodes (dots)",
                 fontsize=9, loc="left")


def figure_graph(sample, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    _panel_mesh_to_graph(axes[0, 0], sample)
    _panel_block(axes[0, 1])
    _panel_reach(axes[1, 0], sample)
    _panel_grid(axes[1, 1], sample)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


# ======================================================================================
# 2. the interpolation floor
# ======================================================================================
def _panel_floor_cartoon(ax) -> None:
    """Schematic: the two averaging steps between the true peak and a nodal value."""
    x = np.linspace(0, 1, 800)
    sigma = 1.0 + 2.0 * np.exp(-((x - 0.30) / 0.055) ** 2)  # a stress concentration
    ax.plot(x, sigma, color="k", lw=1.8, label="true $\\sigma_{vm}(x)$  (peak on the hole boundary)")

    edges = np.linspace(0, 1, 13)  # coarse elements
    centres = 0.5 * (edges[:-1] + edges[1:])
    elem = np.interp(centres, x, sigma)  # P1 strain is constant per element
    ax.step(np.append(edges, 1.0), np.append(np.append(elem[0], elem), elem[-1]), where="pre",
            color=INK, lw=1.3, label="P1 element stress: constant per element")

    nodal = np.zeros(len(edges))
    nodal[1:-1] = 0.5 * (elem[:-1] + elem[1:])
    nodal[0], nodal[-1] = elem[0], elem[-1]
    ax.plot(edges, nodal, "o-", color=RED, ms=5, lw=1.2, label="area-averaged onto nodes, then read at coarse nodes")

    ax.annotate("", xy=(0.30, sigma.max()), xytext=(0.30, nodal.max()),
                arrowprops=dict(arrowstyle="<->", lw=1.2, color="#f08c00"))
    ax.text(0.40, 0.5 * (sigma.max() + nodal.max()),
            "lost before any model is involved:\n"
            f"element $\\to$ node averaging ($\\approx$8%)\n+ restriction to coarse nodes ($\\approx$2%)\n"
            f"$\\Rightarrow$ a {FLOOR_ERR_PCT:.1f}% floor on the peak",
            fontsize=8, color="#f08c00", va="center")

    ax.set_ylim(0.85, 4.1)
    ax.set_xlabel("distance along the boundary (schematic)")
    ax.set_ylabel("$\\sigma_{vm}$")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_title("(a)  two operators sit between the truth and any nodal model,\nand neither commutes with $\\max$  [schematic]", fontsize=9, loc="left")


def _panel_floor_data(ax, blob) -> None:
    fine = blob["max_vm_fine"]
    err = lambda a: 100.0 * np.mean(np.abs(a - fine) / fine)
    ratios = {
        f"target on coarse nodes\n(the floor: {err(blob['max_vm_target']):.1f}% low)": (blob["max_vm_target"] / fine, RED),
        f"coarse FEM solve\n({err(blob['max_vm_coarse']):.1f}% low)": (blob["max_vm_coarse"] / fine, INK),
    }
    bins = np.linspace(0.3, 1.05, 46)
    for label, (r, col) in ratios.items():
        ax.hist(r, bins=bins, alpha=0.65, color=col, label=label)
        ax.axvline(float(np.median(r)), color=col, ls="--", lw=1.2)
    ax.axvline(1.0, color="k", lw=1.4)
    ax.text(1.005, ax.get_ylim()[1] * 0.9, "fine-mesh truth", fontsize=8, rotation=90, va="top")
    ax.set_xlabel("peak von Mises / fine-mesh peak")
    ax.set_ylabel(f"designs (out of {len(fine)})")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("(b)  the same two losses, measured over the training set\n"
                 "both are one-sided: a coarse discretisation is always too stiff", fontsize=9, loc="left")


def figure_floor(blob, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    _panel_floor_cartoon(axes[0])
    _panel_floor_data(axes[1], blob)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


# ======================================================================================
# 3. ranking is not accuracy
# ======================================================================================
def _calibrate_noise(truth: np.ndarray, bias: float, target_rho: float, rng) -> float:
    """Scatter that reproduces a given Spearman correlation against `truth`."""
    lo, hi = 1e-4, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        rho = np.mean([
            spearmanr(truth, truth * (1 + bias) + rng.normal(0, mid * truth.mean(), truth.size)).statistic
            for _ in range(9)
        ])
        if rho > target_rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def figure_ranking(out: Path, n: int = 150, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    truth = np.sort(rng.uniform(19.0, 36.0, n))  # peak stress range of the design study

    s_coarse = _calibrate_noise(truth, COARSE_BIAS, SPEARMAN_COARSE, rng)
    s_gnn = _calibrate_noise(truth, GNN_BIAS, SPEARMAN_GNN, rng)
    coarse = truth * (1 + COARSE_BIAS) + rng.normal(0, s_coarse * truth.mean(), n)
    gnn = truth * (1 + GNN_BIAS) + rng.normal(0, s_gnn * truth.mean(), n)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), constrained_layout=True)
    for ax, (name, pred, col, rho) in zip(axes, [
        ("coarse FEM: pure bias $\\psi$, little scatter $\\varepsilon$", coarse, INK, SPEARMAN_COARSE),
        ("surrogate: bias removed, scatter paid for", gnn, RED, SPEARMAN_GNN),
    ]):
        lo, hi = 0.6 * truth.min(), 1.05 * truth.max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="perfect")
        ax.scatter(truth, pred, s=12, alpha=0.7, color=col)
        ax.set_xlabel("true peak stress")
        ax.set_ylabel("score used to rank")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{name}\nmeasured Spearman = {rho:+.3f}", fontsize=9, loc="left")
        ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("$\\hat s = \\psi(s) + \\varepsilon$:  rank correlation is invariant to any increasing $\\psi$,\n"
                 "and destroyed only by $\\varepsilon$ — so 36% low everywhere ranks better than unbiased-but-noisy.\n"
                 "[synthetic draws, calibrated to the measured bias and Spearman of each strategy]", fontsize=8.5)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figs"))
    ap.add_argument("--sample", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="schematic only, no dataset needed")
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    figure_ranking(args.fig_dir / "ranking_vs_accuracy.png")
    if not args.quick:
        blob = np.load(args.data_dir / "train.npz")
        sample = unpack(blob, args.sample)
        figure_graph(sample, args.fig_dir / "graph_explained.png")
        figure_floor(blob, args.fig_dir / "interpolation_floor.png")


if __name__ == "__main__":
    main()
