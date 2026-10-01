"""Figures, drawn directly on the unstructured mesh (no resampling to a grid)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _tri(ax, sample, values, vmin, vmax, cmap="inferno"):
    nodes, tris = sample["nodes"], sample["tris"]
    im = ax.tripcolor(nodes[:, 0], nodes[:, 1], tris, values, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def stress_comparison(sample: dict, preds: dict[str, np.ndarray], path: Path, title: str = "") -> None:
    """Reference / prediction / error, one column per model."""
    target = sample["vm_target"]
    vmax = float(target.max())
    n = len(preds)
    fig, axes = plt.subplots(2, n + 1, figsize=(3.4 * (n + 1), 3.6), constrained_layout=True)
    axes = np.atleast_2d(axes)

    im = _tri(axes[0, 0], sample, target, 0, vmax)
    axes[0, 0].set_title(f"reference (fine FEM)\nmax = {sample['max_vm_fine']:.2f}", fontsize=9)
    fig.colorbar(im, ax=axes[0, 0], fraction=0.03)
    axes[1, 0].axis("off")
    axes[1, 0].text(0.0, 0.6, title, fontsize=9, va="center")

    for j, (name, p) in enumerate(preds.items(), start=1):
        im = _tri(axes[0, j], sample, p, 0, vmax)
        axes[0, j].set_title(f"{name}\nmax = {p.max():.2f}", fontsize=9)
        fig.colorbar(im, ax=axes[0, j], fraction=0.03)
        err = np.abs(p - target)
        im2 = _tri(axes[1, j], sample, err, 0, 0.25 * vmax, cmap="viridis")
        axes[1, j].set_title(f"|error|, rel L2 = {np.linalg.norm(p - target) / np.linalg.norm(target):.3f}", fontsize=8)
        fig.colorbar(im2, ax=axes[1, j], fraction=0.03)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def ablation_curve(points: dict[str, list[tuple[float, float]]], path: Path, xlabel: str, ylabel: str, logx=False) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
    for name, pts in points.items():
        pts = sorted(pts)
        style = "o--" if "FEM" in name or "floor" in name else "o-"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=name, ms=4)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def bar_comparison(groups: dict[str, dict[str, float]], path: Path, ylabel: str) -> None:
    """Grouped bars: one group per split, one bar per model."""
    splits = list(groups)
    models = list(next(iter(groups.values())))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(1.9 * len(splits) + 2.5, 3.4), constrained_layout=True)
    for k, m in enumerate(models):
        xs = np.arange(len(splits)) + k * width
        ax.bar(xs, [groups[s].get(m, np.nan) for s in splits], width, label=m)
    ax.set_xticks(np.arange(len(splits)) + 0.4 - width / 2)
    ax.set_xticklabels(splits, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def optimisation_summary(history: dict, before: dict, after: dict, path: Path) -> None:
    fig = plt.figure(figsize=(10.5, 3.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.2])

    ax = fig.add_subplot(gs[0])
    ax.plot(history["best_so_far"], lw=1.6)
    ax.set_xlabel("designs evaluated (surrogate)")
    ax.set_ylabel("predicted peak von Mises")
    ax.grid(alpha=0.3)
    ax.set_title("search on the surrogate", fontsize=10)

    for j, (label, blob) in enumerate([("start", before), ("optimised", after)], start=1):
        ax = fig.add_subplot(gs[j])
        vmax = max(float(before["vm"].max()), float(after["vm"].max()))
        im = _tri(ax, blob["sample"], blob["vm"], 0, vmax)
        ax.set_title(f"{label}: peak = {blob['peak']:.2f} (fine FEM)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.savefig(path, dpi=130)
    plt.close(fig)
