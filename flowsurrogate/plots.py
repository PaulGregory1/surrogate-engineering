"""Figures.  Matplotlib only, no seaborn, no styling that needs a network."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELD_LABELS = ["u", "v", "p"]


def _show(ax, field: np.ndarray, mask: np.ndarray, vmin=None, vmax=None, cmap="RdBu_r"):
    masked = np.ma.masked_where(mask < 0.5, field)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("0.35")
    im = ax.imshow(masked, origin="lower", cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def field_comparison(
    true: np.ndarray,
    preds: dict[str, np.ndarray],
    mask: np.ndarray,
    path: Path,
    title: str = "",
    channel: int = 0,
) -> None:
    """One row per model: prediction and error, against the solver reference."""
    names = list(preds)
    fig, axes = plt.subplots(len(names) + 1, 2, figsize=(9, 1.9 * (len(names) + 1)), constrained_layout=True)
    vmax = np.abs(true[channel][mask > 0.5]).max()
    _show(axes[0, 0], true[channel], mask, -vmax, vmax)
    axes[0, 0].set_ylabel("LBM", fontsize=9)
    axes[0, 1].axis("off")
    axes[0, 1].text(0.0, 0.5, f"reference  {FIELD_LABELS[channel]}\n{title}", fontsize=9, va="center")
    for i, name in enumerate(names, start=1):
        err = (preds[name][channel] - true[channel]) * mask
        _show(axes[i, 0], preds[name][channel], mask, -vmax, vmax)
        im = _show(axes[i, 1], err, mask, -0.15 * vmax, 0.15 * vmax, cmap="coolwarm")
        axes[i, 0].set_ylabel(name, fontsize=9)
        axes[i, 1].set_title("error (x6.7 scale)", fontsize=7)
        fig.colorbar(im, ax=axes[i, 1], fraction=0.03)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cd_scatter(results: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    """Predicted vs solver drag coefficient, one panel per model."""
    fig, axes = plt.subplots(1, len(results), figsize=(3.1 * len(results), 3.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (name, (pred, true)) in zip(axes, results.items()):
        lo, hi = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.scatter(true, pred, s=9, alpha=0.6)
        err = 100 * np.abs((pred - true) / true).mean()
        ax.set_title(f"{name}\nMAPE {err:.2f}%", fontsize=9)
        ax.set_xlabel("$C_d$ LBM", fontsize=8)
        ax.set_ylabel("$C_d$ surrogate", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def data_efficiency(curves: dict[str, list[tuple[int, float]]], path: Path, ylabel: str = "test rel. $L_2$") -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.4), constrained_layout=True)
    for name, pts in curves.items():
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=name, ms=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training samples (solver runs)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def optimisation_summary(
    history: dict,
    fields: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Drag history of the optimiser plus the initial/final shape and wake."""
    fig = plt.figure(figsize=(10, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.5])

    ax = fig.add_subplot(gs[:, 0])
    ax.plot(history["cd_surrogate"], label="surrogate $C_d$", lw=1.5)
    for it, cd in history["verified"]:
        ax.plot(it, cd, "kx", ms=8)
    ax.plot([], [], "kx", label="LBM verification")
    ax.set_xlabel("optimiser iteration")
    ax.set_ylabel("$C_d$ (area held fixed)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("gradient descent through the surrogate", fontsize=10)

    for row, key in enumerate(["initial", "optimised"]):
        ax = fig.add_subplot(gs[row, 1])
        vmax = np.abs(fields[key][masks[key] > 0.5]).max()
        _show(ax, fields[key], masks[key], -vmax, vmax)
        ax.set_title(f"{key}: $C_d$ = {history[f'cd_{key}']:.3f} (LBM)", fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def shape_gallery(boundaries: list[np.ndarray], labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)
    for b, lab in zip(boundaries, labels):
        ax.plot(np.append(b[0], b[0][0]), np.append(b[1], b[1][0]), lw=1.6, label=lab)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("shape before / after optimisation", fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)
