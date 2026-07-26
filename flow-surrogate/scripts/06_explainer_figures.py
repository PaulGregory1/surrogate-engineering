"""Figures for BACKGROUND.md: what each surrogate family actually does.

These are teaching figures, not measurements.  Two of the three are computed from the
real dataset (the POD spectrum, the Fourier truncation of a real geometry) so that the
numbers quoted in BACKGROUND.md are the ones this repo actually produces; the third is
a hand-drawn schematic of the three architectures.

    python scripts/06_explainer_figures.py          # -> figs/pod_explained.png
                                                    #    figs/architectures.png
                                                    #    figs/fourier_truncation.png

`--quick` skips the dataset-backed panels (useful without data/).
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
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from flowsurrogate.data import load_splits
from flowsurrogate.geometry import Domain, level_set, solid_mask
from flowsurrogate.metrics import masked_relative_l2

# Measured test-split errors, from RESULTS.md -- drawn as reference lines so the
# projection floor can be read against what the fitted ROMs actually scored.
POD_RIDGE_RELL2 = 0.1439
POD_KRR_RELL2 = 0.1066
N_MODES_USED = 64

INK = "#1f4e9c"
GREY = "0.35"


# ======================================================================================
# 1. POD
# ======================================================================================
def figure_pod(data_dir: Path, out: Path, domain: Domain) -> None:
    """Modes, singular-value spectrum, and the rank-r projection floor."""
    train, _, test, norm = load_splits(data_dir, domain)
    n = len(train)
    flat = train.y.numpy().reshape(n, -1)  # normalised, masked -- exactly what PODRegressor fits
    mean = flat.mean(axis=0)
    u_, s_, vt = np.linalg.svd(flat - mean, full_matrices=False)

    energy = np.cumsum(s_**2) / (s_**2).sum()
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ranks = [r for r in ranks if r < n]

    # Projection floor: the best any regression on these modes could do, i.e. project the
    # *test* snapshots themselves onto the rank-r training subspace.
    test_flat = test.y.numpy().reshape(len(test), -1)
    true_phys = torch.from_numpy(test.fields_raw) * test.m
    floor = []
    for r in ranks:
        basis = vt[:r]
        rec = (test_flat - mean) @ basis.T @ basis + mean
        rec_t = torch.from_numpy(rec.reshape(test.y.shape).astype(np.float32)) * test.m
        pred_phys = norm.decode_fields(rec_t) * test.m
        floor.append(float(masked_relative_l2(pred_phys, true_phys, test.m).mean()))

    fig = plt.figure(figsize=(11.0, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.35])

    # --- top row: the mean and the first three modes, u channel -----------------------
    # Each snapshot is masked by its own geometry before the SVD, so there is no single
    # body to cut out; show the modes outside the cells that are solid in more than half
    # of the training snapshots.
    mask0 = train.m[:, 0].numpy().mean(axis=0) > 0.5
    panels = [("mean snapshot $\\bar{Y}$", mean.reshape(3, domain.ny, domain.nx)[0])]
    for j in range(3):
        panels.append((f"mode $\\psi_{{{j + 1}}}$  ({100 * s_[j] ** 2 / (s_**2).sum():.1f}% energy)",
                       vt[j].reshape(3, domain.ny, domain.nx)[0]))
    for ax, (title, img) in zip([fig.add_subplot(gs[0, k]) for k in range(4)], panels):
        v = np.abs(img[mask0]).max()
        ax.imshow(np.ma.masked_where(~mask0, img), origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8.5)
    fig.text(0.005, 0.985, "(a)  the leading POD modes of $u$ ($640$ snapshots)", fontsize=9.5, va="top")

    # --- bottom left: spectrum ---------------------------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    ax.semilogy(np.arange(1, len(s_) + 1), s_ / s_[0], color=INK, lw=1.6)
    ax.axvline(N_MODES_USED, color="k", ls="--", lw=1.0)
    ax.text(N_MODES_USED * 1.1, 0.35, f"$r = {N_MODES_USED}$ modes kept\n"
            f"{100 * energy[N_MODES_USED - 1]:.2f}% of the energy", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("mode index $j$")
    ax.set_ylabel("$\\sigma_j / \\sigma_1$")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("(b)  singular values: the snapshots live near a low-dimensional subspace", fontsize=9.5, loc="left")

    # --- bottom right: projection floor vs what the ROMs scored ------------------------
    ax = fig.add_subplot(gs[1, 2:])
    ax.loglog(ranks, floor, "o-", color=INK, ms=4, label="rank-$r$ projection of the test fields\n(the basis's own floor)")
    ax.axhline(POD_KRR_RELL2, color="#c1121f", ls="--", lw=1.2, label=f"POD + kernel ridge, measured ({POD_KRR_RELL2:.3f})")
    ax.axhline(POD_RIDGE_RELL2, color="#f08c00", ls=":", lw=1.4, label=f"POD + ridge, measured ({POD_RIDGE_RELL2:.3f})")
    ax.axvline(N_MODES_USED, color="k", ls="--", lw=1.0)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.minorticks_off()
    ax.set_xlabel("number of modes $r$")
    ax.set_ylabel("test rel. $L_2$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("(c)  the ROM's error is the regression, not the basis", fontsize=9.5, loc="left")

    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}  (rank-{N_MODES_USED} projection floor {floor[ranks.index(N_MODES_USED)]:.4f})")


# ======================================================================================
# 2. Architectures
# ======================================================================================
def _box(ax, x, y, w, h, text, fc="#eaf1fb", ec=INK, fontsize=7.5, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3)


def _arrow(ax, p0, p1, style="-|>", color="k", lw=1.1, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=10, lw=lw, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=4))


def _panel_pod(ax) -> None:
    ax.set_title("(a)  POD + regression — a linear reduced-order model", fontsize=9.5, loc="left")
    _box(ax, 0.02, 0.66, 0.16, 0.22, "design vector\n$d = (r_0, \\mathbf{c}, Re)\\in\\mathbb{R}^8$")
    _box(ax, 0.24, 0.66, 0.20, 0.22, "regression $d \\mapsto \\hat\\alpha$\nridge  /  kernel ridge", fc="#fff3bf", ec="#f08c00")
    _box(ax, 0.50, 0.66, 0.16, 0.22, "coefficients\n$\\hat\\alpha \\in \\mathbb{R}^{64}$")
    _box(ax, 0.72, 0.66, 0.26, 0.22, "$\\hat Y = \\bar Y + \\sum_j \\hat\\alpha_j \\psi_j$\nfield on the full lattice")
    for x0, x1 in [(0.18, 0.24), (0.44, 0.50), (0.66, 0.72)]:
        _arrow(ax, (x0, 0.77), (x1, 0.77))

    _box(ax, 0.24, 0.18, 0.42, 0.26,
         "offline, once:  $\\;Y - \\bar Y = U \\Sigma V^\\top$\n"
         "modes $\\psi_j$ = leading right singular vectors\n"
         "(optimal rank-$r$ subspace, Eckart–Young)", fc="#f1f3f5", ec=GREY)
    _arrow(ax, (0.50, 0.44), (0.58, 0.66), rad=0.15, color=GREY, ls="--")
    ax.text(0.70, 0.30, "fitted in seconds;\n$\\sim 2\\times10^4$ faster than the solver\nat inference",
            fontsize=7.5, color=GREY)
    ax.text(0.02, 0.42, "sees only the\n7 CAD parameters\n$\\Rightarrow$ welded to one\nparametrisation",
            fontsize=7.5, color="#c1121f", va="top")


def _panel_unet(ax) -> None:
    ax.set_title("(b)  U-Net — a learned local stencil on a resolution pyramid", fontsize=9.5, loc="left")
    levels = [(96, 192, 32), (48, 96, 64), (24, 48, 128)]
    xs_enc = [0.05, 0.16, 0.27]
    xs_dec = [0.84, 0.73, 0.62]
    ys = [0.60, 0.46, 0.33]
    hs = [0.22, 0.17, 0.12]
    w = 0.085
    xb, yb, hb = 0.42, 0.20, 0.09  # bottleneck

    for i, ((ny, nx, ch), xe, xd, y, h) in enumerate(zip(levels, xs_enc, xs_dec, ys, hs)):
        _box(ax, xe, y, w, h, f"{ny}x{nx}\n{ch} ch", fontsize=6.8)
        _box(ax, xd, y, w, h, f"{ny}x{nx}\n{ch} ch", fontsize=6.8)
        _arrow(ax, (xe + w, y + h / 2), (xd, y + h / 2), color="#2b8a3e", ls="--", lw=1.0)
        ax.text((xe + xd) / 2 + 0.04, y + h / 2 + 0.015, "skip connection", fontsize=6.5, color="#2b8a3e", ha="center")

    _box(ax, xb, yb, 0.11, hb, "12x24, 256 ch\nbottleneck", fontsize=6.8)
    down = [(xs_enc[0] + w, ys[0]), (xs_enc[1] + w, ys[1]), (xs_enc[2] + w, ys[2])]
    dtar = [(xs_enc[1], ys[1] + hs[1]), (xs_enc[2], ys[2] + hs[2]), (xb, yb + hb)]
    for p0, p1 in zip(down, dtar):
        _arrow(ax, p0, p1)
    up = [(xb + 0.11, yb + hb), (xs_dec[2] + w, ys[2] + hs[2]), (xs_dec[1] + w, ys[1] + hs[1])]
    utar = [(xs_dec[2], ys[2]), (xs_dec[1], ys[1]), (xs_dec[0], ys[0])]
    for p0, p1 in zip(up, utar):
        _arrow(ax, p0, p1)
    ax.text(0.20, 0.30, "pool $\\downarrow 2$", fontsize=6.5, color=GREY, rotation=-40)
    ax.text(0.70, 0.30, "upsample $\\uparrow 2$", fontsize=6.5, color=GREY, rotation=40)

    ax.text(0.05, 0.92, "input  $3\\times96\\times192$\n$[\\tanh(\\varphi/L_s),\\ \\mathbf{1}_{\\varphi<0},\\ Re/Re_s]$", fontsize=7.5, va="top")
    _arrow(ax, (0.09, 0.845), (0.09, 0.82))
    ax.text(0.88, 0.92, "output  $3\\times96\\times192$\n$(\\hat u, \\hat v, \\hat p)$", fontsize=7.5, ha="center", va="top")
    _arrow(ax, (0.88, 0.82), (0.88, 0.845))
    ax.text(0.02, 0.08, "each block: $3{\\times}3$ conv $\\to$ GroupNorm $\\to$ GELU, twice.  Pooling is what makes the stencil\n"
                        "reach: receptive field 5 cells after the first block, 68 at the bottleneck, $\\approx$96 at the output —\n"
                        "half the channel length, from nothing but $3{\\times}3$ stencils.",
            fontsize=7.2, color=GREY, va="bottom")
    _box(ax, 0.60, 0.13, 0.30, 0.09, "$\\hat C_d = \\mathrm{MLP}(\\langle z \\rangle_x)$   (drag head)", fc="#fff3bf", ec="#f08c00", fontsize=7.5)
    _arrow(ax, (0.53, 0.24), (0.66, 0.22), color="#f08c00", rad=-0.25)


def _panel_fno(ax) -> None:
    ax.set_title("(c)  FNO — a learned multiplier in Fourier space (global support)", fontsize=9.5, loc="left")
    _box(ax, 0.02, 0.55, 0.13, 0.22, "input\n+ $(x, y)$")
    _box(ax, 0.18, 0.55, 0.10, 0.22, "lift\n$1{\\times}1$")
    _arrow(ax, (0.15, 0.66), (0.18, 0.66))

    _box(ax, 0.34, 0.72, 0.13, 0.16, "$\\mathcal{F}$  (FFT)", fc="#f1f3f5", ec=GREY)
    _box(ax, 0.50, 0.72, 0.18, 0.16, "keep $|k| \\leq 16$,\nmultiply by $R_\\phi$", fc="#fff3bf", ec="#f08c00")
    _box(ax, 0.71, 0.72, 0.13, 0.16, "$\\mathcal{F}^{-1}$", fc="#f1f3f5", ec=GREY)
    _box(ax, 0.50, 0.44, 0.18, 0.14, "$W z$  ($1{\\times}1$ conv)", fc="#f1f3f5", ec=GREY)
    _box(ax, 0.88, 0.55, 0.10, 0.22, "$\\sigma(\\cdot)$")

    _arrow(ax, (0.28, 0.66), (0.34, 0.80), rad=0.15)
    _arrow(ax, (0.28, 0.66), (0.50, 0.51), rad=-0.10)
    _arrow(ax, (0.47, 0.80), (0.50, 0.80))
    _arrow(ax, (0.68, 0.80), (0.71, 0.80))
    _arrow(ax, (0.84, 0.80), (0.90, 0.77), rad=0.15)
    _arrow(ax, (0.68, 0.51), (0.90, 0.55), rad=-0.15)
    ax.text(0.93, 0.44, "$\\times 4$ layers", fontsize=7.5, color=GREY, ha="center")

    ax.text(0.02, 0.30,
            "one layer  $z \\leftarrow \\sigma(Wz + \\mathcal{F}^{-1}[R_\\phi \\cdot \\mathcal{F}z])$\n"
            "$R_\\phi$ is a $16{\\times}16{\\times}32{\\times}32$ complex tensor: a *global* kernel, learned directly in the\n"
            "frequency domain.  Resolution-independent by construction — and unable to represent a\n"
            "sharp moving boundary with 16 modes, which is the failure mode measured in this repo.",
            fontsize=7.5)


def figure_architectures(out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 11.0), constrained_layout=True)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    _panel_pod(axes[0])
    _panel_unet(axes[1])
    _panel_fno(axes[2])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


# ======================================================================================
# 3. Fourier truncation
# ======================================================================================
def _truncate(img: np.ndarray, modes: int) -> np.ndarray:
    """Zero every Fourier coefficient outside the FNO's |k1|, |k2| <= modes box."""
    f = np.fft.fft2(img)
    keep = np.zeros_like(f)
    keep[:modes, :modes] = f[:modes, :modes]
    keep[-modes:, :modes] = f[-modes:, :modes]
    keep[:modes, -modes:] = f[:modes, -modes:]
    keep[-modes:, -modes:] = f[-modes:, -modes:]
    return np.real(np.fft.ifft2(keep))


def figure_fourier(data_dir: Path, out: Path, domain: Domain, modes: int = 16) -> None:
    """What 16 Fourier modes can and cannot represent, on a real sample."""
    raw = np.load(data_dir / "test.npz")
    idx = int(np.argmax(raw["cd"]))
    params, u = raw["params"][idx], raw["u"][idx]
    mask = solid_mask(params, domain)
    phi = level_set(params, domain)

    occ = (~mask).astype(float)  # 1 in the fluid, 0 in the solid: the discontinuity
    occ_t = _truncate(occ, modes)
    u_t = _truncate(u, modes)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 5.4), constrained_layout=True)

    ax = axes[0, 0]
    ax.imshow(occ, origin="lower", cmap="gray", aspect="equal")
    ax.contour(phi, levels=[0], colors=["#c1121f"], linewidths=1.2)
    ax.set_title("(a)  the geometry the network is handed:\na step across $\\partial\\Omega_b$", fontsize=9, loc="left")

    ax = axes[0, 1]
    im = ax.imshow(occ_t, origin="lower", cmap="gray", vmin=-0.3, vmax=1.3, aspect="equal")
    ax.contour(phi, levels=[0], colors=["#c1121f"], linewidths=1.2)
    ax.set_title(f"(b)  the same, through the FNO's ${modes}{{\\times}}{modes}$ mode filter:\nGibbs ringing over the whole channel", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03)

    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])

    row = domain.ny // 2
    ax = axes[1, 0]
    ax.plot(occ[row], color="k", lw=1.4, label="exact occupancy")
    ax.plot(occ_t[row], color="#c1121f", lw=1.2, label=f"{modes} modes")
    ax.plot(_truncate(occ, 2 * modes)[row], color="#f08c00", lw=1.0, ls="--", label=f"{2 * modes} modes")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("$x$ (cells), horizontal cut through the body")
    ax.set_ylabel("occupancy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("(c)  the overshoot does not shrink with more modes\n(Gibbs: it narrows, it does not decay)", fontsize=9, loc="left")

    ax = axes[1, 1]
    for field, lab, col in [(u, "velocity field $u$", INK), (occ, "occupancy (geometry)", "#c1121f")]:
        f = np.abs(np.fft.fft2(field - field.mean()))
        ky = np.fft.fftfreq(f.shape[0]) * f.shape[0]
        kx = np.fft.fftfreq(f.shape[1]) * f.shape[1]
        kk = np.hypot(*np.meshgrid(kx, ky))
        bins = np.arange(0, 48)
        spec = np.array([f[(kk >= b) & (kk < b + 1)].mean() if np.any((kk >= b) & (kk < b + 1)) else np.nan for b in bins])
        ax.semilogy(bins, spec / np.nanmax(spec), color=col, lw=1.4, label=lab)
    ax.axvline(modes, color="k", ls="--", lw=1.0)
    ax.text(modes + 1, 0.4, f"FNO cutoff\n$|k| = {modes}$", fontsize=8)
    ax.set_ylim(1e-5, 2)
    ax.set_xlabel("radial wavenumber $|k|$")
    ax.set_ylabel("normalised amplitude")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("(d)  where the energy sits: the field is smooth,\nthe geometry is not", fontsize=9, loc="left")

    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figs"))
    ap.add_argument("--quick", action="store_true", help="schematic only, no dataset needed")
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    domain = Domain()

    figure_architectures(args.fig_dir / "architectures.png")
    if not args.quick:
        figure_pod(args.data_dir, args.fig_dir / "pod_explained.png", domain)
        figure_fourier(args.data_dir, args.fig_dir / "fourier_truncation.png", domain)


if __name__ == "__main__":
    main()
