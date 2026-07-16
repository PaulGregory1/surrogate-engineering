"""Error measures shared by training and evaluation.

Two families, deliberately kept apart:

* field errors (relative L2 per channel) -- what a loss curve shows;
* engineering errors (drag) and a physics-consistency residual (continuity) -- what
  decides whether a surrogate can actually be put in a design loop.

A surrogate can look excellent on the first and be useless on the last two, which is
why all three are reported side by side in RESULTS.md.
"""

from __future__ import annotations

import numpy as np
import torch


def masked_relative_l2(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample, per-channel ||pred - true|| / ||true||, over fluid cells only."""
    diff = ((pred - true) * mask) ** 2
    ref = (true * mask) ** 2
    return torch.sqrt(diff.sum(dim=(2, 3)) / ref.sum(dim=(2, 3)).clamp_min(1e-12))


def interior_fluid(mask: torch.Tensor) -> torch.Tensor:
    """Fluid cells whose 5-point stencil is entirely fluid (mask is 1 in the fluid)."""
    m = mask
    inner = m[:, :, 1:-1, 1:-1] * m[:, :, :-2, 1:-1] * m[:, :, 2:, 1:-1] * m[:, :, 1:-1, :-2] * m[:, :, 1:-1, 2:]
    return inner


def divergence(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """du/dx + dv/dy with central differences on the interior, shape (B,1,ny-2,nx-2)."""
    du = 0.5 * (u[:, :, 1:-1, 2:] - u[:, :, 1:-1, :-2])
    dv = 0.5 * (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1])
    return du + dv


def continuity_residual(fields: torch.Tensor, mask: torch.Tensor, u_ref: float, l_ref: float) -> torch.Tensor:
    """Mean |div u| L / u, a dimensionless measure of how badly mass is conserved."""
    div = divergence(fields[:, 0:1], fields[:, 1:2])
    w = interior_fluid(mask)
    return (div.abs() * w).sum(dim=(1, 2, 3)) / w.sum(dim=(1, 2, 3)).clamp_min(1.0) * l_ref / u_ref


def r2_score(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def summarise(
    pred_fields: torch.Tensor,
    true_fields: torch.Tensor,
    mask: torch.Tensor,
    pred_cd: np.ndarray,
    true_cd: np.ndarray,
    u_ref: float,
    l_ref: float,
) -> dict[str, float]:
    """All metrics for one split, in physical (de-normalised) units."""
    rel = masked_relative_l2(pred_fields, true_fields, mask)
    out = {
        "rel_l2_u": float(rel[:, 0].mean()),
        "rel_l2_v": float(rel[:, 1].mean()),
        "rel_l2_p": float(rel[:, 2].mean()),
        "rel_l2_mean": float(rel.mean()),
        "cd_mae": float(np.abs(pred_cd - true_cd).mean()),
        "cd_mape_pct": float(100.0 * np.abs((pred_cd - true_cd) / true_cd).mean()),
        "cd_max_err_pct": float(100.0 * np.abs((pred_cd - true_cd) / true_cd).max()),
        "cd_r2": r2_score(pred_cd, true_cd),
        "div_residual": float(continuity_residual(pred_fields, mask, u_ref, l_ref).mean()),
        "div_residual_ref": float(continuity_residual(true_fields, mask, u_ref, l_ref).mean()),
    }
    return out
