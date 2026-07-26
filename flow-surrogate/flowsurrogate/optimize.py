"""Shape optimisation by back-propagating through the surrogate.

The design vector is the 6 Fourier coefficients; the scale r0 is *derived* from them
so that the enclosed area is exactly constant, which turns a constrained problem into
an unconstrained one (no penalty weight to tune, no infeasible iterates).

The one non-obvious ingredient is the occupancy channel.  The networks are trained on
a hard mask, which has zero gradient w.r.t. the shape everywhere.  At optimisation
time it is replaced by sigmoid(-phi / eps) with eps below one cell: numerically the
same image, but the boundary now moves when the coefficients move.

Because a surrogate optimised against itself will happily walk into its own error, the
optimum is always re-run through the LBM (`verify`), and `run_active_learning` closes
the loop by fine-tuning on those verified points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .data import RE_SCALE, SDF_SCALE, Normalizer
from .geometry import (
    Domain,
    N_COEFFS,
    area,
    equivalent_diameter,
    level_set_torch,
    r0_for_area,
    r0_for_area_torch,
    radius,
    sample_shapes,
    solid_mask,
)
from .lbm import LBMSolver, SolverConfig


@dataclass
class OptConfig:
    n_starts: int = 16
    iters: int = 250
    lr: float = 8e-3
    reynolds: float = 80.0
    target_area: float | None = None  # defaults to the median training area
    coeff_bound: float = 0.16  # stay inside the training box
    mask_eps: float = 0.7  # cells; smoothing of the occupancy channel
    seed: int = 0
    verify_every: int = 0  # 0 = only at the end
    model_kwargs: dict = field(default_factory=dict)


def render_inputs(
    coeffs: torch.Tensor,
    target_area: float,
    reynolds: float,
    domain: Domain,
    mask_eps: float,
) -> torch.Tensor:
    """Differentiable version of `data.input_channels`."""
    r0 = r0_for_area_torch(coeffs, target_area).unsqueeze(1)
    params = torch.cat([r0, coeffs], dim=1)
    phi = level_set_torch(params, domain, device=coeffs.device, dtype=coeffs.dtype)
    sdf = torch.tanh(phi / SDF_SCALE)
    occ = torch.sigmoid(-phi / mask_eps)
    re = torch.full_like(sdf, reynolds / RE_SCALE)
    return torch.stack([sdf, occ, re], dim=1)


def to_params(coeffs: np.ndarray, target_area: float) -> np.ndarray:
    """Full shape vector [r0, a_2, b_2, ...] for a set of coefficients."""
    out = np.empty((coeffs.shape[0], N_COEFFS + 1))
    for i, c in enumerate(coeffs):
        out[i] = np.concatenate([[r0_for_area(c, target_area)], c])
    return out


@torch.no_grad()
def _surrogate_cd(model, x: torch.Tensor, norm: Normalizer) -> np.ndarray:
    return norm.decode_cd(model(x)[1]).cpu().numpy()


def optimise(
    model: torch.nn.Module,
    norm: Normalizer,
    domain: Domain,
    cfg: OptConfig,
    device: torch.device,
    init_coeffs: np.ndarray | None = None,
) -> dict:
    """Projected Adam on the shape coefficients.  Returns history + final designs."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    if init_coeffs is None:
        init = sample_shapes(rng, cfg.n_starts, coeff_range=cfg.coeff_bound * 0.6)[:, 1:]
    else:
        init = init_coeffs
    target_area = cfg.target_area
    assert target_area is not None

    coeffs = torch.tensor(init, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([coeffs], lr=cfg.lr)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    history = {"cd_mean": [], "cd_best": []}
    for it in range(cfg.iters):
        opt.zero_grad(set_to_none=True)
        x = render_inputs(coeffs, target_area, cfg.reynolds, domain, cfg.mask_eps)
        _, cd_norm = model(x)
        cd = norm.decode_cd(cd_norm)
        cd.sum().backward()
        opt.step()
        with torch.no_grad():
            coeffs.clamp_(-cfg.coeff_bound, cfg.coeff_bound)
        history["cd_mean"].append(float(cd.mean().detach()))
        history["cd_best"].append(float(cd.min().detach()))

    with torch.no_grad():
        x = render_inputs(coeffs, target_area, cfg.reynolds, domain, cfg.mask_eps)
        cd_final = norm.decode_cd(model(x)[1]).cpu().numpy()
    for p in model.parameters():
        p.requires_grad_(True)

    return {
        "history": history,
        "init_coeffs": init,
        "final_coeffs": coeffs.detach().cpu().numpy(),
        "cd_surrogate_final": cd_final,
        "target_area": target_area,
    }


def verify(
    coeffs: np.ndarray,
    target_area: float,
    reynolds: float,
    domain: Domain,
    solver_cfg: SolverConfig,
    device: str = "auto",
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Run the real solver on a set of designs.  Returns (cd, fields, params)."""
    params = to_params(coeffs, target_area)
    masks = np.stack([solid_mask(p, domain) for p in params])
    d_eq = np.array([equivalent_diameter(p) for p in params])
    solver = LBMSolver(domain, solver_cfg, device=device)
    out = solver.run(masks, np.full(len(params), reynolds), d_eq)
    return out["cd"], out, params


def boundary_xy(params: np.ndarray, n: int = 400) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = radius(theta, params)
    return np.stack([r * np.cos(theta), r * np.sin(theta)])
