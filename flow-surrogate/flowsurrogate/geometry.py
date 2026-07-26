"""Parametric star-shaped obstacles.

The obstacle boundary is a Fourier series in polar coordinates

    r(theta) = r0 * (1 + sum_k a_k cos(k theta) + b_k sin(k theta)),   k = 2..K+1

with the k = 1 modes deliberately excluded (they encode a rigid translation, which
is already fixed by the position of the body in the channel).

Everything here is written twice: a NumPy path used by the data generator, and a
Torch path that is differentiable w.r.t. the shape coefficients.  The Torch path is
what makes gradient-based shape optimisation through the surrogate possible
(`flowsurrogate.optimize`), so the two must agree -- `tests/test_geometry.py`
checks that they do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

#: Fourier modes used for the boundary (k = 2, 3, 4).
MODES: tuple[int, ...] = (2, 3, 4)
#: Number of harmonic coefficients: (a_k, b_k) per mode.
N_COEFFS: int = 2 * len(MODES)
#: Full shape vector is [r0, a_2, b_2, a_3, b_3, a_4, b_4].
N_SHAPE_PARAMS: int = 1 + N_COEFFS

#: Coefficients are sampled in [-COEFF_RANGE, COEFF_RANGE] ...
COEFF_RANGE: float = 0.16
#: ... and rejected if the shape gets too pinched (min radius factor).
MIN_RADIUS_FACTOR: float = 0.55


@dataclass(frozen=True)
class Domain:
    """Lattice geometry, in lattice units (1 unit = 1 cell)."""

    nx: int = 192
    ny: int = 96
    cx: float = 56.0  # obstacle centre
    cy: float = 48.0

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates, shape (ny, nx)."""
        x = np.arange(self.nx, dtype=np.float64) + 0.5
        y = np.arange(self.ny, dtype=np.float64) + 0.5
        return np.meshgrid(x, y)


# --------------------------------------------------------------------------------------
# NumPy path
# --------------------------------------------------------------------------------------


def radius(theta: np.ndarray, params: np.ndarray) -> np.ndarray:
    """r(theta) for one shape vector."""
    r0, coeffs = params[0], params[1:]
    shape_factor = np.ones_like(theta)
    for j, k in enumerate(MODES):
        shape_factor = shape_factor + coeffs[2 * j] * np.cos(k * theta) + coeffs[2 * j + 1] * np.sin(k * theta)
    return r0 * shape_factor


def level_set(params: np.ndarray, domain: Domain) -> np.ndarray:
    """Signed radial distance on the lattice: negative inside the body.

    This is a *radial* signed distance (rho - r(theta)), not a true Euclidean SDF.
    It is exact on the boundary and monotone across it, which is all the network
    needs, and it is cheap and analytically differentiable in the shape parameters.
    """
    x, y = domain.grid()
    dx, dy = x - domain.cx, y - domain.cy
    rho = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    return rho - radius(theta, params)


def solid_mask(params: np.ndarray, domain: Domain) -> np.ndarray:
    """Boolean occupancy of the body on the lattice."""
    return level_set(params, domain) < 0.0


def area(params: np.ndarray) -> float:
    """Enclosed area, in closed form: A = pi r0^2 (1 + 0.5 sum (a_k^2 + b_k^2))."""
    r0, coeffs = params[0], params[1:]
    return float(np.pi * r0**2 * (1.0 + 0.5 * np.sum(coeffs**2)))


def equivalent_diameter(params: np.ndarray) -> float:
    """Diameter of the disc with the same area -- the reference length for C_d."""
    return float(2.0 * np.sqrt(area(params) / np.pi))


def r0_for_area(coeffs: np.ndarray, target_area: float) -> float:
    """Scale factor that gives exactly `target_area` for the given harmonics.

    Used by the shape optimiser to enforce the area constraint by construction
    instead of by penalty.
    """
    return float(np.sqrt(target_area / (np.pi * (1.0 + 0.5 * np.sum(coeffs**2)))))


def is_valid(params: np.ndarray, n_theta: int = 512) -> bool:
    """Reject shapes that pinch (thin necks are badly resolved on a coarse lattice)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    r = radius(theta, params)
    return bool(r.min() > MIN_RADIUS_FACTOR * params[0])


def sample_shapes(
    rng: np.random.Generator,
    n: int,
    r0_range: tuple[float, float] = (8.5, 11.0),
    coeff_range: float = COEFF_RANGE,
) -> np.ndarray:
    """Draw `n` valid shape vectors, shape (n, N_SHAPE_PARAMS)."""
    out = np.empty((n, N_SHAPE_PARAMS))
    filled = 0
    while filled < n:
        r0 = rng.uniform(*r0_range)
        coeffs = rng.uniform(-coeff_range, coeff_range, size=N_COEFFS)
        cand = np.concatenate([[r0], coeffs])
        if is_valid(cand):
            out[filled] = cand
            filled += 1
    return out


# --------------------------------------------------------------------------------------
# Torch path (differentiable w.r.t. the shape vector)
# --------------------------------------------------------------------------------------


def level_set_torch(params: torch.Tensor, domain: Domain, device=None, dtype=torch.float32) -> torch.Tensor:
    """Batched differentiable level set, shape (B, ny, nx).

    `params` is (B, N_SHAPE_PARAMS) and requires_grad is honoured, so gradients of a
    downstream loss flow back to the Fourier coefficients.
    """
    device = device or params.device
    x = torch.arange(domain.nx, device=device, dtype=dtype) + 0.5
    y = torch.arange(domain.ny, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dx, dy = xx - domain.cx, yy - domain.cy
    rho = torch.sqrt(dx**2 + dy**2 + 1e-12)
    theta = torch.atan2(dy, dx)

    r0 = params[:, :1].unsqueeze(-1)  # (B, 1, 1)
    shape_factor = torch.ones_like(theta).unsqueeze(0).expand(params.shape[0], -1, -1).clone()
    for j, k in enumerate(MODES):
        a = params[:, 1 + 2 * j].view(-1, 1, 1)
        b = params[:, 2 + 2 * j].view(-1, 1, 1)
        shape_factor = shape_factor + a * torch.cos(k * theta) + b * torch.sin(k * theta)
    return rho.unsqueeze(0) - r0 * shape_factor


def area_torch(params: torch.Tensor) -> torch.Tensor:
    r0, coeffs = params[:, 0], params[:, 1:]
    return np.pi * r0**2 * (1.0 + 0.5 * (coeffs**2).sum(dim=1))


def r0_for_area_torch(coeffs: torch.Tensor, target_area: float) -> torch.Tensor:
    return torch.sqrt(target_area / (np.pi * (1.0 + 0.5 * (coeffs**2).sum(dim=1))))
