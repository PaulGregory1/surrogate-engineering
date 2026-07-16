"""Batched D2Q9 lattice-Boltzmann solver (BGK), written in Torch.

This is the "expensive model" the rest of the repo tries to replace.  It is a plain
single-relaxation-time LBM with

  * bounce-back on the obstacle (no-slip),
  * specular (free-slip) reflection on the channel walls, so no wall boundary layer
    contaminates the wake,
  * equilibrium Dirichlet inlet, zero-gradient outlet,
  * drag/lift from the momentum-exchange method on the bounce-back links.

Two things make it fast enough to build a dataset on a laptop:

1. The batch dimension is a *design* dimension -- B independent geometries are
   marched in lock-step, which turns a latency-bound stencil into a
   bandwidth-bound one (~3x per-sample speed-up at B >= 32).
2. The whole time step is a single `torch.compile` region.  Eager mode issues ~40
   separate kernels per step over a 40 MB state; fused, the step is ~10x cheaper on
   an M-series GPU.  Set ``compile=False`` to fall back to eager.

Lattice units throughout (dx = dt = 1).  See `scripts/01_validate_solver.py` for the
verification against published cylinder drag data.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

import numpy as np
import torch

from .geometry import Domain

# D2Q9 lattice: rest, 4 axial, 4 diagonal.
C = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)], dtype=np.int64)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
# Specular reflection about a horizontal wall (flip the y-component of c).
MIRROR_Y = np.array([0, 1, 4, 3, 2, 8, 7, 6, 5])
# Populations that are unknown at the bottom / top wall row after streaming.
UP_UNKNOWN = np.array([False, False, True, False, False, True, True, False, False])
DOWN_UNKNOWN = np.array([False, False, False, False, True, False, False, True, True])

CS2 = 1.0 / 3.0  # lattice speed of sound squared


@dataclass(frozen=True)
class SolverConfig:
    """Everything that fixes the cost/accuracy trade-off of the solver."""

    u_in: float = 0.08  # inlet velocity (Mach ~ 0.14, safely incompressible)
    n_steps: int = 9000  # total steps; ~3.7 domain crossings
    n_avg: int = 2500  # trailing window used for time-averaging
    dtype: torch.dtype = torch.float32


def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, c: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Second-order Maxwellian, shape (B, 9, ny, nx)."""
    cu = c[:, 0].view(1, 9, 1, 1) * ux.unsqueeze(1) + c[:, 1].view(1, 9, 1, 1) * uy.unsqueeze(1)
    usq = (ux**2 + uy**2).unsqueeze(1)
    return w.view(1, 9, 1, 1) * rho.unsqueeze(1) * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usq)


class LBMSolver:
    """Marches a batch of geometries to a (statistically) steady state."""

    def __init__(
        self,
        domain: Domain,
        config: SolverConfig | None = None,
        device: str = "auto",
        compile: bool = True,
    ):
        self.domain = domain
        self.cfg = config or SolverConfig()
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        dt = self.cfg.dtype
        self.c = torch.tensor(C, device=self.device, dtype=dt)
        self.w = torch.tensor(W, device=self.device, dtype=dt)
        self.opp = torch.tensor(OPP, device=self.device)
        self.mirror = torch.tensor(MIRROR_Y, device=self.device)
        self.up_unknown = torch.tensor(UP_UNKNOWN, device=self.device).view(1, 9, 1, 1)
        self.down_unknown = torch.tensor(DOWN_UNKNOWN, device=self.device).view(1, 9, 1, 1)
        self._step_fn = torch.compile(self._step, dynamic=False) if compile else self._step

    # ---------------------------------------------------------------- one time step
    def _step(
        self,
        f: torch.Tensor,
        solid: torch.Tensor,
        fluid: torch.Tensor,
        tau: torch.Tensor,
        links: torch.Tensor,
        f_in: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cx = self.c[:, 0].view(1, 9, 1, 1)
        cy = self.c[:, 1].view(1, 9, 1, 1)

        rho = f.sum(dim=1)
        ux = (f * cx).sum(dim=1) / rho * fluid
        uy = (f * cy).sum(dim=1) / rho * fluid

        feq = equilibrium(rho, ux, uy, self.c, self.w)
        f_post = f - (f - feq) / tau  # BGK collision

        # Momentum exchange: every blocked link hands 2 c_i f_i^post to the body.
        mom = f_post * links
        fx = 2.0 * (mom * cx).sum(dim=(1, 2, 3))
        fy = 2.0 * (mom * cy).sum(dim=(1, 2, 3))

        # Bounce-back on the node: solid nodes skip the collision entirely and simply
        # reverse the populations that streamed into them.  Colliding inside the body
        # would relax the stored populations toward a rest equilibrium and corrupt
        # both the no-slip condition and the drag.
        #
        # The direction permutations below are written as constant slices rather than
        # `f[:, index_tensor]`: a gather over a 40 MB state costs ~5 ms on MPS, while
        # constant slices fuse into the surrounding elementwise kernels.
        fb = torch.where(solid, torch.cat([f[:, i : i + 1] for i in OPP], dim=1), f_post)

        # Streaming.
        parts = [fb[:, :1]]
        for i in range(1, 9):
            parts.append(torch.roll(fb[:, i : i + 1], shifts=(int(C[i, 1]), int(C[i, 0])), dims=(2, 3)))
        fs = torch.cat(parts, dim=1)

        # Free-slip walls: only the populations that would have streamed in from
        # outside the domain are unknown; replace them by their specular partner.
        # Only the two wall rows are touched, so the mirror permutation is applied
        # after slicing.
        mirror_top = torch.cat([fs[:, i : i + 1, :1, :] for i in MIRROR_Y], dim=1)
        mirror_bot = torch.cat([fs[:, i : i + 1, -1:, :] for i in MIRROR_Y], dim=1)
        row0 = torch.where(self.up_unknown, mirror_top, fs[:, :, :1, :])
        rown = torch.where(self.down_unknown, mirror_bot, fs[:, :, -1:, :])
        fs = torch.cat([row0, fs[:, :, 1:-1, :], rown], dim=2)

        # Inlet (Dirichlet equilibrium) and outlet (zero gradient).
        fs = torch.cat([f_in, fs[:, :, :, 1:-1], fs[:, :, :, -2:-1]], dim=3)
        return fs, ux, uy, rho, fx, fy

    # -------------------------------------------------------------------- public
    @torch.no_grad()
    def run(
        self,
        masks: np.ndarray,
        reynolds: np.ndarray,
        diameters: np.ndarray,
        progress: bool = False,
    ) -> dict[str, np.ndarray]:
        """Solve a batch of cases.

        Args:
            masks: (B, ny, nx) bool, True inside the obstacle.
            reynolds: (B,) Reynolds numbers, Re = u_in * D_eq / nu.
            diameters: (B,) equivalent diameters D_eq, used for nu and for C_d.

        Returns a dict of NumPy arrays with the time-averaged fields ``u``, ``v``,
        ``p``, the force coefficients ``cd``/``cl``, and ``cd_drift`` -- the relative
        change of C_d between the two halves of the averaging window, which is the
        convergence flag used to filter the dataset.
        """
        cfg, dev, dt = self.cfg, self.device, self.cfg.dtype
        nx, ny = self.domain.nx, self.domain.ny
        b = masks.shape[0]

        solid = torch.as_tensor(np.ascontiguousarray(masks), device=dev).unsqueeze(1)
        fluid = (~solid)[:, 0].to(dt)
        nu = torch.as_tensor(cfg.u_in * diameters / reynolds, device=dev, dtype=dt)
        tau = (3.0 * nu + 0.5).view(b, 1, 1, 1)
        d_eq = torch.as_tensor(diameters, device=dev, dtype=dt)

        # Bounce-back links for the momentum exchange: fluid node x with x + c_i solid.
        links = torch.zeros((b, 9, ny, nx), device=dev, dtype=dt)
        for i in range(1, 9):
            nbr_solid = torch.roll(solid[:, 0], shifts=(-int(C[i, 1]), -int(C[i, 0])), dims=(1, 2))
            links[:, i] = ((~solid[:, 0]) & nbr_solid).to(dt)

        rho = torch.ones((b, ny, nx), device=dev, dtype=dt)
        ux = torch.full((b, ny, nx), cfg.u_in, device=dev, dtype=dt) * fluid
        uy = torch.zeros((b, ny, nx), device=dev, dtype=dt)
        f = equilibrium(rho, ux, uy, self.c, self.w)
        f_in = equilibrium(
            torch.ones((b, ny, 1), device=dev, dtype=dt),
            torch.full((b, ny, 1), cfg.u_in, device=dev, dtype=dt),
            torch.zeros((b, ny, 1), device=dev, dtype=dt),
            self.c,
            self.w,
        )

        acc_u = torch.zeros((b, ny, nx), device=dev, dtype=dt)
        acc_v = torch.zeros_like(acc_u)
        acc_p = torch.zeros_like(acc_u)
        fx_hist: list[torch.Tensor] = []
        fy_hist: list[torch.Tensor] = []
        start_avg = cfg.n_steps - cfg.n_avg

        it = range(cfg.n_steps)
        if progress:
            from tqdm import tqdm

            it = tqdm(it, desc=f"LBM x{b}", leave=False)

        for step in it:
            f, ux, uy, rho, fx, fy = self._step_fn(f, solid, fluid, tau, links, f_in)
            if step >= start_avg:
                acc_u += ux
                acc_v += uy
                acc_p += (rho - 1.0) * CS2
                fx_hist.append(fx)
                fy_hist.append(fy)

        n = float(cfg.n_avg)
        fx_all = torch.stack(fx_hist)  # (n_avg, B)
        fy_all = torch.stack(fy_hist)
        half = cfg.n_avg // 2
        cd_scale = 0.5 * cfg.u_in**2 * d_eq
        cd = fx_all.mean(dim=0) / cd_scale
        cl = fy_all.mean(dim=0) / cd_scale
        drift = (fx_all[half:].mean(dim=0) - fx_all[:half].mean(dim=0)).abs() / fx_all.mean(dim=0).abs().clamp_min(1e-12)

        to_np = lambda t: t.detach().float().cpu().numpy()
        return {
            "u": to_np(acc_u / n),
            "v": to_np(acc_v / n),
            "p": to_np(acc_p / n),
            "cd": to_np(cd),
            "cl": to_np(cl),
            "cd_drift": to_np(drift),
        }

    def benchmark(self, batch: int = 32, n_steps: int = 200) -> float:
        """Seconds per sample per 1000 steps -- used in the cost tables."""
        masks = np.zeros((batch, self.domain.ny, self.domain.nx), dtype=bool)
        masks[:, self.domain.ny // 2 - 5 : self.domain.ny // 2 + 5, 50:60] = True
        probe = LBMSolver(
            self.domain,
            dataclasses.replace(self.cfg, n_steps=n_steps, n_avg=max(2, n_steps // 4)),
            device=str(self.device),
            compile=self._step_fn is not self._step,
        )
        re, d = np.full(batch, 60.0), np.full(batch, 10.0)
        probe.run(masks, re, d)  # warm-up (includes compilation)
        t0 = time.perf_counter()
        probe.run(masks, re, d)
        if self.device.type == "mps":
            torch.mps.synchronize()
        return (time.perf_counter() - t0) / batch / (n_steps / 1000.0)
