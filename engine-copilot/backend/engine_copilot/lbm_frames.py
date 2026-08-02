"""Instrumented LBM run that yields intermediate frames instead of only the
time-averaged final state.

`LBMSolver.run()` (flow-surrogate, untouched) only returns the trailing time-average
once the whole march is done -- exactly what the dataset generator and `verify()`
want, but useless for an animation of the wake developing. This module duplicates
the setup section of `run()` (init populations, bounce-back links, tau) and drives
`solver._step_fn` in a plain Python loop instead, extracting the instantaneous
macroscopic state every `steps // n_frames` steps. It shares one compiled solver
instance (module-level singleton, batch=1) with `flow_service.verify()` so the
`torch.compile` warm-up cost is paid once per process, not once per request.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from flowsurrogate.geometry import Domain, equivalent_diameter, solid_mask
from flowsurrogate.lbm import C, LBMSolver, SolverConfig, equilibrium
from flowsurrogate.optimize import to_params

from . import config
from .device_lock import MPS_LOCK
from .flow_service import downscale

_solver: LBMSolver | None = None


def get_solver(domain: Domain, device: torch.device) -> LBMSolver:
    """Process-wide singleton LBM solver (batch=1) shared by simulate + verify."""
    global _solver
    if _solver is None:
        _solver = LBMSolver(domain, SolverConfig(), device=str(device))
    return _solver


def _vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """omega = dv/dx - du/dy, central differences in lattice units (dx = 1)."""
    dvdx = np.gradient(v, axis=1)
    dudy = np.gradient(u, axis=0)
    return dvdx - dudy


def simulate_streaming(
    domain: Domain,
    device: torch.device,
    coeffs: list[float],
    target_area: float,
    reynolds: float,
    n_frames: int = 60,
    steps: int = 6000,
) -> Iterator[dict]:
    """Yields ~n_frames dicts of instantaneous (u, v, vorticity), then a done event."""
    solver = get_solver(domain, device)
    dt = solver.cfg.dtype
    dev = solver.device
    ny, nx = domain.ny, domain.nx

    params = to_params(np.asarray([coeffs], dtype=np.float64), target_area)
    mask_np = solid_mask(params[0], domain)[None, ...]  # (1, ny, nx)
    d_eq = float(equivalent_diameter(params[0]))

    solid = torch.as_tensor(np.ascontiguousarray(mask_np), device=dev).unsqueeze(1)
    fluid = (~solid)[:, 0].to(dt)
    nu = solver.cfg.u_in * d_eq / reynolds
    tau = torch.full((1, 1, 1, 1), 3.0 * nu + 0.5, device=dev, dtype=dt)

    links = torch.zeros((1, 9, ny, nx), device=dev, dtype=dt)
    for i in range(1, 9):
        nbr_solid = torch.roll(solid[:, 0], shifts=(-int(C[i, 1]), -int(C[i, 0])), dims=(1, 2))
        links[:, i] = ((~solid[:, 0]) & nbr_solid).to(dt)

    rho = torch.ones((1, ny, nx), device=dev, dtype=dt)
    ux = torch.full((1, ny, nx), solver.cfg.u_in, device=dev, dtype=dt) * fluid
    uy = torch.zeros((1, ny, nx), device=dev, dtype=dt)
    f = equilibrium(rho, ux, uy, solver.c, solver.w)
    f_in = equilibrium(
        torch.ones((1, ny, 1), device=dev, dtype=dt),
        torch.full((1, ny, 1), solver.cfg.u_in, device=dev, dtype=dt),
        torch.zeros((1, ny, 1), device=dev, dtype=dt),
        solver.c,
        solver.w,
    )

    frame_every = max(1, steps // n_frames)
    avg_window = max(200, steps // 3)
    start_avg = steps - avg_window
    cd_scale = 0.5 * solver.cfg.u_in**2 * d_eq
    fx_hist: list[float] = []
    frame_index = 0

    # `torch.no_grad()` must NOT span the `yield` below. This generator is resumed from a
    # thread pool (main.py drives it with run_in_executor), autograd's grad mode is
    # thread-local, and a generator suspended inside no_grad() hands its pool thread back
    # with autograd still switched off. Whatever runs on that thread next inherits it —
    # concretely, the Adam loop in flow_service.optimise_streaming then dies on
    # "element 0 of tensors does not require grad and does not have a grad_fn" as soon as
    # a user plays the live animation before clicking Optimiser.
    for step in range(steps):
        with torch.no_grad():
            with MPS_LOCK:
                f, ux, uy, rho, fx, fy = solver._step_fn(f, solid, fluid, tau, links, f_in)
            if step >= start_avg:
                fx_hist.append(float(fx.item()))
            frame_due = step % frame_every == 0 or step == steps - 1
            if frame_due:
                u_np = ux[0].detach().float().cpu().numpy()
                v_np = uy[0].detach().float().cpu().numpy()
        if frame_due:
            omega = _vorticity(u_np, v_np)
            yield {
                "t_step": step,
                "u": downscale(u_np, config.FIELD_DOWNSCALE).tolist(),
                "v": downscale(v_np, config.FIELD_DOWNSCALE).tolist(),
                "vorticity": downscale(omega, config.FIELD_DOWNSCALE).tolist(),
                "frame_index": frame_index,
            }
            frame_index += 1

    cd_final = float(np.mean(fx_hist) / cd_scale) if fx_hist else 0.0
    yield {"done": True, "cd_final": cd_final}
