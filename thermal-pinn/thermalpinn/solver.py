"""The truth generator: 1-D nonlinear heat conduction, Crank-Nicolson + Newton.

    rho c(T) dT/dt = d/dx ( k(T) dT/dx )                      0 < x < L
    -k(T) dT/dx |_0 = q(t) - h (Ts - Tinf) - eps sigma (Ts^4 - Tinf^4)
     dT/dx |_L = 0
     T(x, 0) = T0

Three choices here are load-bearing and none of them is cosmetic.

*Finite volume, vertex centred.*  Nodes carry the unknowns and each owns a control
volume (half-width at the two boundaries).  Fluxes are computed once per face and
subtracted, so the discrete energy balance telescopes exactly -- that is what makes the
1e-8 closure of M1(c) a real check on the boundary condition rather than a formality.
Vertex centred rather than cell centred because x = 0 and x = x_s are then *unknowns*,
not extrapolations, and the surface temperature is the whole point of the repo.

*Enthalpy, not rho c dT/dt.*  The accumulation term advances H(T) = int c dT', which is
known in closed form.  With c depending on temperature this is what keeps conservation
exact instead of merely second order.

*Rannacher startup.*  Crank-Nicolson rings on a step input, and 5 MW/m^2 switching on at
t = 0 is exactly that.  The first few steps are therefore split into pairs of damped
backward-Euler half steps, which kills the oscillation without costing an order -- the
half steps span the same dt, so their local error stays O(dt^2) with a small constant.

The truth is generated on `TRUTH_GRID` and every inverse method works on `INVERSE_GRID`.
That gap is deliberate: sharing a discretisation with the truth is the inverse crime, and
results obtained that way measure the grid rather than the method (see the plan, 8.1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import get_lapack_funcs

from .material import SIGMA_SB, Material
from .scenario import Scenario

FluxFn = Callable[[float], float]
SourceFn = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class Grid:
    """Space-time discretisation, kept separate from `Scenario` so that the truth and
    the inverse model can differ while describing the same physics."""

    nx: int
    dt: float

    def nodes(self, L: float) -> np.ndarray:
        return np.linspace(0.0, L, self.nx)

    def spacing(self, L: float) -> float:
        return L / (self.nx - 1)

    def volumes(self, L: float) -> np.ndarray:
        """Control volume per node, m.  Half cells at both ends."""
        v = np.full(self.nx, self.spacing(L))
        v[0] *= 0.5
        v[-1] *= 0.5
        return v


#: Grid the truth is generated on.
TRUTH_GRID = Grid(nx=801, dt=1.0e-4)
#: Grid every inverse method is allowed to use.  Coarser on purpose.
INVERSE_GRID = Grid(nx=201, dt=1.0e-3)


@dataclass
class Solution:
    """Output of `forward`.

    The full field is subsampled in time (it is 800 x 200 000 otherwise), but the probe
    traces -- surface and thermocouple -- are kept at full temporal resolution, because
    those are what everything downstream consumes.
    """

    x: np.ndarray  # (nx,)
    t: np.ndarray  # (nt,)
    probes: np.ndarray  # (n_probe,)
    T_probe: np.ndarray  # (nt, n_probe)
    q_net: np.ndarray  # (nt,) flux actually entering the solid at x = 0
    t_field: np.ndarray  # (n_snap,)
    T_field: np.ndarray  # (n_snap, nx)
    energy: np.ndarray  # (nt,) stored enthalpy per unit area, J/m^2
    influx: np.ndarray  # (nt,) cumulative energy in, same quadrature as the scheme
    newton_iterations: int
    seconds: float

    def at(self, x: float) -> np.ndarray:
        """Temperature trace at one of the requested probe positions."""
        idx = int(np.argmin(np.abs(self.probes - x)))
        if abs(self.probes[idx] - x) > 1e-12:
            raise KeyError(f"x = {x} is not a probe of this solution ({self.probes})")
        return self.T_probe[:, idx]

    @property
    def T_surface(self) -> np.ndarray:
        return self.at(0.0)

    @property
    def energy_residual(self) -> float:
        """Relative closure of the discrete energy balance.

        Exact to round-off by construction of the scheme; a nonzero value means the
        boundary condition, the control volumes or the enthalpy are wrong somewhere.
        """
        stored = self.energy[-1] - self.energy[0]
        scale = max(abs(self.influx[-1]), abs(stored), 1e-30)
        return float(abs(stored - self.influx[-1]) / scale)


def forward(
    scenario: Scenario,
    material: Material,
    q: FluxFn | np.ndarray | float,
    *,
    grid: Grid = TRUTH_GRID,
    t_end: float | None = None,
    probes: Sequence[float] | None = None,
    source: SourceFn | None = None,
    T_init: np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
    n_snapshots: int = 201,
    theta: float = 0.5,
    n_startup: int = 2,
    newton_tol: float = 1e-10,
    newton_max: int = 40,
) -> Solution:
    """Integrate the heat equation forward in time.

    `q` is the *applied* surface flux in W/m^2: a callable of t, an array on the time
    grid, or a constant.  Convection and radiation are subtracted from it by the
    boundary condition, so `Solution.q_net` is generally smaller than `q`.

    `source` (W/m^3) and `T_init` exist for the method of manufactured solutions and are
    not used by the physical problem.
    """
    L = scenario.L
    x = grid.nodes(L)
    dx = grid.spacing(L)
    vol = grid.volumes(L)
    rho = material.rho
    eps_sb = (scenario.eps * SIGMA_SB) if material.radiation else 0.0
    h, T_inf = scenario.h, scenario.T_inf

    t_end = scenario.t_end if t_end is None else t_end
    n_steps = int(round(t_end / grid.dt))
    if abs(n_steps * grid.dt - t_end) > 1e-9 * max(t_end, 1.0):
        raise ValueError(f"t_end = {t_end} is not a whole number of dt = {grid.dt} steps")
    t = np.arange(n_steps + 1, dtype=float) * grid.dt

    # The applied flux only ever needs to be known at the time levels, so evaluate it
    # once here rather than inside the Newton loop.
    if callable(q):
        q_applied = np.array([float(q(tt)) for tt in t])
    else:
        q_applied = np.broadcast_to(np.asarray(q, dtype=float), (n_steps + 1,)).astype(float)

    probes = np.asarray(probes if probes is not None else (0.0, scenario.x_s), dtype=float)
    lo = np.clip(np.searchsorted(x, probes, side="right") - 1, 0, grid.nx - 2)
    w = (probes - x[lo]) / dx  # linear interpolation onto the probe positions

    if T_init is None:
        T = np.full(grid.nx, float(scenario.T0))
    elif callable(T_init):
        T = np.asarray(T_init(x), dtype=float).copy()
    else:
        T = np.asarray(T_init, dtype=float).copy()

    T_probe = np.empty((n_steps + 1, probes.size))
    q_net = np.empty(n_steps + 1)
    energy = np.empty(n_steps + 1)
    influx = np.zeros(n_steps + 1)
    keep = max(1, n_steps // max(n_snapshots - 1, 1))
    snap_idx = list(range(0, n_steps + 1, keep))
    if snap_idx[-1] != n_steps:
        snap_idx.append(n_steps)
    T_field = np.empty((len(snap_idx), grid.nx))
    snap_at = {n: j for j, n in enumerate(snap_idx)}

    def probe_values(state: np.ndarray) -> np.ndarray:
        return state[lo] * (1.0 - w) + state[lo + 1] * w

    def surface_flux(Ts: float, q_app: float) -> float:
        """Net flux entering the solid at x = 0, once the surface has lost its share."""
        return q_app - h * (Ts - T_inf) - eps_sb * (Ts**4 - T_inf**4)

    def divergence(state: np.ndarray, q_app: float) -> tuple[np.ndarray, np.ndarray]:
        """Net heat into each control volume, W/m^2, plus the face conductivities."""
        kn = material.k(state)
        k_face = 0.5 * (kn[:-1] + kn[1:])
        flux = -k_face * (state[1:] - state[:-1]) / dx  # F_{i+1/2}, positive towards +x
        div = np.empty_like(state)
        div[0] = surface_flux(state[0], q_app) - flux[0]
        div[1:-1] = flux[:-1] - flux[1:]
        div[-1] = flux[-1]
        return div, k_face

    zero_source = np.zeros(grid.nx)

    def volumetric(tt: float) -> np.ndarray:
        if source is None:
            return zero_source
        return vol * np.asarray(source(x, tt), dtype=float)

    dt = grid.dt
    stats = {"newton": 0}
    # The Jacobian is tridiagonal and small, so the LAPACK call is bare: solve_banded's
    # validation costs more than the factorisation itself at this size.
    (gtsv,) = get_lapack_funcs(("gtsv",), (T,))

    def advance(T_old, div_old, dt_sub, th, q_old, q_new, src0, src1):
        """One theta-step of size `dt_sub`.

        `div_old` is the heat divergence at the starting state, which the previous step
        already computed on its way out; returning it saves recomputing k(T) every step.
        Returns (T_new, div_new, q_net_new, energy in).
        """
        H_old = material.enthalpy(T_old)
        rhs_old = (1.0 - th) * dt_sub * (div_old + src0)

        T_new = T_old.copy()
        for _ in range(newton_max):
            div_new, k_face = divergence(T_new, q_new)
            cap = rho * vol * material.c(T_new)
            residual = rho * vol * (material.enthalpy(T_new) - H_old) - th * dt_sub * (div_new + src1) - rhs_old
            if np.max(np.abs(residual) / cap) < newton_tol:
                break
            stats["newton"] += 1

            # d(face flux)/dT of the two nodes the face connects
            dk = material.dk_dT(T_new)
            grad = (T_new[1:] - T_new[:-1]) / dx
            a = k_face / dx - 0.5 * dk[:-1] * grad  # dF_i / dT_i
            b = -k_face / dx - 0.5 * dk[1:] * grad  # dF_i / dT_{i+1}

            main = np.empty(grid.nx)
            main[0] = (-h - 4.0 * eps_sb * T_new[0] ** 3) - a[0]
            main[1:-1] = b[:-1] - a[1:]
            main[-1] = b[-1]

            step = gtsv(-th * dt_sub * a, cap - th * dt_sub * main, th * dt_sub * b, -residual)[3]
            np.clip(step, -1000.0, 1000.0, out=step)  # keep Newton on the rails
            T_new += step
        else:
            raise RuntimeError("Newton did not converge")

        # Same quadrature as the scheme, so the energy balance is a statement about the
        # discretisation and not about how the influx was integrated afterwards.
        qn = surface_flux(T_new[0], q_new)
        gained = dt_sub * (
            th * (qn + float(src1.sum())) + (1.0 - th) * (surface_flux(T_old[0], q_old) + float(src0.sum()))
        )
        return T_new, div_new, qn, gained

    T_probe[0] = probe_values(T)
    q_net[0] = surface_flux(T[0], q_applied[0])
    energy[0] = rho * float(vol @ material.enthalpy(T))
    T_field[0] = T
    t_start = time.perf_counter()

    src_old = volumetric(t[0])
    div_old, _ = divergence(T, q_applied[0])
    for n in range(n_steps):
        src_new = volumetric(t[n + 1])
        if n < n_startup:
            # Rannacher: split the step into two damped backward-Euler half steps.  They
            # span the same dt, so the injected error stays O(dt^2) with a small constant
            # -- full-size Euler steps would also be O(dt^2) but wreck the observed order
            # long before the asymptotic range is reached.
            tm = t[n] + 0.5 * dt
            q_mid = float(q(tm)) if callable(q) else float(np.interp(tm, t, q_applied))
            src_mid = volumetric(tm)
            T_mid, div_mid, _, gained_a = advance(T, div_old, 0.5 * dt, 1.0, q_applied[n], q_mid, src_old, src_mid)
            T_new, div_new, q1, gained_b = advance(
                T_mid, div_mid, 0.5 * dt, 1.0, q_mid, q_applied[n + 1], src_mid, src_new
            )
            gained = gained_a + gained_b
        else:
            T_new, div_new, q1, gained = advance(
                T, div_old, dt, theta, q_applied[n], q_applied[n + 1], src_old, src_new
            )

        influx[n + 1] = influx[n] + gained
        T, src_old, div_old = T_new, src_new, div_new
        T_probe[n + 1] = probe_values(T)
        q_net[n + 1] = q1
        energy[n + 1] = rho * float(vol @ material.enthalpy(T))
        if n + 1 in snap_at:
            T_field[snap_at[n + 1]] = T

    return Solution(
        x=x,
        t=t,
        probes=probes,
        T_probe=T_probe,
        q_net=q_net,
        t_field=t[snap_idx],
        T_field=T_field,
        energy=energy,
        influx=influx,
        newton_iterations=stats["newton"],
        seconds=time.perf_counter() - t_start,
    )


def sensor_signal(
    solution: Solution,
    scenario: Scenario,
    *,
    rng: np.random.Generator | None = None,
    sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample the buried thermocouple at its own rate and add measurement noise.

    Returns (times, noisy, clean).  The clean trace is kept because several of the
    honesty checks -- the inverse-crime test in particular -- need a noise-free signal.
    """
    rng = np.random.default_rng(scenario.noise_seed) if rng is None else rng
    sigma = scenario.sigma_noise if sigma is None else sigma
    ts = scenario.sensor_times()
    clean = np.interp(ts, solution.t, solution.at(scenario.x_s))
    return ts, clean + rng.normal(0.0, sigma, ts.shape), clean
