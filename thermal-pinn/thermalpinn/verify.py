"""Three independent ways of catching a wrong solver, before anything is built on it.

Every inverse result in this repo is a statement about a forward model.  If that model
is wrong the comparison between methods is meaningless, so the solver is checked against
things it cannot accidentally agree with:

    manufactured solutions   an exact answer the code has never seen, driving the *full*
                             nonlinear machinery -- variable k and c, radiation, the flux
                             boundary condition -- and measuring the observed order of
                             accuracy in space and in time.  Order 2 or the scheme is not
                             what it claims to be.
    semi-infinite erfc       the one closed-form solution of the real problem: constant
                             flux into a half space.  Checks the physics, not just the
                             self-consistency of the discretisation.
    energy balance           stored enthalpy minus what came through the boundary.  Zero
                             to round-off, or the boundary condition is wrong.

The three fail differently on purpose: MMS misses conservation errors, the energy balance
misses accuracy errors, and neither would notice a sign error in a physical constant that
erfc catches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

from .material import LAW_A, SIGMA_SB, T_REF, Material
from .scenario import Scenario
from .solver import TRUTH_GRID, Grid, forward

# --------------------------------------------------------------------------------------
# Manufactured solutions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Manufactured:
    """An exact solution the solver is forced to reproduce.

        T(x, t) = T_REF + A phi(t) psi(x)
        psi(x)  = 1 + 0.5 cos(pi x / L) + 0.5 ((x - L)/L)^2
        phi(t)  = 1 + 0.5 sin(2 pi t / period)

    `psi` is built so that `psi'(L) = 0` -- the adiabatic mid-plane is satisfied exactly,
    so the far boundary is tested as it is actually used rather than replaced by a
    Dirichlet condition.  `psi'(0) != 0`, so the flux boundary condition is genuinely
    exercised, and the cosine term keeps the fourth derivative nonzero, without which a
    second-order scheme would look artificially exact.

    With A = 200 K the solution sweeps 70..620 degC, which crosses law C's 450 degC knee:
    the Newton solve is tested where the conductivity actually bends.
    """

    L: float
    period: float = 0.4
    amplitude: float = 200.0

    def phi(self, t: float) -> float:
        return 1.0 + 0.5 * np.sin(2.0 * np.pi * t / self.period)

    def dphi(self, t: float) -> float:
        return (np.pi / self.period) * np.cos(2.0 * np.pi * t / self.period)

    def psi(self, x: np.ndarray) -> np.ndarray:
        return 1.0 + 0.5 * np.cos(np.pi * x / self.L) + 0.5 * ((x - self.L) / self.L) ** 2

    def dpsi(self, x: np.ndarray) -> np.ndarray:
        return -0.5 * (np.pi / self.L) * np.sin(np.pi * x / self.L) + (x - self.L) / self.L**2

    def d2psi(self, x: np.ndarray) -> np.ndarray:
        return -0.5 * (np.pi / self.L) ** 2 * np.cos(np.pi * x / self.L) + 1.0 / self.L**2

    def T(self, x: np.ndarray, t: float) -> np.ndarray:
        return T_REF + self.amplitude * self.phi(t) * self.psi(x)

    def source(self, material: Material, x: np.ndarray, t: float) -> np.ndarray:
        """The volumetric source that makes `T` an exact solution, W/m^3."""
        T = self.T(x, t)
        T_t = self.amplitude * self.dphi(t) * self.psi(x)
        T_x = self.amplitude * self.phi(t) * self.dpsi(x)
        T_xx = self.amplitude * self.phi(t) * self.d2psi(x)
        return material.rho * material.c(T) * T_t - (material.dk_dT(T) * T_x**2 + material.k(T) * T_xx)

    def flux(self, material: Material, scenario: Scenario, t: float) -> float:
        """The applied flux that makes the boundary condition consistent with `T`."""
        Ts = float(self.T(np.array(0.0), t))
        conducted = -material.k(Ts) * self.amplitude * self.phi(t) * float(self.dpsi(np.array(0.0)))
        losses = scenario.h * (Ts - scenario.T_inf)
        if material.radiation:
            losses += scenario.eps * SIGMA_SB * (Ts**4 - scenario.T_inf**4)
        return conducted + losses


def _run_manufactured(scenario: Scenario, material: Material, mms: Manufactured, grid: Grid, t_end: float):
    return forward(
        scenario,
        material,
        lambda tt: mms.flux(material, scenario, tt),
        grid=grid,
        t_end=t_end,
        source=lambda xx, tt: mms.source(material, xx, tt),
        T_init=lambda xx: mms.T(xx, 0.0),
        n_snapshots=2,
    )


def _norms(err: np.ndarray, vol: np.ndarray, L: float) -> tuple[float, float]:
    """Grid-independent L2 (volume weighted) and Linf norms, in kelvin."""
    return float(np.sqrt(vol @ err**2 / L)), float(np.max(np.abs(err)))


def _order(h: np.ndarray, e: np.ndarray) -> float:
    """Observed order of accuracy: slope of log(error) against log(step)."""
    if len(h) < 2:
        return float("nan")
    lh = np.log(np.asarray(h, dtype=float))
    return float(np.polyfit(lh - lh.mean(), np.log(np.asarray(e, dtype=float)), 1)[0])


def mms_convergence(
    material: Material,
    scenario: Scenario | None = None,
    *,
    t_end: float = 0.4,
    space_nx: tuple[int, ...] = (26, 51, 101, 201, 401),
    space_dt: float = 4.0e-5,
    time_nx: int = 201,
    time_steps: tuple[int, ...] = (800, 1600, 3200, 6400),
    time_ref_steps: int = 64000,
) -> dict:
    """Refinement study in space and in time on a manufactured solution.

    Space is measured against the exact solution with `space_dt` small enough that the
    temporal error is negligible (about 1 % of the error at the finest grid).  Time is
    measured against the *same spatial grid* run at `time_ref_steps`, which removes the
    spatial error from the comparison entirely -- otherwise a constant spatial offset
    would flatten the slope at small dt.

    The temporal window starts at dt = 5e-4 rather than at the coarsest dt that runs.  An
    observed order is an asymptotic statement, and below a few hundred steps the Rannacher
    startup -- itself O(dt^2), but with a much larger constant -- is still the dominant
    error, which shows up as a slope near 1.3 that climbs towards 2 as dt shrinks.
    Reporting that slope as "the order of the scheme" would understate a scheme that is
    demonstrably second order once the startup transient is no longer dominant.
    """
    scenario = Scenario() if scenario is None else scenario
    mms = Manufactured(L=scenario.L, period=t_end)

    space_rows = []
    for nx in space_nx:
        grid = Grid(nx=nx, dt=space_dt)
        sol = _run_manufactured(scenario, material, mms, grid, t_end)
        err = sol.T_field[-1] - mms.T(sol.x, t_end)
        l2, linf = _norms(err, grid.volumes(scenario.L), scenario.L)
        space_rows.append({"nx": nx, "dx": grid.spacing(scenario.L), "l2": l2, "linf": linf})

    time_rows = []
    if time_steps:
        ref_grid = Grid(nx=time_nx, dt=t_end / time_ref_steps)
        ref = _run_manufactured(scenario, material, mms, ref_grid, t_end).T_field[-1]
    for steps in time_steps:
        grid = Grid(nx=time_nx, dt=t_end / steps)
        sol = _run_manufactured(scenario, material, mms, grid, t_end)
        l2, linf = _norms(sol.T_field[-1] - ref, grid.volumes(scenario.L), scenario.L)
        time_rows.append({"n_steps": steps, "dt": grid.dt, "l2": l2, "linf": linf})

    return {
        "law": material.name,
        "t_end": t_end,
        "space": {
            "dt": space_dt,
            "rows": space_rows,
            "order_l2": _order([r["dx"] for r in space_rows], [r["l2"] for r in space_rows]),
            "order_linf": _order([r["dx"] for r in space_rows], [r["linf"] for r in space_rows]),
        },
        "time": {
            "nx": time_nx,
            "reference_steps": time_ref_steps,
            "rows": time_rows,
            "order_l2": _order([r["dt"] for r in time_rows], [r["l2"] for r in time_rows]),
            "order_linf": _order([r["dt"] for r in time_rows], [r["linf"] for r in time_rows]),
        },
    }


# --------------------------------------------------------------------------------------
# Semi-infinite half space, constant flux
# --------------------------------------------------------------------------------------


def semi_infinite(q0: float, material: Material, x: np.ndarray, t: float, T0: float) -> np.ndarray:
    """Carslaw & Jaeger: constant flux q0 into a half space at uniform T0.

        T(x, t) = T0 + (2 q0 / k) sqrt(alpha t) ierfc( x / (2 sqrt(alpha t)) )
        ierfc(z) = exp(-z^2)/sqrt(pi) - z erfc(z)
    """
    alpha = material.alpha0
    z = np.asarray(x, dtype=float) / (2.0 * np.sqrt(alpha * t))
    ierfc = np.exp(-(z**2)) / np.sqrt(np.pi) - z * erfc(z)
    return T0 + 2.0 * q0 / material.k0 * np.sqrt(alpha * t) * ierfc


def semi_infinite_check(
    *,
    material: Material = LAW_A,
    grid: Grid = TRUTH_GRID,
    q0: float = 1.0e6,
    times: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05),
    depths: tuple[float, ...] = (0.0, 2.5e-4, 5.0e-4, 1.0e-3),
    min_rise: float = 1.0,
) -> dict:
    """Compare the solver against the half-space solution while the wave is still far
    from the mid-plane.

    Convection and radiation are switched off, since the closed form has neither.  Points
    whose analytical rise is below `min_rise` kelvin are reported but excluded from the
    verdict: at 2 mm and 5 ms the true rise is microkelvins and a relative error there
    measures round-off, not the scheme.
    """
    scenario = Scenario(h=0.0, eps=0.0, t_end=max(times))
    sol = forward(scenario, material, q0, grid=grid, probes=depths, t_end=max(times))

    rows = []
    for t in times:
        n = int(round(t / grid.dt))
        exact = semi_infinite(q0, material, np.asarray(depths), t, scenario.T0)
        for d, num, ex in zip(depths, sol.T_probe[n], exact):
            rise = ex - scenario.T0
            rows.append(
                {
                    "t": t,
                    "x": d,
                    "rise_exact": float(rise),
                    "rise_numeric": float(num - scenario.T0),
                    "rel_error_pct": float(100.0 * (num - ex) / rise) if abs(rise) > 0 else float("nan"),
                    "counted": bool(rise >= min_rise),
                }
            )
    counted = [abs(r["rel_error_pct"]) for r in rows if r["counted"]]
    return {
        "q0": q0,
        "grid": {"nx": grid.nx, "dt": grid.dt},
        "min_rise_K": min_rise,
        "rows": rows,
        "max_abs_error_pct": max(counted) if counted else float("nan"),
        "n_counted": len(counted),
    }


# --------------------------------------------------------------------------------------
# Energy balance
# --------------------------------------------------------------------------------------


def energy_balance_check(material: Material, *, scenario: Scenario | None = None, grid: Grid = TRUTH_GRID) -> dict:
    """Run the real braking event and close the books on it.

    `residual_scheme` uses the quadrature the time integrator itself applies, so it is
    zero to round-off unless the control volumes, the enthalpy or the boundary flux
    disagree with each other.  `residual_trapezoid` re-integrates the boundary flux
    independently over the whole trace; it is not expected to vanish, only to be small,
    and the gap between the two is the backward-Euler startup.

    Law A closes at round-off because its system is linear and Newton solves it exactly.
    Laws B and C close near 4e-9 instead, and that floor is the Newton tolerance rather
    than the scheme: the residual Newton is allowed to leave *is* the per-cell energy
    defect, so 1e-10 K per cell accumulated over 2e5 steps lands exactly there.
    """
    scenario = Scenario() if scenario is None else scenario
    sol = forward(scenario, material, scenario.q_true, grid=grid)
    stored = float(sol.energy[-1] - sol.energy[0])
    trapezoid = float(np.sum(0.5 * (sol.q_net[1:] + sol.q_net[:-1]) * np.diff(sol.t)))
    scale = abs(float(sol.influx[-1]))
    return {
        "law": material.name,
        "grid": {"nx": grid.nx, "dt": grid.dt},
        "stored_J_per_m2": stored,
        "influx_J_per_m2": float(sol.influx[-1]),
        "residual_scheme": abs(stored - float(sol.influx[-1])) / scale,
        "residual_trapezoid": abs(stored - trapezoid) / scale,
        "T_surface_max": float(sol.T_surface.max()),
        "seconds": sol.seconds,
    }
