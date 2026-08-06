"""The classical inverse-heat-conduction theory, in the best version this repo can build.

The comparison the repo exists to make is only worth reading if the baselines are the real
thing, so all three are the textbook methods rather than sketches of them, and all three
are held to the same standard: the flux is estimated as a piecewise-constant history on the
measurement grid, from the same model, on the same coarse grid, with the same data.

    beck       sequential function specification.  Steps through time estimating one flux
               value at a time, and buys stability by assuming the flux stays constant over
               the next `r` measurements.  `r` is the regularisation parameter.
    tikhonov   whole-history deconvolution of the Duhamel sum, penalising the second
               difference of q.  `lambda` is chosen twice over, by the discrepancy
               principle and by the corner of the L-curve, and both are reported.
    alifanov   iterative regularisation: conjugate gradients on the least-squares
               functional, with the gradient from the adjoint problem and the iteration
               stopped by the discrepancy principle.  No parametrisation of q, no
               assumption of linearity -- this is the method the PINN has to beat.

`InverseModel` is what all three share, and it is deliberately *not* the truth generator.
It runs on the coarse grid (nx = 201, dt = 1e-3), it has no Rannacher startup, and it sees
the flux as a staircase on the 100 Hz measurement grid.  Sharing a discretisation with the
truth is the inverse crime and measures the grid instead of the method (see the plan, 8.1);
the one place the crime is committed on purpose is `inverse_crime_data`, which exists to
check that these implementations are correct at all.

Two design decisions are worth stating because they are what makes the adjoint trustworthy.

*The adjoint is discrete, not continuous.*  It differentiates the scheme that is actually
solved, step by transposed step, rather than the PDE the scheme approximates.  The
continuous adjoint would agree with finite differences only to discretisation error, a few
parts in 10^3 here; the discrete one agrees to round-off, which is why M3(b) can demand
10^-4 and get 10^-10.  It also means the tangent-linear model used for the CG step length
is the same set of matrices, transposed back.

*Superposition is used only where it exists.*  Beck and Tikhonov are built on the Duhamel
sum, so they exist for law A and, for laws B and C, only in the form industry actually uses
them -- properties frozen at a reference temperature.  That bias is measured in act 2
rather than argued about.  Alifanov needs none of this and handles the nonlinear law
exactly, which is why it is the baseline that matters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve, get_lapack_funcs, toeplitz

from .material import SIGMA_SB, Material
from .scenario import Scenario
from .solver import INVERSE_GRID, Grid


@dataclass
class Run:
    """One forward solve of the inverse model."""

    obs: np.ndarray  # (n_control + 1,) sensor temperature at the measurement times
    field: np.ndarray | None  # (M + 1, N) full history -- the adjoint needs every step
    newton_iterations: int
    seconds: float


# ======================================================================================
# The model all three baselines share
# ======================================================================================


@dataclass
class InverseModel:
    """Finite-volume theta-method heat conduction, plus its tangent and its adjoint.

    The unknown is `q_control`: one flux value per measurement interval, held constant
    across the ten model steps inside it.  Holding it constant over the whole step (rather
    than interpolating between time levels) is what makes the flux derivative exactly
    `-dt * psi_0` with no theta weighting, and it is the same staircase Beck estimates, so
    the three methods are solving for the same object.
    """

    scenario: Scenario
    material: Material
    grid: Grid = INVERSE_GRID
    theta: float = 0.5
    #: Backward-Euler steps before Crank-Nicolson takes over.  5 MW/m^2 switching on at
    #: t = 0 makes CN ring; damping the first 10 ms costs O(dt) over 1 % of the event.
    n_startup: int = 10
    newton_tol: float = 1e-11
    newton_max: int = 30

    x: np.ndarray = field(init=False)
    thetas: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        sc, grid = self.scenario, self.grid
        self.N = grid.nx
        self.dx = grid.spacing(sc.L)
        self.x = grid.nodes(sc.L)
        self.dt = grid.dt
        self.rho_vol = self.material.rho * grid.volumes(sc.L)

        self.n_control = int(round(sc.t_end * sc.f_sample))
        self.steps_per_control = int(round((1.0 / sc.f_sample) / self.dt))
        self.M = self.n_control * self.steps_per_control
        if abs(self.M * self.dt - sc.t_end) > 1e-9 * sc.t_end:
            raise ValueError("the measurement rate and the model step do not divide the event evenly")

        self.h = sc.h
        self.T_inf = sc.T_inf
        self.eps_sb = (sc.eps * SIGMA_SB) if self.material.radiation else 0.0

        # Sensor position, as linear interpolation between two nodes.
        self.i0 = min(int(np.searchsorted(self.x, sc.x_s, side="right")) - 1, self.N - 2)
        self.w = (sc.x_s - self.x[self.i0]) / self.dx

        self.thetas = np.full(self.M, self.theta)
        self.thetas[: self.n_startup] = 1.0
        (self._gtsv_fn,) = get_lapack_funcs(("gtsv",), (self.x,))

    # ---------------------------------------------------------------------------------
    # Spatial operator and its Jacobian
    # ---------------------------------------------------------------------------------

    def _solve_tri(self, sub: np.ndarray, main: np.ndarray, sup: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return self._gtsv_fn(sub, main, sup, rhs)[3]

    def _k_face(self, T: np.ndarray) -> np.ndarray:
        kn = self.material.k(T)
        return 0.5 * (kn[:-1] + kn[1:])

    def _conduction(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Net heat into each control volume from conduction and surface losses, W/m^2.

        The applied flux is *not* included: it enters the step as a constant over the
        step, which is what keeps its derivative clean.
        """
        k_face = self._k_face(T)
        flux = -k_face * (T[1:] - T[:-1]) / self.dx
        G = np.empty_like(T)
        G[0] = -self.h * (T[0] - self.T_inf) - self.eps_sb * (T[0] ** 4 - self.T_inf**4) - flux[0]
        G[1:-1] = flux[:-1] - flux[1:]
        G[-1] = flux[-1]
        return G, k_face

    def _jacobian(self, T: np.ndarray, k_face: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """dG/dT as (sub, main, sup) of a tridiagonal matrix.

        Same expression as the truth solver's Newton Jacobian, which is the check on it:
        that one is exercised by the manufactured solutions to second order, so an error
        here would already have shown up there.
        """
        dk = self.material.dk_dT(T)
        grad = (T[1:] - T[:-1]) / self.dx
        a = k_face / self.dx - 0.5 * dk[:-1] * grad  # dF_i / dT_i
        b = -k_face / self.dx - 0.5 * dk[1:] * grad  # dF_i / dT_{i+1}
        main = np.empty(self.N)
        main[0] = -self.h - 4.0 * self.eps_sb * T[0] ** 3 - a[0]
        main[1:-1] = b[:-1] - a[1:]
        main[-1] = b[-1]
        return a, main, -b

    @staticmethod
    def _matvec(sub: np.ndarray, main: np.ndarray, sup: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = main * v
        out[1:] += sub * v[:-1]
        out[:-1] += sup * v[1:]
        return out

    @staticmethod
    def _matvec_T(sub: np.ndarray, main: np.ndarray, sup: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = main * v
        out[1:] += sup * v[:-1]
        out[:-1] += sub * v[1:]
        return out

    def _probe(self, T: np.ndarray) -> float:
        return float(T[self.i0] * (1.0 - self.w) + T[self.i0 + 1] * self.w)

    # ---------------------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------------------

    def solve(self, q_control: np.ndarray, *, store: bool = True) -> Run:
        """Integrate forward with the flux staircase `q_control`."""
        q_steps = np.repeat(np.asarray(q_control, dtype=float), self.steps_per_control)
        if q_steps.size != self.M:
            raise ValueError(f"expected {self.n_control} control values, got {np.size(q_control)}")

        spc = self.steps_per_control
        T = np.full(self.N, float(self.scenario.T0))
        field_ = np.empty((self.M + 1, self.N)) if store else None
        if store:
            field_[0] = T
        obs = np.empty(self.n_control + 1)
        obs[0] = self._probe(T)
        newton = 0
        t0 = time.perf_counter()

        for n in range(self.M):
            th = self.thetas[n]
            G_old, _ = self._conduction(T)
            explicit = (1.0 - th) * self.dt * G_old
            H_old = self.material.enthalpy(T)
            T_new = T.copy()
            for _ in range(self.newton_max):
                G_new, k_face = self._conduction(T_new)
                cap = self.rho_vol * self.material.c(T_new)
                R = self.rho_vol * (self.material.enthalpy(T_new) - H_old) - th * self.dt * G_new - explicit
                R[0] -= self.dt * q_steps[n]
                if np.max(np.abs(R) / cap) < self.newton_tol:
                    break
                newton += 1
                sub, main, sup = self._jacobian(T_new, k_face)
                step = self._solve_tri(-th * self.dt * sub, cap - th * self.dt * main, -th * self.dt * sup, -R)
                np.clip(step, -1000.0, 1000.0, out=step)
                T_new += step
            else:
                raise RuntimeError(f"the inverse model's Newton solve did not converge at step {n}")
            T = T_new
            if store:
                field_[n + 1] = T
            if (n + 1) % spc == 0:
                obs[(n + 1) // spc] = self._probe(T)

        return Run(obs=obs, field=field_, newton_iterations=newton, seconds=time.perf_counter() - t0)

    # ---------------------------------------------------------------------------------
    # Adjoint and tangent
    # ---------------------------------------------------------------------------------

    def gradient(self, field_: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """dJ/dq_control for J = 0.5 * sum(residual^2), by one backward sweep.

        `residual[m - 1]` is the model-minus-measurement mismatch at measurement m, for
        m = 1..n_control; t = 0 carries no information about the flux and is left out.

        The recursion is the transpose of the forward step.  Writing A^j for the Jacobian
        of step j-1 with respect to its new state and B^j for that of step j with respect
        to its old state -- both evaluated at T^j, which is why one Jacobian serves both --
        stationarity of the Lagrangian gives

            (A^j)^T psi^j = -dJ/dT^j - (B^j)^T psi^{j+1},   psi^{M+1} = 0

        and the flux enters only through node 0, once per step, so dJ/dq_n = -dt psi^{n+1}_0.
        """
        spc = self.steps_per_control
        psi = np.zeros(self.N)
        per_step = np.empty(self.M)

        for j in range(self.M, 0, -1):
            T_j = field_[j]
            sub, main, sup = self._jacobian(T_j, self._k_face(T_j))
            cap = self.rho_vol * self.material.c(T_j)

            rhs = np.zeros(self.N)
            if j % spc == 0:
                r = residual[j // spc - 1]
                rhs[self.i0] -= (1.0 - self.w) * r
                rhs[self.i0 + 1] -= self.w * r
            if j < self.M:
                # -(B^j)^T psi, with B^j = -diag(cap) - dt (1 - theta_j) dG/dT
                rhs += cap * psi + self.dt * (1.0 - self.thetas[j]) * self._matvec_T(sub, main, sup, psi)

            th = self.thetas[j - 1]
            psi = self._solve_tri(-th * self.dt * sup, cap - th * self.dt * main, -th * self.dt * sub, rhs)
            per_step[j - 1] = -self.dt * psi[0]

        return per_step.reshape(self.n_control, spc).sum(axis=1)

    def tangent(self, field_: np.ndarray, d_control: np.ndarray) -> np.ndarray:
        """Sensor response to a perturbation `d_control` of the flux, linearised about `field_`.

        The step length of the conjugate-gradient iteration needs exactly this, and it is
        built from the same matrices as the adjoint -- so if the gradient check passes, the
        step length is consistent with it by construction rather than by coincidence.
        """
        spc = self.steps_per_control
        d_steps = np.repeat(np.asarray(d_control, dtype=float), spc)
        dT = np.zeros(self.N)
        out = np.zeros(self.n_control + 1)

        T_old = field_[0]
        jac_old = self._jacobian(T_old, self._k_face(T_old))
        cap_old = self.rho_vol * self.material.c(T_old)
        for n in range(self.M):
            th = self.thetas[n]
            T_new = field_[n + 1]
            jac_new = self._jacobian(T_new, self._k_face(T_new))
            cap_new = self.rho_vol * self.material.c(T_new)

            rhs = cap_old * dT + self.dt * (1.0 - th) * self._matvec(*jac_old, dT)
            rhs[0] += self.dt * d_steps[n]
            dT = self._solve_tri(
                -th * self.dt * jac_new[0], cap_new - th * self.dt * jac_new[1], -th * self.dt * jac_new[2], rhs
            )
            if (n + 1) % spc == 0:
                out[(n + 1) // spc] = self._probe(dT)
            jac_old, cap_old = jac_new, cap_new
        return out

    # ---------------------------------------------------------------------------------
    # Duhamel, for the two methods that need superposition
    # ---------------------------------------------------------------------------------

    def control_times(self) -> np.ndarray:
        """Mid-points of the flux intervals -- where a staircase estimate is compared with
        a continuous truth."""
        return (np.arange(self.n_control) + 0.5) / self.scenario.f_sample

    def measurement_times(self) -> np.ndarray:
        return np.arange(self.n_control + 1) / self.scenario.f_sample

    def step_response(self, *, q_scale: float = 1.0e6) -> tuple[np.ndarray, np.ndarray]:
        """Sensor rise from a unit flux switched on at t = 0, and the zero-flux baseline.

        The problem is affine rather than linear -- the disc starts 15 K above ambient, so
        it cools on its own -- so the two are separated here: `baseline` is the sensor with
        no flux at all, and `S` is the response per W/m^2 of applied flux.  Their sum is
        the model, exactly, whenever the properties are constant.

        `q_scale` only conditions the subtraction: at 1 W/m^2 the rise is a microkelvin and
        the difference of two 300 K numbers would lose eight digits to cancellation.
        """
        if not self.material.is_linear:
            raise ValueError(f"Duhamel superposition needs constant properties; law {self.material.name} is not")
        baseline = self.solve(np.zeros(self.n_control), store=False).obs
        unit = self.solve(np.full(self.n_control, q_scale), store=False).obs
        return (unit - baseline) / q_scale, baseline

    def duhamel_matrix(self, S: np.ndarray) -> np.ndarray:
        """Sensitivity of every measurement to every flux interval, K per W/m^2.

        Lower triangular and Toeplitz: measurement m responds to interval j only through
        the elapsed time, so the whole matrix is the one step response, staggered.
        """
        phi = np.diff(S)  # response to one interval of unit flux, by lag
        first_row = np.zeros(self.n_control)
        first_row[0] = phi[0]
        return toeplitz(phi, first_row)


def inverse_crime_data(model: InverseModel) -> tuple[np.ndarray, np.ndarray]:
    """The one place the inverse crime is committed on purpose, and it is named for it.

    Returns a flux staircase and the *exact* observations the model produces from it, with
    no noise and no discretisation gap.  Any correct implementation has to invert this
    almost perfectly, so it separates a broken method from a method defeated by the
    physics.  It says nothing about accuracy on real data, and no result in the repo is
    allowed to come from it.
    """
    q = model.scenario.q_true(model.control_times())
    return q, model.solve(q, store=False).obs


# ======================================================================================
# B1 -- Beck's sequential function specification
# ======================================================================================


def beck(
    model: InverseModel,
    Y: np.ndarray,
    *,
    stride: int = 5,
    r: int | None = None,
    duhamel: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """March through time estimating the flux a block at a time, looking `r` measurements ahead.

    Two things regularise this method, and both are Beck's, not inventions here.

    *The flux step.*  The flux is held constant over `stride` measurement intervals.  The
    classical guidance is to size it by the dimensionless step `alpha dt_q / x_s^2`, which
    should land around 0.1 to 0.5; at 2 mm in cast iron the 100 Hz measurement interval
    gives 0.039, five times too fine, and estimating one flux value per measurement makes
    the sequential step far more ill conditioned than the method needs to be.

    *The look-ahead.*  At each block the flux is provisionally assumed constant over the
    next `r` measurements, which turns an ill-conditioned step into a small least-squares
    problem with a closed form.  The sensitivity of the measurement `i` steps ahead to a
    flux held constant from now is exactly the step response `S_i`, so

        q_b = sum_i (Y_{m+i} - base_{m+i}) S_i / sum_i S_i^2

    with `base` the measurements the flux already fixed would produce.  `r` must exceed
    `stride`, or each block is fitted only to data it fully determines and the recursion
    goes unstable; `r = 2 * stride` by default.

    Why not `stride = 1, r = 1`, the unregularised limit?  Because `S_1` is 4e-9 K per
    W/m^2 -- after 10 ms the sensor at 2 mm has not yet heard about the flux at all -- and a
    recursion that divides by it amplifies round-off by 10^9 per step.  The unregularised
    version of this method does not merely become inaccurate, it fails to run.

    Near the end of the record fewer than `r` measurements remain, so the window shrinks;
    extrapolating the data instead would invent the part of the answer least constrained
    by it.
    """
    S, baseline = model.step_response() if duhamel is None else duhamel
    r = 2 * stride if r is None else r
    n = model.n_control
    base = baseline.copy()
    q = np.zeros(n)

    for start in range(0, n, stride):
        block = min(stride, n - start)
        ahead = min(r, n - start)
        psi = S[1 : ahead + 1]
        value = float((Y[start + 1 : start + ahead + 1] - base[start + 1 : start + ahead + 1]) @ psi / (psi @ psi))
        q[start : start + block] = value
        # The block's own response, by telescoping the step response: `block` intervals of
        # unit flux starting now contribute S_L - S_{L-block} at lag L.
        lags = np.arange(1, n - start + 1)
        base[start + 1 :] += value * (S[lags] - S[np.maximum(lags - block, 0)])

    return {"q": q, "obs": base, "residual": base[1:] - Y[1:], "stride": stride, "r": r}


def beck_by_discrepancy(
    model: InverseModel,
    Y: np.ndarray,
    *,
    sigma: float,
    strides: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25),
    look_ahead_ratio: int = 2,
    duhamel: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Pick the flux step the way `lambda` is picked for Tikhonov, and publish the sweep.

    The *largest* step whose residual still explains the data to within the noise: more
    smoothing always fits worse, so the discrepancy principle points the same way here as
    it does for Tikhonov, and picking the smallest instead would select pure noise
    amplification.  `stride` decides the answer, so how it was chosen is reported with it.
    """
    duhamel = model.step_response() if duhamel is None else duhamel
    sweep, chosen = [], None
    for stride in strides:
        out = beck(model, Y, stride=stride, r=look_ahead_ratio * stride, duhamel=duhamel)
        rms = float(np.sqrt(np.mean(out["residual"] ** 2)))
        sweep.append({"stride": stride, "r": out["r"], "residual_rms": rms})
        if np.isfinite(rms) and rms <= sigma:
            chosen = out  # keep overwriting: the last one that fits is the smoothest one
    if chosen is None:  # nothing fitted; fall back to the coarsest tried
        chosen = out
    chosen["sweep"] = sweep
    return chosen


# ======================================================================================
# B2 -- Tikhonov deconvolution of the Duhamel sum
# ======================================================================================


def _second_difference(n: int) -> np.ndarray:
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i : i + 3] = (1.0, -2.0, 1.0)
    return D


def _menger_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Curvature of the polyline through consecutive triples, for the L-curve corner.

    Menger's formula rather than a fitted derivative: the L-curve is sampled coarsely in
    log-log and a spline through it invents inflections between the samples.
    """
    k = np.zeros(x.size)
    for i in range(1, x.size - 1):
        p, c, q = np.array([x[i - 1], y[i - 1]]), np.array([x[i], y[i]]), np.array([x[i + 1], y[i + 1]])
        area = 0.5 * ((c[0] - p[0]) * (q[1] - p[1]) - (q[0] - p[0]) * (c[1] - p[1]))
        lengths = np.linalg.norm(c - p) * np.linalg.norm(q - c) * np.linalg.norm(q - p)
        k[i] = 4.0 * area / lengths if lengths > 0 else 0.0
    return k


def tikhonov(
    model: InverseModel,
    Y: np.ndarray,
    *,
    sigma: float,
    lambdas: np.ndarray | None = None,
    duhamel: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Deconvolve the whole flux history at once, penalising its second difference.

        min_q  || X q - (Y - baseline) ||^2 + lambda || D2 q ||^2

    `X` is the Duhamel sensitivity matrix, so this is exactly the linear problem -- no
    sequential approximation, no look-ahead window.  It is the strongest thing available
    for law A and the natural yardstick for everything else.

    `lambda` is chosen two ways and both are returned, because they answer different
    questions and disagreeing is informative:

        discrepancy   the largest lambda whose residual still explains the data to within
                      the known noise level.  Uses sigma, which is a real advantage a
                      real experiment would also have.
        l_curve       the corner of log||residual|| against log||D2 q||, where buying more
                      smoothness starts costing fit.  Uses no knowledge of sigma.

    The headline uses the discrepancy choice; the L-curve is reported next to it so the
    reader can see how much the choice was worth.
    """
    S, baseline = model.step_response() if duhamel is None else duhamel
    n = model.n_control
    X = model.duhamel_matrix(S)
    y = Y[1:] - baseline[1:]

    XtX = X.T @ X
    Xty = X.T @ y
    D = _second_difference(n)
    DtD = D.T @ D
    scale = float(np.trace(XtX) / n)  # keeps lambda dimensionless-ish across scenarios

    def solve(lam: float) -> tuple[np.ndarray, float, float]:
        c = cho_factor(XtX + lam * scale * DtD, lower=True)
        q = cho_solve(c, Xty)
        return q, float(np.linalg.norm(X @ q - y)), float(np.linalg.norm(D @ q))

    lambdas = np.logspace(-10.0, 2.0, 25) if lambdas is None else np.asarray(lambdas, dtype=float)
    sweep = []
    for lam in lambdas:
        _, res, pen = solve(float(lam))
        sweep.append({"lambda": float(lam), "residual": res, "penalty": pen})

    corner = int(np.argmax(_menger_curvature(np.log(np.array([s["residual"] for s in sweep])),
                                             np.log(np.array([s["penalty"] for s in sweep]) + 1e-300))))
    lam_lcurve = float(sweep[corner]["lambda"])

    # Discrepancy: the residual norm grows monotonically with lambda, so bisect on log.
    target = np.sqrt(n) * sigma
    lo, hi = np.log(lambdas[0]), np.log(lambdas[-1])
    if solve(float(np.exp(lo)))[1] >= target:
        lam_disc = float(np.exp(lo))
    elif solve(float(np.exp(hi)))[1] <= target:
        lam_disc = float(np.exp(hi))
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if solve(float(np.exp(mid)))[1] < target:
                lo = mid
            else:
                hi = mid
        lam_disc = float(np.exp(0.5 * (lo + hi)))

    q, res, pen = solve(lam_disc)
    q_lcurve, res_l, pen_l = solve(lam_lcurve)
    return {
        "q": q,
        "residual": X @ q - y,
        "lambda": lam_disc,
        "lambda_scale": scale,
        "lambda_l_curve": lam_lcurve,
        "q_l_curve": q_lcurve,
        "residual_rms": res / np.sqrt(n),
        "residual_rms_l_curve": res_l / np.sqrt(n),
        "sweep": sweep,
    }


# ======================================================================================
# B3 -- Alifanov's iterative regularisation
# ======================================================================================


def alifanov(
    model: InverseModel,
    Y: np.ndarray,
    *,
    sigma: float,
    q_init: np.ndarray | None = None,
    max_iter: int = 120,
    discrepancy: float = 1.0,
) -> dict:
    """Conjugate gradients on the least-squares functional, with the adjoint gradient.

    Nothing here assumes superposition, a parametric flux or a linear material: the
    gradient comes from the adjoint of whatever the model is, so laws B and C cost the same
    as law A.  That generality is exactly what the repo is testing the PINN against.

    The regularisation is the *stopping rule*, not a penalty term.  Early conjugate-gradient
    iterations reconstruct the smooth, well-determined part of the flux and later ones start
    fitting noise, so the iteration is stopped as soon as the residual has come down to the
    noise level -- Alifanov's iterative regularisation.  There is no lambda to tune, which
    is a real advantage, and its cost is that the answer depends on where it stopped.

    The step length is exact for the linearised problem: with the tangent-linear sensor
    response `dobs` to the current direction, J(q + a d) is a parabola in a, minimised at
    -(r . dobs)/(dobs . dobs).  For law A that is the exact minimiser; for B and C it is one
    Newton step on the line, which is what Alifanov's method does.
    """
    n = model.n_control
    q = np.zeros(n) if q_init is None else np.array(q_init, dtype=float)
    target = discrepancy * n * sigma**2

    history = []
    g_prev_sq, d = None, np.zeros(n)
    stop = "max_iter"
    t0 = time.perf_counter()

    for it in range(max_iter):
        run = model.solve(q)
        residual = run.obs[1:] - Y[1:]
        J = float(residual @ residual)
        g = model.gradient(run.field, residual)
        g_sq = float(g @ g)
        history.append({"iteration": it, "sse": J, "rms": float(np.sqrt(J / n)), "grad_norm": float(np.sqrt(g_sq))})

        if J <= target:
            stop = "discrepancy"
            break
        if g_sq == 0.0:
            stop = "zero_gradient"
            break

        beta = 0.0 if g_prev_sq is None else g_sq / g_prev_sq
        d = -g + beta * d
        g_prev_sq = g_sq

        dobs = model.tangent(run.field, d)[1:]
        denom = float(dobs @ dobs)
        if denom == 0.0:
            stop = "flat_direction"
            break
        alpha = -float(residual @ dobs) / denom
        if alpha <= 0.0:  # conjugacy lost on the nonlinear problem: restart on steepest descent
            d = -g
            dobs = model.tangent(run.field, d)[1:]
            denom = float(dobs @ dobs)
            alpha = -float(residual @ dobs) / max(denom, 1e-300)
            g_prev_sq = None
        q = q + alpha * d

    return {
        "q": q,
        "residual": residual,
        "iterations": len(history),
        "stop": stop,
        "sigma": sigma,
        "history": history,
        "seconds": time.perf_counter() - t0,
    }
