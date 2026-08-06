"""The same inverse problem, posed as an optimisation over functions instead of over a grid.

The classical methods in `classical.py` all estimate the *same object*: a staircase of flux
values on the 100 Hz measurement grid, pushed through a finite-volume model whose adjoint
had to be derived and transposed by hand.  The PINN replaces both halves of that.  The
temperature field is a network `T_theta(x, t)`, the flux is a second network `q_phi(t)`, and
the heat equation is not solved -- it is *scored*, at scattered collocation points, by
automatic differentiation of `T_theta`.  Nothing is marched, so nothing has to be inverted,
and the gradient of the whole objective with respect to every unknown comes from the same
backward pass that a forward network already needs.

That is the claim the repo exists to test, and act 1 is where it is tested at its *weakest*:
constant properties, one unknown function, superposition available to the opposition.  The
classical route is at its best here, and the expectation set out in the plan is a draw at
best.  This file is written to make that a fair measurement rather than a fixed one.

## The change of variables is not cosmetic

In physical units `x ~ 1e-3`, `alpha ~ 1e-5`, `T ~ 1e3`, and the residual of the heat
equation is a difference of terms around 1e9 that has to cancel to a few parts in 1e3.  A
network initialised to O(1) outputs cannot express that, and the optimiser never leaves the
trivial solution.  Everything therefore happens in

    x_hat = x / L        t_hat = t / t_end        T_hat = (T - T0) / dT_ref
    q_hat = q / q_ref     with  q_ref = k0 dT_ref / L

which turns the whole problem into two dimensionless numbers -- the Fourier number
`Fo = alpha0 t_end / L^2 = 12.6`, and a Biot number `h L / k0 = 5.8e-3` hiding in the
boundary condition -- and puts every quantity the network sees within an order of magnitude
of one.  `Scaling` is the only place the conversion is written, and `tests/test_pinn.py`
holds it to a round trip.

## What the network is allowed to represent

`FourierFeatures` is where the one real prior in this file lives, and it is worth stating
plainly rather than burying in a hyperparameter.  A plain MLP is spectrally biased: it
learns the smooth part of a signal long before the fast part, and the 0.3 s pedal release in
a 20 s record is fast.  Lifting the inputs onto a ladder of sines and cosines removes that
bias -- but the same ladder decides what the flux network *can* oscillate at, and inverse
heat conduction is exactly the problem where unlimited bandwidth means unlimited noise
amplification.

So the ladder stops at the recoverable band.  `naive.cutoff_frequency` says a component
above 6.6 Hz reached the thermocouple at 2 mm attenuated by more than the amplification
budget the repo declared in M2; the flux network's highest feature is placed there.  This is
the PINN's regularisation, in the same sense that `stride` is Beck's and `lambda` is
Tikhonov's -- every method on this problem has one, and pretending otherwise would be the
straw man.  What makes it defensible is that it is set from a property of the *experiment*
(depth, diffusivity, the eta = 0.1 budget) rather than from the answer, and that it is a
soft limit: an MLP downstream of those features can and does synthesise harmonics above
them.  The band edge is where the features stop, not where the network stops.

## What is scored, and what is fixed by construction

    L = mean |T_theta(x_s, t_n) - Y_n|^2                              the sensor
      + lambda_r mean |c_hat T_hat_t + Fo d/dx_hat (k_hat T_hat_x)|^2  the heat equation
      + lambda_b ( mean |F(0) - q_hat + losses|^2 + mean |T_hat_x(1)|^2 )   the two ends

The initial condition is not in that list: `T_hat = t_hat * N_theta(x_hat, t_hat)` satisfies
it identically, which removes a term whose weight would otherwise have had to be guessed.

The residual is written as a flux divergence -- form the dimensionless flux
`F = -k_hat(T_hat) T_hat_x` and differentiate *it* -- rather than expanded into
`k T_xx + k'(T) T_x^2`.  Two reasons, and the second is the one that matters: it is the same
conservative form the finite-volume truth solver uses, so the two discretise the same
statement; and when act 3 replaces `k_hat` by a third network, the chain rule through it is
autodiff's problem rather than a derivative someone has to re-derive.  That substitution is
the whole thesis of the repo, and it is one line here.

`lambda_r` and `lambda_b` are set by balancing gradient norms -- each term is weighted so that
it pulls on the shared parameters as hard as the sensor term does -- measured once at
initialisation and then frozen.  Frozen rather than re-measured as training proceeds, which
is the textbook scheme, because the textbook scheme diverges here; `train` carries the
measurement.  A multiplier on `lambda_r` is then swept over four decades in
`scripts/03_act1.py`, on `scenario.TUNING` and never on the showcase.

What that sweep finds is worth stating in advance, because it is the shape of an answer this
repo has already met once.  The balanced weight is not the good weight: it is set by what the
two terms are worth before either is satisfied, and by the time the sensor term has reached
the noise floor the residual needs three more decades before the heat equation is enforced
rather than merely encouraged.  Push further still and the fit to the sensor degrades past
what the noise justifies and the answer gets worse again.  There is an interior optimum, it
has to be found by sweeping, and finding it needs the truth -- which is why it is found on a
scenario that is not the one being reported.  The PINN has a regularisation parameter, in the
same sense and with the same awkwardness as Tikhonov's `lambda`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from .material import SIGMA_SB, Material, T_REF
from .naive import cutoff_frequency
from .scenario import Scenario


# ======================================================================================
# The change of variables
# ======================================================================================


@dataclass(frozen=True)
class Scaling:
    """Physical units in, dimensionless units out, in one place and one direction each.

    `dT_ref` is a *nominal* rise, not a measured one: 500 K is the order of magnitude of a
    hard stop, and using the actual peak would leak the answer into the normalisation.
    """

    L: float
    t_end: float
    T0: float
    T_inf: float
    h: float
    eps_sb: float  # eps * sigma, zero when the law has no radiation
    k0: float
    c0: float
    rho: float
    dT_ref: float = 500.0

    @classmethod
    def of(cls, scenario: Scenario, material: Material, dT_ref: float = 500.0) -> "Scaling":
        return cls(
            L=scenario.L,
            t_end=scenario.t_end,
            T0=float(scenario.T0),
            T_inf=float(scenario.T_inf),
            h=scenario.h,
            eps_sb=(scenario.eps * SIGMA_SB) if material.radiation else 0.0,
            k0=material.k0,
            c0=material.c0,
            rho=material.rho,
            dT_ref=dT_ref,
        )

    # -- the two dimensionless groups ---------------------------------------------------

    @property
    def alpha0(self) -> float:
        return self.k0 / (self.rho * self.c0)

    @property
    def Fo(self) -> float:
        """Fourier number: how many diffusion times the event lasts.  12.6 here."""
        return self.alpha0 * self.t_end / self.L**2

    @property
    def q_ref(self) -> float:
        """The flux that would drive `dT_ref` across the half thickness, W/m^2."""
        return self.k0 * self.dT_ref / self.L

    @property
    def Bi(self) -> float:
        """Biot number of the convective loss, h L / k0."""
        return self.h * self.L / self.k0

    # -- forwards ------------------------------------------------------------------------

    def x_hat(self, x):
        return np.asarray(x, dtype=float) / self.L

    def t_hat(self, t):
        return np.asarray(t, dtype=float) / self.t_end

    def T_hat(self, T):
        return (np.asarray(T, dtype=float) - self.T0) / self.dT_ref

    def q_hat(self, q):
        return np.asarray(q, dtype=float) / self.q_ref

    # -- and back ------------------------------------------------------------------------

    def x(self, x_hat):
        return np.asarray(x_hat, dtype=float) * self.L

    def t(self, t_hat):
        return np.asarray(t_hat, dtype=float) * self.t_end

    def T(self, T_hat):
        return self.T0 + np.asarray(T_hat, dtype=float) * self.dT_ref

    def q(self, q_hat):
        return np.asarray(q_hat, dtype=float) * self.q_ref


class ReducedMaterial:
    """`Material`, re-expressed in torch and in reduced variables.

    A second implementation of formulae that already exist in `material.py` is a liability,
    so it is treated as one: `tests/test_pinn.py` holds `k_hat` and `c_hat` against the numpy
    `Material` across the whole temperature range the event visits, for all three laws.  The
    duplication is unavoidable -- `Material` calls `np.asarray`, which severs a torch graph --
    and it is confined to these ten lines.

    `k_hat` is `k(T) / k0` as a function of `T_hat`, so it is 1 at the reference temperature
    by construction; that is the same normalisation act 3's learned conductivity is held to.
    """

    def __init__(self, material: Material, scaling: Scaling) -> None:
        self.material = material
        self.scaling = scaling
        self._dT_from_ref = (scaling.T0 - T_REF, scaling.dT_ref)  # T - T_REF = a + b * T_hat

    def _from_ref(self, T_hat: Tensor) -> Tensor:
        a, b = self._dT_from_ref
        return a + b * T_hat

    def k_hat(self, T_hat: Tensor) -> Tensor:
        m = self.material
        dT = self._from_ref(T_hat)
        k = 1.0 + m.k_slope * dT
        if m.knee_drop != 0.0:
            T = dT + T_REF
            step = 0.5 * (1.0 + torch.tanh((T - m.knee_T) / (0.5 * m.knee_width)))
            k = k * (1.0 - m.knee_drop * step)
        return k

    def c_hat(self, T_hat: Tensor) -> Tensor:
        return 1.0 + self.material.c_slope * self._from_ref(T_hat)


# ======================================================================================
# The two networks
# ======================================================================================


class FourierFeatures(nn.Module):
    """Lift a coordinate onto a geometric ladder of sines and cosines.

    The ladder runs from one half period across the domain up to `w_max` of them, so a
    feature at `w` completes `w / 2` cycles over the reduced coordinate's unit interval and
    the top of the ladder is `w_max / 2` cycles.  Geometric rather than dyadic spacing only
    so that the top of the ladder can be placed exactly on the recoverable band instead of
    on the nearest power of two.

    Half periods rather than whole ones because whole ones would make every feature agree at
    both ends of the interval, quietly telling the network that the disc at t = 0 and the
    disc at t = 20 s are the same state.  They are not, and the fundamental of this ladder
    is the feature that says so.
    """

    def __init__(self, w_max: float, n: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        ladder = np.geomspace(1.0, float(w_max), int(n)) if n > 0 else np.zeros(0)
        self.register_buffer("omega", torch.tensor(np.pi * ladder, dtype=dtype))
        self.n_out = 1 + 2 * int(n)

    def forward(self, u: Tensor) -> Tensor:
        phase = u * self.omega
        return torch.cat([u, torch.sin(phase), torch.cos(phase)], dim=-1)


def _mlp(n_in: int, width: int, depth: int, n_out: int = 1) -> nn.Sequential:
    """Glorot-initialised tanh MLP: `depth` hidden layers of `width`."""
    layers: list[nn.Module] = []
    d = n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]
        d = width
    layers += [nn.Linear(d, n_out)]
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class TemperatureNet(nn.Module):
    """`T_hat(x_hat, t_hat)`, with the initial condition satisfied identically.

    The output is `g(t_hat) * N(x_hat, t_hat)` with `g(0) = 0`, so `T_hat(., 0) = 0` -- which
    in these variables *is* `T = T0` -- for every value of every parameter.  One loss term
    fewer and one weight fewer to guess.

    Which `g` is not a free choice, and getting it wrong is expensive in a way that is
    invisible until the answer is plotted.  The obvious `g = t_hat` says the surface
    temperature leaves `T0` linearly in time.  It does not: switch a flux on at t = 0 and the
    heat kernel gives `T_surface - T0 = 2 q sqrt(alpha t / pi) / k`, which rises as the
    *square root* of time and has infinite slope at the origin.  For `t_hat * N` to produce
    that, `N` would have to blow up as `t_hat^(-1/2)` at the one place it is evaluated most,
    and a tanh network cannot -- so the flux it can express is pinned near zero at t = 0 and
    has to climb to meet the truth.  Measured on the showcase, that alone was the whole of a
    34 % band error: the estimate was wrong for the first 1.8 s and tracked the truth,
    pedal release included, everywhere after it.

    `g = sqrt(t_hat)` is the same statement as the residual's own `sqrt(t)` weight, applied to
    the ansatz instead of to the norm, and the two cancel exactly: a field of the right shape
    has `T_hat_t ~ t_hat^(-1/2)`, the weight multiplies by `t_hat^(1/2)`, and what the
    optimiser sees is O(1) down to the origin.
    """

    def __init__(self, w_max_x: float, n_freq_x: int, w_max_t: float, n_freq_t: int,
                 width: int, depth: int, dtype: torch.dtype, ansatz: str = "sqrt_t") -> None:
        super().__init__()
        self.fx = FourierFeatures(w_max_x, n_freq_x, dtype)
        self.ft = FourierFeatures(w_max_t, n_freq_t, dtype)
        self.net = _mlp(self.fx.n_out + self.ft.n_out, width, depth).to(dtype)
        self.ansatz = ansatz

    def forward(self, x_hat: Tensor, t_hat: Tensor) -> Tensor:
        z = torch.cat([self.fx(x_hat), self.ft(t_hat)], dim=-1)
        # Clamped because `torch.rand` does return exactly 0, where `d sqrt(t) / dt` is
        # infinite and autograd would multiply it by a zero and produce a NaN.  Below the
        # floor `clamp_min` is flat, so the derivative there is zero rather than infinite and
        # nothing propagates.  The floor costs the initial condition its exactness: at t = 0
        # the network returns `1e-10 * N` instead of 0, which is under a microkelvin --
        # seven decades below the thermocouple's noise, and the only reason the test asserts
        # a tolerance rather than equality.
        ramp = torch.sqrt(t_hat.clamp_min(1e-20)) if self.ansatz == "sqrt_t" else t_hat
        return ramp * self.net(z)


class FluxNet(nn.Module):
    """`q_hat(t_hat)`, unconstrained in sign.

    The true flux is clipped at zero -- no heat is pumped back into the pad -- and enforcing
    that here would be free accuracy the classical methods do not get, since none of them
    constrains the sign either.  It is left out for that reason, and the negative excursions
    the estimate makes during the cooling phase are part of what is measured.
    """

    def __init__(self, w_max_t: float, n_freq_t: int, width: int, depth: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.ft = FourierFeatures(w_max_t, n_freq_t, dtype)
        self.net = _mlp(self.ft.n_out, width, depth).to(dtype)

    def forward(self, t_hat: Tensor) -> Tensor:
        return self.net(self.ft(t_hat))


# ======================================================================================
# Configuration
# ======================================================================================


@dataclass(frozen=True)
class PINNConfig:
    """Everything that is a choice rather than a consequence.

    The defaults are the plan's architecture (4x64 for the field, 3x32 for the flux, Adam
    then L-BFGS, 4096 collocation points resampled every 100 steps).  `flux_band_hz = None`
    means the Fourier ladder is placed on the recoverable band computed from the sensor
    depth, which is the intended setting; a number overrides it, which is what the bandwidth
    ablation does.
    """

    # architecture
    width_T: int = 64
    depth_T: int = 4
    width_q: int = 32
    depth_q: int = 3
    n_freq_x: int = 4
    #: Highest half period across the disc thickness -- 4 full cycles over 5 mm, far finer
    #: than the smoothest field the data can support, so it is not a constraint in practice.
    w_max_x: float = 8.0
    n_freq_t: int = 10
    flux_band_hz: float | None = None
    #: How the hard initial condition ramps off t = 0: "sqrt_t" matches the heat kernel,
    #: "linear" is the obvious choice and is measurably wrong.  See `TemperatureNet`.
    ic_ansatz: str = "sqrt_t"

    # optimisation
    #
    # The plan called for 20 000 Adam steps followed by 2 000 of L-BFGS.  Measured, the split
    # is close to backwards: past about 4 000 steps Adam is worth very little here -- from
    # 6 000 to 8 000 it moved the residual from 65 to 57 -- while the first 2 000 L-BFGS
    # iterations took it from 57 to 1.0, a factor of fifty, at about 1.2x the cost per
    # iteration.  Adam's job on this problem is to get into the basin, and L-BFGS's is to
    # solve it.  Past roughly 12 000 L-BFGS iterations the residual stops moving at all, so
    # that is where the budget stops.
    n_adam: int = 4000
    n_lbfgs: int = 12000
    lr: float = 1.0e-3
    lr_final: float = 1.0e-5
    n_collocation: int = 4096
    n_boundary: int = 512
    resample_every: int = 100
    #: "sqrt_t" makes the mean squared residual a convergent average; "none" is the ablation.
    #: See `Problem.residual` for why the unweighted version is not merely worse but ill posed.
    residual_weight: str = "sqrt_t"
    #: L-BFGS needs a fixed objective to build curvature from, so the collocation set is
    #: frozen for a round and redrawn between rounds.  One round would let it fit those 4096
    #: points; redrawing every round costs the curvature estimate and keeps it honest.
    lbfgs_round: int = 500

    # loss weights
    lambda_r: float | None = None  # None: balance gradient norms
    lambda_b: float | None = None
    #: The multiplier the sweep moves, applied on top of the balanced weight.
    lambda_r_scale: float = 1.0
    #: 0 measures the balance once, at initialisation, and then freezes it.  Anything larger
    #: re-measures it every that many steps, which is the textbook adaptive scheme and which
    #: diverges on this problem -- see `train` for the measurement.  Kept as an ablation.
    balance_every: int = 0
    balance_ema: float = 0.9

    # bookkeeping
    dT_ref: float = 500.0
    device: str = "cpu"
    dtype: str = "float32"
    log_every: int = 500

    def quick(self) -> "PINNConfig":
        """A tenth of the iterations, for checking that the pipeline runs.

        Not for producing a number: the residual is still two decades above where it settles.
        """
        return replace(self, n_adam=400, n_lbfgs=1200, lbfgs_round=200, log_every=200)


@dataclass
class PINNResult:
    """One trained PINN, evaluated where the classical methods are evaluated.

    `q` is sampled at the mid-points of the same 100 Hz intervals Beck and Tikhonov estimate
    a constant over, so `metrics.band_error` compares like with like.  `T_surface` is the
    network's own answer to the question the repo asks; `sensor_fit` is what it thinks the
    thermocouple read, and the gap between that and the data is the diagnostic that says
    whether it has started fitting noise -- it should sit at sigma, not below it.
    """

    q: np.ndarray
    t_control: np.ndarray
    T_surface: np.ndarray
    t_surface: np.ndarray
    sensor_fit: np.ndarray
    t_sensor: np.ndarray
    history: list[dict]
    weights: dict
    seed: int
    seconds: float
    config: PINNConfig
    scaling: Scaling
    band_hz: float


# ======================================================================================
# Residual and losses
# ======================================================================================


def _d(y: Tensor, u: Tensor) -> Tensor:
    return torch.autograd.grad(y, u, grad_outputs=torch.ones_like(y), create_graph=True)[0]


class Problem:
    """The three loss terms, sharing one set of networks and one scaling."""

    def __init__(self, scenario: Scenario, material: Material, cfg: PINNConfig, band_hz: float,
                 device: torch.device, dtype: torch.dtype) -> None:
        self.scaling = Scaling.of(scenario, material, cfg.dT_ref)
        self.reduced = ReducedMaterial(material, self.scaling)
        self.cfg = cfg
        self.Fo = self.scaling.Fo
        self.x_s_hat = float(scenario.x_s / scenario.L)

        # The flux network's ladder is placed on the band; the field's shares it, because the
        # surface temperature carries whatever the flux does.
        w_max_t = 2.0 * band_hz * scenario.t_end
        self.T_net = TemperatureNet(cfg.w_max_x, cfg.n_freq_x, w_max_t, cfg.n_freq_t,
                                    cfg.width_T, cfg.depth_T, dtype, cfg.ic_ansatz).to(device)
        self.q_net = FluxNet(w_max_t, cfg.n_freq_t, cfg.width_q, cfg.depth_q, dtype).to(device)

        # Surface losses, in units of q_ref, as a function of T_hat at the surface.
        s = self.scaling
        self._loss_lin = s.h * s.dT_ref / s.q_ref
        self._loss_offset = s.h * (s.T0 - s.T_inf) / s.q_ref
        self._eps_sb = s.eps_sb

    def parameters(self):
        return list(self.T_net.parameters()) + list(self.q_net.parameters())

    def surface_losses(self, T_hat0: Tensor) -> Tensor:
        """Convection and radiation off the friction surface, in units of q_ref."""
        out = self._loss_lin * T_hat0 + self._loss_offset
        if self._eps_sb:
            s = self.scaling
            T = s.T0 + s.dT_ref * T_hat0
            out = out + self._eps_sb * (T**4 - s.T_inf**4) / s.q_ref
        return out

    def residual(self, x: Tensor, t: Tensor) -> Tensor:
        """`c_hat T_t + Fo d(k_hat T_x)/dx`, in reduced variables, weighted by `sqrt(t)`.

        The weight is not a fudge, and leaving it out does not merely slow the fit -- it makes
        the objective ill defined.  Switch 5 MW/m^2 on at t = 0 and the surface obeys the heat
        kernel: `T_hat_t = q_hat sqrt(Fo / pi t_hat)`, which is 6 at the end of the record and
        630 a fifth of a millisecond into it.  Square that and average it over collocation
        points drawn uniformly in `t_hat` and the integral is one of `1 / t_hat`, which
        diverges logarithmically: the mean squared residual has no limit as the sample grows,
        it just keeps climbing with the number of points, so it is not an objective so much as
        a function of how finely the corner happened to be sampled.

        What that does in practice is worse than the divergence sounds, because a logarithm
        weights every *decade* of time equally.  Measured on this scenario: the interval
        `t_hat` in [1e-4, 1e-3] carries as much of the residual as [1e-1, 1] does, on a
        thousandth of the points -- so 99.9 % of the collocation budget argues over a tenth of
        the objective, and the network spends itself on the first two milliseconds instead of
        on the flux.

        Multiplying by `sqrt(t_hat)` is exactly the factor that cancels it: the weighted
        residual is O(1) across the whole record, and the average of its square converges.
        Nothing is exempted -- a residual that is zero is still zero after weighting, and the
        equation being enforced is unchanged -- only the relative attention paid to a corner
        where the hard initial condition already holds the answer exactly.  `PINNConfig` can
        turn it off, and `scripts/03_act1.py` reports what that costs.
        """
        # Cloned rather than marked in place: the collocation set outlives this call, and a
        # tensor left requiring grad would collect a gradient from every later backward pass.
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        T = self.T_net(x, t)
        T_x = _d(T, x)
        F = -self.reduced.k_hat(T) * T_x  # dimensionless conduction flux towards +x
        F_x = _d(F, x)
        T_t = _d(T, t)
        r = self.reduced.c_hat(T) * T_t + self.Fo * F_x
        # Detached: the weight is a property of where the point is, not something the
        # backward pass has anything to say about -- and `d sqrt(t) / dt` is infinite at the
        # t = 0 that `torch.rand` does occasionally return.
        return torch.sqrt(t.detach()) * r if self.cfg.residual_weight == "sqrt_t" else r

    def boundary(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Mismatch at the friction surface and at the adiabatic mid-plane."""
        zeros = torch.zeros_like(t).requires_grad_(True)
        T0s = self.T_net(zeros, t)
        F0 = -self.reduced.k_hat(T0s) * _d(T0s, zeros)
        surface = F0 - (self.q_net(t) - self.surface_losses(T0s))

        ones = torch.ones_like(t).requires_grad_(True)
        back = _d(self.T_net(ones, t), ones)
        return surface, back


# ======================================================================================
# Training
# ======================================================================================


def train(
    scenario: Scenario,
    material: Material,
    Y: np.ndarray,
    *,
    config: PINNConfig | None = None,
    seed: int = 0,
    t_meas: np.ndarray | None = None,
    t_surface: np.ndarray | None = None,
    on_log: Callable[[dict], None] | None = None,
) -> PINNResult:
    """Fit `T_theta` and `q_phi` to one noisy sensor record.

    `Y` is the thermocouple trace in kelvin at `scenario.sensor_times()`, exactly the array
    the classical methods are given.  The sample at t = 0 is dropped, as it is for them: the
    initial condition is known, so that reading carries no information about the flux and
    scoring the network against its noise would only fight the hard initial condition.
    """
    cfg = PINNConfig() if config is None else config
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    band_hz = cutoff_frequency(scenario.x_s, material.alpha0) if cfg.flux_band_hz is None else cfg.flux_band_hz

    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    problem = Problem(scenario, material, cfg, band_hz, device, dtype)
    scaling = problem.scaling

    t_meas = scenario.sensor_times() if t_meas is None else np.asarray(t_meas, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.shape != t_meas.shape:
        raise ValueError(f"{Y.size} measurements against {t_meas.size} sampling instants")

    def col(a: np.ndarray) -> Tensor:
        return torch.tensor(np.asarray(a, dtype=float).reshape(-1, 1), dtype=dtype, device=device)

    t_data = col(scaling.t_hat(t_meas[1:]))
    T_data = col(scaling.T_hat(Y[1:]))
    x_data = torch.full_like(t_data, problem.x_s_hat)

    def sample() -> tuple[Tensor, Tensor, Tensor]:
        x = torch.rand(cfg.n_collocation, 1, generator=gen).to(device=device, dtype=dtype)
        t = torch.rand(cfg.n_collocation, 1, generator=gen).to(device=device, dtype=dtype)
        tb = torch.rand(cfg.n_boundary, 1, generator=gen).to(device=device, dtype=dtype)
        return x, t, tb

    params = problem.parameters()
    lam_r = 1.0 if cfg.lambda_r is None else cfg.lambda_r
    lam_b = 1.0 if cfg.lambda_b is None else cfg.lambda_b

    def terms(x_c: Tensor, t_c: Tensor, t_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        data = torch.mean((problem.T_net(x_data, t_data) - T_data) ** 2)
        res = torch.mean(problem.residual(x_c, t_c) ** 2)
        surface, back = problem.boundary(t_b)
        return data, res, torch.mean(surface**2) + torch.mean(back**2)

    def grad_norm(loss: Tensor) -> float:
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return float(torch.sqrt(sum((gi**2).sum() for gi in g if gi is not None)))

    history: list[dict] = []
    sigma = scenario.sigma_noise
    t0 = time.perf_counter()

    def record(step: int, phase: str, data: Tensor, res: Tensor, bc: Tensor) -> None:
        data, res, bc = data.detach(), res.detach(), bc.detach()
        row = {
            "step": step,
            "phase": phase,
            "loss_data": float(data),
            "loss_residual": float(res),
            "loss_boundary": float(bc),
            "lambda_r": lam_r,
            "lambda_b": lam_b,
            # The two numbers that can be read without knowing the scaling: how far the
            # network is from the thermocouple, in kelvin, and how far it is from the heat
            # equation, as a fraction of the temperature rise per diffusion time.
            "sensor_rms_K": float(torch.sqrt(data)) * scaling.dT_ref,
            "residual_rms": float(torch.sqrt(res)),
            "seconds": time.perf_counter() - t0,
        }
        history.append(row)
        if on_log is not None:
            on_log(row)

    # ---------------------------------------------------------------------------- Adam
    opt = torch.optim.Adam(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.n_adam, 1), eta_min=cfg.lr_final)
    x_c, t_c, t_b = sample()

    for step in range(cfg.n_adam):
        if step and step % cfg.resample_every == 0:
            x_c, t_c, t_b = sample()

        rebalance = step == 0 or (cfg.balance_every and step % cfg.balance_every == 0)
        if rebalance and (cfg.lambda_r is None or cfg.lambda_b is None):
            # Balance the pull of each term on the shared parameters.  The reference is the
            # sensor term, because it is the only one carrying information the network could
            # not have invented; the other two are constraints, scaled to push as hard as it
            # does.  An exponential average rather than the raw ratio: the norms are noisy
            # across collocation draws, and a weight that jumps by a decade between steps is a
            # worse optimiser than a slightly stale one.
            #
            # Measured once and then frozen, which is not the textbook scheme and is a
            # correction to it.  Re-measuring as training proceeds diverges here, and the
            # mechanism is a feedback loop rather than bad luck: the better the residual is
            # satisfied the smaller its gradient, the rule reads a small gradient as "weight
            # this more", and the weight climbs, which satisfies the residual further.  Left
            # running it reached 3.8e6 -- four million times its initial value -- and arrived
            # at the trivial solution, `T = T0` everywhere with zero flux, which has an exactly
            # zero residual, an exactly satisfied boundary condition, and a 517 K sensor error
            # that by then weighed nothing.  Frozen, the weight means what it says, and the
            # sweep in `scripts/03_act1.py` is a sweep of the number it is named after.
            d_, r_, b_ = terms(x_c, t_c, t_b)
            n_d = grad_norm(d_)
            if cfg.lambda_r is None:
                target = n_d / max(grad_norm(r_), 1e-30)
                lam_r = target if step == 0 else cfg.balance_ema * lam_r + (1.0 - cfg.balance_ema) * target
            if cfg.lambda_b is None:
                target = n_d / max(grad_norm(b_), 1e-30)
                lam_b = target if step == 0 else cfg.balance_ema * lam_b + (1.0 - cfg.balance_ema) * target

        opt.zero_grad(set_to_none=True)
        data, res, bc = terms(x_c, t_c, t_b)
        (data + cfg.lambda_r_scale * lam_r * res + lam_b * bc).backward()
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.n_adam - 1:
            record(step, "adam", data, res, bc)

    # --------------------------------------------------------------------------- L-BFGS
    # Within a round the collocation set is frozen, because L-BFGS builds its curvature
    # estimate by differencing successive gradients: resample underneath it and the two
    # gradients belong to different functions, so what it learns the curvature of is the
    # sampling noise.  Between rounds the set is redrawn and the estimate discarded, which
    # is the price of not letting a second-order method fit 4096 particular points.
    state: dict = {}
    for done in range(0, cfg.n_lbfgs, max(cfg.lbfgs_round, 1)):
        rounds = min(cfg.lbfgs_round, cfg.n_lbfgs - done)
        x_c, t_c, t_b = sample()
        lbfgs = torch.optim.LBFGS(
            params, lr=1.0, max_iter=rounds, history_size=50, line_search_fn="strong_wolfe",
            tolerance_grad=1e-11, tolerance_change=1e-12,
        )

        def closure() -> Tensor:
            lbfgs.zero_grad(set_to_none=True)
            data, res, bc = terms(x_c, t_c, t_b)
            total = data + cfg.lambda_r_scale * lam_r * res + lam_b * bc
            total.backward()
            state["terms"] = (data.detach(), res.detach(), bc.detach())
            return total

        lbfgs.step(closure)
        record(cfg.n_adam + done + rounds, "lbfgs", *state["terms"])

    seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------------ evaluation
    t_control = (np.arange(int(round(scenario.t_end * scenario.f_sample))) + 0.5) / scenario.f_sample
    t_surface = np.arange(0.0, scenario.t_end + 1e-12, 1.0e-3) if t_surface is None else np.asarray(t_surface, float)

    with torch.no_grad():
        q = scaling.q(problem.q_net(col(scaling.t_hat(t_control))).cpu().numpy().ravel())
        ts = col(scaling.t_hat(t_surface))
        T_surf = scaling.T(problem.T_net(torch.zeros_like(ts), ts).cpu().numpy().ravel())
        tm = col(scaling.t_hat(t_meas))
        fit = scaling.T(problem.T_net(torch.full_like(tm, problem.x_s_hat), tm).cpu().numpy().ravel())

    return PINNResult(
        q=q,
        t_control=t_control,
        T_surface=T_surf,
        t_surface=t_surface,
        sensor_fit=fit,
        t_sensor=t_meas,
        history=history,
        weights={"lambda_r": lam_r * cfg.lambda_r_scale, "lambda_b": lam_b, "sigma": sigma},
        seed=seed,
        seconds=seconds,
        config=cfg,
        scaling=scaling,
        band_hz=band_hz,
    )
