"""What has to be true of the PINN before any number it produces means anything.

The training loop itself is not tested here -- whether it converges well is a *result*, and
it is measured in `scripts/03_act1.py`, not asserted.  What is tested is everything the
result would be meaningless without: that the change of variables is invertible, that the
residual is the heat equation and not something adjacent to it, that the initial condition
holds by construction rather than by convergence, and that two runs of the same seed are the
same run.
"""

import numpy as np
import pytest
import torch
from torch import nn

from thermalpinn.material import LAW_A, LAW_B, LAW_C, LAWS
from thermalpinn.naive import cutoff_frequency
from thermalpinn.pinn import (
    FourierFeatures,
    PINNConfig,
    Problem,
    ReducedMaterial,
    Scaling,
    TemperatureNet,
    train,
)
from thermalpinn.scenario import SHOWCASE, TUNING, Scenario, sample_scenarios
from thermalpinn.solver import TRUTH_GRID, forward, sensor_signal

CPU = torch.device("cpu")
QUICK = Scenario(name="quick", t_end=1.0)


@pytest.fixture(scope="module")
def data():
    """One short noisy record, so the training tests do not each pay for a truth solve."""
    truth = forward(QUICK, LAW_A, QUICK.q_true, grid=TRUTH_GRID)
    _, Y, clean = sensor_signal(truth, QUICK)
    return Y, clean, truth


def problem(scenario=SHOWCASE, material=LAW_A, dtype=torch.float64, **kw):
    cfg = PINNConfig(**kw)
    band = cutoff_frequency(scenario.x_s, material.alpha0)
    return Problem(scenario, material, cfg, band, CPU, dtype)


# ======================================================================================
# The change of variables (plan, 8.2)
# ======================================================================================


@pytest.mark.parametrize("law", ["A", "B", "C"])
def test_the_reduced_variables_survive_a_round_trip(law):
    s = Scaling.of(SHOWCASE, LAWS[law])
    rng = np.random.default_rng(0)
    x, t = rng.uniform(0.0, SHOWCASE.L, 50), rng.uniform(0.0, SHOWCASE.t_end, 50)
    T, q = rng.uniform(300.0, 1500.0, 50), rng.uniform(-1e6, 8e6, 50)
    for value, there, back in ((x, s.x_hat, s.x), (t, s.t_hat, s.t), (T, s.T_hat, s.T), (q, s.q_hat, s.q)):
        assert np.allclose(back(there(value)), value, rtol=1e-12, atol=0)


def test_the_reduced_variables_are_actually_of_order_one():
    """The whole point of the scaling: the network never sees 1e-3 or 1e6."""
    s = Scaling.of(SHOWCASE, LAW_A)
    assert 1.0 == s.x_hat(SHOWCASE.L) == s.t_hat(SHOWCASE.t_end)
    assert 0.5 < abs(s.q_hat(SHOWCASE.q_peak)) < 2.0
    assert 0.5 < abs(s.T_hat(SHOWCASE.T0 + 600.0)) < 2.0
    assert 10.0 < s.Fo < 15.0  # the Fourier number the plan quotes as 12.6
    assert s.Bi < 0.01  # the disc conducts far faster than it convects


def test_the_torch_material_is_the_same_material_as_the_numpy_one():
    """Two implementations of one formula is a liability; this is the leash on it."""
    for law in (LAW_A, LAW_B, LAW_C):
        s = Scaling.of(SHOWCASE, law)
        reduced = ReducedMaterial(law, s)
        T = np.linspace(s.T0 - 50.0, s.T0 + 1300.0, 2000)
        T_hat = torch.tensor(s.T_hat(T), dtype=torch.float64)
        assert np.allclose(reduced.k_hat(T_hat).numpy(), law.k(T) / law.k0, rtol=1e-13, atol=1e-14)
        assert np.allclose(reduced.c_hat(T_hat).numpy(), law.c(T) / law.c0, rtol=1e-13, atol=1e-14)


# ======================================================================================
# The residual is the heat equation
# ======================================================================================


class _Analytic(nn.Module):
    """T = T0 + A exp(-alpha pi^2 t / L^2) cos(pi x / L), in reduced variables.

    An exact solution of the physical problem with both faces adiabatic, which is what makes
    it a check on the residual rather than on a solver: the network is replaced by a closed
    form and the residual has nothing left to be but zero.
    """

    def __init__(self, scaling: Scaling, amplitude: float = 30.0) -> None:
        super().__init__()
        self.s, self.A = scaling, amplitude

    def forward(self, x_hat, t_hat):
        return (self.A / self.s.dT_ref) * torch.exp(-self.s.Fo * np.pi**2 * t_hat) * torch.cos(np.pi * x_hat)


@pytest.mark.parametrize("law", ["A", "B", "C"])
def test_the_residual_vanishes_on_an_exact_solution_of_the_linear_law(law):
    """The separable solution solves the *linear* equation, so laws B and C must reject it.

    Passing on all three would mean the residual had quietly dropped k(T) and c(T); the point
    of running B and C here is that they have to come out non-zero, and by the amount their
    property variation over a 30 K swing accounts for.  Run unweighted, because what is being
    checked is the equation and not the norm it is measured in.
    """
    p = problem(material=LAWS[law], residual_weight="none")
    p.T_net = _Analytic(p.scaling)
    x = torch.rand(4000, 1, dtype=torch.float64)
    t = torch.rand(4000, 1, dtype=torch.float64) * 0.05
    scale = float((p.Fo * np.pi**2 * p.T_net(x, t)).abs().max())  # the size of each term
    worst = float(p.residual(x, t).detach().abs().max()) / scale
    assert worst < 1e-12 if law == "A" else 1e-4 < worst < 1e-1


def test_a_solution_of_the_equation_is_still_a_solution_after_it_is_weighted():
    """The weight multiplies the residual, so it cannot change which fields make it zero."""
    p = problem()
    p.T_net = _Analytic(p.scaling)
    x = torch.rand(500, 1, dtype=torch.float64)
    t = torch.rand(500, 1, dtype=torch.float64)
    assert float(p.residual(x, t).detach().abs().max()) < 1e-12


def test_without_the_weight_the_objective_grows_with_the_number_of_collocation_points():
    """Which is the whole case for the weight: an objective with no limit is not an objective.

    The field here is the heat kernel's own surface response, `T ~ sqrt(t)` -- it does not
    solve the equation, and it is not meant to.  What matters is that its residual carries the
    `t^(-1/2)` term every real solution of this problem has near t = 0.  Averaged over
    uniformly drawn points the square of that integrates as `1 / t`, so the unweighted mean
    climbs like `log N` and never settles, while the weighted residual is flat in `t` and its
    mean is the same number at every sample size.
    """
    weighted, plain = problem(), problem(residual_weight="none")

    class Kernel(nn.Module):
        def forward(self, x_hat, t_hat):
            return 4.0 * torch.sqrt(t_hat) + 0.0 * x_hat

    weighted.T_net = plain.T_net = Kernel()
    means = {"sqrt_t": [], "none": []}
    for n in (512, 4096, 32768):
        g = torch.Generator().manual_seed(0)
        x = torch.rand(n, 1, generator=g, dtype=torch.float64)
        t = torch.rand(n, 1, generator=g, dtype=torch.float64)
        means["sqrt_t"].append(float(torch.mean(weighted.residual(x, t).detach() ** 2)))
        means["none"].append(float(torch.mean(plain.residual(x, t).detach() ** 2)))
    assert means["none"][-1] / means["none"][0] > 1.4  # still climbing after 64x the points
    assert means["sqrt_t"][-1] / means["sqrt_t"][0] == pytest.approx(1.0, abs=1e-9)


def test_the_adiabatic_mid_plane_is_the_condition_the_residual_is_paired_with():
    p = problem()
    p.T_net = _Analytic(p.scaling)
    _, back = p.boundary(torch.rand(500, 1, dtype=torch.float64) * 0.05)
    assert float(back.detach().abs().max()) < 1e-12


def test_the_surface_condition_is_the_flux_the_scenario_asked_for():
    """Feed a field whose surface gradient is known and check what flux it implies.

    With T_x(0) fixed by hand, the boundary term is zero exactly when q_hat equals the
    conducted flux plus the surface losses -- so solving the term for q recovers a flux in
    W/m^2 that can be compared with the physical boundary condition written out by hand.
    """
    p = problem()
    grad_phys = -4.0e4  # K/m at the surface: 5 MW/m^2 into cast iron at k = 52
    s = p.scaling

    class Ramp(nn.Module):
        def forward(self, x_hat, t_hat):
            return torch.zeros_like(x_hat) + (grad_phys * s.L / s.dT_ref) * x_hat + 0.0 * t_hat

    p.T_net = Ramp()
    t = torch.full((7, 1), 0.3, dtype=torch.float64)
    surface, _ = p.boundary(t)
    # The term is F(0) - (q_hat - losses), so adding the network's own q_hat back leaves
    # exactly the flux the field implies -- whatever the untrained q network happens to say.
    with torch.no_grad():
        implied_q = s.q(float((p.q_net(t) + surface).mean()))
        T_s = s.T(float(p.T_net(torch.zeros_like(t), t).mean()))
    by_hand = -LAW_A.k(T_s) * grad_phys + SHOWCASE.h * (T_s - SHOWCASE.T_inf)
    assert implied_q == pytest.approx(by_hand, rel=1e-10)


# ======================================================================================
# What the architecture guarantees without being trained
# ======================================================================================


@pytest.mark.parametrize("ansatz", ["sqrt_t", "linear"])
def test_the_initial_condition_holds_for_any_parameters_at_all(ansatz):
    """It is hard-wired, not learned: no weight can break it, so no loss term needs to guard it."""
    net = TemperatureNet(8.0, 4, 265.0, 10, 32, 3, torch.float64, ansatz)
    for p in net.parameters():
        with torch.no_grad():
            p.copy_(torch.randn_like(p) * 5.0)  # deliberately absurd weights
    x = torch.rand(200, 1, dtype=torch.float64)
    # Stated in kelvin, because that is the unit in which the tolerance is obviously
    # harmless: a hundredth of a millikelvin, against a thermocouple noisy to half a kelvin.
    # It is a tolerance rather than an equality only because of the clamp floor `sqrt_t`
    # needs, and the weights here are deliberately absurd, which inflates it.
    worst_K = float(net(x, torch.zeros_like(x)).detach().abs().max()) * 500.0
    assert worst_K < 1.0e-5


def test_only_the_sqrt_ansatz_can_express_the_rise_a_switched_on_flux_actually_produces():
    """The heat kernel, fitted at the surface: `T - T0 = 2 q sqrt(alpha t / pi) / k`.

    Both ansatze satisfy the initial condition, so both look correct until something has to
    be expressed through them.  This fits each to the closed-form surface rise over the first
    fifth of the record and reports what it can reach: `sqrt(t) * N` fits it because `N` may
    stay bounded, and `t * N` cannot because `N` would have to diverge at the origin.  The
    whole of act 1's early-time flux error was this.
    """
    s = Scaling.of(SHOWCASE, LAW_A)
    t = torch.linspace(1e-6, 0.2, 4000, dtype=torch.float64).reshape(-1, 1)
    q_hat = 1.0
    target = 2.0 * q_hat * torch.sqrt(s.Fo * t / np.pi)  # reduced surface rise, exact
    reached = {}
    for ansatz in ("sqrt_t", "linear"):
        torch.manual_seed(0)
        net = TemperatureNet(8.0, 4, 265.0, 10, 64, 4, torch.float64, ansatz)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        x = torch.zeros_like(t)
        for _ in range(1500):
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((net(x, t) - target) ** 2)
            loss.backward()
            opt.step()
        reached[ansatz] = float(loss) ** 0.5 / float(target.mean())
    assert reached["sqrt_t"] < 0.02
    assert reached["linear"] > 3.0 * reached["sqrt_t"]


def test_the_fourier_ladder_stops_on_the_recoverable_band():
    """The one real prior in the file, held to the number M2 measured rather than to taste."""
    f_c = cutoff_frequency(SHOWCASE.x_s, LAW_A.alpha0)
    p = problem(dtype=torch.float32)
    top = float(p.T_net.ft.omega.max()) / (2.0 * np.pi)  # cycles over the reduced record
    assert top == pytest.approx(f_c * SHOWCASE.t_end, rel=1e-6)  # float32 buffer
    assert p.q_net.ft.omega.shape == p.T_net.ft.omega.shape


def test_the_fourier_features_tell_the_two_ends_of_the_record_apart():
    """Whole periods would make t = 0 and t = t_end identical inputs.  Half periods do not."""
    f = FourierFeatures(265.0, 10, torch.float64)
    ends = f(torch.tensor([[0.0], [1.0]], dtype=torch.float64))
    assert float((ends[0] - ends[1]).abs().max()) > 1.0


def test_the_flux_network_is_free_to_be_negative():
    """Not an oversight: none of the classical baselines constrains the sign either."""
    net = problem().q_net
    q = net(torch.rand(4000, 1, dtype=torch.float64)).detach()
    assert float(q.min()) < 0.0 < float(q.max())


# ======================================================================================
# Reproducibility (plan, 8.7) and the tunables that are not allowed to be tuned (8.3)
# ======================================================================================


def test_the_tuning_scenario_is_neither_the_showcase_nor_any_case_in_the_table():
    """Every weight is chosen on TUNING, so TUNING has to be disjoint from what is reported."""
    cases = sample_scenarios(np.random.default_rng(0))
    assert TUNING.noise_seed != SHOWCASE.noise_seed
    assert all(TUNING.noise_seed != c.noise_seed for c in cases)
    assert all((TUNING.q_peak, TUNING.t_brake) != (c.q_peak, c.t_brake) for c in cases + [SHOWCASE])


def test_the_same_seed_is_the_same_run(data):
    Y, _, _ = data
    cfg = PINNConfig(n_adam=12, n_lbfgs=0, log_every=1000, balance_every=6, n_collocation=128, n_boundary=32)
    a = train(QUICK, LAW_A, Y, config=cfg, seed=3)
    b = train(QUICK, LAW_A, Y, config=cfg, seed=3)
    assert np.array_equal(a.q, b.q)
    assert np.array_equal(a.T_surface, b.T_surface)
    assert a.weights == b.weights


def test_a_different_seed_is_a_different_run(data):
    """Dispersion across seeds is a published number, so it had better exist."""
    Y, _, _ = data
    cfg = PINNConfig(n_adam=12, n_lbfgs=0, log_every=1000, balance_every=6, n_collocation=128, n_boundary=32)
    a = train(QUICK, LAW_A, Y, config=cfg, seed=3)
    b = train(QUICK, LAW_A, Y, config=cfg, seed=4)
    assert not np.allclose(a.q, b.q)


def test_the_weights_are_balanced_against_the_sensor_term_not_left_at_one(data):
    """Gradient-norm balancing, the plan's 8.3: the terms are unequal by many decades raw."""
    Y, _, _ = data
    cfg = PINNConfig(n_adam=4, n_lbfgs=0, log_every=1000, balance_every=2, n_collocation=128, n_boundary=32)
    r = train(QUICK, LAW_A, Y, config=cfg, seed=0)
    assert r.weights["lambda_r"] != 1.0 and r.weights["lambda_b"] != 1.0
    assert r.weights["lambda_r"] > 0.0 and r.weights["lambda_b"] > 0.0


def test_the_sweep_multiplier_moves_only_the_residual_weight(data):
    """What the three-decade sweep actually varies, held to a decade it can be read against.

    With the weight pinned, the multiplier is exact.  With balancing on it is not, and that is
    not a defect: the multiplier changes the trajectory, the balance is re-measured along it,
    and the two compound.  What must still hold is that the *first* balance -- taken before
    either run has taken a step -- does not depend on the multiplier at all, so the sweep
    starts every configuration from the same measured weight and moves it deliberately.
    """
    Y, _, _ = data
    kw = dict(n_adam=4, n_lbfgs=0, log_every=1000, balance_every=2, n_collocation=128, n_boundary=32)
    pinned = dict(lambda_r=1e-6, lambda_b=1e-2, **kw)
    a = train(QUICK, LAW_A, Y, config=PINNConfig(**pinned), seed=0)
    b = train(QUICK, LAW_A, Y, config=PINNConfig(lambda_r_scale=100.0, **pinned), seed=0)
    assert b.weights["lambda_r"] == pytest.approx(100.0 * a.weights["lambda_r"], rel=1e-12)
    assert b.weights["lambda_b"] == a.weights["lambda_b"]

    c = train(QUICK, LAW_A, Y, config=PINNConfig(**kw), seed=0)
    d = train(QUICK, LAW_A, Y, config=PINNConfig(lambda_r_scale=100.0, **kw), seed=0)
    assert d.history[0]["lambda_r"] == c.history[0]["lambda_r"]


# ======================================================================================
# The output is comparable with the classical methods, by construction
# ======================================================================================


def test_the_flux_is_reported_where_the_classical_methods_report_theirs(data):
    """`metrics.band_error` compares two arrays; they have to be on the same grid to mean anything."""
    from thermalpinn.classical import InverseModel

    Y, _, _ = data
    cfg = PINNConfig(n_adam=2, n_lbfgs=0, log_every=1000, n_collocation=64, n_boundary=16)
    r = train(QUICK, LAW_A, Y, config=cfg, seed=0)
    assert np.allclose(r.t_control, InverseModel(QUICK, LAW_A).control_times())


def test_the_network_answers_at_t_zero_with_the_initial_condition_it_was_given(data):
    Y, _, _ = data
    cfg = PINNConfig(n_adam=2, n_lbfgs=0, log_every=1000, n_collocation=64, n_boundary=16)
    r = train(QUICK, LAW_A, Y, config=cfg, seed=0)
    assert r.T_surface[0] == pytest.approx(QUICK.T0, abs=1e-6)  # kelvin; the clamp floor
    assert r.sensor_fit[0] == pytest.approx(QUICK.T0, abs=1e-6)


def test_a_measurement_record_of_the_wrong_length_is_refused(data):
    Y, _, _ = data
    with pytest.raises(ValueError):
        train(QUICK, LAW_A, Y[:-3], config=PINNConfig(n_adam=1, n_lbfgs=0), seed=0)
