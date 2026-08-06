import dataclasses

import numpy as np
import pytest

from thermalpinn.material import LAW_A, LAW_B, LAW_C, LAWS
from thermalpinn.scenario import Scenario
from thermalpinn.solver import INVERSE_GRID, TRUTH_GRID, Grid, forward, sensor_signal
from thermalpinn.verify import Manufactured, mms_convergence, semi_infinite, semi_infinite_check

COARSE = Grid(nx=101, dt=1e-3)
ADIABATIC = Scenario(h=0.0, eps=0.0)


def test_nothing_happens_when_nothing_is_applied():
    sol = forward(ADIABATIC, LAW_A, 0.0, grid=COARSE, t_end=1.0)
    assert np.abs(sol.T_field - ADIABATIC.T0).max() < 1e-9


@pytest.mark.parametrize("law", LAWS.values(), ids=list(LAWS))
def test_energy_balance_closes_to_round_off(law):
    """Conservation is exact by construction; a nonzero residual means the boundary
    condition, the control volumes or the enthalpy disagree with each other."""
    sc = Scenario(t_end=2.0)
    sol = forward(sc, law, sc.q_true, grid=COARSE, t_end=2.0)
    assert sol.energy_residual < 1e-10


def test_long_time_gradient_matches_the_analytical_steady_profile():
    """Constant flux into an insulated slab settles on dT/dt = q/(rho c L) everywhere,
    which fixes the surface-to-midplane difference at q L / 2k."""
    q0 = 1.0e5
    sol = forward(ADIABATIC, LAW_A, q0, grid=COARSE, t_end=10.0, probes=(0.0, ADIABATIC.L))
    drop = sol.at(0.0)[-1] - sol.at(ADIABATIC.L)[-1]
    assert drop == pytest.approx(q0 * ADIABATIC.L / (2.0 * LAW_A.k0), rel=2e-3)
    mean_rise = (sol.energy[-1] - sol.energy[0]) / (LAW_A.rho * LAW_A.c0 * ADIABATIC.L)
    assert mean_rise == pytest.approx(q0 * 10.0 / (LAW_A.rho * LAW_A.c0 * ADIABATIC.L), rel=1e-9)


def test_matches_the_semi_infinite_solution_while_the_wave_is_still_far_from_the_wall():
    r = semi_infinite_check(grid=Grid(nx=401, dt=2e-4))
    assert r["n_counted"] >= 10
    assert r["max_abs_error_pct"] < 0.5


def test_second_order_in_space_on_the_nonlinear_law():
    r = mms_convergence(LAW_C, space_nx=(26, 51, 101), space_dt=1e-4, time_steps=())
    assert 1.9 <= r["space"]["order_l2"] <= 2.1
    assert 1.9 <= r["space"]["order_linf"] <= 2.1


def test_halving_dt_quarters_the_error():
    """Cheap stand-in for the full temporal study in scripts/01_validate_solver.py."""
    sc = Scenario()
    mms = Manufactured(L=sc.L, period=0.4)
    kw = dict(
        source=lambda xx, tt: mms.source(LAW_B, xx, tt),
        T_init=lambda xx: mms.T(xx, 0.0),
        t_end=0.4,
        n_snapshots=2,
    )
    run = lambda steps: forward(  # noqa: E731
        sc, LAW_B, lambda tt: mms.flux(LAW_B, sc, tt), grid=Grid(nx=51, dt=0.4 / steps), **kw
    ).T_field[-1]
    ref = run(32000)
    coarse = np.abs(run(1600) - ref).max()
    fine = np.abs(run(3200) - ref).max()
    assert coarse / fine == pytest.approx(4.0, rel=0.1)


def test_radiation_actually_removes_energy():
    """Law B radiates and law A does not; at 800 K that has to show."""
    hot = Scenario(T0=800.0, t_end=1.0)
    no_rad = dataclasses.replace(LAW_B, name="B-norad", radiation=False)
    with_rad = forward(hot, LAW_B, 0.0, grid=COARSE, t_end=1.0)
    without = forward(hot, no_rad, 0.0, grid=COARSE, t_end=1.0)
    assert with_rad.q_net[0] < without.q_net[0] < 0.0
    assert with_rad.T_surface[-1] < without.T_surface[-1]


def test_the_two_grids_agree_on_the_quantity_the_whole_repo_is_about():
    """The inverse model deliberately does not share the truth's discretisation.  That
    gap has to be small enough to be a modelling error and not the dominant one."""
    sc = Scenario(t_end=1.0)
    truth = forward(sc, LAW_A, sc.q_true, grid=TRUTH_GRID, t_end=1.0)
    inverse = forward(sc, LAW_A, sc.q_true, grid=INVERSE_GRID, t_end=1.0)
    assert abs(truth.T_surface.max() - inverse.T_surface.max()) < 1.0
    assert abs(truth.at(sc.x_s).max() - inverse.at(sc.x_s).max()) < 0.1


def test_probes_interpolate_and_refuse_positions_they_were_not_given():
    sc = Scenario(t_end=0.2)
    mid = 0.5 * (sc.x_s + sc.L)
    sol = forward(sc, LAW_A, sc.q_true, grid=COARSE, t_end=0.2, probes=(0.0, sc.x_s, mid, sc.L))
    assert np.all(sol.at(0.0) >= sol.at(sc.x_s))  # heat flows inwards
    assert np.all(sol.at(sc.x_s) >= sol.at(mid))
    assert np.all(sol.at(mid) >= sol.at(sc.L))
    with pytest.raises(KeyError):
        sol.at(0.123e-3)


def test_sensor_signal_has_the_advertised_rate_and_noise():
    sc = Scenario(t_end=2.0, sigma_noise=0.5)
    sol = forward(sc, LAW_A, sc.q_true, grid=COARSE, t_end=2.0)
    ts, noisy, clean = sensor_signal(sol, sc, rng=np.random.default_rng(0))
    assert ts.size == noisy.size == clean.size == 201
    assert np.std(noisy - clean) == pytest.approx(0.5, rel=0.15)
    # the same seed must give the same measurement, or nothing downstream reproduces
    assert np.array_equal(noisy, sensor_signal(sol, sc, rng=np.random.default_rng(0))[1])


def test_the_analytical_reference_is_the_one_it_claims_to_be():
    """Guard the closed form itself: at x = 0 it must reduce to 2 q sqrt(alpha t / pi)/k."""
    t = 0.02
    got = semi_infinite(1e6, LAW_A, np.array(0.0), t, 0.0)
    assert float(got) == pytest.approx(2e6 * np.sqrt(LAW_A.alpha0 * t / np.pi) / LAW_A.k0, rel=1e-12)


def test_newton_survives_the_knee_at_full_braking_power():
    sc = Scenario(t_end=2.0)
    sol = forward(sc, LAW_C, sc.q_true, grid=COARSE, t_end=2.0)
    assert np.isfinite(sol.T_field).all()
    assert sol.T_surface.max() > LAW_C.knee_T  # the run really does cross the 450 degC knee
    assert sol.newton_iterations < 4 * int(2.0 / COARSE.dt)  # and Newton is not thrashing
