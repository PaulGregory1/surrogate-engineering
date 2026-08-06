import numpy as np
import pytest

from thermalpinn.material import LAW_A, LAW_C
from thermalpinn.naive import (
    _back_region,
    attenuation,
    cutoff_frequency,
    march_divergence,
    measured_attenuation,
    space_march,
    transfer_function,
)
from thermalpinn.scenario import Scenario
from thermalpinn.solver import Grid, forward

COARSE = Grid(nx=201, dt=1e-3)
ALPHA = LAW_A.alpha0


def test_the_transfer_function_is_the_closed_form_it_claims_to_be():
    """Amplitude and phase both decay by beta = d sqrt(pi f / alpha) -- the same number,
    which is why the phase lag is an independent check on the attenuation."""
    f, d = 8.0, 2.0e-3
    beta = d * np.sqrt(np.pi * f / ALPHA)
    g = transfer_function(f, d, ALPHA)
    assert abs(g) == pytest.approx(np.exp(-beta), rel=1e-12)
    assert -np.angle(g) == pytest.approx(beta % (2.0 * np.pi), rel=1e-12)


def test_the_cutoff_is_exactly_where_the_attenuation_reaches_eta():
    for eta in (0.05, 0.1, 0.3):
        for d in (1.0e-3, 2.0e-3, 6.0e-3):
            assert attenuation(cutoff_frequency(d, ALPHA, eta), d, ALPHA) == pytest.approx(eta, rel=1e-10)


def test_the_cutoff_falls_as_one_over_depth_squared():
    """Doubling the depth costs a factor of four in bandwidth: 26 Hz at 1 mm, 6.6 at 2,
    1.7 at 4.  This is the whole reason the 8 Hz ABS ripple is unrecoverable."""
    shallow = cutoff_frequency(1.0e-3, ALPHA)
    assert cutoff_frequency(2.0e-3, ALPHA) == pytest.approx(shallow / 4.0, rel=1e-12)
    assert cutoff_frequency(4.0e-3, ALPHA) == pytest.approx(shallow / 16.0, rel=1e-12)
    assert 6.0 < cutoff_frequency(2.0e-3, ALPHA) < 8.0


def test_the_back_region_agrees_with_the_forward_solver():
    """The march's starting layer comes from the *well posed* half of the domain, so it is
    checked against the solver that owns the whole domain.  Private on purpose -- it is an
    internal step of the march, and a wrong one would flatter it."""
    sc = Scenario(t_end=4.0)
    truth = forward(sc, LAW_A, sc.q_true, grid=COARSE, t_end=4.0, probes=(sc.x_s, sc.x_s + 0.1e-3))
    t = sc.sensor_times()[: int(4.0 * sc.f_sample) + 1]
    measured = np.interp(t, truth.t, truth.at(sc.x_s))
    behind = _back_region(t, measured, sc, LAW_A, 0.1e-3)
    expected = np.interp(t, truth.t, truth.at(sc.x_s + 0.1e-3))
    assert np.abs(behind - expected)[t > 0.2].max() < 0.5  # K, on a 600 K rise


def test_marching_clean_data_recovers_the_surface_the_sensor_cannot_see():
    """The march is not a straw man: with no noise it finds the surface peak, 15 K above
    anything the thermocouple reads, to a fraction of a percent."""
    sc = Scenario(t_end=6.0)
    truth = forward(sc, LAW_A, sc.q_true, grid=COARSE, t_end=6.0)
    t = np.arange(int(6.0 * sc.f_sample) + 1) / sc.f_sample
    clean = np.interp(t, truth.t, truth.at(sc.x_s))
    marched = space_march(t, clean, sc, LAW_A, n_march=20)

    rise = truth.T_surface.max() - sc.T0
    assert abs(marched.T_surface.max() - truth.T_surface.max()) < 0.005 * rise
    assert marched.T_surface.max() > clean.max() + 10.0  # it really did climb off the data


def test_adding_half_a_kelvin_of_noise_destroys_it_and_faster_sampling_destroys_it_more():
    """Ill posed, stated in kelvin: the same code, the same disc, more data, worse answer."""
    sc = Scenario()
    r = march_divergence(sc, LAW_A, grid=COARSE, rates=(100.0, 500.0, 1000.0), n_marches=(20,))
    errors = [row["noisy"]["max_abs_error_K"] for row in r["rows"]]
    assert errors == sorted(errors)
    assert errors[0] < 100.0  # at the sensor's own rate the march is merely wrong
    assert errors[-1] > 1.0e4  # at 1 kHz it is not an answer at all
    # The control that makes the above mean something: clean data is fine at every rate.
    assert max(abs(row["clean"]["T_surface_max_error_pct"]) for row in r["rows"]) < 0.1


def test_the_march_refuses_what_it_cannot_honestly_do():
    sc = Scenario(t_end=1.0)
    t = np.arange(101) / 100.0
    data = np.full_like(t, sc.T0)
    with pytest.raises(ValueError, match="nonlinear"):
        space_march(t, data, sc, LAW_C)
    with pytest.raises(ValueError, match="uniform"):
        space_march(np.sort(np.append(t, 0.005)), np.append(data, sc.T0), sc, LAW_A)


def test_the_measured_attenuation_matches_the_closed_form_in_amplitude_and_in_phase():
    """The bound the repo scores against is measured on the solver, not asserted: one
    multi-tone run, one baseline run, and the tones separate because they are orthogonal
    over the projection window."""
    r = measured_attenuation(grid=COARSE, depths=(1.0e-3, 2.0e-3), settle=2.0, window=2.0, tones=(2, 4, 8, 16))
    assert r["frequencies"] == [1.0, 2.0, 4.0, 8.0]
    for row in r["rows"]:
        assert row["measured"] == pytest.approx(row["analytic"], rel=0.06)
        assert row["phase_lag_measured"] == pytest.approx(row["phase_lag_analytic"], rel=0.06)


def test_the_measured_cutoff_lands_on_the_analytical_one_at_the_sensor_depth():
    """Milestone M2(b), on the coarse grid: the sweep has to bracket 6.6 Hz, so it runs to
    16 Hz, and the crossing has to land within 20 % of alpha ln(1/eta)^2 / pi d^2."""
    r = measured_attenuation(grid=COARSE, depths=(2.0e-3,), settle=2.0, window=2.0, tones=(2, 4, 8, 12, 16, 24, 32))
    p = r["per_depth"][0]
    assert p["penetration_at_f_c_m"] < p["slab_behind_probe_m"]  # half space is legitimate here
    assert abs(p["ratio"] - 1.0) < 0.2
