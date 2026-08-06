import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from thermalpinn.classical import (
    InverseModel,
    alifanov,
    beck,
    beck_by_discrepancy,
    inverse_crime_data,
    tikhonov,
)
from thermalpinn.material import LAW_A, LAW_B, LAW_C, LAWS
from thermalpinn.metrics import band_error, k_curve_error, lowpass, surface_peak_error
from thermalpinn.naive import cutoff_frequency
from thermalpinn.scenario import Scenario
from thermalpinn.solver import TRUTH_GRID, forward, sensor_signal

SHORT = Scenario(t_end=2.0)
F_C = cutoff_frequency(SHORT.x_s, LAW_A.alpha0)
DT_CONTROL = 1.0 / SHORT.f_sample


@pytest.fixture(scope="module")
def inverted():
    """All three baselines run once on the same realistic data, for the tests that compare
    them.  The truth comes from the fine grid through the real measurement chain, so this is
    not the inverse crime -- that has its own test."""
    model = InverseModel(SHORT, LAW_A)
    truth = forward(SHORT, LAW_A, SHORT.q_true, grid=TRUTH_GRID, t_end=SHORT.t_end)
    _, Y, clean = sensor_signal(truth, SHORT, rng=np.random.default_rng(0))
    duhamel = model.step_response()
    q_ref = SHORT.q_true(model.control_times())
    return {
        "model": model,
        "truth": truth,
        "Y": Y,
        "clean": clean,
        "q_ref": q_ref,
        "estimates": {
            "beck": beck_by_discrepancy(model, Y, sigma=SHORT.sigma_noise, duhamel=duhamel)["q"],
            "tikhonov": tikhonov(model, Y, sigma=SHORT.sigma_noise, duhamel=duhamel)["q"],
            "alifanov": alifanov(model, Y, sigma=SHORT.sigma_noise, max_iter=40)["q"],
        },
    }


# ======================================================================================
# The model the baselines share
# ======================================================================================


@pytest.mark.parametrize("law", LAWS.values(), ids=list(LAWS))
def test_the_inverse_model_agrees_with_the_truth_it_is_not_allowed_to_share_a_grid_with(law):
    """The coarse model has to be a modelling error, not the dominant one.  Its gap to the
    truth is the floor no inverse method can beat, so it has to sit far below the 0.5 K of
    measurement noise."""
    model = InverseModel(SHORT, law)
    truth = forward(SHORT, law, SHORT.q_true, grid=TRUTH_GRID, t_end=SHORT.t_end)
    obs = model.solve(SHORT.q_true(model.control_times()), store=False).obs
    expected = np.interp(model.measurement_times(), truth.t, truth.at(SHORT.x_s))
    assert np.abs(obs - expected).max() < 0.1  # K, against a rise of several hundred


@pytest.mark.parametrize("law", LAWS.values(), ids=list(LAWS))
def test_the_adjoint_is_the_gradient_of_the_model_it_belongs_to(law):
    """Milestone M3(b), and the check the rest of the repo rests on: acts 2 and 3 are
    descents of this gradient, so if it is wrong every result after it is decoration.

    The discrete adjoint differentiates the scheme rather than the equation, which is why
    the tolerance can be this tight.  Law A is exactly linear in the flux, so the central
    difference is exact there and only round-off separates them; laws B and C are limited by
    the O(eps^2) truncation of the difference, not by the adjoint.
    """
    tiny = Scenario(t_end=0.6)
    model = InverseModel(tiny, law)
    q_true = tiny.q_true(model.control_times())
    Y = model.solve(q_true, store=False).obs + np.random.default_rng(1).normal(0.0, 0.5, model.n_control + 1)

    q = 0.7 * q_true
    run = model.solve(q)
    g = model.gradient(run.field, run.obs[1:] - Y[1:])
    assert np.linalg.norm(g) > 0.0

    def objective(candidate):
        r = model.solve(candidate, store=False).obs[1:] - Y[1:]
        return 0.5 * float(r @ r)

    eps = 1.0e-2 * tiny.q_peak
    for j in np.random.default_rng(2).choice(model.n_control, size=5, replace=False):
        up, down = q.copy(), q.copy()
        up[j] += eps
        down[j] -= eps
        fd = (objective(up) - objective(down)) / (2.0 * eps)
        assert abs(fd - g[j]) / abs(fd) < (1e-9 if law.is_linear else 1e-6)


@pytest.mark.parametrize("law", [LAW_A, LAW_C], ids=["A", "C"])
def test_the_tangent_model_is_the_derivative_of_the_forward_model(law):
    """The conjugate-gradient step length is exact only if this is, and it shares its
    matrices with the adjoint -- so the two checks together pin the whole gradient."""
    tiny = Scenario(t_end=0.6)
    model = InverseModel(tiny, law)
    q = tiny.q_true(model.control_times())
    run = model.solve(q)
    d = np.random.default_rng(3).normal(0.0, 1.0e5, model.n_control)

    eps = 1e-4
    fd = (model.solve(q + eps * d, store=False).obs - model.solve(q - eps * d, store=False).obs) / (2.0 * eps)
    linear = model.tangent(run.field, d)
    assert np.abs(linear - fd).max() / np.abs(fd).max() < 1e-6


def test_duhamel_superposition_reproduces_the_full_solve_on_the_linear_law():
    """The whole of Beck and Tikhonov rests on this identity, so it is checked directly
    rather than trusted: baseline plus sensitivity matrix times flux *is* the model.

    The Toeplitz structure additionally assumes the scheme is time invariant, which the
    backward-Euler startup breaks.  So the identity is checked twice: exactly, with the
    startup switched off, and then at the default startup, where the violation has to stay
    far below the 0.5 K of measurement noise or Beck and Tikhonov would be inverting a model
    that is not the one being solved.
    """
    q = SHORT.q_true(InverseModel(SHORT, LAW_A).control_times())

    def mismatch(n_startup):
        model = InverseModel(SHORT, LAW_A, n_startup=n_startup)
        S, baseline = model.step_response()
        return np.abs(baseline[1:] + model.duhamel_matrix(S) @ q - model.solve(q, store=False).obs[1:]).max()

    assert mismatch(0) < 1e-9  # exactly time invariant, so the sum is exact
    assert mismatch(InverseModel(SHORT, LAW_A).n_startup) < 0.01  # 50x below the noise


@pytest.mark.parametrize("law", [LAW_B, LAW_C], ids=["B", "C"])
def test_duhamel_refuses_the_laws_it_does_not_hold_for(law):
    with pytest.raises(ValueError, match="superposition"):
        InverseModel(SHORT, law).step_response()


# ======================================================================================
# The three milestone criteria
# ======================================================================================


def test_the_direct_methods_invert_the_inverse_crime():
    """Milestone M3(a) for the two closed-form methods.  Zero noise, the model's own grid: a
    method that cannot recover the flux here is broken, and only one that can is worth
    measuring on real data.

    Alifanov is held to the same 1 % in `scripts/02_run_methods.py` rather than here, because
    conjugate gradients needs a couple of hundred iterations to reach it on noise-free data
    and that is a minute of arithmetic, not a second.  Its convergence towards the crime data
    is checked below instead, and its correctness is pinned by the gradient and tangent tests
    far more tightly than an accuracy threshold could.
    """
    model = InverseModel(SHORT, LAW_A)
    duhamel = model.step_response()
    q_crime, Y_crime = inverse_crime_data(model)
    estimates = {
        "beck": beck(model, Y_crime, stride=2, r=4, duhamel=duhamel)["q"],
        "tikhonov": tikhonov(model, Y_crime, sigma=1e-9, duhamel=duhamel)["q"],
    }
    for name, q in estimates.items():
        assert band_error(q, q_crime, DT_CONTROL, F_C)["rmse_pct"] < 1.0, name


def test_beck_cannot_be_run_unregularised_at_all():
    """Not a limitation of this implementation but of the method, and worth pinning: the
    sequential recursion amplifies by roughly S_2 / S_1 per step, and at 2 mm depth the first
    future measurement carries almost nothing (S_1 = 1.6e-9 K per W/m^2).  One flux value per
    measurement with the shortest look-ahead does not merely lose accuracy, it diverges along
    the record -- on noise-free data, which is the point."""
    model = InverseModel(SHORT, LAW_A)
    duhamel = model.step_response()
    q_crime, Y_crime = inverse_crime_data(model)
    with np.errstate(over="ignore", invalid="ignore"):
        reckless = beck(model, Y_crime, stride=1, r=2, duhamel=duhamel)["q"]
        careful = beck(model, Y_crime, stride=2, r=4, duhamel=duhamel)["q"]
    assert np.max(np.abs(reckless)) > 1e3 * SHORT.q_peak
    assert np.max(np.abs(careful)) < 2.0 * SHORT.q_peak


def test_alifanov_keeps_closing_on_the_crime_data_as_it_iterates():
    """The cheap half of M3(a) for the iterative method: given no noise to stop on, more
    iterations have to keep buying accuracy.  A method that stalled here would be converging
    to something other than the answer."""
    model = InverseModel(SHORT, LAW_A)
    q_crime, Y_crime = inverse_crime_data(model)
    errors = []
    q = None
    for _ in range(3):
        q = alifanov(model, Y_crime, sigma=0.0, q_init=q, max_iter=25)["q"]
        errors.append(band_error(q, q_crime, DT_CONTROL, F_C)["rmse_pct"])
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 5.0


def test_the_three_methods_land_within_a_factor_two_of_each_other(inverted):
    """Milestone M3(c).  A baseline three times worse than its neighbour is a detuned
    baseline, and beating one of those would prove nothing."""
    scores = {
        name: band_error(q, inverted["q_ref"], DT_CONTROL, F_C)["rmse_pct"]
        for name, q in inverted["estimates"].items()
    }
    assert max(scores.values()) / min(scores.values()) <= 2.0, scores


def test_the_classical_methods_find_the_surface_peak_on_the_linear_law(inverted):
    """Act 1 is meant to be a fair fight, and this is the bar: with superposition valid and
    the material known, the classical route reads a surface it cannot see to within a
    percent.  The PINN has to match this, not a straw man."""
    model, truth = inverted["model"], inverted["truth"]
    for name, q in inverted["estimates"].items():
        surface = model.solve(q).field[:, 0]
        assert abs(surface_peak_error(surface, truth.T_surface, SHORT.T0)) < 1.5, name


# ======================================================================================
# Method-specific behaviour
# ======================================================================================


def test_beck_refuses_a_look_ahead_shorter_than_its_own_flux_step():
    model = InverseModel(SHORT, LAW_A)
    with pytest.raises(ValueError, match="shorter than the flux step"):
        beck(model, np.zeros(model.n_control + 1), stride=5, r=3, duhamel=(np.zeros(model.n_control + 1), np.zeros(model.n_control + 1)))


def test_beck_leaves_the_final_look_ahead_window_unresolved_and_says_so():
    """It cannot answer for the last r intervals -- the data it would need are past the end
    of the record -- and reporting that is better than filling it in silently."""
    model = InverseModel(SHORT, LAW_A)
    out = beck(model, inverse_crime_data(model)[1], stride=5, r=10)
    assert 0 < out["unresolved_tail"] <= out["r"]
    assert np.all(out["q"][-out["unresolved_tail"] :] == out["q"][-out["unresolved_tail"] - 1])


def test_the_discrepancy_principle_picks_the_smoothest_setting_that_still_fits(inverted):
    """Both knobs point the same way -- more smoothing fits worse -- so the rule is to take
    the largest still consistent with the noise.  Taking the smallest instead would select
    pure noise amplification, which is how the sweep is checked for direction."""
    model, Y = inverted["model"], inverted["Y"]
    out = beck_by_discrepancy(model, Y, sigma=SHORT.sigma_noise)
    fits = [row for row in out["sweep"] if np.isfinite(row["residual_rms"]) and row["residual_rms"] <= SHORT.sigma_noise]
    assert out["stride"] == max(row["stride"] for row in fits)
    # Direction: over the stable part of the sweep, more smoothing fits worse, monotonically.
    # stride = 1 is left out because it does not merely fit worse, it diverges.
    stable = [row["residual_rms"] for row in out["sweep"] if row["stride"] > 1]
    assert stable == sorted(stable)


def test_the_discrepancy_principle_reproduces_the_noise_it_was_given(inverted):
    """The whole content of the rule: stop tightening the fit once the residual is the size
    of the noise.  If this drifts, every lambda in the repo is chosen by something else."""
    out = tikhonov(inverted["model"], inverted["Y"], sigma=SHORT.sigma_noise)
    assert out["residual_rms"] == pytest.approx(SHORT.sigma_noise, rel=0.25)


def test_the_l_curve_fails_on_this_problem_and_the_repo_says_which_choice_it_uses(inverted):
    """A finding, kept as a test so it cannot rot silently.

    The L-curve is the standard way to choose lambda without knowing the noise level, and on
    this problem it does not work: there is no corner to find, because the penalty axis spans
    five decades while the residual spans under two.  The corner it settles on is decades too
    small and the flux it returns is badly under-regularised.

    So the headline uses the discrepancy principle for a measured reason rather than a
    stylistic one -- and 'the classical route needs a tuning parameter' turns out to
    understate it: the usual way of setting that parameter blind fails here, and knowing
    sigma is what rescues it.
    """
    out = tikhonov(inverted["model"], inverted["Y"], sigma=SHORT.sigma_noise)
    assert out["lambda"] > 0.0 and out["lambda_l_curve"] > 0.0
    assert out["lambda_l_curve"] < 0.01 * out["lambda"]  # decades below: under-regularised

    scored = lambda q: band_error(q, inverted["q_ref"], DT_CONTROL, F_C)["rmse_pct"]  # noqa: E731
    assert scored(out["q_l_curve"]) > 1.3 * scored(out["q"])
    # Rougher, which is the actual complaint, and by much more than the error suggests.
    roughness = lambda q: float(np.linalg.norm(np.diff(q, 2)))  # noqa: E731
    assert roughness(out["q_l_curve"]) > 10.0 * roughness(out["q"])


def test_alifanov_stops_because_it_has_explained_the_noise_not_because_it_ran_out_of_budget(inverted):
    """The stopping rule *is* the regularisation here, so hitting the iteration cap would
    mean the answer was decided by the budget."""
    out = alifanov(inverted["model"], inverted["Y"], sigma=SHORT.sigma_noise, max_iter=200)
    assert out["stop"] == "discrepancy"
    assert out["iterations"] < 200
    sse = [h["sse"] for h in out["history"]]
    assert sse == sorted(sse, reverse=True)  # a descent method has to descend


@pytest.mark.parametrize("law", [LAW_B, LAW_C], ids=["B", "C"])
def test_alifanov_needs_neither_superposition_nor_a_linear_material(law):
    """The reason it is the baseline that matters: laws B and C cost it nothing but time,
    while Beck and Tikhonov cannot be posed at all."""
    model = InverseModel(SHORT, law)
    truth = forward(SHORT, law, SHORT.q_true, grid=TRUTH_GRID, t_end=SHORT.t_end)
    _, Y, _ = sensor_signal(truth, SHORT, rng=np.random.default_rng(0))
    out = alifanov(model, Y, sigma=SHORT.sigma_noise, max_iter=60)
    q_ref = SHORT.q_true(model.control_times())
    assert band_error(out["q"], q_ref, DT_CONTROL, F_C)["rmse_pct"] < 10.0


# ======================================================================================
# The metrics, and the honesty rules they encode
# ======================================================================================


def test_nothing_is_scored_on_what_the_depth_of_the_sensor_destroyed(inverted):
    """Honesty rule 4, as a test.  Adding a component the sensor could never have seen must
    not move the score -- otherwise the metric would be rewarding invention.

    It holds in the interior, which is where the score is taken, and it does *not* hold at the
    boundary: a finite record cannot be band limited at its ends, and about a tenth of the
    added component leaks in there.  That is exactly why the margin exists, and the test
    checks both halves of the claim so the margin cannot quietly be removed.
    """
    q = inverted["estimates"]["tikhonov"]
    t = inverted["model"].control_times()
    polluted = q + 0.3 * SHORT.q_peak * np.sin(2.0 * np.pi * 30.0 * t)  # 30 Hz, far outside

    before = band_error(q, inverted["q_ref"], DT_CONTROL, F_C)
    after = band_error(polluted, inverted["q_ref"], DT_CONTROL, F_C)
    assert after["rmse_pct"] == pytest.approx(before["rmse_pct"], rel=0.05)
    assert before["out_of_band_rms"] > 0.0  # and the truth's own lost part is reported

    unscored = band_error(polluted, inverted["q_ref"], DT_CONTROL, F_C, edge_margin=0)
    assert unscored["rmse"] > 1.5 * after["rmse"]  # the boundary really is the reason


def test_the_verdict_does_not_depend_on_which_low_pass_was_chosen(inverted):
    """The filter is a convention, so the conclusion must not turn on it -- but the
    conclusion is 'these three methods are within a factor of two of each other', not an
    ordering.  They are separated by well under their own spread, so which one comes first
    *is* noise, and a test that pinned the ordering would be pinning the noise.

    A Gaussian matched to the Butterworth at its -3 dB point rolls off far more gently, so it
    admits more out-of-band content and scores everyone worse.  What has to survive is the
    verdict: nobody is a factor of two from anybody, under either filter.
    """
    sigma_samples = 0.8326 / (2.0 * np.pi * F_C * DT_CONTROL)  # matched at -3 dB
    margin = int(round(1.0 / (F_C * DT_CONTROL)))  # the same interior the Butterworth is scored on

    def gaussian_score(q):
        est = gaussian_filter1d(q, sigma_samples, mode="nearest")[margin:-margin]
        ref = gaussian_filter1d(inverted["q_ref"], sigma_samples, mode="nearest")[margin:-margin]
        return 100.0 * np.sqrt(np.mean((est - ref) ** 2)) / np.sqrt(np.mean(ref**2))

    butter = {n: band_error(q, inverted["q_ref"], DT_CONTROL, F_C)["rmse_pct"] for n, q in inverted["estimates"].items()}
    gauss = {n: gaussian_score(q) for n, q in inverted["estimates"].items()}
    for scores in (butter, gauss):
        assert max(scores.values()) / min(scores.values()) <= 2.0, scores
    # The magnitudes are allowed to move, and do -- by about two.  Matching two filters at
    # their -3 dB point does not make them the same band, and the estimates carry enough
    # content above the cutoff that the gentler roll-off admits visibly more of it.  What
    # must not happen is a method being misjudged by more than that.
    for name in butter:
        assert gauss[name] == pytest.approx(butter[name], rel=2.0), (name, butter, gauss)


def test_the_headline_error_is_relative_to_the_rise_not_the_absolute_temperature():
    """Dividing by 900 K of absolute temperature instead of the 600 K rise would flatter
    every method by the same factor of three, which is exactly the kind of quiet choice the
    repo is not allowed to make."""
    T0 = 313.15
    true_trace = np.array([T0, T0 + 600.0, T0 + 100.0])
    assert surface_peak_error(np.array([T0, T0 + 660.0]), true_trace, T0) == pytest.approx(10.0)
    assert surface_peak_error(np.array([T0, T0 + 540.0]), true_trace, T0) == pytest.approx(-10.0)


def test_the_low_pass_leaves_a_signal_already_in_band_alone():
    t = np.arange(0.0, 4.0, DT_CONTROL)
    slow = 1.0 + 0.5 * np.sin(2.0 * np.pi * 0.2 * t)
    assert np.abs(lowpass(slow, DT_CONTROL, F_C) - slow).max() < 0.02


def test_the_conductivity_curve_error_is_relative_and_scale_free():
    k_true = np.linspace(40.0, 52.0, 50)
    assert k_curve_error(k_true, k_true) == 0.0
    assert k_curve_error(1.1 * k_true, k_true) == pytest.approx(10.0)
