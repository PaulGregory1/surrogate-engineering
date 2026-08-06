import numpy as np
import pytest

from thermalpinn.material import LAW_A, LAW_B, LAW_C, LAWS, T_REF, to_celsius, to_kelvin

T = np.linspace(T_REF - 50.0, T_REF + 900.0, 401)


@pytest.mark.parametrize("law", LAWS.values(), ids=list(LAWS))
def test_properties_are_normalised_at_the_reference(law):
    """Every law must agree at 20 degC -- the PINN's learned k is held to the same rule."""
    assert law.k(T_REF) == pytest.approx(law.k0, rel=1e-12)
    assert law.c(T_REF) == pytest.approx(law.c0, rel=1e-12)
    assert law.enthalpy(T_REF) == pytest.approx(0.0, abs=1e-12)
    assert law.kirchhoff(T_REF) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("law", LAWS.values(), ids=list(LAWS))
def test_properties_stay_physical(law):
    assert np.all(law.k(T) > 0.0)
    assert np.all(law.c(T) > 0.0)
    assert np.all(np.diff(law.enthalpy(T)) > 0.0)  # heating always stores energy


@pytest.mark.parametrize("law", LAWS.values(), ids=list(LAWS))
def test_dk_dT_matches_finite_differences(law):
    """The Newton solve is only as good as this derivative."""
    eps = 1e-4
    fd = (law.k(T + eps) - law.k(T - eps)) / (2.0 * eps)
    assert np.abs(law.dk_dT(T) - fd).max() < 1e-8 * law.k0


@pytest.mark.parametrize("law", LAWS.values(), ids=list(LAWS))
def test_enthalpy_differentiates_back_to_c(law):
    eps = 1e-3
    fd = (law.enthalpy(T + eps) - law.enthalpy(T - eps)) / (2.0 * eps)
    assert np.abs(fd - law.c(T)).max() < 1e-6 * law.c0


@pytest.mark.parametrize("law", LAWS.values(), ids=list(LAWS))
def test_kirchhoff_round_trip_and_derivative(law):
    assert np.abs(law.kirchhoff_inverse(law.kirchhoff(T)) - T).max() < 1e-6
    # The step straddles many table cells on purpose: the knee law integrates k
    # numerically, so differentiating its piecewise-linear interpolant across a single
    # cell measures the interpolation and not the transform.
    eps = 0.25
    du = (law.kirchhoff(T + eps) - law.kirchhoff(T - eps)) / (2.0 * eps)
    assert np.abs(du - law.k(T) / law.k0).max() < 1e-5


def test_law_a_is_the_linear_one():
    assert LAW_A.is_linear and not LAW_A.radiation
    assert np.allclose(LAW_A.k(T), LAW_A.k0)
    assert np.allclose(LAW_A.kirchhoff(T), T - T_REF)  # Kirchhoff is the identity here
    assert LAW_A.alpha0 == pytest.approx(1.57e-5, rel=0.02)


def test_law_b_is_smooth_and_radiating():
    assert LAW_B.radiation and not LAW_B.is_linear
    assert np.all(np.diff(LAW_B.k(T)) < 0.0)  # conductivity falls with temperature
    assert np.all(np.diff(LAW_B.c(T)) > 0.0)  # specific heat rises
    # smooth means no curvature: the second difference of k is zero
    assert np.abs(np.diff(LAW_B.k(T), n=2)).max() < 1e-10


def test_law_c_has_a_knee_of_the_advertised_size_and_place():
    """The -12 % step at 450 degC is the whole point of act 3; check it is really there."""
    below, above = to_kelvin(450.0) - 150.0, to_kelvin(450.0) + 150.0
    ratio = lambda t: LAW_C.k(t) / LAW_B.k(t)  # noqa: E731
    assert ratio(below) == pytest.approx(1.0, abs=1e-3)
    assert ratio(above) == pytest.approx(0.88, abs=1e-3)
    # steepest descent of the ratio sits at the knee, and the transition is localised
    grid = np.linspace(below, above, 3001)
    slope = np.gradient(ratio(grid), grid)
    assert to_celsius(grid[np.argmin(slope)]) == pytest.approx(450.0, abs=1.0)
    inside = np.abs(np.diff(ratio(grid))) > 0.01 * np.abs(np.diff(ratio(grid))).max()
    assert np.ptp(grid[1:][inside]) < 4.0 * LAW_C.knee_width


def test_law_c_is_not_a_member_of_the_families_the_classical_route_will_fit():
    """A quadratic in T cannot follow the knee -- that gap is what act 3 measures."""
    band = np.linspace(T_REF, T_REF + 800.0, 401)
    k = LAW_C.k(band)
    residual = k - np.polyval(np.polyfit(band, k, 2), band)
    assert np.abs(residual).max() / LAW_C.k0 > 0.01
