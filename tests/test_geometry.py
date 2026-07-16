import numpy as np
import torch

from flowsurrogate.geometry import (
    Domain,
    area,
    area_torch,
    level_set,
    level_set_torch,
    r0_for_area,
    radius,
    sample_shapes,
)


def test_numpy_and_torch_level_sets_agree():
    """The optimiser differentiates the torch path; it must draw the same shape."""
    domain = Domain()
    params = sample_shapes(np.random.default_rng(0), 4)
    ref = np.stack([level_set(p, domain) for p in params])
    got = level_set_torch(torch.tensor(params, dtype=torch.float64), domain, dtype=torch.float64).numpy()
    assert np.abs(ref - got).max() < 1e-9


def test_area_closed_form_matches_quadrature():
    params = sample_shapes(np.random.default_rng(1), 6)
    theta = np.linspace(0, 2 * np.pi, 20001)
    for p in params:
        numeric = np.trapezoid(0.5 * radius(theta, p) ** 2, theta)
        assert abs(numeric - area(p)) / area(p) < 1e-6


def test_area_torch_matches_numpy():
    params = sample_shapes(np.random.default_rng(2), 5)
    got = area_torch(torch.tensor(params, dtype=torch.float64)).numpy()
    ref = np.array([area(p) for p in params])
    assert np.allclose(got, ref)


def test_r0_for_area_is_exact():
    """The optimiser relies on the area constraint holding by construction."""
    params = sample_shapes(np.random.default_rng(3), 5)
    target = 300.0
    for p in params:
        rescaled = np.concatenate([[r0_for_area(p[1:], target)], p[1:]])
        assert abs(area(rescaled) - target) / target < 1e-10


def test_level_set_gradient_flows_to_coefficients():
    domain = Domain()
    p = torch.tensor(sample_shapes(np.random.default_rng(4), 1), dtype=torch.float32, requires_grad=True)
    level_set_torch(p, domain).sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0
