import numpy as np
import torch

from flowsurrogate.data import Normalizer, input_channels
from flowsurrogate.geometry import Domain, sample_shapes
from flowsurrogate.metrics import continuity_residual, masked_relative_l2
from flowsurrogate.models import build, count_parameters
from flowsurrogate.optimize import render_inputs

SMALL = Domain(nx=64, ny=48, cx=20.0, cy=24.0)


def test_input_channels_shape_and_range():
    params = sample_shapes(np.random.default_rng(0), 3)
    re = np.array([40.0, 80.0, 120.0])
    x = input_channels(params, re, SMALL)
    assert x.shape == (3, 3, SMALL.ny, SMALL.nx)
    assert np.abs(x[:, 0]).max() <= 1.0
    assert set(np.unique(x[:, 1])) <= {0.0, 1.0}


def test_normalizer_round_trip():
    rng = np.random.default_rng(0)
    fields = rng.normal(size=(8, 3, 12, 16)).astype(np.float32)
    mask = np.ones((8, 1, 12, 16), dtype=np.float32)
    cd = rng.normal(size=8).astype(np.float32)
    norm = Normalizer.fit(fields, mask, cd)
    back = norm.decode_fields(torch.from_numpy(norm.encode_fields(fields).astype(np.float32)))
    assert torch.allclose(back, torch.from_numpy(fields), atol=1e-4)


def test_models_run_and_have_two_heads():
    x = torch.randn(2, 3, SMALL.ny, SMALL.nx)
    for name, kwargs in [("unet", {"base": 8, "depth": 2}), ("fno", {"width": 8, "modes": 6, "n_layers": 2})]:
        model = build(name, **kwargs)
        fields, cd = model(x)
        assert fields.shape == (2, 3, SMALL.ny, SMALL.nx)
        assert cd.shape == (2,)
        assert count_parameters(model) > 0


def test_render_inputs_is_differentiable_and_matches_hard_mask():
    """The smooth occupancy channel must only differ from the hard one at the boundary."""
    coeffs = torch.zeros(1, 6, requires_grad=True)
    x = render_inputs(coeffs, target_area=300.0, reynolds=80.0, domain=SMALL, mask_eps=0.7)
    x.sum().backward()
    assert coeffs.grad is not None and coeffs.grad.abs().sum() > 0

    # The two masks may only disagree inside a thin band around the boundary: that
    # is the whole point -- same image, but now with a gradient.
    with torch.no_grad():
        x = render_inputs(coeffs, 300.0, 80.0, SMALL, 0.7)
        smooth = x[0, 1]
        hard = (render_inputs(coeffs, 300.0, 80.0, SMALL, 1e-4)[0, 1] > 0.5).float()
        phi = torch.atanh(x[0, 0].clamp(-0.999, 0.999)) * 16.0  # undo the tanh squash
    disagreement = (smooth - hard).abs() > 0.1
    assert phi[disagreement].abs().max() < 2.0
    assert disagreement.sum() > 0


def test_continuity_residual_is_zero_for_a_divergence_free_field():
    ny, nx = 24, 32
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    # u = y, v = -x is divergence free.
    fields = torch.zeros(1, 3, ny, nx)
    fields[0, 0] = torch.tensor(y, dtype=torch.float32)
    fields[0, 1] = torch.tensor(-x, dtype=torch.float32)
    mask = torch.ones(1, 1, ny, nx)
    assert float(continuity_residual(fields, mask, 0.08, 20.0)) < 1e-5


def test_relative_l2_is_zero_for_a_perfect_prediction():
    y = torch.randn(4, 3, 10, 10)
    m = torch.ones(4, 1, 10, 10)
    assert float(masked_relative_l2(y, y, m).max()) < 1e-6
