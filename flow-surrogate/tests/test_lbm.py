import numpy as np
import pytest

from flowsurrogate.geometry import Domain
from flowsurrogate.lbm import LBMSolver, SolverConfig

SMALL = Domain(nx=64, ny=48, cx=20.0, cy=24.0)


def _cylinder(domain, radius=6.0):
    x, y = domain.grid()
    return ((x - domain.cx) ** 2 + (y - domain.cy) ** 2) < radius**2


def _solver(steps=400, **kw):
    return LBMSolver(SMALL, SolverConfig(n_steps=steps, n_avg=steps // 4), device="cpu", **kw)


def test_empty_channel_stays_uniform():
    """With no obstacle the solution must be the inlet velocity everywhere."""
    masks = np.zeros((1, SMALL.ny, SMALL.nx), dtype=bool)
    out = _solver().run(masks, np.array([50.0]), np.array([12.0]))
    u_in = SolverConfig().u_in
    assert np.abs(out["u"] - u_in).max() < 1e-4
    assert np.abs(out["v"]).max() < 1e-6
    assert abs(float(out["cd"][0])) < 1e-9  # no body, no drag


def test_symmetric_body_produces_no_lift():
    masks = _cylinder(SMALL)[None]
    out = _solver(600).run(masks, np.array([40.0]), np.array([12.0]))
    assert abs(float(out["cl"][0])) < 1e-3 * abs(float(out["cd"][0]))
    assert float(out["cd"][0]) > 0  # drag opposes the flow


def test_drag_decreases_with_reynolds():
    masks = np.stack([_cylinder(SMALL)] * 3)
    out = _solver(1200).run(masks, np.array([20.0, 40.0, 80.0]), np.full(3, 12.0))
    cd = out["cd"]
    assert cd[0] > cd[1] > cd[2]


def test_no_flow_inside_the_body():
    masks = _cylinder(SMALL)[None]
    out = _solver(600).run(masks, np.array([40.0]), np.array([12.0]))
    assert np.abs(out["u"][0][masks[0]]).max() == 0.0
    assert np.abs(out["v"][0][masks[0]]).max() == 0.0


@pytest.mark.parametrize("field", ["u", "cd"])
def test_compiled_matches_eager(field):
    masks = _cylinder(SMALL)[None]
    args = (masks, np.array([40.0]), np.array([12.0]))
    eager = LBMSolver(SMALL, SolverConfig(n_steps=300, n_avg=100), device="cpu", compile=False).run(*args)
    comp = LBMSolver(SMALL, SolverConfig(n_steps=300, n_avg=100), device="cpu", compile=True).run(*args)
    assert np.abs(eager[field] - comp[field]).max() < 1e-5 * max(1.0, np.abs(eager[field]).max())
