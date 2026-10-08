"""Milestone 6 tests for the Option B guidance of fcso/correct.py (SPEC "Guidance, Option B").

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m pytest -s tests/test_guidance.py

Sign test: from the same noisy state x, one guided Euler step must reduce (V - c)^2 more than the unguided
step, in both modes (noisy state, x0). c is set away from V(x) on purpose (as after the noise injection,
where V(x) != V(z_hat)), and lam is set by the calibration rule (guidance norm = 0.3 |d| at that step).
Numbers go to runs/m6/test_results.json.
"""
import json
import os

import pytest
import torch

from fcso import correct as C
from fcso import prior as P
from fcso.volume import Volume

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEBUG = 3


@pytest.fixture(scope="module")
def prior_dm():
    return P.Prior(load_dm=True)


@pytest.fixture(scope="module")
def est(prior_dm):
    return Volume(prior_dm, 64)


@pytest.fixture(scope="module")
def results():
    out = {}
    yield out
    os.makedirs(os.path.join(ROOT, "runs", "m6"), exist_ok=True)
    path = os.path.join(ROOT, "runs", "m6", "test_results.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(out)
    with open(path, "w") as f:
        json.dump(old, f, indent=1)


def _sq(est, x, c):
    return float((est.value(x)[0] - c) ** 2)


@pytest.mark.parametrize("mode", ["noisy", "x0"])
@pytest.mark.parametrize("c_factor", [0.8, 1.2])
def test_guided_step_reduces_sq_more(prior_dm, est, results, mode, c_factor):
    ids = P.read_ids(os.path.join(ROOT, "splits", "opt.txt"))[:DEBUG]
    sig = C.correction_sigmas(0.25, 10).tolist()
    s, s_next = sig[0], sig[1]
    for sid in ids:
        z = prior_dm.encode_id(sid)
        x = z + s * torch.randn(z.shape, generator=torch.Generator().manual_seed(0)).to(z.device)
        c = c_factor * float(est.value(x)[0])  # target off V(x), so (V - c)^2 has room to move
        cal = C.calibrate_lambda(prior_dm, est, z, c, sigma_start=s, seed=0, mode=mode, target_ratio=0.3)
        x_u, _ = C.ode_step(prior_dm, x, s, s_next)
        g = C.VolumeGuidance(prior_dm, est, c, cal["lam"], mode)
        x_g, info = C.ode_step(prior_dm, x, s, s_next, guidance=g)
        sq0, sq_u, sq_g = _sq(est, x, c), _sq(est, x_u, c), _sq(est, x_g, c)
        results[f"sign_{mode}_c{c_factor}_{sid[:8]}"] = dict(lam=cal["lam"], d_norm=info["d_norm"],
                                                             g_norm=info["g_norm"], sq_x=sq0, sq_unguided=sq_u,
                                                             sq_guided=sq_g)
        print(f"{mode} c={c_factor}V {sid[:8]}: lam {cal['lam']:.3g} |d| {info['d_norm']:.2f} "
              f"|g| {info['g_norm']:.2f}  (V-c)^2: x {sq0:.3e}, unguided {sq_u:.3e}, guided {sq_g:.3e}")
        assert sq_g < sq_u, (sid, mode, sq_g, sq_u)
        # and the opposite sign does worse than the right one (the "+" of the SPEC is the right one)
        g_neg = C.VolumeGuidance(prior_dm, est, c, -cal["lam"], mode)
        x_n, _ = C.ode_step(prior_dm, x, s, s_next, guidance=g_neg)
        assert _sq(est, x_n, c) > sq_g


def test_no_grad_reaches_params(prior_dm, est):
    sid = P.read_ids(os.path.join(ROOT, "splits", "opt.txt"))[0]
    z = prior_dm.encode_id(sid)
    for mode in ("noisy", "x0"):
        g = C.VolumeGuidance(prior_dm, est, 0.1, 1.0, mode)
        out = g(z + 0.1 * torch.randn_like(z), 0.1)
        assert torch.isfinite(out).all()
    for m in (prior_dm.ae, prior_dm.dm):
        assert not any(p.requires_grad for p in m.parameters())
        assert all(p.grad is None for p in m.parameters())
