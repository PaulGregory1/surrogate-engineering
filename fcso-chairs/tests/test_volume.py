"""Milestone 4 tests for fcso/volume.py (SPEC "Differentiable volume estimator").

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m pytest -s tests/test_volume.py

Numbers are also written to runs/m4/test_results.json.
"""
import time

import numpy as np
import pytest
import torch

from fcso import mesh_render as R
from fcso import volume as Vol
from fcso.volume import Volume, memory_stats

DEBUG = 3  # first 3 lines of splits/opt.txt


@pytest.fixture(scope="module")
def latents(prior, opt_ids):
    return {sid: prior.encode_id(sid) for sid in opt_ids}


def test_params_frozen(prior):
    assert not any(p.requires_grad for p in prior.ae.parameters())


# ---------------------------------------------------------------------------------------------- gradients
def test_gradcheck_float64(prior64, latents, opt_ids):
    """torch.autograd.gradcheck through the chunked Function (CPU, float64, 6^3 grid, 50-query chunks),
    on a 3-dim slice z(a) = z_s + sum_k a_k u_k. beta = 0.1 so that every query has a non-negligible
    gradient (at beta = 1 most coarse-grid queries are saturated); beta = 1 is checked by finite differences."""
    est = Volume(prior64, res=6, beta=0.1, chunk=50)
    z0 = latents[opt_ids[0]].cpu().double()
    g = torch.Generator().manual_seed(0)
    U = torch.randn(3, *z0.shape[1:], generator=g, dtype=torch.float64)
    U = U / U.flatten(1).norm(dim=1).view(-1, 1, 1)

    def f(a):
        return est(z0 + (a.view(-1, 1, 1) * U).sum(0, keepdim=True))

    a = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (a,), eps=1e-6, atol=1e-8, rtol=1e-5)


def test_finite_difference_beta1(prior64, latents, opt_ids, record):
    """beta = 1, 16^3 grid, CPU float64: directional derivative along the gradient (and along a random
    direction) matches the central difference. V is strongly curved along the gradient on this coarse grid
    (central-difference error 16 % at eps 1e-3, 0.9 % at 1e-4, 9e-5 at 1e-5, 9e-7 at 1e-6: O(eps^2), so the
    analytic gradient is exact), hence the small eps."""
    est = Volume(prior64, res=16, beta=1.0, chunk=1000)
    z0 = latents[opt_ids[2]].cpu().double()
    V, grad = est.value_and_grad(z0)
    assert grad.abs().max() > 0
    out = {}
    gen = torch.Generator().manual_seed(1)
    for name, u, eps in [("grad_dir", grad, 1e-6),
                         ("random_dir", torch.randn(z0.shape, generator=gen, dtype=torch.float64), 1e-5)]:
        u = u / u.norm()
        fd = (est.value(z0 + eps * u) - est.value(z0 - eps * u)) / (2 * eps)
        an = (grad * u).sum()
        rel = float((fd - an).abs() / an.abs().clamp_min(1e-12))
        out[name] = dict(eps=eps, fd=float(fd), analytic=float(an), rel_err=rel)
        assert rel < 1e-5, out
    record["finite_difference_beta1_16cube_f64"] = out


def test_chunked_equals_unchunked(prior, latents, opt_ids, record):
    """MPS float32, 16^3 grid: value_and_grad with 512-query chunks == plain autograd on all queries at
    once; same for the Function path est(z), and for the hook fn(V) = (V - c)^2."""
    est = Volume(prior, res=16, beta=1.0, chunk=512)
    z0 = latents[opt_ids[0]]
    zr = z0.detach().clone().requires_grad_()
    Vr = est.reference(zr)
    Vr.sum().backward()
    V, g = est.value_and_grad(z0)
    torch.testing.assert_close(V, Vr.detach(), rtol=1e-5, atol=1e-7)
    err = float((g - zr.grad).abs().max() / zr.grad.abs().max())
    assert err < 1e-4

    zf = z0.detach().clone().requires_grad_()
    est(zf).sum().backward()
    err_fn = float((zf.grad - zr.grad).abs().max() / zr.grad.abs().max())
    assert err_fn < 1e-4

    c = float(V) * 0.9
    zr.grad = None
    ((est.reference(zr) - c) ** 2).sum().backward()
    V2, g2, F = est.value_and_grad(z0, lambda v: (v - c) ** 2)
    torch.testing.assert_close(F, (V - c) ** 2, rtol=1e-5, atol=1e-9)
    err_hook = float((g2 - zr.grad).abs().max() / zr.grad.abs().max())
    assert err_hook < 1e-4
    # fn(V) = (V - c)^2 is exactly 2 (V - c) dV/dz
    torch.testing.assert_close(g2, 2 * (V - c).view(-1, 1, 1) * g, rtol=1e-4, atol=1e-9)
    record["chunked_vs_unchunked_16cube_mps"] = dict(V=float(V), rel_max_err_grad=err, rel_max_err_function=err_fn,
                                                     rel_max_err_hook=err_hook)


def test_batched_and_through_graph(prior, latents, opt_ids):
    """B = 2 batch gives each chair its own gradient; backward(z) through an upstream graph (stand-in for
    the x0 variant z = D(x; sigma)) accumulates into x.grad like plain autograd."""
    est = Volume(prior, res=16, chunk=1000)
    z = torch.cat([latents[opt_ids[0]], latents[opt_ids[1]]])
    V, g = est.value_and_grad(z)
    for b in range(2):
        Vb, gb = est.value_and_grad(z[b])
        torch.testing.assert_close(V[b:b + 1], Vb, rtol=1e-5, atol=1e-7)
        assert float((g[b] - gb[0]).abs().max() / gb.abs().max()) < 1e-4

    x = latents[opt_ids[0]].detach().clone().requires_grad_()
    c = float(est.value(x)) * 0.8
    V1, F1 = est.backward(torch.tanh(1.1 * x), lambda v: (v - c) ** 2)
    gx = x.grad.clone()
    x.grad = None
    ((est.reference(torch.tanh(1.1 * x)) - c) ** 2).sum().backward()
    assert float((gx - x.grad).abs().max() / x.grad.abs().max()) < 1e-4


# ---------------------------------------------------------------------------------------------- vs mesh
def test_volume_vs_mesh(prior, latents, opt_ids, record):
    """V(z_s) at 64^3, beta = 1, within +-5 % of the trimesh volume of the 128^3 marching-cubes mesh
    (level 0). SPEC asks for 5 opt chairs; all 10 are checked. The vertex-grid variant is logged too."""
    est = Volume(prior, res=64, beta=1.0)
    est_v = Volume(prior, res=64, beta=1.0, centered=False)
    rows = {}
    for sid in opt_ids:
        z = latents[sid]
        mesh = R.logits_to_mesh(prior.decode_grid(z, 128)[0])
        mv = float(mesh.volume)
        v = float(est.value(z))
        vv = float(est_v.value(z))
        est.beta = 10.0
        v10 = float(est.value(z))
        est.beta = 1.0
        rows[sid] = dict(mesh_volume=mv, V64=v, rel=v / mv - 1, V64_beta10=v10, rel_beta10=v10 / mv - 1,
                         V64_vertex_grid=vv, rel_vertex_grid=vv / mv - 1)
        print(f"{sid[:8]} mesh {mv:.4f} V64 {v:.4f} ({100 * (v / mv - 1):+.2f} %)  beta10 {100 * (v10 / mv - 1):+.2f} %"
              f"  vertex-grid {100 * (vv / mv - 1):+.2f} %")
    record["volume_vs_mesh_64"] = rows
    bad = {k[:8]: round(100 * r["rel"], 2) for k, r in rows.items() if abs(r["rel"]) > 0.05}
    assert not bad, bad


def test_res48_vs_64(prior, latents, opt_ids, record):
    """48^3 is allowed during optimization only if it is within 2 % of 64^3 on the 3 debug chairs. The test
    checks that the measured outcome agrees with volume.RES48_ALLOWED (the flag later milestones read)."""
    e64, e48 = Volume(prior, res=64), Volume(prior, res=48)
    rows = {}
    for sid in opt_ids[:DEBUG]:
        v64, v48 = float(e64.value(latents[sid])), float(e48.value(latents[sid]))
        rows[sid] = dict(V64=v64, V48=v48, rel=v48 / v64 - 1)
        print(f"{sid[:8]} V64 {v64:.4f} V48 {v48:.4f} ({100 * (v48 / v64 - 1):+.2f} %)")
    allowed = all(abs(r["rel"]) < 0.02 for r in rows.values())
    record["res48_vs_64"] = dict(rows=rows, allowed=allowed)
    assert allowed == Vol.RES48_ALLOWED


# ---------------------------------------------------------------------------------------------- memory
def test_gradient_step_memory(prior, latents, opt_ids, record):
    """One value+grad step at 64^3 (default 32k chunks): log peak memory, must stay well under 10 GB."""
    z = latents[opt_ids[0]]
    est = Volume(prior, res=64)
    est.value_and_grad(z)  # warm-up
    if prior.device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    before = memory_stats()
    est.trace = {}
    t0 = time.perf_counter()
    V, g = est.value_and_grad(z)
    if prior.device.type == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    after = memory_stats()
    peak_cur = est.trace.get("peak_current_allocated", 0) / 2 ** 30
    out = dict(seconds=dt, before=before, after=after, peak_mps_current_allocated_gb=peak_cur)
    print(out)
    record["gradient_step_memory_64"] = out
    assert np.isfinite(float(V)) and torch.isfinite(g).all()
    assert after.get("mps_driver_gb", 0) < 10 and after["rss_peak_gb"] < 10
