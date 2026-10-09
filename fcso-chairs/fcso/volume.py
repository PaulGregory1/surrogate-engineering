"""Differentiable volume estimator (SPEC "Differentiable volume estimator", paper Eq. 16 and App. F.1).

    V(z) = |B| * mean_{x in G} sigmoid(beta * logit(x, z))

with G a uniform res^3 grid over the normalization box [-1, 1]^3, |B| = 8, logits positive inside
(eval.py), beta = 1 by default. G is cell-centered (points -1 + (i + 1/2) * 2/res): |B| * mean is then
the midpoint rule over the box. The vertex grid linspace(-1, 1, res) used for meshes (`prior.make_grid`)
would bias |B| * mean by (1 - 1/res)^3 (-4.6 % at 64^3) since its res^3 cells cover a box of side
2 res/(res-1); pass centered=False to get it anyway (DECISIONS.md, milestone 4).

Chunked two-stage backward (SPEC), exact because V is a sum over queries:
  1. h = trunk(z) = 24 self-attention blocks over proj(z), computed once, with its graph;
  2. h_leaf = h.detach().requires_grad_();
  3. for each chunk of queries: contribution = |B|/|G| * sum sigmoid(beta * head(h_leaf, chunk)),
     .backward() -> dV/dh accumulates in h_leaf.grad, the chunk's graph is freed;
  4. h.backward(w * h_leaf.grad) propagates to z (and to whatever z depends on, e.g. x through the
     denoiser for the x0 guidance variant), with w = dF/dV for an outer objective F(V) (w = 1 for V itself).

API (all accept z of shape (B, 512, 8) or (512, 8); V_b only depends on z_b):
  est = Volume(prior, res=64)
  est.value(z)                      -> V (B,), no graph (no_grad)
  est.value_and_grad(z)             -> V (B,), dV/dz (B, 512, 8)
  est.value_and_grad(z, fn)         -> V, d(sum fn(V))/dz, fn(V)      e.g. fn = lambda V: (V - c) ** 2
  est.backward(z, fn)               -> V, fn(V); accumulates d(sum fn(V)) into the .grad of z's graph leaves
  est(z)                            -> V as a differentiable tensor usable in any autograd expression
                                       (same chunked mechanism inside a torch.autograd.Function)
Model parameters are frozen by `prior.Prior` (requires_grad=False): gradients only reach latents.
"""
import resource

import torch

from fcso.prior import BOUND, make_grid

EVAL_RES = 64  # thresholds, progress r and reported volumes (SPEC)
# SPEC: 48^3 may be used during optimization only if it is within 2 % of 64^3 on 3 chairs. Measured on the
# debug chairs (tests/test_volume.py::test_res48_vs_64, runs/m4/test_results.json): it is not.
RES48_ALLOWED = False
OPT_RES = 48 if RES48_ALLOWED else 64


class Volume:
    def __init__(self, prior, res=64, beta=1.0, chunk=32768, bound=BOUND, centered=True):
        self.prior = prior
        self.res = int(res)
        self.beta = float(beta)
        self.chunk = int(chunk)
        self.box = (2.0 * bound) ** 3
        self.grid = make_grid(self.res, bound * (1 - 1 / self.res) if centered else bound).to(prior.device)
        self.n = self.grid.shape[0]
        self.trace = None  # set to a dict to record peak MPS allocations inside a gradient step

    # ------------------------------------------------------------------ helpers
    def _prep(self, z):
        z = z.to(self.prior.device)
        return z[None] if z.dim() == 2 else z

    def _chunks(self, dtype):
        for i in range(0, self.n, self.chunk):
            yield self.grid[i:i + self.chunk].to(dtype)

    def _contrib(self, h, q):
        """|B|/|G| * sum over the chunk of sigmoid(beta * logit): (B,)."""
        return torch.sigmoid(self.beta * self.prior.head(h, q)).sum(1) * (self.box / self.n)

    def _note_peak(self):
        if self.trace is not None and self.prior.device.type == "mps":
            cur = torch.mps.current_allocated_memory()
            self.trace["peak_current_allocated"] = max(self.trace.get("peak_current_allocated", 0), cur)

    def _dV_dh(self, h):
        """Steps 2-3: V (B,) and dV/dh (B, 512, 512) from a trunk output h, one query chunk at a time."""
        h_leaf = h.detach().requires_grad_()
        V = torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
        with torch.enable_grad():
            for q in self._chunks(h.dtype):
                c = self._contrib(h_leaf, q)
                self._note_peak()
                c.sum().backward()  # V_b depends on h_b only, so the sum gives each dV_b/dh_b
                V += c.detach()
        return V, h_leaf.grad

    # ------------------------------------------------------------------ API
    @torch.no_grad()
    def value(self, z):
        """V (B,) without any graph."""
        z = self._prep(z)
        h = self.prior.trunk(z)
        V = torch.zeros(z.shape[0], dtype=h.dtype, device=h.device)
        for q in self._chunks(h.dtype):
            V += self._contrib(h, q)
        return V

    def backward(self, z, fn=None):
        """Chunked two-stage backward of F = sum(fn(V(z))) (fn = identity if None). z may be a leaf that
        requires grad or any tensor in a graph (e.g. D(x; sigma)); gradients accumulate in the .grad of the
        graph's leaves, exactly as F.backward() would. Returns (V, fn(V)) detached."""
        z = self._prep(z)
        with torch.enable_grad():
            h = self.prior.trunk(z)                       # 1. trunk once, with graph
            V, g = self._dV_dh(h)                         # 2-3. chunked, dV/dh in h_leaf.grad
            if fn is None:
                F = V.clone()
                w = torch.ones_like(V)
            else:
                v = V.clone().requires_grad_()
                F = fn(v)
                w, = torch.autograd.grad(F.sum(), v)     # dF/dV_b, exact since V_b depends on z_b only
                F = F.detach()
            self._note_peak()
            h.backward(g * w.view(-1, 1, 1))              # 4. back to z
            self._note_peak()
        return V, F

    def value_and_grad(self, z, fn=None):
        """(V, grad) with grad = d sum(V)/dz, or (V, grad, fn(V)) with grad = d sum(fn(V))/dz if fn is given.
        z is detached first: the gradient is w.r.t. this z only."""
        z_ = self._prep(z).detach().requires_grad_()
        V, F = self.backward(z_, fn)
        return (V, z_.grad) if fn is None else (V, z_.grad, F)

    def __call__(self, z):
        """Differentiable V(z) (B,): plain autograd tensor, e.g. ((est(z) - c) ** 2).sum().backward().
        Falls back to value() when no gradient is needed."""
        z = self._prep(z)
        if not (torch.is_grad_enabled() and z.requires_grad):
            return self.value(z)
        return _ChunkedVolume.apply(self.prior.trunk(z), self)

    def reference(self, z):
        """Unchunked plain-autograd V(z), for tests on small grids only (keeps the whole graph)."""
        z = self._prep(z)
        h = self.prior.trunk(z)
        return torch.sigmoid(self.beta * self.prior.head(h, self.grid.to(h.dtype))).mean(1) * self.box


class _ChunkedVolume(torch.autograd.Function):
    """forward: steps 2-3 (V and dV/dh); backward: grad_V * dV/dh, i.e. step 4 is done by autograd."""

    @staticmethod
    def forward(ctx, h, est):
        V, g = est._dV_dh(h)
        ctx.save_for_backward(g)
        return V

    @staticmethod
    def backward(ctx, grad_V):
        g, = ctx.saved_tensors
        return grad_V.view(-1, 1, 1) * g, None


def memory_stats():
    """Memory snapshot in GB: MPS current/driver allocations, process peak RSS (ru_maxrss, bytes on macOS)."""
    out = {"rss_peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30}
    if torch.backends.mps.is_available():
        out["mps_current_gb"] = torch.mps.current_allocated_memory() / 2 ** 30
        out["mps_driver_gb"] = torch.mps.driver_allocated_memory() / 2 ** 30
    return out
