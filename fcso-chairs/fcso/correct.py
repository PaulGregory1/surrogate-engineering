"""Phase 2 of FCSO on an EDM prior: noising + probability-flow ODE, plus a float32 EDM sampler.

SPEC "Adapting Phase 2 to an EDM diffusion prior": a rectified-flow state at t = 1 - tau is equivalent to
an EDM state x = z_hat + sigma * eps with sigma_start = tau / (1 - tau) (0.25 for tau = 0.2). The
correction noises z_hat to sigma_start with fresh noise, then integrates the PF-ODE dx/dsigma = d with
d = (x - D(x; sigma, chair)) / sigma from sigma_start down to 0 in N steps (Euler, Heun as a variant),
on the repo's EDM rho = 7 schedule truncated to [0, sigma_start].

The repo's `models_class_cond.edm_sampler` casts to float64, which MPS does not support; `edm_sample`
below is the same algorithm (18 Heun steps, S_churn = 0, t_N = 0) in float32.

Guidance (milestone 6, Option B of the paper, Eq. 8, adapted in SPEC): `correct(..., guidance=g)` adds
g(x, sigma, denoised) to d at every step, i.e. d_guided = d + g. `VolumeGuidance(prior, est, c, lam, mode)`
is that g for the volume objective: g = lam * grad_x (V(Dec(.)) - c)^2, with c = V(z_hat) at the end of
Phase 1, evaluated
  - mode 'noisy' (default, Eq. 8): on the noisy state x itself (the denoiser stays under no_grad);
  - mode 'x0': on the denoised prediction D(x; sigma), differentiating through the denoiser (one extra
    denoiser forward + backward per evaluation; gradients only reach x, model parameters are frozen).
The sign is "+" because sigma decreases along the integration, so (sigma_next - sigma) * g moves x down the
gradient of (V - c)^2 (tests/test_guidance.py checks it). V is the chunked estimator of fcso/volume.py.
"""
import math

import torch

from fcso.prior import CHAIR_CLASS

SIGMA_MIN = 0.002  # repo edm_sampler defaults
SIGMA_MAX = 80.0
RHO = 7.0


def tau_to_sigma(tau):
    """Rectified-flow noise ratio tau -> equivalent EDM noise level (SPEC): sigma = tau / (1 - tau)."""
    return tau / (1.0 - tau)


def edm_sigmas(num_steps, sigma_max=SIGMA_MAX, sigma_min=SIGMA_MIN, rho=RHO):
    """EDM time steps (Karras et al. Eq. 5, as in the repo's edm_sampler): num_steps decreasing values from
    sigma_max to sigma_min on the rho schedule, followed by t_N = 0. float32 tensor of length num_steps + 1
    on the CPU."""
    i = torch.arange(num_steps, dtype=torch.float64)
    a, b = sigma_max ** (1.0 / rho), sigma_min ** (1.0 / rho)
    t = (a + i / max(num_steps - 1, 1) * (b - a)) ** rho
    if num_steps == 1:
        t = torch.tensor([sigma_max], dtype=torch.float64)
    return torch.cat([t, torch.zeros(1, dtype=torch.float64)]).float()


def correction_sigmas(sigma_start, n_steps=10, sigma_min=SIGMA_MIN, rho=RHO):
    """The rho = 7 schedule truncated to [0, sigma_start]: sigma_max := sigma_start, same sigma_min and
    rho, n_steps values then 0, so n_steps ODE steps in total (the last one, sigma_min -> 0, returns
    D(x; sigma_min) for an Euler step)."""
    return edm_sigmas(n_steps, sigma_max=sigma_start, sigma_min=sigma_min, rho=rho)


def _randn(shape, generator, device):
    """Standard normal noise drawn on the CPU (reproducible across devices), then moved to device."""
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


def _generator(seed):
    if isinstance(seed, torch.Generator):
        return seed
    g = torch.Generator()
    g.manual_seed(int(seed) if seed is not None else torch.seed())
    return g


@torch.no_grad()
def _denoise(prior, x, sigma, class_label):
    return prior.denoise(x, sigma, class_label)


def ode_step(prior, x, sigma, sigma_next, solver="euler", class_label=CHAIR_CLASS, guidance=None):
    """One PF-ODE step from sigma to sigma_next (< sigma). Returns (x_next, info) where info holds the norms
    of d and of the guidance term at this step (for lambda calibration in milestone 6).

    Euler: x_next = x + (sigma_next - sigma) * d, d = (x - D(x; sigma)) / sigma [+ guidance(x, sigma, D)].
    Heun: second evaluation at sigma_next, averaged, skipped when sigma_next = 0 (as in edm_sampler)."""
    sigma, sigma_next = float(sigma), float(sigma_next)
    den = _denoise(prior, x, sigma, class_label)
    d = (x - den) / sigma
    info = {"sigma": sigma, "d_norm": float(d.flatten(1).norm(dim=1).mean())}
    if guidance is not None:
        g = guidance(x, sigma, den)
        info["g_norm"] = float(g.flatten(1).norm(dim=1).mean())
        info.update(getattr(guidance, "last", None) or {})
        d = d + g
    x_next = x + (sigma_next - sigma) * d
    if solver == "heun" and sigma_next > 0:
        den2 = _denoise(prior, x_next, sigma_next, class_label)
        d2 = (x_next - den2) / sigma_next
        if guidance is not None:
            d2 = d2 + guidance(x_next, sigma_next, den2)
        x_next = x + (sigma_next - sigma) * (0.5 * d + 0.5 * d2)
    elif solver not in ("euler", "heun"):
        raise ValueError(f"unknown solver {solver!r}")
    return x_next, info


def correct(prior, z_hat, sigma_start=0.25, n_steps=10, seed=None, solver="euler", class_label=CHAIR_CLASS,
            guidance=None, sigmas=None, return_info=False):
    """Phase 2 correction of latents z_hat (B, 512, 8): x = z_hat + sigma_start * eps (fresh eps from `seed`,
    an int or a torch.Generator; None = random), then N ODE steps down to sigma = 0 on the truncated rho = 7
    schedule (or on `sigmas`, a decreasing sequence ending at 0). Returns the corrected latents (and the
    per-step info list if return_info). Parameters stay frozen; no autograd graph is kept."""
    dev = prior.device
    z_hat = z_hat.to(dev, torch.float32)
    sig = correction_sigmas(sigma_start, n_steps) if sigmas is None else torch.as_tensor(sigmas, dtype=torch.float32)
    eps = _randn(z_hat.shape, _generator(seed), dev)
    x = z_hat + float(sig[0]) * eps
    infos = []
    for s, s_next in zip(sig[:-1].tolist(), sig[1:].tolist()):
        x, info = ode_step(prior, x, s, s_next, solver=solver, class_label=class_label, guidance=guidance)
        x = x.detach()
        infos.append(info)
    return (x, infos) if return_info else x


class VolumeGuidance:
    """Option B guidance term for the volume objective (SPEC "Guidance, Option B"):

        g(x, sigma) = lam * grad_x (V(Dec(y)) - c)^2,   y = x (mode 'noisy') or y = D(x; sigma) (mode 'x0')

    `est` is a fcso.volume.Volume (chunked two-stage backward); c is a float or a (B,) tensor. Model
    parameters never receive gradients (Prior freezes them; the only leaf is a detached copy of x). After
    each call `self.last` holds V(y) and (V(y) - c)^2 (batch means) for logging; `self.n_evals` counts the
    volume-gradient evaluations (one per Euler step, two per Heun step)."""

    def __init__(self, prior, est, c, lam, mode="noisy", class_label=CHAIR_CLASS):
        if mode not in ("noisy", "x0"):
            raise ValueError(f"unknown guidance mode {mode!r}")
        self.prior, self.est, self.lam, self.mode, self.class_label = prior, est, float(lam), mode, class_label
        self.c = torch.as_tensor(c, dtype=torch.float32, device=prior.device).reshape(-1)
        self.last = None
        self.n_evals = 0

    def _fn(self, v):
        return (v - self.c) ** 2

    def raw_grad(self, x, sigma):
        """(grad_x (V(y) - c)^2, V(y), (V(y) - c)^2) without the lam factor, x detached."""
        x = x.detach().to(self.prior.device, torch.float32)
        if self.mode == "noisy":
            V, gx, F = self.est.value_and_grad(x, self._fn)
        else:
            x_leaf = x.clone().requires_grad_(True)
            with torch.enable_grad():
                y = self.prior.denoise(x_leaf, sigma, self.class_label)
                V, F = self.est.backward(y, self._fn)
            gx = x_leaf.grad
        self.n_evals += 1
        return gx.detach(), V.detach(), F.detach()

    def __call__(self, x, sigma, denoised=None):
        gx, V, F = self.raw_grad(x, sigma)
        self.last = {"V": float(V.mean()), "sq": float(F.mean())}
        return self.lam * gx


def calibrate_lambda(prior, est, z_hat, c, sigma_start=0.25, seed=0, mode="noisy", class_label=CHAIR_CLASS,
                     target_ratio=1.0):
    """Ratio |d| / |grad_x (V(Dec(.)) - c)^2| at the first ODE step from x = z_hat + sigma_start * eps
    (same noise as `correct(..., seed=seed)`): the lam that makes the guidance term target_ratio * |d|.
    Returns dict(lam, d_norm, grad_norm, V_x, sq)."""
    dev = prior.device
    z_hat = z_hat.to(dev, torch.float32)
    eps = _randn(z_hat.shape, _generator(seed), dev)
    x = z_hat + float(sigma_start) * eps
    den = _denoise(prior, x, sigma_start, class_label)
    d_norm = float(((x - den) / sigma_start).flatten(1).norm(dim=1).mean())
    g = VolumeGuidance(prior, est, c, 1.0, mode, class_label)
    gx, V, F = g.raw_grad(x, sigma_start)
    gn = float(gx.flatten(1).norm(dim=1).mean())
    return dict(lam=target_ratio * d_norm / max(gn, 1e-30), d_norm=d_norm, grad_norm=gn, V_x=float(V.mean()),
                sq=float(F.mean()))


@torch.no_grad()
def edm_sample(prior, n, seed=0, num_steps=18, class_label=CHAIR_CLASS, shape=(512, 8)):
    """Unconditional-from-noise class-conditional sampling: float32 port of the repo's edm_sampler
    (sigma_max 80, sigma_min 0.002, rho 7, Heun, S_churn 0, round_sigma = identity, t_N = 0).
    Returns latents (n, 512, 8) on prior.device."""
    t = edm_sigmas(num_steps).tolist()
    x = _randn((n, *shape), _generator(seed), prior.device) * t[0]
    for i, (t_cur, t_next) in enumerate(zip(t[:-1], t[1:])):
        x, _ = ode_step(prior, x, t_cur, t_next, solver="heun" if i < num_steps - 1 else "euler",
                        class_label=class_label)
    return x
