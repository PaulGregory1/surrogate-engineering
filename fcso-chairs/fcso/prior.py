"""3DShape2VecSet models for FCSO: KL autoencoder (shape decoder) + class-conditional EDM prior.

Conventions (checked against third_party/3DShape2VecSet, see DECISIONS.md):
- Input surfaces: 2048 points drawn from the 100k-point surface cloud, multiplied by the per-shape
  `scale` (util/shapenet.py). Shapes then span [-1, 1]^3 along their largest axis, y is up.
- Latent z: (B, 512, 8). `encode` returns the KL posterior **mean**. The diffusion prior was trained on
  `ae.encode(surface)[1]`, i.e. a posterior *sample*, with no scaling factor (engine_class_cond.py), and
  EDMPrecond uses sigma_data = 1.
- Decoder output: occupancy logits, positive = inside (eval.py thresholds logits >= 0 against labels).
- Grid: the repo evaluates on linspace(-1, 1) along each axis (eval.py, sample_class_cond.py).
- Chair class label: 18 (category_ids['03001627'] in util/shapenet.py).
"""
import os

import numpy as np
import torch

from fcso import compat

compat.install()
import models_ae  # noqa: E402
import models_class_cond  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = "03001627"
CHAIR_CLASS = 18
BOUND = 1.0  # grid spans [-BOUND, BOUND]^3
POINT_DIR = os.path.join(ROOT, "data", "ShapeNetV2_point", CAT)
SURF_DIR = os.path.join(ROOT, "data", "ShapeNetV2_watertight", CAT, "4_pointcloud")
AE_PTH = os.path.join(ROOT, "weights/ae/kl_d512_m512_l8/checkpoint-199-model.pth")
DM_PTH = os.path.join(ROOT, "weights/dm/kl_d512_m512_l8_d24_edm/checkpoint-499-model.pth")


def default_device():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def read_split(split):
    """Chair ids of a repo split ('train', 'val', 'test'), without the .npz suffix of the .lst files."""
    return [i.replace(".npz", "") for i in read_ids(os.path.join(POINT_DIR, split + ".lst"))]


def read_ids(path):
    with open(path) as f:
        return [l.split("#")[0].strip() for l in f if l.split("#")[0].strip()]


def load_surface(shape_id):
    """Full surface cloud (100k x 3, float32) in the normalized frame used for training (points * scale)."""
    scale = np.load(os.path.join(POINT_DIR, shape_id + ".npy")).item()
    with np.load(os.path.join(SURF_DIR, shape_id + ".npz")) as d:
        return d["points"].astype(np.float32) * np.float32(scale)


def sample_surface(surface, n=2048, seed=0):
    """n points drawn without replacement, as util/shapenet.py does (surface_sampling=True), but seeded."""
    ind = np.random.default_rng(seed).choice(surface.shape[0], n, replace=False)
    return surface[ind]


def make_grid(res, bound=BOUND):
    """(res^3, 3) query points, linspace(-bound, bound, res) per axis, 'ij' order: index = (i*res + j)*res + k
    for the point (x_i, y_j, z_k). Reshape decoded values to (res, res, res) to get a [x, y, z] volume."""
    t = torch.linspace(-bound, bound, res)
    return torch.stack(torch.meshgrid(t, t, t, indexing="ij"), dim=-1).reshape(-1, 3)


def _load(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class Prior:
    """AE + EDM in eval mode, parameters frozen (requires_grad=False): gradients only ever reach latents."""

    def __init__(self, device=None, load_dm=True):
        self.device = torch.device(device) if device is not None else default_device()
        self.ae = _load(models_ae.kl_d512_m512_l8(), AE_PTH).to(self.device)
        self.dm = _load(models_class_cond.kl_d512_m512_l8_d24_edm(), DM_PTH).to(self.device) if load_dm else None

    # ---------------------------------------------------------------- encoder
    @torch.no_grad()
    def encode_posterior(self, pc):
        """pc: (B, 2048, 3) -> (mean, logvar), each (B, 512, 8). Mirrors KLAutoEncoder.encode up to the sample."""
        ae = self.ae
        pc = torch.as_tensor(pc, dtype=torch.float32, device=self.device)
        if pc.dim() == 2:
            pc = pc[None]
        B, N, D = pc.shape
        flat = pc.reshape(B * N, D)
        batch = torch.repeat_interleave(torch.arange(B, device=self.device), N)
        idx = models_ae.fps(flat, batch, ratio=ae.num_latents / ae.num_inputs)
        sampled = flat[idx].view(B, -1, 3)
        q = ae.point_embed(sampled)
        ctx = ae.point_embed(pc)
        cross_attn, cross_ff = ae.cross_attend_blocks
        x = cross_attn(q, context=ctx, mask=None) + q
        x = cross_ff(x) + x
        mean = ae.mean_fc(x)
        logvar = ae.logvar_fc(x).clamp(-30.0, 20.0)  # same clamp as DiagonalGaussianDistribution
        return mean, logvar

    def encode(self, pc):
        """Posterior mean z_s (B, 512, 8): the initial latent for optimization (SPEC: mean, not a sample)."""
        return self.encode_posterior(pc)[0]

    def encode_id(self, shape_id, seed=0):
        return self.encode(sample_surface(load_surface(shape_id), seed=seed))

    # ---------------------------------------------------------------- decoder
    def trunk(self, z):
        """Latent-only part of the decoder: proj + 24 self-attention blocks. (B, 512, 8) -> (B, 512, 512).
        Differentiable; compute once per z, then decode any number of query chunks from it."""
        ae = self.ae
        h = ae.proj(z)
        for self_attn, self_ff in ae.layers:
            h = self_attn(h) + h
            h = self_ff(h) + h
        return h

    def head(self, h, queries):
        """Cross-attention from queries to trunk features. h: (B, 512, 512), queries: (B, Q, 3) or (Q, 3)
        -> occupancy logits (B, Q). Differentiable."""
        ae = self.ae
        if queries.dim() == 2:
            queries = queries[None].expand(h.shape[0], -1, -1)
        out = ae.decoder_cross_attn(ae.point_embed(queries), context=h)
        if ae.decoder_ff is not None:
            out = out + ae.decoder_ff(out)
        return ae.to_outputs(out).squeeze(-1)

    @torch.no_grad()
    def decode(self, z, queries, chunk=65536):
        """No-grad logits of z (B, 512, 8) at queries (Q, 3), chunked. Returns (B, Q) on the CPU."""
        z = z.to(self.device)
        queries = torch.as_tensor(queries, dtype=torch.float32)
        h = self.trunk(z)
        out = [self.head(h, queries[i:i + chunk].to(self.device)).float().cpu()
               for i in range(0, queries.shape[0], chunk)]
        return torch.cat(out, dim=1)

    def decode_grid(self, z, res=128, bound=BOUND, chunk=65536):
        """Logit volume(s) on a res^3 grid: (B, res, res, res) numpy array indexed [b, x, y, z]."""
        return self.decode(z, make_grid(res, bound), chunk).view(-1, res, res, res).numpy()

    # ---------------------------------------------------------------- prior
    def denoise(self, x, sigma, class_label=CHAIR_CLASS):
        """EDM denoiser D(x; sigma, class): x (B, 512, 8), sigma scalar or (B,). Grad flows to x only if the
        caller enables it (parameters are frozen)."""
        x = x.to(self.device, torch.float32)
        B = x.shape[0]
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=self.device).reshape(-1).expand(B)
        labels = torch.full((B,), class_label, dtype=torch.long, device=self.device)
        return self.dm(x, sigma, labels)

    @property
    def sigma_data(self):
        return float(self.dm.sigma_data)
