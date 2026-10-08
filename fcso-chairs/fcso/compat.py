"""Mac compatibility shims for third_party/3DShape2VecSet, installed before its modules are imported.

The repo imports `torch_cluster.fps` (CUDA/C++ extension). We register a pure-PyTorch module under
that name instead of editing the third-party code.
"""
import os
import sys
import types

import torch

VECSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "3DShape2VecSet")


def fps(pos, batch, ratio=0.5, random_start=False):
    """Farthest point sampling with the torch_cluster.fps signature: returns flat indices into `pos`.

    Assumes equal-sized, contiguous batches (the only way models_ae.py calls it). Starts at the first
    point of each cloud unless random_start, so encodings are deterministic.
    """
    B = int(batch.max().item()) + 1
    N = pos.shape[0] // B
    k = int(round(ratio * N))
    p = pos.view(B, N, -1)
    idx = torch.zeros(B, k, dtype=torch.long, device=pos.device)
    dist = torch.full((B, N), float("inf"), device=pos.device)
    cur = torch.randint(N, (B,), device=pos.device) if random_start else torch.zeros(B, dtype=torch.long, device=pos.device)
    ar = torch.arange(B, device=pos.device)
    for i in range(k):
        idx[:, i] = cur
        d = ((p - p[ar, cur].unsqueeze(1)) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        cur = dist.argmax(-1)
    return (idx + ar.unsqueeze(1) * N).view(-1)


def install():
    if "torch_cluster" not in sys.modules:
        mod = types.ModuleType("torch_cluster")
        mod.fps = fps
        sys.modules["torch_cluster"] = mod
    if VECSET_DIR not in sys.path:
        sys.path.insert(0, VECSET_DIR)
