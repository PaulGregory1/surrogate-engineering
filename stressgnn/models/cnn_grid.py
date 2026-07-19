"""Cartesian-grid baseline: rasterise, run a U-Net, sample back at the mesh nodes.

This is the "why not just voxelise it?" answer, and it is included to be argued with
rather than dismissed.  It is given a fair shot -- the grid spacing (dx = 0.0104)
matches the fine reference mesh, and the loss is evaluated at exactly the same mesh
nodes as for the graph network -- so any gap comes from the representation itself:
a Cartesian grid cannot place a sample point *on* a curved hole boundary, which is
precisely where the stress peaks live.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.GELU(),
    )


def bilinear_sample(field: torch.Tensor, uv: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Sample (B, H, W) at (N, 2) normalised coordinates belonging to graphs `batch`."""
    b, h, w = field.shape
    px = (uv[:, 0] * 0.5 + 0.5) * (w - 1)
    py = (uv[:, 1] * 0.5 + 0.5) * (h - 1)
    x0 = px.floor().clamp(0, w - 1)
    y0 = py.floor().clamp(0, h - 1)
    x1 = (x0 + 1).clamp(0, w - 1)
    y1 = (y0 + 1).clamp(0, h - 1)
    wx, wy = px - x0, py - y0
    flat = field.reshape(b, h * w)
    xi0, xi1, yi0, yi1 = x0.long(), x1.long(), y0.long(), y1.long()
    v00 = flat[batch, yi0 * w + xi0]
    v01 = flat[batch, yi0 * w + xi1]
    v10 = flat[batch, yi1 * w + xi0]
    v11 = flat[batch, yi1 * w + xi1]
    return (v00 * (1 - wx) + v01 * wx) * (1 - wy) + (v10 * (1 - wx) + v11 * wx) * wy


class GridUNet(nn.Module):
    def __init__(self, in_ch: int = 4, base: int = 32, depth: int = 3, **_):
        super().__init__()
        chans = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_block(in_ch if i == 0 else chans[i - 1], chans[i]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _block(chans[depth - 1], chans[depth])
        self.up = nn.ModuleList([nn.Conv2d(chans[i + 1], chans[i], 1) for i in range(depth)])
        self.dec = nn.ModuleList([_block(2 * chans[i], chans[i]) for i in range(depth)])
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, graph) -> torch.Tensor:
        h = graph.grid
        skips = []
        for enc in self.enc:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for i in reversed(range(len(self.dec))):
            h = F.interpolate(h, size=skips[i].shape[-2:], mode="bilinear", align_corners=False)
            h = self.up[i](h)
            h = self.dec[i](torch.cat([h, skips[i]], dim=1))
        field = self.head(h)[:, 0]
        return bilinear_sample(field, graph.uv, graph.node_batch)
