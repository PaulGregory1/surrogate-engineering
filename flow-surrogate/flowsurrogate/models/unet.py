"""U-Net surrogate: geometry image in, flow field out.

The obstacle enters only through the input channels (level set + mask), so unlike a
parametric ROM the network never sees the Fourier coefficients and is in principle
usable on any geometry that can be rasterised.
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


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 32, depth: int = 3):
        super().__init__()
        chans = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_block(in_ch if i == 0 else chans[i - 1], chans[i]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _block(chans[depth - 1], chans[depth])
        self.up = nn.ModuleList([nn.Conv2d(chans[i + 1], chans[i], 1) for i in range(depth)])
        self.dec = nn.ModuleList([_block(2 * chans[i], chans[i]) for i in range(depth)])
        self.head = nn.Conv2d(base, out_ch, 1)
        self.scalar_head = nn.Sequential(nn.Linear(chans[depth], 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skips = []
        h = x
        for enc in self.enc:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        cd = self.scalar_head(h.mean(dim=(2, 3))).squeeze(-1)
        for i in reversed(range(len(self.dec))):
            h = F.interpolate(h, size=skips[i].shape[-2:], mode="bilinear", align_corners=False)
            h = self.up[i](h)
            h = self.dec[i](torch.cat([h, skips[i]], dim=1))
        return self.head(h), cd
