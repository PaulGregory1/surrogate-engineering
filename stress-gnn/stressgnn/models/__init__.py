"""Surrogate model zoo.  Every model maps a `Graph` to one value per node."""

from __future__ import annotations

import torch.nn as nn

from .cnn_grid import GridUNet
from .gnn import MeshGraphNet

BUILDERS = {
    "gnn": MeshGraphNet,
    "grid-cnn": GridUNet,
}


def build(name: str, **kwargs) -> nn.Module:
    if name not in BUILDERS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(BUILDERS)}")
    return BUILDERS[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#: Models that need the rasterised image attached to each graph.
NEEDS_GRID = {"grid-cnn"}

__all__ = ["GridUNet", "MeshGraphNet", "BUILDERS", "NEEDS_GRID", "build", "count_parameters"]
