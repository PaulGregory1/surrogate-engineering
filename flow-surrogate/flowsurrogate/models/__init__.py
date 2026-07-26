"""Surrogate model zoo."""

from __future__ import annotations

import torch.nn as nn

from .fno import FNO2d
from .pod import PODRegressor
from .unet import UNet

#: Torch models, all sharing the signature x -> (fields, cd).
TORCH_MODELS = {
    "unet": lambda **kw: UNet(**kw),
    "fno": lambda **kw: FNO2d(**kw),
}


def build(name: str, **kwargs) -> nn.Module:
    if name not in TORCH_MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(TORCH_MODELS)}")
    return TORCH_MODELS[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["FNO2d", "PODRegressor", "UNet", "TORCH_MODELS", "build", "count_parameters"]
