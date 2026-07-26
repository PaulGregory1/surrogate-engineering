"""Turn a trained surrogate into the numbers that go in RESULTS.md."""

from __future__ import annotations

import time

import numpy as np
import torch

from .data import FlowDataset, Normalizer
from .metrics import summarise
from .models import PODRegressor
from .train import L_REF


def _to_physical(pred_norm: torch.Tensor, mask: torch.Tensor, norm: Normalizer) -> torch.Tensor:
    return norm.decode_fields(pred_norm) * mask


def metrics_from_normalized(
    pred_fields_norm: torch.Tensor,
    pred_cd_norm: np.ndarray,
    dataset: FlowDataset,
    norm: Normalizer,
    u_in: float,
) -> dict[str, float]:
    """Metrics in physical units, given predictions in normalised units."""
    mask = dataset.m
    pred_phys = _to_physical(pred_fields_norm.cpu(), mask, norm)
    true_phys = torch.from_numpy(dataset.fields_raw) * mask
    pred_cd = norm.decode_cd(torch.as_tensor(pred_cd_norm)).numpy()
    return summarise(pred_phys, true_phys, mask, pred_cd, dataset.cd_raw, u_in, L_REF)


@torch.no_grad()
def predict_torch(model: torch.nn.Module, dataset: FlowDataset, device, batch: int = 32):
    model.eval()
    fields, cds = [], []
    for i in range(0, len(dataset), batch):
        x = dataset.x[i : i + batch].to(device)
        m = dataset.m[i : i + batch].to(device)
        f, cd = model(x)
        fields.append((f * m).cpu())
        cds.append(cd.cpu())
    return torch.cat(fields), torch.cat(cds).numpy()


def evaluate_torch(model, dataset: FlowDataset, norm: Normalizer, device, u_in: float) -> dict[str, float]:
    fields, cd = predict_torch(model, dataset, device)
    return metrics_from_normalized(fields, cd, dataset, norm, u_in)


def evaluate_pod(pod: PODRegressor, dataset: FlowDataset, norm: Normalizer, u_in: float) -> dict[str, float]:
    shape = dataset.y.shape[1:]
    fields, cd = pod.predict(dataset.params, dataset.re, tuple(shape))
    fields = torch.from_numpy(fields.astype(np.float32)) * dataset.m
    return metrics_from_normalized(fields, cd, dataset, norm, u_in)


@torch.no_grad()
def latency_ms(model: torch.nn.Module, dataset: FlowDataset, device, batch: int = 1, repeats: int = 30) -> float:
    """Wall-clock inference time per sample."""
    model.eval()
    x = dataset.x[:batch].to(device)
    for _ in range(5):
        model(x)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(x)
    if device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / repeats / batch * 1e3
