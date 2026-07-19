"""Metrics.

The comparison that matters is not "how small is the MSE" but:

    a coarse FEM solve already gives you *an* answer, cheaply, and it is wrong by a
    known amount.  Does the surrogate beat it, on the same mesh, and how close does
    it get to the fine reference?

So every metric is reported for three things at once: the coarse FEM solution as-is
(the baseline), the surrogate, and the interpolation floor -- the error a perfect
model would still have, because a 1000-node mesh cannot represent the peak of a
25000-node stress field.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .dataset import GraphDataset, collate


@torch.no_grad()
def predict(model, dataset: GraphDataset, device, batch_size: int = 8) -> list[np.ndarray]:
    """Per-graph predictions in physical units."""
    model.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(dataset), batch_size):
        graphs = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        batch = collate(graphs).to(device)
        pred = dataset.denormalise(model(batch).cpu())
        idx = 0
        for g in graphs:
            n = g.x.shape[0]
            out.append(pred[idx : idx + n].numpy())
            idx += n
    return out


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))


def field_metrics(preds: list[np.ndarray], dataset: GraphDataset) -> dict[str, float]:
    rel, mae, peak_err, peak_signed = [], [], [], []
    for p, sample in zip(preds, dataset.samples):
        target = sample["vm_target"]
        rel.append(_rel_l2(p, target))
        mae.append(float(np.abs(p - target).mean()))
        fine = float(sample["max_vm_fine"])
        peak_err.append(abs(float(p.max()) - fine) / fine)
        peak_signed.append((float(p.max()) - fine) / fine)
    return {
        "rel_l2": float(np.mean(rel)),
        "rel_l2_p95": float(np.percentile(rel, 95)),
        "mae": float(np.mean(mae)),
        "peak_err_pct": 100 * float(np.mean(peak_err)),
        "peak_err_p95_pct": 100 * float(np.percentile(peak_err, 95)),
        "peak_bias_pct": 100 * float(np.mean(peak_signed)),
    }


def baseline_metrics(dataset: GraphDataset) -> dict[str, dict[str, float]]:
    """Coarse FEM as-is, and the interpolation floor."""
    coarse = [s["vm_coarse"] for s in dataset.samples]
    floor = [s["vm_target"] for s in dataset.samples]
    return {
        "coarse FEM (no ML)": field_metrics(coarse, dataset),
        "interpolation floor": field_metrics(floor, dataset),
    }


@torch.no_grad()
def latency_ms(model, dataset: GraphDataset, device, repeats: int = 20) -> float:
    batch = collate([dataset[0]]).to(device)
    for _ in range(5):
        model(batch)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(batch)
    if device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / repeats * 1e3
