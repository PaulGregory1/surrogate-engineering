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
def predict(model, dataset: GraphDataset, device, batch_size: int = 8) -> tuple[list[np.ndarray], np.ndarray]:
    """Per-graph nodal fields and peak-stress predictions, in physical units."""
    model.eval()
    fields: list[np.ndarray] = []
    peaks: list[np.ndarray] = []
    for i in range(0, len(dataset), batch_size):
        graphs = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        batch = collate(graphs).to(device)
        node_out, peak_out = model(batch)
        pred = dataset.denormalise(node_out.cpu())
        peaks.append(dataset.denormalise_peak(peak_out.cpu()).numpy())
        idx = 0
        for g in graphs:
            n = g.x.shape[0]
            fields.append(pred[idx : idx + n].numpy())
            idx += n
    return fields, np.concatenate(peaks)


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))


def field_metrics(
    preds: list[np.ndarray], dataset: GraphDataset, peaks: np.ndarray | None = None
) -> dict[str, float]:
    """Nodal-field errors, plus peak-stress error from the scalar head if given.

    `peak_from_field_*` is always reported as well: it is what you get by taking the
    maximum of the predicted nodal field, and it is bounded below by the interpolation
    floor no matter how good the model is.  The scalar head is not.
    """
    fine = np.array([float(s["max_vm_fine"]) for s in dataset.samples])
    rel, mae, from_field = [], [], []
    for p, sample in zip(preds, dataset.samples):
        target = sample["vm_target"]
        rel.append(_rel_l2(p, target))
        mae.append(float(np.abs(p - target).mean()))
        from_field.append(float(p.max()))
    from_field = np.array(from_field)

    out = {
        "rel_l2": float(np.mean(rel)),
        "rel_l2_p95": float(np.percentile(rel, 95)),
        "mae": float(np.mean(mae)),
        "peak_from_field_err_pct": 100 * float(np.mean(np.abs(from_field - fine) / fine)),
        "peak_from_field_bias_pct": 100 * float(np.mean((from_field - fine) / fine)),
    }
    head = from_field if peaks is None else peaks
    err = np.abs(head - fine) / fine
    out.update(
        {
            "peak_err_pct": 100 * float(np.mean(err)),
            "peak_err_p95_pct": 100 * float(np.percentile(err, 95)),
            "peak_bias_pct": 100 * float(np.mean((head - fine) / fine)),
        }
    )
    return out


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
