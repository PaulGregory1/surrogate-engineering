"""Training loop for the graph and grid surrogates."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import Graph, GraphDataset, collate
from .models import NEEDS_GRID, build, count_parameters


@dataclass
class TrainConfig:
    model: str = "gnn"
    epochs: int = 60
    batch_size: int = 8  # graphs per step
    lr: float = 1e-3
    weight_decay: float = 1e-6
    hidden: int = 64
    n_message_passing: int = 10
    use_global: bool = False
    use_coarse_solution: bool = False
    use_position: bool = True
    lambda_max: float = 0.0  # extra weight on the per-part peak stress
    n_train: int | None = None
    seed: int = 0
    name: str | None = None

    @property
    def tag(self) -> str:
        if self.name:
            return self.name
        bits = [self.model]
        if self.model == "gnn":
            bits.append(f"mp{self.n_message_passing}")
            if self.use_global:
                bits.append("glob")
        if self.use_coarse_solution:
            bits.append("corr")
        if not self.use_position:
            bits.append("nopos")
        if self.lambda_max:
            bits.append("maxloss")
        if self.n_train:
            bits.append(f"n{self.n_train}")
        return "-".join(bits)

    def model_kwargs(self, n_features: int) -> dict:
        if self.model == "gnn":
            return {
                "n_node_features": n_features,
                "hidden": self.hidden,
                "n_message_passing": self.n_message_passing,
                "use_global": self.use_global,
            }
        return {"in_ch": 4, "base": 32, "depth": 3}


def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)


def per_graph_max(values: torch.Tensor, node_batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Differentiable segment max.  Batches are small, so an explicit loop is fine."""
    return torch.stack([values[node_batch == i].max() for i in range(n_graphs)])


def loss_fn(pred: torch.Tensor, batch: Graph, lambda_max: float) -> tuple[torch.Tensor, dict]:
    field = F.mse_loss(pred, batch.y)
    logs = {"field": float(field)}
    total = field
    if lambda_max > 0:
        pm = per_graph_max(pred, batch.node_batch, batch.n_graphs)
        tm = per_graph_max(batch.y, batch.node_batch, batch.n_graphs)
        peak = F.mse_loss(pm, tm)
        total = total + lambda_max * peak
        logs["peak"] = float(peak)
    logs["total"] = float(total)
    return total, logs


def iterate(dataset: GraphDataset, batch_size: int, shuffle: bool, rng: np.random.Generator | None = None):
    idx = np.arange(len(dataset))
    if shuffle and rng is not None:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield collate([dataset[j] for j in idx[i : i + batch_size]])


@torch.no_grad()
def evaluate_loss(model, dataset, cfg, device) -> dict[str, float]:
    model.eval()
    acc, n = {}, 0
    for batch in iterate(dataset, cfg.batch_size, shuffle=False):
        batch = batch.to(device)
        _, logs = loss_fn(model(batch), batch, cfg.lambda_max)
        for k, v in logs.items():
            acc[k] = acc.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in acc.items()}


def train(
    cfg: TrainConfig,
    train_set: GraphDataset,
    val_set: GraphDataset,
    out_dir: Path,
    device: str = "auto",
    verbose: bool = True,
) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = pick_device(device)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.n_train is not None and cfg.n_train < len(train_set):
        keep = rng.permutation(len(train_set))[: cfg.n_train]
        subset = [train_set[i] for i in keep]

        class _Sub:
            def __len__(self):
                return len(subset)

            def __getitem__(self, i):
                return subset[i]

        train_set_used = _Sub()
    else:
        train_set_used = train_set

    model = build(cfg.model, **cfg.model_kwargs(train_set.n_features)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_steps = cfg.epochs * max(1, int(np.ceil(len(train_set_used) / cfg.batch_size)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=n_steps, pct_start=0.15)

    history, best = [], {"val": float("inf"), "epoch": -1}
    ckpt = out_dir / f"{cfg.tag}.pt"
    t0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        run, nb = {}, 0
        for batch in iterate(train_set_used, cfg.batch_size, shuffle=True, rng=rng):
            batch = batch.to(dev)
            opt.zero_grad(set_to_none=True)
            loss, logs = loss_fn(model(batch), batch, cfg.lambda_max)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in logs.items():
                run[k] = run.get(k, 0.0) + v
            nb += 1
        row = {"epoch": epoch, **{f"train_{k}": v / max(nb, 1) for k, v in run.items()}}
        row.update({f"val_{k}": v for k, v in evaluate_loss(model, val_set, cfg, dev).items()})
        history.append(row)
        if row["val_total"] < best["val"]:
            best = {"val": row["val_total"], "epoch": epoch}
            torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg), "n_features": train_set.n_features}, ckpt)
        if verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
            print(f"  [{cfg.tag}] epoch {epoch:3d}  train {row['train_total']:.4f}  val {row['val_total']:.4f}", flush=True)

    wall = time.perf_counter() - t0
    summary = {
        "cfg": asdict(cfg),
        "tag": cfg.tag,
        "n_parameters": count_parameters(model),
        "n_train": len(train_set_used),
        "best_epoch": best["epoch"],
        "best_val_total": best["val"],
        "train_seconds": wall,
        "device": str(dev),
        "history": history,
    }
    (out_dir / f"{cfg.tag}_history.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"  [{cfg.tag}] done in {wall / 60:.1f} min (best epoch {best['epoch']})", flush=True)
    return summary


def load_model(ckpt: Path, device: str = "auto") -> tuple[torch.nn.Module, TrainConfig]:
    dev = pick_device(device)
    blob = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = TrainConfig(**blob["cfg"])
    model = build(cfg.model, **cfg.model_kwargs(blob["n_features"])).to(dev)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, cfg


def needs_grid(cfg: TrainConfig) -> bool:
    return cfg.model in NEEDS_GRID
