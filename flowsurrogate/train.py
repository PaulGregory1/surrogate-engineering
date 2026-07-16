"""Training loop shared by every torch surrogate."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import FlowDataset, Normalizer
from .metrics import divergence, interior_fluid, masked_relative_l2
from .models import build, count_parameters

L_REF: float = 20.0  # reference length for the dimensionless continuity residual


@dataclass
class TrainConfig:
    model: str = "unet"
    epochs: int = 120
    batch_size: int = 16
    lr: float = 2e-3
    weight_decay: float = 1e-5
    lambda_cd: float = 0.2  # weight of the drag head
    lambda_phys: float = 0.0  # weight of the continuity residual
    n_train: int | None = None  # subset size, for the data-efficiency study
    seed: int = 0
    u_in: float = 0.08
    model_kwargs: dict = field(default_factory=dict)
    name: str | None = None  # overrides the auto-generated tag

    @property
    def tag(self) -> str:
        if self.name:
            return self.name
        bits = [self.model]
        if self.lambda_phys > 0:
            bits.append("phys")
        if self.n_train is not None:
            bits.append(f"n{self.n_train}")
        if self.seed:
            bits.append(f"s{self.seed}")
        return "-".join(bits)


def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)


def _losses(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    cfg: TrainConfig,
    norm: Normalizer,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    x, y, m, cd = (t.to(device) for t in batch)
    pred, pred_cd = model(x)
    pred = pred * m

    field_loss = F.mse_loss(pred, y)
    cd_loss = F.mse_loss(pred_cd, cd)
    total = field_loss + cfg.lambda_cd * cd_loss
    logs = {"field": float(field_loss), "cd": float(cd_loss)}

    if cfg.lambda_phys > 0:
        # Continuity is imposed on the de-normalised prediction, so the weight has a
        # physical meaning and does not depend on the normalisation statistics.
        phys = norm.decode_fields(pred) * m
        div = divergence(phys[:, 0:1], phys[:, 1:2]) * interior_fluid(m)
        phys_loss = (div**2).mean() * (L_REF / cfg.u_in) ** 2
        total = total + cfg.lambda_phys * phys_loss
        logs["phys"] = float(phys_loss)

    logs["total"] = float(total)
    return total, logs


@torch.no_grad()
def evaluate_loss(model, loader, cfg, norm, device) -> dict[str, float]:
    model.eval()
    acc: dict[str, float] = {}
    rels = []
    for batch in loader:
        _, logs = _losses(model, batch, cfg, norm, device)
        for k, v in logs.items():
            acc[k] = acc.get(k, 0.0) + v
        x, y, m, _ = (t.to(device) for t in batch)
        pred, _ = model(x)
        rels.append(masked_relative_l2(pred * m, y, m).mean(dim=1).cpu())
    n = len(loader)
    out = {k: v / n for k, v in acc.items()}
    out["rel_l2"] = float(torch.cat(rels).mean())
    return out


def train(
    cfg: TrainConfig,
    train_set: FlowDataset,
    val_set: FlowDataset,
    out_dir: Path,
    device: str = "auto",
    verbose: bool = True,
    init_state_dict: dict | None = None,
) -> dict:
    """Fit one surrogate and write `<tag>.pt` / `<tag>_history.json` to `out_dir`.

    `init_state_dict` warm-starts from an existing model, which is what the
    active-learning loop needs -- retraining from scratch on 30 extra samples would
    throw away everything the surrogate already knows.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = pick_device(device)

    if cfg.n_train is not None and cfg.n_train < len(train_set):
        idx = np.random.default_rng(cfg.seed).permutation(len(train_set))[: cfg.n_train]
        train_set = torch.utils.data.Subset(train_set, idx.tolist())
    norm = val_set.normalizer

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=len(train_set) > cfg.batch_size)
    val_loader = DataLoader(val_set, batch_size=32)

    model = build(cfg.model, **cfg.model_kwargs).to(dev)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * max(1, len(train_loader)), pct_start=0.15
    )

    history: list[dict] = []
    best = {"val": float("inf"), "epoch": -1}
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{cfg.tag}.pt"
    t0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        run: dict[str, float] = {}
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, logs = _losses(model, batch, cfg, norm, dev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in logs.items():
                run[k] = run.get(k, 0.0) + v
        run = {f"train_{k}": v / max(1, len(train_loader)) for k, v in run.items()}
        val = {f"val_{k}": v for k, v in evaluate_loss(model, val_loader, cfg, norm, dev).items()}
        row = {"epoch": epoch, "lr": sched.get_last_lr()[0], **run, **val}
        history.append(row)

        if val["val_total"] < best["val"]:
            best = {"val": val["val_total"], "epoch": epoch}
            torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg)}, ckpt)
        if verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
            print(
                f"  [{cfg.tag}] epoch {epoch:3d}  train {row['train_total']:.4f}  "
                f"val {row['val_total']:.4f}  relL2 {row['val_rel_l2']:.4f}",
                flush=True,
            )

    wall = time.perf_counter() - t0
    summary = {
        "cfg": asdict(cfg),
        "n_parameters": count_parameters(model),
        "n_train": len(train_set),
        "best_epoch": best["epoch"],
        "best_val_total": best["val"],
        "train_seconds": wall,
        "device": str(dev),
        "history": history,
    }
    (out_dir / f"{cfg.tag}_history.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"  [{cfg.tag}] done in {wall / 60:.1f} min, best epoch {best['epoch']}", flush=True)
    return summary


def load_model(ckpt: Path, device: str = "auto") -> tuple[torch.nn.Module, TrainConfig]:
    dev = pick_device(device)
    blob = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = TrainConfig(**blob["cfg"])
    model = build(cfg.model, **cfg.model_kwargs).to(dev)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, cfg
