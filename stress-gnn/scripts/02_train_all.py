"""Train the surrogates and the ablations.

    python scripts/02_train_all.py --epochs 60

Main comparison (all on the same coarse meshes, same targets):
  grid-cnn        rasterise to a Cartesian grid, U-Net, sample back at the nodes
  gnn-mp2/5/10/15 mesh graph network, varying the number of message-passing steps
  gnn-mp10-glob   idem + a global latent (O(1)-hop long-range coupling)
  gnn-...-corr    same network, but fed the coarse FEM solution as an input feature
                  (discretisation-error correction rather than a pure surrogate)
  gnn-...-maxloss adds an explicit loss term on the per-part peak stress
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stressgnn.dataset import GraphDataset
from stressgnn.train import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain models that already have a history file")
    args = ap.parse_args()

    runs = [
        TrainConfig(model="gnn", n_message_passing=10, epochs=args.epochs),
        TrainConfig(model="gnn", n_message_passing=10, use_global=True, epochs=args.epochs),
        TrainConfig(model="grid-cnn", epochs=args.epochs),
    ]
    if not args.skip_ablations:
        runs += [
            TrainConfig(model="gnn", n_message_passing=0, epochs=args.epochs),  # pointwise MLP
            TrainConfig(model="gnn", n_message_passing=2, epochs=args.epochs),
            TrainConfig(model="gnn", n_message_passing=5, epochs=args.epochs),
            TrainConfig(model="gnn", n_message_passing=15, epochs=args.epochs),
            TrainConfig(model="gnn", n_message_passing=10, use_global=True, use_coarse_solution=True, epochs=args.epochs),
            TrainConfig(model="gnn", n_message_passing=10, use_global=True, lambda_max=0.3, epochs=args.epochs),
            TrainConfig(model="gnn", n_message_passing=10, use_global=True, use_position=False, epochs=args.epochs),
        ]

    # One dataset object per feature configuration; the graphs are cached in memory.
    cache: dict[tuple, tuple[GraphDataset, GraphDataset]] = {}
    for cfg in runs:
        # A completed run writes its history file last, so its presence means the run
        # finished.  Skipping those makes the sweep resumable after an interruption.
        if not args.force and (args.out_dir / f"{cfg.tag}_history.json").exists():
            print(f"  [{cfg.tag}] already trained, skipping", flush=True)
            continue
        key = (cfg.use_coarse_solution, cfg.use_position, cfg.model in ("grid-cnn",))
        if key not in cache:
            kw = dict(use_coarse_solution=key[0], use_position=key[1], with_grid=key[2])
            tr = GraphDataset(args.data_dir / "train.npz", **kw)
            va = GraphDataset(
                args.data_dir / "val.npz", stats=(tr.mean, tr.std), peak_stats=(tr.peak_mean, tr.peak_std), **kw
            )
            cache[key] = (tr, va)
            print(f"loaded data for {key}: {len(tr)} train / {len(va)} val, {tr.n_features} node features", flush=True)
        tr, va = cache[key]
        train(cfg, tr, va, args.out_dir, device=args.device)


if __name__ == "__main__":
    main()
