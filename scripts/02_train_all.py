"""Train every surrogate on the same data, then the data-efficiency sweep.

    python scripts/02_train_all.py --epochs 120

Models compared:
  pod-linear   POD + ridge on the 7 shape parameters      (classic ROM)
  pod-rbf      POD + kernel ridge on the same             (strong ROM baseline)
  unet         U-Net on the rasterised level set          (geometry-agnostic)
  unet-phys    idem + continuity residual in the loss
  fno          Fourier neural operator
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flowsurrogate.data import load_splits
from flowsurrogate.geometry import Domain
from flowsurrogate.models import PODRegressor
from flowsurrogate.train import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--sweep-epochs", type=int, default=80)
    ap.add_argument("--sweep-sizes", type=int, nargs="*", default=[80, 160, 320, 640])
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    domain = Domain()
    train_set, val_set, _, norm = load_splits(args.data_dir, domain)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"train {len(train_set)} | val {len(val_set)}", flush=True)

    # ---------------------------------------------------------------- ROM baselines
    for kernel in ("linear", "rbf"):
        t0 = time.perf_counter()
        pod = PODRegressor(n_modes=64, kernel=kernel, alpha=1e-3 if kernel == "rbf" else 1.0)
        pod.fit(train_set.params, train_set.re, train_set.y.numpy(), train_set.cd.numpy())
        wall = time.perf_counter() - t0
        with open(args.out_dir / f"pod-{kernel}.pkl", "wb") as fh:
            pickle.dump(pod, fh)
        (args.out_dir / f"pod-{kernel}_history.json").write_text(
            json.dumps({"cfg": {"model": f"pod-{kernel}"}, "n_parameters": pod.n_parameters,
                        "n_train": len(train_set), "train_seconds": wall}, indent=2)
        )
        print(f"  [pod-{kernel}] fitted in {wall:.1f}s", flush=True)

    # -------------------------------------------------------------- neural surrogates
    runs = [
        TrainConfig(model="unet", epochs=args.epochs, model_kwargs={"base": 32, "depth": 3}),
        TrainConfig(model="unet", epochs=args.epochs, lambda_phys=0.02, model_kwargs={"base": 32, "depth": 3}),
        TrainConfig(model="fno", epochs=args.epochs, model_kwargs={"width": 32, "modes": 16, "n_layers": 4}),
    ]
    for cfg in runs:
        train(cfg, train_set, val_set, args.out_dir, device=args.device)

    # ------------------------------------------------------------- data efficiency
    if not args.skip_sweep:
        for n in args.sweep_sizes:
            if n > len(train_set):
                continue
            cfg = TrainConfig(model="unet", epochs=args.sweep_epochs, n_train=n, model_kwargs={"base": 32, "depth": 3})
            train(cfg, train_set, val_set, args.out_dir, device=args.device)
            pod = PODRegressor(n_modes=min(64, n - 1), kernel="rbf", alpha=1e-3)
            idx = np.random.default_rng(0).permutation(len(train_set))[:n]
            pod.fit(train_set.params[idx], train_set.re[idx], train_set.y.numpy()[idx], train_set.cd.numpy()[idx])
            with open(args.out_dir / f"pod-rbf-n{n}.pkl", "wb") as fh:
                pickle.dump(pod, fh)
            print(f"  [pod-rbf-n{n}] fitted", flush=True)


if __name__ == "__main__":
    main()
