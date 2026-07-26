"""Generate the FEM dataset.

    python scripts/00_generate_data.py --n-train 400 --n-val 60 --n-test 120

Splits:
  train/val/test  coarse mesh size drawn from U[0.035, 0.060]
  res             the test designs re-meshed at h = 0.025, a resolution never seen
                  during training -- checks that the graph network is actually
                  discretisation-invariant rather than memorising one mesh density
  ood             4-6 holes (training tops out at 4) and load angles outside the
                  training range
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from stressgnn.dataset import GenSpec, generate_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=120)
    ap.add_argument("--n-res", type=int, default=60)
    ap.add_argument("--n-ood", type=int, default=60)
    ap.add_argument("--h-fine", type=float, default=0.010)
    args = ap.parse_args()

    specs = [
        GenSpec("train", args.n_train, seed=11, h_fine=args.h_fine),
        GenSpec("val", args.n_val, seed=12, h_fine=args.h_fine),
        GenSpec("test", args.n_test, seed=13, h_fine=args.h_fine),
        GenSpec("res", args.n_res, seed=13, h_fine=args.h_fine, h_coarse_fixed=0.025),
        GenSpec(
            "ood",
            args.n_ood,
            seed=14,
            h_fine=args.h_fine,
            n_holes_choices=(4, 5, 6),
            angle_range=(-np.pi, -np.pi / 2),
        ),
    ]
    for spec in specs:
        if spec.n > 0:
            generate_split(spec, args.data_dir)


if __name__ == "__main__":
    main()
