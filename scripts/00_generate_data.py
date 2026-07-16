"""Generate the LBM dataset.

    python scripts/00_generate_data.py --n-train 640 --n-val 64 --n-test 128

Splits:
  train/val/test  same distribution (Re in [30, 110], Fourier coefficients in +-0.16)
  ood             deliberately outside it (stronger deformations, higher Re) -- used
                  to measure how the surrogates degrade off-distribution rather than
                  only reporting the in-distribution number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowsurrogate.data import GenSpec, generate_split
from flowsurrogate.geometry import Domain
from flowsurrogate.lbm import SolverConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--n-train", type=int, default=640)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--n-ood", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--avg", type=int, default=2500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    domain = Domain()
    solver_cfg = SolverConfig(n_steps=args.steps, n_avg=args.avg)
    specs = [
        GenSpec("train", args.n_train, seed=1),
        GenSpec("val", args.n_val, seed=2),
        GenSpec("test", args.n_test, seed=3),
        GenSpec("ood", args.n_ood, seed=4, re_range=(110.0, 150.0), coeff_range=0.24, r0_range=(7.5, 12.0)),
    ]
    for spec in specs:
        if spec.n > 0:
            generate_split(
                spec,
                domain,
                solver_cfg,
                args.data_dir,
                batch=args.batch,
                device=args.device,
                compile=not args.no_compile,
            )


if __name__ == "__main__":
    main()
