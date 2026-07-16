"""Run the whole pipeline end to end.

    python run_all.py            # full run, ~2 h on an M-series laptop
    python run_all.py --quick    # reduced sizes, ~15 min, for checking it works

Each stage is skipped if its output already exists; pass --force to redo everything.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(name: str, args: list[str], produces: Path, force: bool) -> None:
    if produces.exists() and not force:
        print(f"== {name}: already present at {produces}, skipping")
        return
    print(f"== {name}: {' '.join(args)}", flush=True)
    t0 = time.perf_counter()
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    print(f"== {name}: {(time.perf_counter() - t0) / 60:.1f} min\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    gen = ["scripts/00_generate_data.py"]
    train = ["scripts/02_train_all.py"]
    opt = ["scripts/04_optimize_shape.py"]
    validate = ["scripts/01_validate_solver.py"]
    if args.quick:
        gen += ["--n-train", "128", "--n-val", "32", "--n-test", "32", "--n-ood", "32", "--steps", "4000", "--avg", "1200"]
        train += ["--epochs", "20", "--sweep-epochs", "15", "--sweep-sizes", "32", "64", "128"]
        opt += ["--iters", "80", "--n-starts", "8", "--steps", "4000", "--al-rounds", "1", "--al-epochs", "8"]
        validate += ["--quick"]

    run("generate data", gen, ROOT / "data" / "ood.npz", args.force)
    run("validate solver", validate, ROOT / "results" / "solver_validation.json", args.force)
    run("train surrogates", train, ROOT / "runs" / "fno.pt", args.force)
    run("report", ["scripts/03_report.py"], ROOT / "RESULTS.md", args.force)
    run("optimise shape", opt, ROOT / "results" / "optimization.json", args.force)
    subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
