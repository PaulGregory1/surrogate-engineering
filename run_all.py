"""Run the whole pipeline end to end.

    python run_all.py            # full run, ~50 min on an M-series laptop
    python run_all.py --quick    # reduced sizes, ~8 min, for checking it works

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
    opt = ["scripts/04_optimize.py"]
    if args.quick:
        gen += ["--n-train", "80", "--n-val", "20", "--n-test", "30", "--n-res", "20", "--n-ood", "20", "--h-fine", "0.014"]
        train += ["--epochs", "15", "--skip-ablations"]
        opt += ["--n-candidates", "30"]

    run("generate data", gen, ROOT / "data" / "ood.npz", args.force)
    run(
        "mesh convergence",
        ["scripts/01_validate_fem.py"] + (["--n-designs", "2", "--sizes", "0.06", "0.03", "0.014"] if args.quick else []),
        ROOT / "results" / "fem_convergence.json",
        args.force,
    )
    run("train surrogates", train, ROOT / "runs" / "grid-cnn.pt", args.force)
    run("report", ["scripts/03_report.py"], ROOT / "RESULTS.md", args.force)
    run("design selection", opt, ROOT / "results" / "optimization.json", args.force)
    subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
