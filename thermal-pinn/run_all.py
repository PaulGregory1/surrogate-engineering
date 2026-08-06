"""Run the whole pipeline end to end.

    python run_all.py            # full run
    python run_all.py --quick    # reduced sweeps, for checking it works

Each stage is skipped if its output already exists; pass --force to redo everything.
Stages are added as milestones land; today the repo stops after act 1.

The act 1 stage is the expensive one by a wide margin -- eleven trainings, and a training is
minutes rather than the seconds every solver stage before it costs.  `--quick` cuts its
iteration counts by a factor of ten, which is enough to check that the pipeline runs and not
nearly enough to produce a number worth quoting.

Script numbers follow the repo layout rather than the running order -- 06 owns
figs/illposed.png and runs second, because the fact that the problem is ill posed is what
the methods after it are answers to.
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

    quick = ["--quick"] if args.quick else []

    run(
        "validate solver",
        ["scripts/01_validate_solver.py", *quick],
        ROOT / "results" / "solver_validation.json",
        args.force,
    )
    run(
        "ill-posedness",
        ["scripts/06_explainer_figures.py", *quick],
        ROOT / "results" / "illposedness.json",
        args.force,
    )
    run(
        "classical baselines",
        ["scripts/02_run_methods.py", *quick],
        ROOT / "results" / "baseline_validation.json",
        args.force,
    )
    run(
        "act 1 -- the PINN against the baselines",
        ["scripts/03_act1.py", *quick],
        ROOT / "results" / "act1.json",
        args.force,
    )
    subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
