"""Verify the LBM solver against published cylinder data before trusting the dataset.

A surrogate can only ever be as good as the solver it is fitted to, so this runs the
one case with well-established reference values -- steady flow past a circular
cylinder -- at two blockage ratios and two Reynolds numbers, and reports the drag
after the standard blockage correction C_d,inf ~ C_d (1 - beta)^2.

Reference drag for an unbounded 2-D cylinder (Tritton 1959; Fornberg 1980):
    Re = 20 -> C_d = 2.05      Re = 40 -> C_d = 1.55
Reference recirculation length L_w / D (Coutanceau & Bouard 1977):
    Re = 20 -> 0.93            Re = 40 -> 2.35

    python scripts/01_validate_solver.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flowsurrogate.geometry import Domain
from flowsurrogate.lbm import LBMSolver, SolverConfig

REFERENCE_CD = {20.0: 2.05, 40.0: 1.55}
REFERENCE_LW = {20.0: 0.93, 40.0: 2.35}


def recirculation_length(u: np.ndarray, domain: Domain, radius: float) -> float:
    """Length of the reversed-flow region on the centreline, in diameters."""
    row = int(domain.cy)
    line = u[row, int(domain.cx + radius) :]
    neg = line < 0
    if not neg.any():
        return 0.0
    end = int(np.argmax(~neg)) if (~neg).any() else len(line)
    return end / (2.0 * radius)


def run_case(nx: int, ny: int, cx: int, radius: float, reynolds: list[float], steps: int, device: str) -> list[dict]:
    domain = Domain(nx=nx, ny=ny, cx=float(cx), cy=ny / 2.0)
    x, y = domain.grid()
    mask = ((x - domain.cx) ** 2 + (y - domain.cy) ** 2) < radius**2
    masks = np.stack([mask] * len(reynolds))
    d_eq = np.full(len(reynolds), 2 * radius)
    solver = LBMSolver(domain, SolverConfig(n_steps=steps, n_avg=steps // 4), device=device)
    out = solver.run(masks, np.array(reynolds), d_eq)

    beta = 2 * radius / ny
    rows = []
    for i, re in enumerate(reynolds):
        cd = float(out["cd"][i])
        rows.append(
            {
                "grid": f"{nx}x{ny}",
                "D_cells": 2 * radius,
                "blockage_pct": 100 * beta,
                "Re": re,
                "cd_raw": cd,
                "cd_corrected": cd * (1 - beta) ** 2,
                "cd_reference": REFERENCE_CD[re],
                "cd_error_pct": 100 * (cd * (1 - beta) ** 2 - REFERENCE_CD[re]) / REFERENCE_CD[re],
                "Lw_over_D": recirculation_length(out["u"][i], domain, radius),
                "Lw_reference": REFERENCE_LW[re],
                "cd_drift": float(out["cd_drift"][i]),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/solver_validation.json"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="production grid only")
    args = ap.parse_args()

    cases = [(192, 96, 56, 10.0, 9000)]  # production grid
    if not args.quick:
        cases += [
            (288, 192, 72, 10.0, 14000),  # halved blockage
            (288, 192, 72, 15.0, 14000),  # finer resolution of the body
        ]

    rows: list[dict] = []
    for nx, ny, cx, r, steps in cases:
        rows += run_case(nx, ny, cx, r, [20.0, 40.0], steps, args.device)

    hdr = f"{'grid':>9} {'D':>4} {'block%':>7} {'Re':>5} {'Cd':>7} {'Cd*(1-b)^2':>11} {'ref':>6} {'err%':>7} {'Lw/D':>6} {'ref':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['grid']:>9} {r['D_cells']:>4.0f} {r['blockage_pct']:>7.1f} {r['Re']:>5.0f} "
            f"{r['cd_raw']:>7.3f} {r['cd_corrected']:>11.3f} {r['cd_reference']:>6.2f} "
            f"{r['cd_error_pct']:>+7.1f} {r['Lw_over_D']:>6.2f} {r['Lw_reference']:>5.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
