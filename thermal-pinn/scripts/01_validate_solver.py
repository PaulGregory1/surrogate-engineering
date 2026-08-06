"""Verify the forward solver before any inverse method is allowed to use it.

    python scripts/01_validate_solver.py            # ~3 min
    python scripts/01_validate_solver.py --quick    # ~40 s, coarser refinement windows

Runs the three checks of milestone M1 and writes the numbers, pass/fail included, to
results/solver_validation.json:

  (a) manufactured solutions on all three constitutive laws, observed order 2 in space
      and in time (accepted between 1.9 and 2.1);
  (b) the closed-form semi-infinite constant-flux solution, agreement better than 0.5 %;
  (c) the discrete energy balance on law A, closed to better than 1e-8 relative.

The verdict at the bottom is the gate: nothing downstream is meaningful if it is FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermalpinn.material import LAWS
from thermalpinn.solver import INVERSE_GRID, TRUTH_GRID
from thermalpinn.verify import energy_balance_check, mms_convergence, semi_infinite_check

ORDER_RANGE = (1.9, 2.1)
ERFC_TOL_PCT = 0.5
ENERGY_TOL = 1e-8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/solver_validation.json"))
    ap.add_argument("--quick", action="store_true", help="shorter refinement sweeps")
    args = ap.parse_args()

    # --quick keeps all three checks and all three laws -- it only shrinks the refinement
    # sweeps and drops the fine grid, so a failure still means the same thing.
    mms_kw: dict = {}
    grids = [TRUTH_GRID, INVERSE_GRID]
    if args.quick:
        mms_kw = {"space_nx": (26, 51, 101, 201), "time_steps": (800, 1600, 3200), "time_ref_steps": 32000}
        grids = [INVERSE_GRID]

    t0 = time.perf_counter()
    mms = [mms_convergence(LAWS[name], **mms_kw) for name in ("A", "B", "C")]
    # Always the fine grid: this checks the *truth generator*, it only runs 50 ms, and the
    # coarse grid genuinely cannot resolve the first few milliseconds of a step flux --
    # which is a fact about the inverse model, not a defect of the solver.
    erfc = semi_infinite_check(grid=TRUTH_GRID)
    energy = [energy_balance_check(LAWS[name], grid=grids[0]) for name in ("A", "B", "C")]
    energy += [energy_balance_check(LAWS["A"], grid=g) for g in grids[1:]]

    print("(a) manufactured solutions -- observed order of accuracy")
    print(f"{'law':>4} {'space L2':>9} {'space Linf':>11} {'time L2':>9} {'time Linf':>10}")
    for r in mms:
        print(
            f"{r['law']:>4} {r['space']['order_l2']:>9.3f} {r['space']['order_linf']:>11.3f} "
            f"{r['time']['order_l2']:>9.3f} {r['time']['order_linf']:>10.3f}"
        )
    orders = [v for r in mms for v in (r["space"]["order_l2"], r["space"]["order_linf"], r["time"]["order_l2"], r["time"]["order_linf"])]
    pass_a = all(ORDER_RANGE[0] <= o <= ORDER_RANGE[1] for o in orders)

    print("\n(b) semi-infinite half space, constant flux -- error on the temperature rise")
    print(f"{'t [s]':>7} {'x [mm]':>7} {'rise [K]':>9} {'numeric':>9} {'err %':>8}  counted")
    for row in erfc["rows"]:
        print(
            f"{row['t']:>7.3f} {1e3 * row['x']:>7.2f} {row['rise_exact']:>9.4f} "
            f"{row['rise_numeric']:>9.4f} {row['rel_error_pct']:>+8.4f}  {'yes' if row['counted'] else 'no'}"
        )
    pass_b = erfc["max_abs_error_pct"] < ERFC_TOL_PCT

    print("\n(c) energy balance over the full braking event")
    print(f"{'law':>4} {'grid':>12} {'stored [MJ/m2]':>15} {'scheme':>10} {'trapezoid':>11} {'Tsurf max':>10}")
    for r in energy:
        print(
            f"{r['law']:>4} {r['grid']['nx']:>5}/{r['grid']['dt']:<6.0e} {1e-6 * r['stored_J_per_m2']:>15.4f} "
            f"{r['residual_scheme']:>10.1e} {r['residual_trapezoid']:>11.1e} {r['T_surface_max'] - 273.15:>9.1f}C"
        )
    pass_c = all(r["residual_scheme"] < ENERGY_TOL for r in energy if r["law"] == "A")

    verdict = {
        "order_of_accuracy": {"pass": pass_a, "range": list(ORDER_RANGE), "worst": [min(orders), max(orders)]},
        "semi_infinite": {"pass": pass_b, "tol_pct": ERFC_TOL_PCT, "max_abs_error_pct": erfc["max_abs_error_pct"]},
        "energy_balance": {
            "pass": pass_c,
            "tol": ENERGY_TOL,
            "worst": max(r["residual_scheme"] for r in energy if r["law"] == "A"),
        },
    }
    verdict["all_pass"] = all(v["pass"] for v in verdict.values())

    print()
    for key, v in verdict.items():
        if key != "all_pass":
            print(f"{'PASS' if v['pass'] else 'FAIL'}  {key}")
    print(f"{'PASS' if verdict['all_pass'] else 'FAIL'}  overall   ({time.perf_counter() - t0:.0f} s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"verdict": verdict, "mms": mms, "semi_infinite": erfc, "energy_balance": energy, "quick": args.quick},
            indent=2,
        )
    )
    print(f"wrote {args.out}")
    sys.exit(0 if verdict["all_pass"] else 1)


if __name__ == "__main__":
    main()
