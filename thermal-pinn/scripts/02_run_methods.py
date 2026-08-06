"""Establish that the classical baselines are trustworthy, before anything is compared to them.

    python scripts/02_run_methods.py            # ~11 min, most of it the crime test
    python scripts/02_run_methods.py --quick     # ~1 min, on a 4 s event

This is milestone M3, and its three checks are not formalities.  Every claim the repo will
make about the PINN is a claim about the *difference* between it and these three methods, so
a baseline that is quietly broken, or quietly detuned, would invent the whole result.

  (a) the inverse crime -- zero noise, the model's own grid, no discretisation gap.  Each
      method has to recover the flux to better than 1 % in band.  This is the only place in
      the repo where the crime is committed, and it is committed on purpose: it separates a
      broken implementation from one defeated by the physics.
  (b) the adjoint gradient against central differences, on five components drawn at random,
      for all three constitutive laws.  Better than 1e-4 relative or every act 2 and act 3
      result is void, because they all descend that gradient.
  (c) on realistic noise the three land within a factor 2 of each other.  Spread wider than
      that means one of them is detuned, and a detuned baseline is a straw man.

Writes results/baseline_validation.json.  The three acts themselves arrive with M4 and M5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermalpinn.classical import InverseModel, alifanov, beck, beck_by_discrepancy, inverse_crime_data, tikhonov
from thermalpinn.material import LAWS, LAW_A, to_celsius
from thermalpinn.metrics import band_error, surface_peak_error
from thermalpinn.naive import cutoff_frequency
from thermalpinn.scenario import SHOWCASE, Scenario
from thermalpinn.solver import TRUTH_GRID, forward, sensor_signal

CRIME_TOL_PCT = 1.0
GRADIENT_TOL = 1.0e-4
SPREAD_TOL = 2.0
#: Beck's least-smoothed setting that is still numerically stable, which is not the same as
#: its least-smoothed setting.  The sequential recursion amplifies per step roughly like
#: S_2 / S_1, and after 10 ms the sensor at 2 mm has barely heard of the flux at all
#: (S_1 = 1.6e-9 K per W/m^2), so one flux value per measurement with a two-step look-ahead
#: diverges exponentially along the record -- 10^25 % here.  Two intervals per flux value
#: with a four-step look-ahead is the first setting that both runs and barely smooths.
CRIME_BECK = {"stride": 2, "r": 4}
#: Zero noise leaves the discrepancy principle nothing to stop on, so the iterative method
#: gets an iteration budget instead and the band error it reaches is what the gate reads.
#: Conjugate gradients needs a couple of hundred iterations to get under 1 % here -- 2000
#: unknowns and badly conditioned normal equations -- and that slow convergence is a
#: property of the algorithm, not a doubt about it.  On real data it stops after about 30,
#: because real noise gives it something to stop on.
CRIME_SIGMA = 0.0


def gradient_check(scenario: Scenario, laws, rng: np.random.Generator, n_components: int = 5) -> list[dict]:
    """Central differences against the adjoint, at a point that is not the optimum.

    Run on a 1 s event: the gradient is a property of the model, not of the record length,
    and each finite difference costs a full forward solve.  The perturbation is 1 % of the
    peak flux -- large enough that the difference is not round-off, small enough that the
    O(eps^2) truncation stays under the tolerance.  For law A the model is exactly linear in
    the flux, so the central difference is exact and the comparison is pure round-off.
    """
    rows = []
    for law in laws:
        model = InverseModel(scenario, law)
        q_true = scenario.q_true(model.control_times())
        Y = model.solve(q_true, store=False).obs + rng.normal(0.0, scenario.sigma_noise, model.n_control + 1)

        q = 0.7 * q_true  # off the optimum, where the gradient is not itself near zero
        run = model.solve(q)
        g = model.gradient(run.field, run.obs[1:] - Y[1:])

        def objective(candidate: np.ndarray) -> float:
            r = model.solve(candidate, store=False).obs[1:] - Y[1:]
            return 0.5 * float(r @ r)

        eps = 1.0e-2 * scenario.q_peak
        for j in sorted(rng.choice(model.n_control, size=n_components, replace=False)):
            up, down = q.copy(), q.copy()
            up[j] += eps
            down[j] -= eps
            fd = (objective(up) - objective(down)) / (2.0 * eps)
            rows.append(
                {
                    "law": law.name,
                    "component": int(j),
                    "adjoint": float(g[j]),
                    "central_difference": float(fd),
                    "relative": float(abs(fd - g[j]) / max(abs(fd), 1e-300)),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/baseline_validation.json"))
    ap.add_argument("--quick", action="store_true", help="4 s event, smaller CG budget")
    args = ap.parse_args()

    scenario = Scenario(t_end=4.0) if args.quick else SHOWCASE
    crime_budget = 80 if args.quick else 240
    material = LAW_A
    model = InverseModel(scenario, material)
    f_c = cutoff_frequency(scenario.x_s, material.alpha0)
    dt_control = 1.0 / scenario.f_sample
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()

    print(f"inverse model: nx = {model.N}, dt = {model.dt}, {model.n_control} flux intervals over {scenario.t_end} s")
    print(f"recoverable band: f < {f_c:.3f} Hz\n")

    duhamel = model.step_response()
    q_ref = scenario.q_true(model.control_times())

    # ---------------------------------------------------------------------- (a) the crime
    q_crime, Y_crime = inverse_crime_data(model)
    crime = {}
    crime["beck"] = beck(model, Y_crime, duhamel=duhamel, **CRIME_BECK)
    crime["tikhonov"] = tikhonov(model, Y_crime, sigma=CRIME_SIGMA, duhamel=duhamel)
    crime["alifanov"] = alifanov(model, Y_crime, sigma=CRIME_SIGMA, max_iter=crime_budget)

    print(f"(a) inverse crime -- zero noise, the model's own grid (tolerance {CRIME_TOL_PCT} % in band)")
    crime_rows = []
    for name, out in crime.items():
        e = band_error(out["q"], q_crime, dt_control, f_c)
        knob = {"beck": f"stride {CRIME_BECK['stride']}, r {CRIME_BECK['r']}",
                "tikhonov": f"lambda {out.get('lambda', 0):.2e}",
                "alifanov": f"{out.get('iterations', 0)} iterations"}[name]
        crime_rows.append({"method": name, "band_rmse_pct": e["rmse_pct"], "knob": knob,
                           "pass": bool(e["rmse_pct"] < CRIME_TOL_PCT)})
        print(f"    {name:9s} {e['rmse_pct']:8.4f} %   ({knob})")
    pass_a = all(r["pass"] for r in crime_rows)

    # ------------------------------------------------------------------- (b) the gradient
    print(f"\n(b) adjoint gradient vs central differences (tolerance {GRADIENT_TOL:.0e} relative)")
    grad_rows = gradient_check(Scenario(t_end=1.0), [LAWS[n] for n in ("A", "B", "C")], rng)
    for law in ("A", "B", "C"):
        worst = max(r["relative"] for r in grad_rows if r["law"] == law)
        print(f"    law {law}: worst of 5 components  {worst:.3e}")
    worst_gradient = max(r["relative"] for r in grad_rows)
    pass_b = worst_gradient < GRADIENT_TOL

    # -------------------------------------------------------------------- (c) real noise
    truth = forward(scenario, material, scenario.q_true, grid=TRUTH_GRID, t_end=scenario.t_end)
    _, Y, Y_clean = sensor_signal(truth, scenario)
    model_gap = float(np.max(np.abs(model.solve(q_ref, store=False).obs - Y_clean)))

    print(f"\n(c) realistic noise: sigma = {scenario.sigma_noise} K, truth on {TRUTH_GRID.nx}/{TRUTH_GRID.dt:g}")
    print(f"    inverse model vs truth at the sensor, noise free: {model_gap:.4f} K "
          f"-- the floor no method can beat")
    noisy = {
        "beck": beck_by_discrepancy(model, Y, sigma=scenario.sigma_noise, duhamel=duhamel),
        "tikhonov": tikhonov(model, Y, sigma=scenario.sigma_noise, duhamel=duhamel),
        "alifanov": alifanov(model, Y, sigma=scenario.sigma_noise),
    }
    print(f"\n    {'method':10s} {'band RMSE':>10} {'of peak':>9} {'max err':>11} {'T_surf peak':>12} {'knob':>26}")
    noisy_rows = []
    for name, out in noisy.items():
        e = band_error(out["q"], q_ref, dt_control, f_c)
        surface = model.solve(out["q"]).field[:, 0]
        peak = surface_peak_error(surface, truth.T_surface, scenario.T0)
        knob = {
            "beck": f"stride {out.get('stride')}, r {out.get('r')}",
            "tikhonov": f"lambda {out.get('lambda', 0):.3e}",
            "alifanov": f"{out.get('iterations')} iters, {out.get('stop')}",
        }[name]
        noisy_rows.append(
            {
                "method": name,
                "band_rmse_pct": e["rmse_pct"],
                "band_rmse_pct_of_peak": e["rmse_pct_of_peak"],
                "band_max_abs": e["max_abs"],
                "surface_peak_error_pct": peak,
                "knob": knob,
                "sweep": out.get("sweep"),
            }
        )
        print(f"    {name:10s} {e['rmse_pct']:9.3f} % {e['rmse_pct_of_peak']:8.3f} % {e['max_abs']:11.3e} "
              f"{peak:+11.3f} % {knob:>26}")

    # How much the choice of lambda was worth -- and how badly the choice that does not know
    # sigma does.  Reported rather than dropped: the usual way of tuning the classical method
    # without the noise level fails on this problem.
    tk = noisy["tikhonov"]
    lam_choice = {
        "lambda_discrepancy": tk["lambda"],
        "lambda_l_curve": tk["lambda_l_curve"],
        "band_rmse_pct_discrepancy": band_error(tk["q"], q_ref, dt_control, f_c)["rmse_pct"],
        "band_rmse_pct_l_curve": band_error(tk["q_l_curve"], q_ref, dt_control, f_c)["rmse_pct"],
    }
    print(f"\n    choosing Tikhonov's lambda: discrepancy {lam_choice['lambda_discrepancy']:.3e} "
          f"-> {lam_choice['band_rmse_pct_discrepancy']:.2f} % in band")
    print(f"                                L-curve     {lam_choice['lambda_l_curve']:.3e} "
          f"-> {lam_choice['band_rmse_pct_l_curve']:.2f} % in band")
    print("    the L-curve has no corner to find here: the penalty axis spans five decades "
          "and the residual under two")

    scores = [r["band_rmse_pct"] for r in noisy_rows]
    spread = max(scores) / min(scores)
    pass_c = spread <= SPREAD_TOL
    print(f"\n    spread between best and worst: {spread:.2f}x  (tolerance {SPREAD_TOL}x)")
    print(f"    out of band, unrecoverable by anyone: "
          f"{band_error(q_ref, q_ref, dt_control, f_c)['out_of_band_rms']:.3e} W/m2 rms")
    print(f"    true surface peak {to_celsius(truth.T_surface.max()):.1f} degC, "
          f"sensor peak {to_celsius(Y_clean.max()):.1f} degC")

    verdict = {
        "inverse_crime": {"pass": pass_a, "tol_pct": CRIME_TOL_PCT,
                          "worst_pct": max(r["band_rmse_pct"] for r in crime_rows)},
        "adjoint_gradient": {"pass": pass_b, "tol": GRADIENT_TOL, "worst_relative": worst_gradient},
        "method_spread": {"pass": pass_c, "tol": SPREAD_TOL, "spread": spread},
    }
    verdict["all_pass"] = all(v["pass"] for v in verdict.values() if isinstance(v, dict))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "scenario": {"name": scenario.name, "t_end": scenario.t_end, "x_s": scenario.x_s,
                             "sigma_noise": scenario.sigma_noise, "f_sample": scenario.f_sample},
                "model": {"nx": model.N, "dt": model.dt, "n_control": model.n_control,
                          "theta": model.theta, "n_startup": model.n_startup},
                "f_c": f_c,
                "model_vs_truth_K": model_gap,
                "inverse_crime": crime_rows,
                "gradient_check": grad_rows,
                "realistic_noise": noisy_rows,
                "lambda_selection": lam_choice,
                "alifanov_history": noisy["alifanov"]["history"],
                "quick": args.quick,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")
    for key, v in verdict.items():
        if key != "all_pass":
            print(f"{'PASS' if v['pass'] else 'FAIL'}  {key}")
    print(f"{'PASS' if verdict['all_pass'] else 'FAIL'}  overall   ({time.perf_counter() - t0:.0f} s)")
    sys.exit(0 if verdict["all_pass"] else 1)


if __name__ == "__main__":
    main()
