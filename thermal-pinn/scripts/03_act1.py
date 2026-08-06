"""Act 1: the fair fight, and the first time the PINN is scored against anything.

    python scripts/03_act1.py            # long: eleven trainings
    python scripts/03_act1.py --quick    # a tenth of the iterations, to check it runs

Law A -- constant properties, no radiation, one unknown function.  Superposition holds, so
Beck and Tikhonov are exactly applicable and Alifanov's adjoint is exact.  This is the act
where the classical route is at its strongest, and M3 measured how strong that is: all three
read a surface they cannot see to within 0.14 % of its 600 K rise.  The plan's expectation
here is a draw at best, and its acceptance criterion is written accordingly -- the PINN
either lands within a factor 1.5 of the *best* baseline on the surface peak, or the gap is
measured and written into the README exactly as it came out.  This script decides which, and
it is allowed to return either answer.

Two stages, in this order and never the other way round.

  (1) **The weights, on the tuning scenario and nowhere else.**  `lambda_r` and `lambda_b`
      come from balancing gradient norms, and a multiplier on `lambda_r` is then swept over
      four decades with the whole sweep published, because the weight decides the answer
      (plan, 8.3).  The scenario is `scenario.TUNING`, which is neither the showcase the
      figures are drawn from nor any case in the M6 table: a weight chosen on either of those
      would be a hyperparameter fitted to the thing it is about to be scored on.

      `lambda_r = 0` is run as the left end of the sweep.  With the physics term switched off
      the PINN is an interpolator of noisy readings from 2 mm down, and it has no mechanism at
      all for knowing what the surface did -- so it has to collapse.  It is the control that
      says the residual is doing the work, and it is the cheapest way to catch a setup where
      the answer is really coming from somewhere else.

      One further ablation runs at the chosen weight: the ramp the hard initial condition is
      built from.  `sqrt(t) N` against `t N`, everything else identical.  It is here rather
      than in prose because it was the single most expensive mistake in building this file
      and the claim should be regenerated rather than remembered.

  (2) **Five seeds on the showcase**, at the weight stage 1 chose, against the three
      baselines on the same record.  Every seed is stored rather than only their mean: a PINN
      is a variable object, and concealing that would be the single largest credibility
      defect this repo could have (plan, 8.7).

Writes results/act1.json and figs/act1_linear.png.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermalpinn import plots
from thermalpinn.classical import InverseModel, alifanov, beck_by_discrepancy, tikhonov
from thermalpinn.material import LAW_A, to_celsius
from thermalpinn.metrics import band_error, lowpass, surface_peak_error
from thermalpinn.naive import cutoff_frequency
from thermalpinn.pinn import PINNConfig, train
from thermalpinn.scenario import SHOWCASE, TUNING, Scenario
from thermalpinn.solver import TRUTH_GRID, forward, sensor_signal

#: The plan's M4 gate: within this factor of the best baseline on the surface peak, or the
#: gap goes into the README as measured.
PEAK_FACTOR = 1.5
#: What the sweep is scored on, declared before it is run.  The flux in band rather than the
#: surface peak: the peak is the headline number and choosing a hyperparameter on the
#: headline is how a sweep turns into a fit.
SWEEP_CRITERION = "band_rmse_pct"
#: Four decades of residual weight, plus the ablation at the left end.
#:
#: The range is not centred on the balanced weight, and that is a measurement rather than a
#: preference.  Balancing gradient norms at initialisation sets `lambda_r` by what the two
#: terms are worth *before* either is satisfied, and by the time the sensor term has reached
#: the noise floor the residual needs several decades more weight than that to be enforced at
#: all: at the balanced value the converged network buys a better fit to the noise by
#: violating the heat equation, and it is right to, because at that weight the violation is
#: cheap.  Measured on the tuning scenario, the flux improves monotonically from x1 to x100
#: with no sign of the trivial solution, so the sweep is placed where the answer is.
SWEEP_SCALES = (0.0, 1.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4)
#: How much worse switching the physics off has to be before the control counts as passed.
ABLATION_FACTOR = 2.0
SEEDS = (0, 1, 2, 3, 4)


def score(q: np.ndarray, T_surface: np.ndarray, *, q_ref, T_surface_true, T0, dt_c, f_c) -> dict:
    e = band_error(q, q_ref, dt_c, f_c)
    return {
        "band_rmse_pct": e["rmse_pct"],
        "band_rmse_pct_of_peak": e["rmse_pct_of_peak"],
        "band_max_abs": e["max_abs"],
        "surface_peak_error_pct": surface_peak_error(T_surface, T_surface_true, T0),
    }


def cache_path(root: Path, scenario: Scenario, cfg: PINNConfig, seed: int) -> Path:
    """One file per training, named by everything that decides its result.

    Eleven trainings is over an hour, which is long enough that losing the tenth to a
    machine going to sleep is worth engineering against.  The key is the full configuration
    plus the scenario and the seed, so a cached entry can only ever be returned to the run
    that would have produced it; changing any tunable invalidates the lot.
    """
    key = json.dumps({"scenario": scenario.name, "seed": seed, "law": LAW_A.name,
                      "cfg": asdict(cfg)}, sort_keys=True)
    return root / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"


def run_pinn(scenario: Scenario, Y: np.ndarray, cfg: PINNConfig, seed: int, model: InverseModel,
             cache: Path | None = None, **kw) -> tuple:
    """Train once and score the two answers it gives for the surface.

    `direct` is the network's own surface temperature -- what the method actually claims.
    `via model` pushes its flux through the same finite-volume model the baselines are scored
    with, so the only difference left is the flux itself.  The gap between the two is not a
    detail: it is how far the trained network is from obeying the physics it was scored on,
    and a method whose two answers disagree has not converged whatever its losses say.
    """
    store = None if cache is None else cache_path(cache, scenario, cfg, seed)
    if store is not None and store.exists():
        with np.load(store, allow_pickle=False) as z:
            row = json.loads(str(z["row"]))
            return row, SimpleNamespace(q=z["q"], t_surface=z["t_surface"], T_surface=z["T_surface"])

    r = train(scenario, LAW_A, Y, config=cfg, seed=seed)
    direct = score(r.q, r.T_surface, **kw)
    try:
        via = score(r.q, model.solve(r.q).field[:, 0], **kw)["surface_peak_error_pct"]
    except RuntimeError:
        # A flux wild enough to stop the forward model's Newton solve converging.  That is
        # itself a result about the estimate, not a reason to lose the other ten trainings,
        # so it is recorded as such and the run carries on.
        via = None
    row = {
        "seed": seed,
        **direct,
        "surface_peak_error_pct_via_model": via,
        "sensor_fit_rms_K": float(np.sqrt(np.mean((r.sensor_fit - Y) ** 2))),
        "lambda_r": r.weights["lambda_r"],
        "lambda_b": r.weights["lambda_b"],
        "loss_data": r.history[-1]["loss_data"],
        "loss_residual": r.history[-1]["loss_residual"],
        "loss_boundary": r.history[-1]["loss_boundary"],
        "seconds": r.seconds,
    }
    if store is not None:
        store.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(store, row=np.array(json.dumps(row)), q=r.q,
                            t_surface=r.t_surface, T_surface=r.T_surface)
    return row, r


def summarise(rows: list[dict], key: str) -> dict:
    v = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "median": float(np.median(v)),
        "min": float(v.min()),
        "max": float(v.max()),
        "worst_abs": float(np.abs(v).max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/act1.json"))
    ap.add_argument("--fig", type=Path, default=Path("figs/act1_linear.png"))
    ap.add_argument("--quick", action="store_true", help="a tenth of the iterations, one sweep decade")
    ap.add_argument("--device", default="cpu", help="cpu (default, deterministic) or mps")
    ap.add_argument("--cache", type=Path, default=Path("data/act1_cache"),
                    help="reuse trainings already done with this exact configuration; "
                         "pass an empty string to retrain everything")
    ap.add_argument("--n-adam", type=int, default=None)
    ap.add_argument("--n-lbfgs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cache = args.cache if args.cache and str(args.cache) else None
    base = PINNConfig(device=args.device)
    if args.quick:
        base = base.quick()
    if args.n_adam is not None:
        base = replace(base, n_adam=args.n_adam)
    if args.n_lbfgs is not None:
        base = replace(base, n_lbfgs=args.n_lbfgs)
    scales = (0.0, 1.0, 1.0e1) if args.quick else SWEEP_SCALES

    f_c = cutoff_frequency(SHOWCASE.x_s, LAW_A.alpha0)
    dt_c = 1.0 / SHOWCASE.f_sample
    t0 = time.perf_counter()
    print(f"act 1 -- law A, {base.n_adam} Adam + {base.n_lbfgs} L-BFGS on {args.device}, "
          f"recoverable band f < {f_c:.3f} Hz\n")

    # ------------------------------------------------------- (1) the weight, on TUNING only
    tune_truth = forward(TUNING, LAW_A, TUNING.q_true, grid=TRUTH_GRID)
    _, tune_Y, _ = sensor_signal(tune_truth, TUNING)
    tune_model = InverseModel(TUNING, LAW_A)
    tune_kw = dict(q_ref=TUNING.q_true(tune_model.control_times()), T_surface_true=tune_truth.T_surface,
                   T0=TUNING.T0, dt_c=dt_c, f_c=f_c)

    print("(1) the residual weight, swept on the tuning scenario -- never on the showcase")
    print(f"    scored on {SWEEP_CRITERION}; one seed each, seed 0")
    print(f"    {'lambda_r x':>12} {'lambda_r':>11} {'band RMSE':>11} {'T_surf peak':>12} {'sensor fit':>11}")
    sweep = []
    for scale in scales:
        cfg = replace(base, lambda_r_scale=scale)
        row, _ = run_pinn(TUNING, tune_Y, cfg, 0, tune_model, cache=cache, **tune_kw)
        row["lambda_r_scale"] = scale
        sweep.append(row)
        tag = "  (physics off)" if scale == 0.0 else ""
        print(f"    {scale:12.0e} {row['lambda_r']:11.3e} {row['band_rmse_pct']:10.3f} % "
              f"{row['surface_peak_error_pct']:+11.3f} % {row['sensor_fit_rms_K']:10.3f} K{tag}")

    live = [r for r in sweep if r["lambda_r_scale"] != 0.0]
    chosen = min(live, key=lambda r: r[SWEEP_CRITERION])
    off = next((r for r in sweep if r["lambda_r_scale"] == 0.0), None)
    ablation = None
    if off is not None:
        cost = off[SWEEP_CRITERION] / chosen[SWEEP_CRITERION]
        ablation = {"band_rmse_pct": off[SWEEP_CRITERION], "factor_worse": cost,
                    "tol": ABLATION_FACTOR, "pass": bool(cost >= ABLATION_FACTOR)}
        print(f"\n    with the heat equation switched off the same network is {cost:.1f}x worse in band "
              f"-- the residual is what turns it from an interpolator into an inversion")

    # The other ablation, and the one that cost the most to find: the ramp the hard initial
    # condition uses.  Run at the chosen weight and on the same scenario, so the only thing
    # that differs between this row and `chosen` is `t N` against `sqrt(t) N`.
    linear_cfg = replace(base, lambda_r_scale=chosen["lambda_r_scale"], ic_ansatz="linear")
    linear_row, _ = run_pinn(TUNING, tune_Y, linear_cfg, 0, tune_model, cache=cache, **tune_kw)
    linear_row["ic_ansatz"] = "linear"
    ic_ablation = {
        "sqrt_t": {k: chosen[k] for k in ("band_rmse_pct", "surface_peak_error_pct",
                                          "surface_peak_error_pct_via_model", "loss_residual")},
        "linear": {k: linear_row[k] for k in ("band_rmse_pct", "surface_peak_error_pct",
                                              "surface_peak_error_pct_via_model", "loss_residual")},
        "factor_worse": linear_row["band_rmse_pct"] / chosen["band_rmse_pct"],
    }
    print(f"    the hard initial condition's ramp, at the same weight: "
          f"sqrt(t) N {chosen['band_rmse_pct']:.2f} % in band, "
          f"t N {linear_row['band_rmse_pct']:.2f} % ({ic_ablation['factor_worse']:.1f}x worse)")
    print("    a switched-on flux makes the surface leave T0 as sqrt(t); t N asserts it leaves "
          "linearly,\n    and no weights can recover the difference")
    print(f"\n    chosen: lambda_r x {chosen['lambda_r_scale']:.0e}, and it is frozen from here on\n")

    # ------------------------------------------------------ (2) the showcase, five seeds
    scenario = SHOWCASE
    truth = forward(scenario, LAW_A, scenario.q_true, grid=TRUTH_GRID)
    t_sensor, Y, Y_clean = sensor_signal(truth, scenario)
    model = InverseModel(scenario, LAW_A)
    q_ref = scenario.q_true(model.control_times())
    kw = dict(q_ref=q_ref, T_surface_true=truth.T_surface, T0=scenario.T0, dt_c=dt_c, f_c=f_c)

    print(f"(2) the showcase, sigma = {scenario.sigma_noise} K, "
          f"true surface peak {to_celsius(truth.T_surface.max()):.1f} degC")
    duhamel = model.step_response()
    classical = {
        "beck": beck_by_discrepancy(model, Y, sigma=scenario.sigma_noise, duhamel=duhamel),
        "tikhonov": tikhonov(model, Y, sigma=scenario.sigma_noise, duhamel=duhamel),
        "alifanov": alifanov(model, Y, sigma=scenario.sigma_noise),
    }
    print(f"\n    {'method':10s} {'band RMSE':>11} {'of peak':>9} {'T_surf peak':>13} {'seconds':>9}")
    baselines, surfaces, fluxes = [], {}, {}
    for name, out in classical.items():
        surface = model.solve(out["q"]).field[:, 0]
        row = {"method": name, **score(out["q"], surface, **kw)}
        baselines.append(row)
        surfaces[name] = surface
        fluxes[name] = out["q"]
        print(f"    {name:10s} {row['band_rmse_pct']:10.3f} % {row['band_rmse_pct_of_peak']:8.3f} % "
              f"{row['surface_peak_error_pct']:+12.4f} %")

    cfg = replace(base, lambda_r_scale=chosen["lambda_r_scale"])
    seeds, seed_q, seed_T = [], [], []
    print()
    for seed in SEEDS:
        row, r = run_pinn(scenario, Y, cfg, seed, model, cache=cache, **kw)
        seeds.append(row)
        seed_q.append(r.q)
        seed_T.append(np.interp(model.dt * np.arange(model.M + 1), r.t_surface, r.T_surface))
        via = row["surface_peak_error_pct_via_model"]
        print(f"    pinn s{seed}   {row['band_rmse_pct']:10.3f} % {row['band_rmse_pct_of_peak']:8.3f} % "
              f"{row['surface_peak_error_pct']:+12.4f} % {row['seconds']:8.0f}s   "
              f"(via model {f'{via:+.4f} %' if via is not None else 'forward solve diverged'}, "
              f"fit {row['sensor_fit_rms_K']:.3f} K)")

    peak = summarise(seeds, "surface_peak_error_pct")
    band = summarise(seeds, "band_rmse_pct")
    print(f"\n    across {len(SEEDS)} seeds: T_surf peak {peak['mean']:+.4f} % +/- {peak['std']:.4f} "
          f"(worst |.| {peak['worst_abs']:.4f} %), band {band['mean']:.3f} % +/- {band['std']:.3f}")

    # ----------------------------------------------------------------------- the M4 gate
    best = min(abs(r["surface_peak_error_pct"]) for r in baselines)
    worst = max(abs(r["surface_peak_error_pct"]) for r in baselines)
    ratio = abs(peak["median"]) / best
    gate = {
        "criterion": f"|median PINN peak error| <= {PEAK_FACTOR} x best baseline",
        "best_baseline_abs_pct": best,
        "worst_baseline_abs_pct": worst,
        "pinn_median_abs_pct": abs(peak["median"]),
        "ratio_to_best": ratio,
        "ratio_to_worst": abs(peak["median"]) / worst,
        "factor": PEAK_FACTOR,
        "pass": bool(ratio <= PEAK_FACTOR),
        "seeds_within_factor": int(sum(abs(r["surface_peak_error_pct"]) <= PEAK_FACTOR * best for r in seeds)),
    }
    print(f"\n    gate: |{peak['median']:+.4f} %| / {best:.4f} % = {ratio:.2f}x the best baseline "
          f"(tolerance {PEAK_FACTOR}x)  -- {'PASS' if gate['pass'] else 'measured, not passed'}")
    print(f"          against the worst baseline it is {gate['ratio_to_worst']:.2f}x")

    # ------------------------------------------------------------------- write it all down
    seed_q, seed_T = np.array(seed_q), np.array(seed_T)
    median_seed = int(np.argsort([abs(r["surface_peak_error_pct"]) for r in seeds])[len(seeds) // 2])
    t_model = model.dt * np.arange(model.M + 1)
    traces = {
        "t_q": model.control_times(),
        "q_true": q_ref,
        "q_band": lowpass(q_ref, dt_c, f_c),
        "q_est": {**fluxes, "pinn": seed_q[median_seed]},
        "out_of_band_rms": band_error(q_ref, q_ref, dt_c, f_c)["out_of_band_rms"],
        "t_surface": t_model,
        "T_surface_true": np.interp(t_model, truth.t, truth.T_surface),
        "T_surface_est": {**surfaces, "pinn": seed_T[median_seed]},
        "t_sensor": t_sensor,
        "T_sensor_true": Y_clean,
        "pinn_q_seeds": seed_q,
        "pinn_surface_seeds": seed_T,
    }
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    plots.act1(traces, scenario, args.fig, f_c=f_c)
    print(f"\nwrote {args.fig}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "verdict": {"m4_gate": gate, "ablation_lambda_r_zero": ablation,
                    "all_pass": bool(gate["pass"] and (ablation is None or ablation["pass"]))},
        "act": 1,
        "law": LAW_A.name,
        "scenario": {"name": scenario.name, "t_end": scenario.t_end, "x_s": scenario.x_s,
                     "sigma_noise": scenario.sigma_noise, "f_sample": scenario.f_sample,
                     "surface_peak_C": float(to_celsius(truth.T_surface.max())),
                     "sensor_peak_C": float(to_celsius(Y_clean.max()))},
        "tuning_scenario": {"name": TUNING.name, "noise_seed": TUNING.noise_seed,
                            "q_peak": TUNING.q_peak, "t_brake": TUNING.t_brake},
        "config": {k: v for k, v in cfg.__dict__.items()},
        "f_c": f_c,
        "weight_sweep": {"criterion": SWEEP_CRITERION, "scales": list(scales), "rows": sweep,
                         "chosen_scale": chosen["lambda_r_scale"]},
        "initial_condition_ablation": ic_ablation,
        "baselines": baselines,
        "pinn_seeds": seeds,
        "pinn_summary": {"surface_peak_error_pct": peak, "band_rmse_pct": band,
                         "surface_peak_error_pct_via_model": summarise(seeds, "surface_peak_error_pct_via_model"),
                         "sensor_fit_rms_K": summarise(seeds, "sensor_fit_rms_K"),
                         "seconds": summarise(seeds, "seconds")},
        "quick": args.quick,
        "device": args.device,
        "total_seconds": time.perf_counter() - t0,
    }, indent=2))
    print(f"wrote {args.out}   ({(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
