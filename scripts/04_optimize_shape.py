"""Minimise drag at constant area by descending through the surrogate.

    python scripts/04_optimize_shape.py --model unet --al-rounds 1

Reports three numbers that matter more than any loss curve:
  * what the surrogate *thinks* the optimised drag is,
  * what the LBM says when the design is actually solved,
  * how big the gap is -- and whether one round of active learning closes it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flowsurrogate import plots
from flowsurrogate.data import load_splits
from flowsurrogate.geometry import Domain, area, equivalent_diameter, solid_mask
from flowsurrogate.lbm import SolverConfig
from flowsurrogate.optimize import OptConfig, boundary_xy, optimise, to_params, verify
from flowsurrogate.train import TrainConfig, load_model, pick_device, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--model", default="unet")
    ap.add_argument("--out", type=Path, default=Path("results/optimization.json"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figs"))
    ap.add_argument("--n-starts", type=int, default=16)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--reynolds", type=float, default=80.0)
    ap.add_argument("--al-rounds", type=int, default=1, help="active-learning rounds")
    ap.add_argument("--al-epochs", type=int, default=25)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    domain = Domain()
    device = pick_device(args.device)
    solver_cfg = SolverConfig(n_steps=args.steps, n_avg=args.steps // 4)
    train_set, val_set, _, norm = load_splits(args.data_dir, domain)
    model, _ = load_model(args.runs_dir / f"{args.model}.pt", device=str(device))

    target_area = float(np.median([area(p) for p in train_set.params]))
    cfg = OptConfig(n_starts=args.n_starts, iters=args.iters, reynolds=args.reynolds, target_area=target_area)
    print(f"target area {target_area:.1f} cells^2 (D_eq = {2 * np.sqrt(target_area / np.pi):.2f}), Re = {args.reynolds}")

    log: dict = {"target_area": target_area, "reynolds": args.reynolds, "rounds": []}
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for rnd in range(args.al_rounds + 1):
        t0 = time.perf_counter()
        res = optimise(model, norm, domain, cfg, device)
        t_opt = time.perf_counter() - t0

        # Solver check on the starting designs and on the optimised ones.
        cd_init_lbm, out_init, params_init = verify(
            res["init_coeffs"], target_area, args.reynolds, domain, solver_cfg, str(device)
        )
        t1 = time.perf_counter()
        cd_opt_lbm, out_opt, params_opt = verify(
            res["final_coeffs"], target_area, args.reynolds, domain, solver_cfg, str(device)
        )
        t_verify = time.perf_counter() - t1

        best = int(np.argmin(cd_opt_lbm))
        worst_init = float(cd_init_lbm.max())
        entry = {
            "round": rnd,
            "optimise_seconds": t_opt,
            "verify_seconds": t_verify,
            "surrogate_evals": cfg.iters * cfg.n_starts,
            "cd_init_lbm_best": float(cd_init_lbm.min()),
            "cd_init_lbm_mean": float(cd_init_lbm.mean()),
            "cd_init_lbm_worst": worst_init,
            "cd_opt_surrogate": float(res["cd_surrogate_final"].min()),
            "cd_opt_lbm_best": float(cd_opt_lbm.min()),
            "surrogate_gap_pct": 100.0
            * float(res["cd_surrogate_final"][best] - cd_opt_lbm[best])
            / float(cd_opt_lbm[best]),
            "drag_reduction_pct": 100.0 * (float(cd_init_lbm.min()) - float(cd_opt_lbm.min())) / float(cd_init_lbm.min()),
            "best_params": params_opt[best].tolist(),
        }
        log["rounds"].append(entry)
        print(
            f"[round {rnd}] optimiser {t_opt:.1f}s for {entry['surrogate_evals']} evals | "
            f"Cd best start (LBM) {entry['cd_init_lbm_best']:.3f} -> optimised (LBM) "
            f"{entry['cd_opt_lbm_best']:.3f}  ({entry['drag_reduction_pct']:+.1f}%) | "
            f"surrogate said {entry['cd_opt_surrogate']:.3f} (gap {entry['surrogate_gap_pct']:+.1f}%)",
            flush=True,
        )

        if rnd == 0:
            i0 = int(np.argmin(cd_init_lbm))
            plots.optimisation_summary(
                {
                    "cd_surrogate": res["history"]["cd_best"],
                    "verified": [(cfg.iters - 1, float(cd_opt_lbm.min()))],
                    "cd_initial": float(cd_init_lbm[i0]),
                    "cd_optimised": float(cd_opt_lbm[best]),
                },
                {"initial": out_init["u"][i0], "optimised": out_opt["u"][best]},
                {
                    "initial": (~solid_mask(params_init[i0], domain)).astype(float),
                    "optimised": (~solid_mask(params_opt[best], domain)).astype(float),
                },
                args.fig_dir / "optimisation.png",
            )
            plots.shape_gallery(
                [boundary_xy(params_init[i0]), boundary_xy(params_opt[best])],
                [f"start ($C_d$={cd_init_lbm[i0]:.3f})", f"optimised ($C_d$={cd_opt_lbm[best]:.3f})"],
                args.fig_dir / "optimised_shape.png",
            )

        if rnd < args.al_rounds:
            # Fold the solver-labelled designs back into the training set and
            # fine-tune: the optimiser has just told us exactly where the surrogate
            # is being trusted, which is where it should be accurate.
            all_params = np.concatenate([params_init, params_opt])
            extra = {
                "params": all_params.astype(np.float32),
                "re": np.full(len(all_params), args.reynolds, dtype=np.float32),
                "d_eq": np.array([equivalent_diameter(p) for p in all_params], dtype=np.float32),
                "cd": np.concatenate([cd_init_lbm, cd_opt_lbm]).astype(np.float32),
                "cd_drift": np.concatenate([out_init["cd_drift"], out_opt["cd_drift"]]).astype(np.float32),
                "u": np.concatenate([out_init["u"], out_opt["u"]]).astype(np.float32),
                "v": np.concatenate([out_init["v"], out_opt["v"]]).astype(np.float32),
                "p": np.concatenate([out_init["p"], out_opt["p"]]).astype(np.float32),
            }
            aug = train_set.augmented(extra, domain)
            print(f"  active learning: {len(train_set)} -> {len(aug)} training samples", flush=True)
            # A distinct tag, so the fine-tuned model never clobbers the checkpoint
            # the rest of the report is built from.
            ft = TrainConfig(
                model=args.model,
                epochs=args.al_epochs,
                lr=3e-4,
                model_kwargs={"base": 32, "depth": 3} if args.model == "unet" else {},
                name=f"{args.model}-al{rnd}",
            )
            train(
                ft,
                aug,
                val_set,
                args.runs_dir,
                device=str(device),
                verbose=False,
                init_state_dict=model.state_dict(),
            )
            model, _ = load_model(args.runs_dir / f"{ft.tag}.pt", device=str(device))
            train_set = aug

    args.out.write_text(json.dumps(log, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
