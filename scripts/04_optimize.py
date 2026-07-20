"""Design selection: can the surrogate rank candidate designs better than a cheap solve?

    python scripts/04_optimize.py --n-candidates 150

A design loop never needs the exact stress of every candidate -- it needs the *ranking*
to be right, so that the design it keeps is genuinely the best one.  So this script
does not report an accuracy; it reports a decision.

`--n-candidates` hole layouts are drawn at constant removed area (a lightweighting
study).  Each one is scored three ways:

  fine FEM     ground truth, ~1.7 s per design  (computed here only to grade the others)
  coarse FEM   the cheap-solver-in-the-loop strategy, ~0.15 s per design
  GNN          the surrogate on the same coarse mesh, ~0.16 s per design

and the three are compared on rank correlation and, more importantly, on the *regret*
of the design each one would have made you pick.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy.stats import spearmanr

from stressgnn import plots
from stressgnn.dataset import GraphDataset, build_graph, collate
from stressgnn.fem import interpolate, solve
from stressgnn.geometry import Design, is_valid, sample_design
from stressgnn.mesh import build_mesh
from stressgnn.train import load_model, pick_device

H_FINE = 0.010


def sample_at_fixed_area(rng: np.random.Generator, n_holes: int, target_area: float) -> Design:
    """Random hole layout whose total removed area is exactly `target_area`."""
    for _ in range(5000):
        d = sample_design(rng, n_holes_choices=(n_holes,))
        r = d.holes[:, 2]
        scale = np.sqrt(target_area / (np.pi * (r**2).sum()))
        holes = d.holes.copy()
        holes[:, 2] = r * scale
        if is_valid(holes):
            return Design(holes=holes, load_angle=d.load_angle)
    raise RuntimeError("could not sample at the requested area")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--model", default="gnn-mp10-glob")
    ap.add_argument("--n-candidates", type=int, default=150)
    ap.add_argument("--n-holes", type=int, default=3)
    ap.add_argument("--h-coarse", type=float, default=0.045)
    ap.add_argument("--load-angle", type=float, default=-np.pi / 3)
    ap.add_argument("--removed-area", type=float, default=0.16)  # 8% of the 2x1 plate
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("results/optimization.json"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figs"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, cfg = load_model(args.runs_dir / f"{args.model}.pt", device=str(device))
    base = GraphDataset(args.data_dir / "train.npz")
    mean, std = base.mean, base.std
    peak_mean, peak_std = base.peak_mean, base.peak_std

    rng = np.random.default_rng(args.seed)
    designs = []
    while len(designs) < args.n_candidates:
        d = sample_at_fixed_area(rng, args.n_holes, args.removed_area)
        designs.append(Design(holes=d.holes, load_angle=args.load_angle))

    fine_peak, coarse_peak, gnn_peak = [], [], []
    t_fine = t_mesh = t_solve = t_gnn = 0.0
    samples = []

    for i, design in enumerate(designs):
        # Meshing and solving are timed apart: the plain GNN needs the coarse *mesh*
        # but not the coarse *solve*, and charging it for one would understate it.
        t0 = time.perf_counter()
        cm = build_mesh(design, args.h_coarse)
        t_mesh += time.perf_counter() - t0
        t0 = time.perf_counter()
        cs = solve(cm, design)
        t_solve += time.perf_counter() - t0

        sample = {
            "nodes": cm.nodes.astype(np.float32),
            "tris": cm.tris.astype(np.int32),
            "node_type": cm.node_type.astype(np.int8),
            "local_size": cm.local_size().astype(np.float32),
            "vm_coarse": cs.vm_node.astype(np.float32),
            "u_coarse": cs.u.astype(np.float32),
            "vm_target": np.zeros(cm.n_nodes, dtype=np.float32),  # unknown at search time
            "holes": design.holes.astype(np.float32),
            "load_angle": np.float32(design.load_angle),
            "max_vm_fine": np.float32(0.0),
            "max_vm_coarse": np.float32(cs.vm_node.max()),
            "max_vm_target": np.float32(0.0),
            "h_coarse": np.float32(args.h_coarse),
        }
        t0 = time.perf_counter()
        g = build_graph(
            sample, cfg.use_coarse_solution, cfg.use_position, mean, std,
            with_grid=cfg.model == "grid-cnn", peak_stats=(peak_mean, peak_std),
        )
        with torch.no_grad():
            _, peak_out = model(collate([g]).to(device))
            pred_peak = float(peak_out.cpu().numpy()[0]) * peak_std + peak_mean
        t_gnn += time.perf_counter() - t0

        t0 = time.perf_counter()
        fm = build_mesh(design, H_FINE)
        fs = solve(fm, design)
        t_fine += time.perf_counter() - t0

        sample["vm_target"] = interpolate(fm, fs.vm_node, cm.nodes).astype(np.float32)
        sample["max_vm_fine"] = np.float32(fs.max_vm)
        samples.append(sample)

        fine_peak.append(fs.max_vm)
        coarse_peak.append(float(cs.vm_node.max()))
        gnn_peak.append(pred_peak)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.n_candidates} candidates scored", flush=True)

    fine_peak = np.array(fine_peak)
    coarse_peak = np.array(coarse_peak)
    gnn_peak = np.array(gnn_peak)

    def regret(scores: np.ndarray) -> dict:
        """What you lose by trusting `scores` instead of the truth."""
        pick = np.argsort(scores)[: args.top_k]
        chosen = int(pick[np.argmin(fine_peak[pick])])  # top-k shortlist, verified
        best = int(np.argmin(fine_peak))
        return {
            "chosen_index": chosen,
            "chosen_true_peak": float(fine_peak[chosen]),
            "regret_pct": 100 * float(fine_peak[chosen] - fine_peak[best]) / float(fine_peak[best]),
            "spearman": float(spearmanr(scores, fine_peak).statistic),
            "top1_index": int(np.argmin(scores)),
            "top1_true_peak": float(fine_peak[int(np.argmin(scores))]),
            "top1_regret_pct": 100
            * float(fine_peak[int(np.argmin(scores))] - fine_peak[best])
            / float(fine_peak[best]),
        }

    out = {
        "n_candidates": args.n_candidates,
        "removed_area_fraction": args.removed_area / 2.0,
        "load_angle": args.load_angle,
        "cost_per_design_s": {
            "fine_fem": t_fine / args.n_candidates,
            "coarse_fem": (t_mesh + t_solve) / args.n_candidates,
            "gnn_on_coarse_mesh": (t_mesh + t_gnn + (t_solve if cfg.use_coarse_solution else 0.0))
            / args.n_candidates,
            "coarse_meshing_only": t_mesh / args.n_candidates,
            "coarse_solve_only": t_solve / args.n_candidates,
            "gnn_graph_build_and_inference": t_gnn / args.n_candidates,
        },
        "peak_stress_range": [float(fine_peak.min()), float(fine_peak.max())],
        "strategies": {"coarse FEM": regret(coarse_peak), "GNN": regret(gnn_peak)},
    }
    out["speedup_vs_fine"] = out["cost_per_design_s"]["fine_fem"] / out["cost_per_design_s"]["gnn_on_coarse_mesh"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    best = int(np.argmin(fine_peak))
    worst = int(np.argmax(fine_peak))
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(gnn_peak)
    plots.optimisation_summary(
        {"best_so_far": np.minimum.accumulate(fine_peak[order]).tolist()},
        {"sample": samples[worst], "vm": samples[worst]["vm_target"], "peak": float(fine_peak[worst])},
        {"sample": samples[best], "vm": samples[best]["vm_target"], "peak": float(fine_peak[best])},
        args.fig_dir / "design_selection.png",
    )

    print("\n--- design selection over", args.n_candidates, "candidates ---")
    print(f"true peak stress spans {fine_peak.min():.2f} - {fine_peak.max():.2f}")
    for name, r in out["strategies"].items():
        print(
            f"{name:12s} Spearman {r['spearman']:+.3f} | top-1 regret {r['top1_regret_pct']:5.2f}% | "
            f"top-{args.top_k} shortlist regret {r['regret_pct']:5.2f}%"
        )
    c = out["cost_per_design_s"]
    print(
        f"cost/design: fine FEM {c['fine_fem']:.2f}s | coarse FEM {c['coarse_fem']:.2f}s | "
        f"GNN {c['gnn_on_coarse_mesh']:.2f}s "
        f"(meshing {c['coarse_meshing_only'] * 1000:.0f} ms + network {c['gnn_graph_build_and_inference'] * 1000:.0f} ms)"
    )
    print(f"speed-up of the GNN pipeline over fine FEM: {out['speedup_vs_fine']:.1f}x")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
