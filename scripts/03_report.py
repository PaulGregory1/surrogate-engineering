"""Evaluate every trained model and write RESULTS.md + figures.

    python scripts/03_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from stressgnn import plots
from stressgnn.dataset import GraphDataset
from stressgnn.evaluate import baseline_metrics, field_metrics, latency_ms, predict
from stressgnn.train import load_model, pick_device

PRETTY = {
    "gnn-mp0": "MLP per node (no message passing)",
    "gnn-mp2": "GNN, 2 message-passing steps",
    "gnn-mp5": "GNN, 5 message-passing steps",
    "gnn-mp10": "GNN, 10 message-passing steps",
    "gnn-mp15": "GNN, 15 message-passing steps",
    "gnn-mp10-glob": "GNN, 10 steps + global latent",
    "gnn-mp10-glob-corr": "GNN + global, fed the coarse solve",
    "gnn-mp10-glob-maxloss": "GNN + global, peak-stress loss",
    "gnn-mp10-glob-nopos": "GNN + global, no node coordinates",
    "grid-cnn": "U-Net on a Cartesian grid",
}


def md_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for key, _, fmt in cols:
            v = r.get(key)
            cells.append("-" if v is None else (v if isinstance(v, str) else fmt.format(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figs"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    base = GraphDataset(args.data_dir / "train.npz")
    stats = (base.mean, base.std)
    splits = ["test", "res", "ood"]
    available = [s for s in splits if (args.data_dir / f"{s}.npz").exists()]

    # One dataset per (feature config, split); reused across models.
    cache: dict[tuple, GraphDataset] = {}

    def get(split: str, coarse: bool, pos: bool, grid: bool) -> GraphDataset:
        key = (split, coarse, pos, grid)
        if key not in cache:
            cache[key] = GraphDataset(
                args.data_dir / f"{split}.npz",
                use_coarse_solution=coarse,
                use_position=pos,
                with_grid=grid,
                stats=stats,
            )
        return cache[key]

    solver_meta = json.loads((args.data_dir / "train_meta.json").read_text())

    rows: list[dict] = []
    preds_for_fig: dict[str, np.ndarray] = {}
    fig_index = 0

    for ckpt in sorted(args.runs_dir.glob("*.pt")):
        tag = ckpt.stem
        model, cfg = load_model(ckpt, device=str(device))
        hist = json.loads((args.runs_dir / f"{tag}_history.json").read_text())
        row = {
            "model": PRETTY.get(tag, tag),
            "tag": tag,
            "n_parameters": hist["n_parameters"],
            "train_min": hist["train_seconds"] / 60,
        }
        for split in available:
            ds = get(split, cfg.use_coarse_solution, cfg.use_position, cfg.model == "grid-cnn")
            m = field_metrics(predict(model, ds, device), ds)
            row.update({f"{split}_{k}": v for k, v in m.items()})
            if split == "test" and tag in ("gnn-mp10-glob", "grid-cnn", "gnn-mp0"):
                preds_for_fig[PRETTY.get(tag, tag)] = predict(model, ds, device)[fig_index]
        row["latency_ms"] = latency_ms(model, get("test", cfg.use_coarse_solution, cfg.use_position, cfg.model == "grid-cnn"), device)
        rows.append(row)
        print(f"  {tag:24s} test rel L2 {row['test_rel_l2']:.4f}  peak err {row['test_peak_err_pct']:.2f}%", flush=True)

    # Non-learned references.
    ref_rows = []
    for name, fn_key in [("coarse FEM (no ML)", "coarse FEM (no ML)"), ("interpolation floor", "interpolation floor")]:
        row = {"model": name, "tag": name, "n_parameters": None, "train_min": None, "latency_ms": None}
        for split in available:
            ds = get(split, False, True, False)
            m = baseline_metrics(ds)[fn_key]
            row.update({f"{split}_{k}": v for k, v in m.items()})
        ref_rows.append(row)
        if fn_key == "coarse FEM (no ML)":
            preds_for_fig["coarse FEM (no ML)"] = get("test", False, True, False).samples[fig_index]["vm_coarse"]

    rows.sort(key=lambda r: r["test_peak_err_pct"])
    all_rows = ref_rows + rows
    (args.out_dir / "metrics.json").write_text(json.dumps(all_rows, indent=2))

    # ------------------------------------------------------------------- figures
    ds_test = get("test", False, True, False)
    order = ["coarse FEM (no ML)"] + [k for k in preds_for_fig if k != "coarse FEM (no ML)"]
    plots.stress_comparison(
        ds_test.samples[fig_index],
        {k: preds_for_fig[k] for k in order if k in preds_for_fig},
        args.fig_dir / "stress_comparison.png",
        title=f"{ds_test.samples[fig_index]['nodes'].shape[0]} coarse nodes\n"
        f"vs {int(ds_test.samples[fig_index]['n_nodes_fine'])} fine nodes",
    )

    mp_points = {"GNN": [], "coarse FEM (no ML)": [], "interpolation floor": []}
    for r in rows:
        if r["tag"].startswith("gnn-mp") and r["tag"].count("-") == 1:
            mp_points["GNN"].append((int(r["tag"].split("mp")[1]), r["test_peak_err_pct"]))
    for r in ref_rows:
        mp_points[r["model"]] = [(0, r["test_peak_err_pct"]), (15, r["test_peak_err_pct"])]
    if mp_points["GNN"]:
        plots.ablation_curve(
            mp_points,
            args.fig_dir / "message_passing_ablation.png",
            xlabel="message-passing steps",
            ylabel="peak von Mises error (%)",
        )

    groups = {}
    for split in available:
        label = {"test": "test\n(h ~ 0.035-0.06)", "res": "unseen resolution\n(h = 0.025)", "ood": "OOD\n(5-6 holes, new load)"}[split]
        groups[label] = {r["model"]: r.get(f"{split}_peak_err_pct") for r in all_rows if r["tag"] in
                         ("coarse FEM (no ML)", "gnn-mp10-glob", "grid-cnn", "gnn-mp10-glob-corr")}
    plots.bar_comparison(groups, args.fig_dir / "generalisation.png", "peak von Mises error (%)")

    # ------------------------------------------------------------------ RESULTS.md
    acc_cols = [
        ("model", "model", "{}"),
        ("n_parameters", "params", "{:,.0f}"),
        ("test_rel_l2", "field rel L2", "{:.4f}"),
        ("test_peak_err_pct", "peak err %", "{:.2f}"),
        ("test_peak_err_p95_pct", "peak err p95 %", "{:.2f}"),
        ("test_peak_bias_pct", "peak bias %", "{:+.2f}"),
        ("latency_ms", "inference (ms)", "{:.1f}"),
    ]
    gen_cols = [("model", "model", "{}")] + [
        (f"{s}_peak_err_pct", {"test": "test", "res": "unseen h", "ood": "OOD"}[s] + " peak err %", "{:.2f}")
        for s in available
    ]

    md = [
        "# Results",
        "",
        f"Generated by `scripts/03_report.py`.  Every model predicts the **fine-mesh** von Mises "
        f"field on the **coarse** mesh nodes.  Reference solve: h = 0.010 "
        f"(~{int(ds_test.samples[0]['n_nodes_fine'])} nodes); surrogate input: "
        f"~{ds_test.samples[0]['nodes'].shape[0]} nodes.",
        "",
        f"Dataset cost: **{solver_meta['seconds_per_sample']:.2f} s/sample** (fine + coarse solve + interpolation).",
        "",
        "## Accuracy on the test split",
        "",
        md_table(all_rows, acc_cols),
        "",
        "`peak err` is the error on the maximum von Mises stress of the part -- the number a "
        "structural engineer actually signs off on.  `interpolation floor` is what a *perfect* "
        "model would still get wrong, because a coarse node set cannot sample the peak of a fine "
        "stress field.",
        "",
        "## Generalisation",
        "",
        md_table(all_rows, gen_cols),
        "",
    ]
    Path("RESULTS.md").write_text("\n".join(md))
    print("wrote RESULTS.md")


if __name__ == "__main__":
    main()
