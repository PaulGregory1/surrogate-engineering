"""Evaluate every trained surrogate and write RESULTS.md + figures.

    python scripts/03_report.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from flowsurrogate import plots
from flowsurrogate.data import FlowDataset, load_splits
from flowsurrogate.evaluate import evaluate_pod, evaluate_torch, latency_ms, predict_torch
from flowsurrogate.geometry import Domain
from flowsurrogate.lbm import SolverConfig
from flowsurrogate.train import load_model, pick_device

PRETTY = {
    "pod-linear": "POD + ridge",
    "pod-rbf": "POD + kernel ridge",
    "unet": "U-Net",
    "unet-phys": "U-Net + continuity loss",
    "fno": "FNO",
}


def discover(runs_dir: Path) -> list[str]:
    """Main-table models: excludes the data-efficiency sweep (`-n<N>`) and the
    active-learning fine-tunes (`-al<k>`), which are reported separately."""
    tags = [p.stem for p in sorted(runs_dir.glob("*.pt"))]
    tags += [p.stem for p in sorted(runs_dir.glob("pod-*.pkl"))]
    return [t for t in tags if "-n" not in t and "-al" not in t]


def sweep_tags(runs_dir: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for p in list(runs_dir.glob("*-n*.pt")) + list(runs_dir.glob("*-n*.pkl")):
        base, n = p.stem.rsplit("-n", 1)
        out.setdefault(base, []).append(int(n))
    return {k: sorted(v) for k, v in out.items()}


def load_any(runs_dir: Path, tag: str, device):
    if tag.startswith("pod"):
        with open(runs_dir / f"{tag}.pkl", "rb") as fh:
            return pickle.load(fh), "pod"
    model, _ = load_model(runs_dir / f"{tag}.pt", device=str(device))
    return model, "torch"


def eval_tag(obj, kind, dataset, norm, device, u_in) -> dict:
    if kind == "pod":
        return evaluate_pod(obj, dataset, norm, u_in)
    return evaluate_torch(obj, dataset, norm, device, u_in)


def md_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for key, _, fmt in cols:
            v = r.get(key)
            cells.append("-" if v is None else (fmt.format(v) if not isinstance(v, str) else v))
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

    domain = Domain()
    u_in = SolverConfig().u_in
    device = pick_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set, test_set, norm = load_splits(args.data_dir, domain)
    ood_path = args.data_dir / "ood.npz"
    ood_set = FlowDataset(ood_path, domain, normalizer=norm) if ood_path.exists() else None

    solver_meta = json.loads((args.data_dir / "train_meta.json").read_text())
    solver_s = solver_meta["seconds_per_sample"]

    rows, cd_pairs, preds_for_fig = [], {}, {}
    for tag in discover(args.runs_dir):
        obj, kind = load_any(args.runs_dir, tag, device)
        hist = json.loads((args.runs_dir / f"{tag}_history.json").read_text())
        m_test = eval_tag(obj, kind, test_set, norm, device, u_in)
        m_ood = eval_tag(obj, kind, ood_set, norm, device, u_in) if ood_set else {}

        if kind == "torch":
            lat = latency_ms(obj, test_set, device, batch=1)
            lat32 = latency_ms(obj, test_set, device, batch=32)
            fields, cd = predict_torch(obj, test_set, device)
            preds_for_fig[PRETTY.get(tag, tag)] = norm.decode_fields(fields.cpu()) * test_set.m
            cd_pairs[PRETTY.get(tag, tag)] = (norm.decode_cd(torch.as_tensor(cd)).numpy(), test_set.cd_raw)
        else:
            t0 = time.perf_counter()
            for _ in range(20):
                obj.predict(test_set.params[:1], test_set.re[:1], tuple(test_set.y.shape[1:]))
            lat = (time.perf_counter() - t0) / 20 * 1e3
            t0 = time.perf_counter()
            obj.predict(test_set.params, test_set.re, tuple(test_set.y.shape[1:]))
            lat32 = (time.perf_counter() - t0) / len(test_set) * 1e3
            f, cd = obj.predict(test_set.params, test_set.re, tuple(test_set.y.shape[1:]))
            preds_for_fig[PRETTY.get(tag, tag)] = norm.decode_fields(torch.from_numpy(f.astype(np.float32))) * test_set.m
            cd_pairs[PRETTY.get(tag, tag)] = (norm.decode_cd(torch.as_tensor(cd)).numpy(), test_set.cd_raw)

        rows.append(
            {
                "model": PRETTY.get(tag, tag),
                "tag": tag,
                "n_parameters": hist.get("n_parameters"),
                "train_min": hist.get("train_seconds", 0) / 60,
                "latency_ms": lat,
                "latency_ms_b32": lat32,
                "speedup": solver_s * 1e3 / lat32,
                **{f"test_{k}": v for k, v in m_test.items()},
                **{f"ood_{k}": v for k, v in m_ood.items()},
            }
        )
        print(f"  {tag:12s} relL2 {m_test['rel_l2_mean']:.4f}  Cd MAPE {m_test['cd_mape_pct']:.2f}%", flush=True)

    rows.sort(key=lambda r: r["test_rel_l2_mean"])
    (args.out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))

    # ------------------------------------------------------------------- figures
    plots.cd_scatter(cd_pairs, args.fig_dir / "cd_parity.png")
    i = int(np.argmax(test_set.cd_raw))  # the highest-drag test case
    plots.field_comparison(
        (torch.from_numpy(test_set.fields_raw[i : i + 1]) * test_set.m[i : i + 1])[0].numpy(),
        {k: v[i].numpy() for k, v in preds_for_fig.items()},
        test_set.m[i, 0].numpy(),
        args.fig_dir / "fields_comparison.png",
        title=f"Re = {test_set.re[i]:.0f}, $C_d$ = {test_set.cd_raw[i]:.2f}",
    )

    curves: dict[str, list[tuple[int, float]]] = {}
    for base, sizes in sweep_tags(args.runs_dir).items():
        for n in sizes:
            tag = f"{base}-n{n}"
            obj, kind = load_any(args.runs_dir, tag, device)
            m = eval_tag(obj, kind, test_set, norm, device, u_in)
            curves.setdefault(PRETTY.get(base, base), []).append((n, m["rel_l2_mean"]))
    # Only sweep runs go on the curve: they all share one training budget, so the
    # comparison is about data volume and nothing else.
    if curves:
        plots.data_efficiency(curves, args.fig_dir / "data_efficiency.png")
        (args.out_dir / "data_efficiency.json").write_text(json.dumps(curves, indent=2))

    # ------------------------------------------------------------------ RESULTS.md
    acc_cols = [
        ("model", "model", "{}"),
        ("n_parameters", "params", "{:,.0f}"),
        ("test_rel_l2_u", "rel L2 u", "{:.4f}"),
        ("test_rel_l2_v", "rel L2 v", "{:.4f}"),
        ("test_rel_l2_p", "rel L2 p", "{:.4f}"),
        ("test_cd_mape_pct", "Cd MAPE %", "{:.2f}"),
        ("test_cd_max_err_pct", "Cd worst %", "{:.1f}"),
        ("test_cd_r2", "Cd R2", "{:.4f}"),
        ("test_div_residual", "div residual", "{:.3f}"),
    ]
    cost_cols = [
        ("model", "model", "{}"),
        ("train_min", "train (min)", "{:.1f}"),
        ("latency_ms", "infer b=1 (ms)", "{:.2f}"),
        ("latency_ms_b32", "infer b=32 (ms/sample)", "{:.2f}"),
        ("speedup", "speed-up vs LBM", "{:,.0f}x"),
    ]
    ood_cols = [
        ("model", "model", "{}"),
        ("test_rel_l2_mean", "rel L2 (in-dist)", "{:.4f}"),
        ("ood_rel_l2_mean", "rel L2 (OOD)", "{:.4f}"),
        ("test_cd_mape_pct", "Cd MAPE % (in-dist)", "{:.2f}"),
        ("ood_cd_mape_pct", "Cd MAPE % (OOD)", "{:.2f}"),
    ]

    ref_div = rows[0].get("test_div_residual_ref", None)
    md = [
        "# Results",
        "",
        f"Generated by `scripts/03_report.py`.  Test split: {len(test_set)} unseen geometries; "
        f"OOD split: {len(ood_set) if ood_set else 0} geometries with deformations and Reynolds "
        "numbers outside the training range.",
        "",
        "## Accuracy",
        "",
        md_table(rows, acc_cols),
        "",
        f"`div residual` is the dimensionless continuity error mean|div u| L/U; the LBM reference "
        f"fields themselves sit at {rows[0]['test_div_residual_ref']:.3f}.",
        "",
        "## Cost",
        "",
        f"The solver costs **{solver_s:.2f} s/sample** (batched, `torch.compile`, MPS).",
        "",
        md_table(rows, cost_cols),
        "",
        "## Generalisation",
        "",
        md_table(rows, ood_cols),
        "",
    ]
    (Path("RESULTS.md")).write_text("\n".join(md))
    print("wrote RESULTS.md")


if __name__ == "__main__":
    main()
