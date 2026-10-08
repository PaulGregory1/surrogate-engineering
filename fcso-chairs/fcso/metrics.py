"""Shape-quality metrics for the evaluation (SPEC "Evaluation"): connected components and Chamfer distance.

Meshes are the evaluation meshes: marching cubes at 128^3 on the logits, level 0, every component kept
(`mesh_render.logits_to_mesh`). Timing is recorded by the optimizer itself (`fcso/optimize.py`).

    mesh_stats(m)            -> components, stray_components, stray_share, largest_share, mesh_volume
    chamfer(a, b, n=10000)   -> mean NN distance a->b + mean NN distance b->a (unsquared, eval.py convention)
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree


def component_volumes(m):
    """Absolute volumes of the connected components of m, largest first (empty list if m is None)."""
    if m is None:
        return []
    comps = m.split(only_watertight=False)
    return sorted((abs(float(c.volume)) for c in comps), reverse=True)


def mesh_stats(m):
    """Stray components = every connected component other than the largest one; stray_share = their share
    of the total component volume (0 for a single-piece shape)."""
    vols = component_volumes(m)
    if not vols:
        return dict(components=0, stray_components=0, stray_share=0.0, largest_share=0.0, mesh_volume=0.0)
    tot = max(sum(vols), 1e-12)
    return dict(components=len(vols), stray_components=len(vols) - 1, stray_share=1.0 - vols[0] / tot,
                largest_share=vols[0] / tot, mesh_volume=float(abs(m.volume)))


def sample_points(m, n=10000, seed=0):
    return np.asarray(m.sample(n, seed=seed), dtype=np.float64)


def chamfer(a, b, n=10000, seed=0):
    """Chamfer distance between two meshes (or (N, 3) point arrays): n surface samples each (seeded), sum of
    the two mean unsquared nearest-neighbour distances, the convention of the repo's eval.py (milestone 2).
    NaN if either shape is empty."""
    if a is None or b is None:
        return float("nan")
    pa = a if isinstance(a, np.ndarray) else sample_points(a, n, seed)
    pb = b if isinstance(b, np.ndarray) else sample_points(b, n, seed + 1)
    d_ab, _ = cKDTree(pb).query(pa)
    d_ba, _ = cKDTree(pa).query(pb)
    return float(d_ab.mean() + d_ba.mean())


def evaluate_run(prior, run_dir, res=128, n_points=10000, render_path=None, log=print):
    """Evaluation meshes and metrics of every snapshot of one run (run_dir/ckpt.pt): initial shape, each
    threshold reached (10 ... 70 %), and the final latent if the run stopped on the budget. Writes
    run_dir/eval.json and, if render_path is given, a one-row PNG (fixed camera, stray components in red,
    'not reached' panels left empty). Returns the list of rows."""
    import torch

    from fcso import mesh_render as R
    from fcso.optimize import snapshot_items

    state = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
    items = snapshot_items(state)
    if state["done"] and state["stop_reason"] != "target":
        items.append(("final", dict(threshold=None, step=state["step"], cycle=state["cycle"],
                                    time_s=state["elapsed"], ratio=state["V"] / state["V0"],
                                    r=1 - state["V"] / state["V0"], V=state["V"], z=state["z"])))
    rows, objs, labels, m0 = [], [], [], None
    arrays = {}  # evaluation meshes, saved to run_dir/meshes.npz so the report needs no decoding
    for label, s in items:
        if s is None:
            rows.append(dict(label=label, reached=False))
            objs.append(None)
            labels.append(f"{label}%: not reached")
            continue
        m = R.logits_to_mesh(prior.decode_grid(s["z"].to(prior.device), res)[0])
        if m is not None:
            arrays[f"{label}_v"] = np.asarray(m.vertices, dtype=np.float32)
            arrays[f"{label}_f"] = np.asarray(m.faces, dtype=np.int32)
        if label == "init":
            m0 = m
        st = mesh_stats(m)
        row = dict(label=label, reached=True, step=s["step"], time_s=s["time_s"], ratio=s["ratio"], r=s["r"],
                   **st, mesh_ratio=st["mesh_volume"] / max(mesh_stats(m0)["mesh_volume"], 1e-12),
                   chamfer_to_initial=0.0 if label == "init" else chamfer(m, m0, n_points))
        rows.append(row)
        objs.append(m)
        head = "init" if label == "init" else (f"{label}%" if label != "final" else "final")
        labels.append(f"{head}  V/V0 {s['ratio']:.2f}\nstep {s['step']}  {s['time_s']:.0f} s\n"
                      f"comps {st['components']}  stray {100 * st['stray_share']:.1f}%")
        log(f"[{state['id'][:8]}] eval {label}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                                for k, v in row.items()}))
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(dict(id=state["id"], name=state["cfg"]["name"], rows=rows), f, indent=1)
    tmp = os.path.join(run_dir, "meshes.tmp.npz")
    np.savez_compressed(tmp, res=np.int32(res), **arrays)
    os.replace(tmp, os.path.join(run_dir, "meshes.npz"))
    if render_path:
        R.render_grid(objs, render_path, labels=labels, ncols=len(objs), panel=(320, 360), color_strays=True)
    return rows
