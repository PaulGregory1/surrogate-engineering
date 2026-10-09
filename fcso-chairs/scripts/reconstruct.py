"""Milestone 2: encode (posterior mean) then decode chairs, extract meshes, render them next to their
surface clouds, and report Chamfer / F-score against the full surface cloud (same metric as eval.py).

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/reconstruct.py --n 3          # debug set of opt
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/reconstruct.py --ids a b c
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from scipy.spatial import cKDTree

from fcso import mesh_render as R
from fcso import prior as P


def chamfer_fscore(mesh, surface, n=100000, th=0.02, seed=0):
    """eval.py metric: mesh sampled with n points vs the full surface cloud; Chamfer = sum of the two mean
    (unsquared) nearest-neighbour distances; F-score at threshold th."""
    pred = np.asarray(mesh.sample(n, seed=seed))
    d1, _ = cKDTree(pred).query(surface)   # gt -> pred
    d2, _ = cKDTree(surface).query(pred)   # pred -> gt
    recall, precision = float((d2 < th).mean()), float((d1 < th).mean())
    f = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    return float(d1.mean() + d2.mean()), f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--split-file", default=os.path.join(P.ROOT, "splits", "opt.txt"))
    ap.add_argument("--n", type=int, default=3, help="first n ids of the split file (ignored with --ids)")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0, help="seed of the 2048-point surface subsample")
    ap.add_argument("--out", default=os.path.join(P.ROOT, "runs", "m2"))
    a = ap.parse_args()

    ids = a.ids or P.read_ids(a.split_file)[: a.n]
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    pr = P.Prior(load_dm=False)
    print(f"device {pr.device}, {len(ids)} chairs, grid {a.res}^3, subsample seed {a.seed}", flush=True)

    objs, labels, rows = [], [], []
    for sid in ids:
        surface = P.load_surface(sid)
        t0 = time.time()
        mean, logvar = pr.encode_posterior(P.sample_surface(surface, seed=a.seed))
        vol = pr.decode_grid(mean, res=a.res)[0]
        t_dec = time.time() - t0
        mesh = R.logits_to_mesh(vol)
        cd, f = chamfer_fscore(mesh, surface) if mesh is not None else (float("nan"), 0.0)
        comps = mesh.split(only_watertight=False) if mesh is not None else []
        row = dict(id=sid, chamfer=cd, fscore_0p02=f, components=len(comps),
                   mesh_volume=float(mesh.volume) if mesh is not None else 0.0,
                   grid_inside_frac=float((vol > 0).mean()), encode_decode_s=t_dec,
                   z_mean_std=float(mean.std()), posterior_std_mean=float((0.5 * logvar).exp().mean()))
        rows.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"z_mean": mean.cpu(), "logvar": logvar.cpu(), "seed": a.seed}, os.path.join(a.out, f"{sid}_z.pt"))
        R.render_grid([surface, mesh], os.path.join(a.out, f"{sid}.png"),
                      labels=[f"{sid[:8]} surface", f"recon CD {cd:.4f} F {f:.3f}"], ncols=2)
        objs += [surface, mesh]
        labels += [f"{sid[:8]} surface", f"recon CD {cd:.4f} F {f:.3f}"]

    R.render_grid(objs, os.path.join(a.out, "reconstruction.png"), labels=labels, ncols=2)
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print("saved", os.path.join(a.out, "reconstruction.png"), flush=True)


if __name__ == "__main__":
    main()
