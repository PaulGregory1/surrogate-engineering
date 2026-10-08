"""Milestone 3 diagnostics: why does the unguided correction change the volume of some chairs?

For each chair, volume (64^3 occupancy fraction x 8, no grad) of:
  - the posterior mean z_s, and posterior samples z_s + std * eps (what the prior was trained on);
  - corrections at several sigma_start, Euler vs Heun, N = 10 vs 50 steps;
  - a single denoiser call D(z_s + sigma * eps; sigma) (Tweedie mean, no ODE);
  - corrections with other class labels (sanity check of the chair label).

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/prior_diag.py [--ids ...]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from fcso import correct as C
from fcso import prior as P

BOX = (2 * P.BOUND) ** 3


def V(pr, z):
    return [BOX * float((v > 0).mean()) for v in pr.decode_grid(z, res=64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(P.ROOT, "runs", "m3", "diag.json"))
    a = ap.parse_args()
    ids = a.ids or P.read_ids(os.path.join(P.ROOT, "splits", "opt.txt"))[: a.n]
    pr = P.Prior()
    seeds = list(range(a.seeds))
    out = []
    for sid in ids:
        zp = os.path.join(P.ROOT, "runs", "m2", f"{sid}_z.pt")
        if os.path.exists(zp):
            d = torch.load(zp)
            mean, logvar = d["z_mean"].to(pr.device), d["logvar"].to(pr.device)
        else:
            mean, logvar = pr.encode_posterior(P.sample_surface(P.load_surface(sid)))
        std = (0.5 * logvar).exp()
        v0 = V(pr, mean)[0]
        row = {"id": sid, "V0": v0, "post_std_mean": float(std.mean()), "post_std_rms": float(std.pow(2).mean().sqrt())}

        def rel(vs):
            return [round((v - v0) / v0, 4) for v in vs]

        g = torch.Generator().manual_seed(123)
        samples = torch.cat([mean + std * torch.randn(mean.shape, generator=g).to(pr.device) for _ in seeds])
        row["posterior_sample"] = rel(V(pr, samples))

        for s_start in (0.05, 0.1, 0.25, 0.5):
            zs = torch.cat([C.correct(pr, mean, s_start, 10, seed=s) for s in seeds])
            row[f"euler10_s{s_start}"] = rel(V(pr, zs))
        zs = torch.cat([C.correct(pr, mean, 0.25, 10, seed=s, solver="heun") for s in seeds])
        row["heun10_s0.25"] = rel(V(pr, zs))
        zs = torch.cat([C.correct(pr, mean, 0.25, 50, seed=s) for s in seeds])
        row["euler50_s0.25"] = rel(V(pr, zs))
        zs = torch.cat([C.correct(pr, samples[i:i + 1], 0.25, 10, seed=s) for i, s in enumerate(seeds)])
        row["euler10_s0.25_from_posterior_sample"] = rel(V(pr, zs))

        with torch.no_grad():
            for s in (0.05, 0.25):
                g = torch.Generator().manual_seed(7)
                x = torch.cat([mean + s * torch.randn(mean.shape, generator=g).to(pr.device) for _ in seeds])
                row[f"tweedie_D_s{s}"] = rel(V(pr, pr.denoise(x, s)))
        # Label sanity: same correction with the chair label (18) vs a few other ShapeNet class ids.
        for lab in (0, 18, 47, 49):  # airplane, chair, sofa, table
            zs = torch.cat([C.correct(pr, mean, 0.25, 10, seed=s, class_label=lab) for s in seeds])
            row[f"label{lab}_euler10_s0.25"] = rel(V(pr, zs))
        out.append(row)
        print(json.dumps(row), flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
