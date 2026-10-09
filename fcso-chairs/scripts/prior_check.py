"""Milestone 3: sample chairs from noise with the class-conditioned EDM prior, and apply the unguided
Phase 2 correction (sigma_start = 0.25, N = 10 Euler steps, truncated rho = 7 schedule, fresh noise) to
encoded opt chairs. Volume = no-grad occupancy fraction on a 64^3 grid over [-1, 1]^3 times 8 (the
differentiable estimator is milestone 4).

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -W default scripts/prior_check.py
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

from fcso import correct as C
from fcso import mesh_render as R
from fcso import prior as P

BOX = (2 * P.BOUND) ** 3


def occ_volume(pr, z, res=64):
    """No-grad volume estimate: |B| * fraction of grid points with logit > 0, per latent in z (B, 512, 8)."""
    return [BOX * float((v > 0).mean()) for v in pr.decode_grid(z, res=res)]


def mesh_of(pr, z):
    return R.logits_to_mesh(pr.decode_grid(z[None] if z.dim() == 2 else z, 128)[0])


def mesh_stats(m):
    if m is None:
        return dict(components=0, largest_share=0.0, mesh_volume=0.0)
    comps = m.split(only_watertight=False)
    vols = sorted([abs(float(c.volume)) for c in comps], reverse=True)
    return dict(components=len(comps), largest_share=vols[0] / max(sum(vols), 1e-12), mesh_volume=float(m.volume))


def mem():
    out = {"rss_gb": None}
    try:
        import resource
        out["rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30  # bytes on macOS
    except Exception:
        pass
    if torch.backends.mps.is_available():
        out["mps_driver_gb"] = torch.mps.driver_allocated_memory() / 2 ** 30
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--n-chairs", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="correction noise seeds")
    ap.add_argument("--sigma-start", type=float, default=C.tau_to_sigma(0.2))
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(P.ROOT, "runs", "m3"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pr = P.Prior()
    print(f"device {pr.device}, sigma_data {pr.sigma_data}, sigma_start {a.sigma_start}, N {a.n_steps}", flush=True)
    print("correction sigmas", [round(s, 5) for s in C.correction_sigmas(a.sigma_start, a.n_steps).tolist()], flush=True)
    res = {"sigma_start": a.sigma_start, "n_steps": a.n_steps,
           "correction_sigmas": C.correction_sigmas(a.sigma_start, a.n_steps).tolist()}

    # ---------------------------------------------------------------- 1. samples from noise
    t0 = time.time()
    zs = C.edm_sample(pr, a.n_samples, seed=a.sample_seed)
    if pr.device.type == "mps":
        torch.mps.synchronize()
    t_s = time.time() - t0
    vols = occ_volume(pr, zs)
    objs, labels, rows = [], [], []
    for i in range(a.n_samples):
        m = mesh_of(pr, zs[i])
        st = mesh_stats(m)
        row = dict(sample=i, z_std=float(zs[i].std()), z_mean=float(zs[i].mean()), occ_volume_64=vols[i], **st)
        rows.append(row)
        print("sample", json.dumps(row), flush=True)
        objs.append(m)
        labels.append(f"sample {i}  V64 {vols[i]:.3f}  comps {st['components']}")
    R.render_grid(objs, os.path.join(a.out, "samples.png"), labels=labels, ncols=a.n_samples)
    torch.save({"z": zs.cpu(), "seed": a.sample_seed}, os.path.join(a.out, "samples_z.pt"))
    res["samples"] = dict(seed=a.sample_seed, seconds_18_heun_steps=t_s, rows=rows)
    print(f"sampling {a.n_samples} chairs: {t_s:.1f} s", flush=True)

    # ---------------------------------------------------------------- 2. unguided correction
    ids = P.read_ids(os.path.join(P.ROOT, "splits", "opt.txt"))[: a.n_chairs]
    res["corrections"] = []
    objs, labels = [], []
    for sid in ids:
        zp = os.path.join(P.ROOT, "runs", "m2", f"{sid}_z.pt")
        z_hat = torch.load(zp)["z_mean"].to(pr.device) if os.path.exists(zp) else pr.encode_id(sid)
        v0 = occ_volume(pr, z_hat)[0]
        st0 = mesh_stats(m0 := mesh_of(pr, z_hat))
        objs.append(m0)
        labels.append(f"{sid[:8]} encoded  V64 {v0:.3f}")
        for seed in a.seeds:
            t0 = time.time()
            z1, info = C.correct(pr, z_hat, a.sigma_start, a.n_steps, seed=seed, return_info=True)
            if pr.device.type == "mps":
                torch.mps.synchronize()
            dt = time.time() - t0
            v1 = occ_volume(pr, z1)[0]
            m1 = mesh_of(pr, z1)
            st1 = mesh_stats(m1)
            row = dict(id=sid, seed=seed, V0_64=v0, V1_64=v1, dV_rel=(v1 - v0) / v0,
                       rms_dz=float((z1 - z_hat).pow(2).mean().sqrt()), seconds=dt,
                       comps0=st0["components"], comps1=st1["components"],
                       mesh_V0=st0["mesh_volume"], mesh_V1=st1["mesh_volume"],
                       d_norm_first=info[0]["d_norm"], d_norm_last=info[-1]["d_norm"])
            res["corrections"].append(row)
            print("correct", json.dumps(row), flush=True)
            torch.save({"z": z1.cpu(), "seed": seed}, os.path.join(a.out, f"{sid}_corr_s{seed}_z.pt"))
            objs.append(m1)
            labels.append(f"corrected s{seed}  V64 {v1:.3f}  dV {100 * (v1 - v0) / v0:+.1f}%")
    ncols = 1 + len(a.seeds)
    R.render_grid(objs, os.path.join(a.out, "correction.png"), labels=labels, ncols=ncols)
    dv = np.array([r["dV_rel"] for r in res["corrections"]])
    res["summary"] = dict(max_abs_dV_rel=float(np.abs(dv).max()), mean_abs_dV_rel=float(np.abs(dv).mean()),
                          criterion_lt_5pct=bool(np.abs(dv).max() < 0.05), memory=mem())
    print("summary", json.dumps(res["summary"]), flush=True)
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
