"""Milestone 6: calibrate the Option B guidance scale lambda (SPEC "Guidance, Option B").

For each debug chair: z_hat = z_s after k x 15 Adam steps (the Phase 1 of FCSO, k = 1 and 3), c = V(z_hat).
At the first ODE step (sigma_start = 0.25, noise seeds 0-2), measure |d| and |grad_x (V(Dec(x)) - c)^2|:
lam_1 = |d| / |grad| is the lambda for which the guidance term equals |d|; the SPEC window 0.1-1 x |d| is
[0.1 lam_1, lam_1]. Then run full corrections (10 Euler steps) with lam in {0, candidates} and report
V_after / c and (V_after - c)^2.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./.venv/bin/python scripts/calibrate_guidance.py [--mode noisy|x0]
         [--lams 0 1000 3000 ...] [--solver euler]
Output: runs/m6/calibration_<mode>.json, log logs/m6_calibrate_<mode>.log.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fcso import correct as C  # noqa: E402
from fcso import prior as P  # noqa: E402
from fcso.optimize import OptConfig, phase1  # noqa: E402
from fcso.volume import Volume  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="noisy")
    ap.add_argument("--solver", default="euler")
    ap.add_argument("--lams", nargs="*", type=float, default=None)
    ap.add_argument("--cycles", nargs="*", type=int, default=[1, 3])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tag = f"{a.mode}{'_' + a.solver if a.solver != 'euler' else ''}{a.tag}"
    os.makedirs(os.path.join(P.ROOT, "runs", "m6"), exist_ok=True)
    logf = open(os.path.join(P.ROOT, "logs", f"m6_calibrate_{tag}.log"), "a")

    def log(msg):
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    prior = P.Prior(load_dm=True)
    est = Volume(prior, 64)
    cfg = OptConfig.from_yaml(os.path.join(P.ROOT, "configs", "base.yaml"))
    ids = P.read_ids(os.path.join(P.ROOT, "splits", "opt.txt"))[:3]
    out = dict(mode=a.mode, solver=a.solver, chairs={})
    states = {}
    for sid in ids:
        z_s = prior.encode_id(sid).reshape(1, 512, 8)
        V0 = float(est.value(z_s)[0])
        z = z_s.clone()
        for k in range(1, max(a.cycles) + 1):
            z, _, _ = phase1(est, z, z_s, cfg, cfg.steps_per_cycle)
            if k in a.cycles:
                states[(sid, k)] = (z.clone(), V0)
    # 1. ratio at the first ODE step
    lam1 = []
    for (sid, k), (z_hat, V0) in states.items():
        c = float(est.value(z_hat)[0])
        rows = []
        for seed in a.seeds:
            cal = C.calibrate_lambda(prior, est, z_hat, c, 0.25, seed=seed, mode=a.mode)
            rows.append(cal)
            lam1.append(cal["lam"])
            log(f"{sid[:8]} cycle {k} seed {seed}: c/V0 {c / V0:.3f} V(x)/c {cal['V_x'] / c:.3f} |d| {cal['d_norm']:.2f} "
                f"|grad (V-c)^2| {cal['grad_norm']:.4g} -> lam for |g| = |d|: {cal['lam']:.4g}")
        out["chairs"].setdefault(sid, {})[f"cycle{k}"] = dict(c=c, V0=V0, first_step=rows)
    med = float(np.median(lam1))
    out["lam1_median"] = med
    out["lam1_all"] = lam1
    log(f"lam_1 (|g| = |d| at the first step): median {med:.4g}, range {min(lam1):.4g} - {max(lam1):.4g}")
    # 2. full corrections
    lams = a.lams if a.lams is not None else [0.0]
    for (sid, k), (z_hat, V0) in states.items():
        c = float(est.value(z_hat)[0])
        for lam in lams:
            res = []
            for seed in a.seeds:
                g = C.VolumeGuidance(prior, est, c, lam, a.mode) if lam else None
                t0 = time.perf_counter()
                z_new, infos = C.correct(prior, z_hat, 0.25, 10, seed=seed, solver=a.solver, guidance=g,
                                         return_info=True)
                if prior.device.type == "mps":
                    torch.mps.synchronize()
                dt = time.perf_counter() - t0
                V = float(est.value(z_new)[0])
                ratios = [i.get("g_norm", 0.0) / i["d_norm"] for i in infos]
                res.append(dict(seed=seed, V_over_c=V / c, sq=(V - c) ** 2, time_s=dt, g_over_d=ratios,
                                d_norm=[i["d_norm"] for i in infos], V_x=[i.get("V") for i in infos]))
            vc = [r["V_over_c"] for r in res]
            log(f"{sid[:8]} cycle {k} lam {lam:<8.4g}: V_after/c {np.mean(vc):.4f} ({', '.join(f'{v:.3f}' for v in vc)})  "
                f"|g|/|d| step0 {np.mean([r['g_over_d'][0] for r in res]):.3f} max {np.max([max(r['g_over_d']) for r in res]):.3f}  "
                f"{np.mean([r['time_s'] for r in res]):.2f} s/correction")
            out["chairs"][sid][f"cycle{k}"].setdefault("corrections", {})[f"{lam:g}"] = res
    with open(os.path.join(P.ROOT, "runs", "m6", f"calibration_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
