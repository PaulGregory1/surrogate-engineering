"""Milestone 4: time and measure the memory of one volume value+grad step (fcso/volume.py) at 64^3 and 48^3.

    PYTORCH_ENABLE_MPS_FALLBACK=1 /usr/bin/time -l .venv/bin/python scripts/volume_check.py

Per resolution and chunk size: value-only time, value+grad time (mean of --reps steps after one warm-up),
peak MPS current allocation inside the step (sampled after each chunk forward and around the trunk
backward), MPS driver allocation after the step (empty_cache before it), process peak RSS.
Output: runs/m4/timing.json (+ stdout, tee to logs/m4_volume_check.log).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from fcso import prior as P
from fcso.volume import Volume, memory_stats


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


def timed(fn, dev, reps):
    fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t0)
    return sum(ts) / len(ts), min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-chairs", type=int, default=3)
    args = ap.parse_args()
    pr = P.Prior(load_dm=False)
    dev = pr.device
    ids = P.read_ids(os.path.join(P.ROOT, "splits", "opt.txt"))[:args.n_chairs]
    zs = [pr.encode_id(i) for i in ids]
    out = dict(device=str(dev), torch=torch.__version__, chairs=ids, reps=args.reps, runs=[])
    configs = [(64, 32768), (64, 16384), (64, 65536), (48, 32768)]
    for res, chunk in configs:
        est = Volume(pr, res=res, chunk=chunk)
        t_val, t_grad, peaks, drivers = [], [], [], []
        for z in zs:
            t_val.append(timed(lambda: est.value(z), dev, args.reps)[0])
            t_grad.append(timed(lambda: est.value_and_grad(z, lambda v: (v - 0.1) ** 2), dev, args.reps)[0])
            if dev.type == "mps":
                torch.mps.empty_cache()
            est.trace = {}
            est.value_and_grad(z)
            sync(dev)
            peaks.append(est.trace.get("peak_current_allocated", 0) / 2 ** 30)
            drivers.append(memory_stats().get("mps_driver_gb", 0.0))
            est.trace = None
        zb = torch.cat(zs)
        t_batch = timed(lambda: est.value_and_grad(zb), dev, args.reps)[0]
        row = dict(res=res, chunk=chunk, value_s=sum(t_val) / len(t_val), value_grad_s=sum(t_grad) / len(t_grad),
                   value_grad_s_per_chair=t_grad, value_grad_batch_s=t_batch, batch=len(zs),
                   peak_mps_current_gb=max(peaks), mps_driver_after_gb=max(drivers),
                   rss_peak_gb=memory_stats()["rss_peak_gb"])
        out["runs"].append(row)
        print(json.dumps(row), flush=True)
    os.makedirs(os.path.join(P.ROOT, "runs", "m4"), exist_ok=True)
    with open(os.path.join(P.ROOT, "runs", "m4", "timing.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
