"""Run one method (configs/<method>.yaml) on the opt chairs, one chair at a time, resumable.

    PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i ./.venv/bin/python scripts/run_opt.py --config configs/gd.yaml
    ... --debug                       3 debug chairs, 3 cycles (45 steps), into runs/debug/<name>/
    ... --n-chairs 3                  first 3 chairs of splits/opt.txt (the debug set) with the full budget
    ... --set optimizer=sgd lr=0.01   override config keys (use --name to keep runs apart)

Outputs: runs/<name>/<id>/{ckpt.pt, result.json, cycles.json, eval.json}, runs/<name>/<id>_snapshots.png
(initial + each threshold, stray components in red), runs/<name>/summary.json; log: logs/<name>.log.
A chair whose ckpt says done is skipped (its evaluation is redone only if eval.json is missing or --re-eval).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
import yaml  # noqa: E402

from fcso import metrics as MT  # noqa: E402
from fcso import optimize as O  # noqa: E402
from fcso import prior as P  # noqa: E402
from fcso.volume import Volume, memory_stats  # noqa: E402


def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", help="run name (default: config 'name' or file stem)")
    ap.add_argument("--set", nargs="*", default=[], help="key=value config overrides")
    ap.add_argument("--debug", action="store_true", help="3 chairs, 3 cycles, runs/debug/")
    ap.add_argument("--n-chairs", type=int, help="first n chairs of the split file (default all, 3 with --debug)")
    ap.add_argument("--ids", nargs="*", help="explicit chair ids (prefixes allowed)")
    ap.add_argument("--split-file", default=os.path.join(P.ROOT, "splits", "opt.txt"))
    ap.add_argument("--max-steps", type=int, help="override the gradient-step budget")
    ap.add_argument("--restart", action="store_true", help="delete existing checkpoints of these chairs first")
    ap.add_argument("--no-eval", action="store_true", help="skip the 128^3 evaluation meshes and renders")
    ap.add_argument("--re-eval", action="store_true", help="redo the evaluation even if eval.json exists")
    a = ap.parse_args()

    ov = parse_set(a.set)
    if a.name:
        ov["name"] = a.name
    if a.max_steps is not None:
        ov["max_steps"] = a.max_steps
    elif a.debug:
        ov["max_steps"] = 3 * ov.get("steps_per_cycle", 15)
    cfg = O.OptConfig.from_yaml(a.config, ov)

    all_ids = P.read_ids(a.split_file)
    if a.ids:
        ids = [next(i for i in all_ids if i.startswith(p)) if len(p) < 30 else p for p in a.ids]
    else:
        ids = all_ids[: (a.n_chairs or (3 if a.debug else len(all_ids)))]
    out = os.path.join(P.ROOT, "runs", "debug" if a.debug else "", cfg.name)
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(P.ROOT, "logs"), exist_ok=True)
    logf = open(os.path.join(P.ROOT, "logs", ("debug_" if a.debug else "") + cfg.name + ".log"), "a")

    def log(msg):
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"=== run_opt {cfg.name}: {len(ids)} chairs, out {out}, config {json.dumps(cfg.to_dict())}")
    t_load = time.time()
    prior = P.Prior(load_dm=cfg.method != "gd")
    est = Volume(prior, cfg.opt_res, beta=cfg.beta)
    est_eval = est if cfg.eval_res == cfg.opt_res else Volume(prior, cfg.eval_res, beta=cfg.beta)
    phase2 = O.build_phase2(cfg, prior, est)
    log(f"device {prior.device}, models loaded in {time.time() - t_load:.1f} s")

    summaries = []
    for sid in ids:
        rdir = os.path.join(out, sid)
        if a.restart and os.path.isdir(rdir):
            for f in os.listdir(rdir):
                os.remove(os.path.join(rdir, f))
        summ = O.run_shape(prior, est, est_eval, sid, cfg, rdir, log=log, phase2=phase2)
        if not a.no_eval and (a.re_eval or not os.path.exists(os.path.join(rdir, "eval.json"))):
            t0 = time.time()
            MT.evaluate_run(prior, rdir, render_path=os.path.join(out, f"{sid}_snapshots.png"), log=log)
            log(f"[{sid[:8]}] evaluation {time.time() - t0:.1f} s")
        summaries.append(summ)
        if prior.device.type == "mps":
            torch.mps.empty_cache()
        log(f"memory {json.dumps({k: round(v, 2) for k, v in memory_stats().items()})}")

    rows = [{k: s[k] for k in ("id", "stop_reason", "steps", "elapsed_s", "ratio_final", "norm_dz_final",
                               "failures")} for s in summaries]
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(dict(config=cfg.to_dict(), chairs=rows), f, indent=1)
    for r in rows:
        log("summary " + json.dumps(r))
    logf.close()


if __name__ == "__main__":
    main()
