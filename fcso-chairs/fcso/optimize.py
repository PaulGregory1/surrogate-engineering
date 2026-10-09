"""Shared optimization driver for every method (SPEC "FCSO algorithm and hyperparameters", "Methods compared").

One run = one method config applied to one chair, as a loop of cycles:

    cycle k:  Phase 1: n = min(M, budget left) gradient steps on V (+ lam_reg ||z - z_s||^2) from z^(k)
              Phase 2: guided correction of z_hat (FCSO / FMG, `build_phase2`; None for GD)
              r = 1 - V(z^(k+1)) / V_initial at 64^3 (EVAL_RES)
              first time r >= 10 %, 20 %, ..., 70 %: snapshot (latent, step count, wall time, V/V_init)
    stop at r >= target (70 %) or after max_steps (1000) gradient steps; thresholds never crossed are
    recorded as failures.

Phase 1 optimizer: 'sgd' (Eq. 5, z <- z - lr * g) or 'adam' (torch.optim.Adam, state reset at the start of
every cycle, SPEC). For GD the cycle is only the bookkeeping unit (r, snapshots, checkpoint), so GD is
exactly FCSO with Phase 2 removed (App. B). Checkpoints are written after every cycle (atomic replace) and
hold everything needed to resume and to build the report: z, z_s, snapshots with their latents, the per-cycle
log, the accumulated wall time. Wall time counts only optimization work (gradient steps, Phase 2, the r
evaluation), not encoding, checkpoint I/O or evaluation meshes.
"""
import json
import os
import time
from dataclasses import asdict, dataclass, field, fields

import torch
import yaml

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


@dataclass
class OptConfig:
    name: str = "gd"
    method: str = "gd"              # gd | fcso | fmg  (FCSO without guidance = fcso with lam_guid 0)
    # Phase 1
    optimizer: str = "adam"         # sgd | adam (Adam state reset every cycle)
    lr: float = 0.005               # Table 6
    steps_per_cycle: int = 15       # M, Table 6
    lam_reg: float = 0.0            # GD regularization lam_reg ||z - z_s||^2 (App. B)
    # budget and progress
    max_steps: int = 1000           # gradient steps (App. E, Table 5)
    target: float = 0.7
    thresholds: tuple = THRESHOLDS
    opt_res: int = 64               # grid of the optimized volume (volume.OPT_RES)
    eval_res: int = 64              # grid of r and of the thresholds (volume.EVAL_RES)
    beta: float = 1.0
    seed: int = 0                   # per-shape seed; Phase 2 noise seed of cycle k = seed * 100003 + k
    encode_seed: int = 0            # seed of the 2048-point surface subsample (same as milestones 2-4)
    # Phase 2 (milestone 6)
    sigma_start: float = 0.25       # tau = 0.2 -> sigma = tau / (1 - tau)
    n_ode: int = 10
    solver: str = "euler"
    guidance: str = "noisy"         # noisy | x0 | none
    lam_guid: float = 0.0
    fmg_factor: float = 0.97        # FMG target c = fmg_factor * V_current
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path, overrides=None):
        """Load a config; a 'base: other.yaml' key (relative to the file) is loaded first and overridden."""
        d = _load_yaml(path)
        d.setdefault("name", os.path.splitext(os.path.basename(path))[0])
        d.update(overrides or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        if "thresholds" in d:
            d["thresholds"] = tuple(float(t) for t in d["thresholds"])
        return cls(**d)

    def to_dict(self):
        d = asdict(self)
        d["thresholds"] = list(self.thresholds)
        return d


def _load_yaml(path):
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    base = d.pop("base", None)
    if base:
        b = _load_yaml(os.path.join(os.path.dirname(path), base))
        b.pop("name", None)
        b.update(d)
        d = b
    return d


def _th_key(th):
    return f"{int(round(100 * th))}"


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------------------------- Phase 1
def phase1(est, z, z_s, cfg, n_steps):
    """n_steps gradient steps on V(z) + lam_reg ||z - z_s||^2 from z (1, 512, 8). Adam state is fresh.
    Returns the new z (detached) and per-step V (before each step) and gradient norms."""
    p = z.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=cfg.lr) if cfg.optimizer == "adam" else None
    if cfg.optimizer not in ("adam", "sgd"):
        raise ValueError(cfg.optimizer)
    Vs, gnorms = [], []
    for _ in range(n_steps):
        V, g = est.value_and_grad(p)
        if cfg.lam_reg:
            g = g + 2.0 * cfg.lam_reg * (p.detach() - z_s)
        Vs.append(float(V[0]))
        gnorms.append(float(g.norm()))
        with torch.no_grad():
            if opt is None:
                p -= cfg.lr * g
            else:
                p.grad = g
                opt.step()
    return p.detach(), Vs, gnorms


def build_phase2(cfg, prior, est):
    """Phase 2 callable (z_hat, V_hat, cycle_seed) -> (z, info dict with 'c'), or None for GD.

    fcso: c = V_hat (V at the end of Phase 1), guided correction (Option B, Eq. 8 as adapted in SPEC) with
          lam = cfg.lam_guid on the noisy state ('noisy') or on D(x; sigma) ('x0'); lam = 0 or guidance 'none'
          is the unguided correction (FCSO without guidance, App. E).
    fmg:  c = fmg_factor * V_hat with V_hat the current volume (no Phase 1), same guided correction.
    The returned info holds c, the norms of d and of the guidance term at the first ODE step, the max
    |g|/|d| over the steps, V(y) at the last step, and 'grad_evals' (volume-gradient evaluations in Phase 2,
    which count as gradient steps for FMG only)."""
    if cfg.method == "gd":
        return None
    if cfg.method not in ("fcso", "fmg"):
        raise ValueError(f"unknown method {cfg.method!r}")
    from fcso import correct as C

    guided = cfg.lam_guid > 0 and cfg.guidance != "none"

    def phase2(z_hat, V_hat, seed):
        c = V_hat if cfg.method == "fcso" else cfg.fmg_factor * V_hat
        g = C.VolumeGuidance(prior, est, c, cfg.lam_guid, cfg.guidance) if guided else None
        z, infos = C.correct(prior, z_hat, cfg.sigma_start, cfg.n_ode, seed=seed, solver=cfg.solver, guidance=g,
                             return_info=True)
        info = dict(c=c, d_norm0=infos[0]["d_norm"], grad_evals=g.n_evals if g else 0)
        if g is not None:
            ratios = [i["g_norm"] / i["d_norm"] for i in infos]
            info.update(g_norm0=infos[0]["g_norm"], g_over_d0=ratios[0], g_over_d_max=max(ratios),
                        V_x0=infos[0]["V"], V_x_last=infos[-1]["V"])
        return z, info

    return phase2


# ---------------------------------------------------------------------------------------------- one shape
def _save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def summarize(state):
    """JSON-friendly summary of a run state (no tensors)."""
    snaps = {}
    for th in state["cfg"]["thresholds"]:
        s = state["snapshots"].get(_th_key(th))
        snaps[_th_key(th)] = {"reached": s is not None, **({k: v for k, v in s.items() if k != "z"} if s else {})}
    return dict(id=state["id"], name=state["cfg"]["name"], method=state["cfg"]["method"], seed=state["cfg"]["seed"],
                done=state["done"], stop_reason=state["stop_reason"], steps=state["step"], cycles=state["cycle"],
                elapsed_s=state["elapsed"], V_initial=state["V0"], V_final=state["V"],
                ratio_final=state["V"] / state["V0"], r_final=1 - state["V"] / state["V0"],
                norm_dz_final=state["log"][-1]["norm_dz"] if state["log"] else 0.0,
                failures=[_th_key(th) for th in state["cfg"]["thresholds"] if snaps[_th_key(th)]["reached"] is False],
                snapshots=snaps, config=state["cfg"])


def run_shape(prior, est, est_eval, shape_id, cfg, run_dir, log=print, z_s=None, phase2=None):
    """Optimize one chair with cfg, resuming from run_dir/ckpt.pt if present. Returns the summary dict."""
    os.makedirs(run_dir, exist_ok=True)
    ckpt = os.path.join(run_dir, "ckpt.pt")
    dev = prior.device
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if state["cfg"] != cfg.to_dict():
            raise RuntimeError(f"{ckpt} was written with another config; use --restart or another --name.\n"
                               f"saved: {state['cfg']}\nnow:   {cfg.to_dict()}")
        log(f"[{shape_id[:8]}] resume at cycle {state['cycle']}, step {state['step']}, "
            f"r {1 - state['V'] / state['V0']:.3f}, {state['elapsed']:.1f} s, done={state['done']}")
    else:
        if z_s is None:
            z_s = prior.encode_id(shape_id, seed=cfg.encode_seed)
        z_s = z_s.detach().reshape(1, 512, 8).cpu()
        V0 = float(est_eval.value(z_s)[0])
        state = dict(id=shape_id, cfg=cfg.to_dict(), z_s=z_s, z=z_s.clone(), V0=V0, V=V0, step=0, cycle=0,
                     elapsed=0.0, done=False, stop_reason=None, log=[],
                     snapshots={"0": dict(threshold=0.0, step=0, cycle=0, time_s=0.0, ratio=1.0, r=0.0, V=V0,
                                          z=z_s.clone())})
        log(f"[{shape_id[:8]}] start: V0 {V0:.5f} (64^3), method {cfg.method}, {cfg.optimizer} lr {cfg.lr}, "
            f"lam_reg {cfg.lam_reg}, seed {cfg.seed}")
        _save_atomic(state, ckpt)

    z_s = state["z_s"].to(dev)
    z = state["z"].to(dev)
    while not state["done"]:
        k = state["cycle"]
        n = min(cfg.steps_per_cycle, cfg.max_steps - state["step"])
        _sync(dev)
        t0 = time.perf_counter()
        if cfg.method == "fmg":  # no Phase 1: Phase 2 alone, repeated, from the current latent
            n = 0
            z_hat, Vs, gn, V_hat = z, [], [], state["V"]
        else:
            z_hat, Vs, gn = phase1(est, z, z_s, cfg, n)
            V_hat = float(est_eval.value(z_hat)[0])
        info2 = {}
        if phase2 is not None:
            z, info2 = phase2(z_hat, V_hat, cfg.seed * 100003 + k)
            V = float(est_eval.value(z)[0])
        else:
            z, V = z_hat, V_hat
        _sync(dev)
        dt = time.perf_counter() - t0
        if cfg.method == "fmg":  # FMG's gradient steps are its guidance gradients (DECISIONS.md)
            n = max(int(info2.get("grad_evals", 0)), 1)
        state["elapsed"] += dt
        state["step"] += n
        state["cycle"] = k + 1
        state["V"] = V
        r = 1 - V / state["V0"]
        c = info2.get("c")
        row = dict(cycle=k + 1, step=state["step"], V=V, r=r, ratio=V / state["V0"], V_phase1=V_hat,
                   c=c, sq_before=None if c is None else (V_hat - c) ** 2,
                   sq_after=None if c is None else (V - c) ** 2,
                   norm_dz=float((z - z_s).norm()), norm_z=float(z.norm()), time_s=dt, elapsed_s=state["elapsed"],
                   V_steps=Vs, grad_norm_first=gn[0] if gn else None, grad_norm_last=gn[-1] if gn else None,
                   noise_seed=None if phase2 is None else cfg.seed * 100003 + k,
                   **{f"p2_{a}": b for a, b in info2.items() if a != "c"})
        state["log"].append(row)
        for th in cfg.thresholds:
            key = _th_key(th)
            if key not in state["snapshots"] and r >= th - 1e-12:
                state["snapshots"][key] = dict(threshold=th, step=state["step"], cycle=k + 1, time_s=state["elapsed"],
                                               ratio=V / state["V0"], r=r, V=V, z=z.detach().cpu().clone())
                log(f"[{shape_id[:8]}]   snapshot {key}%: step {state['step']}, {state['elapsed']:.1f} s, "
                    f"V/V0 {V / state['V0']:.3f}")
        if r >= cfg.target - 1e-12:
            state["done"], state["stop_reason"] = True, "target"
        elif state["step"] >= cfg.max_steps:
            state["done"], state["stop_reason"] = True, "budget"
        state["z"] = z.detach().cpu().clone()
        if k % 5 == 0 or state["done"]:
            p2 = "" if c is None else (f"  P1 {V_hat / state['V0']:.4f} -> P2 {V / state['V0']:.4f} "
                                       f"(c/V0 {c / state['V0']:.4f}, |g|/|d| {info2.get('g_over_d0', 0):.2f})")
            log(f"[{shape_id[:8]}] cycle {k + 1:3d} step {state['step']:4d}  V/V0 {V / state['V0']:.4f}{p2}  "
                f"|z-z_s| {row['norm_dz']:.2f}  |g| {row['grad_norm_last'] or 0:.4f}  {dt:.1f} s/cycle  "
                f"total {state['elapsed']:.1f} s")
        _save_atomic(state, ckpt)
        _write_json(summarize(state), os.path.join(run_dir, "result.json"))
        _write_json(state["log"], os.path.join(run_dir, "cycles.json"))
    summ = summarize(state)
    _write_json(summ, os.path.join(run_dir, "result.json"))
    _write_json(state["log"], os.path.join(run_dir, "cycles.json"))
    log(f"[{shape_id[:8]}] done ({state['stop_reason']}): steps {state['step']}, V/V0 {summ['ratio_final']:.3f}, "
        f"{state['elapsed']:.1f} s, failures {summ['failures'] or 'none'}")
    return summ


def snapshot_items(state):
    """(label, snapshot dict) for the initial shape, every threshold (None if not reached) and the final z."""
    out = [("init", state["snapshots"]["0"])]
    for th in state["cfg"]["thresholds"]:
        out.append((_th_key(th), state["snapshots"].get(_th_key(th))))
    return out

