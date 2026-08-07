"""Wraps flowsurrogate.* so the rest of the backend never touches torch directly.

Loads the training-set normaliser once (needed to decode physical units — the
checkpoints don't carry it) and caches models by key. flowsurrogate/ is treated
as a library: nothing here reimplements its logic, it only renders inputs, calls
the model, and reshapes outputs for JSON.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from flowsurrogate.data import Normalizer, input_channels, load_splits
from flowsurrogate.geometry import Domain, area, solid_mask
from flowsurrogate.lbm import SolverConfig
from flowsurrogate.optimize import boundary_xy, render_inputs, to_params
from flowsurrogate.optimize import verify as lbm_verify
from flowsurrogate.train import load_model, pick_device

from . import config
from .device_lock import MPS_LOCK


def downscale(field: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool an (ny, nx) field by `factor` for lighter JSON payloads."""
    ny, nx = field.shape
    ny2, nx2 = ny // factor, nx // factor
    return field[: ny2 * factor, : nx2 * factor].reshape(ny2, factor, nx2, factor).mean(axis=(1, 3))


class FlowService:
    def __init__(self) -> None:
        self.domain = Domain()
        self.device = pick_device("auto")
        train_ds, _, _, norm = load_splits(config.FLOW_DATA_DIR, self.domain, limit_train=256)
        self.norm: Normalizer = norm
        self.default_target_area = float(np.median([area(p) for p in train_ds.params]))
        self._models: dict[str, torch.nn.Module] = {}

    def _get_model(self, key: str) -> torch.nn.Module:
        if key not in self._models:
            ckpt = config.FLOW_RUNS_DIR / f"{key}.pt"
            model, _cfg = load_model(ckpt, device="auto")
            self._models[key] = model
        return self._models[key]

    def warmup(self, keys: tuple[str, ...] = ("unet-phys",)) -> None:
        for key in keys:
            self.predict([0.0] * 6, reynolds=80.0, model_key=key)

    def predict(
        self,
        coeffs: list[float],
        reynolds: float,
        model_key: str,
        target_area: float | None = None,
    ) -> dict:
        t0 = time.perf_counter()
        target_area = target_area or self.default_target_area
        model = self._get_model(model_key)

        coeffs_arr = np.asarray([coeffs], dtype=np.float64)
        params = to_params(coeffs_arr, target_area)
        x = input_channels(params, np.asarray([reynolds]), self.domain)
        x_t = torch.from_numpy(x).to(self.device)

        with MPS_LOCK, torch.no_grad():
            fields_norm, cd_norm = model(x_t)
            fields = self.norm.decode_fields(fields_norm)[0].cpu().numpy()
            cd = float(self.norm.decode_cd(cd_norm)[0].cpu())

        mask = solid_mask(params[0], self.domain)
        boundary = boundary_xy(params[0])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        quality = config.FLOW_MODEL_QUALITY.get(model_key, {"cd_mape_pct": None})

        return {
            "cd_surrogate": cd,
            "cd_error_pct_expected": quality["cd_mape_pct"],
            "fields": {
                "u": downscale(fields[0], config.FIELD_DOWNSCALE).tolist(),
                "v": downscale(fields[1], config.FIELD_DOWNSCALE).tolist(),
                "p": downscale(fields[2], config.FIELD_DOWNSCALE).tolist(),
            },
            "mask": downscale(mask.astype(np.float64), config.FIELD_DOWNSCALE).round().astype(int).tolist(),
            "domain": {
                "nx": self.domain.nx,
                "ny": self.domain.ny,
                "cx": self.domain.cx,
                "cy": self.domain.cy,
                "downscale": config.FIELD_DOWNSCALE,
            },
            "boundary_xy": boundary.tolist(),
            "target_area": target_area,
            "latency_ms": latency_ms,
            "model": model_key,
        }

    def optimise_streaming(
        self,
        reynolds: float,
        n_starts: int,
        iters: int,
        model_key: str,
        seed: int,
        init_coeffs: list[float] | None = None,
    ):
        """Adam optimisation over the shape coefficients, yielding progress every 2 iters.

        Mirrors flowsurrogate.optimize.optimise() but as a generator so /api/flow/optimize
        can stream it over SSE instead of only returning the final result.

        `init_coeffs` (the design currently on screen) seeds start #0. Without it every
        start comes from `sample_shapes`, so the run is identical for every user design at
        a given seed and the UI ends up showing "Start: your shape" beside a result it had
        no part in. Any remaining starts stay random, which is what `n_starts > 1` is for.
        """
        from flowsurrogate.geometry import sample_shapes

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = self._get_model(model_key)
        target_area = self.default_target_area
        coeff_bound = 0.16

        init = sample_shapes(rng, n_starts, coeff_range=coeff_bound * 0.6)[:, 1:]
        if init_coeffs is not None:
            # Clamped to the same box the optimiser projects onto each step, so start #0
            # is a point the run can actually occupy.
            init[0] = np.clip(np.asarray(init_coeffs, dtype=init.dtype), -coeff_bound, coeff_bound)
        coeffs = torch.tensor(init, dtype=torch.float32, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([coeffs], lr=8e-3)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        history: dict[str, list[float]] = {"cd_mean": [], "cd_best": []}
        for it in range(iters):
            # enable_grad() is not redundant: grad mode is thread-local and this generator
            # runs on whatever pool thread run_in_executor hands it, so it must not trust
            # the ambient state. This is the only place in the backend that needs autograd
            # switched ON, and it is the one that fails silently-then-loudly if it isn't.
            with MPS_LOCK, torch.enable_grad():
                opt.zero_grad(set_to_none=True)
                x = render_inputs(coeffs, target_area, reynolds, self.domain, mask_eps=0.7)
                _, cd_norm = model(x)
                cd = self.norm.decode_cd(cd_norm)
                cd.sum().backward()
                opt.step()
                with torch.no_grad():
                    coeffs.clamp_(-coeff_bound, coeff_bound)
            cd_mean = float(cd.mean().detach())
            cd_best = float(cd.min().detach())
            history["cd_mean"].append(cd_mean)
            history["cd_best"].append(cd_best)

            if it % 2 == 0 or it == iters - 1:
                best_idx = int(cd.argmin().detach())
                best_coeffs = coeffs[best_idx].detach().cpu().numpy()
                best_params = to_params(best_coeffs[None, :], target_area)[0]
                boundary = boundary_xy(best_params)
                yield {
                    "it": it,
                    "cd_best": cd_best,
                    "cd_mean": cd_mean,
                    "coeffs_best": best_coeffs.tolist(),
                    "boundary_xy": boundary.tolist(),
                }

        for p in model.parameters():
            p.requires_grad_(True)

        with MPS_LOCK, torch.no_grad():
            x = render_inputs(coeffs, target_area, reynolds, self.domain, mask_eps=0.7)
            cd_final = self.norm.decode_cd(model(x)[1]).cpu().numpy()
        best_idx = int(np.argmin(cd_final))
        final_coeffs = coeffs.detach().cpu().numpy()[best_idx]

        yield {
            "done": True,
            "final_coeffs": final_coeffs.tolist(),
            "history": history,
            "cd_surrogate_final": float(cd_final[best_idx]),
            "target_area": target_area,
        }

    def verify(
        self,
        coeffs: list[float],
        reynolds: float,
        target_area: float | None = None,
        steps: int = 9000,
        model_key: str = "unet-phys",
    ) -> dict:
        """Runs the real LBM on `coeffs` and compares it against the surrogate.

        Uses `flowsurrogate.optimize.verify()` unmodified (own compiled solver
        instance) rather than the shared singleton in `lbm_frames.py`: verify wants
        the fully time-averaged, high-step-count answer, while the shared singleton
        is tuned for the shorter, differently-configured runs `/api/flow/simulate`
        streams frame-by-frame.
        """
        t0 = time.perf_counter()
        target_area = target_area or self.default_target_area
        solver_cfg = SolverConfig(n_steps=steps, n_avg=min(2500, max(200, steps // 3)))
        coeffs_arr = np.asarray([coeffs], dtype=np.float64)
        # Not held across self.predict() below — that call takes MPS_LOCK itself,
        # and the lock isn't reentrant.
        with MPS_LOCK:
            cd_lbm, fields, _params = lbm_verify(coeffs_arr, target_area, reynolds, self.domain, solver_cfg, device=str(self.device))

        pred = self.predict(coeffs, reynolds, model_key, target_area)
        cd_surrogate = pred["cd_surrogate"]
        cd_lbm_val = float(cd_lbm[0])
        gap_pct = 100.0 * abs(cd_lbm_val - cd_surrogate) / max(abs(cd_lbm_val), 1e-9)
        wall_time_s = time.perf_counter() - t0

        return {
            "cd_lbm": cd_lbm_val,
            "cd_surrogate": cd_surrogate,
            "gap_pct": gap_pct,
            "fields": {
                "u": downscale(fields["u"][0], config.FIELD_DOWNSCALE).tolist(),
                "v": downscale(fields["v"][0], config.FIELD_DOWNSCALE).tolist(),
                "p": downscale(fields["p"][0], config.FIELD_DOWNSCALE).tolist(),
            },
            "wall_time_s": wall_time_s,
        }


flow_service = FlowService()
