"""Wraps stressgnn.* the way flow_service.py wraps flowsurrogate.* — a thin,
JSON-facing layer around a library treated as read-only.

Unlike the flow surrogate (a CNN on a fixed-size grid), the GNN is mesh-native: it
runs message passing on the actual triangulation, so every request needs a real
gmsh mesh of the *current* hole layout. gmsh's C++ state is process-global and not
thread-safe, so mesh building is serialised behind `_GMSH_LOCK` even though FastAPI
may run requests in different worker threads (see main.py's run_in_executor calls).
"""

from __future__ import annotations

import threading
import time
from typing import Iterator

import numpy as np
import torch
from matplotlib.tri import LinearTriInterpolator, Triangulation

from stressgnn.dataset import GraphDataset, build_graph, collate
from stressgnn.fem import solve
from stressgnn.geometry import HEIGHT, MAX_R, MIN_R, WIDTH, Design, is_valid
from stressgnn.mesh import build_mesh
from stressgnn.train import load_model, pick_device

from . import config
from .device_lock import MPS_LOCK

_GMSH_LOCK = threading.Lock()


class StressService:
    def __init__(self) -> None:
        self.device = pick_device("auto")
        # Normalisation stats are purely a property of vm_target across train.npz —
        # identical for every checkpoint regardless of that model's input features —
        # so one reference load is enough to serve all model keys.
        ref = GraphDataset(config.STRESS_DATA_DIR / "train.npz")
        self.mean, self.std = ref.mean, ref.std
        self.peak_mean, self.peak_std = ref.peak_mean, ref.peak_std
        self._models: dict[str, tuple[torch.nn.Module, object]] = {}

        # gmsh.initialize() registers a SIGINT handler, which CPython only allows
        # from the main thread of the main interpreter. `StressService()` is a
        # module-level singleton built at import time, i.e. on the main thread —
        # force gmsh's one-time init here so the *first* mesh build never happens
        # inside one of FastAPI's run_in_executor worker threads (main.py), where
        # it would raise "signal only works in main thread".
        build_mesh(Design(), config.STRESS_H_COARSE)

    def _get_model(self, key: str) -> tuple[torch.nn.Module, object]:
        if key not in self._models:
            ckpt = config.STRESS_RUNS_DIR / f"{key}.pt"
            model, cfg = load_model(ckpt, device=str(self.device))
            self._models[key] = (model, cfg)
        return self._models[key]

    def warmup(self, keys: tuple[str, ...] = ("gnn-mp10-glob",)) -> None:
        for key in keys:
            self.predict([[0.5, 0.5, 0.12], [1.3, 0.4, 0.10]], load_angle_deg=-90.0, model_key=key)

    def _mesh_and_solve(self, design: Design):
        with _GMSH_LOCK:
            cm = build_mesh(design, config.STRESS_H_COARSE)
        cs = solve(cm, design)
        return cm, cs

    def _predict_field(self, holes: np.ndarray, load_angle_rad: float, model_key: str):
        """Mesh + coarse-solve + GNN forward pass. Returns (mesh, fem_solution, vm_nodal, peak_vm)."""
        design = Design(holes=holes, load_angle=load_angle_rad)
        cm, cs = self._mesh_and_solve(design)
        model, cfg = self._get_model(model_key)

        sample = {
            "nodes": cm.nodes.astype(np.float32),
            "tris": cm.tris.astype(np.int32),
            "node_type": cm.node_type.astype(np.int8),
            "local_size": cm.local_size().astype(np.float32),
            "vm_coarse": cs.vm_node.astype(np.float32),
            "u_coarse": cs.u.astype(np.float32),
            "vm_target": np.zeros(cm.n_nodes, dtype=np.float32),  # unknown at inference time
            "holes": holes.astype(np.float32),
            "load_angle": np.float32(load_angle_rad),
            "max_vm_fine": np.float32(0.0),
            "max_vm_coarse": np.float32(cs.vm_node.max()),
            "max_vm_target": np.float32(0.0),
            "h_coarse": np.float32(config.STRESS_H_COARSE),
        }
        g = build_graph(
            sample,
            cfg.use_coarse_solution,
            cfg.use_position,
            self.mean,
            self.std,
            with_grid=False,
            peak_stats=(self.peak_mean, self.peak_std),
        )
        with MPS_LOCK, torch.no_grad():
            field_out, peak_out = model(collate([g]).to(self.device))
            vm_nodal = field_out.cpu().numpy() * self.std + self.mean
            peak_vm = float(peak_out.cpu().numpy()[0]) * self.peak_std + self.peak_mean
        return cm, cs, vm_nodal, peak_vm

    def _grid_field(self, nodes: np.ndarray, tris: np.ndarray, values: np.ndarray) -> list[list[float | None]]:
        """Resample a nodal field onto a regular grid for the heatmap.

        `LinearTriInterpolator` naturally masks points outside the triangulation —
        i.e. inside a hole or outside the plate — which is exactly the "no material
        here" signal the frontend needs (same convention as FieldViewer's solid mask).
        """
        xs = (np.arange(config.STRESS_GRID_NX) + 0.5) / config.STRESS_GRID_NX * WIDTH
        ys = (np.arange(config.STRESS_GRID_NY) + 0.5) / config.STRESS_GRID_NY * HEIGHT
        gx, gy = np.meshgrid(xs, ys)
        tri = Triangulation(nodes[:, 0], nodes[:, 1], tris)
        zi = LinearTriInterpolator(tri, values)(gx, gy)
        mask = np.ma.getmaskarray(zi)
        data = np.ma.filled(zi, np.nan)
        return [
            [None if mask[j, i] or not np.isfinite(data[j, i]) else float(data[j, i]) for i in range(config.STRESS_GRID_NX)]
            for j in range(config.STRESS_GRID_NY)
        ]

    def predict(self, holes: list[list[float]], load_angle_deg: float, model_key: str = "gnn-mp10-glob") -> dict:
        t0 = time.perf_counter()
        holes_arr = np.asarray(holes, dtype=np.float64)
        if not is_valid(holes_arr):
            raise ValueError("Invalid hole layout: holes overlap, or sit too close to the plate edge / each other.")

        load_angle_rad = float(np.deg2rad(load_angle_deg))
        cm, cs, vm_nodal, peak_vm = self._predict_field(holes_arr, load_angle_rad, model_key)
        peak_idx = int(np.argmax(vm_nodal))
        quality = config.STRESS_MODEL_QUALITY.get(model_key, {"peak_err_pct": None})

        return {
            "peak_vm_surrogate": peak_vm,
            "peak_error_pct_expected": quality["peak_err_pct"],
            "field": {
                "vm": self._grid_field(cm.nodes, cm.tris, vm_nodal),
                "extent": [0.0, WIDTH, 0.0, HEIGHT],
            },
            "mesh": {
                "nodes": cm.nodes.tolist(),
                "triangles": cm.tris.tolist(),
                "vm_nodal": vm_nodal.tolist(),
            },
            "peak_location": cm.nodes[peak_idx].tolist(),
            "vm_coarse_fem": self._grid_field(cm.nodes, cm.tris, cs.vm_node),
            "peak_vm_coarse_fem": float(cs.vm_node.max()),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "model": model_key,
        }

    def optimise_streaming(
        self,
        holes: list[list[float]],
        load_angle_deg: float,
        model_key: str,
        n_iters: int,
        seed: int,
    ) -> Iterator[dict]:
        """(1+1)-style randomised local search over hole positions/radii at fixed removed area.

        The GNN runs message passing on the mesh graph itself, which changes topology
        with every hole move — there is no gradient of "peak stress" w.r.t. hole
        position the way there is a gradient of Cd w.r.t. the flow project's Fourier
        coefficients (see stress-gnn/BACKGROUND.md). The plan calls this out and
        explicitly sanctions search-in-the-loop over the surrogate instead (§3.2,
        option A) — every candidate is a real re-mesh + re-solve + GNN forward pass,
        so this is a genuine (if not massively parallel) optimisation.
        """
        holes0 = np.asarray(holes, dtype=np.float64)
        if not is_valid(holes0):
            raise ValueError("Invalid hole layout: holes overlap, or sit too close to the plate edge / each other.")
        rng = np.random.default_rng(seed)
        load_angle_rad = float(np.deg2rad(load_angle_deg))
        n_holes = len(holes0)
        removed_area = float(np.pi * (holes0[:, 2] ** 2).sum())

        def area_locked(h: np.ndarray) -> np.ndarray:
            scale = np.sqrt(removed_area / (np.pi * (h[:, 2] ** 2).sum()))
            out = h.copy()
            out[:, 2] *= scale
            return out

        def evaluate(h: np.ndarray) -> float:
            return self._predict_field(h, load_angle_rad, model_key)[3]

        best = holes0
        best_peak = evaluate(best)
        initial_peak = best_peak
        step = 0.12  # fraction of plate size; annealed down over iterations

        yield {"it": 0, "peak_vm_current": best_peak, "best_holes": best.tolist()}

        for it in range(1, n_iters + 1):
            sigma = step * (0.96**it)
            cand = best.copy()
            cand[:, :2] += rng.normal(0, sigma, size=(n_holes, 2))
            cand[:, 2] += rng.normal(0, sigma * 0.3, size=n_holes)
            cand[:, 2] = np.clip(cand[:, 2], MIN_R, MAX_R)
            cand = area_locked(cand)
            if not is_valid(cand):
                continue
            try:
                peak = evaluate(cand)
            except Exception:
                continue
            if peak < best_peak:
                best, best_peak = cand, peak
            if it % 2 == 0 or it == n_iters:
                yield {"it": it, "peak_vm_current": best_peak, "best_holes": best.tolist()}

        yield {
            "done": True,
            "final_holes": best.tolist(),
            "peak_vm_initial": initial_peak,
            "peak_vm_final": best_peak,
        }


stress_service = StressService()
