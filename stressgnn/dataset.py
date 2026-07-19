"""Dataset generation and graph construction.

Every sample is one design solved twice:

  * on a **fine** mesh (h = 0.010, ~25k nodes) -- the reference,
  * on a **coarse** mesh (h ~ U[0.035, 0.06], ~1k nodes) -- what the surrogate sees.

The learning target is the fine von Mises field interpolated onto the coarse nodes,
so a perfect model would give fine-mesh answers at coarse-mesh cost.  The coarse FEM
solution is stored too: it is both the baseline to beat and, optionally, an input
(the "correction model" variant).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .fem import interpolate, solve
from .geometry import Design, sample_designs
from .mesh import N_NODE_TYPES, Mesh, build_mesh

H_FINE: float = 0.010
LENGTH_SCALE: float = 0.05  # normalises edge vectors and the local-size feature


@dataclass
class GenSpec:
    name: str
    n: int
    seed: int
    h_coarse_range: tuple[float, float] = (0.035, 0.060)
    h_fine: float = H_FINE
    n_holes_choices: tuple[int, ...] = (2, 3, 4)
    angle_range: tuple[float, float] = (-np.pi / 2, 0.0)
    h_coarse_fixed: float | None = None  # forces one resolution (resolution study)


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def solve_pair(design: Design, h_coarse: float, h_fine: float = H_FINE) -> dict:
    """Fine reference + coarse discretisation of a single design."""
    t0 = time.perf_counter()
    fine = build_mesh(design, h_fine)
    sol_fine = solve(fine, design)
    t_fine = time.perf_counter() - t0

    t0 = time.perf_counter()
    coarse = build_mesh(design, h_coarse)
    sol_coarse = solve(coarse, design)
    t_coarse = time.perf_counter() - t0

    target = interpolate(fine, sol_fine.vm_node, coarse.nodes)
    return {
        "nodes": coarse.nodes.astype(np.float32),
        "tris": coarse.tris.astype(np.int32),
        "node_type": coarse.node_type.astype(np.int8),
        "local_size": coarse.local_size().astype(np.float32),
        "vm_coarse": sol_coarse.vm_node.astype(np.float32),
        "u_coarse": sol_coarse.u.astype(np.float32),
        "vm_target": target.astype(np.float32),
        "holes": design.holes.astype(np.float32),
        "load_angle": np.float32(design.load_angle),
        "h_coarse": np.float32(h_coarse),
        "n_nodes_fine": np.int32(fine.n_nodes),
        "max_vm_fine": np.float32(sol_fine.max_vm),
        "max_vm_coarse": np.float32(sol_coarse.vm_node.max()),
        "max_vm_target": np.float32(target.max()),
        "compliance_fine": np.float32(sol_fine.compliance),
        "compliance_coarse": np.float32(sol_coarse.compliance),
        "t_fine": np.float32(t_fine),
        "t_coarse": np.float32(t_coarse),
    }


VAR_KEYS = ("nodes", "tris", "node_type", "local_size", "vm_coarse", "u_coarse", "vm_target", "holes")


def pack(samples: list[dict]) -> dict[str, np.ndarray]:
    """Concatenate variable-length per-sample arrays and record the offsets."""
    out: dict[str, np.ndarray] = {}
    for key in VAR_KEYS:
        arrays = [s[key] for s in samples]
        out[key] = np.concatenate(arrays) if arrays else np.zeros(0)
        out[f"{key}_ptr"] = np.cumsum([0] + [len(a) for a in arrays]).astype(np.int64)
    for key in samples[0]:
        if key not in VAR_KEYS:
            out[key] = np.array([s[key] for s in samples])
    return out


def unpack(blob, i: int) -> dict:
    out = {}
    for key in VAR_KEYS:
        ptr = blob[f"{key}_ptr"]
        out[key] = blob[key][ptr[i] : ptr[i + 1]]
    for key in blob.files if hasattr(blob, "files") else blob:
        if key not in VAR_KEYS and not key.endswith("_ptr"):
            out[key] = blob[key][i]
    return out


def generate_split(spec: GenSpec, out_dir: Path, verbose: bool = True) -> Path:
    rng = np.random.default_rng(spec.seed)
    designs = sample_designs(rng, spec.n, n_holes_choices=spec.n_holes_choices, angle_range=spec.angle_range)
    samples = []
    t0 = time.perf_counter()
    for i, design in enumerate(designs):
        h = spec.h_coarse_fixed if spec.h_coarse_fixed else float(rng.uniform(*spec.h_coarse_range))
        samples.append(solve_pair(design, h, spec.h_fine))
        if verbose and (i + 1) % 25 == 0:
            el = time.perf_counter() - t0
            print(
                f"  [{spec.name}] {i + 1}/{spec.n}  {el:.0f}s  {el / (i + 1):.2f} s/sample  "
                f"eta {el / (i + 1) * (spec.n - i - 1) / 60:.1f} min",
                flush=True,
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.name}.npz"
    np.savez_compressed(path, **pack(samples))
    (out_dir / f"{spec.name}_meta.json").write_text(
        json.dumps({"spec": asdict(spec), "seconds_per_sample": (time.perf_counter() - t0) / spec.n}, indent=2)
    )
    if verbose:
        print(f"  [{spec.name}] wrote {path} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
    return path


def generate_resolution_split(source: Path, out_dir: Path, name: str, h: float, seed: int, n: int) -> Path:
    """Re-mesh the *same* designs at an unseen resolution (fine solve is reused)."""
    blob = np.load(source)
    ptr = blob["holes_ptr"]
    samples = []
    for i in range(min(n, len(blob["load_angle"]))):
        design = Design(holes=blob["holes"][ptr[i] : ptr[i + 1]].astype(np.float64), load_angle=float(blob["load_angle"][i]))
        samples.append(solve_pair(design, h))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, **pack(samples))
    return path


# --------------------------------------------------------------------------------------
# Graphs
# --------------------------------------------------------------------------------------


GRID_H, GRID_W = 96, 192  # dx = 0.0104, i.e. the fine-mesh element size


def rasterise(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    """Signed-distance image of the part + normalised node coordinates.

    Only needed by the Cartesian-grid baseline; the graph models never see it.
    """
    from .geometry import HEIGHT, WIDTH

    xs = (np.arange(GRID_W) + 0.5) / GRID_W * WIDTH
    ys = (np.arange(GRID_H) + 0.5) / GRID_H * HEIGHT
    gx, gy = np.meshgrid(xs, ys)
    # Negative inside the material: outside the plate or inside a hole is positive.
    d_rect = np.maximum.reduce([-gx, gx - WIDTH, -gy, gy - HEIGHT])
    phi = d_rect
    for cx, cy, r in sample["holes"]:
        phi = np.maximum(phi, r - np.hypot(gx - cx, gy - cy))
    angle = float(sample["load_angle"])
    grid = np.stack(
        [
            np.tanh(phi / 0.05),
            (phi < 0).astype(np.float32),
            np.full_like(gx, np.cos(angle)),
            np.full_like(gx, np.sin(angle)),
        ]
    ).astype(np.float32)
    uv = np.stack(
        [sample["nodes"][:, 0] / WIDTH * 2 - 1, sample["nodes"][:, 1] / HEIGHT * 2 - 1], axis=1
    ).astype(np.float32)
    return grid, uv


@dataclass
class Graph:
    """One mesh as a graph.  Batches are built by concatenation (see `collate`)."""

    x: torch.Tensor  # (N, F) node features
    edge_index: torch.Tensor  # (2, E) directed, both orientations present
    edge_attr: torch.Tensor  # (E, 3)
    y: torch.Tensor  # (N,) normalised target
    node_batch: torch.Tensor  # (N,) sample index within the batch
    n_graphs: int = 1
    meta: dict = field(default_factory=dict)
    grid: torch.Tensor | None = None  # (B, 4, H, W), grid baseline only
    uv: torch.Tensor | None = None  # (N, 2) in [-1, 1]

    def to(self, device) -> "Graph":
        move = lambda t: None if t is None else t.to(device)
        return Graph(
            self.x.to(device),
            self.edge_index.to(device),
            self.edge_attr.to(device),
            self.y.to(device),
            self.node_batch.to(device),
            self.n_graphs,
            self.meta,
            move(self.grid),
            move(self.uv),
        )


def build_graph(
    sample: dict,
    use_coarse_solution: bool,
    use_position: bool,
    target_mean: float,
    target_std: float,
    with_grid: bool = False,
) -> Graph:
    nodes = sample["nodes"].astype(np.float32)
    tris = sample["tris"].astype(np.int64)

    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    ei = np.concatenate([e, e[:, ::-1]]).T  # both orientations

    d = nodes[ei[1]] - nodes[ei[0]]
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    edge_attr = np.concatenate([d, dist], axis=1) / LENGTH_SCALE

    onehot = np.zeros((len(nodes), N_NODE_TYPES), dtype=np.float32)
    onehot[np.arange(len(nodes)), sample["node_type"].astype(np.int64)] = 1.0
    angle = float(sample["load_angle"])
    load = np.tile(np.array([np.cos(angle), np.sin(angle)], dtype=np.float32), (len(nodes), 1))
    feats = [onehot, load, (sample["local_size"] / LENGTH_SCALE)[:, None].astype(np.float32)]
    if use_position:
        feats.append(nodes)
    if use_coarse_solution:
        feats.append(sample["u_coarse"] / max(np.abs(sample["u_coarse"]).max(), 1e-9))
        feats.append(((sample["vm_coarse"] - target_mean) / target_std)[:, None])

    x = np.concatenate(feats, axis=1).astype(np.float32)
    y = ((sample["vm_target"] - target_mean) / target_std).astype(np.float32)
    grid, uv = rasterise(sample) if with_grid else (None, None)
    return Graph(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(ei.astype(np.int64)),
        edge_attr=torch.from_numpy(edge_attr.astype(np.float32)),
        y=torch.from_numpy(y),
        node_batch=torch.zeros(len(nodes), dtype=torch.long),
        meta={k: float(sample[k]) for k in ("max_vm_fine", "max_vm_coarse", "max_vm_target", "h_coarse")},
        grid=None if grid is None else torch.from_numpy(grid).unsqueeze(0),
        uv=None if uv is None else torch.from_numpy(uv),
    )


def collate(graphs: list[Graph]) -> Graph:
    offsets = np.cumsum([0] + [g.x.shape[0] for g in graphs])
    has_grid = graphs[0].grid is not None
    return Graph(
        x=torch.cat([g.x for g in graphs]),
        edge_index=torch.cat([g.edge_index + offsets[i] for i, g in enumerate(graphs)], dim=1),
        edge_attr=torch.cat([g.edge_attr for g in graphs]),
        y=torch.cat([g.y for g in graphs]),
        node_batch=torch.cat([torch.full((g.x.shape[0],), i, dtype=torch.long) for i, g in enumerate(graphs)]),
        n_graphs=len(graphs),
        meta={"per_graph": [g.meta for g in graphs]},
        grid=torch.cat([g.grid for g in graphs]) if has_grid else None,
        uv=torch.cat([g.uv for g in graphs]) if has_grid else None,
    )


class GraphDataset:
    """Holds one split in memory as a list of `Graph`."""

    def __init__(
        self,
        path: Path,
        use_coarse_solution: bool = False,
        use_position: bool = True,
        stats: tuple[float, float] | None = None,
        limit: int | None = None,
        with_grid: bool = False,
    ):
        blob = np.load(path)
        self.n = len(blob["load_angle"]) if limit is None else min(limit, len(blob["load_angle"]))
        self.samples = [unpack(blob, i) for i in range(self.n)]
        if stats is None:
            all_t = np.concatenate([s["vm_target"] for s in self.samples])
            stats = (float(all_t.mean()), float(all_t.std()))
        self.mean, self.std = stats
        self.graphs = [
            build_graph(s, use_coarse_solution, use_position, self.mean, self.std, with_grid)
            for s in self.samples
        ]
        self.n_features = self.graphs[0].x.shape[1]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, i: int) -> Graph:
        return self.graphs[i]

    def denormalise(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean
