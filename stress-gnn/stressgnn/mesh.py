"""Triangular meshing of a perforated plate, via the gmsh OCC kernel.

Each design is meshed at several target element sizes.  That is not a detail: the
central experiment of this repo is "solve on the coarse mesh, predict the fine-mesh
answer", so coarse and fine discretisations of the *same* geometry are first-class
objects here.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass

import numpy as np

from .geometry import HEIGHT, WIDTH, Design

_GMSH_READY = False

# Node tags, used as one-hot input features by the networks.
NODE_INTERIOR, NODE_CLAMPED, NODE_LOADED, NODE_HOLE, NODE_FREE = range(5)
N_NODE_TYPES = 5


def _ensure_gmsh():
    global _GMSH_READY
    import gmsh

    if not _GMSH_READY:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        atexit.register(gmsh.finalize)
        _GMSH_READY = True
    return gmsh


@dataclass
class Mesh:
    """A P1 triangulation plus everything the FEM and the graph builder need."""

    nodes: np.ndarray  # (N, 2)
    tris: np.ndarray  # (M, 3) int
    node_type: np.ndarray  # (N,) int, see NODE_* constants
    boundary_edges: np.ndarray  # (E, 2) int, edges on the outer/inner boundary
    h_target: float

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_tris(self) -> int:
        return len(self.tris)

    def edges(self) -> np.ndarray:
        """Unique undirected mesh edges, shape (E, 2)."""
        e = np.concatenate([self.tris[:, [0, 1]], self.tris[:, [1, 2]], self.tris[:, [2, 0]]])
        e = np.sort(e, axis=1)
        return np.unique(e, axis=0)

    def local_size(self) -> np.ndarray:
        """Mean incident edge length per node -- the network's resolution cue."""
        e = self.edges()
        lengths = np.linalg.norm(self.nodes[e[:, 0]] - self.nodes[e[:, 1]], axis=1)
        acc = np.zeros(self.n_nodes)
        cnt = np.zeros(self.n_nodes)
        for col in (0, 1):
            np.add.at(acc, e[:, col], lengths)
            np.add.at(cnt, e[:, col], 1.0)
        return acc / np.maximum(cnt, 1.0)


def _classify_nodes(nodes: np.ndarray, design: Design, tol: float) -> np.ndarray:
    kind = np.full(len(nodes), NODE_INTERIOR, dtype=np.int64)
    x, y = nodes[:, 0], nodes[:, 1]
    on_free = (np.abs(y) < tol) | (np.abs(y - HEIGHT) < tol)
    kind[on_free] = NODE_FREE
    for cx, cy, r in design.holes:
        on_hole = np.abs(np.hypot(x - cx, y - cy) - r) < tol
        kind[on_hole] = NODE_HOLE
    kind[np.abs(x - WIDTH) < tol] = NODE_LOADED
    kind[np.abs(x) < tol] = NODE_CLAMPED  # clamping wins over every other tag
    return kind


def _boundary_edges(tris: np.ndarray) -> np.ndarray:
    """Edges belonging to exactly one triangle."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq[counts == 1]


def build_mesh(design: Design, h: float, algorithm: int = 5) -> Mesh:
    """Mesh one design with target element size `h`."""
    gmsh = _ensure_gmsh()
    gmsh.model.add("plate")
    try:
        rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, WIDTH, HEIGHT)
        holes = [gmsh.model.occ.addDisk(cx, cy, 0.0, r, r) for cx, cy, r in design.holes]
        if holes:
            gmsh.model.occ.cut([(2, rect)], [(2, h_) for h_ in holes])
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        gmsh.model.mesh.generate(2)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = coords.reshape(-1, 3)[:, :2]
        order = np.argsort(tags)
        tags, coords = tags[order], coords[order]
        remap = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        remap[tags] = np.arange(len(tags))

        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        tri_block = [i for i, t in enumerate(etypes) if t == 2]
        if not tri_block:
            raise RuntimeError("gmsh produced no triangles")
        tris = remap[enodes[tri_block[0]].reshape(-1, 3).astype(np.int64)]
    finally:
        gmsh.model.remove()

    return Mesh(
        nodes=coords,
        tris=tris,
        node_type=_classify_nodes(coords, design, tol=0.25 * h),
        boundary_edges=_boundary_edges(tris),
        h_target=h,
    )


def loaded_edges(mesh: Mesh) -> np.ndarray:
    """Boundary edges lying on the right-hand edge, used for the traction load."""
    x = mesh.nodes[:, 0]
    tol = 0.25 * mesh.h_target
    on_right = np.abs(x - WIDTH) < tol
    e = mesh.boundary_edges
    return e[on_right[e[:, 0]] & on_right[e[:, 1]]]
