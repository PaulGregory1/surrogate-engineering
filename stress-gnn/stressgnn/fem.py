"""Plane-stress linear elasticity on P1 (constant-strain) triangles.

Vectorised assembly, direct sparse solve.  This is the reference the surrogate is
measured against, and the thing whose cost we are trying to avoid inside a design
loop: one solve is cheap, but a shape study is thousands of solves, each preceded by
a remesh.

Conventions: unit thickness, E and nu passed in, displacements ordered
[u_x0, u_y0, u_x1, u_y1, ...].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import Design
from .mesh import NODE_CLAMPED, Mesh, loaded_edges

E_DEFAULT: float = 1.0
NU_DEFAULT: float = 0.3


def plane_stress_D(E: float = E_DEFAULT, nu: float = NU_DEFAULT) -> np.ndarray:
    return E / (1 - nu**2) * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1 - nu) / 2]])


def element_operators(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Strain-displacement matrices B (M, 3, 6) and triangle areas (M,)."""
    p = mesh.nodes[mesh.tris]  # (M, 3, 2)
    x, y = p[:, :, 0], p[:, :, 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    two_a = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    area = 0.5 * np.abs(two_a)

    B = np.zeros((len(mesh.tris), 3, 6))
    inv = 1.0 / two_a[:, None]
    for k in range(3):
        B[:, 0, 2 * k] = b[:, k] * inv[:, 0]
        B[:, 1, 2 * k + 1] = c[:, k] * inv[:, 0]
        B[:, 2, 2 * k] = c[:, k] * inv[:, 0]
        B[:, 2, 2 * k + 1] = b[:, k] * inv[:, 0]
    return B, area


def assemble_stiffness(mesh: Mesh, B: np.ndarray, area: np.ndarray, D: np.ndarray) -> sp.csr_matrix:
    ke = np.einsum("eki,kl,elj,e->eij", B, D, B, area)  # (M, 6, 6)
    dofs = np.empty((len(mesh.tris), 6), dtype=np.int64)
    dofs[:, 0::2] = 2 * mesh.tris
    dofs[:, 1::2] = 2 * mesh.tris + 1
    rows = np.repeat(dofs, 6, axis=1).ravel()
    cols = np.tile(dofs, (1, 6)).ravel()
    n = 2 * mesh.n_nodes
    return sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def load_vector(mesh: Mesh, design: Design) -> np.ndarray:
    """Consistent nodal loads from a uniform traction on the right-hand edge."""
    f = np.zeros(2 * mesh.n_nodes)
    edges = loaded_edges(mesh)
    lengths = np.linalg.norm(mesh.nodes[edges[:, 0]] - mesh.nodes[edges[:, 1]], axis=1)
    t = design.traction
    for col in (0, 1):
        np.add.at(f, 2 * edges[:, col], 0.5 * lengths * t[0])
        np.add.at(f, 2 * edges[:, col] + 1, 0.5 * lengths * t[1])
    return f


def von_mises(sigma: np.ndarray) -> np.ndarray:
    """Plane-stress von Mises from (sx, sy, txy)."""
    sx, sy, txy = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    return np.sqrt(np.maximum(sx**2 - sx * sy + sy**2 + 3 * txy**2, 0.0))


def elements_to_nodes(mesh: Mesh, values: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Area-weighted averaging of a per-element field onto the nodes."""
    acc = np.zeros(mesh.n_nodes)
    wsum = np.zeros(mesh.n_nodes)
    for k in range(3):
        np.add.at(acc, mesh.tris[:, k], values * area)
        np.add.at(wsum, mesh.tris[:, k], area)
    return acc / np.maximum(wsum, 1e-30)


@dataclass
class FEMSolution:
    u: np.ndarray  # (N, 2) displacements
    sigma_elem: np.ndarray  # (M, 3)
    vm_elem: np.ndarray  # (M,)
    vm_node: np.ndarray  # (N,)
    compliance: float
    max_vm: float


def solve(mesh: Mesh, design: Design, E: float = E_DEFAULT, nu: float = NU_DEFAULT) -> FEMSolution:
    D = plane_stress_D(E, nu)
    B, area = element_operators(mesh)
    K = assemble_stiffness(mesh, B, area, D)
    f = load_vector(mesh, design)

    clamped = np.where(mesh.node_type == NODE_CLAMPED)[0]
    fixed = np.concatenate([2 * clamped, 2 * clamped + 1])
    free = np.setdiff1d(np.arange(2 * mesh.n_nodes), fixed)

    u = np.zeros(2 * mesh.n_nodes)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])

    ue = u[np.stack([2 * mesh.tris, 2 * mesh.tris + 1], axis=2).reshape(-1, 6)]
    sigma = np.einsum("kl,elj,ej->ek", D, B, ue)
    vm = von_mises(sigma)
    return FEMSolution(
        u=u.reshape(-1, 2),
        sigma_elem=sigma,
        vm_elem=vm,
        vm_node=elements_to_nodes(mesh, vm, area),
        compliance=float(f @ u),
        max_vm=float(vm.max()),
    )


def interpolate(mesh: Mesh, values: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Barycentric interpolation of a nodal field onto arbitrary points.

    Uses the actual triangulation (not a Delaunay of the point cloud) so that values
    are never interpolated *across* a hole.  Points that fall marginally outside the
    fine mesh -- the two discretisations approximate the curved hole boundaries by
    different polygons -- fall back to the nearest node.
    """
    from matplotlib.tri import LinearTriInterpolator, Triangulation

    tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris)
    out = np.asarray(LinearTriInterpolator(tri, values)(points[:, 0], points[:, 1]))
    missing = ~np.isfinite(out)
    if missing.any():
        from scipy.spatial import cKDTree

        _, idx = cKDTree(mesh.nodes).query(points[missing])
        out[missing] = values[idx]
    return np.asarray(out, dtype=np.float64)
