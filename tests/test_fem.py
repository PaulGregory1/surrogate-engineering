import numpy as np
import pytest
import scipy.sparse.linalg as spla

from stressgnn.fem import (
    assemble_stiffness,
    element_operators,
    load_vector,
    plane_stress_D,
    solve,
    von_mises,
)
from stressgnn.geometry import HEIGHT, WIDTH, Design
from stressgnn.mesh import build_mesh


@pytest.fixture(scope="module")
def plain_mesh():
    return build_mesh(Design(), h=0.08)


@pytest.fixture(scope="module")
def holed():
    design = Design(holes=np.array([[0.7, 0.5, 0.15], [1.4, 0.35, 0.12]]), load_angle=-np.pi / 3)
    return design, build_mesh(design, h=0.05)


def test_element_areas_sum_to_the_material_area(holed):
    design, mesh = holed
    _, area = element_operators(mesh)
    expected = WIDTH * HEIGHT - np.pi * (design.holes[:, 2] ** 2).sum()
    assert abs(area.sum() - expected) / expected < 2e-3  # polygonal holes, so not exact


def test_patch_test_uniform_tension(plain_mesh):
    """The standard FEM verification: a constant-stress state must be reproduced exactly.

    Roller support on the left edge (u_x = 0) plus one pinned node, so the plate is
    free to contract laterally and the exact solution really is uniform.
    """
    mesh = plain_mesh
    D = plane_stress_D(E=1.0, nu=0.3)
    B, area = element_operators(mesh)
    K = assemble_stiffness(mesh, B, area, D)

    traction = 0.37
    design = Design(load_angle=0.0, load_magnitude=traction)
    f = load_vector(mesh, design)

    left = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    corner = left[np.argmin(mesh.nodes[left, 1])]
    fixed = np.concatenate([2 * left, [2 * corner + 1]])
    free = np.setdiff1d(np.arange(2 * mesh.n_nodes), fixed)

    u = np.zeros(2 * mesh.n_nodes)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    ue = u[np.stack([2 * mesh.tris, 2 * mesh.tris + 1], axis=2).reshape(-1, 6)]
    sigma = np.einsum("kl,elj,ej->ek", D, B, ue)

    assert np.abs(sigma[:, 0] - traction).max() < 1e-9
    assert np.abs(sigma[:, 1]).max() < 1e-9
    assert np.abs(sigma[:, 2]).max() < 1e-9


def test_energy_balance(holed):
    """External work must equal twice the stored strain energy."""
    design, mesh = holed
    sol = solve(mesh, design)
    D = plane_stress_D()
    B, area = element_operators(mesh)
    ue = sol.u.ravel()[np.stack([2 * mesh.tris, 2 * mesh.tris + 1], axis=2).reshape(-1, 6)]
    strain = np.einsum("ekj,ej->ek", B, ue)
    energy = float(np.einsum("ek,kl,el,e->", strain, D, strain, area))
    assert abs(sol.compliance - energy) / abs(sol.compliance) < 1e-9


def test_linearity(holed):
    design, mesh = holed
    a = solve(mesh, design)
    doubled = Design(holes=design.holes, load_angle=design.load_angle, load_magnitude=2.0)
    b = solve(mesh, doubled)
    assert np.allclose(b.u, 2 * a.u, rtol=1e-8, atol=1e-12)
    assert abs(b.max_vm - 2 * a.max_vm) / a.max_vm < 1e-8


def test_zero_load_gives_zero_displacement(holed):
    design, mesh = holed
    quiet = Design(holes=design.holes, load_angle=0.0, load_magnitude=0.0)
    sol = solve(mesh, quiet)
    assert np.abs(sol.u).max() < 1e-14


def test_clamped_nodes_do_not_move(holed):
    from stressgnn.mesh import NODE_CLAMPED

    design, mesh = holed
    sol = solve(mesh, design)
    clamped = mesh.node_type == NODE_CLAMPED
    assert clamped.sum() > 0
    assert np.abs(sol.u[clamped]).max() < 1e-14


def test_von_mises_of_pure_shear():
    sigma = np.array([[0.0, 0.0, 1.0]])
    assert abs(float(von_mises(sigma)[0]) - np.sqrt(3.0)) < 1e-12


def test_refinement_reduces_the_peak_stress_error(holed):
    """A coarse mesh must under-predict the stress concentration: the premise of the repo."""
    design, _ = holed
    peaks = [solve(build_mesh(design, h), design).max_vm for h in (0.06, 0.03, 0.015)]
    assert peaks[0] < peaks[1] < peaks[2]
