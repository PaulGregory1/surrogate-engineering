import numpy as np
import pytest
import torch

from stressgnn.dataset import build_graph, collate, rasterise
from stressgnn.fem import interpolate, solve
from stressgnn.geometry import Design, is_valid, sample_designs
from stressgnn.mesh import NODE_CLAMPED, NODE_LOADED, build_mesh
from stressgnn.models import build


@pytest.fixture(scope="module")
def sample():
    design = Design(holes=np.array([[0.7, 0.5, 0.15], [1.4, 0.35, 0.12]]), load_angle=-np.pi / 3)
    mesh = build_mesh(design, h=0.06)
    sol = solve(mesh, design)
    return {
        "nodes": mesh.nodes.astype(np.float32),
        "tris": mesh.tris.astype(np.int32),
        "node_type": mesh.node_type.astype(np.int8),
        "local_size": mesh.local_size().astype(np.float32),
        "vm_coarse": sol.vm_node.astype(np.float32),
        "u_coarse": sol.u.astype(np.float32),
        "vm_target": sol.vm_node.astype(np.float32),
        "holes": design.holes.astype(np.float32),
        "load_angle": np.float32(design.load_angle),
        "max_vm_fine": np.float32(sol.max_vm),
        "max_vm_coarse": np.float32(sol.max_vm),
        "max_vm_target": np.float32(sol.max_vm),
        "h_coarse": np.float32(0.06),
    }


def test_mesh_node_tags_are_populated(sample):
    types = sample["node_type"]
    assert (types == NODE_CLAMPED).sum() > 0
    assert (types == NODE_LOADED).sum() > 0


def test_graph_edges_are_bidirectional(sample):
    g = build_graph(sample, False, True, 0.0, 1.0)
    src, dst = g.edge_index
    forward = {(int(a), int(b)) for a, b in zip(src, dst)}
    assert all((b, a) in forward for a, b in forward)
    assert torch.isfinite(g.x).all() and torch.isfinite(g.edge_attr).all()


def test_collate_offsets_keep_graphs_disconnected(sample):
    g = build_graph(sample, False, True, 0.0, 1.0)
    batch = collate([g, g, g])
    n = g.x.shape[0]
    assert batch.x.shape[0] == 3 * n
    assert batch.n_graphs == 3
    # No edge may cross a graph boundary.
    assert torch.equal(batch.node_batch[batch.edge_index[0]], batch.node_batch[batch.edge_index[1]])


def test_coarse_solution_features_change_the_input_width(sample):
    a = build_graph(sample, False, True, 0.0, 1.0)
    b = build_graph(sample, True, True, 0.0, 1.0)
    assert b.x.shape[1] == a.x.shape[1] + 3


def test_models_predict_one_value_per_node(sample):
    g = build_graph(sample, False, True, 0.0, 1.0, with_grid=True)
    batch = collate([g, g])
    gnn = build("gnn", n_node_features=g.x.shape[1], hidden=16, n_message_passing=3, use_global=True)
    cnn = build("grid-cnn", in_ch=4, base=8, depth=2)
    assert gnn(batch).shape == (batch.x.shape[0],)
    assert cnn(batch).shape == (batch.x.shape[0],)


def test_zero_message_passing_is_a_pointwise_mlp(sample):
    g = build_graph(sample, False, True, 0.0, 1.0)
    model = build("gnn", n_node_features=g.x.shape[1], hidden=16, n_message_passing=0)
    single = model(collate([g]))
    # Duplicating a node must not change any prediction if there is no communication.
    assert single.shape == (g.x.shape[0],)
    perm = torch.randperm(g.x.shape[0])
    g2 = build_graph(sample, False, True, 0.0, 1.0)
    g2.x = g.x[perm]
    assert torch.allclose(model(collate([g2])), single[perm], atol=1e-5)


def test_rasterise_marks_holes_as_void(sample):
    grid, uv = rasterise(sample)
    assert grid.shape[0] == 4
    assert uv.min() >= -1.0 and uv.max() <= 1.0
    # A point at the centre of the first hole must be outside the material.
    from stressgnn.dataset import GRID_H, GRID_W
    from stressgnn.geometry import HEIGHT, WIDTH

    cx, cy, _ = sample["holes"][0]
    i = int(cy / HEIGHT * GRID_H)
    j = int(cx / WIDTH * GRID_W)
    assert grid[1, i, j] == 0.0


def test_interpolation_is_exact_for_a_linear_field():
    design = Design(holes=np.array([[0.8, 0.5, 0.15]]))
    fine = build_mesh(design, 0.02)
    coarse = build_mesh(design, 0.07)
    f = lambda p: 1.5 + 2.0 * p[:, 0] - 0.7 * p[:, 1]
    got = interpolate(fine, f(fine.nodes), coarse.nodes)
    assert np.abs(got - f(coarse.nodes)).max() < 1e-6


def test_sampled_designs_respect_the_ligament_constraints():
    designs = sample_designs(np.random.default_rng(0), 40)
    assert all(is_valid(d.holes) for d in designs)
