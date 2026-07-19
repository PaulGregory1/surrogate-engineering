"""MeshGraphNet-style encode-process-decode network.

Message passing happens on the mesh edges themselves, so the model is indifferent to
node count and connectivity: the same weights run on a 700-node and a 3000-node mesh.

Two knobs exist because they are the two things that actually decide whether this
works on a stress field:

`n_message_passing`
    Information travels one edge per block.  A load applied on the right edge has to
    reach the far side of the plate, which is ~40 edges away on a coarse mesh, so a
    plain 5-block network structurally *cannot* see it.  Sweeping this exposes the
    limitation instead of hiding it.

`use_global`
    A global latent, pooled from and broadcast to every node in each block, gives an
    O(1)-hop shortcut for exactly that long-range coupling -- much cheaper than
    stacking 40 blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(sizes: list[int], layer_norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    if layer_norm:
        layers.append(nn.LayerNorm(sizes[-1]))
    return nn.Sequential(*layers)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, src.shape[1], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt = torch.zeros(n, 1, device=src.device, dtype=src.dtype)
    cnt.index_add_(0, index, torch.ones(src.shape[0], 1, device=src.device, dtype=src.dtype))
    return out / cnt.clamp_min(1.0)


class MessagePassingBlock(nn.Module):
    def __init__(self, hidden: int, use_global: bool):
        super().__init__()
        self.use_global = use_global
        self.edge_mlp = mlp([3 * hidden, hidden, hidden])
        node_in = 2 * hidden + (hidden if use_global else 0)
        self.node_mlp = mlp([node_in, hidden, hidden])
        if use_global:
            self.global_mlp = mlp([2 * hidden, hidden, hidden])

    def forward(self, v, e, edge_index, g, node_batch, n_graphs):
        src, dst = edge_index[0], edge_index[1]
        e = e + self.edge_mlp(torch.cat([e, v[src], v[dst]], dim=-1))

        agg = torch.zeros_like(v)
        agg.index_add_(0, dst, e)
        parts = [v, agg]
        if self.use_global:
            parts.append(g[node_batch])
        v = v + self.node_mlp(torch.cat(parts, dim=-1))

        if self.use_global:
            g = g + self.global_mlp(torch.cat([g, scatter_mean(v, node_batch, n_graphs)], dim=-1))
        return v, e, g


class MeshGraphNet(nn.Module):
    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int = 3,
        hidden: int = 64,
        n_message_passing: int = 10,
        use_global: bool = False,
        out_dim: int = 1,
    ):
        super().__init__()
        self.use_global = use_global
        self.node_encoder = mlp([n_node_features, hidden, hidden])
        self.edge_encoder = mlp([n_edge_features, hidden, hidden])
        if use_global:
            self.global_init = nn.Parameter(torch.zeros(1, hidden))
        self.blocks = nn.ModuleList([MessagePassingBlock(hidden, use_global) for _ in range(n_message_passing)])
        self.decoder = mlp([hidden, hidden, out_dim], layer_norm=False)

    def forward(self, graph) -> torch.Tensor:
        v = self.node_encoder(graph.x)
        e = self.edge_encoder(graph.edge_attr)
        g = self.global_init.expand(graph.n_graphs, -1) if self.use_global else None
        for block in self.blocks:
            v, e, g = block(v, e, graph.edge_index, g, graph.node_batch, graph.n_graphs)
        return self.decoder(v).squeeze(-1)
