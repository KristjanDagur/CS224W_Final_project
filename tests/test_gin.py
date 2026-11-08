"""Model and sampling tests: subgraph forwards must equal full-graph forwards (eval mode)."""

import numpy as np
import pytest
import torch
from torch_geometric.loader import NeighborLoader

from amlinterp.data.synthetic import CirculantConfig, generate_circulant_graph
from amlinterp.data.transforms import FWD, NODE, REV, add_ego_ids, build_ladder_graph, model_inputs
from amlinterp.models.gin import RUNGS, GINLadder

L = 4
N = 300


def _graph(rung):
    ei, t = generate_circulant_graph(CirculantConfig(n=N, d=6.0, r=5.0), np.random.default_rng(0))
    ei, t = torch.from_numpy(ei), torch.from_numpy(t).float()
    return build_ladder_graph(torch.ones(N, 1), ei, torch.ones(ei.shape[1], 1), t, rung, N)


def _model(g, rung):
    torch.manual_seed(0)
    m = GINLadder(node_in=1 + int(rung.ego), edge_in=g[FWD].edge_attr.shape[1], out_dim=3, hidden=16,
                  num_layers=L, reverse_mp=rung.reverse_mp, readout="node")
    # Non-trivial BatchNorm running stats, so eval mode is not a near-identity.
    for bn in m.norms:
        bn.running_mean.uniform_(-0.5, 0.5)
        bn.running_var.uniform_(0.5, 2.0)
    return m.eval()


@pytest.mark.parametrize("rung_name", ["R0", "R1", "R2", "R3"])
def test_disjoint_subgraph_forward_equals_full_graph(rung_name):
    rung = RUNGS[rung_name]
    g = _graph(rung)
    model = _model(g, rung)
    seeds = torch.tensor([3, 50, 51, 299])
    loader = NeighborLoader(g, num_neighbors={et: [-1] * L for et in g.edge_types}, input_nodes=(NODE, seeds),
                            batch_size=len(seeds), shuffle=False, disjoint=True)
    batch = next(iter(loader))
    if rung.ego:
        batch = add_ego_ids(batch, len(seeds))
    with torch.no_grad():
        sub = model(**model_inputs(batch), target=torch.arange(len(seeds)))
        for i, s in enumerate(seeds.tolist()):
            full_g = g.clone()
            if rung.ego:  # exact ego: mark only this seed in the full graph
                full_g[NODE].x = torch.cat([full_g[NODE].x, (torch.arange(N) == s).float()[:, None]], 1)
            full = model(**model_inputs(full_g), target=torch.tensor([s]))
            torch.testing.assert_close(sub[i : i + 1], full, rtol=1e-4, atol=1e-5)


def test_reverse_edges_swap_port_columns():
    g = _graph(RUNGS["R2"])
    fwd, rev = g[FWD].edge_attr, g[REV].edge_attr
    assert torch.equal(g[REV].edge_index, g[FWD].edge_index.flip(0))
    assert torch.equal(rev[:, -2], fwd[:, -1]) and torch.equal(rev[:, -1], fwd[:, -2])
    assert torch.equal(rev[:, :-2], fwd[:, :-2])


def test_rung_input_sizes():
    sizes = {r: (_graph(RUNGS[r])[FWD].edge_attr.shape[1], REV in _graph(RUNGS[r]).edge_types) for r in RUNGS}
    assert sizes == {"R0": (1, False), "R1": (1, True), "R2": (3, True), "R3": (3, True)}


def test_edge_readout_shape_and_hidden_states():
    rung = RUNGS["R2"]
    g = _graph(rung)
    torch.manual_seed(0)
    m = GINLadder(node_in=1, edge_in=3, out_dim=2, hidden=16, num_layers=2, reverse_mp=True, readout="edge").eval()
    out, hidden = m(**model_inputs(g), target=torch.arange(10), return_hidden=True)
    assert out.shape == (10, 2) and len(hidden) == 3 and hidden[-1].shape == (N, 16)
