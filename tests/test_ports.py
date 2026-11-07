"""Check the vectorized port numbering against a re-implementation of IBM Multi-GNN's loop."""

import numpy as np
import torch

from amlinterp.data.ports import compute_ports


def _ibm_reference_ports(edge_index: torch.Tensor, edge_time: torch.Tensor) -> torch.Tensor:
    """Loop version following IBM Multi-GNN `data_util.ports` (Apache-2.0), ties excluded."""

    def ports(ei, owner_row):
        adj = {}
        for (a, b), t in zip(ei.T.tolist(), edge_time.tolist()):
            owner, nbr = (b, a) if owner_row == 1 else (a, b)
            adj.setdefault(owner, []).append((nbr, t))
        port_of = {}
        for owner, nbs in adj.items():
            seen = []
            for nbr, _ in sorted(nbs, key=lambda x: x[1]):
                if nbr not in seen:
                    seen.append(nbr)
            for i, nbr in enumerate(seen):
                port_of[(owner, nbr)] = i
        return [port_of[(b, a) if owner_row == 1 else (a, b)] for a, b in ei.T.tolist()]

    return torch.tensor([ports(edge_index, 1), ports(edge_index, 0)], dtype=torch.float).T


def test_ports_match_ibm_loop():
    rng = np.random.default_rng(0)
    n, m = 30, 400  # dense enough to have many parallel edges
    ei = torch.from_numpy(rng.integers(0, n, size=(2, m)))
    t = torch.from_numpy(rng.permutation(m).astype(np.float32))  # distinct times: no ties
    torch.testing.assert_close(compute_ports(ei, t, n), _ibm_reference_ports(ei, t))


def test_parallel_edges_share_a_port():
    # 0->1 at t=5 and t=1; 2->1 at t=3. Node 1's in-neighbors by first contact: 0 (t=1), 2 (t=3).
    ei = torch.tensor([[0, 0, 2], [1, 1, 1]])
    t = torch.tensor([5.0, 1.0, 3.0])
    p = compute_ports(ei, t, 3)
    assert p[:, 0].tolist() == [0.0, 0.0, 1.0]  # in-ports at node 1
    assert p[:, 1].tolist() == [0.0, 0.0, 0.0]  # each sender has a single out-neighbor
