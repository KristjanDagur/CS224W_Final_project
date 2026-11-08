# Port numbering, re-implemented (vectorized) from IBM Multi-GNN `data_util.py`
# (`GraphData.add_ports`, `ports`, `to_adj_nodes_with_times`).
# Original: https://github.com/IBM/Multi-GNN, Copyright IBM Corp., Apache License 2.0.
"""Port numbers for directed multigraphs (Egressy et al., 2024; Sato et al., 2019).

Every node numbers its counterparties separately for incoming and outgoing edges:

* ``in_port[e]`` for an edge ``e = (u, v)`` is the rank of ``u`` among the distinct
  in-neighbors of ``v``.
* ``out_port[e]`` is the rank of ``v`` among the distinct out-neighbors of ``u``.

Neighbors are ranked by the timestamp of their *first* edge with the node, starting at 0.
All parallel edges between the same ordered pair therefore share one port number. This
matches IBM's implementation. IBM breaks ties in first-time with an unstable
``argsort``; here ties are broken by neighbor ID so the result is deterministic.

IBM's version is a Python loop over all edges (slow on AMLworld's 5M edges). This one
is O(m log m) numpy and takes about a second on AMLworld.
"""

from __future__ import annotations

import numpy as np
import torch


def _rank_neighbors(owner: np.ndarray, nbr: np.ndarray, t: np.ndarray, num_nodes: int) -> np.ndarray:
    """Rank each edge's neighbor among the distinct neighbors of the edge's owner node.

    Args:
        owner: (m,) node that assigns the port (e.g. the head ``v`` for in-ports).
        nbr: (m,) the counterparty being numbered (e.g. the tail ``u`` for in-ports).
        t: (m,) edge timestamps; neighbors are ordered by their earliest timestamp.
        num_nodes: number of nodes, used to encode (owner, nbr) pairs as one int64 key.

    Returns:
        (m,) int64 port number of each edge.
    """
    key = owner.astype(np.int64) * num_nodes + nbr.astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    first_t = np.full(uniq.shape[0], np.inf)
    np.minimum.at(first_t, inv, t.astype(np.float64))
    pair_owner, pair_nbr = uniq // num_nodes, uniq % num_nodes
    # Sort pairs by owner, then by first-contact time, then by neighbor ID (tie-break).
    order = np.lexsort((pair_nbr, first_t, pair_owner))
    sorted_owner = pair_owner[order]
    group_start = np.r_[0, np.flatnonzero(np.diff(sorted_owner)) + 1]
    group_len = np.diff(np.r_[group_start, sorted_owner.shape[0]])
    rank_sorted = np.arange(sorted_owner.shape[0]) - np.repeat(group_start, group_len)
    pair_port = np.empty_like(rank_sorted)
    pair_port[order] = rank_sorted
    return pair_port[inv]


def compute_ports(edge_index: torch.Tensor, edge_time: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute (in_port, out_port) for every edge of a directed multigraph.

    Args:
        edge_index: (2, m) long tensor of ``(source, target)`` pairs.
        edge_time: (m,) timestamps used to order counterparties by first contact.
        num_nodes: number of nodes in the graph.

    Returns:
        (m, 2) float tensor ``[in_port, out_port]``. This is the column order IBM appends to
        ``edge_attr``. For reverse-direction message passing these two columns are swapped.
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    t = edge_time.cpu().numpy()
    in_port = _rank_neighbors(owner=dst, nbr=src, t=t, num_nodes=num_nodes)
    out_port = _rank_neighbors(owner=src, nbr=dst, t=t, num_nodes=num_nodes)
    return torch.from_numpy(np.stack([in_port, out_port], axis=1)).float()
