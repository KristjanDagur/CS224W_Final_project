"""Turn a base graph into the input for one ladder rung, and add ego IDs to sampled batches.

Building a rung's graph:

1. **Ports.** If ``rung.ports``, append ``[in_port, out_port]`` to the edge features.
2. **Reverse edges.** If ``rung.reverse_mp``, add the edge type ``("node", "rev_to", "node")``.
   It carries the flipped ``edge_index`` and the same features, with the two port columns
   swapped: seen from the reversed edge, the in-port and out-port trade places.
3. **Ego IDs.** If ``rung.ego``, append a ``{0, 1}`` column to the node features *per batch*,
   after sampling, because the marked nodes depend on the batch (``add_ego_ids``).

The graph is a ``HeteroData`` with one node type, ``"node"``, for every rung, so that
all rungs go through the same loader code.

Ego modes (AGENTS.md technical fact 1):

* ``exact``: each target sits in its own disjoint subgraph (``disjoint=True`` in the
  loader), and only that target is marked.
* ``batch``: one shared subgraph per batch, and every seed in the batch is marked. This
  is IBM's ``AddEgoIds`` behavior.

Either way, the seeds are the first ``batch_size`` nodes of the sampled batch, so the
marking code is the same and only the loader's ``disjoint`` flag differs.
"""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from amlinterp.data.ports import compute_ports
from amlinterp.models.gin import Rung

NODE = "node"
FWD = (NODE, "to", NODE)
REV = (NODE, "rev_to", NODE)
EGO_MODES = ("exact", "batch")


def build_ladder_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_time: torch.Tensor,
    rung: Rung,
    num_nodes: int,
    ports: torch.Tensor | None = None,
    edge_transform=None,
) -> HeteroData:
    """Build the ``HeteroData`` graph that a rung trains on (without ego IDs).

    Args:
        x: (N, F) node features.
        edge_index: (2, E) directed edges.
        edge_attr: (E, D) edge features.
        edge_time: (E,) timestamps that order counterparties for port numbering.
        rung: which adaptations to apply.
        num_nodes: N.
        ports: optional precomputed (E, 2) ``[in_port, out_port]``. Pass this for large
            graphs whose ports are cached; otherwise they are computed here.
        edge_transform: optional function applied to the full edge-feature matrix, ports
            included, before the reverse edges are built. AML uses it for IBM's z-norm,
            which also normalizes the port columns.

    Returns:
        ``HeteroData`` with ``data["node"].x``, ``data[FWD].edge_index/edge_attr``, and,
        when ``rung.reverse_mp``, ``data[REV].edge_index/edge_attr``.
    """
    if rung.ports:
        if ports is None:
            ports = compute_ports(edge_index, edge_time, num_nodes)
        edge_attr = torch.cat([edge_attr, ports.to(edge_attr.dtype)], dim=1)
    if edge_transform is not None:
        edge_attr = edge_transform(edge_attr)
    data = HeteroData()
    data[NODE].x = x
    data[NODE].num_nodes = num_nodes
    data[FWD].edge_index = edge_index
    data[FWD].edge_attr = edge_attr
    if rung.reverse_mp:
        rev_attr = edge_attr.clone()
        if rung.ports:
            rev_attr[:, [-2, -1]] = rev_attr[:, [-1, -2]]
        data[REV].edge_index = edge_index.flip(0)
        data[REV].edge_attr = rev_attr
    return data


def add_ego_ids(batch: HeteroData, num_seeds: int) -> HeteroData:
    """Append the ego bit to node features: 1 on the first ``num_seeds`` nodes, else 0.

    For node tasks, the seeds are the target nodes. In ``exact`` mode (disjoint loader)
    each seed sits in its own subgraph; in ``batch`` mode all seeds share one subgraph and
    all of them are marked. Edge tasks mark the endpoints of the target edges instead;
    that is added together with the AML loader.
    """
    x = batch[NODE].x
    ego = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
    ego[:num_seeds] = 1
    batch[NODE].x = torch.cat([x, ego], dim=1)
    return batch


def add_ego_ids_edges(batch: HeteroData) -> HeteroData:
    """Append the ego bit for edge tasks: 1 on both endpoints of every seed edge in the batch.

    This is IBM's ``AddEgoIds``: it marks the nodes of
    ``batch[FWD].edge_label_index``, which is ``batch`` mode for edge tasks. With a
    disjoint loader, each subgraph holds a single seed, so the same code gives ``exact``
    mode.
    """
    x = batch[NODE].x
    ego = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
    ego[torch.unique(batch[FWD].edge_label_index.reshape(-1))] = 1
    batch[NODE].x = torch.cat([x, ego], dim=1)
    return batch


def model_inputs(batch: HeteroData) -> dict[str, torch.Tensor | None]:
    """Unpack a (sampled) ``HeteroData`` into ``GINLadder.forward`` keyword arguments."""
    has_rev = REV in batch.edge_types
    return {
        "x": batch[NODE].x,
        "edge_index": batch[FWD].edge_index,
        "edge_attr": batch[FWD].edge_attr,
        "rev_edge_index": batch[REV].edge_index if has_rev else None,
        "rev_edge_attr": batch[REV].edge_attr if has_rev else None,
    }
