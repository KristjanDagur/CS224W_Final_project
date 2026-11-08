# GIN-with-edge-features ladder, adapted from IBM Multi-GNN `models.py` (class `GINe`) and the
# `to_hetero(..., aggr='mean')` reverse message passing in `training.py`.
# Original: https://github.com/IBM/Multi-GNN, Copyright IBM Corp., Apache License 2.0.
"""The expressiveness ladder: GIN -> +reverse MP -> +ports -> +ego IDs (Multi-GIN).

All four rungs share one architecture. The rung only changes:

* ``reverse_mp`` (a model flag): each layer also aggregates over reversed edges, with
  separate weights, and averages the two directions. This equals IBM's
  ``to_hetero(model, metadata, aggr='mean')`` over the edge types ``to`` and ``rev_to``:
  separate edge embeddings and convs per edge type, and one BatchNorm per node type
  applied after the mean.
* ``ports`` and ``ego`` (data flags): extra input columns. Port numbers are appended to
  the edge features, and the ego bit is appended to the node features (see
  ``amlinterp.data.transforms``). The model sees them only through its input sizes.

Layer update (IBM GINe, residual=True)::

    m = GINEConv_to(x, e)                                  # plain GIN
    m = (GINEConv_to(x, e) + GINEConv_rev(x, e_rev)) / 2   # with reverse MP
    x = (x + relu(BatchNorm(m))) / 2

Readouts:

* ``edge``, used for AML: ``MLP([relu(h_u) || relu(h_v) || e_uv])``, sizes 3h -> 50 -> 25 -> out.
* ``node``, used for synthetic: ``MLP(relu(h_v))``, sizes h -> 50 -> 25 -> out.

``forward(..., return_hidden=True)`` also returns the node states after the embedding
and after each layer, which the activation hooks use.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.nn import GINEConv


@dataclass(frozen=True)
class Rung:
    """One rung of the ladder: which of Egressy et al.'s adaptations are switched on."""

    name: str
    reverse_mp: bool
    ports: bool
    ego: bool


RUNGS: dict[str, Rung] = {
    "R0": Rung("R0", reverse_mp=False, ports=False, ego=False),  # GIN
    "R1": Rung("R1", reverse_mp=True, ports=False, ego=False),   # + reverse MP
    "R2": Rung("R2", reverse_mp=True, ports=True, ego=False),    # + port numbers
    "R3": Rung("R3", reverse_mp=True, ports=True, ego=True),     # + ego IDs (Multi-GIN)
}


def _gine(hidden: int) -> GINEConv:
    """GINEConv as in IBM GINe: 2-layer MLP, with an edge Linear inside (``edge_dim=hidden``)."""
    mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
    return GINEConv(mlp, edge_dim=hidden)


def _decoder(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """IBM GINe readout MLP: in -> 50 -> 25 -> out, with dropout after each hidden layer."""
    return nn.Sequential(
        nn.Linear(in_dim, 50), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(50, 25), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(25, out_dim),
    )


class GINLadder(nn.Module):
    """GIN with edge features, with optional reverse message passing (IBM GINe).

    Args:
        node_in: number of node input features (+1 when ego IDs are on).
        edge_in: number of edge input features (+2 when ports are on).
        out_dim: number of output logits (2 for AML, 11 for synthetic).
        hidden: hidden size, shared by node and edge embeddings.
        num_layers: number of message-passing layers.
        reverse_mp: also pass messages along reversed edges (separate weights).
        readout: ``"edge"`` (score edges from endpoint states) or ``"node"``.
        final_dropout: dropout in the readout MLP.
    """

    def __init__(
        self,
        node_in: int,
        edge_in: int,
        out_dim: int,
        hidden: int = 64,
        num_layers: int = 2,
        reverse_mp: bool = False,
        readout: str = "edge",
        final_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if readout not in ("edge", "node"):
            raise ValueError(f"readout must be 'edge' or 'node', got {readout!r}")
        self.hidden, self.num_layers = hidden, num_layers
        self.reverse_mp, self.readout = reverse_mp, readout
        self.node_emb = nn.Linear(node_in, hidden)
        self.edge_emb = nn.Linear(edge_in, hidden)
        self.convs = nn.ModuleList(_gine(hidden) for _ in range(num_layers))
        if reverse_mp:
            self.edge_emb_rev = nn.Linear(edge_in, hidden)
            self.convs_rev = nn.ModuleList(_gine(hidden) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(num_layers))
        dec_in = 3 * hidden if readout == "edge" else hidden
        self.decoder = _decoder(dec_in, out_dim, final_dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        rev_edge_index: Tensor | None = None,
        rev_edge_attr: Tensor | None = None,
        target: Tensor | None = None,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Run message passing and the readout.

        Args:
            x: (N, node_in) node features.
            edge_index: (2, E) forward edges ``src -> dst``.
            edge_attr: (E, edge_in) forward edge features.
            rev_edge_index: (2, E') reversed edges. Required when ``reverse_mp``.
            rev_edge_attr: (E', edge_in) features of the reversed edges. With ports, their
                port columns are already swapped.
            target: which items to score. For ``readout="node"``, node indices. For
                ``readout="edge"``, indices into ``edge_index``. ``None`` scores all.
            return_hidden: also return ``[h_0, h_1, ..., h_L]``, the node states after
                the embedding and after each layer, each of shape (N, hidden).

        Returns:
            Logits of shape (num_targets, out_dim), plus the hidden states if requested.
        """
        if self.reverse_mp and (rev_edge_index is None or rev_edge_attr is None):
            raise ValueError("reverse_mp=True needs rev_edge_index and rev_edge_attr")
        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)
        e_rev = self.edge_emb_rev(rev_edge_attr) if self.reverse_mp else None
        hidden = [h]
        for i in range(self.num_layers):
            m = self.convs[i](h, edge_index, e)
            if self.reverse_mp:
                m = (m + self.convs_rev[i](h, rev_edge_index, e_rev)) / 2
            h = (h + torch.relu(self.norms[i](m))) / 2
            hidden.append(h)

        if self.readout == "node":
            z = torch.relu(h if target is None else h[target])
        else:
            src, dst = edge_index if target is None else edge_index[:, target]
            e_t = e if target is None else e[target]
            z = torch.cat([torch.relu(h[src]), torch.relu(h[dst]), e_t], dim=-1)
        out = self.decoder(z)
        return (out, hidden) if return_hidden else out
