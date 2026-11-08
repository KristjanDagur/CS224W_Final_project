# AMLworld edge-classification task, following IBM Multi-GNN `data_loading.get_data`,
# `train_util.get_loaders`/`AddEgoIds`/`add_arange_ids` and `training.train_hetero`.
# Original: https://github.com/IBM/Multi-GNN, Copyright IBM Corp., Apache License 2.0.
"""AMLworld HI-Small: is a transaction laundering? (Egressy et al., Table 2.)

The protocol follows IBM Multi-GNN exactly, including its known bugs (AGENTS.md, fact 7):

* **Nested graphs.** Train, val and test are message-passing graphs over edges
  ``[0, n_train)``, ``[0, n_train + n_val)`` and all edges. Each split scores only its own
  edges. Edges are time-sorted, so these are prefixes.
* **Features.** Edge features are ``[Timestamp, Amount Received, Received Currency,
  Payment Format]`` (+ ``[in_port, out_port]``), z-normalized *separately on each split's
  graph* (IBM bug 1). Node features are ``z_norm(ones)``, which is all zeros.
* **Sampling.** ``LinkNeighborLoader`` with ``num_neighbors=[100, 100]`` for every edge
  type, ``batch_size=8192``, and the training seeds shuffled.
* **Ego IDs** (``ego_mode="batch"``): both endpoints of every seed edge in the batch are
  marked, as in IBM's ``AddEgoIds``.
* **Scored seeds.** A seed is scored only if the sampler included the seed edge itself in
  the subgraph (IBM's ``isin(edge ID, seed IDs)`` mask). Missing seeds are dropped from
  the loss and the metrics (IBM bug 2). The task reports how many seeds were scored
  (``last_batch_info``), and ``train.py`` logs that as ``seed_coverage``. We identify edges
  by PyG's ``e_id``, which equals IBM's prepended ``arange`` ID column.
* **Loss and metric.** Two-class cross-entropy with class weights ``w_ce`` (IBM
  ``model_settings.json``, ``gin``). The metric is F1 of the laundering class at
  probability 0.5, which is IBM's argmax.

IBM bug 3 is harmless and not reproduced. In ``create_hetero_obj`` the forward and reverse
edge types share one tensor, so IBM's in-place port swap hits both, and the reverse edges
end up with the *same* column order as the forward ones. Each edge type has its own
embedding ``Linear``, so swapping or not is equivalent up to a permutation of its weights.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from amlinterp.data.aml import SPLITS, edge_features
from amlinterp.data.transforms import FWD, add_ego_ids_edges, build_ladder_graph, model_inputs
from amlinterp.metrics import minority_f1, pr_auc
from amlinterp.models.gin import Rung


def z_norm(data: Tensor) -> Tensor:
    """IBM ``data_util.z_norm``: per-column (x - mean) / std, with std 0 replaced by 1."""
    std = data.std(0, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return (data - data.mean(0, keepdim=True)) / std


def _tag_seeds(batch: HeteroData, offset: int) -> HeteroData:
    """Loader transform: positions of this batch's seed edges in the split graph (IBM ``inds[input_id]``)."""
    batch[FWD].seed_pos = batch[FWD].input_id + offset
    return batch


class AMLTask:
    """AMLworld edge classification for one rung; same interface as ``SyntheticTask``.

    Args:
        cfg: run config. Uses ``data_dir``, ``batch_size``, ``eval_batch_size``,
            ``num_neighbors``, ``w_ce``, ``ego_mode`` and ``num_workers``.
        rung: the ladder rung to build inputs for.
    """

    readout = "edge"
    label_names = ["laundering"]

    def __init__(self, cfg: dict, rung: Rung) -> None:
        if cfg["ego_mode"] != "batch":
            raise NotImplementedError("AML training uses IBM's batch ego IDs; exact mode is for analysis")
        self.rung, self._cfg = rung, cfg
        g = torch.load(Path(cfg["data_dir"]) / "graph.pt", weights_only=False)
        sizes = [len(g["split"][k]) for k in SPLITS]
        self.offset = dict(zip(SPLITS, np.cumsum([0] + sizes[:-1]).tolist()))  # first scored edge
        self.n_scored = dict(zip(SPLITS, sizes))
        x = z_norm(torch.ones(g["num_nodes"], 1))  # IBM: placeholder feature of ones, z-normed -> zeros
        self.graphs: dict[str, HeteroData] = {}
        for split in SPLITS:
            end = self.offset[split] + self.n_scored[split]
            graph = build_ladder_graph(
                x, g["edge_index"][:, :end], edge_features(g, torch.arange(end), ports=False),
                g["edge_time"][:end], rung, g["num_nodes"],
                ports=g["ports"][:end] if rung.ports else None,
                edge_transform=z_norm,  # IBM bug 1: statistics of this split's graph
            )
            graph[FWD].y = g["y"][:end]
            self.graphs[split] = graph
        self.node_in = 1 + int(rung.ego)
        self.edge_in = self.graphs["train"][FWD].edge_attr.shape[1]
        self.out_dim = 2
        self.minority = np.array([1])
        self._w_ce = torch.tensor(cfg["w_ce"], dtype=torch.float)
        self.last_batch_info: dict | None = None

    def loader(self, split: str, shuffle: bool) -> LinkNeighborLoader:
        """IBM ``get_loaders``: neighbor-sampled subgraphs around this split's edges."""
        g, off, m = self.graphs[split], self.offset[split], self.n_scored[split]
        return LinkNeighborLoader(
            g,
            num_neighbors=list(self._cfg["num_neighbors"]),
            edge_label_index=(FWD, g[FWD].edge_index[:, off:off + m]),
            edge_label=g[FWD].y[off:off + m],
            batch_size=self._cfg["batch_size"] if split == "train" else self._cfg["eval_batch_size"],
            shuffle=shuffle,
            transform=partial(_tag_seeds, offset=off),
            num_workers=self._cfg.get("num_workers", 0),
        )

    def loss_fn(self, logits: Tensor, y: Tensor) -> Tensor:
        """Class-weighted cross-entropy, as in IBM: weights ``[w_ce1, w_ce2]`` for licit and laundering."""
        return F.cross_entropy(logits, y, weight=self._w_ce.to(logits.device))

    def forward(self, model: nn.Module, batch: HeteroData, device: torch.device, return_hidden: bool = False):
        """Score the seed edges that the sampler included in the batch.

        Returns ``(logits, labels)``. With ``return_hidden``, returns ``(logits, labels, hidden, ids)``:
        ``hidden[l]`` is ``[h_l(u) || h_l(v)]``, shape (n_scored, 2 * hidden), for each scored edge
        ``u -> v`` (``l = 0`` is the embedding), and ``ids`` are the scored edges' global indices
        (the split graphs are time prefixes, so split-graph and global edge indices agree).
        """
        st = batch[FWD]
        mask = torch.isin(st.e_id, st.seed_pos)  # IBM: seeds absent from the subgraph are dropped
        self.last_batch_info = {"seeds": int(st.seed_pos.numel()), "scored": int(mask.sum())}
        if self.rung.ego:
            batch = add_ego_ids_edges(batch)
        batch = batch.to(device)
        target = mask.nonzero().squeeze(1).to(device)
        out = model(**model_inputs(batch), target=target, return_hidden=return_hidden)
        y = batch[FWD].y[target]
        if not return_hidden:
            return out, y
        logits, hidden = out
        src, dst = batch[FWD].edge_index[:, target]
        return logits, y, [torch.cat([h[src], h[dst]], dim=1) for h in hidden], batch[FWD].e_id[target]

    def probs(self, logits: Tensor) -> Tensor:
        """P(laundering) as an (N, 1) column. Thresholding at 0.5 equals IBM's argmax."""
        return torch.softmax(logits, dim=-1)[:, 1:2]

    def label_matrix(self, y: Tensor) -> Tensor:
        """Labels as an (N, 1) column."""
        return y[:, None]

    def metrics(self, y: np.ndarray, prob: np.ndarray, thresholds: np.ndarray | float = 0.5) -> dict:
        """Laundering-class F1 (%) and PR-AUC (%). ``mean_f1`` is the selection key, as in the synthetic task."""
        f1 = float(minority_f1(y, prob, self.minority, thresholds)[0] * 100)
        ap = float(pr_auc(y, prob, self.minority)[0] * 100)
        return {"mean_f1": f1, "mean_pr_auc": ap, "f1": {"laundering": round(f1, 2)}, "pr_auc": {"laundering": round(ap, 2)}}
