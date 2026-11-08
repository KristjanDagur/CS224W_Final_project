"""Synthetic pattern-detection task: 11-way multi-label node classification (Egressy et al., Table 1).

Each split is an independent random circulant graph (``amlinterp.data.synthetic``). Following
Egressy F.3, there is no neighbor subsampling: each target node gets its full L-hop
neighborhood (``num_neighbors=-1`` for every hop, in both edge directions if reverse MP is on).

The sampling protocol is the same for every rung and is set by ``ego_mode``. ``exact`` uses
one disjoint subgraph per target node (``disjoint=True``); ``batch`` uses one shared subgraph
per batch. Ego IDs are added only for rungs with ``rung.ego``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from amlinterp.data.synthetic import LABEL_NAMES
from amlinterp.data.transforms import EGO_MODES, NODE, add_ego_ids, build_ladder_graph, model_inputs
from amlinterp.metrics import minority_class, minority_f1, pr_auc
from amlinterp.models.gin import Rung

SPLITS = ("train", "val", "test")


class SyntheticTask:
    """Loads the synthetic splits for one rung and provides loaders, a batch step and metrics.

    Args:
        cfg: run config. Uses ``data_dir``, ``num_layers``, ``batch_size``,
            ``eval_batch_size``, ``ego_mode`` and ``num_workers``.
        rung: the ladder rung to build inputs for.
    """

    readout = "node"
    label_names = list(LABEL_NAMES)

    def __init__(self, cfg: dict, rung: Rung) -> None:
        if cfg["ego_mode"] not in EGO_MODES:
            raise ValueError(f"ego_mode must be one of {EGO_MODES}")
        self.rung = rung
        self.graphs: dict[str, HeteroData] = {}
        self.labels: dict[str, Tensor] = {}
        for split in SPLITS:
            d = torch.load(Path(cfg["data_dir"]) / f"{split}.pt", weights_only=False)
            g = build_ladder_graph(d.x, d.edge_index, d.edge_attr, d.edge_time, rung, d.num_nodes)
            g[NODE].y = d.y
            self.graphs[split] = g
            self.labels[split] = d.y
        g = self.graphs["train"]
        self.node_in = g[NODE].x.shape[1] + int(rung.ego)
        self.edge_in = g[g.edge_types[0]].edge_attr.shape[1]
        self.out_dim = len(LABEL_NAMES)
        self.minority = minority_class(self.labels["train"].numpy())
        self.loss_fn = nn.BCEWithLogitsLoss()
        self._cfg = cfg

    def loader(self, split: str, shuffle: bool) -> NeighborLoader:
        """Full-neighborhood loader over all nodes of a split, seeded by the global torch RNG."""
        g = self.graphs[split]
        bs = self._cfg["batch_size"] if split == "train" else self._cfg["eval_batch_size"]
        return NeighborLoader(
            g,
            num_neighbors={et: [-1] * self._cfg["num_layers"] for et in g.edge_types},
            input_nodes=NODE,
            batch_size=bs,
            shuffle=shuffle,
            disjoint=self._cfg["ego_mode"] == "exact",
            num_workers=self._cfg.get("num_workers", 0),
        )

    def forward(self, model: nn.Module, batch: HeteroData, device: torch.device, return_hidden: bool = False):
        """Score the seed nodes of a batch.

        Returns ``(logits, labels)``. With ``return_hidden``, returns ``(logits, labels, hidden, ids)``:
        ``hidden[l]`` is the (n_seed, hidden) node state after layer ``l`` (``l = 0`` is the
        embedding) at the seed nodes, and ``ids`` are the seeds' node indices in the split graph.
        """
        n_seed = batch[NODE].batch_size
        if self.rung.ego:
            batch = add_ego_ids(batch, n_seed)
        batch = batch.to(device)
        target = torch.arange(n_seed, device=device)
        out = model(**model_inputs(batch), target=target, return_hidden=return_hidden)
        y = batch[NODE].y[:n_seed]
        if not return_hidden:
            return out, y
        logits, hidden = out
        return logits, y, [h[:n_seed] for h in hidden], batch[NODE].n_id[:n_seed]

    def metrics(self, y: np.ndarray, prob: np.ndarray, thresholds: np.ndarray | float = 0.5) -> dict:
        """Per-label and mean minority-class F1 (in %) and PR-AUC. The selection key is ``mean_f1``."""
        f1 = minority_f1(y, prob, self.minority, thresholds) * 100
        ap = pr_auc(y, prob, self.minority) * 100
        return {
            "mean_f1": float(f1.mean()),
            "mean_pr_auc": float(ap.mean()),
            "f1": dict(zip(self.label_names, np.round(f1, 2).tolist())),
            "pr_auc": dict(zip(self.label_names, np.round(ap, 2).tolist())),
        }
