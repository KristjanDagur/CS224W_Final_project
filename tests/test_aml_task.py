"""AML task on a tiny fake preprocessed graph: shapes, IBM masking, per-split z-norm, all rungs run."""

import numpy as np
import pytest
import torch

from amlinterp.data.transforms import FWD, NODE, REV
from amlinterp.models.gin import RUNGS, GINLadder
from amlinterp.tasks.aml import AMLTask


@pytest.fixture()
def data_dir(aml_data_dir):
    return aml_data_dir


def _cfg(data_dir):
    return {"data_dir": str(data_dir), "batch_size": 64, "eval_batch_size": 64, "num_neighbors": [3, 3],
            "w_ce": [1.0, 6.0], "ego_mode": "batch", "num_workers": 0}


@pytest.mark.parametrize("rung_name", ["R0", "R1", "R2", "R3"])
def test_all_rungs_score_only_sampled_seeds(data_dir, rung_name):
    rung = RUNGS[rung_name]
    task = AMLTask(_cfg(data_dir), rung)
    assert task.edge_in == (6 if rung.ports else 4) and task.node_in == 1 + int(rung.ego)
    assert (REV in task.graphs["train"].edge_types) == rung.reverse_mp
    model = GINLadder(task.node_in, task.edge_in, 2, hidden=8, num_layers=2, reverse_mp=rung.reverse_mp, readout="edge")
    batch = next(iter(task.loader("val", shuffle=False)))
    seeds = batch[FWD].seed_pos.clone()
    logits, y = task.forward(model, batch, torch.device("cpu"))
    info = task.last_batch_info
    assert info["seeds"] == len(seeds) == 64 and 0 < info["scored"] <= 64 and logits.shape == (info["scored"], 2)
    assert seeds.min() >= 300 and seeds.max() < 400  # val seeds are val edges of the train+val graph
    assert task.loss_fn(logits, y).isfinite()
    assert task.probs(logits).shape == (info["scored"], 1)


def test_features_are_z_normed_per_split_graph(data_dir):
    task = AMLTask(_cfg(data_dir), RUNGS["R2"])
    for split, n_edges in (("train", 300), ("val", 400), ("test", 500)):
        ea = task.graphs[split][FWD].edge_attr
        assert ea.shape[0] == n_edges
        torch.testing.assert_close(ea.mean(0), torch.zeros(6), atol=1e-5, rtol=0)  # IBM bug 1: own stats
    assert torch.equal(task.graphs["train"][NODE].x, torch.zeros(60, 1))  # z_norm(ones) = zeros
