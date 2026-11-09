"""Tests for the intervention hook and mean-ablation (analysis.ablation)."""

import numpy as np
import pytest
import torch

from amlinterp.analysis.ablation import mean_ablation, test_f1 as ablation_f1, unit_means
from amlinterp.data.synthetic import CirculantConfig, build_dataset
from amlinterp.data.transforms import model_inputs
from amlinterp.models.gin import RUNGS, GINLadder
from amlinterp.tasks.synthetic import SyntheticTask

L, H = 3, 8


@pytest.fixture(scope="module")
def task_and_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    build_dataset(d, CirculantConfig(n=150, d=6.0, r=4.0))
    cfg = {"data_dir": str(d), "num_layers": L, "batch_size": 64, "eval_batch_size": 64, "ego_mode": "exact",
           "num_workers": 0}
    task = SyntheticTask(cfg, RUNGS["R3"])
    torch.manual_seed(0)
    model = GINLadder(task.node_in, task.edge_in, task.out_dim, hidden=H, num_layers=L, reverse_mp=True,
                      readout="node").eval()
    return task, model


def _full_graph_forward(task, model, **kw):
    g = task.graphs["test"].clone()
    g["node"].x = torch.cat([g["node"].x, torch.zeros(g["node"].num_nodes, 1)], 1)  # ego bit off everywhere
    with torch.no_grad():
        return model(**model_inputs(g), return_hidden=True, **kw)


def test_identity_intervention_changes_nothing(task_and_model):
    task, model = task_and_model
    out0, hid0 = _full_graph_forward(task, model)
    seen = []
    model.set_intervention(lambda layer, h: (seen.append(layer), h)[1])
    try:
        out1, hid1 = _full_graph_forward(task, model)
    finally:
        model.set_intervention(None)
    assert seen == list(range(L + 1))
    torch.testing.assert_close(out0, out1)


def test_mean_ablation_sets_unit_and_propagates(task_and_model):
    task, model = task_and_model
    means = {layer: torch.full((H,), 0.123) for layer in range(L + 1)}
    out0, hid0 = _full_graph_forward(task, model)
    model.set_intervention(mean_ablation([(1, 2), (1, 5)], means))
    try:
        out1, hid1 = _full_graph_forward(task, model)
    finally:
        model.set_intervention(None)
    assert torch.allclose(hid1[1][:, [2, 5]], torch.tensor(0.123))            # ablated units hold the mean
    keep = [k for k in range(H) if k not in (2, 5)]
    torch.testing.assert_close(hid1[1][:, keep], hid0[1][:, keep])            # other units at that layer untouched
    torch.testing.assert_close(hid1[0], hid0[0])                              # earlier layers untouched
    assert not torch.allclose(hid1[2], hid0[2]) and not torch.allclose(out1, out0)  # later layers and output change


def test_unit_means_cover_all_layers_and_match_identity_f1(task_and_model):
    task, model = task_and_model
    means = unit_means(model, task, "train")
    assert sorted(means) == list(range(L + 1)) and all(m.shape == (H,) for m in means.values())
    thr = np.full(task.out_dim, 0.5)
    torch.manual_seed(0)
    a = ablation_f1(model, task, thr)
    b = ablation_f1(model, task, thr, intervention=lambda layer, h: h)
    np.testing.assert_allclose(a, b)
    assert model._intervention is None                                         # cleared after use
