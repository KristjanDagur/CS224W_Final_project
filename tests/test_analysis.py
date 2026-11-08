"""Analysis tests: dissection scores against sklearn and planted units; activation extraction and ego modes."""

import json

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from amlinterp.analysis.dissection import (
    average_precision, bootstrap_coverage, chance_level, coverage, iou, permutation_null, stack_units, unit_scores,
)
from amlinterp.analysis.hooks import extract_run, load_run
from amlinterp.data.synthetic import CirculantConfig, make_split
from amlinterp.models.gin import RUNGS
from amlinterp.train import build_model, make_task

# --------------------------------------------------------------------------- dissection


def test_average_precision_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 5, size=(300, 7)).astype(float)  # many ties
    acts[:, 0] = 1.0                                        # a constant unit
    concepts = rng.random((300, 3)) < [0.1, 0.4, 0.8]
    w = rng.integers(0, 3, size=300).astype(float)
    ours, ours_w = average_precision(acts, concepts), average_precision(acts, concepts, w)
    for k in range(7):
        for c in range(3):
            assert ours[k, c] == pytest.approx(average_precision_score(concepts[:, c], acts[:, k]))
            assert ours_w[k, c] == pytest.approx(average_precision_score(concepts[:, c], acts[:, k], sample_weight=w))


def test_planted_units_are_found_with_their_sign():
    rng = np.random.default_rng(1)
    n = 2000
    concepts = np.stack([rng.random(n) < 0.3, rng.random(n) < 0.7], axis=1)
    noise = rng.normal(size=(n, 5))
    acts = np.column_stack([noise, concepts[:, 0] + 0.1 * rng.normal(size=n),   # unit 5: high on concept 0
                            -concepts[:, 1].astype(float) + 0.1 * rng.normal(size=n),  # unit 6: low on concept 1
                            np.zeros(n)])                                       # unit 7: constant
    s = unit_scores(acts, concepts)
    assert s["iou"][5, 0] == pytest.approx(1.0) and s["iou_sign"][5, 0] == 1
    assert s["iou"][6, 1] == pytest.approx(1.0) and s["iou_sign"][6, 1] == -1
    assert s["ap"][6, 1] == pytest.approx(1.0) and s["ap_sign"][6, 1] == -1
    assert np.all(s["iou"][7] == 0)                                       # ties at tau are excluded
    assert s["ap"][7] == pytest.approx(concepts.mean(0))                  # constant score -> base rate
    cov = coverage(s, np.array(["h0"] * 5 + ["h1"] * 3))
    assert list(cov["iou"]["unit"]) == [5, 6]
    assert cov["iou"]["per_layer"]["h0"].max() < 0.6
    # Noise units sit near the single-unit chance level.
    assert s["iou"][:5, 0].mean() == pytest.approx(chance_level("iou", concepts[:, 0].mean()), abs=0.05)


def test_fixed_quantile_iou_and_unit_weights():
    rng = np.random.default_rng(2)
    acts, concepts = rng.normal(size=(500, 4)), rng.random((500, 2)) < 0.2
    assert np.allclose(iou(acts, concepts, weights=np.ones(500)), iou(acts, concepts))
    planted = np.column_stack([acts, concepts[:, 0].astype(float)])
    assert iou(planted, concepts, quantile=concepts[:, 0].mean())[4, 0] == pytest.approx(1.0)


def test_null_and_bootstrap_separate_a_real_detector_from_chance():
    rng = np.random.default_rng(3)
    n = 1500
    concepts = rng.random((n, 1)) < 0.25
    acts = np.column_stack([rng.normal(size=(n, 20)), concepts[:, 0] + 0.5 * rng.normal(size=n)])
    real = np.nanmax(unit_scores(acts, concepts)["iou"], axis=0)
    null = permutation_null(acts, concepts, n_perm=5)
    boot = bootstrap_coverage(acts, concepts, n_boot=10, groups=np.arange(n) // 3)  # groups of 3 items
    assert null["iou"].shape == (5, 1) and boot["iou"].shape == (10, 1)
    assert real[0] > null["iou"].max() + 0.2
    assert np.percentile(boot["iou"][:, 0], 2.5) > null["iou"].max()


# --------------------------------------------------------------------------- activation extraction


def _synth_run(tmp_path, rung, ego_mode, layers=3, hidden=16):
    """A finished-looking synthetic run (random weights) on tiny circulant graphs."""
    data_dir = tmp_path / "data"
    if not data_dir.exists():
        data_dir.mkdir()
        for i, split in enumerate(("train", "val", "test")):
            torch.save(make_split(CirculantConfig(n=120, d=6.0, r=4.0), seed=i), data_dir / f"{split}.pt")
    cfg = {"dataset": "synth", "data_dir": str(data_dir), "num_layers": layers, "hidden": hidden,
           "final_dropout": 0.1, "batch_size": 32, "eval_batch_size": 32, "ego_mode": ego_mode,
           "num_workers": 0, "rung": rung, "seed": 0, "run_id": f"synth_{rung}_{ego_mode}",
           **{k: v for k, v in vars(RUNGS[rung]).items() if k != "name"}}
    run_dir = tmp_path / cfg["run_id"]
    run_dir.mkdir()
    task = make_task(cfg)
    torch.manual_seed(0)
    model = build_model(cfg, task)
    for bn in model.norms:  # non-trivial eval statistics
        bn.running_mean.uniform_(-0.5, 0.5)
        bn.running_var.uniform_(0.5, 2.0)
    torch.save(model.state_dict(), run_dir / "best.pt")
    (run_dir / "config.json").write_text(json.dumps(cfg))
    return run_dir


def _by_id(d):
    order = d["ids"].argsort()
    return {k: v[order] for k, v in d["acts"].items()}


def test_extract_synthetic_shapes_ids_and_meta(tmp_path):
    run_dir = _synth_run(tmp_path, "R3", "exact")
    d = torch.load(extract_run(run_dir, "test"), weights_only=False)
    assert set(d["acts"]) == {"h0", "h1", "h2", "h3", "dec1", "dec2"}
    assert d["acts"]["h3"].shape == (120, 16) and d["acts"]["dec1"].shape == (120, 50) and d["acts"]["dec2"].shape == (120, 25)
    assert torch.equal(d["ids"].sort().values, torch.arange(120))
    assert d["logits"].shape == (120, 11) and d["y"].shape == (120, 11)
    assert d["meta"]["protocol"]["ego_mode"] == "exact" and d["meta"]["protocol"]["disjoint"]
    assert d["meta"]["protocol"]["ego"] and d["meta"]["units"]["h0"] == 16
    # Logits from the hooks equal a plain forward of the restored model.
    _, task, model = load_run(run_dir)
    batch = next(iter(task.loader("test", shuffle=False)))
    logits, _ = task.forward(model, batch, "cpu")
    assert torch.allclose(d["logits"][: len(logits)], logits, atol=1e-6)
    # The decoder hooks capture post-ReLU activations.
    assert (d["acts"]["dec1"] >= 0).all()
    acts, names, layers = stack_units(d["acts"])
    assert acts.shape == (120, 4 * 16 + 50 + 25) and names[16] == "h1:0" and layers[-1] == "dec2"


def test_exact_ego_activations_do_not_depend_on_the_batch_but_batch_mode_does(tmp_path):
    exact = _synth_run(tmp_path, "R3", "exact")
    a = _by_id(torch.load(extract_run(exact, "test", eval_batch_size=7), weights_only=False))
    b = _by_id(torch.load(extract_run(exact, "test", eval_batch_size=50), weights_only=False))
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5), k
    batch = _synth_run(tmp_path, "R3", "batch")
    c = _by_id(torch.load(extract_run(batch, "test", eval_batch_size=50), weights_only=False))
    assert not torch.allclose(a["h3"], c["h3"], atol=1e-3)  # shared subgraph: other seeds are marked too
    # Override tags keep protocols apart on disk.
    assert (exact / "activations_test_eval_batch_size7.pt").exists()


def test_extract_aml_edges(aml_data_dir, tmp_path):
    cfg = {"dataset": "amlhis", "data_dir": str(aml_data_dir), "num_layers": 2, "hidden": 8, "final_dropout": 0.1,
           "batch_size": 64, "eval_batch_size": 64, "num_neighbors": [3, 3], "w_ce": [1.0, 6.0],
           "ego_mode": "batch", "num_workers": 0, "rung": "R3", "seed": 0, "run_id": "amlhis_R3_test",
           **{k: v for k, v in vars(RUNGS["R3"]).items() if k != "name"}}
    run_dir = tmp_path / cfg["run_id"]
    run_dir.mkdir()
    task = make_task(cfg)
    torch.save(build_model(cfg, task).state_dict(), run_dir / "best.pt")
    (run_dir / "config.json").write_text(json.dumps(cfg))
    d = torch.load(extract_run(run_dir, "test"), weights_only=False)
    g = torch.load(aml_data_dir / "graph.pt", weights_only=False)
    n = len(d["ids"])
    assert 0 < n <= 100 and d["acts"]["h2"].shape == (n, 16)          # [h_u || h_v]
    assert ((d["ids"] >= 400) & (d["ids"] < 500)).all()                 # test edges only
    assert len(d["ids"].unique()) == n
    assert torch.equal(d["y"], g["y"][d["ids"]])
    assert d["meta"]["protocol"]["item"] == "edge" and d["meta"]["protocol"]["ego_mode"] == "batch"
