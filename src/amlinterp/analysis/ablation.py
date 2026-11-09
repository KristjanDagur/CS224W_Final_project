"""Causal mean-ablation of concept units (hypothesis H2: are the detectors used?).

For one trained synthetic run:

1. **Unit selection on validation.** For each concept, rank the GNN units (layers h1..hL) by signed
   AUROC on the validation activations (``activations_val.pt`` from ``analysis.hooks``). The best
   unit, and the best ``k``, are that concept's candidate detectors. Test is not used for selection.
2. **Means on train.** Each unit's ablation value is its mean over *all nodes* of the training-split
   subgraphs: every node whose state feeds the scored node, under the same exact-ego protocol.
3. **Ablate and evaluate on test.** The unit is replaced by its mean at every node at that layer
   (``GINLadder.set_intervention``), so the change propagates through later message passing and the
   readout. The run then scores test F1 for every concept at the decision thresholds tuned on
   validation *before* ablating (``thresholds.json``; AGENTS.md fact 8).
4. **Random-unit control.** ``n_random`` single GNN units are drawn uniformly and ablated the same
   way, giving the drop that ablating an arbitrary unit causes.

The effect of ablating a set of units is ``delta[c] = F1_clean[c] - F1_ablated[c]``, in F1 points. A
unit is a used, concept-specific detector if ablating it hurts its own concept clearly more than
the other concepts and more than the random units do.

Usage (one run; ``scripts/slurm/ablate_synth.sbatch`` runs all of them as an array)::

    python -m amlinterp.analysis.ablation --run runs/synth_R3_L6_h64_s1 --out runs/analysis/ablation_synth
    python -m amlinterp.analysis.ablation --summarize runs/analysis/ablation_synth/*.json --out runs/analysis/ablation_synth
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from amlinterp.analysis.dissection import stack_units, unit_scores
from amlinterp.analysis.hooks import load_run
from amlinterp.metrics import minority_f1
from amlinterp.train import predict, seed_everything


@torch.no_grad()
def unit_means(model, task, split: str = "train", device: str = "cpu") -> dict[int, torch.Tensor]:
    """Mean of every unit of every node-state layer over all nodes the model processes on ``split``.

    Returns:
        ``{layer: (hidden,) tensor}`` for layers 0..L (0 is the embedding).
    """
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}

    def record(layer: int, h: torch.Tensor) -> torch.Tensor:
        sums[layer] = sums.get(layer, 0) + h.sum(0).double().cpu()
        counts[layer] = counts.get(layer, 0) + h.shape[0]
        return h

    model.set_intervention(record)
    try:
        predict(model, task, task.loader(split, shuffle=False), device)
    finally:
        model.set_intervention(None)
    return {layer: (sums[layer] / counts[layer]).float() for layer in sums}


def mean_ablation(units: list[tuple[int, int]], means: dict[int, torch.Tensor]):
    """Intervention that sets each ``(layer, unit)`` to its mean at every node."""
    by_layer: dict[int, list[int]] = {}
    for layer, unit in units:
        by_layer.setdefault(layer, []).append(unit)

    def fn(layer: int, h: torch.Tensor) -> torch.Tensor:
        cols = by_layer.get(layer)
        if not cols:
            return h
        h = h.clone()
        h[:, cols] = means[layer][cols].to(h.device, h.dtype)
        return h

    return fn


def test_f1(model, task, thresholds: np.ndarray, device: str = "cpu", intervention=None) -> np.ndarray:
    """Per-concept test minority-class F1 (%) at fixed thresholds, optionally under an intervention."""
    model.set_intervention(intervention)
    try:
        y, prob, _ = predict(model, task, task.loader("test", shuffle=False), device)
    finally:
        model.set_intervention(None)
    return minority_f1(y, prob, task.minority, thresholds) * 100


def select_units(acts_val: Path, top_k: int) -> tuple[list[str], dict[str, list[tuple[int, int]]], dict[str, list[float]]]:
    """Best ``top_k`` GNN units per concept by signed AUROC on validation activations.

    Returns:
        ``(label_names, {concept: [(layer, unit), ...]}, {concept: [auroc, ...]})``, best first.
    """
    d = torch.load(acts_val, weights_only=False)
    gnn = {k: v for k, v in d["acts"].items() if k.startswith("h") and k != "h0"}
    a, names, _ = stack_units(gnn)
    y = d["y"].float().numpy().reshape(len(a), -1) > 0.5
    au = unit_scores(a, y)["auroc"]
    labels = d["meta"]["label_names"]
    chosen, scores = {}, {}
    for j, c in enumerate(labels):
        order = np.argsort(-np.nan_to_num(au[:, j], nan=-1))[:top_k]
        chosen[c] = [(int(names[k].split(":")[0][1:]), int(names[k].split(":")[1])) for k in order]
        scores[c] = [round(float(au[k, j]), 4) for k in order]
    return labels, chosen, scores


def ablate_run(run_dir: Path, top_k: int = 5, n_random: int = 20, seed: int = 0, device: str = "cpu") -> dict:
    """The full ablation experiment for one run (see the module docstring). Returns a JSON-ready dict."""
    run_dir = Path(run_dir)
    cfg, task, model = load_run(run_dir, device)
    seed_everything(seed)
    labels, chosen, val_auroc = select_units(run_dir / "activations_val.pt", top_k)
    thr_map = json.loads((run_dir / "thresholds.json").read_text())
    thresholds = np.array([thr_map[c] for c in labels])
    means = unit_means(model, task, "train", device)
    clean = test_f1(model, task, thresholds, device)

    def delta(units):
        return (clean - test_f1(model, task, thresholds, device, mean_ablation(units, means))).round(3).tolist()

    per_concept = {}
    for c in labels:
        per_concept[c] = {
            "units": [f"h{l}:{u}" for l, u in chosen[c]], "val_auroc": val_auroc[c],
            "delta_top1": delta(chosen[c][:1]), f"delta_top{top_k}": delta(chosen[c]),
        }
    rng = np.random.default_rng(seed)
    hidden, n_layers = cfg["hidden"], cfg["num_layers"]
    random_units = [(int(rng.integers(1, n_layers + 1)), int(rng.integers(0, hidden))) for _ in range(n_random)]
    random_deltas = [delta([u]) for u in random_units]
    return {
        "run_id": cfg["run_id"], "rung": cfg["rung"], "num_layers": n_layers, "labels": labels,
        "protocol": {"ego_mode": cfg["ego_mode"], "means": "train split, all nodes", "selection": "val, signed AUROC",
                     "thresholds": "val-tuned (thresholds.json)", "top_k": top_k, "n_random": n_random, "seed": seed},
        "clean_f1": clean.round(3).tolist(), "concepts": per_concept,
        "random_units": [f"h{l}:{u}" for l, u in random_units], "random_deltas": random_deltas,
    }


def summarize(results: list[dict], top_k: int = 5) -> list[dict]:
    """Per run group (rung, depth) and target concept: own-concept drop, mean off-target drop, random-unit drops (mean over seeds)."""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        groups.setdefault((r["rung"], r["num_layers"]), []).append(r)
    rows = []
    for (rung, depth), runs in sorted(groups.items()):
        labels = runs[0]["labels"]
        for j, c in enumerate(labels):
            row = {"rung": rung, "layers": depth, "concept": c, "n_seeds": len(runs)}
            for key in ("delta_top1", f"delta_top{top_k}"):
                d = np.array([r["concepts"][c][key] for r in runs])           # (seeds, concepts)
                row[f"{key}_own"] = round(float(d[:, j].mean()), 2)
                row[f"{key}_offtarget"] = round(float(np.delete(d, j, axis=1).mean()), 2)
            rand = np.array([np.array(r["random_deltas"])[:, j] for r in runs])  # (seeds, n_random)
            row["random_mean"] = round(float(rand.mean()), 2)
            row["random_p95"] = round(float(np.percentile(rand, 95)), 2)
            row["clean_f1"] = round(float(np.mean([r["clean_f1"][j] for r in runs])), 2)
            rows.append(row)
    return rows


def main() -> None:
    """CLI entry point; see the module docstring."""
    p = argparse.ArgumentParser(description="Mean-ablation of concept units.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path, help="runs/<run_id> (needs activations_val.pt and thresholds.json)")
    src.add_argument("--summarize", type=Path, nargs="+", help="per-run .json files to merge into a table")
    p.add_argument("--out", type=Path, required=True, help="output directory (per-run .json) or table prefix")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.run:
        try:  # array tasks create this concurrently; on NFS, mkdir(exist_ok=True) can still raise on a race
            args.out.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            if not args.out.is_dir():
                raise
        res = ablate_run(args.run, args.top_k, args.n_random, args.seed)
        path = args.out / f"{res['run_id']}.json"
        path.write_text(json.dumps(res, indent=2))
        print(f"wrote {path}")
        return
    results = [json.loads(f.read_text()) for f in args.summarize]
    rows = summarize(results, args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{args.out}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}_summary.csv")


if __name__ == "__main__":
    main()
