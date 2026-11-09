"""Structural baselines for dissection: how much of a concept is explained by degree alone?

A unit can score well on a concept it does not compute, because the concept correlates with
something simpler. On the random circulant graphs, being on a k-cycle, being a scatter-gather sink
and being a biclique sink all correlate with degree. A model that cannot detect cycles (R0-R2)
still has "cycle units" that are really degree units. This module scores each concept with plain
node statistics (in-/out-/total degree, fan-in/out) as the AUROC of the better direction. The
results are directly comparable with the dissection's signed AUROC coverage.

Usage::

    python -m amlinterp.analysis.baselines --graph data/processed/synthetic/test.pt \\
        --dissection runs/analysis/dissection_synth_test.json --out runs/analysis/figures/degree_baseline.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def degree_features(edge_index: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Per-node in-/out-degree (parallel edges counted), fan-in/out (distinct neighbours) and total degree."""
    src, dst = edge_index
    pairs = np.unique(src.astype(np.int64) * n + dst)
    f = {
        "in_degree": np.bincount(dst, minlength=n),
        "out_degree": np.bincount(src, minlength=n),
        "fan_in": np.bincount(pairs % n, minlength=n),
        "fan_out": np.bincount(pairs // n, minlength=n),
    }
    f["total_degree"] = f["in_degree"] + f["out_degree"]
    return f


def signed_auroc(y: np.ndarray, score: np.ndarray) -> float:
    """AUROC of the better direction, max(A, 1 - A), the same convention as the dissection's signed units."""
    a = roc_auc_score(y, score)
    return float(max(a, 1.0 - a))


def degree_baseline(graph_path: Path) -> list[dict]:
    """Signed AUROC of every degree statistic for every concept of a synthetic split graph."""
    d = torch.load(graph_path, weights_only=False)
    feats = degree_features(d.edge_index.numpy(), d.num_nodes)
    y = d.y.numpy() > 0.5
    rows = []
    for j, c in enumerate(d.label_names):
        scores = {k: signed_auroc(y[:, j], v) for k, v in feats.items()}
        best = max(scores, key=scores.get)
        rows.append({"concept": c, "best_stat": best, "best_auroc": round(scores[best], 4),
                     **{f"auroc_{k}": round(v, 4) for k, v in scores.items()}})
    return rows


def add_coverage(rows: list[dict], dissection: list[dict]) -> list[dict]:
    """Append each rung's mean GNN-layer (h1..hL) AUROC coverage, plus its margin over the best degree statistic."""
    for row in rows:
        for rung in sorted({r["protocol"]["rung"] for r in dissection}):
            vals = [max(v for k, v in r["concepts"][row["concept"]]["auroc"]["per_layer"].items()
                        if k.startswith("h") and k != "h0")
                    for r in dissection if r["protocol"]["rung"] == rung]
            row[f"{rung}_gnn_auroc"] = round(float(np.mean(vals)), 4)
            row[f"{rung}_minus_degree"] = round(float(np.mean(vals)) - row["best_auroc"], 4)
    return rows


def main() -> None:
    """CLI entry point; see the module docstring."""
    p = argparse.ArgumentParser(description="Degree-only baselines for dissection coverage.")
    p.add_argument("--graph", type=Path, required=True, help="synthetic split file (e.g. data/processed/synthetic/test.pt)")
    p.add_argument("--dissection", type=Path, default=None, help="combined dissection .json to compare against")
    p.add_argument("--out", type=Path, required=True, help="output .csv")
    args = p.parse_args()
    rows = degree_baseline(args.graph)
    if args.dissection:
        rows = add_coverage(rows, json.loads(args.dissection.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["concept"], r["best_stat"], r["best_auroc"], {k: v for k, v in r.items() if k.endswith("_minus_degree")})


if __name__ == "__main__":
    main()
