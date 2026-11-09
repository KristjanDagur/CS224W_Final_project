"""Depth experiment: does a k-cycle detector need k+1 layers?

On the 6-layer R3 models, the first GNN layer with a unit at AUROC >= 0.99 was h(k+1) for C_k,
k = 2..5, in all three seeds, and C6 was never reached. The rule predicts that a 7-layer R3 detects
C6 at h7 and keeps C2-C5 at h3-h6. This module compares dissections of models that differ only in
depth, and writes:

* ``first_layer``: per concept and seed, the first layer h1..hL reaching ``--level`` AUROC, and the
  best AUROC at the last layer;
* ``f1``: test minority-class F1 per concept (mean ± sd over seeds) from each run's ``metrics.json``.

Usage::

    python -m amlinterp.analysis.depth \\
        --dissection L6=runs/analysis/dissection_synth_test.json L7=runs/analysis/dissection_synth_L7_test.json \\
        --rung R3 --out runs/analysis/depth_R3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CONCEPTS = ("deg_in", "deg_out", "fan_in", "fan_out", "C2", "C3", "C4", "C5", "C6", "scatter_gather", "biclique")


def first_layer(run: dict, concept: str, level: float) -> tuple[str, float, str]:
    """First GNN layer (h1..hL) whose best unit reaches ``level`` AUROC, the best AUROC there, and the last layer's name."""
    per_layer = run["concepts"][concept]["auroc"]["per_layer"]
    layers = sorted((k for k in per_layer if k.startswith("h") and k != "h0"), key=lambda k: int(k[1:]))
    hit = next((k for k in layers if per_layer[k] >= level), "none")
    return hit, per_layer[layers[-1]], layers[-1]


def test_f1(run_id: str, runs_dir: Path) -> dict[str, float]:
    """Per-concept test F1 (%) at threshold 0.5 from ``runs/<run_id>/metrics.json``."""
    return json.loads((runs_dir / run_id / "metrics.json").read_text())["test"]["f1"]


def main() -> None:
    """CLI entry point; see the module docstring."""
    p = argparse.ArgumentParser(description="Compare dissections of models that differ only in depth.")
    p.add_argument("--dissection", nargs="+", required=True, metavar="NAME=PATH", help="combined dissection .json per depth")
    p.add_argument("--rung", default="R3")
    p.add_argument("--level", type=float, default=0.99, help="AUROC that counts as a detector")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, required=True, help="output prefix (writes _first_layer.csv, _f1.csv, .md)")
    args = p.parse_args()

    first_rows, f1_rows, md = [], [], []
    md.append(f"First GNN layer with a unit at AUROC ≥ {args.level} (per seed; AUROC of the last layer in brackets)\n")
    md.append("| Concept | " + " | ".join(n.split("=")[0] for n in args.dissection) + " |")
    md.append("|---|" + "---|" * len(args.dissection))
    depths = {}
    for spec in args.dissection:
        name, path = spec.split("=", 1)
        runs = sorted((r for r in json.loads(Path(path).read_text()) if r["protocol"]["rung"] == args.rung),
                      key=lambda r: r["run_id"])
        depths[name] = runs
    for c in CONCEPTS:
        cells = []
        for name, runs in depths.items():
            parts = []
            for r in runs:
                hit, last_auc, last = first_layer(r, c, args.level)
                parts.append(f"{hit} [{last_auc:.3f}]")
                first_rows.append({"depth": name, "run_id": r["run_id"], "concept": c, "first_layer": hit,
                                   "last_layer": last, "auroc_last_layer": round(last_auc, 4)})
            cells.append(", ".join(parts))
        md.append(f"| {c} | " + " | ".join(cells) + " |")

    md.append("\nTest F1 (%) at 0.5, mean ± sd over seeds\n")
    md.append("| Concept | " + " | ".join(depths) + " |")
    md.append("|---|" + "---|" * len(depths))
    f1 = {name: [test_f1(r["run_id"], args.runs_dir) for r in runs] for name, runs in depths.items()}
    for c in CONCEPTS:
        cells = []
        for name in depths:
            v = np.array([m[c] for m in f1[name]])
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            cells.append(f"{v.mean():.1f} ± {sd:.1f}")
            f1_rows.append({"depth": name, "concept": c, "mean": round(float(v.mean()), 2), "sd": round(float(sd), 2),
                            "n_seeds": len(v)})
        md.append(f"| {c} | " + " | ".join(cells) + " |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for rows, suffix in ((first_rows, "_first_layer.csv"), (f1_rows, "_f1.csv")):
        with open(f"{args.out}{suffix}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    Path(f"{args.out}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
