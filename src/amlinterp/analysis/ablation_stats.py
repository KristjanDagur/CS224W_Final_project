"""Significance of the ablation drops (``analysis.ablation``): seed-level t-tests with FDR control.

The unit of replication is the training seed: each seed is an independently trained model with its
own selected units. For every model group (rung, depth), concept ``c`` and ablation size ``k``
(top-1, top-k), each seed ``s`` gives two differences:

* **Effect beyond chance (A):** ``d_s = drop_c(top-k units of c) - mean drop_c(random k-unit sets)``.
  Random sets have the same size, so the test asks whether *these* units matter more than *any* k
  units. Top-1 uses the random single units; top-k uses the random k-unit sets.
* **Specificity (B):** ``d_s = drop_c(top-k units of c) - mean over c' != c of drop_c'(same ablation)``.
  This asks whether removing c's units hurts c more than it hurts the other concepts.

Each is tested with a one-sided one-sample t-test, ``H1: mean(d) > 0``, df = n_seeds - 1. All tests
(both kinds, all groups, concepts and sizes) form one family, and the Benjamini-Hochberg procedure
controls the false discovery rate at ``q``.

With 3 seeds, df = 2, so only large and consistent effects reach significance. A non-significant test
means "not shown", not "no effect".

Usage::

    python -m amlinterp.analysis.ablation_stats --results runs/analysis/ablation_synth/*.json \\
        --out runs/analysis/ablation_synth_stats
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


def seed_differences(runs: list[dict], concept_idx: int, k: int, top_k: int) -> dict[str, np.ndarray]:
    """Per-seed differences for tests A and B, for one concept and ablation size ``k`` (1 or ``top_k``)."""
    key = "delta_top1" if k == 1 else f"delta_top{top_k}"
    a, b = [], []
    for r in runs:
        c = r["labels"][concept_idx]
        drops = np.asarray(r["concepts"][c][key], dtype=float)                 # (concepts,)
        own = drops[concept_idx]
        rand = np.asarray(r["random_deltas"] if k == 1 else r["random_sets"][str(k)]["deltas"], dtype=float)
        a.append(own - rand[:, concept_idx].mean())
        b.append(own - np.delete(drops, concept_idx).mean())
    return {"A_vs_random": np.array(a), "B_specificity": np.array(b)}


def one_sided_t(d: np.ndarray) -> tuple[float, float]:
    """One-sample t-test of ``mean(d) > 0``. Returns ``(t, p)``; ``p`` is NaN if it is undefined (n < 2 or zero variance)."""
    if len(d) < 2 or np.allclose(d, d[0]):
        return float("nan"), float("nan")
    res = stats.ttest_1samp(d, 0.0, alternative="greater")
    return float(res.statistic), float(res.pvalue)


def run_tests(results: list[dict], q: float = 0.05) -> list[dict]:
    """All tests with BH-FDR over the whole family. Returns one row per test."""
    top_k = int(results[0]["protocol"]["top_k"])
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        groups.setdefault((r["rung"], r["num_layers"]), []).append(r)
    rows = []
    for (rung, depth), runs in sorted(groups.items()):
        missing = [r["run_id"] for r in runs if str(top_k) not in r.get("random_sets", {})]
        if missing:
            raise ValueError(f"runs without the size-{top_k} random control: {missing}")
        for j, c in enumerate(runs[0]["labels"]):
            for k in (1, top_k):
                for test, d in seed_differences(runs, j, k, top_k).items():
                    t, p = one_sided_t(d)
                    rows.append({"rung": rung, "layers": depth, "concept": c, "k": k, "test": test,
                                 "n_seeds": len(d), "mean_diff": round(float(d.mean()), 3),
                                 "sd_diff": round(float(d.std(ddof=1)), 3) if len(d) > 1 else None,
                                 "per_seed": " / ".join(f"{x:.2f}" for x in d), "t": t, "p": p})
    p = np.array([r["p"] for r in rows])
    ok = ~np.isnan(p)
    qvals = np.full(len(p), np.nan)
    qvals[ok] = stats.false_discovery_control(p[ok], method="bh")
    for r, qv in zip(rows, qvals):
        r["q_bh"] = float(qv)
        r["significant"] = bool(qv <= q) if not np.isnan(qv) else False
        r["t"] = round(r["t"], 3) if not np.isnan(r["t"]) else None
        r["p"] = float(f"{r['p']:.3g}") if not np.isnan(r["p"]) else None
        r["q_bh"] = float(f"{qv:.3g}") if not np.isnan(qv) else None
    return rows


def main() -> None:
    """CLI entry point; see the module docstring."""
    p = argparse.ArgumentParser(description="Seed-level t-tests with BH-FDR for ablation drops.")
    p.add_argument("--results", type=Path, nargs="+", required=True, help="per-run ablation .json files")
    p.add_argument("--q", type=float, default=0.05, help="FDR level")
    p.add_argument("--out", type=Path, required=True, help="output prefix (writes .csv and .md)")
    args = p.parse_args()
    rows = run_tests([json.loads(f.read_text()) for f in args.results], args.q)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_sig = sum(r["significant"] for r in rows)
    n_undef = sum(r["p"] is None for r in rows)
    lines = [f"BH-FDR q = {args.q} over {len(rows) - n_undef} tests ({n_undef} undefined): {n_sig} significant.", "",
             "| Group | Concept | k | Test | Mean diff (per seed) | t | p | q | Sig. |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['rung']} L{r['layers']} | {r['concept']} | {r['k']} | {r['test']} | {r['mean_diff']} "
                     f"({r['per_seed']}) | {r['t']} | {r['p']} | {r['q_bh']} | {'✓' if r['significant'] else ''} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(lines[0])


if __name__ == "__main__":
    main()
