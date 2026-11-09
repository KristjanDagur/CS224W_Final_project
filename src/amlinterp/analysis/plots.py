"""Figures for the dissection results (``analysis.dissection`` JSON).

Two static figures for the report, plus a CSV of every plotted number (the table view):

* ``coverage_<metric>``: rung x concept heatmap of concept coverage, the best single unit over
  the GNN layers h1..hL, mean over seeds. Cells outlined in black are the rung at which the theory
  (Egressy et al., 2024) first makes the concept detectable, so H1 predicts a jump there.
* ``layers_<rung>_<metric>``: concept x layer heatmap for one rung (where in depth each detector
  appears).

AP is shown *chance-corrected*, ``(AP - p) / (1 - p)`` with ``p`` the concept's base rate, so 0 is a
random unit and 1 a perfect detector; raw AP would make frequent concepts (C6, p = 0.78) look covered
by default. IoU is shown raw (its single-unit chance is in the JSON). A cell is greyed out with "–"
unless every seed's coverage exceeds that seed's permutation-null 95th percentile.

Usage::

    python -m amlinterp.analysis.plots --results runs/analysis/dissection_synth_test.json \\
        --out runs/analysis/figures --layers-rung R3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

CONCEPT_LABELS = {
    "deg_in": "deg-in", "deg_out": "deg-out", "fan_in": "fan-in", "fan_out": "fan-out",
    "C2": "C2", "C3": "C3", "C4": "C4", "C5": "C5", "C6": "C6",
    "scatter_gather": "S-G", "biclique": "B-C",
}
RUNG_LABELS = {"R0": "R0 GIN", "R1": "R1 +reverse", "R2": "R2 +ports", "R3": "R3 +ego IDs"}
# First rung at which each synthetic concept becomes detectable (Egressy et al., 2024, Fig. 1/Table 1).
UNLOCKED_AT = {
    "deg_in": "R0", "deg_out": "R1", "fan_in": "R2", "fan_out": "R2",
    "C2": "R3", "C3": "R3", "C4": "R3", "C5": "R3", "C6": "R3", "scatter_gather": "R3", "biclique": "R3",
}

# Sequential blue ramp (dataviz reference palette, steps 100-700) and chart ink.
_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", _BLUES)
INK, INK2, MUTED, SURFACE, NS_FILL = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#f0efec"


def _score(entry: dict, base_rate: float, metric: str, value: float) -> float:
    """Chance-corrected AP, or raw IoU."""
    if metric == "ap":
        return (value - base_rate) / (1 - base_rate)
    return value


def gnn_coverage(run: dict, concept: str, metric: str) -> tuple[float, str]:
    """Best score over GNN layers h1..hL (excluding the input h0 and the decoder) and its layer."""
    per_layer = run["concepts"][concept][metric]["per_layer"]
    layers = [k for k in per_layer if k.startswith("h") and k != "h0"]
    best = max(layers, key=lambda k: per_layer[k])
    return per_layer[best], best


def coverage_table(results: list[dict], metric: str) -> dict:
    """Per rung and concept: per-seed chance-corrected GNN-layer coverage, null and best layers."""
    table: dict[str, dict[str, dict]] = {}
    for r in results:
        rung = r["protocol"]["rung"]
        for c, e in r["concepts"].items():
            v, layer = gnn_coverage(r, c, metric)
            p = e["base_rate"]
            cell = table.setdefault(rung, {}).setdefault(c, {"vals": [], "null": [], "layers": [], "seeds": []})
            cell["vals"].append(_score(e, p, metric, v))
            cell["null"].append(_score(e, p, metric, e[metric]["null_p95"]) if "null_p95" in e[metric] else np.nan)
            cell["layers"].append(layer)
            cell["seeds"].append(r["run_id"])
    return table


def _cell_text_color(v: float) -> str:
    return "#ffffff" if v > 0.62 else INK


def plot_coverage(table: dict, metric: str, path: Path) -> list[dict]:
    """Rung x concept heatmap; returns the plotted rows."""
    rungs = sorted(table)
    concepts = [c for c in CONCEPT_LABELS if c in table[rungs[0]]]
    fig, ax = plt.subplots(figsize=(9.2, 3.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    rows = []
    for i, rung in enumerate(rungs):
        for j, c in enumerate(concepts):
            cell = table[rung][c]
            vals, nulls = np.array(cell["vals"]), np.array(cell["null"])
            m, sd = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            sig = bool(np.all(vals > nulls))
            rows.append({"rung": rung, "concept": c, "metric": metric, "mean": round(m, 4), "sd": round(sd, 4),
                         "above_null_all_seeds": sig, "best_layers": " ".join(cell["layers"]),
                         "n_seeds": len(vals)})
            color = CMAP(max(m, 0.0)) if sig else NS_FILL
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=color, edgecolor=SURFACE, linewidth=2))
            ax.text(j + 0.5, i + 0.5, f"{m:.2f}" if sig else "–", ha="center", va="center", fontsize=8,
                    color=_cell_text_color(m) if sig else MUTED)
            if UNLOCKED_AT.get(c) == rung:
                ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fill=False, edgecolor=INK, linewidth=1.6))
    ax.set_xlim(0, len(concepts))
    ax.set_ylim(len(rungs), 0)
    ax.set_xticks(np.arange(len(concepts)) + 0.5, [CONCEPT_LABELS[c] for c in concepts], fontsize=8, color=INK2)
    ax.set_yticks(np.arange(len(rungs)) + 0.5, [RUNG_LABELS.get(r, r) for r in rungs], fontsize=8, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    what = "chance-corrected AP" if metric == "ap" else "IoU"
    ax.set_title(f"Best single unit per concept ({what}, GNN layers, mean of seeds); "
                 "outline = rung where theory unlocks it", fontsize=8.5, color=INK, loc="left")
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK2, length=0)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
    return rows


def plot_layers(results: list[dict], rung: str, metric: str, path: Path) -> list[dict]:
    """Concept x layer heatmap for one rung (mean over its seeds); returns the plotted rows."""
    runs = [r for r in results if r["protocol"]["rung"] == rung]
    layers = [k for k in runs[0]["concepts"]["deg_in"][metric]["per_layer"] if k.startswith("h")]
    concepts = [c for c in CONCEPT_LABELS if c in runs[0]["concepts"]]
    grid = np.array([[[_score(r["concepts"][c], r["concepts"][c]["base_rate"], metric,
                              r["concepts"][c][metric]["per_layer"][layer]) for layer in layers]
                      for c in concepts] for r in runs])  # (seeds, concepts, layers)
    mean = grid.mean(0)
    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    rows = []
    for i, c in enumerate(concepts):
        for j, layer in enumerate(layers):
            m = float(mean[i, j])
            rows.append({"rung": rung, "concept": c, "layer": layer, "metric": metric, "mean": round(m, 4),
                         "sd": round(float(grid[:, i, j].std(ddof=1)), 4) if len(runs) > 1 else 0.0})
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=CMAP(max(m, 0.0)), edgecolor=SURFACE, linewidth=2))
            ax.text(j + 0.5, i + 0.5, f"{m:.2f}", ha="center", va="center", fontsize=6.5, color=_cell_text_color(m))
    ax.set_xlim(0, len(layers))
    ax.set_ylim(len(concepts), 0)
    ax.set_xticks(np.arange(len(layers)) + 0.5, layers, fontsize=8, color=INK2)
    ax.set_yticks(np.arange(len(concepts)) + 0.5, [CONCEPT_LABELS[c] for c in concepts], fontsize=8, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    what = "chance-corrected AP" if metric == "ap" else "IoU"
    ax.set_title(f"{RUNG_LABELS.get(rung, rung)}: best unit per layer ({what})", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    """CLI entry point; see the module docstring."""
    p = argparse.ArgumentParser(description="Plot dissection results.")
    p.add_argument("--results", type=Path, required=True, help="combined dissection .json")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--metrics", nargs="+", default=["ap", "iou"], choices=["ap", "iou"])
    p.add_argument("--layers-rung", nargs="*", default=["R3"], help="rungs to draw the per-layer figure for")
    args = p.parse_args()
    results = json.loads(args.results.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for metric in args.metrics:
        rows = plot_coverage(coverage_table(results, metric), metric, args.out / f"coverage_{metric}")
        _write_csv(rows, args.out / f"coverage_{metric}.csv")
        for rung in args.layers_rung:
            rows = plot_layers(results, rung, metric, args.out / f"layers_{rung}_{metric}")
            _write_csv(rows, args.out / f"layers_{rung}_{metric}.csv")
    print(f"wrote figures and tables to {args.out}")


if __name__ == "__main__":
    main()
