"""Figures for the dissection results (``analysis.dissection`` JSON).

Two static figures for the report, plus a CSV of every plotted number (the table view):

* ``coverage_<metric>``: rung x concept heatmap of concept coverage, the best single unit over
  the GNN layers h1..hL, mean over seeds. Cells outlined in black are the rung at which the theory
  (Egressy et al., 2024) first makes the concept detectable, so H1 predicts a jump there.
* ``layers_<rung>_<metric>``: concept x layer heatmap for one rung (where in depth each detector
  appears).
* ``pr_curves`` (needs ``--acts-select`` and ``--acts-eval``): one panel per concept with the
  precision-recall curve of each rung's best unit. The unit (and its sign) is **selected on one split
  (validation) and evaluated on another (test)**, so the curves and their AP are not inflated by
  picking the best of hundreds of units on the data they are scored on. Curves are the mean over
  seeds of precision at a fixed recall grid; the grey line is the concept's base rate (chance).

AUROC (chance 0.5 for every concept) is the headline metric on synthetic data. AP is shown *chance-corrected*, ``(AP - p) / (1 - p)`` with ``p`` the concept's base rate, so 0 is a
random unit and 1 a perfect detector; raw AP would make frequent concepts (C6, p = 0.78) look covered
by default. IoU is shown raw (its single-unit chance is in the JSON). A cell is greyed out with "–"
unless every seed's coverage exceeds that seed's permutation-null 95th percentile.

Usage::

    python -m amlinterp.analysis.plots --results runs/analysis/dissection_synth_test.json \\
        --out runs/analysis/figures --layers-rung R3 \\
        --acts-select 'runs/synth_R*_L6_h64_s*/activations_val.pt' \\
        --acts-eval 'runs/synth_R*_L6_h64_s*/activations_test.pt'
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score  # noqa: E402

from amlinterp.analysis.dissection import stack_units, unit_scores  # noqa: E402

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
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# Categorical slots 1-4 (blue, orange, aqua, yellow), fixed per rung; validated with the dataviz
# validator (adjacent CVD dE >= 9.1). Aqua and yellow are below 3:1 on the surface, so the figure
# carries a legend and a CSV table.
RUNG_COLORS = {"R0": "#2a78d6", "R1": "#eb6834", "R2": "#1baf7a", "R3": "#eda100"}
RECALL_GRID = np.linspace(0.0, 1.0, 201)


def _score(entry: dict, base_rate: float, metric: str, value: float) -> float:
    """Chance-corrected AP, or raw IoU / AUROC."""
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


# Colour-scale range per metric: chance-corrected AP and IoU run 0..1, AUROC runs from chance (0.5) to 1.
SCALE = {"ap": (0.0, 1.0), "iou": (0.0, 1.0), "auroc": (0.5, 1.0)}
WHAT = {"ap": "chance-corrected AP", "iou": "IoU", "auroc": "AUROC"}


def _norm(v: float, metric: str) -> float:
    lo, hi = SCALE[metric]
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def _cell_text_color(t: float) -> str:
    """Ink for a cell whose normalized colour position is ``t``."""
    return "#ffffff" if t > 0.62 else INK


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
            t = _norm(m, metric)
            color = CMAP(t) if sig else NS_FILL
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=color, edgecolor=SURFACE, linewidth=2))
            ax.text(j + 0.5, i + 0.5, f"{m:.2f}" if sig else "–", ha="center", va="center", fontsize=8,
                    color=_cell_text_color(t) if sig else MUTED)
            if UNLOCKED_AT.get(c) == rung:
                ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fill=False, edgecolor=INK, linewidth=1.6))
    ax.set_xlim(0, len(concepts))
    ax.set_ylim(len(rungs), 0)
    ax.set_xticks(np.arange(len(concepts)) + 0.5, [CONCEPT_LABELS[c] for c in concepts], fontsize=8, color=INK2)
    ax.set_yticks(np.arange(len(rungs)) + 0.5, [RUNG_LABELS.get(r, r) for r in rungs], fontsize=8, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    what = WHAT[metric]
    ax.set_title(f"Best single unit per concept ({what}, GNN layers, mean of seeds); "
                 "outline = rung where theory unlocks it", fontsize=8.5, color=INK, loc="left")
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(*SCALE[metric]))
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
            t = _norm(m, metric)
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=CMAP(t), edgecolor=SURFACE, linewidth=2))
            ax.text(j + 0.5, i + 0.5, f"{m:.2f}", ha="center", va="center", fontsize=6.5, color=_cell_text_color(t))
    ax.set_xlim(0, len(layers))
    ax.set_ylim(len(concepts), 0)
    ax.set_xticks(np.arange(len(layers)) + 0.5, layers, fontsize=8, color=INK2)
    ax.set_yticks(np.arange(len(concepts)) + 0.5, [CONCEPT_LABELS[c] for c in concepts], fontsize=8, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    what = WHAT[metric]
    ax.set_title(f"{RUNG_LABELS.get(rung, rung)}: best unit per layer ({what})", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
    return rows


def _gnn_units(acts: dict) -> dict:
    return {k: v for k, v in acts.items() if k.startswith("h") and k != "h0"}


def precision_at_recall(y: np.ndarray, score: np.ndarray, grid: np.ndarray = RECALL_GRID) -> np.ndarray:
    """Precision of the highest threshold that reaches each recall level (no interpolation).

    sklearn returns points in increasing-threshold order, so recall is non-increasing along the
    arrays. With tied scores several thresholds share one recall level (e.g. every threshold below
    a tied block that holds all positives has recall 1), so the point must be chosen by threshold,
    not by recall alone. The final sentinel point (recall 0, precision 1) has no threshold and is
    dropped.
    """
    prec, rec, _ = precision_recall_curve(y, score)
    prec, rec = prec[:-1], rec[:-1]
    n_reach = np.searchsorted(-rec, -grid, side="right")   # points with rec >= r (-rec is non-decreasing)
    return prec[np.clip(n_reach - 1, 0, len(rec) - 1)]   # the last of them = highest threshold


def pr_curves(select_files: list[Path], eval_files: list[Path]) -> tuple[dict, list[dict]]:
    """Select each run's best GNN unit per concept (AP, signed) on one split and score it on another.

    Args:
        select_files, eval_files: activation files of the same runs (matched by run ID), e.g. the
            validation and test splits.

    Returns:
        ``(curves, rows)``: ``curves[rung][concept]`` is a list over seeds of precision on
        ``RECALL_GRID``; ``rows`` has one record per run and concept (selected unit, sign, AP and
        AUROC on both splits, base rate on the evaluation split).
    """
    by_run = {}
    for f in eval_files:
        d = torch.load(f, weights_only=False)
        by_run[d["meta"]["run_id"]] = d
    curves: dict[str, dict[str, list]] = {}
    rows = []
    for f in select_files:
        sel = torch.load(f, weights_only=False)
        run_id = sel["meta"]["run_id"]
        if run_id not in by_run:
            continue
        ev = by_run[run_id]
        rung = sel["meta"]["protocol"]["rung"]
        a_sel, names, _ = stack_units(_gnn_units(sel["acts"]))
        a_ev, names_ev, _ = stack_units(_gnn_units(ev["acts"]))
        assert names == names_ev
        y_sel = sel["y"].float().numpy().reshape(len(a_sel), -1) > 0.5
        y_ev = ev["y"].float().numpy().reshape(len(a_ev), -1) > 0.5
        s = unit_scores(a_sel, y_sel)
        for j, c in enumerate(sel["meta"]["label_names"]):
            k = int(np.nanargmax(s["ap"][:, j]))
            sign = int(s["ap_sign"][k, j])
            score = sign * a_ev[:, k].astype(np.float64)
            curves.setdefault(rung, {}).setdefault(c, []).append(precision_at_recall(y_ev[:, j], score))
            rows.append({
                "rung": rung, "run_id": run_id, "concept": c, "unit": names[k], "sign": sign,
                "ap_select": round(float(s["ap"][k, j]), 4),
                "ap_eval": round(float(average_precision_score(y_ev[:, j], score)), 4),
                "auroc_eval": round(float(roc_auc_score(y_ev[:, j], score)), 4),
                "base_rate_eval": round(float(y_ev[:, j].mean()), 4),
                "split_select": sel["meta"]["split"], "split_eval": ev["meta"]["split"],
            })
    return curves, rows


def plot_pr(curves: dict, rows: list[dict], path: Path) -> None:
    """Small multiples: one panel per concept, one curve per rung (mean over seeds)."""
    rungs = sorted(curves)
    concepts = [c for c in CONCEPT_LABELS if c in curves[rungs[0]]]
    ncol = 4
    nrow = int(np.ceil((len(concepts) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.2, 2.25 * nrow), dpi=200, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = axes.ravel()
    for ax, c in zip(axes, concepts):
        ax.set_facecolor(SURFACE)
        base = np.mean([r["base_rate_eval"] for r in rows if r["concept"] == c])
        ax.axhline(base, color=AXIS, linewidth=1.0, zorder=1)
        for rung in rungs:
            prec = np.mean(curves[rung][c], axis=0)
            ax.plot(RECALL_GRID, prec, color=RUNG_COLORS.get(rung, MUTED), linewidth=1.4,
                    solid_joinstyle="round", solid_capstyle="round", zorder=2)
        ax.set_title(CONCEPT_LABELS[c], fontsize=8, color=INK, loc="left", pad=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(labelsize=6.5, colors=INK2, length=0)
        for side, sp in ax.spines.items():
            sp.set_visible(side in ("left", "bottom"))
            sp.set_color(AXIS)
            sp.set_linewidth(0.6)
    for ax in axes[len(concepts):]:
        ax.axis("off")
    # Legend in the spare panel: one entry per rung, plus chance (per-run AP/AUROC are in the CSV).
    legend_ax = axes[len(concepts)]
    handles = [plt.Line2D([], [], color=RUNG_COLORS.get(r, MUTED), linewidth=1.8) for r in rungs]
    handles.append(plt.Line2D([], [], color=AXIS, linewidth=1.0))
    legend_ax.legend(handles, [RUNG_LABELS.get(r, r) for r in rungs] + ["base rate (chance)"],
                     loc="center left", frameon=False, fontsize=7.5, labelcolor=INK2)
    for i, ax in enumerate(axes[:len(concepts)]):
        if i + ncol >= len(concepts):                       # lowest panel in its column
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel("recall", fontsize=7, color=INK2)
    for ax in axes[::ncol]:
        ax.set_ylabel("precision", fontsize=7, color=INK2)
    split_sel, split_ev = rows[0]["split_select"], rows[0]["split_eval"]
    fig.suptitle(f"Precision-recall of the best single GNN unit per concept (unit chosen on {split_sel}, "
                 f"curve on {split_ev}; mean of seeds)", fontsize=8.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)


def _expand(patterns: list[str]) -> list[Path]:
    return [Path(f) for p in patterns for f in sorted(glob.glob(p))]


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
    p.add_argument("--metrics", nargs="+", default=["auroc", "ap", "iou"], choices=["auroc", "ap", "iou"])
    p.add_argument("--layers-rung", nargs="*", default=["R3"], help="rungs to draw the per-layer figure for")
    p.add_argument("--acts-select", nargs="*", default=[],
                   help="activation files (globs) used to choose each concept's unit for the PR curves")
    p.add_argument("--acts-eval", nargs="*", default=[], help="activation files (globs) the PR curves are drawn on")
    args = p.parse_args()
    results = json.loads(args.results.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if args.acts_select and args.acts_eval:
        curves, rows = pr_curves(_expand(args.acts_select), _expand(args.acts_eval))
        plot_pr(curves, rows, args.out / "pr_curves")
        _write_csv(rows, args.out / "pr_curves.csv")
    for metric in args.metrics:
        rows = plot_coverage(coverage_table(results, metric), metric, args.out / f"coverage_{metric}")
        _write_csv(rows, args.out / f"coverage_{metric}.csv")
        for rung in args.layers_rung:
            rows = plot_layers(results, rung, metric, args.out / f"layers_{rung}_{metric}")
            _write_csv(rows, args.out / f"layers_{rung}_{metric}.csv")
    print(f"wrote figures and tables to {args.out}")


if __name__ == "__main__":
    main()
