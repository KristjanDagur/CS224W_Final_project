"""Neuron dissection: how well does each hidden unit detect each concept? (Bau et al., 2017; Xuanyuan et al., 2023)

For unit ``k`` with activations ``a_k`` over the scored items and a binary concept mask ``C_c``:

* **IoU.** Binarize the unit as ``M_k = {i : a_k(i) > tau}`` and score ``|M_k & C_c| / |M_k | C_c|``.
  By default ``tau`` is *rate-matched*: the ``(1 - p_c)`` quantile of ``a_k``, where ``p_c`` is the
  concept's base rate, so the unit's mask has the concept's size. The synthetic concepts have base
  rates of 0.2-0.8, and a fixed small top quantile (Bau et al. use 0.5%) would cap IoU near 0.01.
  ``quantile=q`` gives the proposal's fixed top-quantile form instead. Ties at ``tau`` are
  excluded, so a constant unit gets IoU 0.
* **AP.** Average precision of ``a_k`` as a score for ``C_c``. It is threshold-free and is the better
  measure for rare concepts (AGENTS.md, fact 6). Chance level is ``p_c``.
* **AUROC.** The probability that a random concept item activates the unit more than a random
  non-concept item (ties count one half). Chance is 0.5 for every concept, so scores are comparable
  across concepts with different base rates (the synthetic ones range from 0.19 to 0.78). For very
  rare concepts it is dominated by the many easy negatives, so AP stays the headline metric there.
* **Sign.** Hidden states are not ReLU'd, so a unit can encode a concept by being *low*. With
  ``signed=True`` (default) each unit is scored as ``a_k`` and as ``-a_k``, and the better
  direction is kept (recorded as +1 / -1).

**Concept coverage** of a model is ``max_k score(k, c)`` over all units of all layers (and the same
max per layer). Because it is a max over hundreds of units, its chance level is far above the
single-unit chance, so ``permutation_null`` recomputes it with the concept masks shuffled across
items. ``bootstrap_coverage`` gives confidence intervals that re-select the best unit in every
resample, optionally resampling *groups* of items (AML: laundering attempts, AGENTS.md fact 6).

Usage::

    python -m amlinterp.analysis.dissection --acts runs/synth_R*_L6_h64_s*/activations_test.pt \\
        --out runs/analysis/dissection_synth_test

writes ``<out>.json`` (coverage per run, concept and layer) and ``<out>_summary.csv``
(rung x concept, mean and sd over seeds). Runs dissected separately (one ``--out`` per run, e.g. in
parallel) are merged with ``--combine runs/analysis/<dir>/*.json --out <prefix>``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

CHUNK = 64
METRICS = ("iou", "ap", "auroc")
"""Units processed at once in ``average_precision`` (bounds memory at about N * CHUNK * M floats)."""


# --------------------------------------------------------------------------- unit x concept scores


def average_precision(acts: np.ndarray, concepts: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Average precision of every unit as a score for every concept, with sklearn's tie handling.

    Equivalent to ``sklearn.metrics.average_precision_score(concepts[:, c], acts[:, k],
    sample_weight=weights)`` for each pair, vectorized over units and concepts.

    Args:
        acts: (N, K) unit activations.
        concepts: (N, M) binary concept masks.
        weights: optional (N,) non-negative item weights (bootstrap counts).

    Returns:
        (K, M) AP. Concepts with no (weighted) positives get NaN.
    """
    n, k_units = acts.shape
    y = concepts.astype(np.float64)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    wy = y * w[:, None]                                    # (N, M)
    total_pos = wy.sum(0)                                  # (M,)
    out = np.empty((k_units, y.shape[1]))
    rows = np.arange(n)[:, None]
    for start in range(0, k_units, CHUNK):
        a = acts[:, start:start + CHUNK]
        order = np.argsort(-a, axis=0, kind="stable")      # (N, k) descending
        s = np.take_along_axis(a, order, axis=0)
        # Index of the last item of each item's tie block (precision is evaluated there).
        is_last = np.ones_like(s, dtype=bool)
        is_last[:-1] = s[1:] != s[:-1]
        block_end = np.where(is_last, rows, n)
        block_end = np.minimum.accumulate(block_end[::-1], axis=0)[::-1]
        tp = np.cumsum(wy[order], axis=0)                  # (N, k, M)
        cw = np.cumsum(w[order], axis=0)                   # (N, k)
        tp_end = np.take_along_axis(tp, block_end[..., None], axis=0)
        cw_end = np.take_along_axis(cw, block_end, axis=0)[..., None]
        prec = np.divide(tp_end, cw_end, out=np.zeros_like(tp_end), where=cw_end > 0)
        out[start:start + a.shape[1]] = (wy[order] * prec).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return out / total_pos


def auroc(acts: np.ndarray, concepts: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """ROC AUC of every unit as a score for every concept (ties count one half).

    AUROC is the probability that a random concept item gets a higher activation than a random
    non-concept item (the Mann-Whitney statistic), so chance is 0.5 whatever the base rate.
    Equivalent to ``sklearn.metrics.roc_auc_score(concepts[:, c], acts[:, k], sample_weight=weights)``.

    Args:
        acts: (N, K) unit activations.
        concepts: (N, M) binary concept masks.
        weights: optional (N,) non-negative item weights (bootstrap counts).

    Returns:
        (K, M) AUROC. Concepts without both (weighted) positives and negatives get NaN.
    """
    n, k_units = acts.shape
    y = concepts.astype(np.float64)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    wy, wn = y * w[:, None], (1.0 - y) * w[:, None]       # (N, M)
    denom = wy.sum(0) * wn.sum(0)                          # (M,)
    out = np.empty((k_units, y.shape[1]))
    rows = np.arange(n)[:, None]
    for start in range(0, k_units, CHUNK):
        a = acts[:, start:start + CHUNK]
        order = np.argsort(a, axis=0, kind="stable")       # (N, k) ascending
        s = np.take_along_axis(a, order, axis=0)
        # First and last index of each item's tie block.
        is_first = np.ones_like(s, dtype=bool)
        is_first[1:] = s[1:] != s[:-1]
        is_last = np.ones_like(s, dtype=bool)
        is_last[:-1] = s[1:] != s[:-1]
        block_start = np.maximum.accumulate(np.where(is_first, rows, 0), axis=0)
        block_end = np.minimum.accumulate(np.where(is_last, rows, n)[::-1], axis=0)[::-1]
        cn = np.cumsum(wn[order], axis=0)                  # (N, k, M) negatives at or below
        cn = np.concatenate([np.zeros((1,) + cn.shape[1:]), cn], axis=0)  # cn[i] = negatives before i
        below = np.take_along_axis(cn, block_start[..., None], axis=0)    # strictly lower
        upto = np.take_along_axis(cn, block_end[..., None] + 1, axis=0)   # lower or tied
        out[start:start + a.shape[1]] = (wy[order] * 0.5 * (below + upto)).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return out / denom


def iou(acts: np.ndarray, concepts: np.ndarray, quantile: float | None = None,
        weights: np.ndarray | None = None) -> np.ndarray:
    """IoU between each unit's binarized activation mask and each concept mask.

    Args:
        acts: (N, K) unit activations.
        concepts: (N, M) binary concept masks.
        quantile: fraction of items each unit marks (its top ``quantile``). ``None`` matches each
            concept's base rate (see the module docstring).
        weights: optional (N,) item weights. Thresholds always come from the unweighted
            activations, so a bootstrap resample re-scores fixed unit masks.

    Returns:
        (K, M) IoU in [0, 1].
    """
    c = concepts.astype(np.float64)
    w = np.ones(acts.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    rates = c.mean(0) if quantile is None else np.full(c.shape[1], quantile)
    out = np.zeros((acts.shape[1], c.shape[1]))
    wc = c * w[:, None]
    for rate in np.unique(rates):
        cols = np.flatnonzero(rates == rate)
        tau = np.quantile(acts, 1.0 - rate, axis=0)        # (K,)
        b = (acts > tau).astype(np.float64)                 # (N, K)
        inter = b.T @ wc[:, cols]                           # (K, m)
        union = (b * w[:, None]).sum(0)[:, None] + wc[:, cols].sum(0)[None, :] - inter
        out[:, cols] = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return out


def unit_scores(acts: np.ndarray, concepts: np.ndarray, signed: bool = True, quantile: float | None = None,
                weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """IoU, AP and AUROC of every unit for every concept, keeping each unit's better direction.

    Returns:
        ``{metric}``: (K, M) scores for each metric in ``METRICS``; ``{metric}_sign``: (K, M) +1 if
        the unit detects the concept by being high, -1 if by being low.
    """
    acts = np.asarray(acts, dtype=np.float64)
    out = {}
    for name, fn in (("iou", lambda a: iou(a, concepts, quantile, weights)),
                     ("ap", lambda a: average_precision(a, concepts, weights)),
                     ("auroc", lambda a: auroc(a, concepts, weights))):
        pos = fn(acts)
        if signed:
            neg = 1.0 - pos if name == "auroc" else fn(-acts)  # AUROC(-a) = 1 - AUROC(a) exactly
            better = np.nan_to_num(neg, nan=-1) > np.nan_to_num(pos, nan=-1)
            out[name] = np.where(better, neg, pos)
            out[f"{name}_sign"] = np.where(better, -1, 1)
        else:
            out[name], out[f"{name}_sign"] = pos, np.ones_like(pos, dtype=int)
    return out


# --------------------------------------------------------------------------- coverage and baselines


def stack_units(acts: dict[str, torch.Tensor | np.ndarray]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Concatenate per-layer activations into one (N, K) matrix.

    Returns:
        ``(matrix, unit_names, unit_layer)``: names like ``"h3:17"``; ``unit_layer`` is each
        unit's layer name (for per-layer coverage).
    """
    mats, names, layers = [], [], []
    for layer, a in acts.items():
        a = a.float().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
        mats.append(a)
        names += [f"{layer}:{j}" for j in range(a.shape[1])]
        layers += [layer] * a.shape[1]
    return np.concatenate(mats, axis=1), names, np.array(layers)


def coverage(scores: dict[str, np.ndarray], unit_layer: np.ndarray | None = None) -> dict[str, dict]:
    """Best unit per concept, overall and per layer.

    Returns:
        For each metric in ``METRICS``: ``{"value": (M,), "unit": (M,) best-unit index,
        "sign": (M,), "per_layer": {layer: (M,)}}``.
    """
    out = {}
    for metric in METRICS:
        s = np.nan_to_num(scores[metric], nan=-np.inf)
        best = s.argmax(0)
        res = {"value": s.max(0), "unit": best, "sign": scores[f"{metric}_sign"][best, np.arange(s.shape[1])]}
        if unit_layer is not None:
            res["per_layer"] = {layer: s[unit_layer == layer].max(0) for layer in dict.fromkeys(unit_layer)}
        out[metric] = res
    return out


def chance_level(metric: str, rate: float, quantile: float | None = None) -> float:
    """Expected score of a single unit unrelated to the concept (base rate ``rate``).

    AP: ``rate``. AUROC: 0.5. IoU: a random mask of size ``q`` (``q = rate`` when rate-matched) overlaps the
    concept in ``q * rate`` of the items, so IoU = ``q*rate / (q + rate - q*rate)``.
    """
    if metric == "ap":
        return rate
    if metric == "auroc":
        return 0.5
    q = rate if quantile is None else quantile
    return q * rate / (q + rate - q * rate)


def permutation_null(acts: np.ndarray, concepts: np.ndarray, n_perm: int = 20, seed: int = 0,
                     **kwargs) -> dict[str, np.ndarray]:
    """Coverage (max over units) when concept masks are shuffled across items.

    This is the chance level of ``coverage``: it keeps every unit's activation distribution and
    every concept's base rate, and breaks only the item-level link between them.

    Returns:
        ``{metric: (n_perm, M)}`` for each metric in ``METRICS``.
    """
    rng = np.random.default_rng(seed)
    out = {m: [] for m in METRICS}
    for _ in range(n_perm):
        s = unit_scores(acts, concepts[rng.permutation(len(concepts))], **kwargs)
        for metric in out:
            out[metric].append(np.nanmax(s[metric], axis=0))
    return {k: np.stack(v) for k, v in out.items()}


def bootstrap_coverage(acts: np.ndarray, concepts: np.ndarray, n_boot: int = 100, groups: np.ndarray | None = None,
                       seed: int = 0, **kwargs) -> dict[str, np.ndarray]:
    """Bootstrap distribution of coverage, re-selecting the best unit in every resample.

    Args:
        acts, concepts: as in ``unit_scores``.
        n_boot: number of resamples.
        groups: optional (N,) group ID per item. Groups are resampled with replacement and all
            items of a drawn group enter together (AML: attempt-level CIs). ``None`` resamples items.
        seed: RNG seed.

    Returns:
        ``{metric: (n_boot, M)}`` for each metric in ``METRICS``.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(acts))
    uniq, inv = np.unique(groups, return_inverse=True)
    out = {m: [] for m in METRICS}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        s = unit_scores(acts, concepts, weights=counts[inv].astype(np.float64), **kwargs)
        for metric in out:
            out[metric].append(np.nanmax(s[metric], axis=0))
    return {k: np.stack(v) for k, v in out.items()}


# --------------------------------------------------------------------------- run-level analysis


def dissect_file(path: Path, n_boot: int = 100, n_perm: int = 20, quantile: float | None = None,
                 seed: int = 0) -> dict:
    """Dissect one activations file (``analysis.hooks``) against its task labels as concepts.

    For synthetic runs the concepts are the 11 pattern labels. The labels are binarized at 0.5
    and the positive class is the concept (also for C5/C6, whose minority class is 0).

    Returns:
        A JSON-ready dict: run metadata, concept base rates, and per concept the coverage (IoU, AP),
        best unit and its direction, per-layer coverage, bootstrap 95% CI and permutation-null
        95th percentile.
    """
    d = torch.load(path, weights_only=False)
    meta = d["meta"]
    acts, names, unit_layer = stack_units(d["acts"])
    concepts = (d["y"].float().numpy().reshape(len(acts), -1) > 0.5)
    label_names = meta["label_names"]
    kw = {"quantile": quantile}
    scores = unit_scores(acts, concepts, **kw)
    cov = coverage(scores, unit_layer)
    boot = bootstrap_coverage(acts, concepts, n_boot, seed=seed, **kw) if n_boot else None
    null = permutation_null(acts, concepts, n_perm, seed=seed, **kw) if n_perm else None
    rates = concepts.mean(0)
    per_concept = {}
    for j, c in enumerate(label_names):
        entry = {"base_rate": float(rates[j])}
        for m in METRICS:
            e = {
                "value": float(cov[m]["value"][j]),
                "unit": names[cov[m]["unit"][j]],
                "sign": int(cov[m]["sign"][j]),
                "per_layer": {layer: float(v[j]) for layer, v in cov[m]["per_layer"].items()},
                "single_unit_chance": chance_level(m, float(rates[j]), quantile),
            }
            if boot is not None:
                e["ci95"] = [float(x) for x in np.nanpercentile(boot[m][:, j], [2.5, 97.5])]
            if null is not None:
                e["null_p95"] = float(np.nanpercentile(null[m][:, j], 95))
            entry[m] = e
        per_concept[c] = entry
    return {
        "file": str(path), "run_id": meta["run_id"], "split": meta["split"], "protocol": meta["protocol"],
        "n_items": int(len(acts)), "n_units": int(acts.shape[1]), "units": meta["units"],
        "settings": {"quantile": quantile, "signed": True, "n_boot": n_boot, "n_perm": n_perm, "seed": seed},
        "concepts": per_concept,
    }


def _sd(x: np.ndarray) -> float:
    return round(float(x.std(ddof=1)), 4) if len(x) > 1 else 0.0


def summarize(results: list[dict], metric: str = "iou") -> list[dict]:
    """Rung x concept table: mean and sd over the runs (seeds) of each rung.

    ``mean``/``sd`` use coverage over all units; ``gnn_mean``/``gnn_sd`` only the GNN layers h1..hL.
    """
    by_rung: dict[str, list[dict]] = {}
    for r in results:
        by_rung.setdefault(r["protocol"]["rung"], []).append(r)
    rows = []
    for rung in sorted(by_rung):
        runs = by_rung[rung]
        for c in runs[0]["concepts"]:
            vals = np.array([r["concepts"][c][metric]["value"] for r in runs])
            # GNN layers only (h1..hL): the decoder layers sit next to the output, so any accurate
            # model covers its own target concepts there and the rung differences are hidden.
            gnn = np.array([max(v for k, v in r["concepts"][c][metric]["per_layer"].items()
                                if k.startswith("h") and k != "h0") for r in runs])
            nulls = [r["concepts"][c][metric].get("null_p95") for r in runs]
            rows.append({
                "rung": rung, "concept": c, "metric": metric, "n_runs": len(runs),
                "mean": round(float(vals.mean()), 4), "sd": _sd(vals),
                "gnn_mean": round(float(gnn.mean()), 4), "gnn_sd": _sd(gnn),
                "null_p95_mean": round(float(np.mean(nulls)), 4) if None not in nulls else None,
                "base_rate": round(runs[0]["concepts"][c]["base_rate"], 4),
            })
    return rows


def main() -> None:
    """CLI entry point; see the module docstring."""
    p = argparse.ArgumentParser(description="Neuron dissection of extracted activations.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--acts", type=Path, nargs="+", help="activations_<split>.pt files")
    src.add_argument("--combine", type=Path, nargs="+",
                     help="merge earlier --out .json files (e.g. one per run from a job array) instead")
    p.add_argument("--out", type=Path, required=True, help="output prefix (writes .json and _summary.csv)")
    p.add_argument("--n-boot", type=int, default=100)
    p.add_argument("--n-perm", type=int, default=20)
    p.add_argument("--quantile", type=float, default=None, help="fixed top quantile (default: rate-matched)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    results = []
    for f in args.combine or []:
        results.extend(json.loads(f.read_text()))
    for f in args.acts or []:
        r = dissect_file(f, args.n_boot, args.n_perm, args.quantile, args.seed)
        results.append(r)
        print(r["run_id"], {c: round(v["iou"]["value"], 3) for c, v in r["concepts"].items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(results, indent=2))
    rows = [row for m in METRICS for row in summarize(results, m)]
    with open(f"{args.out}_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out.with_suffix('.json')} and {args.out}_summary.csv")


if __name__ == "__main__":
    main()
