"""AML ladder results table: our runs next to Egressy et al. (2024) Table 2.

For each rung, over its seeds (mean ± sd, sd with ddof=1):

* **F1 @ 0.5.** Laundering-class F1 on test, flagging when P(laundering) > 0.5. This is IBM's
  argmax and the number Egressy Table 2 reports, so it is the comparable column.
* **F1 @ val-tuned threshold.** The same F1 with the cut-off chosen to maximize validation F1
  (grid 0.05..0.95, ``metrics.tune_thresholds``), plus the threshold each seed picked.
* **PR-AUC.** Test average precision, threshold-free. A random scorer's expected AP equals the
  positive rate, so the table prints the test illicit ratio as the "random" column.

Usage::

    python -m amlinterp.analysis.aml_table --runs 'runs/amlhis_R?_L2_h64_s?' \\
        --stats data/processed/aml_hi_small/stats.json --out runs/analysis/aml_ladder_table
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

EGRESSY_TABLE2 = {"R0": (28.70, 1.13), "R1": (46.79, 4.97), "R2": (56.85, 2.64), "R3": (57.15, 4.99)}
"""Egressy et al. (2024) Table 2, AML Small HI, minority-class F1 (%), mean ± sd over 5 seeds."""
RUNG_NAMES = {"R0": "GIN", "R1": "+reverse MP", "R2": "+ports", "R3": "+ego IDs"}


def load_runs(patterns: list[str]) -> dict[str, list[dict]]:
    """Finished runs (those with ``metrics.json``) grouped by rung, sorted by seed."""
    by_rung: dict[str, list[dict]] = {}
    for d in sorted({p for pat in patterns for p in glob.glob(pat)}):
        d = Path(d)
        if not (d / "metrics.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        m = json.loads((d / "metrics.json").read_text())
        thr = json.loads((d / "thresholds.json").read_text())["laundering"]
        by_rung.setdefault(cfg["rung"], []).append({
            "seed": cfg["seed"], "run_id": m["run_id"], "best_epoch": m["best_epoch"],
            "f1": m["test"]["mean_f1"], "f1_tuned": m["test_tuned"]["mean_f1"], "threshold": thr,
            "pr_auc": m["test"]["mean_pr_auc"], "val_f1": m["val"]["mean_f1"],
            "gpu": cfg.get("host", {}).get("gpu"),
        })
    return {r: sorted(v, key=lambda x: x["seed"]) for r, v in sorted(by_rung.items())}


def _ms(x: list[float]) -> str:
    """'mean ± sd' (sd only with two or more values)."""
    a = np.asarray(x, dtype=float)
    return f"{a.mean():.2f} ± {a.std(ddof=1):.2f}" if len(a) > 1 else f"{a.mean():.2f}"


def build_table(by_rung: dict[str, list[dict]], random_ap: float) -> list[dict]:
    """One row per rung with the reported columns (also used for the CSV)."""
    rows = []
    for rung, runs in by_rung.items():
        ref = EGRESSY_TABLE2.get(rung)
        rows.append({
            "rung": f"{rung} {RUNG_NAMES.get(rung, '')}".strip(),
            "seeds": " ".join(str(r["seed"]) for r in runs),
            "f1_at_0.5": _ms([r["f1"] for r in runs]),
            "egressy_table2": f"{ref[0]:.2f} ± {ref[1]:.2f}" if ref else "",
            "f1_val_tuned": _ms([r["f1_tuned"] for r in runs]),
            "tuned_thresholds": " / ".join(f"{r['threshold']:.2f}" for r in runs),
            "test_pr_auc": _ms([r["pr_auc"] for r in runs]),
            "pr_auc_random": f"{random_ap:.2f}",
            "best_epochs": " / ".join(str(r["best_epoch"]) for r in runs),
        })
    return rows


def to_markdown(rows: list[dict], random_ap: float) -> str:
    """Markdown table for the report, with a note on how to read each column."""
    head = ("| Rung | Seeds | F1 @ 0.5 | Egressy Table 2 | F1 @ val-tuned thr. | Thresholds picked | "
            "Test PR-AUC | PR-AUC of random |")
    lines = [head, "|" + "---|" * 8]
    for r in rows:
        lines.append(f"| {r['rung']} | {r['seeds']} | {r['f1_at_0.5']} | {r['egressy_table2']} | {r['f1_val_tuned']} | "
                     f"{r['tuned_thresholds']} | {r['test_pr_auc']} | {r['pr_auc_random']} |")
    lines.append("")
    lines.append("All values are percentages, i.e. ×100.")
    lines.append("- **F1 @ 0.5** is laundering-class F1 when a transaction is flagged at P > 0.5. This is IBM's "
                 "metric, the one Egressy Table 2 reports.")
    lines.append("- **F1 @ val-tuned thr.** uses each seed's cut-off maximizing validation F1. The picked "
                 "cut-offs are listed per seed.")
    lines.append(f"- **PR-AUC** is threshold-free. A random scorer gets the positive rate, {random_ap:.2f}, "
                 "which is the test illicit ratio.")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point; see the module docstring."""
    p = argparse.ArgumentParser(description="AML ladder table vs Egressy Table 2.")
    p.add_argument("--runs", nargs="+", default=["runs/amlhis_R?_L2_h64_s?"], help="run directories (globs)")
    p.add_argument("--stats", type=Path, default=Path("data/processed/aml_hi_small/stats.json"))
    p.add_argument("--out", type=Path, default=Path("runs/analysis/aml_ladder_table"), help="output prefix")
    args = p.parse_args()
    random_ap = 100 * json.loads(args.stats.read_text())["per_split"]["test"]["illicit_ratio"]
    by_rung = load_runs(args.runs)
    rows = build_table(by_rung, random_ap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = to_markdown(rows, random_ap)
    args.out.with_suffix(".md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
