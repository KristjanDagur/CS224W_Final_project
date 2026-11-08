"""Pick the synthetic learning rate by parabolic interpolation in log10(lr).

Reads ``runs/lr_sweep/*/metrics.json`` (from ``scripts/slurm/synthetic_lr_sweep.sbatch``),
averages the best validation mean minority-F1 over the four rungs at each lr (one lr is
shared by every rung, so no rung is favored), and fits ``F1 = a x^2 + b x + c`` with
``x = log10(lr)``.

* If the parabola is concave (``a < 0``) and its vertex lies inside the tested range, the
  proposal is the vertex. Run it to confirm it beats the best tested point.
* Otherwise the proposal is the best tested lr, and the script suggests extending the range
  toward it instead of extrapolating.

After the proposal has been run on all rungs (seed 0), ``--select`` writes
``selected_lr.json`` with the lr that has the best observed mean over rungs among all lrs tested.
That covers both "the vertex did not beat the best tested point" and "it did".

Validation only; test metrics are never read. Usage::

    python scripts/fit_lr.py [--sweep-dir runs/lr_sweep] [--out runs/lr_sweep/lr_fit.png]
    python scripts/fit_lr.py --select
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical palette, slots 1-4 in fixed order (validated adjacent pairs), plus ink tokens.
RUNG_STYLE = {"R0": ("#2a78d6", "o"), "R1": ("#eb6834", "s"), "R2": ("#1baf7a", "^"), "R3": ("#eda100", "D")}
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
RUN_RE = re.compile(r"^synth_(R\d)_L\d+_h\d+_s\d+_lr(.+)$")


def load_sweep(sweep_dir: Path) -> dict[float, dict[str, float]]:
    """Map lr -> {rung: best validation mean F1} for every finished run in the sweep."""
    out: dict[float, dict[str, float]] = defaultdict(dict)
    for mpath in sorted(sweep_dir.glob("*/metrics.json")):
        match = RUN_RE.match(mpath.parent.name)
        if match:
            out[float(match.group(2))][match.group(1)] = json.loads(mpath.read_text())["val"]["mean_f1"]
    return dict(sorted(out.items()))


def propose(lrs: np.ndarray, score: np.ndarray) -> dict:
    """Fit a parabola in log10(lr) and return the proposed lr with the reasoning."""
    x = np.log10(lrs)
    a, b, c = np.polyfit(x, score, 2)
    best_i = int(score.argmax())
    res = {"coef": [float(a), float(b), float(c)], "best_tested_lr": float(lrs[best_i]),
           "best_tested_f1": float(score[best_i])}
    if a < 0 and x.min() < -b / (2 * a) < x.max():
        xv = -b / (2 * a)
        res.update(kind="vertex", lr=float(f"{10**xv:.2g}"), predicted_f1=float(np.polyval([a, b, c], xv)))
    else:
        side = "lower" if best_i == 0 else "higher" if best_i == len(lrs) - 1 else None
        res.update(kind="best_tested", lr=float(lrs[best_i]), predicted_f1=float(score[best_i]),
                   note=f"no interior maximum; extend the range to {side} lr" if side else "no interior maximum")
    return res


def plot(sweep: dict[float, dict[str, float]], lrs: np.ndarray, score: np.ndarray, res: dict, out: Path) -> None:
    """Per-rung curves, the cross-rung mean, the fitted parabola and the proposed lr."""
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for rung, (color, marker) in RUNG_STYLE.items():
        pts = [(lr, s[rung]) for lr, s in sweep.items() if rung in s]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=8, mec=SURFACE, mew=2, label=rung, zorder=3)
        ax.annotate(rung, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    xf = np.linspace(np.log10(lrs.min()), np.log10(lrs.max()), 200)
    ax.plot(10**xf, np.polyval(res["coef"], xf), color=INK_2, lw=2, ls="--", label="quadratic fit to mean", zorder=2)
    ax.plot(lrs, score, color=INK, ls="none", marker="o", ms=9, mec=SURFACE, mew=2, label="mean over rungs", zorder=4)
    ax.axvline(res["lr"], color=MUTED, lw=1, zorder=1)
    ax.annotate(f"proposed lr = {res['lr']:.2g}\n({res['kind'].replace('_', ' ')})", (res["lr"], ax.get_ylim()[0]),
                xytext=(6, 8), textcoords="offset points", fontsize=9, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log scale)", color=INK_2)
    ax.set_ylabel("best val mean minority F1 (%)", color=INK_2)
    ax.set_title("Synthetic ladder: learning-rate sweep (seed 0, validation)", color=INK, fontsize=11, loc="left")
    ax.grid(True, color="#e6e5e0", lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK_2)
    ax.legend(frameon=False, fontsize=8, loc="best", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)


def main() -> None:
    """CLI: print the table and the proposal, save the plot and ``lr_fit.json``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sweep-dir", type=Path, default=Path("runs/lr_sweep"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--select", action="store_true", help="write selected_lr.json (best observed mean) instead of fitting")
    args = p.parse_args()
    sweep = load_sweep(args.sweep_dir)
    complete = {lr: s for lr, s in sweep.items() if len(s) == len(RUNG_STYLE)}
    print(f"{'lr':>8} " + " ".join(f"{r:>6}" for r in RUNG_STYLE) + "   mean")
    for lr, s in sweep.items():
        row = " ".join(f"{s[r]:6.2f}" if r in s else "     -" for r in RUNG_STYLE)
        print(f"{lr:8.1e} {row}  " + (f"{np.mean(list(s.values())):6.2f}" if lr in complete else "  (incomplete)"))
    if len(complete) < 3:
        raise SystemExit(f"need >= 3 lrs with all four rungs finished, have {len(complete)}")
    lrs = np.array(list(complete))
    score = np.array([np.mean(list(s.values())) for s in complete.values()])
    if args.select:
        i = int(score.argmax())
        sel = {"lr": float(lrs[i]), "mean_val_f1": float(score[i]),
               "rule": "best observed mean val F1 over R0-R3 (seed 0) among all tested lrs",
               "tested": {f"{lr:g}": round(float(s), 2) for lr, s in zip(lrs, score)}}
        (args.sweep_dir / "selected_lr.json").write_text(json.dumps(sel, indent=2))
        print(json.dumps(sel, indent=2))
        return
    res = propose(lrs, score)
    print(json.dumps(res, indent=2))
    out = args.out or args.sweep_dir / "lr_fit.png"
    plot(sweep, lrs, score, res, out)
    (args.sweep_dir / "lr_fit.json").write_text(json.dumps({"sweep": {str(k): v for k, v in sweep.items()}, **res}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
