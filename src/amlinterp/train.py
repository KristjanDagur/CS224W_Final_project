"""Train and evaluate one ladder run: one config, one rung, one seed.

Usage::

    python -m amlinterp.train --config configs/synthetic.yaml --rung R3 --seed 1
    python -m amlinterp.train --config configs/synthetic.yaml --rung R0 --seed 1 --set epochs=2

Writes to ``runs/<run_id>/``, where ``run_id = {dataset}_{rung}_L{layers}_h{hidden}_s{seed}[_{tag}]``:

* ``config.json``: fully resolved config, with rung flags, ego mode, git commit and device;
* ``history.jsonl``: one line per epoch (train loss, val metrics, time);
* ``last.pt``: checkpoint written every epoch (model, optimizer, RNG states). A rerun
  resumes from it, so a job killed at its time limit can continue;
* ``best.pt``: model state with the best validation ``mean_f1`` at threshold 0.5;
* ``thresholds.json``: per-label thresholds tuned on validation for ``best.pt``;
* ``metrics.json``: val and test metrics of ``best.pt`` at 0.5 and at the tuned thresholds.

Model selection and threshold tuning use the validation split only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from amlinterp.models.gin import RUNGS, GINLadder

TASKS = {"synth": "amlinterp.tasks.synthetic:SyntheticTask", "amlhis": "amlinterp.tasks.aml:AMLTask"}


# --------------------------------------------------------------------------- setup helpers


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path: Path) -> None:
    """``torch.save`` to a temporary file, then rename it, so a job killed mid-write never leaves a corrupt checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def git_state() -> dict:
    """Current commit hash, and whether the working tree has uncommitted (or untracked, non-ignored) changes."""
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def _parse_value(value: str):
    """Parse a CLI override as YAML. Also accepts ``3e-4``, which YAML 1.1 would read as a string."""
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):
        try:
            return float(parsed)
        except ValueError:
            pass
    return parsed


def load_config(path: Path, rung: str, seed: int, overrides: list[str]) -> dict:
    """Read a YAML config, apply ``key=value`` overrides (values parsed as YAML), add rung and seed."""
    cfg = yaml.safe_load(path.read_text())
    for kv in overrides:
        key, _, value = kv.partition("=")
        cfg[key] = _parse_value(value)
    cfg["rung"], cfg["seed"] = rung, seed
    cfg.update({k: v for k, v in vars(RUNGS[rung]).items() if k != "name"})
    tag = f"_{cfg['tag']}" if cfg.get("tag") else ""
    cfg["run_id"] = f"{cfg['dataset']}_{rung}_L{cfg['num_layers']}_h{cfg['hidden']}_s{seed}{tag}"
    return cfg


def make_task(cfg: dict):
    """Instantiate the task class registered for ``cfg['dataset']``."""
    module, cls = TASKS[cfg["dataset"]].split(":")
    return getattr(__import__(module, fromlist=[cls]), cls)(cfg, RUNGS[cfg["rung"]])


# --------------------------------------------------------------------------- train / eval


def _probs_and_labels(task, logits, y):
    """Task-specific ``(labels, probabilities)`` as (N, T) arrays. The default is multi-label sigmoid."""
    probs = task.probs(logits) if hasattr(task, "probs") else torch.sigmoid(logits)
    labels = task.label_matrix(y) if hasattr(task, "label_matrix") else y
    return labels, probs


class _Coverage:
    """Counts seeds vs scored seeds for tasks that report ``task.last_batch_info`` (AML drops some seeds, as IBM does)."""

    def __init__(self) -> None:
        self.seeds = self.scored = 0

    def add(self, task) -> None:
        info = getattr(task, "last_batch_info", None)
        if info:
            self.seeds += info["seeds"]
            self.scored += info["scored"]

    def summary(self) -> dict | None:
        if not self.seeds:
            return None
        return {"seeds": self.seeds, "scored": self.scored, "frac_scored": round(self.scored / self.seeds, 5)}


def train_epoch(model, task, loader, optimizer, device) -> tuple[float, dict | None]:
    """One pass over the training loader. Returns the mean loss per scored seed and the seed coverage."""
    model.train()
    total, count, cov = 0.0, 0, _Coverage()
    for batch in loader:
        logits, y = task.forward(model, batch, device)
        cov.add(task)
        loss = task.loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * y.shape[0]
        count += y.shape[0]
    return total / count, cov.summary()


@torch.no_grad()
def predict(model, task, loader, device) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """Labels and probabilities for every scored seed of a loader, in loader order, plus the seed coverage."""
    model.eval()
    ys, probs, cov = [], [], _Coverage()
    for batch in loader:
        logits, y = task.forward(model, batch, device)
        cov.add(task)
        labels, prob = _probs_and_labels(task, logits, y)
        ys.append(labels.cpu())
        probs.append(prob.cpu())
    return torch.cat(ys).numpy(), torch.cat(probs).numpy(), cov.summary()


def _evaluate(task, y, prob, cov, thresholds=0.5) -> dict:
    """``task.metrics`` plus the seed coverage, if the task reports it."""
    out = task.metrics(y, prob, thresholds)
    if cov is not None:
        out["seed_coverage"] = cov
    return out


def run(cfg: dict) -> dict:
    """Train with per-epoch checkpoints, then evaluate the best checkpoint. Returns the final metrics."""
    from amlinterp.metrics import tune_thresholds

    run_dir = Path(cfg.get("runs_dir", "runs")) / cfg["run_id"]
    if (run_dir / "metrics.json").exists():
        # Finished runs are never re-evaluated: neighbor-sampled evaluation is stochastic, so a
        # rerun would silently change the reported numbers. Delete metrics.json to force it.
        print(f"[{cfg['run_id']}] already finished; keeping {run_dir / 'metrics.json'}")
        return json.loads((run_dir / "metrics.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if "SLURM_CPUS_PER_TASK" in os.environ:
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    cfg = {**cfg, "device": str(device), "git": git_state(), "torch": torch.__version__}

    seed_everything(cfg["seed"])
    task = make_task(cfg)
    model = GINLadder(
        node_in=task.node_in, edge_in=task.edge_in, out_dim=task.out_dim, hidden=cfg["hidden"],
        num_layers=cfg["num_layers"], reverse_mp=cfg["reverse_mp"], readout=task.readout,
        final_dropout=cfg["final_dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    cfg["num_params"] = sum(p.numel() for p in model.parameters())
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    start_epoch, best = 0, {"mean_f1": -1.0, "epoch": -1}
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)  # device-agnostic resume
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        print(f"resumed from epoch {ck['epoch']} (best val mean_f1 {best['mean_f1']:.2f} at epoch {best['epoch']})")

    train_loader = task.loader("train", shuffle=True)
    val_loader = task.loader("val", shuffle=False)
    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        loss, train_cov = train_epoch(model, task, train_loader, optimizer, device)
        val = _evaluate(task, *predict(model, task, val_loader, device))
        if val["mean_f1"] > best["mean_f1"]:
            best = {"mean_f1": val["mean_f1"], "epoch": epoch}
            atomic_save(model.state_dict(), run_dir / "best.pt")
        rec = {"epoch": epoch, "train_loss": round(loss, 5), "val_mean_f1": round(val["mean_f1"], 2),
               "val_f1": val["f1"], "best_epoch": best["epoch"], "sec": round(time.time() - t0, 1)}
        if train_cov is not None:
            rec.update(train_seed_coverage=train_cov["frac_scored"], val_seed_coverage=val["seed_coverage"]["frac_scored"])
        with open(run_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch:3d}  loss {loss:.4f}  val mean F1 {val['mean_f1']:6.2f}  best {best['mean_f1']:6.2f}@{best['epoch']}  {rec['sec']}s", flush=True)
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best": best,
                     "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}},
                    last)

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    y_val, p_val, cov_val = predict(model, task, val_loader, device)
    y_test, p_test, cov_test = predict(model, task, task.loader("test", shuffle=False), device)
    thr = tune_thresholds(y_val, p_val, task.minority)
    (run_dir / "thresholds.json").write_text(json.dumps(dict(zip(task.label_names, thr.tolist())), indent=2))
    results = {
        "run_id": cfg["run_id"], "best_epoch": best["epoch"],
        "val": _evaluate(task, y_val, p_val, cov_val), "test": _evaluate(task, y_test, p_test, cov_test),
        "val_tuned": _evaluate(task, y_val, p_val, cov_val, thr), "test_tuned": _evaluate(task, y_test, p_test, cov_test, thr),
    }
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"[{cfg['run_id']}] best epoch {best['epoch']}: test mean F1 {results['test']['mean_f1']:.2f} "
          f"(tuned thr {results['test_tuned']['mean_f1']:.2f})")
    print("test F1:", results["test"]["f1"])
    return results


def main() -> None:
    """CLI entry point; see the module docstring."""
    p = argparse.ArgumentParser(description="Train one ladder run.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--rung", choices=sorted(RUNGS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    args = p.parse_args()
    run(load_config(args.config, args.rung, args.seed, args.set))


if __name__ == "__main__":
    main()
