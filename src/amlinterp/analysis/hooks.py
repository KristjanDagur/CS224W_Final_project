"""Activation extraction for trained ladder runs.

For every scored item of a split (synthetic: nodes; AML: transactions), this records

* ``h0 ... hL``: the node states after the embedding and after each message-passing layer,
  taken at the item (synthetic: ``h_l(v)``; AML: ``[h_l(u) || h_l(v)]`` for the edge ``u -> v``);
* ``dec1``, ``dec2``: the two hidden ReLU layers of the readout MLP (50 and 25 units), captured
  with forward hooks;
* the logits, the labels and the item IDs (node or global edge indices), so activations can be
  joined with any concept mask computed on the split graph.

Ego IDs make activations depend on the sampling protocol (AGENTS.md, fact 1). In ``exact`` mode
(disjoint subgraphs) an item's activations do not depend on which other items share its batch;
in ``batch`` mode they do, and AML neighbor sampling is also random. Every saved file therefore
carries ``meta`` with the ego mode, the batch protocol (batch size, disjoint, neighbor fan-out),
the extraction seed, the checkpoint and the git state, and a file without it should not be used.

Usage::

    python -m amlinterp.analysis.hooks --run runs/synth_R3_L6_h64_s1 --split test

writes ``runs/<run_id>/activations_<split>.pt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from amlinterp.train import build_model, git_state, make_task, seed_everything

DECODER_UNITS = {"dec1": 1, "dec2": 4}
"""Readout-MLP layers to record, as indices of the ReLU modules in ``GINLadder.decoder``."""


def load_run(run_dir: Path, device: torch.device | str = "cpu", checkpoint: str = "best.pt", **overrides):
    """Rebuild a finished run's task and model from ``config.json`` and a checkpoint.

    Args:
        run_dir: ``runs/<run_id>``.
        device: where to put the model.
        checkpoint: state-dict file inside ``run_dir`` (``best.pt`` is the selected model).
        **overrides: config entries to replace, e.g. ``data_dir`` when the data moved, or
            ``ego_mode``/``eval_batch_size`` to change the extraction protocol.

    Returns:
        ``(cfg, task, model)`` with the model in eval mode.
    """
    cfg = {**json.loads((Path(run_dir) / "config.json").read_text()), **overrides}
    task = make_task(cfg)
    model = build_model(cfg, task)
    model.load_state_dict(torch.load(Path(run_dir) / checkpoint, map_location="cpu", weights_only=True))
    return cfg, task, model.to(device).eval()


def protocol(cfg: dict, task) -> dict:
    """The batch protocol that activations of this config depend on (stored with every extraction)."""
    return {
        "dataset": cfg["dataset"],
        "rung": cfg["rung"],
        "ego": bool(cfg.get("ego", task.rung.ego)),
        "ego_mode": cfg["ego_mode"],
        "item": "node" if task.readout == "node" else "edge",
        "eval_batch_size": cfg["eval_batch_size"],
        "disjoint": cfg["ego_mode"] == "exact" and task.readout == "node",
        "num_neighbors": cfg.get("num_neighbors", "full"),
        "num_layers": cfg["num_layers"],
    }


@torch.no_grad()
def extract(model: nn.Module, task, split: str, device: torch.device | str = "cpu", seed: int = 0,
            dtype: torch.dtype = torch.float32) -> dict:
    """Run a split through the model and collect activations of every scored item.

    Args:
        model: a ``GINLadder`` (put in eval mode here).
        task: the task the model was trained on (``SyntheticTask`` or ``AMLTask``).
        split: ``"train"``, ``"val"`` or ``"test"``.
        device: device to run on.
        seed: seeds the loader and the neighbor sampler, so extraction is repeatable.
        dtype: storage dtype of the activations (``torch.float16`` halves the size of AML dumps).

    Returns:
        ``{"acts": {name: (N, d) tensor}, "logits": (N, out), "y": (N, ...), "ids": (N,)}``,
        with items in loader order. ``acts`` keys are ``h0 ... hL``, then ``dec1``, ``dec2``.
    """
    model.eval()
    seed_everything(seed)
    captured: dict[str, Tensor] = {}
    handles = [
        model.decoder[i].register_forward_hook(lambda _m, _inp, out, name=name: captured.__setitem__(name, out))
        for name, i in DECODER_UNITS.items()
    ]
    chunks: dict[str, list[Tensor]] = {}
    try:
        for batch in task.loader(split, shuffle=False):
            logits, y, hidden, ids = task.forward(model, batch, device, return_hidden=True)
            parts = {f"h{l}": h for l, h in enumerate(hidden)} | captured
            parts |= {"logits": logits, "y": y, "ids": ids}
            for name, value in parts.items():
                is_act = name.startswith(("h", "dec"))
                chunks.setdefault(name, []).append(value.to("cpu", dtype if is_act else value.dtype))
    finally:
        for h in handles:
            h.remove()
    out = {name: torch.cat(vals) for name, vals in chunks.items()}
    acts = {k: out.pop(k) for k in list(out) if k.startswith(("h", "dec"))}
    return {"acts": acts, **out}


def extract_run(run_dir: Path, split: str, device: torch.device | str = "cpu", seed: int = 0,
                dtype: torch.dtype = torch.float32, out: Path | None = None, **overrides) -> Path:
    """Extract activations of a finished run and save them with their metadata.

    Args:
        run_dir: ``runs/<run_id>``.
        split: which split to run.
        device, seed, dtype: see ``extract``.
        out: output file (default ``run_dir/activations_<split>.pt``; a tag is added when the
            protocol is overridden, so a batch-mode dump never overwrites an exact-mode one).
        **overrides: config overrides passed to ``load_run``.

    Returns:
        The path of the saved file.
    """
    run_dir = Path(run_dir)
    cfg, task, model = load_run(run_dir, device, **overrides)
    result = extract(model, task, split, device, seed, dtype)
    tag = "".join(f"_{k}{v}" for k, v in sorted(overrides.items()) if k != "data_dir")
    out = out or run_dir / f"activations_{split}{tag}.pt"
    result["meta"] = {
        "run_id": cfg["run_id"], "split": split, "checkpoint": "best.pt", "seed": seed,
        "dtype": str(dtype), "protocol": protocol(cfg, task), "overrides": overrides,
        "git": git_state(), "train_git": cfg.get("git"),
        "units": {name: a.shape[1] for name, a in result["acts"].items()},
        "label_names": list(task.label_names),
    }
    torch.save(result, out)
    return out


def main() -> None:
    """CLI entry point; see the module docstring."""
    p = argparse.ArgumentParser(description="Extract activations of a trained run.")
    p.add_argument("--run", type=Path, nargs="+", required=True, help="one or more runs/<run_id> directories")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp16", action="store_true", help="store activations as float16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config overrides, e.g. ego_mode=batch eval_batch_size=64")
    args = p.parse_args()
    from amlinterp.train import _parse_value
    overrides = {k: _parse_value(v) for k, _, v in (kv.partition("=") for kv in args.set)}
    for run in args.run:
        path = extract_run(run, args.split, args.device, args.seed,
                           torch.float16 if args.fp16 else torch.float32, **overrides)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
