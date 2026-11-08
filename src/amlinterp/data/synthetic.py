"""Synthetic pattern-detection data: random circulant multigraphs with exact labels.

This follows Egressy et al. (AAAI 2024), Appendix D. A random circulant graph
``G(n, d, r)`` has ``n`` nodes on a ring and ``floor(n * d / 2)`` directed edges. For each
edge the tail ``u`` is uniform and the head is ``v = round(X) mod n`` with ``X ~ N(u, r)``.
Patterns therefore arise *naturally* at a rate set by ``d`` and ``r``. Nothing is planted,
which avoids the degree shortcuts that planting creates.

Each node gets 11 binary labels (one multi-task node classification problem), found by
exhaustive enumeration:

====================  =========================================================
``deg_in/deg_out``    in-/out-degree > 3, counting parallel edges
``fan_in/fan_out``    number of distinct in-/out-neighbors > 3
``C2`` ... ``C6``     node lies on a simple directed cycle of exactly that length
``scatter_gather``    node is the sink of a scatter-gather: some source ``s`` reaches
                      it through >= 2 distinct intermediates, ``s -> w_i -> v``
``biclique``          node is a sink of a directed K_{2,2}: sources ``s1 != s2`` and
                      sinks ``v != v'``, all four edges ``s_i -> sink_j`` present
====================  =========================================================

Conventions where the paper is silent (documented so they can be changed in one place):

* Self-loops are rejected and re-drawn, so every graph has exactly ``floor(n*d/2)``
  non-loop edges. Algorithm 2 does not state this. Keeping self-loops (about 3.6% of draws
  at r=11.1) thins the graph and undershoots Table 3 by 2-5 points on C4-C6,
  scatter-gather and biclique. Re-drawing matches every Table 3 rate to within about
  one standard deviation across generator seeds (checked 2026-11-07).
* Synthetic edges have no attributes. Each edge gets a uniform random timestamp
  (``edge_time``) that is used only to order counterparties for port numbering.

Expected positive rates (paper Table 3): deg-in .352, deg-out .349, fan-in .324,
fan-out .323, C2 .191, C3 .344, C4 .527, C5 .677, C6 .779, S-G .321, B-C .318.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

LABEL_NAMES: tuple[str, ...] = (
    "deg_in", "deg_out", "fan_in", "fan_out",
    "C2", "C3", "C4", "C5", "C6",
    "scatter_gather", "biclique",
)
"""Column order of ``Data.y``; matches the column order of Egressy Table 1."""

PAPER_POSITIVE_RATE: dict[str, float] = dict(zip(
    LABEL_NAMES, (0.352, 0.349, 0.324, 0.323, 0.191, 0.344, 0.527, 0.677, 0.779, 0.321, 0.318)
))
"""Positive rate of each label reported in Egressy et al. Table 3 (for sanity checks)."""

DEGREE_THRESHOLD = 3
MAX_CYCLE_LEN = 6


@dataclass(frozen=True)
class CirculantConfig:
    """Parameters of the random circulant generator (defaults: Egressy App. D)."""

    n: int = 8192
    d: float = 6.0
    r: float = 11.1


# --------------------------------------------------------------------------- generator


def generate_circulant_graph(cfg: CirculantConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample a random circulant directed multigraph (Egressy et al., Algorithm 2).

    Args:
        cfg: generator parameters ``n``, ``d``, ``r``.
        rng: numpy random generator (the only source of randomness).

    Returns:
        ``edge_index`` of shape (2, m) (int64, 0-based, ``m = floor(n*d/2)``, no self-loops) and
        ``edge_time`` of shape (m,) (float64 in [0, 1), used only for port ordering).
    """
    m = int(cfg.n * cfg.d // 2)
    us, vs, have = [], [], 0
    while have < m:  # rejection-sample self-loops (about 3.6% of draws at r=11.1)
        u = rng.integers(0, cfg.n, size=m)
        v = np.floor(rng.normal(loc=u, scale=cfg.r) + 0.5).astype(np.int64) % cfg.n
        keep = u != v
        us.append(u[keep])
        vs.append(v[keep])
        have += int(keep.sum())
    u, v = np.concatenate(us)[:m], np.concatenate(vs)[:m]
    edge_time = rng.random(m)
    return np.stack([u, v]).astype(np.int64), edge_time


# --------------------------------------------------------------------------- labelers


def _simple_adjacency(edge_index: np.ndarray, n: int) -> sp.csr_matrix:
    """Binary adjacency with parallel edges merged and self-loops removed."""
    src, dst = edge_index
    keep = src != dst
    a = sp.csr_matrix((np.ones(int(keep.sum()), dtype=np.int32), (src[keep], dst[keep])), shape=(n, n))
    a.data[:] = 1  # merge parallel edges (duplicates are summed on construction)
    return a


def degree_fan_labels(edge_index: np.ndarray, n: int, threshold: int = DEGREE_THRESHOLD) -> np.ndarray:
    """Degree and fan labels: (n, 4) bool for ``[deg_in, deg_out, fan_in, fan_out] > threshold``.

    Degrees count every edge, including parallel edges. Fans count distinct neighbors.
    The generator produces no self-loops; if given any, a self-loop counts toward degree
    and makes the node its own neighbor.
    """
    src, dst = edge_index
    deg_in = np.bincount(dst, minlength=n)
    deg_out = np.bincount(src, minlength=n)
    pairs = np.unique(src.astype(np.int64) * n + dst)
    fan_out = np.bincount(pairs // n, minlength=n)
    fan_in = np.bincount(pairs % n, minlength=n)
    return np.stack([deg_in, deg_out, fan_in, fan_out], axis=1) > threshold


def cycle_labels(adj: sp.csr_matrix, max_len: int = MAX_CYCLE_LEN) -> np.ndarray:
    """Mark the nodes that lie on a simple directed cycle of each length 2..max_len.

    Every simple cycle is enumerated exactly once, rooted at its smallest node: a
    depth-bounded DFS from root ``s`` that only visits nodes ``> s``. The circulant graph
    is sparse and local, so this takes a few seconds for n=8192, d=6.

    Args:
        adj: simple binary adjacency (CSR, no self-loops).
        max_len: longest cycle length to enumerate.

    Returns:
        (n, max_len - 1) bool; column ``k - 2`` marks membership in some k-cycle.
    """
    n = adj.shape[0]
    nbrs = [adj.indices[adj.indptr[i]:adj.indptr[i + 1]].tolist() for i in range(n)]
    on_cycle = np.zeros((n, max_len - 1), dtype=bool)
    for s in range(n):
        _cycles_from_root(s, nbrs, max_len, on_cycle)
    return on_cycle


def _cycles_from_root(s: int, nbrs: list[list[int]], max_len: int, on_cycle: np.ndarray) -> None:
    """Enumerate simple cycles of length <= ``max_len`` whose smallest node is ``s``.

    Sets ``on_cycle[node, k - 2] = True`` for every node of every k-cycle found.
    """
    path = [s]

    def dfs(node: int) -> None:
        for w in nbrs[node]:
            if w == s:
                if len(path) >= 2:
                    on_cycle[path, len(path) - 2] = True
            elif w > s and len(path) < max_len and w not in path:
                path.append(w)
                dfs(w)
                path.pop()

    dfs(s)


def scatter_gather_labels(adj: sp.csr_matrix, min_intermediates: int = 2) -> np.ndarray:
    """Sinks of scatter-gather patterns: some ``s != v`` has >= ``min_intermediates`` paths ``s->w->v``.

    ``(A @ A)[s, v]`` counts the intermediates ``w``. Without self-loops, ``w`` differs
    from both ``s`` and ``v`` automatically.
    """
    a2 = (adj @ adj).tolil()
    a2.setdiag(0)
    return np.asarray(a2.tocsr().max(axis=0).todense()).ravel() >= min_intermediates


def biclique_labels(adj: sp.csr_matrix) -> np.ndarray:
    """Sinks of a directed K_{2,2}: ``v`` and some ``v' != v`` share >= 2 in-neighbors.

    ``(A^T @ A)[v, v']`` counts the common in-neighbors. Without self-loops, these are
    distinct from ``v`` and ``v'``, so the four nodes are distinct.
    """
    c = (adj.T @ adj).tolil()
    c.setdiag(0)
    return np.asarray(c.tocsr().max(axis=1).todense()).ravel() >= 2


def compute_labels(edge_index: np.ndarray, n: int) -> np.ndarray:
    """All 11 pattern labels for a graph, as an (n, 11) bool array ordered like ``LABEL_NAMES``."""
    adj = _simple_adjacency(edge_index, n)
    return np.concatenate(
        [
            degree_fan_labels(edge_index, n),
            cycle_labels(adj),
            scatter_gather_labels(adj)[:, None],
            biclique_labels(adj)[:, None],
        ],
        axis=1,
    )


# --------------------------------------------------------------------------- dataset


def make_split(cfg: CirculantConfig, seed: int) -> Data:
    """Generate one independent graph with labels as a PyG ``Data`` object.

    Fields: ``x`` (n, 1) constant ones; ``edge_index`` (2, m); ``edge_attr`` (m, 1)
    constant ones (synthetic edges carry no information); ``edge_time`` (m,) for port
    ordering; ``y`` (n, 11) float labels; ``label_names``; ``gen_seed``.
    """
    rng = np.random.default_rng(seed)
    edge_index, edge_time = generate_circulant_graph(cfg, rng)
    y = compute_labels(edge_index, cfg.n)
    m = edge_index.shape[1]
    return Data(
        x=torch.ones(cfg.n, 1),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.ones(m, 1),
        edge_time=torch.from_numpy(edge_time).float(),
        y=torch.from_numpy(y).float(),
        num_nodes=cfg.n,
        label_names=list(LABEL_NAMES),
        gen_seed=seed,
    )


SPLIT_SEEDS: dict[str, int] = {"train": 0, "val": 1, "test": 2}
"""Generator seed of each split. Fixed across all rungs and training seeds."""


def build_dataset(out_dir: Path, cfg: CirculantConfig = CirculantConfig(), overwrite: bool = False) -> dict[str, Data]:
    """Generate (or load cached) train/val/test graphs and save them to ``out_dir``.

    Writes ``{split}.pt`` and ``stats.json`` (config, sizes and positive rates next to
    the paper's Table 3 rates).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, Data] = {}
    for name, seed in SPLIT_SEEDS.items():
        path = out_dir / f"{name}.pt"
        if path.exists() and not overwrite:
            splits[name] = torch.load(path, weights_only=False)
        else:
            splits[name] = make_split(cfg, seed)
            torch.save(splits[name], path)
    stats = {
        "config": asdict(cfg),
        "split_seeds": SPLIT_SEEDS,
        "splits": {
            name: {
                "num_nodes": int(d.num_nodes),
                "num_edges": int(d.edge_index.shape[1]),
                "positive_rate": dict(zip(LABEL_NAMES, d.y.mean(0).round(decimals=4).tolist())),
            }
            for name, d in splits.items()
        },
        "paper_positive_rate": PAPER_POSITIVE_RATE,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return splits


def main() -> None:
    """CLI: ``python -m amlinterp.data.synthetic --out data/processed/synthetic``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("data/processed/synthetic"))
    p.add_argument("--n", type=int, default=CirculantConfig.n)
    p.add_argument("--d", type=float, default=CirculantConfig.d)
    p.add_argument("--r", type=float, default=CirculantConfig.r)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    splits = build_dataset(args.out, CirculantConfig(args.n, args.d, args.r), overwrite=args.overwrite)
    print(f"{'label':>15} " + " ".join(f"{s:>7}" for s in splits) + "   paper")
    for j, name in enumerate(LABEL_NAMES):
        rates = " ".join(f"{d.y[:, j].mean().item():7.3f}" for d in splits.values())
        print(f"{name:>15} {rates}   {PAPER_POSITIVE_RATE[name]:.3f}")


if __name__ == "__main__":
    main()
