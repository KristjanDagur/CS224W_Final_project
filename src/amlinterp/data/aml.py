# AMLworld preprocessing, adapted from IBM Multi-GNN `format_kaggle_files.py` and
# `data_loading.get_data` (chronological day-based split).
# Original: https://github.com/IBM/Multi-GNN, Copyright IBM Corp., Apache License 2.0.
"""Preprocess AMLworld (``*_Trans.csv`` + ``*_Patterns.txt``) once into cached tensors.

What matches IBM Multi-GNN:

* **Accounts.** An account is ``bank + account`` (string concatenation). Node IDs follow
  order of first appearance in raw row order, ``from`` before ``to``.
* **Categorical codes.** Currencies share one vocabulary, coded in first-appearance order
  (receiving, then payment, row by row). Payment formats are coded the same way.
* **Edge order.** Edges are stable-sorted by timestamp, as in the last step of IBM's
  ``format_kaggle_files.py`` (``sort(3)``). ``graph["raw_row"]`` maps each edge back to its
  raw CSV row. Because of the sort, each split's indices form one contiguous range, so
  global and positional indices agree. IBM's loaders rely on that.
* **Timestamps.** Seconds since the earliest transaction.
* **Split.** Days are ``[k*86400, (k+1)*86400)`` from the earliest timestamp. IBM picks the
  two day boundaries ``(i, j)`` whose train/val/test transaction shares minimize the
  largest relative error against 60/20/20. Each split's index list is its days
  concatenated in order.
* **Edge features.** Built at load time, not here (see ``edge_features``):
  ``[Timestamp, Amount Received, Received Currency, Payment Format]``, plus
  ``[in_port, out_port]`` when ports are on.

What differs:

* **Ports** are computed once on the full graph (``amlinterp.data.ports``). Splits are time
  prefixes, so every edge's ports are identical in the train, train+val and full graphs;
  ``preprocess`` asserts this. Ties in first-contact time are broken by neighbor ID,
  whereas IBM's unstable argsort is arbitrary.

Outputs, in ``out_dir``:

* ``graph.pt`` holds raw edge columns, labels, ports, split indices and vocabularies.
* ``patterns.pt`` holds per-edge ``attempt_id``/``pattern_type`` and the attempts table.
* ``stats.json`` holds sizes, split boundaries, illicit ratios, pattern counts per split and
  the cycle-length distribution.

Usage::

    python -m amlinterp.data.aml --raw data/raw --name HI-Small --out data/processed/aml_hi_small
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from amlinterp.data.patterns import PATTERN_TYPES, RAW_COLUMNS, match_patterns, parse_patterns
from amlinterp.data.ports import compute_ports

SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
SPLITS = ("train", "val", "test")
DAY = 24 * 3600


def read_transactions(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read ``*_Trans.csv`` with every column as a string, keeping bank-code leading zeros."""
    return pd.read_csv(path, header=0, names=list(RAW_COLUMNS), dtype=str, nrows=nrows)


def seconds_since_start(timestamps: pd.Series) -> np.ndarray:
    """AMLworld ``YYYY/MM/DD HH:MM`` strings -> int64 seconds since the earliest one.

    Uses ``total_seconds()``, which does not depend on the datetime64 unit; pandas 3 may
    store these as microseconds rather than nanoseconds.
    """
    ts = pd.to_datetime(timestamps, format="%Y/%m/%d %H:%M")
    return (ts - ts.min()).dt.total_seconds().to_numpy().astype(np.int64)


def _first_appearance_codes(*cols: pd.Series) -> tuple[list[np.ndarray], list[str]]:
    """Code several columns with one vocabulary, in IBM order (row by row, column by column)."""
    inter = np.stack([c.to_numpy(dtype=object) for c in cols], axis=1).ravel()
    codes, vocab = pd.factorize(inter)
    codes = codes.reshape(-1, len(cols))
    return [codes[:, k].astype(np.int64) for k in range(len(cols))], list(vocab)


def chronological_split(edge_time: np.ndarray, fractions=SPLIT_FRACTIONS) -> tuple[dict[str, np.ndarray], dict]:
    """IBM's day-based 60/20/20 split.

    Args:
        edge_time: (E,) seconds since the first transaction.
        fractions: target train/val/test shares of transactions.

    Returns:
        Split name -> edge indices (days concatenated in order, increasing within a day),
        and an info dict with the chosen day boundaries and the achieved shares.
    """
    n_days = int(edge_time.max() // DAY + 1)
    if n_days < 3:
        raise ValueError(f"need at least 3 days of transactions for a day-based split, got {n_days}")
    day = edge_time // DAY
    daily_inds = [np.flatnonzero(day == d) for d in range(n_days)]
    totals = np.array([len(x) for x in daily_inds])
    scores = {}
    for i, j in itertools.combinations(range(n_days), 2):
        parts = np.array([totals[:i].sum(), totals[i:j].sum(), totals[j:].sum()]) / totals.sum()
        scores[(i, j)] = max(abs(p - t) / t for p, t in zip(parts, fractions))
    i, j = min(scores, key=scores.get)
    days = {"train": range(0, i), "val": range(i, j), "test": range(j, n_days)}
    split = {k: np.concatenate([daily_inds[d] for d in v]).astype(np.int64) for k, v in days.items()}
    info = {
        "n_days": n_days,
        "boundary_days": [int(i), int(j)],
        "days": {k: [v.start, v.stop - 1] for k, v in days.items()},
        "transactions_per_day": totals.tolist(),
        "shares": {k: round(len(v) / len(edge_time), 4) for k, v in split.items()},
    }
    return split, info


def edge_features(graph: dict, idx: np.ndarray | torch.Tensor, ports: bool) -> torch.Tensor:
    """IBM edge features for the edges ``idx``: [Timestamp, Amount Received, Received Currency, Payment Format] (+ports).

    These are raw values. IBM z-normalizes each split graph's features with that graph's
    own mean and std, which happens in the AML task.
    """
    idx = torch.as_tensor(idx)
    cols = [graph["edge_time"][idx].double(), graph["amount_received"][idx].double(),
            graph["receiving_currency"][idx].double(), graph["payment_format"][idx].double()]
    feats = torch.stack(cols, dim=1)
    if ports:
        feats = torch.cat([feats, graph["ports"][idx].double()], dim=1)
    return feats.float()


def preprocess(raw_dir: Path, name: str, out_dir: Path, nrows: int | None = None) -> dict:
    """Run the full preprocessing and write ``graph.pt``, ``patterns.pt`` and ``stats.json``. Returns the stats."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    trans = read_transactions(raw_dir / f"{name}_Trans.csv", nrows=nrows)
    print(f"read {len(trans):,} transactions ({time.time() - t0:.0f}s)", flush=True)

    edge_time = seconds_since_start(trans["Timestamp"])
    (src, dst), accounts = _first_appearance_codes(trans["From Bank"] + trans["From Account"],
                                                   trans["To Bank"] + trans["To Account"])
    (recv_cur, pay_cur), currencies = _first_appearance_codes(trans["Receiving Currency"], trans["Payment Currency"])
    (fmt,), formats = _first_appearance_codes(trans["Payment Format"])
    # IBM codes IDs in raw order, then stable-sorts all edges by timestamp (format_kaggle_files.py).
    raw_row = np.argsort(edge_time, kind="stable")
    trans = trans.iloc[raw_row].reset_index(drop=True)
    edge_time, src, dst = edge_time[raw_row], src[raw_row], dst[raw_row]
    recv_cur, pay_cur, fmt = recv_cur[raw_row], pay_cur[raw_row], fmt[raw_row]
    y = trans["Is Laundering"].astype(np.int64).to_numpy()
    num_nodes = len(accounts)
    edge_index = torch.from_numpy(np.stack([src, dst]))
    t_edge = torch.from_numpy(edge_time)

    split, split_info = chronological_split(edge_time)
    if not np.array_equal(np.concatenate([split[k] for k in SPLITS]), np.arange(len(edge_time))):
        raise AssertionError("after the time sort, train/val/test must be consecutive index ranges")
    ports = compute_ports(edge_index, t_edge.double(), num_nodes).to(torch.int32)
    # Splits are time prefixes, so ports in the train and train+val graphs must equal the full graph's.
    for name_, idx in (("train", split["train"]), ("train+val", np.concatenate([split["train"], split["val"]]))):
        sub = compute_ports(edge_index[:, idx], t_edge[idx].double(), num_nodes).to(torch.int32)
        if not torch.equal(sub, ports[idx]):
            raise AssertionError(f"ports of the {name_} graph differ from full-graph ports")
    print(f"ports computed and prefix-checked ({time.time() - t0:.0f}s)", flush=True)

    graph = {
        "edge_index": edge_index,
        "edge_time": t_edge,
        "amount_received": torch.from_numpy(trans["Amount Received"].astype(np.float64).to_numpy()),
        "amount_paid": torch.from_numpy(trans["Amount Paid"].astype(np.float64).to_numpy()),
        "receiving_currency": torch.from_numpy(recv_cur),
        "payment_currency": torch.from_numpy(pay_cur),
        "payment_format": torch.from_numpy(fmt),
        "y": torch.from_numpy(y),
        "raw_row": torch.from_numpy(raw_row),  # edge e is row raw_row[e] of the raw CSV (0-based, header excluded)
        "ports": ports,
        "split": {k: torch.from_numpy(v) for k, v in split.items()},
        "num_nodes": num_nodes,
        "vocab": {"currency": currencies, "payment_format": formats},
        "source": f"{name}_Trans.csv" + (f" (first {nrows} rows)" if nrows else ""),
    }
    torch.save(graph, out_dir / "graph.pt")

    rows, attempts = parse_patterns(raw_dir / f"{name}_Patterns.txt")
    if nrows:  # subsample (development only): keep pattern rows that can match
        rows = rows[rows.merge(trans.drop_duplicates(), how="left", on=list(RAW_COLUMNS), indicator=True)["_merge"].eq("both").to_numpy()]
    attempt_id, pattern_type, match_report = match_patterns(trans, rows, attempts)
    torch.save({"attempt_id": torch.from_numpy(attempt_id), "pattern_type": torch.from_numpy(pattern_type),
                "type_names": list(PATTERN_TYPES), "attempts": attempts.to_dict(orient="records")},
               out_dir / "patterns.pt")

    stats = summarize(y, edge_time, split, split_info, attempt_id, attempts, ports, num_nodes)
    stats["pattern_matching"] = match_report
    stats["self_loops"] = int((src == dst).sum())
    stats["parallel_edges"] = int(len(src) - len(np.unique(src * num_nodes + dst)))
    stats["source"] = graph["source"]
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"done ({time.time() - t0:.0f}s): {out_dir}", flush=True)
    return stats


def summarize(y, edge_time, split, split_info, attempt_id, attempts, ports, num_nodes) -> dict:
    """Dataset statistics: sizes, illicit ratio per split, pattern counts per split, cycle lengths."""
    att = attempts.set_index("attempt_id")
    per_split = {}
    for k, idx in split.items():
        a = attempt_id[idx]
        a_in = a[a >= 0]
        types = att.loc[a_in, "type"].to_numpy() if len(a_in) else np.array([], dtype=object)
        attempt_types = att.loc[np.unique(a_in), "type"] if len(a_in) else pd.Series(dtype=object)
        cyc = att.loc[np.unique(a_in)]
        cyc = cyc[cyc["type"] == "CYCLE"]["param"]
        per_split[k] = {
            "transactions": int(len(idx)),
            "laundering": int(y[idx].sum()),
            "illicit_ratio": round(float(y[idx].mean()), 6),
            "laundering_in_patterns": int(len(a_in)),
            "pattern_coverage_of_laundering": round(len(a_in) / max(int(y[idx].sum()), 1), 4),
            "pattern_transactions_by_type": {t: int((types == t).sum()) for t in PATTERN_TYPES},
            "attempts_touching_split_by_type": {t: int((attempt_types == t).sum()) for t in PATTERN_TYPES},
            "cycle_attempts_by_hops": {int(h): int(c) for h, c in cyc.value_counts().sort_index().items()},
        }
    in_pattern = attempt_id >= 0
    spans = pd.Series(attempt_id).to_frame("a").assign(s=np.select(
        [np.isin(np.arange(len(y)), split[k]) for k in SPLITS], SPLITS, default="?"))[in_pattern]
    return {
        "num_nodes": int(num_nodes),
        "num_edges": int(len(y)),
        "laundering": int(y.sum()),
        "illicit_ratio": round(float(y.mean()), 6),
        "time_span_days": round(float(edge_time.max()) / DAY, 2),
        "split": split_info,
        "per_split": per_split,
        "attempts_total": int(len(attempts)),
        "attempts_by_type": attempts["type"].value_counts().reindex(PATTERN_TYPES).fillna(0).astype(int).to_dict(),
        "attempts_spanning_split_boundary": int((spans.groupby("a")["s"].nunique() > 1).sum()),
        "cycle_attempts_by_hops_all": {int(h): int(c) for h, c in
                                       attempts[attempts["type"] == "CYCLE"]["param"].value_counts().sort_index().items()},
        "max_port": {"in": int(ports[:, 0].max()), "out": int(ports[:, 1].max())},
    }


def main() -> None:
    """CLI entry point; see the module docstring."""
    p = argparse.ArgumentParser(description="Preprocess AMLworld into cached tensors.")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--name", default="HI-Small")
    p.add_argument("--out", type=Path, default=Path("data/processed/aml_hi_small"))
    p.add_argument("--nrows", type=int, default=None, help="only the first N rows (development)")
    args = p.parse_args()
    stats = preprocess(args.raw, args.name, args.out, args.nrows)
    print(json.dumps({k: stats[k] for k in ("num_nodes", "num_edges", "laundering", "illicit_ratio")}, indent=2))
    print(json.dumps(stats["split"]["shares"]), json.dumps(stats["pattern_matching"]))


if __name__ == "__main__":
    main()
