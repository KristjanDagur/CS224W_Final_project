"""Parse AMLworld ``*_Patterns.txt`` and match its rows to transactions.

File format, verified on ``HI-Small_Patterns.txt`` (2026-11-07). The file has 370 attempts
and 3,209 transaction rows; each attempt looks like::

    BEGIN LAUNDERING ATTEMPT - <TYPE>: <detail>
    <transaction rows, same 11-column CSV format as *_Trans.csv, no header>
    END LAUNDERING ATTEMPT - <TYPE>

``<TYPE>`` is one of ``PATTERN_TYPES``. ``<detail>`` depends on the type:

* ``Max N hops`` for CYCLE and RANDOM;
* ``Max N-degree Fan-In`` for FAN-IN and GATHER-SCATTER;
* ``Max N-degree Fan-Out`` for FAN-OUT;
* empty for BIPARTITE, SCATTER-GATHER and STACK.

``N`` is parsed into ``attempts.param``.

Matching: a pattern row is matched to every transaction whose raw text is identical in all
11 columns. The transactions file contains a few exact duplicate rows, so one pattern row
can match several identical transactions. ``match_patterns`` reports how often that happens.
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

PATTERN_TYPES: tuple[str, ...] = (
    "FAN-IN", "FAN-OUT", "GATHER-SCATTER", "SCATTER-GATHER", "CYCLE", "BIPARTITE", "STACK", "RANDOM",
)
"""Pattern type names; ``pattern_type`` codes index this tuple (-1 = not in any attempt)."""

RAW_COLUMNS: tuple[str, ...] = (
    "Timestamp", "From Bank", "From Account", "To Bank", "To Account", "Amount Received",
    "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering",
)
"""Column names of ``*_Trans.csv`` (the raw header repeats "Account"; these are unambiguous)."""

_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z-]+):?\s*(.*)$")
_END = re.compile(r"^END LAUNDERING ATTEMPT - ([A-Z-]+)\s*$")
_PARAM = re.compile(r"Max (\d+)")


def parse_patterns(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a patterns file.

    Returns:
        ``rows``: one row per pattern transaction, with the raw columns (all strings, as
        in the file) plus ``attempt_id``.
        ``attempts``: one row per attempt, with ``attempt_id``, ``type``, ``detail``,
        ``param`` (the N of the detail, or -1) and ``n_rows``.
    """
    attempts, lines, owner = [], [], []
    current = None
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if m := _BEGIN.match(line):
            if current is not None:
                raise ValueError(f"line {lineno}: BEGIN inside attempt {current}")
            ptype, detail = m.group(1), m.group(2).strip()
            if ptype not in PATTERN_TYPES:
                raise ValueError(f"line {lineno}: unknown pattern type {ptype!r}")
            pm = _PARAM.search(detail)
            current = len(attempts)
            attempts.append({"attempt_id": current, "type": ptype, "detail": detail,
                             "param": int(pm.group(1)) if pm else -1, "n_rows": 0})
        elif m := _END.match(line):
            if current is None or m.group(1) != attempts[current]["type"]:
                raise ValueError(f"line {lineno}: unmatched END {m.group(1)!r}")
            current = None
        else:
            if current is None:
                raise ValueError(f"line {lineno}: transaction row outside an attempt")
            lines.append(line)
            owner.append(current)
            attempts[current]["n_rows"] += 1
    if current is not None:
        raise ValueError("file ended inside an attempt")
    rows = pd.read_csv(StringIO("\n".join(lines)), header=None, names=list(RAW_COLUMNS), dtype=str)
    rows["attempt_id"] = np.asarray(owner, dtype=np.int32)
    return rows, pd.DataFrame(attempts)


def _row_key(df: pd.DataFrame) -> pd.Series:
    """One string per row joining all raw columns (exact-text match key)."""
    key = df[RAW_COLUMNS[0]].astype(str)
    for c in RAW_COLUMNS[1:]:
        key = key + "," + df[c].astype(str)
    return key


def match_patterns(trans: pd.DataFrame, rows: pd.DataFrame, attempts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Assign each transaction its laundering attempt and pattern type.

    Args:
        trans: raw transactions with string columns ``RAW_COLUMNS`` (row order = edge order).
        rows, attempts: output of ``parse_patterns``.

    Returns:
        ``attempt_id`` (E,) int32 and ``pattern_type`` (E,) int8, both -1 for transactions in
        no attempt, plus a report dict with match counts.

    Raises:
        ValueError: if a pattern row matches no transaction, or if one transaction would
        belong to two different attempts.
    """
    tkey = pd.DataFrame({"key": _row_key(trans), "edge": np.arange(len(trans))})
    pkey = pd.DataFrame({"key": _row_key(rows), "attempt_id": rows["attempt_id"].to_numpy()})
    merged = pkey.merge(tkey, on="key", how="left")
    unmatched = int(merged["edge"].isna().sum())
    if unmatched:
        raise ValueError(f"{unmatched} pattern rows match no transaction")
    per_edge = merged.groupby("edge")["attempt_id"].nunique()
    if (per_edge > 1).any():
        raise ValueError(f"{int((per_edge > 1).sum())} transactions belong to more than one attempt")
    attempt_id = np.full(len(trans), -1, dtype=np.int32)
    attempt_id[merged["edge"].to_numpy(dtype=np.int64)] = merged["attempt_id"].to_numpy()
    type_code = attempts["type"].map({t: i for i, t in enumerate(PATTERN_TYPES)}).to_numpy(dtype=np.int8)
    pattern_type = np.where(attempt_id >= 0, type_code[np.maximum(attempt_id, 0)], -1).astype(np.int8)
    matches_per_row = merged.groupby(["key", "attempt_id"]).size()  # pattern rows are unique
    report = {
        "pattern_rows": int(len(rows)),
        "matched_edges": int((attempt_id >= 0).sum()),
        "pattern_rows_matching_duplicate_transactions": int((matches_per_row > 1).sum()),
    }
    return attempt_id, pattern_type, report
