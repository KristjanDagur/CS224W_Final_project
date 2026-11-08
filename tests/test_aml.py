"""Tests for the AMLworld patterns parser, pattern matching and the chronological split."""

import numpy as np
import pandas as pd
import pytest

from amlinterp.data.aml import DAY, chronological_split, seconds_since_start
from amlinterp.data.patterns import PATTERN_TYPES, RAW_COLUMNS, match_patterns, parse_patterns

ROWS = [
    "2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1",
    "2022/09/01 04:33,021174,800737690,020,80020C5B0,8630.40,Euro,8630.40,Euro,ACH,1",
    "2022/09/01 00:03,01467,8013C4030,020,80BC62F10,58702.10,Yuan,58702.10,Yuan,ACH,1",
    "2022/09/01 02:52,020,80BC62F10,01467,8013C4030,7332.87,Yuan,7332.87,Yuan,ACH,1",
]
PATTERNS = f"""BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
{ROWS[0]}
{ROWS[1]}
END LAUNDERING ATTEMPT - FAN-OUT

BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops
{ROWS[2]}
{ROWS[3]}
END LAUNDERING ATTEMPT - CYCLE

BEGIN LAUNDERING ATTEMPT - STACK
END LAUNDERING ATTEMPT - STACK
"""


def _trans(lines):
    rows = [l.split(",") for l in lines]
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS))


def test_parse_patterns(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text(PATTERNS)
    rows, attempts = parse_patterns(f)
    assert attempts[["type", "param", "n_rows"]].values.tolist() == [["FAN-OUT", 16, 2], ["CYCLE", 2, 2], ["STACK", -1, 0]]
    assert rows["attempt_id"].tolist() == [0, 0, 1, 1]
    assert rows["From Bank"].iloc[2] == "01467"  # leading zero kept


def test_match_patterns_including_duplicate_transactions(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text(PATTERNS)
    rows, attempts = parse_patterns(f)
    normal = "2022/09/01 01:00,010,8000EBD30,010,8000EBD30,10.00,US Dollar,10.00,US Dollar,Cash,0"
    trans = _trans([normal, ROWS[3], ROWS[0], ROWS[2], ROWS[1], ROWS[0]])  # ROWS[0] duplicated
    attempt_id, ptype, report = match_patterns(trans, rows, attempts)
    assert attempt_id.tolist() == [-1, 1, 0, 1, 0, 0]
    assert [PATTERN_TYPES[t] if t >= 0 else None for t in ptype] == [None, "CYCLE", "FAN-OUT", "CYCLE", "FAN-OUT", "FAN-OUT"]
    assert report["pattern_rows_matching_duplicate_transactions"] == 1


def test_match_patterns_rejects_unmatched_rows(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text(PATTERNS)
    rows, attempts = parse_patterns(f)
    with pytest.raises(ValueError, match="match no transaction"):
        match_patterns(_trans(ROWS[:3]), rows, attempts)


def test_chronological_split_is_day_aligned_and_near_60_20_20():
    rng = np.random.default_rng(0)
    t = np.sort(rng.integers(0, 10 * DAY, size=10_000))
    split, info = chronological_split(t)
    days = {k: np.unique(t[v] // DAY) for k, v in split.items()}
    assert days["train"].max() < days["val"].min() and days["val"].max() < days["test"].min()
    assert sorted(np.concatenate(list(split.values())).tolist()) == list(range(len(t)))
    assert abs(info["shares"]["train"] - 0.6) < 0.1 and abs(info["shares"]["test"] - 0.2) < 0.1


def test_seconds_since_start_is_unit_independent():
    s = pd.Series(["2022/09/01 00:20", "2022/09/01 00:00", "2022/09/10 23:59"])
    assert seconds_since_start(s).tolist() == [1200, 0, 9 * DAY + 23 * 3600 + 59 * 60]
