"""Evaluation metrics: minority-class F1 (the benchmark metric), PR-AUC and threshold tuning."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def minority_class(y_train: np.ndarray) -> np.ndarray:
    """Minority class (0 or 1) of each task, decided on the training labels.

    Args:
        y_train: (N, T) binary labels.

    Returns:
        (T,) int array. Usually 1; it is 0 for tasks that are mostly positive (synthetic C5, C6).
    """
    return (y_train.mean(axis=0) > 0.5).astype(int) ^ 1


def minority_f1(y_true: np.ndarray, prob: np.ndarray, minority: np.ndarray, thresholds: np.ndarray | float = 0.5) -> np.ndarray:
    """F1 of the minority class for each task, predicting 1 when ``prob > threshold``.

    Args:
        y_true: (N, T) binary labels.
        prob: (N, T) predicted probabilities of label 1.
        minority: (T,) minority class per task (see ``minority_class``).
        thresholds: scalar or (T,) decision thresholds.

    Returns:
        (T,) minority-class F1 in [0, 1].
    """
    thr = np.broadcast_to(np.asarray(thresholds, dtype=float), (y_true.shape[1],))
    pred = prob > thr
    return np.array([
        f1_score(y_true[:, t], pred[:, t], pos_label=int(minority[t]), zero_division=0.0)
        for t in range(y_true.shape[1])
    ])


def pr_auc(y_true: np.ndarray, prob: np.ndarray, minority: np.ndarray) -> np.ndarray:
    """Average precision of the minority class for each task (scores flipped when the minority is 0)."""
    out = []
    for t in range(y_true.shape[1]):
        if minority[t] == 1:
            out.append(average_precision_score(y_true[:, t], prob[:, t]))
        else:
            out.append(average_precision_score(1 - y_true[:, t], 1 - prob[:, t]))
    return np.array(out)


def tune_thresholds(y_val: np.ndarray, prob_val: np.ndarray, minority: np.ndarray) -> np.ndarray:
    """Per-task decision threshold that maximizes validation minority F1 (over ``THRESHOLD_GRID``)."""
    scores = np.stack([minority_f1(y_val, prob_val, minority, thr) for thr in THRESHOLD_GRID])  # (G, T)
    return THRESHOLD_GRID[scores.argmax(axis=0)]
