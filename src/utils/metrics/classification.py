"""Task 1a classification metrics (accuracy, F1, F2, precision, recall)."""

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: Sequence,
    y_pred: Sequence,
) -> dict:
    """Compute the five Task 1a metrics for a single artifact head.

    Follows the official LISA 2026 Task 1a ranking:
    aggregate = mean(accuracy, F1_macro, F2_macro, precision_macro, recall_macro).

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Dict with keys: accuracy, f1_macro, f2_macro, precision_macro,
        recall_macro, aggregate.
    """
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0))
    pre = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    agg = float(np.mean([acc, f1, f2, pre, rec]))
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "f2_macro": f2,
        "precision_macro": pre,
        "recall_macro": rec,
        "aggregate": agg,
    }
