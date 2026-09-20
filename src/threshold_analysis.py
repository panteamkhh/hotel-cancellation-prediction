"""
Cost-sensitive decision-threshold analysis for the cancellation model.

The default 0.5 cutoff treats a missed cancellation and a false alarm as
equally costly, which they are not. These helpers sweep the threshold and pick
the one that minimises the expected business cost:

    expected_cost = false_negatives * cost_fn + false_positives * cost_fp
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def threshold_curve(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """Precision / recall / F1 for each candidate probability threshold."""
    thresholds = np.linspace(0.05, 0.95, 19) if thresholds is None else np.asarray(thresholds)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    rows = []
    for threshold in thresholds:
        predicted = (y_proba >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 3),
                "precision": precision_score(y_true, predicted, zero_division=0),
                "recall": recall_score(y_true, predicted, zero_division=0),
                "f1": f1_score(y_true, predicted, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def expected_cost(
    y_true,
    y_proba,
    threshold: float = 0.5,
    cost_false_negative: float = 10.0,
    cost_false_positive: float = 1.0,
) -> float:
    """Total weighted cost of the errors made at a given threshold."""
    y_true = np.asarray(y_true)
    predicted = (np.asarray(y_proba) >= threshold).astype(int)

    false_negatives = int(((y_true == 1) & (predicted == 0)).sum())
    false_positives = int(((y_true == 0) & (predicted == 1)).sum())
    return false_negatives * cost_false_negative + false_positives * cost_false_positive


def best_threshold(
    y_true,
    y_proba,
    cost_false_negative: float = 10.0,
    cost_false_positive: float = 1.0,
    thresholds=None,
) -> float:
    """Threshold in ``[0.05, 0.95]`` that minimises :func:`expected_cost`."""
    candidates = np.linspace(0.05, 0.95, 91) if thresholds is None else np.asarray(thresholds)
    costs = [
        expected_cost(y_true, y_proba, t, cost_false_negative, cost_false_positive)
        for t in candidates
    ]
    return float(candidates[int(np.argmin(costs))])
