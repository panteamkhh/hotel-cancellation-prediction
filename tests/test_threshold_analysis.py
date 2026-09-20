import numpy as np

from src.threshold_analysis import best_threshold, expected_cost, threshold_curve


def test_threshold_curve_has_expected_columns_and_rows():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.4, 0.6, 0.9]
    curve = threshold_curve(y_true, y_proba, thresholds=[0.2, 0.5, 0.8])

    assert list(curve.columns) == ["threshold", "precision", "recall", "f1"]
    assert len(curve) == 3
    assert curve["precision"].between(0, 1).all()


def test_expected_cost_counts_weighted_errors():
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.9, 0.1, 0.9]
    # at 0.5: one false positive (0.9) and one false negative (0.1)
    cost = expected_cost(
        y_true, y_proba, threshold=0.5, cost_false_negative=10, cost_false_positive=1
    )
    assert cost == 11


def test_best_threshold_prefers_high_recall_when_misses_are_costly():
    y_true = [0, 1]
    y_proba = [0.4, 0.4]
    # a missed cancellation (cost 10) is worse than a false alarm (cost 1)
    threshold = best_threshold(y_true, y_proba, cost_false_negative=10, cost_false_positive=1)
    assert threshold <= 0.4


def test_best_threshold_is_within_grid():
    y_true = [0, 1, 0, 1]
    y_proba = [0.2, 0.7, 0.3, 0.8]
    threshold = best_threshold(y_true, y_proba)
    assert 0.05 <= threshold <= 0.95
    assert np.isclose(threshold, round(threshold, 2))
