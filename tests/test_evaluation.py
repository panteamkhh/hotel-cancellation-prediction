import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation import evaluate_model


class DummyModel:
    """Minimal estimator without ``predict_proba``, to exercise the None path."""

    def predict(self, X):
        return (np.arange(len(X)) % 2).astype(int)


def fitted_model():
    X = pd.DataFrame({"lead_time": [1, 2, 90, 100], "adr": [80, 90, 120, 130]})
    y = [0, 0, 1, 1]
    return LogisticRegression().fit(X, y), X, y


def test_evaluate_model_returns_all_metric_keys():
    model, X, y = fitted_model()
    metrics = evaluate_model(model, X, y, model_name="LR")

    assert metrics["model"] == "LR"
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_evaluate_model_without_predict_proba_has_no_roc_auc():
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    y = [0, 1, 0, 1]
    metrics = evaluate_model(DummyModel(), X, y, model_name="Dummy")
    assert metrics["roc_auc"] is None
