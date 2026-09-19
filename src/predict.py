"""
Score new bookings with a trained cancellation model.

The saved model was fit on one-hot encoded features, so this module keeps
the encoding step explicit: pass an already engineered/encoded frame plus
the column list the model expects (``feature_columns``), and it returns the
input rows with a cancellation probability and a 0/1 flag attached.

Example
-------
>>> bundle = load_model()
>>> scored = score_bookings(bundle["model"], encoded_bookings, bundle["feature_columns"])
"""

import joblib
import pandas as pd

from src.config import BEST_MODEL_PATH


def load_model(path=BEST_MODEL_PATH) -> dict:
    """
    Load a saved model and normalise it to a bundle dict.

    Both the raw estimator written by ``notebooks/09_modeling.ipynb`` and the
    ``{"model", "feature_columns", "model_name"}`` bundle written by
    ``src/train.py`` are supported.
    """
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "model" in bundle:
        return {
            "model": bundle["model"],
            "feature_columns": bundle.get("feature_columns"),
            "model_name": bundle.get("model_name", "model"),
        }
    return {"model": bundle, "feature_columns": None, "model_name": type(bundle).__name__}


def align_features(df: pd.DataFrame, feature_columns) -> pd.DataFrame:
    """Reorder/reindex features to match the training columns (missing -> 0)."""
    if not feature_columns:
        return df
    return df.reindex(columns=list(feature_columns), fill_value=0)


def score_bookings(
    model,
    df: pd.DataFrame,
    feature_columns=None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Return a copy of ``df`` with ``cancel_probability`` and
    ``predicted_canceled`` columns added.
    """
    X = align_features(df, feature_columns)
    probabilities = model.predict_proba(X)[:, 1]

    scored = df.copy()
    scored["cancel_probability"] = probabilities
    scored["predicted_canceled"] = (probabilities >= threshold).astype(int)
    return scored
