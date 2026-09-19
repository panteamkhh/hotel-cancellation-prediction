import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.predict import align_features, load_model, score_bookings


def fitted_model():
    X = pd.DataFrame({"lead_time": [1, 2, 90, 100], "adr": [80, 90, 120, 130]})
    y = [0, 0, 1, 1]
    return LogisticRegression().fit(X, y)


def test_align_features_reorders_and_fills_missing():
    df = pd.DataFrame({"b": [1, 2], "c": [3, 4]})
    aligned = align_features(df, ["a", "b", "c"])
    assert list(aligned.columns) == ["a", "b", "c"]
    assert (aligned["a"] == 0).all()


def test_align_features_no_columns_is_passthrough():
    df = pd.DataFrame({"b": [1]})
    assert align_features(df, None) is df


def test_score_bookings_adds_probability_and_flag():
    model = fitted_model()
    df = pd.DataFrame({"lead_time": [1, 2, 90, 100], "adr": [80, 90, 120, 130]})
    scored = score_bookings(model, df)

    assert "cancel_probability" in scored.columns
    assert "predicted_canceled" in scored.columns
    assert scored["cancel_probability"].between(0, 1).all()
    assert set(scored["predicted_canceled"].unique()).issubset({0, 1})
    assert list(scored["lead_time"]) == [1, 2, 90, 100]


def test_score_bookings_threshold_controls_flag():
    model = fitted_model()
    df = pd.DataFrame({"lead_time": [50], "adr": [100]})
    low = score_bookings(model, df, threshold=0.0)
    high = score_bookings(model, df, threshold=1.0)
    assert low["predicted_canceled"].iloc[0] == 1
    assert high["predicted_canceled"].iloc[0] == 0


def test_load_model_accepts_raw_estimator(tmp_path):
    path = tmp_path / "model.pkl"
    joblib.dump(fitted_model(), path)

    bundle = load_model(path)
    assert "model" in bundle
    assert bundle["feature_columns"] is None


def test_load_model_accepts_bundle(tmp_path):
    path = tmp_path / "bundle.pkl"
    joblib.dump(
        {"model": fitted_model(), "feature_columns": ["lead_time", "adr"], "model_name": "LR"},
        path,
    )

    bundle = load_model(path)
    assert bundle["feature_columns"] == ["lead_time", "adr"]
    assert bundle["model_name"] == "LR"
