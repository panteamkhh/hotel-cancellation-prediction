"""
Train and compare the candidate cancellation models, then persist the best.

This is the CLI-reproducible version of the comparison in
``notebooks/09_modeling.ipynb``:

    python -m src.train

It reads the feature-engineered CSV, fits Logistic Regression, Random Forest,
XGBoost and LightGBM on the same stratified split, prints a comparison table,
saves the best model to ``models/best_model.pkl`` and the metrics to
``models/model_metrics.json``.
"""

import json

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_PATH,
    MODEL_METRICS_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET,
)
from src.data_loader import load_features_data
from src.evaluation import evaluate_model


def prepare_xy(df: pd.DataFrame, target: str = TARGET):
    """Split a dataframe into a feature matrix and the target vector."""
    return df.drop(columns=[target]), df[target]


def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """One-hot encode categorical columns, keeping train and test aligned."""
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    return X_train.align(X_test, join="left", axis=1, fill_value=0)


def candidate_models(random_state: int = RANDOM_STATE) -> dict:
    """The four models compared in the report, in a stable order."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1, verbose=-1
        ),
    }


def train_and_compare(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    save: bool = True,
):
    """
    Fit every candidate model, return ``(results_df, best_model)`` and —
    when ``save`` is set — write the best model and the metrics table to disk.
    """
    df = load_features_data() if df is None else df
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test = encode_categoricals(X_train, X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    fitted = {}
    for name, model in candidate_models(random_state).items():
        is_linear = name == "Logistic Regression"
        train_X = X_train_scaled if is_linear else X_train
        test_X = X_test_scaled if is_linear else X_test

        model.fit(train_X, y_train)
        results.append(evaluate_model(model, test_X, y_test, model_name=name))
        fitted[name] = model

    results_df = (
        pd.DataFrame(results).set_index("model").sort_values("roc_auc", ascending=False).round(4)
    )
    best_name = results_df.index[0]
    best_model = fitted[best_name]

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, BEST_MODEL_PATH)
        with open(MODEL_METRICS_PATH, "w", encoding="utf-8") as handle:
            json.dump(results_df.to_dict(orient="index"), handle, indent=2)
        print(f"\nBest model: {best_name} (ROC-AUC {results_df.loc[best_name, 'roc_auc']:.4f})")
        print(f"Saved model -> {BEST_MODEL_PATH}")
        print(f"Saved metrics -> {MODEL_METRICS_PATH}")

    return results_df, best_model


def main() -> None:
    results_df, _ = train_and_compare()
    print("\nModel comparison (sorted by ROC-AUC):")
    print(results_df.to_string())


if __name__ == "__main__":
    main()
