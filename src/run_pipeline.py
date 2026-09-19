"""
End-to-end CLI: raw CSV -> clean data -> features -> trained model.
Lets the whole pipeline be reproduced with one command instead of
re-running every notebook by hand:

    python -m src.run_pipeline
"""

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from src.config import (
    BEST_MODEL_PATH,
    CLEAN_DATA_PATH,
    FEATURES_DATA_PATH,
    RANDOM_STATE,
    TARGET,
)
from src.data_cleaning import (
    drop_duplicates,
    drop_leakage_columns,
    handle_missing_values,
    load_raw_data,
    remove_adr_outliers,
    remove_zero_guest_bookings,
)
from src.evaluation import evaluate_model
from src.feature_engineering import engineer_features


def run():
    print("1/4 — Loading and cleaning raw data...")
    df = load_raw_data()
    df = drop_duplicates(df)
    df = handle_missing_values(df)
    df = remove_zero_guest_bookings(df)
    df = remove_adr_outliers(df)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"   Saved {len(df):,} clean rows -> {CLEAN_DATA_PATH}")

    print("2/4 — Engineering features...")
    df = engineer_features(df)
    df = drop_leakage_columns(df)
    df.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"   Saved {df.shape[1]} columns -> {FEATURES_DATA_PATH}")

    print("3/4 — Training the model (LightGBM)...")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    model = LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)

    print("4/4 — Evaluating and saving the model...")
    evaluate_model(model, X_test, y_test, model_name="LightGBM")
    joblib.dump(model, BEST_MODEL_PATH)
    print(f"   Saved trained model -> {BEST_MODEL_PATH}")


if __name__ == "__main__":
    run()
