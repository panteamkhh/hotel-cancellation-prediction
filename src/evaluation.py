"""
Shared model evaluation helpers for notebooks/09_modeling.ipynb.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import theme

theme.apply_theme()


def evaluate_model(model, X_test, y_test, model_name: str = "Model", y_proba=None) -> dict:
    """
    Print a full classification report and return a dict of metrics,
    so results can be collected into a comparison table across models.
    """
    y_pred = model.predict(X_test)

    if y_proba is None and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
    }

    print(f"--- {model_name} ---")
    print(classification_report(y_test, y_pred))
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    return metrics


def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Model"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Canceled", "Canceled"],
        yticklabels=["Not Canceled", "Canceled"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()


def plot_roc_curve(model, X_test, y_test, model_name: str = "Model", ax=None):
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=model_name, ax=ax)


def plot_feature_importance(model, feature_names, top_n: int = 15, model_name: str = "Model"):
    """Works for tree-based models exposing feature_importances_."""
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} has no feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    order = importances.argsort()[::-1][:top_n]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[order], y=[feature_names[i] for i in order])
    plt.title(f"Top {top_n} Feature Importances - {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
