"""
Lightweight data-quality checks for the bookings data.

Produces the handful of numbers that matter before any analysis: missing
values, exact duplicates, constant columns and the target balance. Used as a
sanity gate at the start of a pipeline run.
"""

import pandas as pd

from src.config import TARGET


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count and share per column, worst first (empty columns omitted)."""
    missing = df.isna().sum()
    report = pd.DataFrame({"missing": missing, "share": missing / len(df)})
    return report[report["missing"] > 0].sort_values("missing", ascending=False)


def duplicate_count(df: pd.DataFrame) -> int:
    """Number of exact duplicate rows."""
    return int(df.duplicated().sum())


def constant_columns(df: pd.DataFrame) -> list:
    """Columns that hold a single value (or all-missing) across the frame."""
    return [column for column in df.columns if df[column].nunique(dropna=False) <= 1]


def target_balance(df: pd.DataFrame, target: str = TARGET) -> pd.Series:
    """Share of each target class, sorted by class label."""
    if target not in df.columns:
        raise KeyError(f"target column '{target}' not found in dataframe")
    return df[target].value_counts(normalize=True).sort_index()


def data_quality_report(df: pd.DataFrame, target: str = TARGET) -> dict:
    """Collect the headline quality metrics into a plain dict."""
    report = {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicates": duplicate_count(df),
        "missing_cells": int(df.isna().sum().sum()),
        "constant_columns": constant_columns(df),
        "target_balance": None,
    }
    if target in df.columns:
        report["target_balance"] = target_balance(df, target).round(4).to_dict()
    return report


def print_report(df: pd.DataFrame, target: str = TARGET) -> dict:
    """Print a human-readable report and return the underlying dict."""
    report = data_quality_report(df, target)
    print(f"Rows: {report['rows']:,}  Columns: {report['columns']}")
    print(f"Duplicate rows: {report['duplicates']:,}")
    print(f"Missing cells: {report['missing_cells']:,}")
    print(f"Constant columns: {report['constant_columns'] or 'none'}")
    if report["target_balance"] is not None:
        balance = ", ".join(
            f"{label}: {share:.1%}" for label, share in report["target_balance"].items()
        )
        print(f"Target balance: {balance}")
    return report
