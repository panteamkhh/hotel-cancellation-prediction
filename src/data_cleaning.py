"""
Reusable data cleaning functions for the hotel bookings dataset.
Used by notebooks/01_data_cleaning.ipynb.
"""

import pandas as pd

from src.data_loader import load_raw_data

__all__ = [
    "load_raw_data",
    "drop_duplicates",
    "handle_missing_values",
    "remove_zero_guest_bookings",
    "remove_adr_outliers",
    "drop_leakage_columns",
    "clean_pipeline",
]


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill / drop missing values.
    - children: fill with mode (very few NaNs)
    - company: drop column (mostly missing)
    - agent: fill with 'no agent' (NaN means no agent involved)
    - country: fill with mode
    """
    df = df.copy()

    if "children" in df.columns:
        df["children"] = df["children"].fillna(df["children"].mode()[0])

    if "company" in df.columns:
        df = df.drop(columns=["company"])

    if "agent" in df.columns:
        df["agent"] = df["agent"].fillna("no agent")

    if "country" in df.columns:
        df["country"] = df["country"].fillna(df["country"].mode()[0])

    return df


def remove_zero_guest_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where adults + children + babies == 0 (invalid bookings)."""
    total_members = df["adults"] + df["children"] + df["babies"]
    return df[total_members != 0].copy()


def remove_adr_outliers(df: pd.DataFrame, upper_quantile: float = 0.999) -> pd.DataFrame:
    """Remove negative ADR and extreme high-end ADR outliers."""
    df = df.copy()
    df = df[df["adr"] >= 0]
    df = df[df["adr"] < df["adr"].quantile(upper_quantile)]
    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that directly encode the outcome (target leakage).
    reservation_status / reservation_status_date are only known AFTER
    the cancellation decision, so they can't be used as predictors.
    """
    return df.drop(columns=["reservation_status", "reservation_status_date"], errors="ignore")


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline in order."""
    df = drop_duplicates(df)
    df = handle_missing_values(df)
    df = remove_zero_guest_bookings(df)
    df = remove_adr_outliers(df)
    return df
