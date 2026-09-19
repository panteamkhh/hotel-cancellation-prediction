"""
Reusable feature engineering functions for the hotel bookings dataset.
Used by notebooks/08_feature_engineering.ipynb and 09_modeling.ipynb.
"""

import pandas as pd


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add total_members, total_guests, has_family."""
    df = df.copy()
    df["total_members"] = df["adults"] + df["babies"] + df["children"]
    df["total_guests"] = df["babies"] + df["children"] + df["adults"]
    df["has_family"] = ((df["babies"] > 0) | (df["children"] > 0)).astype(int)
    return df


def add_stay_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add total_stay (weekend + week nights)."""
    df = df.copy()
    df["total_stay"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    return df


def add_room_changed_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag whether the assigned room differs from the reserved room."""
    df = df.copy()
    df["room_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    return df


def add_lead_time_features(df: pd.DataFrame, threshold: int = 90) -> pd.DataFrame:
    """Flag bookings made far in advance."""
    df = df.copy()
    df["long_lead_time"] = (df["lead_time"] > threshold).astype(int)
    return df


def add_arrival_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket arrival week number into a quarter (Q1-Q4).
    Upper bin edge is 53, not 52: ISO week numbering can reach week 53
    in some years, and leaving it out of range would silently produce
    NaN for those rows. Labels are non-numeric strings ("Q1"..."Q4")
    so the column round-trips through CSV as text, not a NaN-prone float.
    """
    df = df.copy()
    df["arrival_quarter"] = pd.cut(
        df["arrival_date_week_number"],
        bins=[0, 13, 26, 39, 53],
        labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    return df


def add_previous_booking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add total_previous_bookings and prev_cancel_ratio."""
    df = df.copy()
    df["total_previous_bookings"] = (
        df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    )
    df["prev_cancel_ratio"] = df["previous_cancellations"] / df["total_previous_bookings"].replace(
        0, 1
    )
    return df


def add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue = adr * total_stay. Requires total_stay to already exist."""
    df = df.copy()
    if "total_stay" not in df.columns:
        df = add_stay_features(df)
    df["revenue"] = df["adr"] * df["total_stay"]
    return df


def encode_deposit_type(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode deposit_type (No Deposit < Refundable < Non Refund risk-wise)."""
    df = df.copy()
    df["deposit_type"] = df["deposit_type"].replace(
        {"No Deposit": 0, "Non Refund": 1, "Refundable": 2}
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline in order."""
    df = add_family_features(df)
    df = df[df["total_members"] != 0].copy()
    df = add_stay_features(df)
    df = add_room_changed_flag(df)
    df = add_lead_time_features(df)
    df = add_arrival_quarter(df)
    df = add_previous_booking_features(df)
    df = add_revenue(df)
    df = encode_deposit_type(df)
    return df
