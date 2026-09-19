"""
Loading and schema validation for the project's CSV files.

Keeping every read in one place means a missing or malformed file fails
loudly and early (with a clear message) instead of surfacing later as a
confusing ``KeyError`` deep inside an analysis function.
"""

from pathlib import Path

import pandas as pd

from src.config import CLEAN_DATA_PATH, FEATURES_DATA_PATH, RAW_DATA_PATH

# Columns the raw Kaggle CSV is expected to contain.
REQUIRED_RAW_COLUMNS = [
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status",
    "reservation_status_date",
]


class DataValidationError(ValueError):
    """Raised when a data file does not match the expected schema."""


def validate_schema(
    df: pd.DataFrame, required=REQUIRED_RAW_COLUMNS, name: str = "dataframe"
) -> None:
    """Raise ``DataValidationError`` if any required column is missing."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataValidationError(
            f"{name} is missing {len(missing)} required column(s): {', '.join(missing)}"
        )


def load_csv(path, required=None, name: str | None = None) -> pd.DataFrame:
    """Read a CSV, raising a helpful error if it is absent, then validate it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the raw CSV from Kaggle "
            "(see data/raw/README.md) or run the pipeline first."
        )
    df = pd.read_csv(path)
    if required:
        validate_schema(df, required, name or path.name)
    return df


def load_raw_data(path=None) -> pd.DataFrame:
    """Load and validate the raw hotel bookings CSV."""
    return load_csv(path or RAW_DATA_PATH, REQUIRED_RAW_COLUMNS, "raw bookings CSV")


def load_clean_data(path=None) -> pd.DataFrame:
    """Load the cleaned CSV produced by the cleaning stage."""
    return load_csv(path or CLEAN_DATA_PATH, name="clean bookings CSV")


def load_features_data(path=None) -> pd.DataFrame:
    """Load the feature-engineered CSV produced by the feature stage."""
    return load_csv(path or FEATURES_DATA_PATH, name="features CSV")
