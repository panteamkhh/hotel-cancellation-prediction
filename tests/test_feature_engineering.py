import pandas as pd

from src.feature_engineering import (
    add_arrival_quarter,
    add_family_features,
    add_lead_time_features,
    add_previous_booking_features,
    add_revenue,
    add_room_changed_flag,
    add_stay_features,
    encode_deposit_type,
    engineer_features,
)


def base_df():
    return pd.DataFrame(
        {
            "hotel": ["City Hotel", "Resort Hotel"],
            "adults": [2, 1],
            "children": [0, 1],
            "babies": [0, 0],
            "stays_in_weekend_nights": [1, 2],
            "stays_in_week_nights": [2, 3],
            "reserved_room_type": ["A", "B"],
            "assigned_room_type": ["A", "C"],
            "lead_time": [10, 200],
            "arrival_date_week_number": [5, 53],
            "previous_cancellations": [0, 2],
            "previous_bookings_not_canceled": [0, 3],
            "adr": [100.0, 50.0],
            "deposit_type": ["No Deposit", "Non Refund"],
        }
    )


def test_add_family_features():
    df = add_family_features(base_df())
    assert list(df["total_members"]) == [2, 2]
    assert list(df["has_family"]) == [0, 1]


def test_add_stay_features():
    df = add_stay_features(base_df())
    assert list(df["total_stay"]) == [3, 5]


def test_add_room_changed_flag():
    df = add_room_changed_flag(base_df())
    assert list(df["room_changed"]) == [0, 1]


def test_add_lead_time_features_threshold():
    df = add_lead_time_features(base_df(), threshold=90)
    assert list(df["long_lead_time"]) == [0, 1]


def test_add_arrival_quarter_handles_week_53_without_nan():
    df = add_arrival_quarter(base_df())
    assert df["arrival_quarter"].isnull().sum() == 0
    assert df.loc[1, "arrival_quarter"] == "Q4"


def test_add_previous_booking_features_ratio():
    df = add_previous_booking_features(base_df())
    assert df.loc[0, "total_previous_bookings"] == 0
    assert df.loc[0, "prev_cancel_ratio"] == 0  # guarded against div-by-zero
    assert df.loc[1, "prev_cancel_ratio"] == 2 / 5


def test_add_revenue_computes_from_stay_and_adr():
    df = add_revenue(base_df())
    assert list(df["revenue"]) == [300.0, 250.0]


def test_encode_deposit_type_is_ordinal():
    df = encode_deposit_type(base_df())
    assert list(df["deposit_type"]) == [0, 1]


def test_engineer_features_produces_no_nan_and_expected_columns():
    df = engineer_features(base_df())
    assert df.isnull().sum().sum() == 0
    for col in [
        "total_members",
        "total_stay",
        "room_changed",
        "long_lead_time",
        "arrival_quarter",
        "prev_cancel_ratio",
        "revenue",
    ]:
        assert col in df.columns
