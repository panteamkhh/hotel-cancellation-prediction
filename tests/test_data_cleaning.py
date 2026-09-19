import pandas as pd
import pytest

from src.data_cleaning import (
    clean_pipeline,
    drop_duplicates,
    drop_leakage_columns,
    handle_missing_values,
    remove_adr_outliers,
    remove_zero_guest_bookings,
)


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "adults": [2, 2, 0, 1, 1],
            "children": [0.0, None, 0.0, 1.0, 0.0],
            "babies": [0, 0, 0, 0, 0],
            "company": [None, 123.0, None, None, None],
            "agent": [9.0, None, 9.0, 240.0, 9.0],
            "country": ["PRT", None, "GBR", "PRT", "PRT"],
            "adr": [100.0, -5.0, 90.0, 95.0, 10000.0],
            "reservation_status": ["Check-Out"] * 5,
            "reservation_status_date": ["2017-01-01"] * 5,
        }
    )


def test_drop_duplicates_removes_exact_dupes():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    result = drop_duplicates(df)
    assert len(result) == 2


def test_handle_missing_values_fills_and_drops(raw_df):
    result = handle_missing_values(raw_df)
    assert "company" not in result.columns
    assert result["children"].isnull().sum() == 0
    assert result["agent"].isnull().sum() == 0
    assert result["country"].isnull().sum() == 0
    assert (result["agent"] == "no agent").sum() == 1


def test_remove_zero_guest_bookings_drops_all_zero_rows(raw_df):
    result = remove_zero_guest_bookings(raw_df)
    # row index 2 has adults=0, children=0, babies=0 -> should be removed
    assert 2 not in result.index
    assert len(result) == len(raw_df) - 1


def test_remove_adr_outliers_drops_negative_and_extreme(raw_df):
    result = remove_adr_outliers(raw_df, upper_quantile=0.8)
    assert (result["adr"] >= 0).all()
    assert 10000.0 not in result["adr"].values


def test_drop_leakage_columns_removes_target_leak(raw_df):
    result = drop_leakage_columns(raw_df)
    assert "reservation_status" not in result.columns
    assert "reservation_status_date" not in result.columns


def test_drop_leakage_columns_is_safe_if_columns_missing():
    df = pd.DataFrame({"a": [1, 2]})
    result = drop_leakage_columns(df)
    assert list(result.columns) == ["a"]


def test_clean_pipeline_runs_end_to_end(raw_df):
    result = clean_pipeline(raw_df)
    assert result.isnull().sum().sum() == 0
    assert (result["adr"] >= 0).all()
