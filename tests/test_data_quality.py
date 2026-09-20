import pandas as pd

from src.data_quality import (
    constant_columns,
    data_quality_report,
    duplicate_count,
    missing_value_report,
    target_balance,
)


def sample_df():
    return pd.DataFrame(
        {
            "a": [1, 2, 2, None],
            "b": [1, 1, 1, 1],
            "is_canceled": [0, 1, 1, 0],
        }
    )


def test_missing_value_report_lists_only_gaps():
    report = missing_value_report(sample_df())
    assert list(report.index) == ["a"]
    assert report.loc["a", "missing"] == 1
    assert report.loc["a", "share"] == 0.25


def test_duplicate_count_detects_exact_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 3]})
    assert duplicate_count(df) == 1


def test_constant_columns_finds_single_value_columns():
    assert constant_columns(sample_df()) == ["b"]


def test_target_balance_sums_to_one():
    balance = target_balance(sample_df())
    assert balance.sum() == 1.0
    assert balance.loc[1] == 0.5


def test_target_balance_missing_column_raises():
    df = pd.DataFrame({"a": [1, 2]})
    try:
        target_balance(df)
    except KeyError as error:
        assert "not found" in str(error)
    else:
        raise AssertionError("expected KeyError")


def test_data_quality_report_shape_and_keys():
    report = data_quality_report(sample_df())
    assert report["rows"] == 4
    assert report["columns"] == 3
    assert report["duplicates"] == 1
    assert report["missing_cells"] == 1
    assert report["constant_columns"] == ["b"]
    assert report["target_balance"] == {0: 0.5, 1: 0.5}
