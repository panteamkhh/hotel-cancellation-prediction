import pandas as pd

from src.analysis import (
    cancellation_by_lead_time_bucket,
    cancellation_rate_by,
    cancellation_rate_overall,
    channel_performance_table,
    hotel_type_distribution,
    revenue_by_hotel,
)


def sample_df():
    return pd.DataFrame(
        {
            "hotel": ["City Hotel", "City Hotel", "Resort Hotel", "Resort Hotel"],
            "is_canceled": [1, 0, 0, 0],
            "lead_time": [5, 40, 100, 200],
            "revenue": [100, 200, 150, 300],
            "distribution_channel": ["TA/TO", "Direct", "TA/TO", "Direct"],
        }
    )


def test_hotel_type_distribution_counts():
    counts = hotel_type_distribution(sample_df())
    assert counts["City Hotel"] == 2
    assert counts["Resort Hotel"] == 2


def test_cancellation_rate_overall():
    assert cancellation_rate_overall(sample_df()) == 0.25


def test_cancellation_rate_by_hotel():
    rates = cancellation_rate_by(sample_df(), "hotel")
    assert rates["City Hotel"] == 0.5
    assert rates["Resort Hotel"] == 0.0


def test_cancellation_by_lead_time_bucket_has_all_labels():
    result = cancellation_by_lead_time_bucket(sample_df())
    assert list(result.index) == ["0-7 days", "8-30 days", "31-90 days", "90+ days"]


def test_cancellation_by_lead_time_bucket_survives_short_lead_times():
    # Regression test: when no booking is made more than 90 days ahead, the
    # derived upper bin edge used to make the bins non-monotonic and crash.
    df = pd.DataFrame({"lead_time": [1, 5, 20, 40, 60], "is_canceled": [0, 1, 0, 0, 1]})
    result = cancellation_by_lead_time_bucket(df)
    assert list(result.index) == ["0-7 days", "8-30 days", "31-90 days", "90+ days"]


def test_revenue_by_hotel_sums_correctly():
    result = revenue_by_hotel(sample_df())
    assert result["City Hotel"] == 300
    assert result["Resort Hotel"] == 450


def test_channel_performance_table_columns():
    table = channel_performance_table(sample_df())
    assert set(table.columns) == {"bookings", "revenue", "cancellation_rate"}
    assert table.loc["TA/TO", "bookings"] == 2
