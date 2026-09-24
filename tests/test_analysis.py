import pandas as pd
import pytest

from src.analysis import (
    adr_by_hotel,
    business_vs_vacation_segments,
    cancellation_by_lead_time_bucket,
    cancellation_rate_by,
    cancellation_rate_by_top_countries,
    cancellation_rate_overall,
    channel_performance_table,
    customer_type_profile,
    hotel_type_distribution,
    lead_time_by_cancellation,
    monthly_booking_trend,
    monthly_revenue_trend,
    previous_cancellations_effect,
    revenue_by_channel,
    revenue_by_country,
    revenue_by_hotel,
    stay_duration_by_hotel,
    top_countries_by_volume,
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


def full_df():
    return pd.DataFrame(
        {
            "hotel": ["City Hotel", "City Hotel", "Resort Hotel", "Resort Hotel", "City Hotel"],
            "is_canceled": [1, 0, 0, 0, 1],
            "lead_time": [5, 40, 100, 200, 10],
            "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
            "distribution_channel": ["TA/TO", "Direct", "TA/TO", "Direct", "TA/TO"],
            "arrival_date_month": ["January", "August", "August", "January", "December"],
            "country": ["PRT", "GBR", "PRT", "FRA", "PRT"],
            "customer_type": ["Transient", "Transient", "Contract", "Group", "Transient"],
            "total_stay": [2, 5, 4, 6, 1],
            "adr": [100.0, 120.0, 90.0, 110.0, 95.0],
            "stays_in_weekend_nights": [0, 2, 2, 3, 0],
            "previous_cancellations": [0, 1, 0, 0, 2],
        }
    )


def test_monthly_booking_trend_is_calendar_ordered():
    trend = monthly_booking_trend(full_df())
    assert trend.index[0] == "January"
    assert trend.index[-1] == "December"
    assert len(trend) == 12
    assert trend["January"] == 2
    assert trend["August"] == 2


def test_adr_by_hotel_returns_hotel_adr_columns():
    result = adr_by_hotel(full_df())
    assert list(result.columns) == ["hotel", "adr"]
    assert len(result) == len(full_df())


def test_cancellation_rate_by_top_countries():
    rates = cancellation_rate_by_top_countries(full_df())
    assert rates["PRT"] == pytest.approx(2 / 3)


def test_previous_cancellations_effect_groups_by_history():
    effect = previous_cancellations_effect(full_df())
    assert effect["Has prior cancellation"] == pytest.approx(0.5)
    assert effect["No prior cancellation"] == pytest.approx(1 / 3)


def test_revenue_by_country_and_channel():
    df = full_df()
    assert revenue_by_country(df)["PRT"] == 500
    assert revenue_by_channel(df)["TA/TO"] == 500


def test_monthly_revenue_trend_sums_by_month():
    trend = monthly_revenue_trend(full_df())
    assert trend["January"] == 400
    assert trend["August"] == 350


def test_customer_type_profile_shares_sum_to_one():
    profile = customer_type_profile(full_df())
    assert {"bookings", "avg_stay", "avg_adr", "cancellation_rate", "share_of_bookings"} <= set(
        profile.columns
    )
    assert profile["share_of_bookings"].sum() == pytest.approx(1.0)


def test_business_vs_vacation_segments_detects_both():
    profile = business_vs_vacation_segments(full_df())
    assert "Business traveler" in profile.index
    assert "Vacation traveler" in profile.index
    assert profile["share_of_bookings"].sum() == pytest.approx(1.0)


def test_stay_duration_by_hotel_averages_nights():
    stays = stay_duration_by_hotel(full_df())
    assert stays["City Hotel"] == pytest.approx(8 / 3)
    assert stays["Resort Hotel"] == pytest.approx(5.0)


def test_lead_time_by_cancellation_returns_expected_columns():
    result = lead_time_by_cancellation(full_df())
    assert list(result.columns) == ["lead_time", "is_canceled"]
    assert len(result) == 5


def test_top_countries_by_volume_ranks_by_count():
    top = top_countries_by_volume(full_df(), top_n=2)
    assert top.index[0] == "PRT"
    assert top["PRT"] == 3
