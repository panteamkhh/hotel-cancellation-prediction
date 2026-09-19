"""
Regenerate every report figure and summary table from the processed data.

This is the CLI counterpart to notebooks 02-07: it produces the same
``reports/figures/*.png`` charts and writes the underlying numbers as CSVs
into ``output/`` so the written report can be rebuilt without opening a
notebook:

    python -m src.run_analysis

It expects ``data/processed/hotel_bookings_clean.csv`` to exist (run
``python -m src.run_pipeline`` first if it does not).
"""

from src import analysis, visualization
from src.config import OUTPUT_DIR
from src.data_loader import load_clean_data
from src.feature_engineering import add_revenue, add_stay_features


def _export(table, name: str) -> None:
    """Write a Series/DataFrame to ``output/<name>.csv``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / name, header=True)
    print(f"   saved output/{name}")


def run_eda(clean) -> None:
    """Figures 01-05: composition, seasonality, ADR and correlations."""
    visualization.plot_hotel_type_distribution(analysis.hotel_type_distribution(clean))
    visualization.plot_monthly_booking_trend(analysis.monthly_booking_trend(clean))
    visualization.plot_lead_time_distribution(clean)
    visualization.plot_adr_by_hotel(analysis.adr_by_hotel(clean))
    visualization.plot_correlation_heatmap(clean)
    _export(analysis.hotel_type_distribution(clean), "hotel_type_distribution.csv")
    _export(analysis.monthly_booking_trend(clean), "monthly_booking_trend.csv")


def run_cancellation(clean) -> None:
    """Figures 06, 09-14: cancellation rates by segment and lead time."""
    visualization.plot_cancellation_by_lead_time_bucket(
        analysis.cancellation_by_lead_time_bucket(clean)
    )
    visualization.plot_lead_time_by_cancellation(analysis.lead_time_by_cancellation(clean))
    visualization.plot_cancellation_rate_by(
        analysis.cancellation_rate_by(clean, "hotel"),
        "Cancellation Rate by Hotel Type",
        "10_cancel_rate_by_hotel.png",
    )
    visualization.plot_cancellation_rate_by(
        analysis.cancellation_rate_by_top_countries(clean, top_n=15),
        "Cancellation Rate by Country (Top 15 by Volume)",
        "11_cancel_rate_by_country.png",
    )
    visualization.plot_cancellation_rate_by(
        analysis.cancellation_rate_by(clean, "customer_type"),
        "Cancellation Rate by Customer Type",
        "12_cancel_rate_by_customer_type.png",
    )
    visualization.plot_cancellation_rate_by(
        analysis.cancellation_rate_by(clean, "deposit_type"),
        "Cancellation Rate by Deposit Type",
        "13_cancel_rate_by_deposit.png",
    )
    visualization.plot_cancellation_rate_by(
        analysis.previous_cancellations_effect(clean),
        "Cancellation Rate by Prior Cancellation History",
        "14_cancel_rate_by_prior_history.png",
    )
    _export(
        analysis.cancellation_rate_by_top_countries(clean, top_n=15),
        "cancel_rate_top_countries.csv",
    )


def run_revenue(revenue_df) -> None:
    """Figures 07, 15-17: revenue totals, seasonality and channel mix."""
    visualization.plot_monthly_revenue_trend(analysis.monthly_revenue_trend(revenue_df))
    visualization.plot_revenue_by(
        analysis.revenue_by_hotel(revenue_df),
        "Total Revenue by Hotel Type",
        "Revenue",
        "15_revenue_by_hotel.png",
    )
    visualization.plot_revenue_by(
        analysis.revenue_by_country(revenue_df, top_n=10),
        "Top 10 Countries by Revenue",
        "Revenue",
        "16_revenue_by_country.png",
    )
    visualization.plot_revenue_by(
        analysis.revenue_by_channel(revenue_df),
        "Revenue by Distribution Channel",
        "Revenue",
        "17_revenue_by_channel.png",
    )
    _export(analysis.channel_performance_table(revenue_df), "channel_performance.csv")


def run_behavior(clean_with_stay) -> None:
    """Figure 08: average stay duration by hotel type."""
    visualization.plot_stay_duration_by_hotel(analysis.stay_duration_by_hotel(clean_with_stay))
    _export(analysis.customer_type_profile(clean_with_stay), "customer_type_profile.csv")
    _export(analysis.business_vs_vacation_segments(clean_with_stay), "customer_segments.csv")


def run_geography(clean) -> None:
    """Figure 18: top countries by booking volume."""
    visualization.plot_top_countries(
        analysis.top_countries_by_volume(clean, top_n=10),
        "Top 10 Countries by Booking Volume",
        "Bookings",
        "18_top_countries_volume.png",
    )


def run() -> None:
    print("Loading clean bookings data...")
    clean = load_clean_data()
    clean_with_stay = add_stay_features(clean)
    revenue_df = add_revenue(clean)

    print("1/5 - EDA figures...")
    run_eda(clean)
    print("2/5 - Cancellation figures...")
    run_cancellation(clean)
    print("3/5 - Revenue figures...")
    run_revenue(revenue_df)
    print("4/5 - Booking-behavior figures...")
    run_behavior(clean_with_stay)
    print("5/5 - Geography figures...")
    run_geography(clean)

    print("\nDone. Figures -> reports/figures/, tables -> output/")


if __name__ == "__main__":
    run()
