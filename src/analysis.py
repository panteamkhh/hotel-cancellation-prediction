"""
One function per business question. Each function takes the cleaned/
feature-engineered DataFrame and returns a small, notebook-ready
pandas object (Series/DataFrame) — no plotting here, see
visualization.py for the matching chart of each result.
"""

import pandas as pd

from src.config import MONTH_ORDER


# ---------------------------------------------------------------- EDA ---
def hotel_type_distribution(df: pd.DataFrame) -> pd.Series:
    """Booking count per hotel type."""
    return df["hotel"].value_counts()


def monthly_booking_trend(df: pd.DataFrame) -> pd.Series:
    """Booking count per arrival month, in calendar order (not alphabetical)."""
    counts = df["arrival_date_month"].value_counts()
    return counts.reindex(MONTH_ORDER)


def adr_by_hotel(df: pd.DataFrame) -> pd.DataFrame:
    """Average Daily Rate distribution per hotel type (for a boxplot)."""
    return df[["hotel", "adr"]]


# ------------------------------------------------------- Cancellation ---
def cancellation_rate_overall(df: pd.DataFrame) -> float:
    return df["is_canceled"].mean()


def cancellation_rate_by(df: pd.DataFrame, column: str) -> pd.Series:
    """Cancellation rate grouped by any categorical column, sorted descending."""
    return df.groupby(column)["is_canceled"].mean().sort_values(ascending=False)


def cancellation_rate_by_top_countries(df: pd.DataFrame, top_n: int = 15) -> pd.Series:
    """Cancellation rate for the top-N countries by booking volume (avoids noisy small-N countries)."""
    top_countries = df["country"].value_counts().head(top_n).index
    subset = df[df["country"].isin(top_countries)]
    return subset.groupby("country")["is_canceled"].mean().sort_values(ascending=False)


LEAD_TIME_BUCKETS = ["0-7 days", "8-30 days", "31-90 days", "90+ days"]


def cancellation_by_lead_time_bucket(
    df: pd.DataFrame,
    labels: list | None = None,
) -> pd.Series:
    """
    Cancellation rate bucketed by how far in advance the booking was made.

    The upper bin edge is derived from the data but floored at 91 so the
    bin edges are always strictly increasing: with a small subset where no
    booking is made more than 90 days ahead, using ``lead_time.max()``
    directly would produce non-monotonic bins and ``pd.cut`` would raise.
    Empty buckets are kept (as NaN) so the returned index is stable.
    """
    labels = labels or LEAD_TIME_BUCKETS
    upper = max(int(df["lead_time"].max()) + 1, 91) if len(df) else 91
    buckets = pd.cut(
        df["lead_time"],
        bins=[-1, 7, 30, 90, upper],
        labels=labels,
        include_lowest=True,
    )
    return df.groupby(buckets, observed=False)["is_canceled"].mean().reindex(labels)


def previous_cancellations_effect(df: pd.DataFrame) -> pd.Series:
    """Cancellation rate for guests with vs. without a prior cancellation on record."""
    has_prior = (df["previous_cancellations"] > 0).map(
        {True: "Has prior cancellation", False: "No prior cancellation"}
    )
    return df.groupby(has_prior)["is_canceled"].mean()


# ------------------------------------------------------------ Revenue ---
def revenue_by_hotel(df: pd.DataFrame) -> pd.Series:
    return df.groupby("hotel")["revenue"].sum().sort_values(ascending=False)


def monthly_revenue_trend(df: pd.DataFrame) -> pd.Series:
    revenue = df.groupby("arrival_date_month")["revenue"].sum()
    return revenue.reindex(MONTH_ORDER)


def revenue_by_country(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    return df.groupby("country")["revenue"].sum().sort_values(ascending=False).head(top_n)


def revenue_by_channel(df: pd.DataFrame) -> pd.Series:
    return df.groupby("distribution_channel")["revenue"].sum().sort_values(ascending=False)


def channel_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """bookings, revenue, and cancellation rate per distribution channel."""
    table = df.groupby("distribution_channel").agg(
        bookings=("hotel", "count"),
        revenue=("revenue", "sum"),
        cancellation_rate=("is_canceled", "mean"),
    )
    return table.sort_values("revenue", ascending=False)


# ---------------------------------------------------------- Segments ---
def customer_type_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Share of bookings, avg stay, avg adr and cancel rate per customer_type."""
    profile = df.groupby("customer_type").agg(
        bookings=("hotel", "count"),
        avg_stay=("total_stay", "mean"),
        avg_adr=("adr", "mean"),
        cancellation_rate=("is_canceled", "mean"),
    )
    profile["share_of_bookings"] = profile["bookings"] / profile["bookings"].sum()
    return profile.sort_values("bookings", ascending=False)


def business_vs_vacation_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based split: short, weekday-heavy stays booked with little
    lead time look like business travel; longer, weekend-heavy stays
    booked further out look like vacation travel. Everything else is
    'Mixed/Other'.
    """
    d = df.copy()
    weekend_share = d["stays_in_weekend_nights"] / d["total_stay"].replace(0, 1)

    is_business = (d["total_stay"] <= 3) & (weekend_share < 0.3) & (d["lead_time"] < 30)
    is_vacation = (d["total_stay"] >= 4) & (weekend_share >= 0.3)

    segment = pd.Series("Mixed/Other", index=d.index)
    segment[is_business] = "Business traveler"
    segment[is_vacation] = "Vacation traveler"
    d["segment"] = segment

    profile = d.groupby("segment").agg(
        bookings=("hotel", "count"),
        avg_stay=("total_stay", "mean"),
        avg_adr=("adr", "mean"),
        cancellation_rate=("is_canceled", "mean"),
    )
    profile["share_of_bookings"] = profile["bookings"] / profile["bookings"].sum()
    return profile.sort_values("bookings", ascending=False)


# ----------------------------------------------------- Booking behavior ---
def stay_duration_by_hotel(df: pd.DataFrame) -> pd.Series:
    return df.groupby("hotel")["total_stay"].mean()


def lead_time_by_cancellation(df: pd.DataFrame) -> pd.DataFrame:
    """Raw lead_time + is_canceled, for a hue-split histogram."""
    return df[["lead_time", "is_canceled"]]


# --------------------------------------------------------- Geography ---
def top_countries_by_volume(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    return df["country"].value_counts().head(top_n)
