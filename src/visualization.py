"""
One plotting function per analysis result in analysis.py.
Every function saves its figure into reports/figures/ (so notebooks
and any future CLI produce the same artifacts) and also returns the
matplotlib Axes so a notebook can keep tweaking it if needed.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from src import theme

theme.apply_theme()


def _save(fig, name: str):
    theme.save_figure(fig, name)


def plot_hotel_type_distribution(counts, save_as="01_hotel_type_distribution.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=theme.PALETTE_CATEGORICAL,
        legend=False,
        ax=ax,
    )
    ax.set_title("Bookings by Hotel Type")
    ax.set_ylabel("Bookings")
    ax.set_xlabel("")
    _save(fig, save_as)
    return ax


def plot_monthly_booking_trend(counts, save_as="02_monthly_booking_trend.png"):
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(x=counts.index, y=counts.values, marker="o", ax=ax)
    ax.set_title("Booking Volume by Arrival Month")
    ax.set_ylabel("Bookings")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, save_as)
    return ax


def plot_lead_time_distribution(df, save_as="03_lead_time_distribution.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["lead_time"], bins=30, kde=True, ax=ax)
    ax.set_title("Lead Time Distribution")
    _save(fig, save_as)
    return ax


def plot_adr_by_hotel(df, save_as="04_adr_by_hotel.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        x="hotel",
        y="adr",
        data=df,
        hue="hotel",
        palette=theme.PALETTE_HOTEL,
        legend=False,
        ax=ax,
    )
    ax.set_title("ADR by Hotel Type")
    _save(fig, save_as)
    return ax


def plot_correlation_heatmap(df, save_as="05_correlation_heatmap.png"):
    import numpy as np

    num_cols = df.select_dtypes(include=[np.number]).columns
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric Feature Correlation")
    _save(fig, save_as)
    return ax


def plot_cancellation_rate_by(series, title, save_as):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        x=series.index,
        y=series.values,
        hue=series.index,
        palette=theme.PALETTE_CANCELLATION,
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Cancellation rate")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, save_as)
    return ax


def plot_cancellation_by_lead_time_bucket(series, save_as="06_cancellation_by_lead_time.png"):
    return plot_cancellation_rate_by(series, "Cancellation Rate by Lead Time", save_as)


def plot_revenue_by(series, title, ylabel, save_as):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        x=series.index,
        y=series.values,
        hue=series.index,
        palette=theme.PALETTE_REVENUE,
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, save_as)
    return ax


def plot_monthly_revenue_trend(series, save_as="07_monthly_revenue_trend.png"):
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(x=series.index, y=series.values, marker="o", color="seagreen", ax=ax)
    ax.set_title("Revenue by Arrival Month")
    ax.set_ylabel("Revenue")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, save_as)
    return ax


def plot_top_countries(series, title, ylabel, save_as):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        x=series.index,
        y=series.values,
        hue=series.index,
        palette=theme.PALETTE_COUNTRY,
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, save_as)
    return ax


def plot_stay_duration_by_hotel(series, save_as="08_stay_duration_by_hotel.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=series.index,
        y=series.values,
        hue=series.index,
        palette=theme.PALETTE_STAY,
        legend=False,
        ax=ax,
    )
    ax.set_title("Average Stay Duration by Hotel Type")
    ax.set_ylabel("Avg nights")
    ax.set_xlabel("")
    _save(fig, save_as)
    return ax


def plot_lead_time_by_cancellation(df, save_as="09_lead_time_by_cancellation.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=df,
        x="lead_time",
        hue="is_canceled",
        bins=30,
        kde=True,
        ax=ax,
        palette=theme.CANCEL_COLORS,
    )
    ax.set_title("Lead Time Distribution: Canceled vs. Not Canceled")
    _save(fig, save_as)
    return ax
