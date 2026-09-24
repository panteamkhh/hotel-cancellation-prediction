import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import theme, visualization  # noqa: E402


@pytest.fixture(autouse=True)
def temp_figures(tmp_path, monkeypatch):
    """Send every saved figure to a temp dir instead of reports/figures/."""
    monkeypatch.setattr(theme, "FIGURES_DIR", tmp_path)
    yield tmp_path
    plt.close("all")


def test_plot_hotel_type_distribution(temp_figures):
    counts = pd.Series([3, 2], index=["City Hotel", "Resort Hotel"])
    visualization.plot_hotel_type_distribution(counts)
    assert (temp_figures / "01_hotel_type_distribution.png").exists()


def test_plot_monthly_booking_trend(temp_figures):
    counts = pd.Series([10, 20], index=["January", "February"])
    visualization.plot_monthly_booking_trend(counts)
    assert (temp_figures / "02_monthly_booking_trend.png").exists()


def test_plot_lead_time_distribution(temp_figures):
    df = pd.DataFrame({"lead_time": [1, 5, 30, 90, 200]})
    visualization.plot_lead_time_distribution(df)
    assert (temp_figures / "03_lead_time_distribution.png").exists()


def test_plot_adr_by_hotel(temp_figures):
    df = pd.DataFrame({"hotel": ["City Hotel", "Resort Hotel"], "adr": [100.0, 90.0]})
    visualization.plot_adr_by_hotel(df)
    assert (temp_figures / "04_adr_by_hotel.png").exists()


def test_plot_correlation_heatmap(temp_figures):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6], "c": ["x", "y", "z"]})
    visualization.plot_correlation_heatmap(df)
    assert (temp_figures / "05_correlation_heatmap.png").exists()


def test_plot_cancellation_rate_by(temp_figures):
    series = pd.Series([0.3, 0.2], index=["City Hotel", "Resort Hotel"])
    visualization.plot_cancellation_rate_by(series, "Title", "10_cancel_rate_by_hotel.png")
    assert (temp_figures / "10_cancel_rate_by_hotel.png").exists()


def test_plot_cancellation_by_lead_time_bucket(temp_figures):
    series = pd.Series([0.1, 0.3], index=["0-7 days", "8-30 days"])
    visualization.plot_cancellation_by_lead_time_bucket(series)
    assert (temp_figures / "06_cancellation_by_lead_time.png").exists()


def test_plot_revenue_by(temp_figures):
    series = pd.Series([300.0, 200.0], index=["City Hotel", "Resort Hotel"])
    visualization.plot_revenue_by(series, "Revenue", "Revenue", "15_revenue_by_hotel.png")
    assert (temp_figures / "15_revenue_by_hotel.png").exists()


def test_plot_monthly_revenue_trend(temp_figures):
    series = pd.Series([100.0, 200.0], index=["January", "February"])
    visualization.plot_monthly_revenue_trend(series)
    assert (temp_figures / "07_monthly_revenue_trend.png").exists()


def test_plot_top_countries(temp_figures):
    series = pd.Series([100, 80], index=["PRT", "GBR"])
    visualization.plot_top_countries(series, "Top", "Bookings", "18_top_countries_volume.png")
    assert (temp_figures / "18_top_countries_volume.png").exists()


def test_plot_stay_duration_by_hotel(temp_figures):
    series = pd.Series([3.0, 4.0], index=["City Hotel", "Resort Hotel"])
    visualization.plot_stay_duration_by_hotel(series)
    assert (temp_figures / "08_stay_duration_by_hotel.png").exists()


def test_plot_lead_time_by_cancellation(temp_figures):
    df = pd.DataFrame({"lead_time": [1, 2, 90, 100], "is_canceled": [0, 0, 1, 1]})
    visualization.plot_lead_time_by_cancellation(df)
    assert (temp_figures / "09_lead_time_by_cancellation.png").exists()
