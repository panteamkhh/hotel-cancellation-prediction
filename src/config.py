"""
Central place for project paths and shared constants.
Every notebook and module imports paths from here instead of
hardcoding relative strings, so the project still works if it's
run from a different working directory (e.g. via pytest or a CLI).
"""

from pathlib import Path

# project root = two levels up from this file (src/config.py -> project root)
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "hotel_bookings.csv"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "hotel_bookings_clean.csv"
FEATURES_DATA_PATH = PROCESSED_DATA_DIR / "hotel_bookings_features.csv"

MODELS_DIR = ROOT_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

OUTPUT_DIR = ROOT_DIR / "output"

RANDOM_STATE = 42
TARGET = "is_canceled"

# Chronological month order — the raw column is a month *name*, not a number
MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

for _dir in (PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, OUTPUT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
