# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [1.1.0] - 2026-09-19

### Added

- `src/data_loader.py` — central CSV loading with schema validation and clear
  errors for missing/malformed files.
- `src/theme.py` — shared chart theme so notebooks and CLI produce identical
  styling; `src/visualization.py` and `src/evaluation.py` now use it.
- `src/train.py` — CLI that fits and compares all four models and writes
  `models/best_model.pkl` + `models/model_metrics.json`.
- `src/predict.py` — score new bookings and attach a cancellation probability.
- `src/run_analysis.py` — regenerate all 18 report figures and export summary
  tables to `output/` without opening a notebook.
- `docs/data_dictionary.md`, `docs/model_card.md`, `docs/project_workflow.md`
  and `notebooks/README.md`.
- `Makefile`, `.gitattributes`, `.editorconfig`, `CITATION.cff`, GitHub issue
  and pull-request templates, and `output/` for generated CSV exports.
- Tests for the new loader, theme and prediction modules (suite now 36 tests).
- Console entry points: `hotel-pipeline`, `hotel-analysis`, `hotel-train`.

### Fixed

- `analysis.cancellation_by_lead_time_bucket` no longer crashes when no booking
  has a lead time above 90 days (non-monotonic bin edges).
- `data_cleaning.remove_adr_outliers` works on a copy, avoiding chained
  assignment warnings.

### Changed

- Raw and processed CSVs and the trained model are now committed so the repo is
  reproducible without downloading from Kaggle.
- Runtime requirements no longer include Jupyter (moved to dev extras).
- Project metadata, URLs and README rewritten around the new structure.

## [1.0.0] - 2026-09-14

### Added

- Full data-cleaning pipeline (`src/data_cleaning.py`): duplicate removal,
  missing-value handling, invalid/outlier-row removal, target-leakage
  column removal.
- Feature engineering pipeline (`src/feature_engineering.py`): family/stay
  features, room-change flag, lead-time flag, arrival quarter, prior-booking
  ratio, revenue, ordinal deposit encoding.
- Business-question analysis functions (`src/analysis.py`) and matching
  chart functions (`src/visualization.py`), one pair per question.
- Nine end-to-end notebooks (`notebooks/01`–`09`) covering cleaning, EDA,
  cancellation analysis, revenue analysis, customer segmentation, booking
  behavior, channel/geography analysis, feature engineering, and modeling —
  fully executed with real outputs and saved figures in `reports/figures/`.
- Cancellation-prediction models: Logistic Regression, Random Forest,
  XGBoost, LightGBM, compared on precision/recall/F1/ROC-AUC
  (`src/evaluation.py`). Best model (LightGBM) saved to `models/best_model.pkl`.
- `src/run_pipeline.py` — single-command CLI that reproduces the full
  pipeline (clean → feature-engineer → train → evaluate) outside the
  notebooks.
- `tests/` — pytest suite covering `data_cleaning.py`, `feature_engineering.py`,
  and `analysis.py`.
- `reports/final_report.md` — written summary of findings and recommendations.
- CI (`.github/workflows/ci.yml`): lint (ruff), format check (black), tests
  (pytest) on Python 3.10–3.12. Pre-commit hooks for the same.

### Changed

- `add_arrival_quarter` now bins on `(0, 53]` instead of `(0, 52]` and uses
  string labels (`Q1`–`Q4`) instead of numeric-looking ones — the previous
  version silently produced `NaN` for ISO week 53 and lost its categorical
  dtype on a CSV round-trip, which broke one-hot encoding downstream.

### Removed

- Power BI dashboard and all dashboard-related files/sections — this
  project is Python-only end-to-end (no BI tool dependency).
