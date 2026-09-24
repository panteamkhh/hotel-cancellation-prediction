# Roadmap

Tracked improvements, in rough priority order. Each item says why it matters.

## Modeling

- **Time-based validation.** Replace the random train/test split with a
  chronological split (train on earlier arrivals, test on later ones). The
  current random split is optimistic for deployment; this is the single most
  important next step (`docs/model_card.md`).
- **Threshold tuning in production.** Wire `src/threshold_analysis.py` into the
  evaluation path so the reported operating point reflects the real cost ratio
  between a missed cancellation and a false alarm.
- **Probability calibration.** Add isotonic/Platt calibration so predicted
  probabilities can be compared across segments.
- **Feature versioning.** Persist the exact one-hot column list with the model
  so `src/predict.py` can align features without guesswork.
- **SHAP explanations.** Per-booking reason codes, not just global importance.

## Data

- **Refresh the data.** 2015–2017 behavior may have shifted; retrain on recent
  bookings before any real use.
- **True revenue join.** Replace the `adr * total_stay` proxy with actual folio
  revenue, including fees and no-show charges.
- **Automated data quality gate.** Call `src/data_quality.py` at the start of
  `run_pipeline` and fail fast on unexpected missing/duplicate rates.

## Engineering

- **Add mypy.** Type-check `src/` in CI.
- **Parquet intermediates.** Faster to read than the 12 MB CSVs for repeated runs.
- **Container image.** A `Dockerfile` so the pipeline runs identically anywhere.
- **Notebook CI execution.** Run `nbconvert --execute` for the notebooks that
  do not require the boosting libraries.

## Reporting

- **Model card in the report.** Surface `docs/model_card.md` from
  `reports/final_report.md`.
- **Interactive dashboard.** Optional; the project stays Python-only by design.
