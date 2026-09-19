# Model card — hotel cancellation predictor

## Overview

| | |
|---|---|
| **Task** | Binary classification — will a booking be canceled? |
| **Target** | `is_canceled` (1 = canceled) |
| **Best model** | LightGBM (`LGBMClassifier`, 300 estimators) |
| **Data** | 87,139 cleaned bookings, 2015–2017, City + Resort Hotel |
| **Split** | 80/20 stratified train/test, `random_state=42` |
| **Features** | 573 columns after one-hot encoding the categoricals |
| **Artifact** | `models/best_model.pkl`, metrics in `models/model_metrics.json` |
| **Reproduce** | `python -m src.train` (comparison) or `python -m src.run_pipeline` (fast path) |

## Intended use

Flag bookings that are likely to be canceled **at booking time**, so the hotel
can prioritize overbooking decisions, deposit policy and retention offers.
Scores are decision support, not an automated cancellation.

## Not intended for

- Any decision after arrival or checkout (features are booking-time only).
- Treating a high score as proof of intent to cancel.
- Extrapolating to other hotels, markets or time periods without retraining.

## Performance (held-out test set)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.692 | 0.549 | 0.612 | 0.871 |
| Random Forest | 0.783 | 0.628 | 0.697 | 0.910 |
| XGBoost | 0.760 | 0.689 | 0.723 | 0.919 |
| **LightGBM** | 0.762 | 0.697 | 0.728 | **0.922** |

LightGBM has the best ROC-AUC and F1. Random Forest is marginally more
precise but misses far more actual cancellations — the wrong trade-off when
the cost of a missed cancellation is high.

## Top predictors

1. `lead_time` — strongest behavioral signal.
2. `deposit_type` — `Non Refund` bookings cancel ~94.7% of the time.
3. `country` — origin effects.
4. `previous_cancellations` — prior behavior predicts future behavior.
5. `market_segment` / `distribution_channel`.

## Limitations

- **No temporal validation.** The split is random, not time-ordered, so the
  reported metrics are optimistic relative to a real forward-in-time deployment.
- **Derived revenue.** `revenue = adr * total_stay` is a proxy; the dataset has
  no true revenue field.
- **Non-refundable paradox.** `Non Refund` deposits correlate with a specific
  high-risk channel, so the model may partly learn channel artifacts.
- **Class imbalance.** ~27.5% positive class; precision/recall depend on the
  0.5 threshold and should be tuned to the business cost ratio.
- **Data vintage.** 2015–2017 behavior may not reflect current booking patterns.

## Ethical notes

- `country` is a proxy for guest origin. It is used here for analysis only and
  should not be used to deny service or price individual guests unfairly.
- No personally identifiable information is present in the dataset.
