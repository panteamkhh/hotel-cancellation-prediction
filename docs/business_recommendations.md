# Business recommendations

These follow directly from the analysis in `reports/final_report.md` and the
model in `models/best_model.pkl`. Each one names the evidence behind it.

## 1. Act on long-lead-time bookings

Bookings made 90+ days ahead cancel at 36.9% versus 8.4% within a week — a
4.4× gap — and they are made far enough out that there is time to intervene.

- Send a confirmation/re-confirmation touchpoint 7–14 days before arrival.
- Consider a small, clearly disclosed deposit for the longest lead times.

*Evidence:* `reports/figures/06_cancellation_by_lead_time.png`.

## 2. Treat prior cancellation history as a first-class risk signal

A guest with at least one previous cancellation cancels again 68.0% of the
time, versus 26.7% for guests with none.

- Surface this flag to the reservations team at booking time.
- Use it to prioritize which bookings get a retention offer.

*Evidence:* `reports/figures/14_cancel_rate_by_prior_history.png`.

## 3. Re-examine the non-refundable deposit policy

`Non Refund` bookings cancel 94.7% of the time — higher than `No Deposit`
(26.7%) and `Refundable` (24.3%). The deposit is not acting as a deterrent.

- Audit which channel is applying `Non Refund` and why.
- Test channel-specific terms instead of a blanket non-refundable policy.

*Evidence:* `reports/figures/13_cancel_rate_by_deposit.png`.

## 4. Plan around the seasonal revenue curve

Revenue peaks in August (~$7.1M) and bottoms in January (~$1.1M) — a >6× swing.

- Build rate and inventory strategy ahead of the July–August peak.
- Use flexible terms or promotions in the January–February trough.

*Evidence:* `reports/figures/07_monthly_revenue_trend.png`.

## 5. Protect the high-volume TA/TO channel

Travel agents / tour operators drive the most bookings and revenue ($28.5M) but
also the highest cancellation rate among meaningful channels (30.9%).

- A one-point reduction in TA/TO cancellations moves more revenue than the same
  reduction anywhere else.
- Review cancellation terms with the largest TA/TO partners.

*Evidence:* `reports/figures/17_revenue_by_channel.png`.

## 6. Use a cost-aware decision threshold

The default 0.5 cutoff treats a missed cancellation and a false alarm as equal.
`src/threshold_analysis.py` lets you set the real cost ratio and pick the
threshold that minimises expected cost.

```python
from src.threshold_analysis import best_threshold
threshold = best_threshold(y_true, y_proba, cost_false_negative=10, cost_false_positive=1)
```
