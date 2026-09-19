# Hotel Booking Cancellation Analysis — Final Report

## 1. Business Problem

A hotel chain operating a City Hotel and a Resort Hotel wants to understand
customer booking behavior, revenue drivers, and — most importantly — why
bookings get canceled, so it can intervene before a cancellation happens
rather than reacting to it. Every canceled booking is lost revenue and a
planning headache (staffing, inventory, overbooking strategy), so a model
that flags high-risk bookings at the time they're made has direct
operational value.

## 2. Data

- Source: Kaggle — Hotel Booking Demand
- Rows analyzed: 87,139 (after cleaning; 119,390 raw, including 31,994 exact
  duplicates removed)
- Time period: 2015–2017
- Hotels: City Hotel (61.1% of bookings), Resort Hotel (38.9%)

## 3. Cancellation Analysis

- **Overall cancellation rate: 27.5%** (post-cleaning; a commonly-quoted
  ~37% figure comes from not removing duplicates first).
- By hotel type: City Hotel 30.1% vs Resort Hotel 23.5%.
- By lead time: cancellation rate climbs steadily with how far ahead the
  booking is made — 8.4% (0–7 days), 25.4% (8–30 days), 32.0% (31–90 days),
  **36.9% (90+ days)** — a 4.4x spread between the shortest and longest
  lead-time buckets.
- By deposit type: "Non Refund" bookings cancel **94.7%** of the time, vs
  26.7% for "No Deposit" and 24.3% for "Refundable." This is
  counterintuitive on its face, but in this dataset it reflects a specific
  high-risk booking channel that happens to require a non-refundable
  deposit, not a causal effect of the deposit itself.
- By prior history: guests with at least one previous cancellation cancel
  again 68.0% of the time, vs 26.7% for guests with no such history —
  by far the single strongest behavioral signal available before lead time.
- Highest-risk countries (top 15 by volume): Brazil (36.5%), Portugal
  (35.3%), and Italy (35.1%) have the highest cancellation rates; Austria
  (18.0%) and the Netherlands (18.3%) the lowest.
- Chart: `figures/06_cancellation_by_lead_time.png`, `figures/13_cancel_rate_by_deposit.png`

## 4. Revenue Analysis

- **Total revenue (cleaned dataset): ~$34.3M**, computed as
  `adr * (weekend nights + week nights)` per booking.
- By hotel: City Hotel $18.7M, Resort Hotel $15.6M.
- Monthly revenue is highly seasonal, peaking in August (~$7.1M) and
  bottoming out in January (~$1.1M) — a >6x swing between the best and
  worst month.
- Top revenue country: Portugal (~$9.1M), followed by the UK (~$4.7M) and
  France (~$3.6M).
- Top revenue driver (channel): **TA/TO** (travel agents / tour operators)
  — $28.5M, dwarfing Direct ($4.9M), Corporate ($0.83M), and GDS ($44k).
- Chart: `figures/07_monthly_revenue_trend.png`, `figures/17_revenue_by_channel.png`

## 5. Customer Segments

Rule-based segmentation on stay length, weekend-night share, and lead time
(see `notebooks/05_customer_segmentation.ipynb` for the exact rule):

| Segment | Profile | Share of bookings | Cancellation rate |
|---|---|---|---|
| Business traveler | Short stay (≤3 nights), weekday-heavy, booked <30 days out | 19.7% | 13.9% |
| Vacation traveler | Longer stay (≥4 nights), weekend-heavy | 14.9% | 31.2% |
| Mixed/Other | Doesn't fit either pattern cleanly | 65.4% | 30.8% |

By the dataset's own `customer_type` field, "Transient" guests dominate
(82.4% of bookings) and cancel the most (30.1%), while "Group" bookings are
rare (0.6%) but far more reliable (9.8% cancellation rate).

## 6. Predictive Model

Feature-engineered dataset (87,139 rows, 39 columns before encoding, 573
after one-hot encoding categorical fields), 80/20 stratified split:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.692 | 0.549 | 0.612 | 0.871 |
| Random Forest | 0.783 | 0.628 | 0.697 | 0.910 |
| XGBoost | 0.760 | 0.689 | 0.723 | 0.919 |
| **LightGBM** | **0.762** | **0.697** | **0.728** | **0.922** |

**LightGBM is the best overall model** (highest ROC-AUC, recall, and F1),
trading a small amount of precision against Random Forest for a much
better catch rate on actual cancellations — the more operationally useful
trade-off, since missing a cancellation is costlier than a false alarm.

**Top predictive features:** lead time, deposit type (particularly the
non-refundable flag), country, and prior cancellation history — consistent
with the patterns found in the cancellation analysis above. See
`figures/` (feature importance chart) and `models/best_model.pkl` for the
saved model. Intended use, limitations and ethical notes are documented in
[`docs/model_card.md`](../docs/model_card.md).

## 7. Business Recommendations

1. **Flag long-lead-time, non-refundable-deposit bookings for proactive
   outreach** (a confirmation call/email close to arrival) — this segment
   carries both the highest cancellation rate and, being booked far out,
   the most lead time to act on it.
2. **Treat a guest's prior-cancellation history as a first-class risk
   signal** at booking time; a returning guest who canceled before is
   roughly 2.5x more likely to cancel again than a first-timer.
3. **Revisit the "Non Refund" deposit policy for its associated channel** —
   it is not functioning as a deterrent (94.7% cancellation rate); either
   the channel itself needs tighter terms or the deposit type is being
   applied in a context where it doesn't change guest behavior.
4. **Use the seasonal revenue curve for staffing and pricing**: build up
   rate and inventory management ahead of the July–August peak, and
   consider promotions or flexible cancellation terms in the January–February
   trough to smooth demand.
5. **Prioritize retention efforts on high-volume, high-cancellation
   channels** — TA/TO brings the most bookings and revenue but also the
   highest cancellation rate among channels with meaningful volume; a
   small reduction in TA/TO cancellations would have an outsized revenue
   impact simply because of its volume.

## 8. Limitations & Next Steps

- Data covers 2015–2017 only; booking and cancellation behavior may have
  shifted materially since (post-pandemic travel patterns, OTA policy
  changes), so the model should be retrained on more recent data before
  any production use.
- Revenue is a derived proxy (`adr * total_stay`), not an accounting
  figure — it excludes discounts, fees, and no-show penalties the hotel
  chain may actually apply.
- The "business traveler" / "vacation traveler" split is a heuristic
  built for this analysis, not a field the hotel actually collects — it's
  useful directionally, not as a precise headcount.
- Next steps: retrain periodically as new booking data arrives; consider a
  cost-sensitive threshold (rather than the default 0.5 cutoff) for the
  cancellation model, since a false negative (missed cancellation) is
  likely more costly to the business than a false positive; and validate
  the deposit-type finding against actual channel-level policy data before
  recommending a policy change.
