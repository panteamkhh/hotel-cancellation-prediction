# FAQ

**Why is the cancellation rate 27.5% here but ~37% elsewhere?**
Most published figures are computed on the raw file, which contains 31,994
exact duplicate rows. Those duplicates are concentrated in non-canceled
bookings, so keeping them understates the cancellation share. After
de-duplication the rate is 27.5%.

**Why is `revenue` not in the original dataset?**
It is not. The dataset has no revenue column, so this project uses the standard
derived proxy `revenue = adr * total_stay`. It excludes fees, discounts and
no-show penalties.

**Why drop `reservation_status` and `reservation_status_date`?**
Both are only known *after* the cancellation decision. Leaving them in would
leak the target and produce a model that looks perfect offline and is useless
at booking time.

**Why does a `Non Refund` deposit cancel 94.7% of the time?**
Because in this dataset it is effectively a marker for a specific high-risk
booking channel rather than a deterrent. The deposit does not appear to change
guest behavior — see the caveat in `reports/final_report.md`.

**Which model should I use?**
LightGBM: it has the best ROC-AUC and F1, and the best recall at a small cost in
precision versus Random Forest. See `docs/model_card.md`.

**Why is the split random and not time-based?**
A random split is the common baseline and lets the notebook compare models
quickly. It is optimistic for a real deployment, which is called out as a
limitation in the model card; a time-ordered split is on the roadmap.

**Can I run this without Jupyter?**
Yes. `python -m src.run_pipeline` cleans, engineers features and trains the
model; `python -m src.run_analysis` regenerates every figure and table. The
notebooks are for exploring, not for reproducing.

**Do I need to download the Kaggle data?**
No — the raw CSV is committed. See `data/raw/README.md` if you want to refresh it.
