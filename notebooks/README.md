# Notebooks

Run them in order — each one reads what the previous stage wrote into
`data/processed/`, `reports/figures/` or `models/`. Every notebook is fully
executed with its real outputs committed, so you can read them on GitHub
without running anything.

| # | Notebook | Question it answers | Writes |
|---|---|---|---|
| 01 | `01_data_cleaning.ipynb` | How dirty is the data and what do we remove? | `data/processed/hotel_bookings_clean.csv` |
| 02 | `02_eda.ipynb` | How is the business composed and how does it move over time? | figures 01–05 |
| 03 | `03_cancellation_analysis.ipynb` | Who cancels, and what predicts it? | figures 06, 10–14 |
| 04 | `04_revenue_analysis.ipynb` | Where does revenue come from? | figures 07, 15–17 |
| 05 | `05_customer_segmentation.ipynb` | How do customer types and travel styles differ? | profile tables |
| 06 | `06_booking_behavior.ipynb` | How does lead time / stay length relate to cancellation? | figures 08–09 |
| 07 | `07_channel_geo_analysis.ipynb` | Which channels and countries matter? | figure 18 |
| 08 | `08_feature_engineering.ipynb` | What features go into the model? | `data/processed/hotel_bookings_features.csv` |
| 09 | `09_modeling.ipynb` | How well can we predict cancellation? | `models/best_model.pkl` |

## Running

From the repository root:

```bash
jupyter notebook notebooks/01_data_cleaning.ipynb
```

The notebooks add the project root to `sys.path` themselves, so they work
regardless of the directory Jupyter is launched from — but `data/raw/hotel_bookings.csv`
must exist first (see `data/raw/README.md`).

## Style

Analysis logic lives in `src/`, not in the notebooks. A notebook should only
load data, call a function, and display or plot the result — that is what keeps
them reproducible and testable.
