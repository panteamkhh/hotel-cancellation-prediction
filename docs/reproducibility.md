# Reproducibility

Every number, table and figure in this repository is produced by code in `src/`
from the committed raw CSV. There are no manual steps.

## Environment

- Python 3.10+ (CI runs 3.10, 3.11 and 3.12)
- Dependencies pinned with lower bounds in `requirements.txt`
- Determinism: `RANDOM_STATE = 42` is used for the train/test split and for
  every model, so repeated runs produce the same split and comparable metrics

## Full pipeline

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m src.run_pipeline    # clean -> features -> train -> evaluate
python -m src.run_analysis    # regenerate all 18 figures + output/ tables
python -m src.train           # fit and compare all four models
```

`src.run_pipeline` writes `data/processed/hotel_bookings_clean.csv` and
`data/processed/hotel_bookings_features.csv`; `src.run_analysis` writes
`reports/figures/*.png` and `output/*.csv`; `src.train` writes
`models/best_model.pkl` and `models/model_metrics.json`.

## Notebooks

The notebooks in `notebooks/` are the exploratory front-end to the same
functions. Run them in order (01 → 09) if you want the narrative, but nothing
in the notebooks is required to reproduce the artifacts.

## Determinism notes

- LightGBM/XGBoost training is single-threaded with respect to results because
  `random_state` is fixed; minor floating-point differences across CPU
  architectures can still move metrics in the fourth decimal.
- Plot rendering can differ slightly across matplotlib versions; the committed
  PNGs were produced with the versions resolved by `requirements.txt`.

## Verifying the current state

```bash
ruff check .
black --check .
pytest
```

These are exactly the checks CI runs on every push.
