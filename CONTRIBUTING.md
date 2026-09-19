# Contributing

## Setup

```bash
git clone https://github.com/panteamkhh/hotel-cancellation-prediction.git
cd hotel-cancellation-prediction

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements-dev.txt
pre-commit install
```

Download `hotel_bookings.csv` from Kaggle into `data/raw/` (see
`data/raw/README.md`) before running anything that touches the data.

## Workflow

- Reusable logic lives in `src/` (loading, cleaning, feature engineering,
  analysis, visualization, evaluation, training, prediction). Notebooks call
  into `src/` rather than redefining the same function twice — that's what
  keeps the notebooks and the CLIs in sync.
- Add a test in `tests/` for any new function in `src/`.
- Run the checks locally before opening a PR (or just `make check`):

```bash
ruff check .
black --check .
pytest
pytest --cov=src      # optional coverage
```

- If you change something notebooks depend on, re-run the affected
  notebooks top-to-bottom (`jupyter nbconvert --to notebook --execute
  --inplace notebooks/<name>.ipynb`) so committed outputs stay accurate.

## Commit messages

Keep them short and in the imperative mood ("Add cancellation-rate helper",
not "Added" or "Adds"). Reference the notebook or module you touched when
it's not obvious from the message alone.

## Pull requests

- One logical change per PR.
- CI (lint + format check + tests) must pass before merge.
- Update `CHANGELOG.md` under "Unreleased" for any user-facing change.
