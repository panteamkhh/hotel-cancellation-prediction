.PHONY: help install install-dev pipeline analysis train test lint format check clean

help:
	@echo "Common tasks:"
	@echo "  make install      install runtime dependencies"
	@echo "  make install-dev  install dev dependencies + git hooks"
	@echo "  make pipeline     clean -> features -> train LightGBM -> evaluate"
	@echo "  make analysis     regenerate all report figures and output/ tables"
	@echo "  make train        fit and compare all four models"
	@echo "  make test         run the pytest suite"
	@echo "  make lint         run ruff"
	@echo "  make format       run black"
	@echo "  make check        lint + format check + tests (what CI runs)"
	@echo "  make clean        remove caches"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

pipeline:
	python -m src.run_pipeline

analysis:
	python -m src.run_analysis

train:
	python -m src.train

test:
	pytest

lint:
	ruff check .

format:
	black .

check:
	ruff check .
	black --check .
	pytest

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ output/*.csv
