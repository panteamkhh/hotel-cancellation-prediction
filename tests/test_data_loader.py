import pandas as pd
import pytest

from src.config import RAW_DATA_PATH
from src.data_loader import (
    DataValidationError,
    load_csv,
    load_raw_data,
    validate_schema,
)


def test_validate_schema_passes_when_all_columns_present():
    df = pd.DataFrame(columns=["a", "b"])
    validate_schema(df, required=["a", "b"])


def test_validate_schema_raises_on_missing_columns():
    df = pd.DataFrame(columns=["a"])
    with pytest.raises(DataValidationError, match="b"):
        validate_schema(df, required=["a", "b"], name="test")


def test_load_csv_raises_clear_error_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_csv(tmp_path / "does_not_exist.csv")


def test_load_csv_validates_required_columns(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)

    result = load_csv(path, required=["a", "b"])
    assert list(result.columns) == ["a", "b"]

    with pytest.raises(DataValidationError):
        load_csv(path, required=["a", "c"])


@pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="raw CSV not downloaded")
def test_load_raw_data_reads_real_file():
    df = load_raw_data()
    assert "is_canceled" in df.columns
    assert len(df) > 0
