"""Unit tests for the DataEngine class."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_engine import DataEngine

MOCK_CSV = """Date,Open,High,Low,Close,Volume,Dividends,Stock Splits,Company
2020-01-01 00:00:00-05:00,149.0,151.0,148.0,150.0,1000000,0,0,AAPL
2020-01-02 00:00:00-05:00,150.0,152.0,149.0,151.0,1100000,0,0,AAPL
2020-01-03 00:00:00-05:00,151.0,153.0,150.0,152.0,1200000,0,0,AAPL
2020-01-01 00:00:00-05:00,299.0,301.0,298.0,300.0,2000000,0,0,MSFT
2020-01-02 00:00:00-05:00,300.0,303.0,299.0,302.0,2100000,0,0,MSFT
2020-01-01 00:00:00-05:00,99.0,101.0,98.0,100.0,500000,0,0,SPY
2020-01-02 00:00:00-05:00,100.0,102.0,99.0,105.0,550000,0,0,SPY
"""


def _write_csv(tmp_path, csv_text: str = MOCK_CSV) -> str:
    """Persist StringIO-style CSV text to a temporary file for loading tests."""
    csv_file = tmp_path / "prices.csv"
    csv_file.write_text(csv_text, encoding="utf-8")
    return str(csv_file)


def _csv_dataframe(csv_text: str = MOCK_CSV) -> pd.DataFrame:
    """Parse mock CSV content from an in-memory StringIO buffer."""
    return pd.read_csv(StringIO(csv_text))


@pytest.fixture
def engine(tmp_path) -> DataEngine:
    return DataEngine(_write_csv(tmp_path))


class TestDataEngineInit:
    def test_init_requires_existing_file(self, tmp_path) -> None:
        missing = tmp_path / "missing.csv"
        with pytest.raises(ValueError, match="CSV file not found"):
            DataEngine(str(missing))

    def test_init_rejects_empty_path(self) -> None:
        with pytest.raises(ValueError, match="file_path must not be empty"):
            DataEngine("")


class TestLoadAndPivotData:
    def test_load_and_pivot_data_success(self, tmp_path) -> None:
        engine = DataEngine(_write_csv(tmp_path))
        expected = _csv_dataframe()

        result = engine.load_and_pivot_data(
            ["AAPL", "MSFT"], "2020-01-01", "2020-01-02"
        )

        assert list(result.columns) == ["AAPL", "MSFT"]
        assert len(result) == 2
        assert result.loc["2020-01-01", "AAPL"] == pytest.approx(150.0)
        assert result.loc["2020-01-02", "MSFT"] == pytest.approx(302.0)
        assert set(result.columns) <= set(expected["Company"].unique())

    def test_load_and_pivot_data_single_ticker(self, tmp_path) -> None:
        engine = DataEngine(_write_csv(tmp_path))

        result = engine.load_and_pivot_data(["SPY"], "2020-01-01", "2020-01-02")

        assert list(result.columns) == ["SPY"]
        assert len(result) == 2
        assert result.iloc[0]["SPY"] == pytest.approx(100.0)

    def test_load_and_pivot_data_prefers_adjusted_close(self, tmp_path) -> None:
        csv_with_adj = """Date,Close,Adjusted Close,Company
2020-01-01 00:00:00-05:00,100.0,101.0,AAPL
2020-01-02 00:00:00-05:00,110.0,111.0,AAPL
"""
        engine = DataEngine(_write_csv(tmp_path, csv_with_adj))

        result = engine.load_and_pivot_data(["AAPL"], "2020-01-01", "2020-01-02")

        assert result.loc["2020-01-01", "AAPL"] == pytest.approx(101.0)
        assert result.loc["2020-01-02", "AAPL"] == pytest.approx(111.0)

    @patch("src.data_engine.pd.read_csv")
    def test_load_and_pivot_data_read_error(
        self,
        mock_read_csv: patch,
        engine: DataEngine,
    ) -> None:
        mock_read_csv.side_effect = OSError("Permission denied")

        with pytest.raises(ValueError, match="Failed to read CSV file"):
            engine.load_and_pivot_data(["AAPL"], "2020-01-01", "2020-01-02")

    @patch("src.data_engine.pd.read_csv")
    def test_load_and_pivot_data_empty_csv(
        self,
        mock_read_csv: patch,
        engine: DataEngine,
    ) -> None:
        mock_read_csv.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="CSV file is empty"):
            engine.load_and_pivot_data(["AAPL"], "2020-01-01", "2020-01-02")

    def test_load_and_pivot_data_no_matching_date_range(self, tmp_path) -> None:
        engine = DataEngine(_write_csv(tmp_path))

        with pytest.raises(ValueError, match="No data for tickers"):
            engine.load_and_pivot_data(["AAPL"], "2019-01-01", "2019-12-31")

    def test_load_and_pivot_data_missing_ticker(self, tmp_path) -> None:
        engine = DataEngine(_write_csv(tmp_path))

        with pytest.raises(ValueError, match="No price data found for tickers"):
            engine.load_and_pivot_data(["AAPL", "GOOGL"], "2020-01-01", "2020-01-02")

    def test_load_and_pivot_data_empty_tickers(self, engine: DataEngine) -> None:
        with pytest.raises(ValueError, match="At least one ticker"):
            engine.load_and_pivot_data([], "2020-01-01", "2020-01-02")


class TestCalculateLogReturns:
    def test_calculate_log_returns(self, engine: DataEngine) -> None:
        prices = pd.DataFrame(
            {"AAPL": [100.0, 110.0, 121.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        result = engine.calculate_log_returns(prices)

        expected_first = np.log(110.0 / 100.0)
        expected_second = np.log(121.0 / 110.0)
        assert len(result) == 2
        assert result.iloc[0]["AAPL"] == pytest.approx(expected_first)
        assert result.iloc[1]["AAPL"] == pytest.approx(expected_second)

    def test_calculate_log_returns_drops_nan_rows(self, engine: DataEngine) -> None:
        prices = pd.DataFrame(
            {"AAPL": [100.0, 110.0, 121.0], "MSFT": [200.0, 220.0, 242.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        result = engine.calculate_log_returns(prices)

        assert len(result) == 2
        assert result.index[0] == prices.index[1]
        assert not result.isna().any().any()

    def test_calculate_log_returns_empty_input(self, engine: DataEngine) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            engine.calculate_log_returns(pd.DataFrame())
