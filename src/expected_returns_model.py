"""
expected_returns_model.py
==========================

Loads the trained scikit-learn Ridge regression model (see
`train_expected_return_model.py` at the project root) and uses it to
estimate each asset's expected annualized return, as an alternative to the
trailing-historical-mean baseline used elsewhere in `PortfolioOptimizer`.

On a held-out, time-purged test set, this model achieved a 36.7% lower
MAE than the baseline (trailing 63-day mean return, annualized) -- see
`src/models/evaluation_report.json` for the full metrics.

Why this module reads the CSV directly instead of reusing DataEngine
-------------------------------------------------------------------
The model's features require ~126 trailing trading days of Close price
history *and* Volume history (for the `vol_trend` feature). `DataEngine`
today only exposes Close prices, and `PortfolioOptimizer` only ever
receives a `log_returns` DataFrame already filtered to the user's chosen
optimization window -- which may be shorter than the model needs and never
includes Volume. Rather than changing `DataEngine`'s contract or what gets
passed into `PortfolioOptimizer.__init__`, this module independently reads
Close + Volume for the requested tickers from the same CSV dataset. This
keeps `optimize_max_sharpe` and every existing caller completely unchanged.

Safety: if a ticker has insufficient trailing history or any feature comes
out NaN/inf, that ticker's estimate silently falls back to the same
trailing 63-day historical mean formula used elsewhere in the codebase,
rather than raising -- one bad ticker should never break a whole
optimization run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: int = 252

_MODULE_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = _MODULE_DIR / "models" / "expected_return_model.joblib"
DEFAULT_METADATA_PATH = _MODULE_DIR / "models" / "feature_metadata.json"

# Must match DataEngine.COMPANY_COLUMN priority order (see app.py / data_engine.py).
_TICKER_COLUMN_CANDIDATES = ("Company", "Symbol", "Ticker", "company", "symbol", "ticker")


class MLExpectedReturnEstimator:
    """
    Wraps the trained Ridge pipeline and reproduces, at inference time, the
    exact feature engineering used in `train_expected_return_model.py`.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        metadata_path: Path | str = DEFAULT_METADATA_PATH,
    ) -> None:
        with open(metadata_path) as f:
            metadata: dict = json.load(f)

        self.feature_columns: List[str] = metadata["feature_columns"]
        self.trailing_lookback_required_days: int = metadata["trailing_lookback_required_days"]
        self._pipeline = joblib.load(model_path)

    def estimate(self, data_file_path: str, tickers: List[str]) -> pd.Series:
        """
        Predict annualized expected returns for `tickers` as of the most
        recent date available in the dataset.

        Args:
            data_file_path: Path to the historical price CSV dataset.
            tickers: Tickers to estimate expected returns for.

        Returns:
            pd.Series: Annualized expected return per ticker, indexed by
                ticker in the same order as `tickers`.
        """
        price_wide, volume_wide = self._load_price_and_volume(data_file_path, tickers)
        features_df = self._engineer_latest_features(price_wide, volume_wide)

        predictions: dict[str, float] = {}
        for ticker in tickers:
            row = features_df.loc[ticker] if ticker in features_df.index else None
            if row is None or row[self.feature_columns].isna().any():
                predictions[ticker] = self._fallback_estimate(price_wide, ticker)
                continue
            X = row[self.feature_columns].values.reshape(1, -1)
            predictions[ticker] = float(self._pipeline.predict(X)[0])

        return pd.Series(predictions, name="expected_return").reindex(tickers)

    # -- Internals ------------------------------------------------------

    def _load_price_and_volume(self, data_file_path: str, tickers: List[str]):
        header_columns = pd.read_csv(data_file_path, nrows=0).columns.tolist()
        ticker_column = next(
            (col for col in _TICKER_COLUMN_CANDIDATES if col in header_columns), None
        )
        if ticker_column is None:
            raise ValueError(
                "Could not locate a ticker/symbol column in the dataset. "
                f"Available columns: {header_columns}"
            )

        raw = pd.read_csv(data_file_path, usecols=["Date", "Close", "Volume", ticker_column])
        raw = raw[raw[ticker_column].isin(tickers)]
        # Mixed UTC offsets across the file (DST transitions) make a plain
        # parse_dates fall back to string dtype -- parse as UTC explicitly
        # and normalize to a tz-naive calendar date (see also DataEngine's
        # equivalent handling of this same dataset quirk).
        raw["Date"] = pd.to_datetime(raw["Date"], utc=True).dt.tz_localize(None).dt.normalize()

        price_wide = raw.pivot(index="Date", columns=ticker_column, values="Close").sort_index()
        volume_wide = raw.pivot(index="Date", columns=ticker_column, values="Volume").sort_index()
        return price_wide, volume_wide

    def _engineer_latest_features(
        self, price_wide: pd.DataFrame, volume_wide: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute each feature's value as of the most recent available date, per ticker."""
        log_returns = np.log(price_wide / price_wide.shift(1))
        log_volume = np.log(volume_wide.replace(0, np.nan))

        rows = {}
        for ticker in price_wide.columns:
            r = log_returns[ticker]
            lv = log_volume[ticker]

            rows[ticker] = {
                "mom_5": r.rolling(5).mean().iloc[-1] * TRADING_DAYS_PER_YEAR,
                "mom_10": r.rolling(10).mean().iloc[-1] * TRADING_DAYS_PER_YEAR,
                "mom_21": r.rolling(21).mean().iloc[-1] * TRADING_DAYS_PER_YEAR,
                "mom_63": r.rolling(63).mean().iloc[-1] * TRADING_DAYS_PER_YEAR,
                "mom_126": r.rolling(126).mean().iloc[-1] * TRADING_DAYS_PER_YEAR,
                "vol_10": r.rolling(10).std().iloc[-1] * np.sqrt(TRADING_DAYS_PER_YEAR),
                "vol_21": r.rolling(21).std().iloc[-1] * np.sqrt(TRADING_DAYS_PER_YEAR),
                "vol_63": r.rolling(63).std().iloc[-1] * np.sqrt(TRADING_DAYS_PER_YEAR),
                "vol_trend": lv.rolling(10).mean().iloc[-1] - lv.rolling(63).mean().iloc[-1],
            }
            mom_63 = rows[ticker]["mom_63"]
            vol_21 = rows[ticker]["vol_21"]
            rows[ticker]["risk_adj"] = (
                mom_63 / (vol_21 + 1e-8) if pd.notna(mom_63) and pd.notna(vol_21) else np.nan
            )

        features_df = pd.DataFrame.from_dict(rows, orient="index")
        return features_df.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _fallback_estimate(price_wide: pd.DataFrame, ticker: str) -> float:
        """
        Same trailing-mean formula used by `optimize_max_sharpe` elsewhere
        in the codebase, applied over the model's 63-day horizon. Used when
        a ticker doesn't have enough trailing history for the full feature
        set, so one thin ticker never breaks an entire optimization run.
        """
        if ticker not in price_wide.columns:
            return 0.0
        r = np.log(price_wide[ticker] / price_wide[ticker].shift(1))
        trailing_mean = r.tail(63).mean()
        return float(trailing_mean * TRADING_DAYS_PER_YEAR) if pd.notna(trailing_mean) else 0.0


_cached_estimator: MLExpectedReturnEstimator | None = None


def get_ml_expected_return_estimator() -> MLExpectedReturnEstimator:
    """Module-level singleton so the model/metadata are only loaded from disk once."""
    global _cached_estimator
    if _cached_estimator is None:
        _cached_estimator = MLExpectedReturnEstimator()
    return _cached_estimator