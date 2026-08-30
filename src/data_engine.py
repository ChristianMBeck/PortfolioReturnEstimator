"""Local financial data loading and transformation from CSV files."""


from __future__ import annotations


import logging
from pathlib import Path


import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


COMPANY_COLUMN = "Company"
PRICE_COLUMNS = ("Adjusted Close", "Close")




class DataEngine:
   """Load historical prices from a local CSV and compute log returns in memory."""


   def __init__(self, file_path: str) -> None:
       """Initialize the engine with a path to a Yahoo Finance CSV dataset.


       Args:
           file_path: Absolute or relative path to the CSV file.


       Raises:
           ValueError: If ``file_path`` is empty or the file does not exist.
       """
       if not file_path:
           raise ValueError("file_path must not be empty.")


       path = Path(file_path)
       if not path.is_file():
           raise ValueError(f"CSV file not found: {file_path}")


       self.file_path = str(path)


   def load_and_pivot_data(
       self,
       tickers: list[str],
       start_date: str,
       end_date: str,
   ) -> pd.DataFrame:
       """Load, filter, and pivot local CSV price data into a wide DataFrame.


       Reads the CSV configured at construction time, keeps rows whose
       ``Company`` value is in ``tickers`` and whose ``Date`` falls within
       ``[start_date, end_date]``, then pivots so dates index rows and
       tickers index columns. Price values use ``Adjusted Close`` when
       present, otherwise ``Close``.


       Args:
           tickers: Ticker symbols to include (matched against ``Company``).
           start_date: Inclusive start date in ``YYYY-MM-DD`` format.
           end_date: Inclusive end date in ``YYYY-MM-DD`` format.


       Returns:
           DataFrame of close prices indexed by date with one column per ticker.


       Raises:
           ValueError: If ``tickers`` is empty, the file cannot be read,
               required columns are missing, or no rows match the filters.
       """
       if not tickers:
           raise ValueError("At least one ticker symbol is required.")


       try:
           raw = pd.read_csv(self.file_path)
       except Exception as exc:
           logger.exception("Failed to read CSV at %s", self.file_path)
           raise ValueError(f"Failed to read CSV file {self.file_path}: {exc}") from exc


       if raw.empty:
           raise ValueError(f"CSV file is empty: {self.file_path}")


       self._validate_columns(raw)


       filtered = raw.loc[raw[COMPANY_COLUMN].isin(tickers)].copy()
       if filtered.empty:
           raise ValueError(f"No rows found for tickers={tickers}.")


       filtered["Date"] = pd.to_datetime(
           filtered["Date"].astype(str).str.slice(0, 10)
       )
       start = pd.Timestamp(start_date).normalize()
       end = pd.Timestamp(end_date).normalize()
       filtered = filtered.loc[(filtered["Date"] >= start) & (filtered["Date"] <= end)]
       if filtered.empty:
           raise ValueError(
               f"No data for tickers={tickers} between {start_date} and {end_date}."
           )


       price_col = self._resolve_price_column(raw)
       pivoted = filtered.pivot(index="Date", columns=COMPANY_COLUMN, values=price_col)
       pivoted = pivoted.sort_index().dropna(how="all")


       missing_tickers = sorted(set(tickers) - set(pivoted.columns))
       if missing_tickers:
           raise ValueError(f"No price data found for tickers: {missing_tickers}")


       return pivoted[list(tickers)]


   def calculate_log_returns(self, price_df: pd.DataFrame) -> pd.DataFrame:
       """Convert price levels into daily log returns and drop NaN rows.


       Computes ln(P_t / P_{t-1}) for each asset column.


       Args:
           price_df: DataFrame of price levels indexed by date.


       Returns:
           DataFrame of daily log returns with NaN rows removed.


       Raises:
           ValueError: If ``price_df`` is empty.
       """
       if price_df.empty:
           raise ValueError("price_df must not be empty.")


       log_returns = np.log(price_df / price_df.shift(1))
       return log_returns.dropna()


   @staticmethod
   def _validate_columns(df: pd.DataFrame) -> None:
       """Ensure the CSV contains the columns required for pivoting."""
       missing = {COMPANY_COLUMN, "Date"} - set(df.columns)
       if missing:
           raise ValueError(f"CSV is missing required columns: {sorted(missing)}")


       if not any(col in df.columns for col in PRICE_COLUMNS):
           raise ValueError(
               f"CSV must contain one of: {', '.join(PRICE_COLUMNS)}."
           )


   @staticmethod
   def _resolve_price_column(df: pd.DataFrame) -> str:
       """Return the preferred price column name present in ``df``."""
       for column in PRICE_COLUMNS:
           if column in df.columns:
               return column
       raise ValueError(f"CSV must contain one of: {', '.join(PRICE_COLUMNS)}.")



