"""Persist and load financial data via SQLite."""

import pandas as pd


def save_prices(db_path: str, prices: pd.DataFrame) -> None:
    raise NotImplementedError


def load_prices(db_path: str, tickers: list[str] | None = None) -> pd.DataFrame:
    raise NotImplementedError
