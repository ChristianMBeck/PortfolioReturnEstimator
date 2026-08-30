"""Download raw historical price data from an external source."""

import pandas as pd


def fetch_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices indexed by date."""
    raise NotImplementedError
