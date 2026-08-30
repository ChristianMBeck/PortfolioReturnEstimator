"""
app.py
======

Main frontend application for the Quantitative Portfolio Optimization
Dashboard.

This Streamlit application connects directly to the real backend modules:

- `src.data_engine.DataEngine`: loads and pivots raw historical Close
  prices from the local CSV dataset, then converts them to daily log
  returns via `calculate_log_returns`.
- `src.optimizer.PortfolioOptimizer`: runs Max Sharpe Ratio optimization
  over a user-selected basket of tickers. It expects daily log returns
  as input, not raw price levels.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
from typing import List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_engine import DataEngine
from src.optimizer import PortfolioOptimizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_FILE_PATH: str = "data/stock_details_5_years.csv"
DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL"]
DEFAULT_LOOKBACK_YEARS: int = 5

# Weights at or below this threshold are treated as an effective zero
# allocation. SLSQP can leave tiny floating-point residuals (e.g. 1e-10)
# on bound-constrained assets instead of an exact 0.0, so a strict
# `weight > 0.0` check is not sufficient to keep the chart/legend clean.
ZERO_WEIGHT_TOLERANCE: float = 1e-4


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_data_engine(file_path: str) -> DataEngine:
    """
    Instantiate (and cache) the DataEngine for the given CSV file path.

    Cached as a resource so the underlying engine/file handle is created
    exactly once per Streamlit session, rather than on every rerun.

    Args:
        file_path (str): Path to the local historical price CSV dataset.

    Returns:
        DataEngine: The initialized data engine instance.
    """
    return DataEngine(file_path)


@st.cache_data(show_spinner="Scanning dataset for available tickers...")
def get_available_tickers(file_path: str) -> List[str]:
    """
    Scan the full dataset once and return the sorted list of unique tickers.

    Reads only the ticker/symbol column from the CSV so that the whole
    dashboard has a single, cached source of truth for the ticker universe
    instead of re-reading the file from disk on every interaction.

    Args:
        file_path (str): Path to the local historical price CSV dataset.

    Returns:
        List[str]: Sorted list of unique ticker symbols found in the file.
    """
    # The raw CSV is long-format (one row per Date/Company/Close, etc.).
    # `DataEngine` itself filters on a column literally named "Company"
    # (see `DataEngine.COMPANY_COLUMN`), so that must take priority here —
    # the ticker values shown in the dropdown must come from the same
    # column `DataEngine.load_and_pivot_data` filters against, or
    # selections made in the UI will never match any rows.
    candidate_columns: Tuple[str, ...] = (
        "Company",
        "Symbol",
        "Ticker",
        "company",
        "symbol",
        "ticker",
    )

    header_columns: List[str] = pd.read_csv(file_path, nrows=0).columns.tolist()
    ticker_column = next(
        (col for col in candidate_columns if col in header_columns), None
    )

    if ticker_column is None:
        raise ValueError(
            "Could not locate a ticker/symbol column in the dataset. "
            f"Available columns: {header_columns}"
        )

    unique_tickers: List[str] = (
        pd.read_csv(file_path, usecols=[ticker_column])[ticker_column]
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(unique_tickers)


@st.cache_data(show_spinner="Loading and pivoting price history...")
def load_price_data(
    file_path: str, tickers: List[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Load and pivot historical Close prices for the requested tickers.

    Args:
        file_path (str): Path to the local historical price CSV dataset.
        tickers (List[str]): Ticker symbols to load.
        start_date (str): Start of the optimization window (YYYY-MM-DD).
        end_date (str): End of the optimization window (YYYY-MM-DD).

    Returns:
        pd.DataFrame: Wide-format DataFrame of Close prices, indexed by
            Date with one column per ticker.
    """
    engine: DataEngine = get_data_engine(file_path)
    return engine.load_and_pivot_data(tickers, start_date, end_date)


@st.cache_data(show_spinner="Computing daily log returns...")
def get_log_returns(file_path: str, prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a wide price-level DataFrame into daily log returns.

    `PortfolioOptimizer` expects daily log returns (it internally does
    `mean() * 252` and `cov() * 252`), not raw price levels, so this
    conversion step is mandatory before optimization -- feeding it raw
    prices silently produces wildly inflated "returns" and "volatility"
    since it would be annualizing price levels rather than returns.

    Args:
        file_path (str): Path to the local historical price CSV dataset,
            used only to obtain the cached `DataEngine` instance.
        prices_df (pd.DataFrame): Wide-format DataFrame of Close prices,
            as returned by `load_price_data`.

    Returns:
        pd.DataFrame: Daily log returns, indexed by date, with the
            leading NaN row (from the first difference) dropped.
    """
    engine: DataEngine = get_data_engine(file_path)
    return engine.calculate_log_returns(prices_df)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def render_metrics_row(
    annualized_return: float, annualized_volatility: float, sharpe_ratio: float
) -> None:
    """
    Render the top metrics row: return, volatility, and Sharpe ratio.

    Args:
        annualized_return (float): Optimized annualized portfolio return.
        annualized_volatility (float): Optimized annualized portfolio
            volatility.
        sharpe_ratio (float): Optimized Sharpe ratio.
    """
    col_return, col_volatility, col_sharpe = st.columns(3)

    with col_return:
        st.metric("Annualized Return", f"{annualized_return * 100:.2f}%")
    with col_volatility:
        st.metric("Annualized Volatility", f"{annualized_volatility * 100:.2f}%")
    with col_sharpe:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")


def build_allocation_dataframe(weights: dict[str, float]) -> pd.DataFrame:
    """
    Build a clean, sorted allocation table from a weights dictionary.

    Zero-weight tickers are dropped and the remaining allocations are
    sorted from highest to lowest weight.

    Args:
        weights (dict[str, float]): Mapping of ticker to portfolio weight.

    Returns:
        pd.DataFrame: Columns ['Ticker', 'Allocation'] where 'Allocation'
            is a formatted percentage string, sorted descending by weight.
    """
    allocation_df: pd.DataFrame = (
        pd.Series(weights, name="Weight")
        .rename_axis("Ticker")
        .reset_index()
    )
    allocation_df = allocation_df[allocation_df["Weight"] > ZERO_WEIGHT_TOLERANCE]
    allocation_df = allocation_df.sort_values("Weight", ascending=False)
    allocation_df["Allocation"] = allocation_df["Weight"].apply(
        lambda w: f"{w * 100:.2f}%"
    )
    return allocation_df[["Ticker", "Allocation"]].reset_index(drop=True)


def render_allocation_visuals(weights: dict[str, float]) -> None:
    """
    Render the allocation pie chart and allocation table side by side.

    Args:
        weights (dict[str, float]): Mapping of ticker to portfolio weight,
            as returned by `PortfolioOptimizer.optimize_max_sharpe`.
    """
    # Filter out any zero-weight tickers so the pie chart and legend
    # only show assets that actually received an allocation.
    non_zero_weights: dict[str, float] = {
        ticker: weight
        for ticker, weight in weights.items()
        if weight > ZERO_WEIGHT_TOLERANCE
    }

    col_chart, col_table = st.columns(2)

    with col_chart:
        st.subheader("Optimal Allocation")
        pie_chart = px.pie(
            names=list(non_zero_weights.keys()),
            values=list(non_zero_weights.values()),
            hole=0.35,
        )
        pie_chart.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(pie_chart, use_container_width=True)

    with col_table:
        st.subheader("Allocation Breakdown")
        allocation_df: pd.DataFrame = build_allocation_dataframe(weights)
        st.dataframe(allocation_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the full Quantitative Portfolio Optimization Dashboard."""
    st.set_page_config(
        page_title="Quantitative Portfolio Optimization Dashboard",
        layout="wide",
    )
    st.title("📊 Quantitative Portfolio Optimization Dashboard")
    st.caption("Modern Portfolio Theory — Max Sharpe Ratio Optimization")

    # -- Sidebar configuration -------------------------------------------------
    with st.sidebar:
        st.header("Configuration")

        available_tickers: List[str] = get_available_tickers(DATA_FILE_PATH)
        default_selection: List[str] = [
            ticker for ticker in DEFAULT_TICKERS if ticker in available_tickers
        ]

        selected_tickers: List[str] = st.multiselect(
            "Select assets",
            options=available_tickers,
            default=default_selection,
        )

        today: dt.date = dt.date.today()
        default_start: dt.date = today.replace(
            year=today.year - DEFAULT_LOOKBACK_YEARS
        )
        start_date, end_date = st.date_input(
            "Optimization timeframe",
            value=(default_start, today),
            min_value=today.replace(year=today.year - 20),
            max_value=today,
        )

        risk_free_rate: float = st.slider(
            "Risk-Free Rate",
            min_value=0.0,
            max_value=0.10,
            value=0.0,
            step=0.0025,
            format="%.4f",
        )

    # -- Defensive check: require at least 2 assets -----------------------
    if len(selected_tickers) < 2:
        st.warning("Please select at least 2 assets to optimize a portfolio.")
        st.stop()

    # -- Load data and run optimization ------------------------------------
    prices_df: pd.DataFrame = load_price_data(
        DATA_FILE_PATH,
        selected_tickers,
        start_date.isoformat(),
        end_date.isoformat(),
    )

    if prices_df.empty:
        st.warning(
            "No price data found for the selected tickers and date range. "
            "Try widening the timeframe or choosing different assets."
        )
        st.stop()

    # Convert price levels to daily log returns -- PortfolioOptimizer
    # expects returns, not raw prices (see get_log_returns docstring).
    log_returns_df: pd.DataFrame = get_log_returns(DATA_FILE_PATH, prices_df)

    if log_returns_df.empty:
        st.warning(
            "Not enough price history in the selected date range to "
            "compute returns. Try widening the timeframe."
        )
        st.stop()

    try:
        optimizer = PortfolioOptimizer(log_returns_df)
        result: dict = optimizer.optimize_max_sharpe(
            tickers=selected_tickers, risk_free_rate=risk_free_rate
        )
    except ValueError as error:
        st.error(f"Portfolio optimization failed: {error}")
        st.stop()

    # -- Row 1: Headline metrics --------------------------------------------
    render_metrics_row(
        annualized_return=result["return"],
        annualized_volatility=result["volatility"],
        sharpe_ratio=result["sharpe_ratio"],
    )

    st.divider()

    # -- Row 2: Allocation pie chart + table --------------------------------
    render_allocation_visuals(result["weights"])


if __name__ == "__main__":
    main()