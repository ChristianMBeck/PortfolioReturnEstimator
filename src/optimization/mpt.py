"""Modern Portfolio Theory optimization."""

import numpy as np
import pandas as pd


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Return optimal asset weights that maximize Sharpe ratio."""
    raise NotImplementedError


def efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (volatilities, returns) arrays along the efficient frontier."""
    raise NotImplementedError
