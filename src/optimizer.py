"""
optimizer.py
============

Modern Portfolio Theory (MPT) utilities for a Quantitative Portfolio
Optimization Dashboard.

This module exposes a single class, `PortfolioOptimizer`, which wraps a
DataFrame of daily log returns (index = Dates, columns = tickers) and
provides methods to:

1. Compute annualized portfolio performance metrics (return, volatility,
   Sharpe ratio) for an arbitrary weight vector.
2. Solve for the Maximum Sharpe Ratio portfolio subject to a
   fully-invested, long-only constraint using `scipy.optimize.minimize`,
   using either the raw sample covariance matrix or a Ledoit-Wolf
   shrinkage estimate of the covariance matrix.

Author: Quantitative Portfolio Optimization Dashboard
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# Number of trading days used to annualize daily statistics.
TRADING_DAYS_PER_YEAR: int = 252


class PortfolioOptimizer:
    """
    Encapsulates Modern Portfolio Theory (MPT) calculations and
    Max Sharpe Ratio optimization for a universe of assets.

    The class is initialized with a DataFrame of daily log returns and
    exposes methods to evaluate portfolio performance and to solve for
    the weight allocation that maximizes the Sharpe ratio, subject to
    a fully-invested, long-only constraint.

    Attributes:
        log_returns (pd.DataFrame): Daily log returns indexed by date,
            with one column per ticker.
    """

    def __init__(self, log_returns: pd.DataFrame) -> None:
        """
        Initialize the PortfolioOptimizer with a daily log returns dataset.

        Args:
            log_returns (pd.DataFrame): A pivoted DataFrame where the
                index represents Dates and the columns are individual
                stock ticker strings (e.g., 'AAPL', 'MSFT', 'GOOGL').
                Values are daily log returns.

        Raises:
            TypeError: If `log_returns` is not a pandas DataFrame.
            ValueError: If `log_returns` is empty.
        """
        if not isinstance(log_returns, pd.DataFrame):
            raise TypeError("log_returns must be a pandas DataFrame.")
        if log_returns.empty:
            raise ValueError("log_returns DataFrame cannot be empty.")

        self.log_returns: pd.DataFrame = log_returns

    def calculate_portfolio_performance(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Calculate annualized portfolio return, volatility, and Sharpe ratio.

        Args:
            weights (np.ndarray): Array of portfolio weights, one per asset,
                assumed to be aligned with the order of `expected_returns`
                and `cov_matrix`.
            expected_returns (pd.Series): Annualized expected returns per
                asset (already annualized, e.g., mean daily return * 252).
            cov_matrix (pd.DataFrame): Annualized covariance matrix of
                asset returns (already annualized, e.g., daily covariance
                * 252).
            risk_free_rate (float, optional): The risk-free rate used in
                the Sharpe ratio calculation. Defaults to 0.0.

        Returns:
            Tuple[float, float, float]: A tuple of
                (annualized_portfolio_return, annualized_portfolio_volatility,
                sharpe_ratio).
        """
        weights = np.asarray(weights, dtype=float)

        annualized_portfolio_return: float = float(np.dot(weights, expected_returns))

        annualized_portfolio_variance: float = float(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        )
        annualized_portfolio_volatility: float = float(
            np.sqrt(annualized_portfolio_variance)
        )

        if annualized_portfolio_volatility == 0.0:
            sharpe_ratio: float = 0.0
        else:
            sharpe_ratio = (
                annualized_portfolio_return - risk_free_rate
            ) / annualized_portfolio_volatility


        print("--- QUANT DIAGNOSTIC CHECK ---")
        print("Log Returns Sample:\n", expected_returns.head(2))
        print("Expected Returns (Annualized):\n", expected_returns.head(5))
        print("Covariance Matrix Sample:\n", cov_matrix.iloc[:3, :3])
        print("------------------------------")

        return (
            annualized_portfolio_return,
            annualized_portfolio_volatility,
            sharpe_ratio,
        )

    def _negative_sharpe_ratio(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float,
    ) -> float:
        """
        Objective function for the optimizer: the negative Sharpe ratio.

        `scipy.optimize.minimize` only minimizes, so to find the weights
        that *maximize* the Sharpe ratio, we minimize its negation.

        Args:
            weights (np.ndarray): Candidate portfolio weights.
            expected_returns (pd.Series): Annualized expected returns per asset.
            cov_matrix (pd.DataFrame): Annualized covariance matrix.
            risk_free_rate (float): Risk-free rate used in the Sharpe
                ratio calculation.

        Returns:
            float: The negative Sharpe ratio for the given weights.
        """
        _, _, sharpe_ratio = self.calculate_portfolio_performance(
            weights, expected_returns, cov_matrix, risk_free_rate
        )
        return -sharpe_ratio

    def _get_filtered_returns(self, tickers: List[str]) -> pd.DataFrame:
        """
        Validate the requested tickers and filter the internal dataset.

        Args:
            tickers (List[str]): List of ticker strings to include in the
                optimization universe.

        Returns:
            pd.DataFrame: The internal log returns filtered down to the
                requested tickers, with any rows containing NaNs dropped.

        Raises:
            ValueError: If `tickers` is empty or contains tickers not
                found in the dataset.
        """
        if not tickers:
            raise ValueError("tickers list cannot be empty.")

        missing_tickers: List[str] = [
            ticker for ticker in tickers if ticker not in self.log_returns.columns
        ]
        if missing_tickers:
            raise ValueError(
                f"The following tickers were not found in the dataset: "
                f"{missing_tickers}"
            )

        return self.log_returns[tickers].dropna()

    def _solve_max_sharpe(
        self,
        tickers: List[str],
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float,
    ) -> Dict[str, object]:
        """
        Run the constrained SLSQP optimization to maximize the Sharpe ratio.

        Shared by all `optimize_max_sharpe*` variants once each has
        produced its own estimate of `expected_returns` and `cov_matrix`.

        Args:
            tickers (List[str]): Ticker strings, aligned with the order of
                `expected_returns` and `cov_matrix`.
            expected_returns (pd.Series): Annualized expected returns per
                asset.
            cov_matrix (pd.DataFrame): Annualized covariance matrix.
            risk_free_rate (float): The risk-free rate used in the Sharpe
                ratio calculation.

        Returns:
            Dict[str, object]: A dictionary with the following keys:
                - 'weights' (Dict[str, float]): Optimal weight per ticker.
                - 'return' (float): Optimized annualized portfolio return.
                - 'volatility' (float): Optimized annualized portfolio
                  volatility.
                - 'sharpe_ratio' (float): Optimized (maximized) Sharpe ratio.

        Raises:
            ValueError: If the optimizer fails to converge.
        """
        num_assets: int = len(tickers)
        initial_weights: np.ndarray = np.repeat(1.0 / num_assets, num_assets)

        # Constraint: sum of weights must equal 1.0 (fully invested).
        constraints: Tuple[Dict[str, object], ...] = (
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        )

        # Bounds: no short-selling, weights strictly within [0.0, 1.0].
        bounds: Tuple[Tuple[float, float], ...] = tuple(
            (0.0, 1.0) for _ in range(num_assets)
        )

        optimization_result = minimize(
            fun=self._negative_sharpe_ratio,
            x0=initial_weights,
            args=(expected_returns, cov_matrix, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not optimization_result.success:
            raise ValueError(
                f"Portfolio optimization failed to converge: "
                f"{optimization_result.message}"
            )

        optimal_weights: np.ndarray = optimization_result.x

        (
            optimized_return,
            optimized_volatility,
            optimized_sharpe_ratio,
        ) = self.calculate_portfolio_performance(
            optimal_weights, expected_returns, cov_matrix, risk_free_rate
        )

        weights_dict: Dict[str, float] = {
            ticker: float(weight) for ticker, weight in zip(tickers, optimal_weights)
        }

        return {
            "weights": weights_dict,
            "return": optimized_return,
            "volatility": optimized_volatility,
            "sharpe_ratio": optimized_sharpe_ratio,
        }

    def optimize_max_sharpe(
        self, tickers: List[str], risk_free_rate: float = 0.0
    ) -> Dict[str, object]:
        """
        Solve for the portfolio weights that maximize the Sharpe ratio.

        Filters the internal log returns dataset down to the requested
        tickers, computes historical annualized expected returns and the
        sample annualized covariance matrix, and runs a constrained SLSQP
        optimization (fully invested, long-only) to find the
        Max Sharpe Ratio portfolio.

        Note:
            The sample covariance matrix used here is a standard historical
            estimate. For a large number of assets relative to the number
            of observations, this estimate can be noisy/ill-conditioned;
            see `optimize_max_sharpe_shrinkage` for a more robust
            Ledoit-Wolf shrinkage alternative.

        Args:
            tickers (List[str]): List of ticker strings to include in the
                optimization universe. Must all be present as columns in
                the internal log returns DataFrame.
            risk_free_rate (float, optional): The risk-free rate used in
                the Sharpe ratio calculation. Defaults to 0.0.

        Returns:
            Dict[str, object]: A dictionary with the following keys:
                - 'weights' (Dict[str, float]): Optimal weight per ticker.
                - 'return' (float): Optimized annualized portfolio return.
                - 'volatility' (float): Optimized annualized portfolio
                  volatility.
                - 'sharpe_ratio' (float): Optimized (maximized) Sharpe ratio.

        Raises:
            ValueError: If `tickers` is empty, contains tickers not found
                in the dataset, or if the optimizer fails to converge.
        """
        filtered_returns: pd.DataFrame = self._get_filtered_returns(tickers)

        # Historical expected annualized returns (mean daily return * 252).
        expected_returns: pd.Series = filtered_returns.mean() * TRADING_DAYS_PER_YEAR

        # Sample annualized covariance matrix (daily covariance * 252).
        cov_matrix: pd.DataFrame = filtered_returns.cov() * TRADING_DAYS_PER_YEAR

        return self._solve_max_sharpe(
            tickers, expected_returns, cov_matrix, risk_free_rate
        )

    def optimize_max_sharpe_shrinkage(
        self, tickers: List[str], risk_free_rate: float = 0.0
    ) -> Dict[str, object]:
        """
        Solve for the Max Sharpe Ratio portfolio using a Ledoit-Wolf
        shrinkage estimate of the covariance matrix.

        This mirrors `optimize_max_sharpe` exactly (same expected-returns
        estimate, same constraints, bounds, and objective), but replaces
        the raw sample covariance matrix with a Ledoit-Wolf shrinkage
        estimator (`sklearn.covariance.LedoitWolf`). Shrinkage pulls the
        sample covariance matrix toward a structured, better-conditioned
        target, which tends to produce more stable, less extreme weight
        allocations than the raw sample covariance -- particularly useful
        when the number of assets is large relative to the number of
        historical observations.

        Args:
            tickers (List[str]): List of ticker strings to include in the
                optimization universe. Must all be present as columns in
                the internal log returns DataFrame.
            risk_free_rate (float, optional): The risk-free rate used in
                the Sharpe ratio calculation. Defaults to 0.0.

        Returns:
            Dict[str, object]: A dictionary with the following keys:
                - 'weights' (Dict[str, float]): Optimal weight per ticker.
                - 'return' (float): Optimized annualized portfolio return.
                - 'volatility' (float): Optimized annualized portfolio
                  volatility.
                - 'sharpe_ratio' (float): Optimized (maximized) Sharpe ratio.

        Raises:
            ValueError: If `tickers` is empty, contains tickers not found
                in the dataset, or if the optimizer fails to converge.
        """
        filtered_returns: pd.DataFrame = self._get_filtered_returns(tickers)

        # Historical expected annualized returns (mean daily return * 252).
        # Kept identical to optimize_max_sharpe so the two methods are
        # comparable on a like-for-like basis, differing only in how the
        # covariance matrix is estimated.
        expected_returns: pd.Series = filtered_returns.mean() * TRADING_DAYS_PER_YEAR

        # Ledoit-Wolf shrinkage estimate of the daily covariance matrix,
        # then annualized (daily covariance * 252).
        ledoit_wolf_estimator: LedoitWolf = LedoitWolf().fit(filtered_returns.values)
        shrunk_cov_values: np.ndarray = (
            ledoit_wolf_estimator.covariance_ * TRADING_DAYS_PER_YEAR
        )
        cov_matrix: pd.DataFrame = pd.DataFrame(
            shrunk_cov_values, index=tickers, columns=tickers
        )

        return self._solve_max_sharpe(
            tickers, expected_returns, cov_matrix, risk_free_rate
        )