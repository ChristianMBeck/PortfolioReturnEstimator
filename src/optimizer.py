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
   shrinkage estimate of the covariance matrix, or a Ridge-regression
   expected-return model in place of the historical-mean estimate.
Author: Quantitative Portfolio Optimization Dashboard
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.expected_returns_model import get_ml_expected_return_estimator

# Number of trading days used to annualize daily statistics.
TRADING_DAYS_PER_YEAR: int = 252

# Default path to the historical price CSV dataset, used only by
# `optimize_max_sharpe_ml` to fetch the extra trailing history (and
# Volume data) its expected-return model needs beyond what's in
# `self.log_returns`. Matches DATA_FILE_PATH in app.py.
DEFAULT_DATA_FILE_PATH: str = "data/stock_details_5_years.csv"


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
        bounds: Tuple[Tuple[float, float], ...] | None = None,
        extra_constraints: List[Dict[str, object]] | None = None,
        failure_context: str = "",
    ) -> Dict[str, object]:
        """
        Run the constrained SLSQP optimization to maximize the Sharpe ratio.
        Shared by all `optimize_max_sharpe*`/`optimize_with_constraints`
        variants once each has produced its own estimate of
        `expected_returns` and `cov_matrix`.
        Args:
            tickers (List[str]): Ticker strings, aligned with the order of
                `expected_returns` and `cov_matrix`.
            expected_returns (pd.Series): Annualized expected returns per
                asset.
            cov_matrix (pd.DataFrame): Annualized covariance matrix.
            risk_free_rate (float): The risk-free rate used in the Sharpe
                ratio calculation.
            bounds (Tuple[Tuple[float, float], ...] | None, optional): Per-
                asset weight bounds. Defaults to `None`, which resolves to
                the standard long-only `(0.0, 1.0)` bound for every asset
                -- i.e. identical behavior to the original implementation.
            extra_constraints (List[Dict[str, object]] | None, optional):
                Additional `scipy.optimize.minimize` constraint dicts to
                layer on top of the mandatory fully-invested equality
                constraint (e.g. a minimum-return or maximum-volatility
                inequality constraint). Defaults to `None` (no extra
                constraints), matching the original implementation.
            failure_context (str, optional): Extra text appended to the
                convergence-failure error message to help identify *why*
                a constrained solve failed (e.g. which constraints were
                active). Defaults to `""`.
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
        constraints: List[Dict[str, object]] = [
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        ]
        if extra_constraints:
            constraints.extend(extra_constraints)

        # Bounds: no short-selling by default, weights strictly within
        # [0.0, 1.0], unless the caller supplied custom per-asset bounds.
        if bounds is None:
            bounds = tuple((0.0, 1.0) for _ in range(num_assets))

        optimization_result = minimize(
            fun=self._negative_sharpe_ratio,
            x0=initial_weights,
            args=(expected_returns, cov_matrix, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=tuple(constraints),
        )

        if not optimization_result.success:
            raise ValueError(
                f"Portfolio optimization failed to converge: "
                f"{optimization_result.message}{failure_context}"
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

    def optimize_with_constraints(
        self,
        tickers: List[str],
        risk_free_rate: float = 0.0,
        min_return: float | None = None,
        max_volatility: float | None = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> Dict[str, object]:
        """
        Generate a Max Sharpe Ratio portfolio subject to user-supplied
        return, volatility, and allocation constraints.
        This mirrors `optimize_max_sharpe` (same historical sample
        covariance/expected-return estimates, same fully-invested,
        long-only-by-default objective), but layers on up to three
        optional constraints:
        - A minimum acceptable annualized portfolio return.
        - A maximum acceptable annualized portfolio volatility.
        - A per-asset allocation band (the same `[min_weight, max_weight]`
          bound is applied to every asset in `tickers`), e.g. to cap
          concentration in any single position or to force a minimum
          stake in every selected asset.
        Any constraint left at its default is simply not imposed, so
        calling this with no optional arguments reproduces the same
        result as `optimize_max_sharpe`.
        Args:
            tickers (List[str]): List of ticker strings to include in the
                optimization universe. Must all be present as columns in
                the internal log returns DataFrame.
            risk_free_rate (float, optional): The risk-free rate used in
                the Sharpe ratio calculation. Defaults to 0.0.
            min_return (float | None, optional): Minimum acceptable
                annualized portfolio return (e.g. `0.10` for 10%). `None`
                (the default) imposes no return floor.
            max_volatility (float | None, optional): Maximum acceptable
                annualized portfolio volatility (e.g. `0.20` for 20%).
                `None` (the default) imposes no volatility ceiling.
            min_weight (float, optional): Minimum allocation any single
                asset may receive, as a fraction of the portfolio (e.g.
                `0.05` for 5%). Defaults to `0.0`.
            max_weight (float, optional): Maximum allocation any single
                asset may receive, as a fraction of the portfolio (e.g.
                `0.30` for 30%). Defaults to `1.0`.
        Returns:
            Dict[str, object]: A dictionary with the following keys:
                - 'weights' (Dict[str, float]): Optimal weight per ticker.
                - 'return' (float): Optimized annualized portfolio return.
                - 'volatility' (float): Optimized annualized portfolio
                  volatility.
                - 'sharpe_ratio' (float): Optimized (maximized) Sharpe ratio.
        Raises:
            ValueError: If `tickers` is empty, contains tickers not found
                in the dataset; if `min_weight`/`max_weight` are invalid
                or make the fully-invested constraint infeasible on their
                own (e.g. `min_weight` too high for the number of assets);
                or if the optimizer fails to converge given the requested
                constraints (which most often means `min_return` and
                `max_volatility` are jointly unreachable for this asset
                universe).
        """
        if not (0.0 <= min_weight <= max_weight <= 1.0):
            raise ValueError(
                "Invalid allocation bounds: require "
                f"0.0 <= min_weight <= max_weight <= 1.0 "
                f"(got min_weight={min_weight}, max_weight={max_weight})."
            )

        num_assets: int = len(tickers)

        # A per-asset floor of `min_weight` across `num_assets` assets can
        # only be met if the assets collectively allow at least 100%
        # allocation; likewise a per-asset ceiling of `max_weight` must
        # allow at least 100% in aggregate. Catch these infeasible cases
        # up front with a clear message instead of letting SLSQP fail
        # silently with a generic non-convergence error.
        if min_weight * num_assets > 1.0 + 1e-9:
            raise ValueError(
                f"Infeasible allocation constraint: a minimum of "
                f"{min_weight:.2%} per asset across {num_assets} assets "
                f"requires at least {min_weight * num_assets:.2%} total "
                "allocation, which exceeds 100%. Lower min_weight or "
                "select more tickers."
            )
        if max_weight * num_assets < 1.0 - 1e-9:
            raise ValueError(
                f"Infeasible allocation constraint: a maximum of "
                f"{max_weight:.2%} per asset across {num_assets} assets "
                f"allows at most {max_weight * num_assets:.2%} total "
                "allocation, which is less than 100%. Raise max_weight or "
                "select more tickers."
            )

        filtered_returns: pd.DataFrame = self._get_filtered_returns(tickers)

        # Same historical annualized expected-return and sample covariance
        # estimates as `optimize_max_sharpe`, so the only difference in
        # the result comes from the constraints applied below.
        expected_returns: pd.Series = filtered_returns.mean() * TRADING_DAYS_PER_YEAR
        cov_matrix: pd.DataFrame = filtered_returns.cov() * TRADING_DAYS_PER_YEAR

        extra_constraints: List[Dict[str, object]] = []
        failure_context_parts: List[str] = []

        if min_return is not None:
            # Inequality constraint form for SLSQP is `fun(weights) >= 0`,
            # so `portfolio_return - min_return >= 0`.
            extra_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights, er=expected_returns: (
                        float(np.dot(weights, er)) - min_return
                    ),
                }
            )
            failure_context_parts.append(f"min_return={min_return:.2%}")

        if max_volatility is not None:
            # `max_volatility - portfolio_volatility >= 0`.
            extra_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights, cm=cov_matrix: (
                        max_volatility
                        - float(np.sqrt(np.dot(weights.T, np.dot(cm, weights))))
                    ),
                }
            )
            failure_context_parts.append(f"max_volatility={max_volatility:.2%}")

        bounds: Tuple[Tuple[float, float], ...] = tuple(
            (min_weight, max_weight) for _ in range(num_assets)
        )
        failure_context_parts.append(
            f"allocation range=[{min_weight:.2%}, {max_weight:.2%}] per asset"
        )
        failure_context: str = (
            " Active constraints: " + ", ".join(failure_context_parts) + ". "
            "Try relaxing one or more constraints."
        )

        return self._solve_max_sharpe(
            tickers,
            expected_returns,
            cov_matrix,
            risk_free_rate,
            bounds=bounds,
            extra_constraints=extra_constraints,
            failure_context=failure_context,
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

    def optimize_max_sharpe_ml(
        self,
        tickers: List[str],
        risk_free_rate: float = 0.0,
        data_file_path: str = DEFAULT_DATA_FILE_PATH,
    ) -> Dict[str, object]:
        """
        Solve for the Max Sharpe Ratio portfolio using a trained Ridge
        regression model to estimate expected returns, in place of the
        trailing-historical-mean estimate used by `optimize_max_sharpe`.

        This mirrors `optimize_max_sharpe` exactly in every other respect
        (same sample covariance matrix, same fully-invested, long-only
        objective and constraints) -- the only difference is how
        `expected_returns` is produced.

        The model (see `train_expected_return_model.py` at the project
        root and `src/expected_returns_model.py`) predicts each asset's
        annualized return over the next ~63 trading days from trailing
        momentum, volatility, and volume-trend features. On a held-out,
        time-purged test set it achieved a 36.7% lower mean absolute error
        than the trailing-historical-mean baseline (see
        `src/models/evaluation_report.json`).

        Note:
            The model needs ~126 trailing trading days of price/volume
            history *and* access to the original CSV dataset (for Volume,
            which `self.log_returns` doesn't carry) -- see
            `data_file_path`. Any ticker with insufficient history falls
            back to the same trailing-mean formula `optimize_max_sharpe`
            uses, rather than failing the whole optimization.

        Args:
            tickers (List[str]): List of ticker strings to include in the
                optimization universe. Must all be present as columns in
                the internal log returns DataFrame.
            risk_free_rate (float, optional): The risk-free rate used in
                the Sharpe ratio calculation. Defaults to 0.0.
            data_file_path (str, optional): Path to the historical price
                CSV dataset the expected-return model reads its extra
                trailing history and Volume data from. Defaults to
                `DEFAULT_DATA_FILE_PATH` (matches `DATA_FILE_PATH` in
                app.py).
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

        estimator = get_ml_expected_return_estimator()
        expected_returns: pd.Series = estimator.estimate(data_file_path, tickers)

        # Same sample annualized covariance matrix as optimize_max_sharpe,
        # so the only difference in the result comes from expected_returns.
        cov_matrix: pd.DataFrame = filtered_returns.cov() * TRADING_DAYS_PER_YEAR

        return self._solve_max_sharpe(
            tickers, expected_returns, cov_matrix, risk_free_rate
        )