"""Plotly chart builders for the dashboard."""

import pandas as pd
import plotly.graph_objects as go


def plot_efficient_frontier(
    volatilities: list[float],
    returns: list[float],
) -> go.Figure:
    raise NotImplementedError


def plot_portfolio_weights(weights: pd.Series) -> go.Figure:
    raise NotImplementedError
