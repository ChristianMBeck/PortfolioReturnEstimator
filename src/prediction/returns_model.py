"""Predictive returns estimation using scikit-learn."""

import pandas as pd


def train_returns_model(features: pd.DataFrame, targets: pd.Series) -> object:
    raise NotImplementedError


def predict_returns(model: object, features: pd.DataFrame) -> pd.Series:
    raise NotImplementedError
