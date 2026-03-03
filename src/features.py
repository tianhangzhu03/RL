"""Feature engineering functions for the trading task."""

from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = ["rolling_mean_return", "rolling_vol", "momentum"]


def build_features(
    data: pd.DataFrame,
    mean_window: int = 10,
    vol_window: int = 20,
    momentum_window: int = 10,
) -> pd.DataFrame:
    """Create hand-crafted features and asset return series."""
    df = data.copy()
    df["asset_return"] = df["Close"].pct_change()
    df["rolling_mean_return"] = df["asset_return"].rolling(mean_window).mean()
    df["rolling_vol"] = df["asset_return"].rolling(vol_window).std(ddof=0)
    df["momentum"] = df["Close"] / df["Close"].shift(momentum_window) - 1.0

    df = df.dropna().reset_index(drop=True)

    required_cols = ["Date", "Close", "asset_return"] + FEATURE_COLUMNS
    return df[required_cols].copy()
