"""Data loading and splitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import yfinance as yf


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def download_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data using yfinance and return a clean frame."""
    data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data downloaded for {symbol} in range {start}..{end}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    expected_cols = {"Date", "Close"}
    missing = expected_cols.difference(data.columns)
    if missing:
        raise ValueError(f"Downloaded data missing columns: {missing}")

    return data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()


def split_dataset(
    data: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> DatasetSplit:
    """Chronologically split data into train/validation/test sets."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0,1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0,1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    n = len(data)
    if n < 100:
        raise ValueError("Dataset is too small; expected at least 100 rows")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = data.iloc[:train_end].reset_index(drop=True)
    val = data.iloc[train_end:val_end].reset_index(drop=True)
    test = data.iloc[val_end:].reset_index(drop=True)

    return DatasetSplit(train=train, val=val, test=test)
