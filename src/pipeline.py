"""Shared dataset pipeline helpers for training and experiment scripts."""

from __future__ import annotations

from typing import Any, Dict

from src.data import DatasetSplit, download_price_data, split_dataset
from src.features import build_features


def prepare_dataset_split(config: Dict[str, Any]) -> DatasetSplit:
    """Download raw prices, build features, and create train/val/test split."""
    dataset_cfg = config["dataset"]
    raw = download_price_data(
        symbol=dataset_cfg["symbol"],
        start=dataset_cfg["start"],
        end=dataset_cfg["end"],
    )

    feature_cfg = config["features"]
    featured = build_features(
        raw,
        mean_window=feature_cfg["mean_window"],
        vol_window=feature_cfg["vol_window"],
        momentum_window=feature_cfg["momentum_window"],
    )

    split_cfg = config["split"]
    return split_dataset(
        featured,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
    )
