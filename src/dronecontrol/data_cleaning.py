"""Data cleaning routines for UAV datasets."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def remove_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    mask = pd.Series(True, index=df.index)
    if method.lower() == "iqr":
        q1 = df[numeric_cols].quantile(0.25)
        q3 = df[numeric_cols].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask &= ~((df[numeric_cols] < lower) | (df[numeric_cols] > upper)).any(axis=1)
    elif method.lower() == "zscore":
        z = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std(ddof=0)
        mask &= ~(np.abs(z) > threshold).any(axis=1)
    else:
        raise ValueError(f"Unsupported outlier removal method: {method}")
    filtered = df[mask].reset_index(drop=True)
    LOGGER.info("Removed %d outlier rows using %s", len(df) - len(filtered), method)
    return filtered


def interpolate_missing(df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
    interpolated = df.interpolate(method=method, limit_direction="both")
    interpolated = interpolated.ffill().bfill()
    missing = interpolated.isna().sum().sum()
    if missing:
        LOGGER.warning("%d missing values remain after interpolation", missing)
    return interpolated


def normalize(
    df: pd.DataFrame,
    scope: str = "per_feature",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if scope == "global":
        mean = float(df[numeric_cols].values.mean())
        std = float(df[numeric_cols].values.std(ddof=0)) or 1.0
        df_norm = df.copy()
        df_norm[numeric_cols] = (df_norm[numeric_cols] - mean) / std
        params = {"global": {"mean": mean, "std": std}}
    elif scope == "per_feature":
        stats = df[numeric_cols].agg(["mean", "std"])
        stats.loc["std", :] = stats.loc["std"].replace(to_replace=0, value=1.0)
        df_norm = df.copy()
        df_norm[numeric_cols] = (df_norm[numeric_cols] - stats.loc["mean"]) / stats.loc["std"]
        params = {
            col: {"mean": float(stats.loc["mean", col]), "std": float(stats.loc["std", col])}
            for col in numeric_cols
        }
    else:
        raise ValueError(f"Unknown normalization scope: {scope}")
    LOGGER.info("Applied %s normalization", scope)
    return df_norm, params
