"""Data loading and serialization utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def dataframe_to_arrays(inputs: pd.DataFrame, outputs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = inputs.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    Y = outputs.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    if Y.ndim == 1:
        Y = Y[:, None]
    return X, Y


def train_val_test_split(
    X: np.ndarray,
    Y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("Validation and test ratios must sum to less than 1")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    n_total = len(X)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test
    split = {
        "train": (X_shuffled[:n_train], Y_shuffled[:n_train]),
        "val": (X_shuffled[n_train:n_train + n_val], Y_shuffled[n_train:n_train + n_val]),
        "test": (X_shuffled[n_train + n_val:], Y_shuffled[n_train + n_val:]),
    }
    LOGGER.info(
        "Split dataset into train=%d, val=%d, test=%d",
        len(split["train"][0]),
        len(split["val"][0]),
        len(split["test"][0]),
    )
    return {
        "X_train": split["train"][0],
        "Y_train": split["train"][1],
        "X_val": split["val"][0],
        "Y_val": split["val"][1],
        "X_test": split["test"][0],
        "Y_test": split["test"][1],
    }


def save_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    LOGGER.info("Persisted dataset to %s", path)


def load_npz(path: str) -> Tuple[np.ndarray, ...]:
    with np.load(path) as data:
        X_train = data["X_train"]
        Y_train = data["Y_train"]
        X_val = data["X_val"]
        Y_val = data["Y_val"]
        X_test = data["X_test"]
        Y_test = data["Y_test"]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test
