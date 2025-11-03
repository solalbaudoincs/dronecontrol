"""Data augmentation helpers for UAV control pipeline."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def add_gaussian_noise(
    inputs: np.ndarray,
    outputs: np.ndarray,
    noise_std: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if noise_std <= 0:
        return inputs, outputs
    rng = np.random.default_rng(seed)
    noisy_inputs = inputs + rng.normal(scale=noise_std, size=inputs.shape)
    noisy_outputs = outputs + rng.normal(scale=noise_std, size=outputs.shape)
    augmented_inputs = np.concatenate([inputs, noisy_inputs], axis=0)
    augmented_outputs = np.concatenate([outputs, noisy_outputs], axis=0)
    LOGGER.info("Augmented data with Gaussian noise std=%s", noise_std)
    return augmented_inputs, augmented_outputs


def slice_windows(
    inputs: np.ndarray,
    outputs: np.ndarray,
    window_size: int,
    stride: int,
) -> Dict[str, np.ndarray]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    sequences_x = []
    sequences_y = []
    for start in range(0, len(inputs) - window_size + 1, stride):
        end = start + window_size
        sequences_x.append(inputs[start:end])
        sequences_y.append(outputs[end - 1])
    if not sequences_x:
        raise ValueError("Insufficient samples to create any sliding windows")
    X = np.stack(sequences_x)
    Y = np.stack(sequences_y)
    LOGGER.info("Created %d sliding windows of size %d", len(X), window_size)
    return {"X": X, "Y": Y}


def append_simulink_samples(
    inputs: np.ndarray,
    outputs: np.ndarray,
    sim_df: Optional[pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray]:
    if sim_df is None or sim_df.empty:
        return inputs, outputs
    sim_data = sim_df.select_dtypes(include=[np.number]).to_numpy()
    padded_outputs = np.zeros((sim_data.shape[0], outputs.shape[1]), dtype=outputs.dtype)
    combined_inputs = np.concatenate([inputs, sim_data], axis=0)
    combined_outputs = np.concatenate([outputs, padded_outputs], axis=0)
    LOGGER.info("Appended %d Simulink samples", sim_data.shape[0])
    return combined_inputs, combined_outputs
