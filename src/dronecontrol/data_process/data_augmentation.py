"""Data augmentation helpers for UAV control pipeline."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


LOGGER = logging.getLogger(__name__)




class DataAugmenter(ABC):
    """Base class for data augmentation strategies."""

    @abstractmethod
    def augment(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment the input and target data.

        Args:
            input_data: Original input data.
            target_data: Original target data.

        Returns:
            A tuple of augmented input and target data.
        """
        pass


class NoOpAugmenter(DataAugmenter):
    """No-operation augmenter that returns data unchanged."""

    def augment(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return input_data, target_data