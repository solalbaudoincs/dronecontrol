from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import AVDataset

class AVDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        input: np.ndarray,
        target: np.ndarray,
        batch_size: int,
        val_split: float = 0.2,
        test_split: float = 0.1,
        seed: int | None = None,
    ) -> None:

        super().__init__()
        self.input = input
        self.target = target
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed if seed is not None else 42

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:

        # First split: train + val vs test
        input_train_val, input_test, target_train_val, target_test = train_test_split(
            self.input,
            self.target,
            test_size=self.test_split,
            random_state=self.seed,
            shuffle=False,
        )

        # Second split: train vs val
        val_size_adjusted = self.val_split / (1 - self.test_split)
        input_train, input_val, target_train, target_val = train_test_split(
            input_train_val,
            target_train_val,
            test_size=val_size_adjusted,
            random_state=self.seed,
            shuffle=False,
        )

        # Create datasets
        self.train_dataset = AVDataset(input=input_train, target=target_train)
        self.val_dataset = AVDataset(input=input_val, target=target_val)
        self.test_dataset = AVDataset(input=input_test, target=target_test)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Data module not set up before accessing train_dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Data module not set up before accessing val_dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Data module not set up before accessing test_dataloader")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
        )
