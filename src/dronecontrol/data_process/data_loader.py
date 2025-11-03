import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

from .dataset import AVDataset

class AVDataLoader(pl.LightningDataModule):
    def __init__(
            self, 
            input: np.ndarray,
            target: np.ndarray,
            batch_size: int
            ) -> None:
        
        super().__init__()
        self.input = input
        self.target = target
        self.batch_size = batch_size
        
    def setup(self, stage: str) -> None:
        self.dataset = AVDataset(
            input=self.input,
            target=self.target
        )

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
