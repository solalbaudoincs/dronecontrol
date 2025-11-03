"""Base Lightning module definitions."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn


class BaseLightningModel(pl.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
