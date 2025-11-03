"""GRU Lightning module."""

from __future__ import annotations

import torch
from torch import nn

from .base_module import BaseModel


class GRU(BaseModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        lr: float,
        **_: object,
    ):
        super().__init__(input_dim, output_dim, lr)

        self.save_hyperparameters({
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }) 

        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        
        x = self.batchnorm(x)
        h, _ = self.gru(x)
        return self.regressor(h)