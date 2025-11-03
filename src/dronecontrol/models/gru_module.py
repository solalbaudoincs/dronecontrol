"""GRU Lightning module."""

from __future__ import annotations

import torch
from torch import nn

from .base_module import BaseLightningModel


class GRULightningModel(BaseLightningModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lr: float = 1e-3,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        **_: object,
    ):
        super().__init__(input_dim, output_dim, lr)
        self.save_hyperparameters({
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }) 
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim > 1 else 1),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2 if hidden_dim > 1 else 1, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        return self.regressor(last_hidden)
