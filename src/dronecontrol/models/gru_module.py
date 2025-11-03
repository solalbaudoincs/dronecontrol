"""GRU Lightning module."""

from __future__ import annotations

import torch
from torch import nn

from .base_module import BaseModel


class GRU(BaseModel):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        lr: float = 1e-3,
        **_: object,
    ):
        super().__init__(lr)

        self.save_hyperparameters({
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }) 

        self.gru = nn.GRU(
            input_size=1,
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
        if x.ndim == 2:
            x = x.unsqueeze(1)
        h, _ = self.gru(x)
        return self.regressor(h)