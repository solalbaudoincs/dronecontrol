"""LSTM Lightning module."""

from __future__ import annotations

import torch
from torch import nn

from .base_module import BaseModel


class LSTM(BaseModel):
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
        self.save_hyperparameters()

        self.batch_norm = nn.BatchNorm1d(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]

        x = x.permute(0, 2, 1)          # [batch, input_dim, seq_len]
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)          # [batch, seq_len, input_dim]

        h, _ = self.lstm(x)
        out = self.regressor(h)

        return out