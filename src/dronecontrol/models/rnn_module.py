"""Vanilla RNN Lightning module."""

from __future__ import annotations

import torch
from torch import nn

from .base_module import BaseLightningModel


class RNNLightningModel(BaseLightningModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        lr: float = 1e-3,
        **_: object,
    ):

        super().__init__(input_dim, output_dim, lr)
        
        self.save_hyperparameters({
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        })
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        _, hidden = self.rnn(x)
        last_hidden = hidden[-1]
        return self.regressor(last_hidden)
