"""Vanilla RNN Lightning module."""

from __future__ import annotations

import torch
from torch import nn

from .base_module import BaseModel


class RNN(BaseModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        lr: float,
        **_,
    ):

        super().__init__(input_dim, output_dim, lr, hidden_dim=hidden_dim, scheduler_type=_['scheduler_type'], scheduler_kwargs=_['scheduler_kwargs'])
        
        self.save_hyperparameters({
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        })
        
        # Recurrent layers
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        if hidden is not None:
            out, h = self.rnn(x, hidden)
        else:
            out, h = self.rnn(x)

        return self.regressor(out), h