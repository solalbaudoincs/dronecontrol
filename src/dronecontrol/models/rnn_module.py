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
        print(input_dim)
        # Save model hyperparameters for logging and checkpointing
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
        
        # Output regression layer
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.regressor = nn.Linear(hidden_dim, 2000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass through the RNN model.
        
        Args:
            x: Input tensor of shape (batch_size, T, 1)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Data is 1D so we unsqueeze the last dimension
        x = self.batchnorm(x)
        
        h, _ = self.rnn(x)
        return self.regressor(h)