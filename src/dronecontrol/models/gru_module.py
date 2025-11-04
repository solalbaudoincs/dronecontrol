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
        dropout: float = 0.0,
        lr: float = 1e-3,
        **_,
    ):
        super().__init__(input_dim, output_dim, lr, scheduler_type=_['scheduler_type'], scheduler_kwargs=_['scheduler_kwargs'])

        self.save_hyperparameters({
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }) 

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.batchnorm = nn.BatchNorm1d(input_dim, affine=False)
        self.regressor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass through the GRU model.
        
        Args:
            x: Input tensor of shape (batch_size, T, input_dim)

        Returns:
            Output tensor of shape (batch_size, T, output_dim)
        """
        # Data is 1D so we unsqueeze the last dimension
        x = self.batchnorm(x.permute(0,2,1)).permute(0,2,1)
        
        h, _ = self.gru(x)
        return self.regressor(h)