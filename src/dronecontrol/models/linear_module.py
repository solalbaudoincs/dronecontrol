"""Linear regression Lightning module."""

from __future__ import annotations

import torch
from torch import nn

from .base_module import BaseModel


class LinearModel(BaseModel):
    def __init__(
            self, 
            input_dim: int,
            output_dim: int,
            lr: float, 
            **_: object
            ):
        super().__init__(input_dim, output_dim, lr)
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)
