"""Linear regression Lightning module."""

from __future__ import annotations

import torch
from torch import nn

from .base_module import BaseModel


class LinearModel(BaseModel):
    def __init__(
            self, 
            lr: float = 1e-3, 
            **_: object
            ):
        super().__init__(lr)
        self.model = nn.Sequential(
            nn.Linear(1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)
