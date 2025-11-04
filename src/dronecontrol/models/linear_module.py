"""Linear regression Lightning module."""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional

from .base_module import BaseModel


class LinearModel(nn.Module):
    def __init__(
            self, 
            input_dim: int,
            output_dim: int,
            **_
            ):
        
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)
