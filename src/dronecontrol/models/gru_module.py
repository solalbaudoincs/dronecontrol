"""GRU Lightning module."""

from __future__ import annotations

from typing import Optional
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
        scheduler_type: Optional[str] = "ReduceLROnPlateau",
        scheduler_kwargs: Optional[dict] = None,
        dropout: float = 0.0,
        lr: float = 1e-2,
        **_,
    ):
        super().__init__(
            input_dim, 
            output_dim, 
            hidden_dim, 
            lr,
            scheduler_type=scheduler_type,
            scheduler_kwargs=scheduler_kwargs
            )

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

        self.regressor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]

        if hidden is not None:
            out, h = self.gru(x, hidden)
        else:
            out, h = self.gru(x)
        return self.regressor(out), h