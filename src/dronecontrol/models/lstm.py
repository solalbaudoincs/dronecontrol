"""LSTM Lightning module."""

from typing import Tuple
import torch
from torch import nn
from typing import Optional

from .base_module import BaseModel


class LSTM(BaseModel):
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
        **_: object,
    ):

        super().__init__(
            input_dim, 
            output_dim, 
            hidden_dim, 
            lr,
            num_layers=num_layers,
            scheduler_type=scheduler_type,
            scheduler_kwargs=scheduler_kwargs
            )
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

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        # x: [batch, seq_len, input_dim]

        if hidden is not None:
            out, h = self.lstm(x, hidden)
        else:
            out, h = self.lstm(x)

        out = self.regressor(out)

        return out, h