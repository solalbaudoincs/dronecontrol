"""Run control optimization using a learned accel-model checkpoint.

This module provides `ControlRunner` which loads an acceleration model
from a checkpoint (or accepts an instance), wraps it into `TrajectoryLoss`,
and runs `SimpleOptimizer` to compute an optimal control trajectory.
"""

from __future__ import annotations

from typing import Optional, Callable, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .loss import TrajectoryLoss
from .simple_optimizer import SimpleOptimizer


class ControlRunner:
    """Utility to run control optimization.

    Usage examples:
      runner = ControlRunner(ckpt_path='models/accel.ckpt', model_class=MyModel)
      u_opt = runner.run(x_ref, x0=x0, v0=v0)
    """

    def __init__(
        self,
        model_class: pl.LightningModule,
        ckpt_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """Initialize ControlRunner. Loads model from checkpoint if provided. Model kwargs are passed to the model constructor."""
        
        self.model_class = model_class
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None
        self.device = torch.device(device)
        
        # Load model immediately during init
        self.accel_model = self._load_model()

    def _load_model(self) -> nn.Module:
        """Load or instantiate the acceleration model and move it to device.
        
        Two scenarios:
          1. With checkpoint: use Lightning's load_from_checkpoint if LightningModule
          2. Without checkpoint: instantiate fresh model with model_kwargs
        """
        # Scenario 1: Load from checkpoint
        if isinstance(self.ckpt_path, str) and self.ckpt_path.exists():
            try:
                model = self.model_class.load_from_checkpoint(
                    self.ckpt_path,
                    map_location=self.device,
                )
                model.to(self.device)
                model.eval()
                return model
            except (TypeError, AttributeError):
                pass
            
            raise RuntimeError(f"Failed to load model from checkpoint: {self.ckpt_path}")
        
        # Scenario 2: Instantiate fresh model without checkpoint
        model = self.model_class
        model.to(self.device)
        model.eval()
        return model


    def run(
        self,
        x_ref: torch.Tensor,
        x0: float,
        v0: float,
        horizon: int,
        dt: float,
        Q: torch.Tensor,
        R: torch.Tensor,
        lr: float,
        max_iter: int,
        history_size: int,
        max_epochs: int,
    ) -> torch.Tensor:
        """Run the optimizer and return optimized controls `u`.

        Args:
            x_ref: reference positions (horizon x 1) torch.Tensor
            x0: initial position (scalar)
            v0: initial velocity (scalar)
            horizon: if provided, overrides x_ref.shape[0]
            dt: time step
            Q, R: optional weighting matrices (horizon x horizon)
            optimizer_kwargs: passed to `SimpleOptimizer` constructor
        Returns:
            Optimized control `u` as torch.Tensor with same shape as `x_ref`.
        """
        # Model is already loaded in __init__
        model = self.accel_model

        device = self.device

        # Ensure types
        Q = Q.to(device)
        R = R.to(device)
        x_ref = x_ref.to(device)

        loss_module = TrajectoryLoss(accel_model=model, horizon=horizon, dt=dt, Q_tensor=Q, R_tensor=R)

        # wrapper to match SimpleOptimizer signature
        def loss_wrapper(u, xref_local):
            # TrajectoryLoss expects (u, x_ref, v0, x0)
            return loss_module(u, xref_local, torch.tensor([v0], device=device), torch.tensor([x0], device=device))

        optimizer = Step(
            trajectory_loss_fn=loss_wrapper, lr=lr, max_iter=max_iter, history_size=history_size, max_epochs=max_epochs
        )

        # initial control guess
        u_init = torch.zeros_like(x_ref, device=device)

        u_opt = optimizer.optimize(u_init, x_ref, verbose=False)

        return u_opt
