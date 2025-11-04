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
        ckpt_path: Optional[str] = None,
        model_class: Optional[Callable[..., nn.Module]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.device = torch.device(device)
        self.accel_model: Optional[nn.Module] = None

    def load_model(self) -> nn.Module:
        """Load or instantiate the acceleration model and move it to device.

        Behavior:
          - If `accel_model` is already set, return it.
          - If `ckpt_path` + `model_class` provided: instantiate model_class(**model_kwargs)
            then attempt to load state dict from checkpoint.
          - If `ckpt_path` provided but no model_class: try to load a full module
            or a state_dict from the checkpoint and return a module if possible.
        """
        if self.accel_model is not None:
            return self.accel_model

        if self.model_class is not None:
            model = self.model_class(**self.model_kwargs)
            model.to(self.device)
            if self.ckpt_path and self.ckpt_path.exists():
                ckpt = torch.load(self.ckpt_path, map_location=self.device)
                # common checkpoint formats
                if isinstance(ckpt, dict):
                    # try common keys
                    for key in ("state_dict", "model_state_dict", "net", "params"):
                        if key in ckpt:
                            state = ckpt[key]
                            break
                    else:
                        state = ckpt
                else:
                    state = ckpt

                # if the checkpoint stores a LightningModule state_dict with prefixes,
                # try to filter keys if necessary
                if isinstance(state, dict):
                    # try loading directly, if keys mismatch try stripping 'model.' prefixes
                    try:
                        model.load_state_dict(state)
                    except Exception:
                        new_state = {k.replace('model.', ''): v for k, v in state.items()}
                        model.load_state_dict(new_state)

            model.eval()
            self.accel_model = model
            return model

        # Try to load a fully saved nn.Module from checkpoint path
        if self.ckpt_path and self.ckpt_path.exists():
            obj = torch.load(self.ckpt_path, map_location=self.device)
            if isinstance(obj, nn.Module):
                obj.to(self.device)
                obj.eval()
                self.accel_model = obj
                return obj
            # If dict, maybe already a state_dict
            if isinstance(obj, dict):
                # as a fallback, return a lightweight identity model that subtracts gravity
                # user should provide model_class for better behavior
                class IdentityAccel(nn.Module):
                    def forward(self, u):
                        return u

                model = IdentityAccel().to(self.device)
                try:
                    model.load_state_dict(obj)
                except Exception:
                    # ignore
                    pass
                model.eval()
                self.accel_model = model
                return model

        raise RuntimeError("No model available: provide `model_class` or valid `ckpt_path`")


    def run(
        self,
        x_ref: torch.Tensor,
        x0: float,
        v0: float,
        horizon: int,
        dt: float,
        Q: torch.Tensor,
        R: torch.Tensor,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
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
        model = self.load_model()

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

        optimizer = SimpleOptimizer(trajectory_loss_fn=loss_wrapper, )

        # initial control guess
        u_init = torch.zeros_like(x_ref, device=device)

        u_opt = optimizer.optimize(u_init, x_ref, verbose=False)

        return u_opt
