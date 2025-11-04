import torch
import torch.nn as nn
from typing import Callable, Tuple

from .loss import TrajectoryLoss
from dronecontrol.models.base_module import BaseModel


class Optimizer:
    """Simple trajectory optimizer using LBFGS."""

    def __init__(
        self,
        lr: float,
        accel_model: BaseModel,
        dt: float,
        max_iter: int,
        horizon: int,
        nb_steps: int,
        Q_tensor: torch.Tensor,
        R_tensor: torch.Tensor,
        max_epochs: int
    ):
        """
        Initialize the optimizer.

        Args:
            trajectory_loss_fn: Function that computes loss given control input u and reference x_ref
            lr: Learning rate for LBFGS
            max_iter: Maximum iterations per LBFGS step
            history_size: History size for LBFGS
            max_epochs: Maximum number of optimization epochs
        """

        self.trajectory_loss_fn = TrajectoryLoss(
            accel_model=accel_model,
            horizon=horizon,
            dt=dt,
            Q_tensor=Q_tensor,
            R_tensor=R_tensor
        )
        self.accel_model = accel_model
        self.dt = dt
        self.hidden_dim = accel_model.hidden_dim
        self.lr = lr
        self.max_iter = max_iter
        self.horizon = horizon
        self.max_epochs = max_epochs
        self.nb_steps = nb_steps

    def step(
        self,
        x_ref: torch.Tensor,
        hidden: torch.Tensor,
        vk: torch.Tensor,
        xk: torch.Tensor,
        horizon: int,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Optimize the control trajectory.

        Args:
            u_init: Initial control input tensor (requires_grad=True)
            x_ref: Reference trajectory
            verbose: Whether to print progress

        Returns:
            Optimized control input tensor
        """
        u = torch.zeros(horizon, requires_grad=True)


        optimizer = torch.optim.LBFGS(
            [u],
            lr=self.lr,
            max_iter=self.max_iter
        )

        def closure():
            optimizer.zero_grad()
            loss = self.trajectory_loss_fn(u, hidden, x_ref, vk, xk)
            loss.backward()
            return loss

        losses = []
        for epoch in range(self.max_epochs):
            loss = optimizer.step(closure)
            losses.append(loss.item())

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.max_epochs}, Loss: {loss.item():.6f}")

        if verbose:
            print(f"Optimization completed. Final loss: {losses[-1]:.6f}")

        return u.detach()[0]

    @staticmethod
    def get_tensions(tensions: torch.Tensor, u_min: float, u_max: float) -> torch.Tensor:
        return u_min + (u_max - u_min) * torch.sigmoid(tensions)
    
    def optimize(
            self,
            x_ref: torch.Tensor,
            v0: torch.Tensor,
            x0: torch.Tensor,
            verbose: bool = True
        ) -> torch.Tensor:
        """Optimize the control trajectory over multiple steps."""

        u = torch.zeros(self.horizon, requires_grad=True)
        vk = v0
        xk = x0
        hidden = torch.zeros(self.hidden_dim)  # Placeholder for hidden state

        for step in range(self.nb_steps):
            u[step] = self.step(x_ref, hidden, vk, xk, horizon=self.horizon, verbose=verbose)
            a_hat_k, vk, xk, hidden = self.update_state(vk, xk, hidden, u[step])
            a_k = Simulink.get_acceleration(u[:step])

        return u

    def update_state(
            self, 
            vk: torch.Tensor, 
            xk: torch.Tensor, 
            hidden: torch.Tensor, 
            u: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update the state based on control input u using the accel model."""

        a_hat_next, hidden_next = self.accel_model(u.unsqueeze(0), hidden.unsqueeze(0))
        a_hat_next = a_hat_next.squeeze(0)

        vk_next = vk + a_hat_next * self.dt
        xk_next = xk + vk * self.dt + 0.5 * a_hat_next * self.dt**2

        return a_hat_next, vk_next, xk_next, hidden_next.squeeze(0)