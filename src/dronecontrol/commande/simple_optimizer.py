import torch
from typing import Callable


class SimpleOptimizer:
    """Simple trajectory optimizer using LBFGS."""

    def __init__(
        self,
        trajectory_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        lr: float,
        max_iter: int,
        history_size: int,
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
        self.trajectory_loss_fn = trajectory_loss_fn
        self.lr = lr
        self.max_iter = max_iter
        self.history_size = history_size
        self.max_epochs = max_epochs

    def optimize(
        self,
        u_init: torch.Tensor,
        x_ref: torch.Tensor,
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
        u = u_init.clone().detach().requires_grad_(True)


        optimizer = torch.optim.LBFGS(
            [u],
            lr=self.lr,
            max_iter=self.max_iter,
            history_size=self.history_size
        )

        def closure():
            optimizer.zero_grad()
            loss = self.trajectory_loss_fn(u, x_ref)
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

        return u.detach()

    @staticmethod
    def get_tensions(tensions: torch.Tensor, u_min: float, u_max: float) -> torch.Tensor:
        return u_min + (u_max - u_min) * torch.sigmoid(tensions)