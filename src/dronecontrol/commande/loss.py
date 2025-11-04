import torch
from torch import nn
from typing import Tuple

class TrajectoryLoss(nn.Module):

    def __init__(
            self,
            accel_model : nn.Module,
            horizon: int,
            dt: float,
            Q_tensor: torch.Tensor,
            R_tensor: torch.Tensor,
            ) :
            
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.accel_model = accel_model
        self.horizon = horizon
        self.dt = dt
        self.Q = Q_tensor
        self.R = R_tensor

        assert Q_tensor.shape == (horizon, horizon), "Q tensor shape mismatch"
        assert R_tensor.shape == (horizon, horizon), "R tensor shape mismatch"

    def compute_trajectory(self, u: torch.Tensor) -> torch.Tensor:

        assert u.shape[0] == self.horizon, "Control input horizon mismatch"

        a_hat = self.accel_model(u)
        v_hat = torch.zeros_like(a_hat)
        v_hat[1:] = v_hat[:-1] + a_hat[:-1] * self.dt
        x_hat = torch.zeros_like(v_hat)
        x_hat[1:] = x_hat[:-1] + v_hat[:-1] * self.dt + 0.5 * a_hat[:-1] * self.dt**2

        return x_hat

    def forward(self, u: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:

        x_pred = self.compute_trajectory(u)

        total_loss = torch.mean( (x_pred - x_ref).T @ self.R @ (x_pred - x_ref) + (u.T @ self.Q @ u))

        return total_loss