import torch
from torch import diff, nn
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
            
    def compute_trajectory(self, u: torch.Tensor, hidden: torch.Tensor, vk: torch.Tensor, xk: torch.Tensor) -> torch.Tensor:

        assert u.shape[1] == self.horizon, "Control input horizon mismatch"

        # Prédire toutes les accélérations
        a_hat, _ = self.accel_model(u, hidden)  # shape: [1, horizon, 1]
        
        # Initialiser les tenseurs pour la vitesse et la position
        batch_size = u.shape[0]
        v_hat = torch.zeros(batch_size, self.horizon, 1, device=u.device, dtype=u.dtype)
        x_hat = torch.zeros(batch_size, self.horizon, 1, device=u.device, dtype=u.dtype)
        
        # Conditions initiales
        v_hat[:, 0] = vk
        x_hat[:, 0] = xk
        
        # Intégration séquentielle (boucle nécessaire pour les dépendances temporelles)
        for t in range(1, self.horizon):
            # Mise à jour de la vitesse: v[t] = v[t-1] + a[t-1] * dt
            v_hat[:, t] = v_hat[:, t-1] + a_hat[:, t-1] * self.dt
            
            # Mise à jour de la position: x[t] = x[t-1] + v[t-1] * dt + 0.5 * a[t-1] * dt^2
            x_hat[:, t] = x_hat[:, t-1] + v_hat[:, t-1] * self.dt + 0.5 * a_hat[:, t-1] * self.dt**2

        return x_hat
    
    def forward(self, x_ref: torch.Tensor, u: torch.Tensor, hidden: torch.Tensor, vk: torch.Tensor, xk: torch.Tensor) -> torch.Tensor:

        x_pred = self.compute_trajectory(u, hidden, vk, xk)

        diff = x_pred.squeeze() - x_ref
        u_sq = u.squeeze()
        total_loss = (diff @ self.R @ diff.T) + (u_sq @ self.Q @ u_sq.T)

        return total_loss