import numpy as np
import torch
from typing import Union

from dronecontrol.models.base_module import BaseModel

ArrayLike = Union[np.ndarray, torch.Tensor]

class NeuralEKF:
    """
    Extended Kalman Filter wrapper for a neural network dynamics model.
    """

    def __init__(
            self, 
            model: BaseModel, 
            hidden_dim: int, 
            input_dim: int, 
            Q_weight: float = 1.0, 
            R_weight: float = 1.0
            ):
        
        self.model = model
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.device = next(model.parameters()).device

        # État de l'EKF (sans filterpy)
        self.h = torch.zeros(hidden_dim, device=self.device)           # État caché estimé
        self.P = torch.eye(hidden_dim, device=self.device) * 0.1       # Covariance de l'état
        self.Q = torch.eye(hidden_dim, device=self.device) * Q_weight  # Bruit de processus
        self.R = torch.tensor(input_dim, device=self.device, dtype=torch.float32) * R_weight               # Bruit de mesure

    def state_transition(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """State transition: h_k+1 = f(h_k, u_k) using the neural network."""
        h_t = h
        u_t = u
        
        with torch.no_grad():
            _, h_next = self.model(u_t, h_t)

        return h_next

    def measurement_function(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Measurement function: a_k = h(h_k, u_k) - predict accel from hidden state."""
        h_t = h
        u_t = u
        
        with torch.no_grad():
            a_hat, _ = self.model(u_t, h_t)
        
        return a_hat
    
    def compute_H(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:

        """
        Compute Jacobian of measurement function with respect to hidden state h.
        """
        h_t = h.to(self.device).requires_grad_(True)
        u_t = u
        
        def func(h):
            a_hat, _ = self.model(u_t, h)
            return a_hat
        
        H = torch.autograd.functional.jacobian(func, h_t)
        
        return H
    

    def compute_F(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian of func with respect to hidden state h.
        """
        device = next(self.model.parameters()).device
        h = h.to(device).requires_grad_(True)
        u = u.to(device)
            
        def func(h):
            _, h_next = self.model(u, h)
            return h_next
        
        F = torch.autograd.functional.jacobian(func, h)
        
        return F

    def step(self, u: torch.Tensor, hk: torch.Tensor, a_measured: torch.Tensor) -> torch.Tensor:
        """
        Perform one EKF step (predict + update) starting from a given hidden state.
        EKF is applied only on the last layer's hidden state.
        """
        
        # hk shape: (num_layers, hidden_dim)
        # Extract last layer for EKF
        num_layers = hk.shape[0] if hk.dim() > 1 else 1
        if hk.dim() > 1:
            h_last = hk[-1].to(self.device)  # Shape: (hidden_dim,)
        else:
            h_last = hk.to(self.device)
        
        self.h = h_last
        u = u.to(self.device)
        hk = hk.to(self.device)
        # ===== PREDICT STEP =====
        # For multi-layer, we need to compute jacobian w.r.t last layer only
        F = self.compute_F(u, hk)  # Jacobian of last layer
        
        # Extract last layer jacobian if needed
        if F.dim() == 4:  # (num_layers, hidden_dim, num_layers, hidden_dim)
            F = F[-1, :, -1, :]  # (hidden_dim, hidden_dim)
        elif F.dim() == 2:
            pass  # Already (hidden_dim, hidden_dim)
        
        h_pred_full = self.state_transition(u, hk)  # Full state transition
        if h_pred_full.dim() > 1:
            h_pred = h_pred_full[-1]  # Last layer only
        else:
            h_pred = h_pred_full

        P_pred = F @ self.P @ F.T + self.Q
        
        # ===== UPDATE STEP =====
        H = self.compute_H(u, hk if hk.dim() > 1 else hk.unsqueeze(0))
        
        # Extract jacobian w.r.t last layer
        if H.dim() == 4:  # (output_dim, num_layers, hidden_dim)
            H = H[0, :, -1, :]  # (output_dim, hidden_dim)
        
        z_pred = self.measurement_function(u, hk if hk.dim() > 1 else hk.unsqueeze(0))
        
        # Innovation
        y = a_measured.to(self.device) - z_pred
        if y.dim() == 0:
            y = y.unsqueeze(0)
        
        # Innovation covariance
        if self.R.dim() == 0:
            R_mat = self.R.unsqueeze(0).unsqueeze(0)
        else:
            R_mat = self.R if self.R.dim() == 2 else torch.diag(self.R)
        
        S = H @ P_pred @ H.T + R_mat
        
        # Kalman gain
        K = P_pred @ H.T @ torch.linalg.inv(S)
        
        # Update last layer state
        self.h = h_pred.unsqueeze(-1) + K @ y
        self.h = self.h.squeeze(-1)
        
        # Update covariance
        I = torch.eye(self.hidden_dim, device=self.device)
        self.P = (I - K @ H) @ P_pred
        
        # Reconstruct full hidden state with updated last layer
        if hk.dim() > 1 and num_layers > 1:
            h_updated = h_pred_full.clone()
            h_updated[-1] = self.h
            return h_updated.detach()
        else:
            return self.h.unsqueeze(0).detach() if num_layers > 1 else self.h.detach()

    def reset(self, x0=None):
        """Reset EKF state."""
        if x0 is None:
            self.h = torch.zeros(self.hidden_dim, device=self.device)
        else:
            self.h = torch.as_tensor(x0, device=self.device, dtype=torch.float32)
        self.P = torch.eye(self.hidden_dim, device=self.device) * 0.1

    def get_state(self) -> torch.Tensor:
        """Get current hidden state estimate."""
        return self.h.clone()

    def get_covariance(self) -> torch.Tensor:
        """Get current state covariance matrix."""
        return self.P.clone()