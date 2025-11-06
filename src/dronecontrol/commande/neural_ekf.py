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

        # État de l'EKF (sans filterpy)
        self.x = np.zeros(hidden_dim)           # État caché estimé
        self.P = np.eye(hidden_dim) * 0.1       # Covariance de l'état
        self.Q = np.eye(hidden_dim) * Q_weight  # Bruit de processus
        self.R = np.array(input_dim) * R_weight               # Bruit de mesure

    def state_transition(self, u: np.ndarray, h: np.ndarray) -> np.ndarray:
        """State transition: h_k+1 = f(h_k, u_k) using the neural network."""
        device = next(self.model.parameters()).device
        h_t = torch.tensor(h, dtype=torch.float32, device=device).unsqueeze(0)
        u_t = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            _, h_next = self.model(u_t, h_t)
        
        return h_next.squeeze(0).detach().cpu().numpy()

    def measurement_function(self, u: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Measurement function: a_k = h(h_k, u_k) - predict accel from hidden state."""
        device = next(self.model.parameters()).device
        h_t = torch.tensor(h, dtype=torch.float32, device=device).unsqueeze(0)
        u_t = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            a_hat, _ = self.model(u_t, h_t)
        
        return a_hat.squeeze(0).detach().cpu().numpy()
    
    def compute_H(self, u: np.ndarray, h: np.ndarray) -> np.ndarray:

        """
        Compute Jacobian of measurement function with respect to hidden state h.
        """
        device = next(self.model.parameters()).device
        h_t = torch.tensor(h, dtype=torch.float32, requires_grad=True, device=device).unsqueeze(0)
        u_t = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0)
        
        def func(h):
            a_hat, _ = self.model(u_t, h)
            return a_hat
        
        H = torch.autograd.functional.jacobian(func, h_t)
        
        return H.squeeze(0).squeeze(1).detach().cpu().numpy()
    

    def compute_F(self, u: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of func with respect to hidden state h.
        """
        device = next(self.model.parameters()).device
        h_t = torch.tensor(h, dtype=torch.float32, requires_grad=True, device=device).unsqueeze(0)
        u_t = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0)
        
        def func(h):
            _, h_next = self.model(u_t, h)
            return h_next
        
        F = torch.autograd.functional.jacobian(func, h_t)
        
        return F.squeeze(0).squeeze(1).detach().cpu().numpy()

    def step(self, u: np.ndarray, hk: np.ndarray, a_measured: np.ndarray) -> np.ndarray:
        """
        Perform one EKF step (predict + update) starting from a given hidden state.
        """
        
        # Use provided hidden state as starting point
        self.x = hk
        
        # ===== PREDICT STEP =====
        F = self.compute_F(u, self.x)
        x_pred = self.state_transition(u, self.x)
        P_pred = F @ self.P @ F.T + self.Q
        
        # ===== UPDATE STEP =====
        H = self.compute_H(u, x_pred)
        z_pred = self.measurement_function(u, x_pred)
        
        # Innovation
        y = a_measured - z_pred
        
        # Innovation covariance
        S = H @ P_pred @ H.T + self.R
        
        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = x_pred + K @ y
        
        # Update covariance
        I = np.eye(self.hidden_dim)
        self.P = (I - K @ H) @ P_pred
        
        return self.x.copy()

    def reset(self, x0=None):
        """Reset EKF state."""
        if x0 is None:
            self.x = np.zeros(self.hidden_dim)
        else:
            self.x = np.array(x0, dtype=np.float32)
        self.P = np.eye(self.hidden_dim) * 0.1

    def get_state(self) -> np.ndarray:
        """Get current hidden state estimate."""
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Get current state covariance matrix."""
        return self.P.copy()