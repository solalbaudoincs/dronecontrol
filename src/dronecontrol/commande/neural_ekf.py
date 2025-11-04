import numpy as np
import torch


class NeuralEKF:
    """
    Extended Kalman Filter wrapper for a neural network dynamics model.
    """

    def __init__(self, model, hidden_dim: int, meas_dim: int, 
                 Q_scale: float = 1e-3, R_scale: float = 1e-2):
        self.model = model
        self.hidden_dim = hidden_dim
        self.meas_dim = meas_dim

        # État de l'EKF (sans filterpy)
        self.x = np.zeros(hidden_dim)           # État caché estimé
        self.P = np.eye(hidden_dim) * 0.1       # Covariance de l'état
        self.Q = np.eye(hidden_dim) * Q_scale   # Bruit de processus
        self.R = np.eye(meas_dim) * R_scale     # Bruit de mesure

    def state_transition(self, u: np.ndarray, h: np.ndarray) -> np.ndarray:
        """State transition: h_k+1 = f(h_k, u_k) using the neural network."""
        h_t = torch.tensor(h, dtype=torch.float32).unsqueeze(0)
        u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            _, h_next = self.model(u_t, h_t)
        
        return h_next.squeeze(0).numpy()

    def measurement_function(self, u: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Measurement function: a_k = h(h_k, u_k) - predict accel from hidden state."""
        h_t = torch.tensor(h, dtype=torch.float32).unsqueeze(0)
        u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            a_hat, _ = self.model(u_t, h_t)
        
        return a_hat.squeeze(0).numpy()

    def compute_jacobian_wrt_hidden(self, func_type: str, u: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of func with respect to hidden state h.
        """
        h_t = torch.tensor(h, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
        
        # Forward pass
        a_hat, h_next = self.model(u_t, h_t)
        
        # Select which output to differentiate
        if func_type == 'state':
            output = h_next
        elif func_type == 'measurement':
            output = a_hat
        else:
            raise ValueError(f"Unknown func_type: {func_type}")
        
        # Compute Jacobian
        jacobian = []
        output_flat = output.squeeze(0)
        
        for i in range(output_flat.shape[0]):
            if h_t.grad is not None:
                h_t.grad.zero_()
            
            output_flat[i].backward(retain_graph=True)
            jacobian.append(h_t.grad.squeeze(0).clone().numpy())
        
        return np.array(jacobian)

    def step(self, u: np.ndarray, hk: np.ndarray, a_measured: np.ndarray) -> np.ndarray:
        """
        Perform one EKF step (predict + update) starting from a given hidden state.
        """
        
        # Use provided hidden state as starting point
        self.x = hk
        
        # ===== PREDICT STEP =====
        F = self.compute_jacobian_wrt_hidden('state', u, self.x)
        x_pred = self.state_transition(u, self.x)
        P_pred = F @ self.P @ F.T + self.Q
        
        # ===== UPDATE STEP =====
        H = self.compute_jacobian_wrt_hidden('measurement', u, x_pred)
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