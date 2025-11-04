import torch
import torch.nn as nn
from typing import Callable, Tuple
from filterpy.kalman import ExtendedKalmanFilter
import numpy as np

from .loss import TrajectoryLoss
from dronecontrol.models.base_module import BaseModel
from dronecontrol.simulink.simulator import DroneSimulator
import numpy as np
import torch
import torch.nn as nn
from filterpy.kalman import ExtendedKalmanFilter


class NeuralEKF:
    """
    Extended Kalman Filter wrapper for a neural network dynamics model.

    Args:
        model: nn.Module or pl.LightningModule
            Must implement forward(u, h) -> (a_hat, h_next)
        hidden_dim: int
            Dimension of the hidden state
        meas_dim: int
            Dimension of the measured variable (e.g., acceleration)
        Q_scale: float
            Process noise scaling
        R_scale: float
            Measurement noise scaling
    """

    def __init__(self, model, hidden_dim: int, meas_dim: int, 
                 Q_scale: float = 1e-3, R_scale: float = 1e-2):
        self.model = model
        self.hidden_dim = hidden_dim
        self.meas_dim = meas_dim

        # Initialize EKF
        self.ekf = ExtendedKalmanFilter(dim_x=hidden_dim, dim_z=meas_dim)
        self.ekf.x = np.zeros(hidden_dim)
        self.ekf.P = np.eye(hidden_dim) * 0.1
        self.ekf.Q = np.eye(hidden_dim) * Q_scale
        self.ekf.R = np.eye(meas_dim) * R_scale

    # ------------------------------------------------------------
    # Internal helper functions
    # ------------------------------------------------------------

    def state_transition(self, h: np.ndarray, u: np.ndarray) -> np.ndarray:
        """State transition: h_k+1 = f(h_k, u_k) using the neural network."""
        h_t = torch.tensor(h, dtype=torch.float32).unsqueeze(0)
        u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            _, h_next = self.model(u_t, h_t)
        
        return h_next.squeeze(0).numpy()

    def measurement_function(self, h: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Measurement function: a_k = h(h_k, u_k) - predict accel from hidden state."""
        h_t = torch.tensor(h, dtype=torch.float32).unsqueeze(0)
        u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            a_hat, _ = self.model(u_t, h_t)
        
        return a_hat.squeeze(0).numpy()

    def compute_jacobian_wrt_hidden(self, func_type: str, h: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of func with respect to hidden state h.
        
        Args:
            func_type: 'state' for f(h,u) or 'measurement' for h(h,u)
            h: hidden state
            u: input/control
        
        Returns:
            Jacobian matrix [output_dim, hidden_dim]
        """
        h_t = torch.tensor(h, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
        
        # Forward pass
        a_hat, h_next = self.model(u_t, h_t)
        
        # Select which output to differentiate
        if func_type == 'state':
            output = h_next  # derivative of h_next wrt h
        elif func_type == 'measurement':
            output = a_hat   # derivative of a_hat wrt h
        else:
            raise ValueError(f"Unknown func_type: {func_type}")
        
        # Compute Jacobian
        jacobian = []
        output_flat = output.squeeze(0)
        
        for i in range(output_flat.shape[0]):
            # Compute gradient of output[i] wrt h
            if h_t.grad is not None:
                h_t.grad.zero_()
            
            output_flat[i].backward(retain_graph=True)
            jacobian.append(h_t.grad.squeeze(0).clone().numpy())
        
        return np.array(jacobian)

    def compute_jacobian_wrt_input(self, func_type: str, h: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of func with respect to input u.
        
        Args:
            func_type: 'state' for f(h,u) or 'measurement' for h(h,u)
            h: hidden state
            u: input/control
        
        Returns:
            Jacobian matrix [output_dim, input_dim]
        """
        h_t = torch.tensor(h, dtype=torch.float32).unsqueeze(0)
        u_t = torch.tensor(u, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        
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
            if u_t.grad is not None:
                u_t.grad.zero_()
            
            output_flat[i].backward(retain_graph=True)
            jacobian.append(u_t.grad.squeeze(0).clone().numpy())
        
        return np.array(jacobian)

    # ------------------------------------------------------------
    # EKF step
    # ------------------------------------------------------------
    def step(self, u: np.ndarray, a_measured: np.ndarray, hk: np.ndarray) -> np.ndarray:
        """
        Perform one EKF step (predict + update) starting from a given hidden state.

        Args:
            u: control input at time k [input_dim]
            a_measured: true measured acceleration at time k [meas_dim]
            hk: current hidden state at time k [hidden_dim] - from your RNN
        
        Returns:
            Updated hidden state estimate [hidden_dim]
        """
        
        # Use provided hidden state as starting point
        self.ekf.x = hk
        
        # ===== PREDICT STEP =====
        # Compute Jacobian F = ∂f/∂h (state transition wrt hidden state)
        F = self.compute_jacobian_wrt_hidden('state', self.ekf.x, u)
        
        # Predict next state
        x_pred = self.state_transition(self.ekf.x, u)
        
        # Predict covariance: P = F·P·F^T + Q
        P_pred = F @ self.ekf.P @ F.T + self.ekf.Q
        
        # ===== UPDATE STEP =====
        # Compute Jacobian H = ∂h/∂h (measurement function wrt predicted state)
        H = self.compute_jacobian_wrt_hidden('measurement', x_pred, u)
        
        # Predicted measurement
        z_pred = self.measurement_function(x_pred, u)
        
        # Innovation
        y = a_measured - z_pred
        
        # Innovation covariance: S = H·P·H^T + R
        S = H @ P_pred @ H.T + self.ekf.R
        
        # Kalman gain: K = P·H^T·S^(-1)
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # Update state: x = x_pred + K·y
        self.ekf.x = x_pred + K @ y
        
        # Update covariance: P = (I - K·H)·P_pred
        I = np.eye(self.hidden_dim)
        self.ekf.P = (I - K @ H) @ P_pred
        
        return self.ekf.x.copy()

    def get_jacobians_debug(self, u: np.ndarray) -> dict:
        """
        Get all Jacobians for debugging purposes.
        
        Returns dict with:
            - F_h: ∂f/∂h (state transition wrt hidden)
            - F_u: ∂f/∂u (state transition wrt input)
            - H_h: ∂h/∂h (measurement wrt hidden)
            - H_u: ∂h/∂u (measurement wrt input)
        """
        u = np.array(u, dtype=np.float32)
        
        return {
            'F_h': self.compute_jacobian_wrt_hidden('state', self.ekf.x, u),
            'F_u': self.compute_jacobian_wrt_input('state', self.ekf.x, u),
            'H_h': self.compute_jacobian_wrt_hidden('measurement', self.ekf.x, u),
            'H_u': self.compute_jacobian_wrt_input('measurement', self.ekf.x, u),
        }

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------

    def reset(self, x0=None):
        """Reset EKF state."""
        if x0 is None:
            self.ekf.x = np.zeros(self.hidden_dim)
        else:
            self.ekf.x = np.array(x0, dtype=np.float32)
        self.ekf.P = np.eye(self.hidden_dim) * 0.1

    def get_state(self) -> np.ndarray:
        """Get current hidden state estimate."""
        return self.ekf.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Get current state covariance matrix."""
        return self.ekf.P.copy()


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
        max_epochs: int,
        use_ekf: bool
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
        self.use_ekf = use_ekf

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
            x0: torch.Tensor,
            v0: torch.Tensor,
            verbose: bool = True
        ) -> torch.Tensor:
        """Optimize the control trajectory over multiple steps."""

        u = torch.zeros(self.nb_steps, requires_grad=True)
        vk = v0
        xk = x0
        hk = torch.zeros(self.hidden_dim)  # Placeholder for hidden state
        droneSimulator = DroneSimulator()
        neuralEKF = NeuralEKF(
                model=self.accel_model,
                hidden_dim=self.hidden_dim,
                meas_dim=1
            )

        for step in range(self.nb_steps):

            u[step] = self.step(x_ref, hk, vk, xk, horizon=self.horizon, verbose=verbose)
            ak = torch.tensor(droneSimulator.accel(control_input=u.detach().numpy()))  # assuming 4 motors
            vk = torch.tensor(droneSimulator.vel)
            xk = torch.tensor(droneSimulator.pos)

            if self.use_ekf :
                # 2. L'EKF corrige cet état avec la mesure réelle
                hk = torch.tensor(neuralEKF.step(
                    u=u[step].detach().numpy(),
                    a_measured=ak.detach().numpy(), 
                    hk=hk.detach().numpy()
                ))
            else:
                hk = self.update_state(u[step], hk)

        return u

    def update_state(
            self, 
            u: torch.Tensor,            
            hidden: torch.Tensor, 
            ) -> torch.Tensor:
        """Update the state based on control input u using the accel model."""

        _, hidden_next = self.accel_model(u.unsqueeze(0), hidden.unsqueeze(0))

        return hidden_next.squeeze()