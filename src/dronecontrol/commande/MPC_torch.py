import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union

<<<<<<< HEAD
from dronecontrol.commande.loss import TrajectoryLoss
from dronecontrol.commande.neural_ekf import NeuralEKF
=======
from dronecontrol.commande.base_mpc import MPC, ArrayLike
>>>>>>> 73a6d4f (fixed torch mpc works great)
from dronecontrol.models.base_module import BaseModel

try:
    from dronecontrol.simulink.simulator import DroneSimulator
except ImportError:
    print("Simulink simulator not available.")

<<<<<<< HEAD
    
@dataclass
class MPCState:
    """Container for MPC state at each time step."""
    step: int
    position: float
    velocity: float
    acceleration: float
    control: float
    hidden_state: np.ndarray
    tracking_error: float
    
    def __str__(self):
        return (f"Step {self.step:3d} | "
                f"x={self.position:7.3f} | "
                f"v={self.velocity:7.3f} | "
                f"a={self.acceleration:7.3f} | "
                f"u={self.control:7.3f} | "
                f"err={self.tracking_error:7.3f}")
=======
>>>>>>> 73a6d4f (fixed torch mpc works great)


class MPCTorch(MPC):
    """
    Model Predictive Control using PyTorch optimization.
    
    Uses gradient descent (Adam) to optimize control sequence.
    """

    def __init__(
        self,
        accel_model: BaseModel,
        dt: float,
        horizon: int,
        nb_steps: int,
        Q: ArrayLike,               # Control effort weight matrix
        R: ArrayLike,               # Tracking error weight matrix
        lr: float = 0.01,
        max_epochs: int = 50,
        u_min: float = -5.0,
        u_max: float = 5.0,
        use_ekf: bool = False,
        use_simulink: bool = False
    ):
        """
        Initialize MPC controller.

        Args:
            accel_model: Neural network predicting acceleration
            dt: Time step
            horizon: Prediction horizon
            nb_steps: Total number of control steps
            Q: Weight matrix on control effort
            R: Weight matrix on tracking error
            lr: Learning rate for Adam optimizer
            max_epochs: Number of optimization iterations per step
            u_min, u_max: Control bounds
            use_ekf: Whether to use EKF for state estimation
            use_simulink: Whether to use Simulink for simulation
        """
        # Call parent constructor
        super().__init__(
            accel_model=accel_model,
            dt=dt,
            horizon=horizon,
            nb_steps=nb_steps,
            Q=Q,
            R=R,
            lr=lr,
            max_epochs=max_epochs,
            u_min=u_min,
            u_max=u_max,
            use_ekf=use_ekf,
            use_simulink=use_simulink
        )
        
        # Store scalar weights for loss computation
        self.Q = torch.tensor(Q, dtype=torch.float32)
        self.R = torch.tensor(R, dtype=torch.float32)

    def project_control(self, u: torch.Tensor) -> torch.Tensor:
        """Project control to satisfy bounds."""
        return torch.clamp(u, self.u_min, self.u_max)

    def optimize_control(
        self,
        x_ref: torch.Tensor,
        u: torch.Tensor,
        hk: torch.Tensor,
        xk: float,
        vk: float,
        verbose: bool = False
    ) -> float:
        """
        Optimize control sequence over the horizon using PyTorch Adam optimizer.

        Args:
            x_ref: Reference trajectory [horizon]
            u: Initial control sequence [1, horizon, 1]
            hk: Initial hidden state [num_layers, 1, hidden_dim]
            xk: Initial position
            vk: Initial velocity
            verbose: Whether to print optimization progress

        Returns:
            u_optimal: Optimized control for first step
        """
        horizon = x_ref.shape[0]
        # Initialize control sequence
        u = torch.zeros(1, horizon, 1, requires_grad=True)

        # Optimizer
        optimizer = torch.optim.Adam([u], lr=self.lr)
        
        # Optimization loop
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            
            # Compute trajectory and loss
            x_pred, _, _ = self._compute_trajectory_wrt_NN(
                u, xk, vk, hk
            )
            
            # Loss: tracking + control effort
            # ||x - x_ref||_R^2 = (x - x_ref)^T R (x - x_ref)
            error = x_pred - x_ref
            R = self.R[:horizon, :horizon]
            tracking_loss = torch.dot(error, R @ error)
            
            # ||u||_Q^2 = u^T Q u
            u_flat = u.squeeze(0).squeeze(-1)  # [horizon]
            Q = self.Q[:horizon, :horizon]
            control_loss = torch.dot(u_flat, Q @ u_flat)

            loss = tracking_loss + control_loss
            
            # Backward and step
            loss.backward()
            optimizer.step()
            
            # Project to bounds
            with torch.no_grad():
                u.data = self.project_control(u.data)
            
            
            # Log first epoch, last epoch, or every 10 epochs if verbose
            should_log = (epoch == 0 or epoch == self.max_epochs - 1 or 
                         (verbose and (epoch + 1) % 10 == 0))
            
            if should_log:
                print(f"    Epoch {epoch+1:3d}/{self.max_epochs}: "
                      f"loss={loss.item():8.4f} "
                      f"(tracking={tracking_loss.item():7.3f}, "
                      f"control={control_loss.item():7.3f})")
        
        # Extract optimal control
        u_optimal = u.detach()[0, 0, 0].item()
        
        return u_optimal


# Example usage
if __name__ == "__main__":
    print("MPC with PyTorch - Example usage:")
    print("""
    mpc = MPCTorch(
        accel_model=your_trained_model,
        dt=0.1,
        horizon=10,
        nb_steps=50,
        Q_weight=0.1,
        R_weight=10.0,
        lr=0.01,
        max_epochs=50,
        use_ekf=True
    )
    
    # Generate reference trajectory
    x_ref = np.linspace(0, 10, 50)
    
    # Run MPC
    u_history, state_history = mpc.solve(
        x_ref=x_ref, 
        x0=0.0, 
        v0=0.0, 
        verbose=True
    )
    
    # Access detailed states
    for state in state_history:
        print(state)
    """)