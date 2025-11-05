import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional

from dronecontrol.commande.loss import TrajectoryLoss
from dronecontrol.commande.neural_ekf import NeuralEKF
from dronecontrol.models.base_module import BaseModel

try:
    from dronecontrol.simulink.simulator import DroneSimulator
except ImportError:
    print("Simulink simulator not available.")

    
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


class MPCTorch:
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
        Q_weight: float = 0.1,      # Control effort weight
        R_weight: float = 10.0,     # Tracking error weight
        lr: float = 0.01,
        max_epochs: int = 50,
        u_min: float = -5.0,
        u_max: float = 5.0,
        use_ekf: bool = False
    ):
        """
        Initialize MPC controller.

        Args:
            accel_model: Neural network predicting acceleration
            dt: Time step
            horizon: Prediction horizon
            nb_steps: Total number of control steps
            Q_weight: Weight on control effort (scalar)
            R_weight: Weight on tracking error (scalar)
            lr: Learning rate for Adam optimizer
            max_epochs: Number of optimization iterations per step
            u_min, u_max: Control bounds
            use_ekf: Whether to use EKF for state estimation
        """
        self.accel_model = accel_model
        self.dt = dt
        self.horizon = horizon
        self.nb_steps = nb_steps
        self.Q_weight = Q_weight
        self.R_weight = R_weight
        self.lr = lr
        self.max_epochs = max_epochs
        self.u_min = u_min
        self.u_max = u_max
        self.use_ekf = use_ekf
        self.hidden_dim = accel_model.hidden_dim
        
    def compute_trajectory(
        self,
        u: torch.Tensor,
        x0: float,
        v0: float,
        hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute predicted trajectory given control sequence.
        
        Returns:
            x: Positions [horizon]
            v: Velocities [horizon]
            a: Accelerations [horizon]
        """
        # Predict accelerations
        with torch.set_grad_enabled(u.requires_grad):
            a, _ = self.accel_model(u, hidden)
        
        a = a.squeeze()
        
        # Integrate dynamics
        x = torch.zeros(self.horizon)
        v = torch.zeros(self.horizon)
        
        x[0] = float(x0)
        v[0] = float(v0)
        
        for t in range(1, self.horizon):
            v[t] = v[t-1] + a[t-1] * self.dt
            x[t] = x[t-1] + v[t-1] * self.dt + 0.5 * a[t-1] * self.dt**2
        
        return x, v, a

    def project_control(self, u: torch.Tensor) -> torch.Tensor:
        """Project control to satisfy bounds."""
        return torch.clamp(u, self.u_min, self.u_max)

    def step(
        self,
        x_ref: torch.Tensor,
        x0: float,
        v0: float,
        hidden: torch.Tensor,
        verbose: bool = False
    ) -> tuple[float, dict]:
        """
        Optimize control for one MPC step.
        
        Args:
            x_ref: Reference trajectory [horizon]
            x0: Current position
            v0: Current velocity
            hidden: Hidden state [1, 1, hidden_dim]
            verbose: Print optimization progress
            
        Returns:
            u_optimal: Optimal control for first step
            info: Dictionary with optimization info
        """
        # Detach inputs
        hidden = hidden.detach()
        x_ref = x_ref.detach()
        
        # Initialize control sequence
        u = torch.zeros(1, self.horizon, 1, requires_grad=True)
        
        # Optimizer
        optimizer = torch.optim.Adam([u], lr=self.lr)
        
        # Optimization loop
        losses = []
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            
            # Compute trajectory and loss
            x_pred, v_pred, a_pred = self.compute_trajectory(
                u, x0, v0, hidden
            )
            
            # Loss: tracking + control effort
            tracking_loss = self.R_weight * torch.sum((x_pred - x_ref)**2)
            control_loss = self.Q_weight * torch.sum(u.squeeze()**2)
            loss = tracking_loss + control_loss
            
            # Backward and step
            loss.backward()
            optimizer.step()
            
            # Project to bounds
            with torch.no_grad():
                u.data = self.project_control(u.data)
            
            losses.append(loss.item())
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}/{self.max_epochs}: "
                      f"loss={loss.item():8.4f} "
                      f"(tracking={tracking_loss.item():7.3f}, "
                      f"control={control_loss.item():7.3f})")
        
        # Extract optimal control
        u_optimal = u.detach()[0, 0, 0].item()
        
        # Compute final trajectory for info
        with torch.no_grad():
            x_pred, v_pred, a_pred = self.compute_trajectory(
                u.detach(), x0, v0, hidden
            )
        
        info = {
            'loss_history': losses,
            'final_loss': losses[-1],
            'u_sequence': u.detach().squeeze().numpy(),
            'x_predicted': x_pred.numpy(),
            'v_predicted': v_pred.numpy(),
            'a_predicted': a_pred.numpy(),
            'tracking_error': torch.mean((x_pred - x_ref)**2).item()
        }
        
        return u_optimal, info

    def solve(
        self,
        x_ref: np.ndarray,
        x0: float,
        v0: float,
        verbose: bool = True,
        return_history: bool = True
    ) -> tuple[np.ndarray, Optional[list[MPCState]]]:
        """
        Execute full MPC control loop.
        
        Args:
            x_ref: Reference trajectory [nb_steps]
            x0: Initial position
            v0: Initial velocity
            verbose: Print progress
            return_history: Return detailed state history
            
        Returns:
            u_history: Applied control sequence [nb_steps]
            state_history: List of MPCState objects (if return_history=True)
        """
        # Initialize
        u_history = np.zeros(self.nb_steps)
        state_history = [] if return_history else None
        
        x_current = x0
        v_current = v0
        h_current = torch.zeros(1, 1, self.hidden_dim)
        
        # Setup simulator and EKF
        simulator = DroneSimulator()
        
        if self.use_ekf:
            ekf = NeuralEKF(
                model=self.accel_model,
                hidden_dim=self.hidden_dim,
                meas_dim=1
            )
            h_current_np = np.zeros(self.hidden_dim)
        
        if verbose:
            print("\n" + "="*80)
            print("MPC CONTROL LOOP (PyTorch)")
            print("="*80)
            print(f"Horizon: {self.horizon} | Steps: {self.nb_steps} | "
                  f"Q: {self.Q_weight} | R: {self.R_weight}")
            print("-"*80)
        
        # MPC loop
        for step in range(self.nb_steps):
            # Extract reference for current horizon
            horizon_end = min(step + self.horizon, len(x_ref))
            x_ref_horizon = x_ref[step:horizon_end]
            
            # Pad if necessary
            if len(x_ref_horizon) < self.horizon:
                x_ref_horizon = np.pad(
                    x_ref_horizon,
                    (0, self.horizon - len(x_ref_horizon)),
                    mode='edge'
                )
            
            x_ref_torch = torch.tensor(x_ref_horizon, dtype=torch.float32)
            
            # Solve MPC
            u_optimal, info = self.step(
                x_ref=x_ref_torch,
                x0=x_current,
                v0=v_current,
                hidden=h_current,
                verbose=verbose and (step % 10 == 0)  # Print every 10 steps
            )
            
            u_history[step] = u_optimal
            
            # Apply control to simulator (4 motors)
            a_measured = simulator.accel(control_input=[u_optimal] * 4)[-1]
            v_new = simulator.vel[-1]
            x_new = simulator.pos[-1]
            
            # Compute tracking error
            tracking_error = abs(x_new - x_ref[step]) if step < len(x_ref) else 0.0
            
            # Save state
            if return_history:
                state_history.append(MPCState(
                    step=step,
                    position=x_new,
                    velocity=v_new,
                    acceleration=a_measured,
                    control=u_optimal,
                    hidden_state=h_current_np if self.use_ekf else h_current.squeeze().numpy(),
                    tracking_error=tracking_error
                ))
            
            if verbose:
                if step % 5 == 0:  # Print every 5 steps
                    print(f"Step {step:3d}/{self.nb_steps} | "
                          f"x={x_new:7.3f} (ref={x_ref[step]:7.3f}) | "
                          f"v={v_new:7.3f} | "
                          f"a={a_measured:7.3f} | "
                          f"u={u_optimal:7.3f} | "
                          f"err={tracking_error:7.4f}")
            
            # Update state
            x_current = x_new
            v_current = v_new
            
            if self.use_ekf:
                # EKF update
                h_current_np = ekf.step(
                    u=np.array([u_optimal]),
                    a_measured=np.array([a_measured]),
                    hk=h_current_np
                )
                h_current = torch.tensor(
                    h_current_np, dtype=torch.float32
                ).reshape(1, 1, -1)
            else:
                # Open-loop hidden state update
                with torch.no_grad():
                    u_torch = torch.tensor(
                        [[u_optimal]], dtype=torch.float32
                    ).reshape(1, 1, 1)
                    _, h_current = self.accel_model(u_torch, h_current)
        
        if verbose:
            print("-"*80)
            print(f"✓ MPC completed")
            print(f"  Final position: {x_current:.3f}")
            print(f"  Final velocity: {v_current:.3f}")
            if return_history:
                avg_error = np.mean([s.tracking_error for s in state_history])
                max_error = np.max([s.tracking_error for s in state_history])
                print(f"  Average tracking error: {avg_error:.4f}")
                print(f"  Maximum tracking error: {max_error:.4f}")
            print("="*80 + "\n")
        
        return u_history, state_history


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