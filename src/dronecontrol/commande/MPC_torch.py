from typing import Optional, Tuple
import torch

from dronecontrol.commande.base_mpc import MPC, ArrayLike
from dronecontrol.models.base_module import BaseModel



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
        Q: ArrayLike,               # Control effort weight matrix
        R: ArrayLike,               # Tracking error weight matrix
        S: ArrayLike,               # Speed overshoot weight matrix
        tau: float,                 # Speed overshoot weight
        lr: float = 0.01,
        max_epochs: int = 50,
        u_min: float = -5.0,
        u_max: float = 5.0,
        use_ekf: bool = False,
        use_simulink: bool = False,
        optimize_trajectory: bool = False,
        max_accel: float = 9.81,
        smoothing: bool = True,
        smoothing_alpha: float = 0.3,
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
            lr: Learning rate for Adam optimizer (not used for LBFGS)
            max_epochs: Number of optimization iterations per step
            u_min, u_max: Control bounds
            use_ekf: Whether to use EKF for state estimation
            use_simulink: Whether to use Simulink for simulation
            optimizer_type: Type of optimizer ("adam" or "lbfgs")
        """
        # Call parent constructor
        super().__init__(
            accel_model=accel_model,
            dt=dt,
            horizon=horizon,
            Q=Q,
            R=R,
            u_min=u_min,
            u_max=u_max,
            use_ekf=use_ekf,
            use_simulink=use_simulink,
            optimize_trajectory=optimize_trajectory,
            max_accel=max_accel,
            smoothing=smoothing,
            smoothing_alpha=smoothing_alpha,
        )
        
        self.lr = lr
        self.max_epochs = max_epochs
        self.early_stop_tol = 1e-5
        self.early_stop_patience = 10
        # Store scalar weights for loss computation
        self.Q = torch.tensor(Q, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
        self.R = torch.tensor(R, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
        self.S = torch.tensor(S, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
        self.tau = tau

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
        verbose: bool = False,
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
        device = next(self.accel_model.parameters()).device
        horizon = x_ref.shape[0]
        # Ensure tensors are on the model's device
        x_ref = x_ref.to(device)
        hk = hk.to(device)
        # Initialize control sequence on device
        u_stable = 0.386  # Stable hover control
        u = torch.full((horizon, 1), u_stable, requires_grad=True, device=device)
        
        optimizer = torch.optim.Adam([u], lr=self.lr)
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        
        # Optimization loop
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            
            # Compute trajectory and loss
            x_pred, v_pred, _ = self._compute_trajectory_wrt_NN(
                u, xk, vk, hk
            )
            
            # Loss: tracking + control effort

            # Control tracking error for trajectory
            # ||x - x_ref||_R^2 = (x - x_ref)^T R (x - x_ref)
            error = x_pred - x_ref  # Adjusted reference
            R = self.R[:horizon, :horizon]
            tracking_loss = torch.dot(error, R @ error)

            # # Control tracking effort for speed
            # # x' = 1/tau * (x_pred - x_ref)
            # lag = self.tau * v_pred - (x_pred - x_ref)
            # S = self.S[:horizon, :horizon]
            # speed_tracking_loss = torch.dot(lag.view(-1), S @ lag.view(-1))

            # ||u||_Q^2 = u^T Q u
            u_flat = u.squeeze(-1) # [horizon]
            Q = self.Q[:horizon, :horizon]
            control_loss = torch.dot(u_flat, Q @ u_flat)

            loss = tracking_loss + control_loss #+ speed_tracking_loss

            # Backward and step
            loss.backward()
            optimizer.step()
            
            # Project to bounds
            with torch.no_grad():
                u.data = self.project_control(u.data)
            
            # Early stopping check
            if loss.item() < best_loss - self.early_stop_tol:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stop_patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}/{self.max_epochs}")
                break
        
        # Extract optimal control
        u_optimal = u.detach()[0, 0].item()

        
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
    u_history, state_ y = mpc.solve(
        x_ref=x_ref, 
        x0=0.0, 
        v0=0.0, 
        verbose=True
    )
    
    # Access detailed states
    for state in state_history:
        print(state)
    """)