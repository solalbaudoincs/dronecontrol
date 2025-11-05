import cvxpy as cp
import numpy as np
import torch
from typing import Optional, Tuple

from dronecontrol.commande.base_mpc import MPC, ArrayLike
from dronecontrol.models.base_module import BaseModel


class MPCCVXPY(MPC):
    """
    Model Predictive Control using CVXPY for convex optimization.
    
    Key improvements over PyTorch version:
    - Guaranteed convergence for convex problems
    - Native constraint handling
    - Much faster (specialized QP solvers)
    - No gradient graph issues
    """

    def __init__(
        self,
        accel_model: BaseModel,
        dt: float,
        horizon: int,
        nb_steps: int,
        Q: ArrayLike,               # Control effort weight matrix
        R: ArrayLike,               # Tracking error weight matrix
        u_min: float = -5.0,
        u_max: float = 5.0,
        use_ekf: bool = False,
        use_simulink: bool = False,
        linearization_iters: int = 3  # Number of SQP iterations
    ):
        """
        Initialize MPC controller.

        Args:
            accel_model: Neural network predicting acceleration from control
            dt: Time step
            horizon: Prediction horizon (number of future steps)
            nb_steps: Total number of control steps to execute
            Q: Weight matrix on control effort
            R: Weight matrix on tracking error
            u_min, u_max: Control input bounds
            use_ekf: Whether to use EKF for state estimation
            use_simulink: Whether to use Simulink for simulation
            linearization_iters: Number of SQP iterations for better accuracy
        """
        # Call parent constructor
        super().__init__(
            accel_model=accel_model,
            dt=dt,
            horizon=horizon,
            nb_steps=nb_steps,
            Q=Q,
            R=R,
            u_min=u_min,
            u_max=u_max,
            use_ekf=use_ekf,
            use_simulink=use_simulink
        )
        
        # Store matrices for cost computation
        self.Q_matrix = np.array(Q, dtype=np.float32) 
        self.R_matrix = np.array(R, dtype=np.float32) 
        self.linearization_iters = linearization_iters

    def linearize_model(
        self,
        u_nominal: np.ndarray,
        hidden: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize neural network dynamics around nominal control sequence.
        
        Approximation: a(u) ≈ a_nom + B·(u - u_nom)
        
        Returns:
            a_nominal: Accelerations at nominal point [horizon]
            B: Sensitivity matrix da/du [horizon, horizon] (diagonal)
        """
        # Convert to torch
        u_torch = torch.tensor(u_nominal, dtype=torch.float32).reshape(1, -1, 1)
        h_torch = hidden.detach() if isinstance(hidden, torch.Tensor) else torch.tensor(hidden, dtype=torch.float32)
        
        # Ensure correct shape [num_layers, 1, hidden_dim]
        if h_torch.dim() == 2:
            h_torch = h_torch.unsqueeze(1)
        
        # Nominal trajectory
        with torch.no_grad():
            a_nominal_torch, _ = self.accel_model(u_torch, h_torch)
            a_nominal = a_nominal_torch.squeeze(0).squeeze(-1).cpu().numpy()
        
        # Compute sensitivity using finite differences
        epsilon = 1e-4
        horizon = len(u_nominal)
        B = np.zeros(horizon)
        
        for i in range(horizon):
            u_pert = u_nominal.copy()
            u_pert[i] += epsilon
            u_pert_torch = torch.tensor(u_pert, dtype=torch.float32).reshape(1, -1, 1)
            
            with torch.no_grad():
                a_pert_torch, _ = self.accel_model(u_pert_torch, h_torch)
                a_pert = a_pert_torch.squeeze(0).squeeze(-1).cpu().numpy()
            
            B[i] = (a_pert[i] - a_nominal[i]) / epsilon
        
        return a_nominal, B

    def solve_qp(
        self,
        x_ref: np.ndarray,
        x0: float,
        v0: float,
        a_nominal: np.ndarray,
        B: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Solve the quadratic program for MPC.
        
        Variables: u (control sequence), x (positions), v (velocities)
        Objective: min R·||x - x_ref||² + Q·||u||²
        Constraints: 
            - Dynamics: x[t+1] = x[t] + v[t]·dt + 0.5·a[t]·dt²
            - Control bounds: u_min ≤ u ≤ u_max
        
        Returns:
            u_optimal: Optimal control sequence [horizon], or None if infeasible
        """
        horizon = len(x_ref)
        # Define optimization variables
        u = cp.Variable(horizon)
        x = cp.Variable(horizon)
        v = cp.Variable(horizon)
        
        # Constraints
        constraints = []
        
        # Initial conditions
        constraints.append(x[0] == x0)
        constraints.append(v[0] == v0)
        
        # System dynamics with linearized model
        for t in range(horizon - 1):
            # Linearized acceleration: a[t] = a_nominal[t] + B[t]·u[t]
            a_t = a_nominal[t] + B[t] * u[t]
            
            # Euler integration
            constraints.append(v[t+1] == v[t] + a_t * self.dt)
            constraints.append(
                x[t+1] == x[t] + v[t] * self.dt + 0.5 * a_t * self.dt**2
            )
        
        # Control bounds
        constraints.append(u >= self.u_min)
        constraints.append(u <= self.u_max)
        
        # Objective function using matrix quadratic forms
        # ||x - x_ref||_R^2 = (x - x_ref)^T R (x - x_ref)
        error = x - x_ref
        R_mat = self.R_matrix[:horizon, :horizon]
        tracking_cost = cp.quad_form(error, R_mat)
        
        # ||u||_Q^2 = u^T Q u
        Q_mat = self.Q_matrix[:horizon, :horizon]
        control_cost = cp.quad_form(u, Q_mat)
        
        objective = cp.Minimize(tracking_cost + control_cost)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-5, eps_rel=1e-5)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return u.value
            else:
                print(f"⚠️  Solver status: {problem.status}")
                return None
                
        except Exception as e:
            print(f"Solver error: {e}")
            return None

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
        Optimize control sequence over the horizon using CVXPY with Sequential Quadratic Programming.

        Args:
            x_ref: Reference trajectory [horizon] (torch.Tensor)
            u: Initial control sequence [1, horizon, 1] (not used, CVXPY initializes internally)
            hk: Initial hidden state [num_layers, 1, hidden_dim]
            xk: Initial position
            vk: Initial velocity
            verbose: Whether to print optimization progress

        Returns:
            u_optimal: Optimized control for first step
        """
        # Convert torch tensors to numpy
        x_ref_np = x_ref.detach().cpu().numpy()
        horizon = len(x_ref_np)
        
        # Initialize with zero control
        u_current = np.zeros(horizon)
        
        # Sequential Quadratic Programming (SQP-like approach)
        for iteration in range(self.linearization_iters):
            # 1. Linearize around current control
            a_nominal, B = self.linearize_model(u_current, hk)
            
            # 2. Solve QP
            u_new = self.solve_qp(x_ref_np, xk, vk, a_nominal, B)
            
            if u_new is None:
                print(f"⚠️  Optimization failed at iteration {iteration}")
                break
            
            # 3. Update
            u_current = u_new
            
            # Log first iteration, last iteration, or all if verbose
            should_log = (iteration == 0 or iteration == self.linearization_iters - 1 or verbose)
            
            if should_log:
                # Compute cost for debugging
                u_torch = torch.tensor(u_current, dtype=torch.float32).reshape(1, -1, 1)
                with torch.no_grad():
                    a_torch, _ = self.accel_model(u_torch, hk)
                    a = a_torch.squeeze().numpy()
                
                # Simulate trajectory
                x_pred = np.zeros(horizon)
                v_pred = np.zeros(horizon)
                x_pred[0], v_pred[0] = xk, vk
                
                for t in range(horizon - 1):
                    v_pred[t+1] = v_pred[t] + a[t] * self.dt
                    x_pred[t+1] = x_pred[t] + v_pred[t] * self.dt + 0.5 * a[t] * self.dt**2
                
                tracking_error = np.mean((x_pred - x_ref_np)**2)
                control_effort = np.mean(u_current**2)
                
                print(f"  Iter {iteration+1:3d}/{self.linearization_iters}: u[0]={u_current[0]:.4f}, "
                      f"tracking_err={tracking_error:.4f}, control_effort={control_effort:.4f}")
        
        return float(u_current[0])


# Example usage
if __name__ == "__main__":
    # This is just for illustration
    print("MPC with CVXPY - Example usage:")
    print("""
    mpc = MPCCVXPY(
        accel_model=your_trained_model,
        dt=0.1,
        horizon=10,
        nb_steps=50,
        Q_weight=0.1,
        R_weight=10.0,
        use_ekf=True
    )
    
    # Generate reference trajectory
    x_ref = np.linspace(0, 10, 50)
    
    # Run MPC
    u_history = mpc.solve(x_ref=x_ref, x0=0.0, v0=0.0, verbose=True)
    """)