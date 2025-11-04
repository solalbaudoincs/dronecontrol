import cvxpy as cp
import numpy as np
import torch
from typing import Optional, Tuple

from dronecontrol.commande.neural_ekf import NeuralEKF
from dronecontrol.simulink.simulator import DroneSimulator
from dronecontrol.models.base_module import BaseModel


class MPCCVXPY:
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
        Q_weight: float = 0.1,      # Control effort weight
        R_weight: float = 10.0,     # Tracking error weight
        u_min: float = -5.0,
        u_max: float = 5.0,
        use_ekf: bool = False,
        linearization_iters: int = 3  # Number of SQP iterations
    ):
        """
        Initialize MPC controller.

        Args:
            accel_model: Neural network predicting acceleration from control
            dt: Time step
            horizon: Prediction horizon (number of future steps)
            nb_steps: Total number of control steps to execute
            Q_weight: Weight on control effort (higher = smoother control)
            R_weight: Weight on tracking error (higher = better tracking)
            u_min, u_max: Control input bounds
            use_ekf: Whether to use EKF for state estimation
            linearization_iters: Number of SQP iterations for better accuracy
        """
        self.accel_model = accel_model
        self.dt = dt
        self.horizon = horizon
        self.nb_steps = nb_steps
        self.Q_weight = Q_weight
        self.R_weight = R_weight
        self.u_min = u_min
        self.u_max = u_max
        self.use_ekf = use_ekf
        self.linearization_iters = linearization_iters
        self.hidden_dim = accel_model.hidden_dim

    def linearize_model(
        self,
        u_nominal: np.ndarray,
        hidden: np.ndarray
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
        h_torch = torch.tensor(hidden, dtype=torch.float32).reshape(1, 1, -1)
        
        # Nominal trajectory
        with torch.no_grad():
            a_nominal_torch, _ = self.accel_model(u_torch, h_torch)
            a_nominal = a_nominal_torch.squeeze().cpu().numpy()
        
        # Compute sensitivity using finite differences
        epsilon = 1e-4
        B = np.zeros(self.horizon)
        
        for i in range(self.horizon):
            u_pert = u_nominal.copy()
            u_pert[i] += epsilon
            u_pert_torch = torch.tensor(u_pert, dtype=torch.float32).reshape(1, -1, 1)
            
            with torch.no_grad():
                a_pert_torch, _ = self.accel_model(u_pert_torch, h_torch)
                a_pert = a_pert_torch.squeeze().cpu().numpy()
            
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
        # Define optimization variables
        u = cp.Variable(self.horizon)
        x = cp.Variable(self.horizon)
        v = cp.Variable(self.horizon)
        
        # Constraints
        constraints = []
        
        # Initial conditions
        constraints.append(x[0] == x0)
        constraints.append(v[0] == v0)
        
        # System dynamics with linearized model
        for t in range(self.horizon - 1):
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
        
        # Objective function
        tracking_cost = self.R_weight * cp.sum_squares(x - x_ref)
        control_cost = self.Q_weight * cp.sum_squares(u)
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
            print(f"❌ Solver error: {e}")
            return None

    def step(
        self,
        x_ref: np.ndarray,
        x0: float,
        v0: float,
        hidden: np.ndarray,
        verbose: bool = False
    ) -> float:
        """
        Compute optimal control for one MPC step using Sequential Quadratic Programming.
        
        Args:
            x_ref: Reference trajectory [horizon]
            x0: Current position
            v0: Current velocity
            hidden: Current hidden state [hidden_dim]
            verbose: Print optimization info
            
        Returns:
            u_optimal: Optimal control for first step (scalar)
        """
        # Initialize with zero control
        u_current = np.zeros(self.horizon)
        
        # Sequential Quadratic Programming (SQP-like approach)
        for iteration in range(self.linearization_iters):
            # 1. Linearize around current control
            a_nominal, B = self.linearize_model(u_current, hidden)
            
            # 2. Solve QP
            u_new = self.solve_qp(x_ref, x0, v0, a_nominal, B)
            
            if u_new is None:
                print(f"⚠️  Optimization failed at iteration {iteration}")
                break
            
            # 3. Update
            u_current = u_new
            
            if verbose:
                # Compute cost for debugging
                u_torch = torch.tensor(u_current, dtype=torch.float32).reshape(1, -1, 1)
                h_torch = torch.tensor(hidden, dtype=torch.float32).reshape(1, 1, -1)
                with torch.no_grad():
                    a_torch, _ = self.accel_model(u_torch, h_torch)
                    a = a_torch.squeeze().numpy()
                
                # Simulate trajectory
                x_pred = np.zeros(self.horizon)
                v_pred = np.zeros(self.horizon)
                x_pred[0], v_pred[0] = x0, v0
                
                for t in range(self.horizon - 1):
                    v_pred[t+1] = v_pred[t] + a[t] * self.dt
                    x_pred[t+1] = x_pred[t] + v_pred[t] * self.dt + 0.5 * a[t] * self.dt**2
                
                tracking_error = np.mean((x_pred - x_ref)**2)
                control_effort = np.mean(u_current**2)
                
                print(f"  Iter {iteration}: u[0]={u_current[0]:.4f}, "
                      f"tracking_err={tracking_error:.4f}, control_effort={control_effort:.4f}")
        
        return float(u_current[0])

    def solve(
        self,
        x_ref: np.ndarray,
        x0: float,
        v0: float,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Execute full MPC control loop.
        
        Args:
            x_ref: Reference trajectory for the entire horizon [nb_steps]
            x0: Initial position
            v0: Initial velocity
            verbose: Print progress
            
        Returns:
            u_history: Applied control sequence [nb_steps]
        """
        # Initialize
        u_history = np.zeros(self.nb_steps)
        x_current = x0
        v_current = v0
        h_current = np.zeros(self.hidden_dim)
        
        # Setup simulator and EKF
        simulator = DroneSimulator()
        
        if self.use_ekf:
            ekf = NeuralEKF(
                model=self.accel_model,
                hidden_dim=self.hidden_dim,
                meas_dim=1
            )
        
        # MPC loop
        for step in range(self.nb_steps):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Step {step+1}/{self.nb_steps}: x={x_current:.3f}, v={v_current:.3f}")
            
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
            
            # Solve MPC
            u_optimal = self.step(
                x_ref=x_ref_horizon,
                x0=x_current,
                v0=v_current,
                hidden=h_current,
                verbose=verbose
            )
            
            u_history[step] = u_optimal
            
            if verbose:
                print(f"→ Optimal control: u={u_optimal:.4f}")
            
            # Apply control to simulator (4 motors with same input)
            a_measured = simulator.accel(control_input=[u_optimal] * 4)[-1]
            v_current = simulator.vel[-1]
            x_current = simulator.pos[-1]
            
            if verbose:
                print(f"→ System response: a={a_measured:.4f}, x_new={x_current:.3f}")
            
            # Update hidden state
            if self.use_ekf:
                h_current = ekf.step(
                    u=np.array([u_optimal]),
                    a_measured=np.array([a_measured]),
                    hk=h_current
                )
            else:
                # Open-loop: propagate hidden state through model
                u_torch = torch.tensor([[u_optimal]], dtype=torch.float32).reshape(1, 1, 1)
                h_torch = torch.tensor(h_current, dtype=torch.float32).reshape(1, 1, -1)
                
                with torch.no_grad():
                    _, h_next = self.accel_model(u_torch, h_torch)
                    h_current = h_next.squeeze().cpu().numpy()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✓ MPC completed. Final position: {x_current:.3f}")
        
        return u_history


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