from abc import ABC, abstractmethod
from typing import List, Union
import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from dronecontrol.models.base_module import BaseModel
from .neural_ekf import NeuralEKF

try :
    from dronecontrol.simulink.simulator import DroneSimulator
except ImportError:
    DroneSimulator = None


ArrayLike = Union[torch.Tensor, np.ndarray]

class MPC(ABC):
    
    def __init__(
        self,
        accel_model: BaseModel,
        dt: float,
        horizon: int,
        nb_steps: int,
        Q: ArrayLike,
        R: ArrayLike,
        lr: float,
        max_epochs: int,
        u_min: float,
        u_max: float,
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
        self.Q = torch.tensor(Q)
        self.R = torch.tensor(R)
        self.lr = lr
        self.max_epochs = max_epochs
        self.u_min = u_min
        self.u_max = u_max

        self.num_layers = accel_model.num_layers
        self.hidden_dim = accel_model.hidden_dim

        self.use_simulink = use_simulink
        self.use_ekf = use_ekf

        if use_ekf:
            self.ekf = NeuralEKF(
                model=accel_model, 
                hidden_dim=self.hidden_dim,
                input_dim=1
                )
        else:
            self.ekf = None

        if use_simulink and DroneSimulator is not None:
            self.simulator = DroneSimulator()
        else:
            self.simulator = None

    def _compute_trajectory_wrt_NN(
        self,
        u: torch.Tensor,
        xk: float,
        vk: float,
        hk: torch.Tensor
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
            a, _ = self.accel_model(u, hk)  # a: [1, horizon, 1]
        
        a = a.squeeze(0).squeeze(-1)  # [horizon]
        horizon = a.shape[0]

        # Integrate dynamics
        x = torch.zeros(horizon)
        v = torch.zeros(horizon)

        x[0] = xk
        v[0] = vk

        v[1:] = vk + torch.cumsum(a[:-1] * self.dt, dim=0)
        x[1:] = xk + torch.cumsum(v[:-1] * self.dt + 0.5 * a[:-1] * self.dt**2, dim=0)

        return x, v, a

    def _update_states_NN(
        self,
        u: float,        
        hk: torch.Tensor,
        xk: float,
        vk: float,
    ) -> tuple[torch.Tensor, float, float, float]:
        """
        Compute predicted trajectory given control sequence.
        
        Returns:
            x: Positions [horizon]
            v: Velocities [horizon]
            a: Accelerations [horizon]
        """
        # Predict accelerations
        u_tensor = torch.tensor(u).view(1, -1, 1)  # [1, 1, 1]
        a, h_new = self.accel_model(u_tensor, hk)

        x_new = xk + vk * self.dt + 0.5 * a.item() * self.dt**2
        v_new = vk + a.item() * self.dt
        a_new = a.item()

        return h_new, x_new, v_new, a_new

    def _update_states_simulink(
        self,
        u: float,
        hk: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, float]:
        """
        Compute predicted trajectory using Simulink model.

        Returns:
            x: Positions [horizon]
            v: Velocities [horizon]
            a: Accelerations [horizon]
        """

        # Apply control to simulator (4 motors)

        if self.simulator is None:
            raise RuntimeError("Simulink simulator not available.")


        u_tensor = torch.tensor(u).view(1, -1, 1)  # [1, 1, 1]
        _, h_new = self.accel_model(u_tensor, hk)

        a_measured = self.simulator.accel(control_input=[u] * 4)[-1]
        v_new = self.simulator.vel[-1]
        x_new = self.simulator.pos[-1]

        return h_new, x_new, v_new, a_measured

    def update_states(
        self,
        u: float,
        hk: torch.Tensor,
        xk: float,
        vk: float,
    ) -> tuple[torch.Tensor, float, float, float]:
        """
        Compute predicted trajectory given control sequence.

        Returns:
            x: Positions [horizon]
            v: Velocities [horizon]
            a: Accelerations [horizon]
        """
        if self.use_simulink:
            return self._update_states_simulink(u, hk)
        else:
            return self._update_states_NN(u, hk, xk, vk)

    
    def solve(
        self,
        x_ref: ArrayLike,
        x0: float,
        v0: float,
        a0: float = 0.0,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Solve MPC optimization problem.

        Args:
            x_ref: Reference trajectory [nb_steps]
            x0: Initial position
            v0: Initial velocity
            verbose: Whether to print progress
            return_history: Whether to return optimization history

        Returns:
            u_opt: Optimized control trajectory [nb_steps]
            history: Optional list of MPCState objects
        """
        
        # Initialize
        u_history = torch.zeros(self.nb_steps, dtype=torch.float32, requires_grad=False)
        x_ref = torch.tensor(x_ref, dtype=torch.float32, requires_grad=False)
        
        x_current = x0
        v_current = v0
        a_current = a0
        h_current = torch.zeros(self.num_layers, 1, self.hidden_dim)


        for step in range(self.nb_steps):

            u_opt, h_current, x_current, v_current, a_current = self.step(
                x_ref=x_ref[step:step + self.horizon],
                hk=h_current,
                xk=x_current,
                vk=v_current,
                verbose=verbose
            )

            u_history[step] = u_opt
            
            # Compute tracking error
            tracking_error = abs(x_current - x_ref[step]) if step < len(x_ref) else 0.0
            
            if verbose:
                if step % 5 == 0:  # Print every 5 steps
                    print(f"Step {step:3d}/{self.nb_steps} | "
                          f"x={x_current:7.3f} (ref={x_ref[step]:7.3f}) | "
                          f"v={v_current:7.3f} | "
                          f"a={a_current:7.3f} | "
                          f"u={u_opt:7.3f} | "
                          f"err={tracking_error:7.4f}")

            # Apply EKF if enabled
            if self.use_ekf and self.ekf is not None:
                # EKF update
                h_new = np.zeros((self.num_layers, self.hidden_dim))
                h_current_np = h_current.squeeze().numpy()

                for i in range(self.num_layers):
                    h_new[i, :] = self.ekf.step(
                        u=np.array([u_opt]),
                        hk=h_current_np[i, :],
                        a_measured=np.array([a_current])
                    )

                h_current = torch.tensor(h_new, dtype=torch.float32)

            else:

                h_current = h_current.detach()
            
        return u_history


    @abstractmethod
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
        Optimize control sequence over the horizon.

        Args:
            u: Initial control sequence [1, horizon, 1]
            hidden: Initial hidden state [1, 1, hidden_dim]
            x_ref: Reference trajectory [horizon]
            x0: Initial position
            v0: Initial velocity
            verbose: Whether to print optimization progress

        Returns:
            u_opt: optimized control sequence
        """
        pass  # To be implemented in subclasses


    
    def step(
        self,
        x_ref: torch.Tensor,
        hk: torch.Tensor,
        xk: float,
        vk: float,
        verbose: bool = False
    ) -> Tuple[float, torch.Tensor, float, float, float]:
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
        hk = hk.detach()
        x_ref = x_ref.detach()

        horizon = x_ref.shape[0]
        
        # Initialize control sequence
        u = torch.zeros(self.num_layers, horizon, 1, requires_grad=True)

        u_optimal = self.optimize_control(            
            x_ref=x_ref,
            u=u,
            hk=hk,
            xk=xk,
            vk=vk,
            verbose=verbose
        )
        
        h_new, x_new, v_new, a_new = self.update_states(
                u=u_optimal,
                xk=xk,
                vk=vk,
                hk=hk
            )

        return u_optimal, h_new, x_new, v_new, a_new