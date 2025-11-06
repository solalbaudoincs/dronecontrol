from abc import ABC, abstractmethod
from typing import List, Union, Tuple
import torch
import numpy as np

from dronecontrol.models.base_module import BaseModel
from .neural_ekf import NeuralEKF

from dronecontrol.simulink.simulator import DroneSimulator


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
        self.Q = torch.tensor(Q, requires_grad=False).to(device=next(accel_model.parameters()).device)
        self.R = torch.tensor(R, requires_grad=False).to(device=next(accel_model.parameters()).device)
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
                input_dim=1,
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
        xk: torch.Tensor,
        vk: torch.Tensor,
        hk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute predicted trajectory given control sequence.
        
        Returns:
            x: Positions [horizon]
            v: Velocities [horizon]
            a: Accelerations [horizon]
        """
        # Ensure inputs are on the model's device
        device = next(self.accel_model.parameters()).device
        u = u
        hk = hk

        # Predict accelerations
        with torch.set_grad_enabled(u.requires_grad):
            a, _ = self.accel_model(u, hk)  # a: [1, horizon, 1]
        
        a = a.squeeze(0).squeeze(-1)  # [horizon]
        device, dtype = a.device, a.dtype
        horizon = a.shape[0]

        # cumulative velocity increments
        dv = a * self.dt                        # [horizon]
        v_prefix = torch.cumsum(dv, dim=0)      # [horizon]
        # v[0] should be vk, v[i] = vk + sum_{j=0..i-1} a[j]*dt for i>=1
        v = vk + torch.cat((a.new_zeros(1), v_prefix[:-1]))  # [horizon]

        # increments for position: use v[:-1] and a[:-1] (same semantics as original)
        dx = v[:-1] * self.dt + 0.5 * a[:-1] * (self.dt ** 2)  # [horizon-1]
        x_prefix = torch.cumsum(dx, dim=0)                     # [horizon-1]
        x = xk + torch.cat((a.new_zeros(1), x_prefix))         # [horizon]

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
        # Predict accelerations on the model's device
        device = next(self.accel_model.parameters()).device
        u_tensor = torch.tensor(u, device=device).view(1, -1, 1)  # [1, 1, 1]
        hk = hk
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

        
        device = next(self.accel_model.parameters()).device
        u_tensor = torch.tensor(u, device=device).view(1, -1, 1)  # [1, 1, 1]
        hk = hk.to(device)
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
    ) -> dict:
        """
        Compute predicted trajectory given control sequence.

        Returns:
            x: Positions [horizon]
            v: Velocities [horizon]
            a: Accelerations [horizon]
        """
        
        # if self.use_simulink:
        #     return self._update_states_simulink(u, hk)
        # else:
        #     return self._update_states_NN(u, hk, xk, vk)

        return {
            "simulink": self._update_states_simulink(u, hk),
            "nn": self._update_states_NN(u, hk, xk, vk)
        } #type: ignore
    
    def solve(
        self,
        x_ref: ArrayLike,
        x0: float,
        v0: float,
        a0: float = 0.0,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Solve MPC optimization problem.

        Args:
            x_ref: Reference trajectory [nb_steps]
            x0: Initial position
            v0: Initial velocity
            verbose: Whether to print progress
            return_history: Whether to return optimization history

        Returns:
            u_history: Optimized control inputs [nb_steps]
            x_history: Predicted positions [nb_steps]
            v_history: Predicted velocities [nb_steps]
            a_history: Predicted accelerations [nb_steps]
        """
        
        # Initialize
        u_history = torch.zeros(self.nb_steps, dtype=torch.float32, requires_grad=False)
        h_hist_simulink = torch.zeros((self.nb_steps, self.num_layers, self.hidden_dim), dtype=torch.float32, requires_grad=False)
        x_hist_simulink = torch.zeros(self.nb_steps, dtype=torch.float32, requires_grad=False)
        v_hist_simulink = torch.zeros(self.nb_steps, dtype=torch.float32, requires_grad=False)
        a_hist_simulink = torch.zeros(self.nb_steps, dtype=torch.float32, requires_grad=False)

        h_hist_nn = torch.zeros((self.nb_steps, self.num_layers, self.hidden_dim), dtype=torch.float32, requires_grad=False)
        x_hist_nn = torch.zeros(self.nb_steps, dtype=torch.float32, requires_grad=False)    
        v_hist_nn = torch.zeros(self.nb_steps, dtype=torch.float32, requires_grad=False)
        a_hist_nn = torch.zeros(self.nb_steps, dtype=torch.float32, requires_grad=False)
        x_ref = torch.tensor(x_ref, dtype=torch.float32, requires_grad=False)
        
        v_current = v0
        a_current = a0
        x_current = x0
        if self.use_simulink:
            initial_state = np.array([x0] + [0.0]*11, dtype=np.float32)  # Assuming 6 state variables in total
            self.simulator.reset(initial_state) #type: ignore
            
        h_current = torch.zeros(self.num_layers, 1, self.hidden_dim)


        for step in range(self.nb_steps):

            u_opt, state_update_dict = self.step(
                x_ref=x_ref[step:step + self.horizon],
                hk=h_current,
                xk=x_current,
                vk=v_current,
                verbose=verbose
            )
            u_history[step] = torch.tensor(u_opt, dtype=torch.float32)

            h_hist_simulink[step], x_hist_simulink[step], v_hist_simulink[step], a_hist_simulink[step] = state_update_dict["simulink"]
            h_hist_nn[step], x_hist_nn[step], v_hist_nn[step], a_hist_nn[step] = state_update_dict["nn"]

            key = "simulink" if self.use_simulink else "nn"
            h_current, x_current, v_current, a_current = state_update_dict[key]
            
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
                h_current_np = h_current.squeeze(1).detach().cpu().numpy()

                for i in range(self.num_layers):
                    h_new[i, :] = self.ekf.step(
                        u=np.array([u_opt]),
                        hk=h_current_np[i, :],
                        a_measured=np.array([a_current])
                    )

                h_current = torch.tensor(h_new, dtype=torch.float32).unsqueeze(1)

            else:

                h_current = h_current.detach()
        
        histories = {
            "simulink": {
                "h": h_hist_simulink,
                "x": x_hist_simulink,
                "v": v_hist_simulink,
                "a": a_hist_simulink
            },
            "nn": {
                "h": h_hist_nn,
                "x": x_hist_nn,
                "v": v_hist_nn,
                "a": a_hist_nn
            }
        }

        return u_history, histories


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
    ) -> Tuple[float,dict]:
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
        u = torch.zeros(self.num_layers, horizon, 1, requires_grad=True, device=next(self.accel_model.parameters()).device)

        u_optimal = self.optimize_control(            
            x_ref=x_ref,
            u=u,
            hk=hk,
            xk=xk,
            vk=vk,
            verbose=verbose
        )
        
        

        state_update_dict = self.update_states(
                u=u_optimal,
                xk=xk,
                vk=vk,
                hk=hk
            )

        return u_optimal, state_update_dict