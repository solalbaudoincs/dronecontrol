from dronecontrol.simulink.engine import initialize_matlab_engine, colvec
import matlab
import matplotlib.pyplot as plt
from typing import Optional, Any
import numpy as np
from tqdm import tqdm
INPUT_FILTER_TAU = 0.2  # time constant for input low-pass filter (seconds)
DT = 0.05  # Default time step for simulation (in seconds)  

DEFAULT_P_CODE_PATH = "matlabfiles"


class DroneSimulator:

    def __init__(self, initial_state : np.ndarray = np.zeros(12), path_to_pcode: str = DEFAULT_P_CODE_PATH) -> None:
        self.eng: Any = initialize_matlab_engine(path_to_pcode)
        self.state : np.ndarray = initial_state
        # filtered input state (initialized lazily on first step)
        self.filtered_input = None
        # detect MATLAB-side helpers to reduce call overhead
        try:
            res: Any = self.eng.feval('exist', 'rk4_step', 'file', nargout=1)
            exists_rk4 = int(res)
        except Exception:
            exists_rk4 = 0
        self._has_rk4_step = exists_rk4 > 0
        try:
            res2: Any = self.eng.feval('exist', 'simulate_rk4', 'file', nargout=1)
            exists_seq = int(res2)
        except Exception:
            exists_seq = 0
        self._has_simulate_rk4 = exists_seq > 0

    @property
    def pos(self):
        return self.state[0:3]
    
    @property
    def vel(self):
        return self.state[3:6]

    @property
    def angles(self):
        return self.state[6:9]

    def accel(self, control_input: np.ndarray | list[float]) -> np.ndarray:
        """ Control input is a list of 4 floats: [thrust, tau_phi, tau_theta, tau_psi] wich are the tensions/torques applied by the motors."""
        control = np.asarray(control_input, dtype=float)
        _, dxdt = self.step(control)
        return dxdt[3:6]

    @property
    def ang_vel(self):
        return self.state[9:12]
    

    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        if initial_state is not None:
            self.state = initial_state
        self.filtered_input = None

    def _compute_deriv(self, state: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """Helper to compute dxdt using the MATLAB model."""
        x_matlab = colvec(state.tolist())
        u_matlab = colvec(control_input.tolist())
        # Specify nargout to reduce overhead
        dxdt_matlab = self.eng.feval('quadcopter_model', x_matlab, u_matlab, nargout=1)  # 12x1 output
        return np.array(dxdt_matlab).flatten()
    
    def step(self, control_input: np.ndarray, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
        """Advance the simulation by one time step using RK4 integration."""
        # Apply first-order low-pass filter to the control input (per-channel)
        control = np.asarray(control_input, dtype=float)
        if self.filtered_input is None:
            self.filtered_input = np.zeros_like(control)
        tau = INPUT_FILTER_TAU
        alpha = dt / (tau + dt) if (tau + dt) != 0 else 1.0
        # exponential smoothing / discrete-time first-order filter
        self.filtered_input = self.filtered_input + alpha * (control - self.filtered_input)
        voltage = 10*np.tanh(self.filtered_input)
        # Prefer MATLAB-side single-call RK4 if available
        if self._has_rk4_step:
            x_matlab = colvec(self.state.tolist())
            u_matlab = colvec(voltage.tolist())
            res: Any = self.eng.feval('rk4_step', x_matlab, u_matlab, float(dt), nargout=2)
            x_next_mat, dxdt_avg_mat = res
            x_next_np = np.array(x_next_mat).flatten()
            dxdt_np = np.array(dxdt_avg_mat).flatten()
            self.state = x_next_np
            return self.state, dxdt_np
        
        # Fallback: RK4 via 4 MATLAB calls from Python
        k1 = self._compute_deriv(self.state, voltage)
        k2 = self._compute_deriv(self.state + 0.5 * dt * k1, voltage)
        k3 = self._compute_deriv(self.state + 0.5 * dt * k2, voltage)
        k4 = self._compute_deriv(self.state + dt * k3, voltage)
        dxdt_avg = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        self.state += dt * dxdt_avg
        return self.state, dxdt_avg

    def rollout(self, control_seq: np.ndarray, dt: float = DT, return_trace: bool = False):
        """Run a multi-step simulation efficiently.
        control_seq can be shape (T,), (T,1), or (T,4).
        Returns (final_state, trace_dict?) depending on return_trace.
        """
        U = np.asarray(control_seq, dtype=float)
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        if U.shape[1] == 1:
            U = np.repeat(U, 4, axis=1)

        if self._has_simulate_rk4:
            # MATLAB-side full rollout with LPF and saturation
            x0_mat = colvec(self.state.tolist())
            # Convert to explicit MATLAB numeric matrix to avoid 'like' errors
            U_mat = matlab.single(U.tolist())
            if return_trace:
                res: Any = self.eng.feval('simulate_rk4', x0_mat, U_mat, float(dt), float(INPUT_FILTER_TAU), nargout=2)
                X_mat, DX_mat = res
                X_np = np.array(X_mat)
                DX_np = np.array(DX_mat)
                if X_np.ndim == 2 and X_np.shape[0] >= 1:
                    self.state = X_np[-1].astype(float)
                return self.state, {"X": X_np, "DX": DX_np}
            else:
                # Return only final state to minimize data transfer
                xN_mat: Any = self.eng.feval('simulate_rk4_final', x0_mat, U_mat, float(dt), float(INPUT_FILTER_TAU), nargout=1)
                self.state = np.array(xN_mat).flatten().astype(float)
                return self.state, None
        
        # Python-side loop fallback using single-step step() (still uses rk4_step if available)
        traj = [self.state.copy()]
        acc = []
        # Reset LPF for new rollout
        self.filtered_input = None
        for t in range(U.shape[0]):
            _, dxdt = self.step(U[t])
            traj.append(self.state.copy())
            acc.append(dxdt)
        X_np = np.vstack(traj)
        DX_np = np.vstack(acc)
        if return_trace:
            return self.state, {"X": X_np, "DX": DX_np}
        return self.state, None
    


if __name__ == "__main__":
    # we validate the correctness of  our python bindings
        
    from pathlib import Path
    import yaml
    import numpy as np    
    from dronecontrol.data_process.preparation import prepare_scenario_data
    from dronecontrol.data_process.data_loader import AVDataset
    import argparse
    import cProfile
    import pstats
    import io
    import sys

    def load_config(path: Path):
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("Configuration file must define a mapping")
        config.setdefault("scenarios", [])
        config.setdefault("general", {})
        return config  # type: ignore[return-value]

    def run_demo():
        cfg_path = Path("config.yaml")
        cfg = load_config(cfg_path)
        print("Preparing data...")

        dl, _ = prepare_scenario_data(cfg["scenarios"][0], cfg_path.parent)
        dl.setup("fit")
        data : AVDataset = dl.train_dataset  # type: ignore[attr-defined]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sim = DroneSimulator(initial_state=np.zeros(12))
        print("Starting simulation comparison...")
        for idx, i in enumerate([0,1,2]):
            # Load sample
            u : np.ndarray = data[i][0].squeeze().numpy()
            a = data[i][1].squeeze().numpy()

            # 1) Efficient rollout path
            sim.reset()
            _, trace = sim.rollout(u, dt=DT, return_trace=True)
            if trace is not None and "DX" in trace:
                dx_roll = trace["DX"]
                a_roll = dx_roll[:,5]
            else:
                a_roll = np.array([])

            # 2) Classic per-step path using .step
            sim.reset()
            a_step = []
            for t in tqdm(range(len(u)), desc=f"sample {i} step sim", leave=True):
                _, dxdt = sim.step(np.full((4,), u[t]))  # shape (4,)
                a_step.append(dxdt[5])
            a_step = np.asarray(a_step)

            # Plot both simulated vs reference
            axes[idx].plot(a_roll, label="sim rollout")
            axes[idx].plot(a_step, label="sim step")
            axes[idx].plot(a, label="reference")
            axes[idx].set_title(f'dxdt for sample {i}')
            axes[idx].legend()
        plt.show()

    parser = argparse.ArgumentParser(description="Run simulator example (optionally profiled)")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile and write stats to file")
    parser.add_argument("--profile-out", default="simulator.prof", help="Filename to write profiler stats to")
    parser.add_argument("--print-top", type=int, default=50, help="Print top N lines from the profiler by cumulative time")
    args = parser.parse_args()

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        try:
            run_demo()
        finally:
            pr.disable()
            pr.dump_stats(args.profile_out)
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            ps.print_stats(args.print_top)
            print(f"Profile written to: {args.profile_out}")
            print(s.getvalue())
    else:
        run_demo()
        