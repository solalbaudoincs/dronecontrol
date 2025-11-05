import matlab
import matplotlib.pyplot as plt
from typing import Optional, Any
import numpy as np
from tqdm import tqdm
import ctypes
import os
from numpy.ctypeslib import ndpointer
import platform
INPUT_FILTER_TAU = 0.2  # time constant for input low-pass filter (seconds)
DT = 0.05  # Default time step for simulation (in seconds)  

DEFAULT_P_CODE_PATH = "matlabfiles"

import logging
logger = logging.getLogger(__name__)
import time


def colvec(seq: list[float]) -> matlab.single:
    """Convert Python sequence to MATLAB column vector (Nx1 double)."""
    return matlab.single([[float(v)] for v in seq])


class DroneSimulator:

    def __init__(self, initial_state: np.ndarray = np.zeros(12), path_to_pcode: str = DEFAULT_P_CODE_PATH) -> None:
        self.eng = None
        self.state = np.asarray(initial_state, dtype=np.float64)
        self.filtered_input = None
        self._dll = None
        self._dll_available = False

        # Detect operating system
        system_name = platform.system().lower()  # 'windows', 'linux', 'darwin'
        is_windows = system_name.startswith("win")
        is_linux = system_name.startswith("linux")

        try:
            # Choose the right extension and folder
            lib_folder = os.path.join(path_to_pcode, "codegen", "dll" if is_windows else "lib", "quadcopter_model")
            dll_name = "quadcopter_model.dll" if is_windows else "libquadcopter_model.so"
            dll_path = os.path.join(lib_folder, dll_name)

            if os.path.isfile(dll_path):
                dll = ctypes.CDLL(dll_path)
                logger.info(f"Loaded native library: {dll_path}")

                # Optional initialize function
                try:
                    init_fn = getattr(dll, "quadcopter_model_initialize")
                    init_fn.restype = None
                    init_fn()
                    logger.debug("Initialized native library (quadcopter_model_initialize).")
                except AttributeError:
                    pass

                # Define main function prototype
                try:
                    core = getattr(dll, "quadcopter_model")
                    core.argtypes = [
                        ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # x[12]
                        ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # u[4]
                        ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # dxdt[12]
                    ]
                    core.restype = None
                    self._dll = dll
                    self._dll_core = core
                    self._dll_available = True
                    logger.info("✅ Native quadcopter model loaded successfully.")
                except AttributeError:
                    logger.warning("⚠️ Core symbol 'quadcopter_model' not found in library.")
                    self._dll = None
                    self._dll_available = False
            else:
                logger.warning(f"⚠️ Native library not found at {dll_path}")

        except Exception as e:
            logger.exception(f"❌ Failed to load native library: {e}")
            self._dll = None
            self._dll_available = False

        # Fallback: use MATLAB engine if no DLL/SO is available
        if not self._dll_available:
            logger.warning("Native library unavailable, starting MATLAB engine...")
            time.sleep(2)
            from dronecontrol.simulink.engine import initialize_matlab_engine
            self.eng = initialize_matlab_engine(path_to_pcode)
            self._has_rk4_step = False
            self._has_simulate_rk4 = False

            if self.eng is not None:
                try:
                    exists_rk4 = int(self.eng.feval('exist', 'rk4_step', 'file', nargout=1))
                    self._has_rk4_step = exists_rk4 > 0
                except Exception:
                    pass
                try:
                    exists_seq = int(self.eng.feval('exist', 'simulate_rk4', 'file', nargout=1))
                    self._has_simulate_rk4 = exists_seq > 0
                except Exception:
                    pass

    @property
    def pos(self):
        return self.state[0:3].astype(np.float64, copy=False)
    
    @property
    def vel(self):
        return self.state[3:6].astype(np.float64, copy=False)

    @property
    def angles(self):
        return self.state[6:9].astype(np.float64, copy=False)

    def accel(self, control_input: np.ndarray | list[float]) -> np.ndarray:
        """ Control input is a list of 4 floats: [thrust, tau_phi, tau_theta, tau_psi] wich are the tensions/torques applied by the motors."""
        control = np.asarray(control_input, dtype=float)
        _, dxdt = self.step(control)
        # Ensure consistent dtype for downstream consumers (float64)
        return np.asarray(dxdt[3:6], dtype=np.float64)

    @property
    def ang_vel(self):
        return self.state[9:12]
    

    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        if initial_state is not None:
            self.state = np.asarray(initial_state, dtype=np.float64)
        self.filtered_input = None

    def _compute_deriv(self, state: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """Compute dxdt using the native DLL if available, else MATLAB."""
        # Prefer native DLL (fast, no engine round-trip)
        if self._dll_available:
            x = np.ascontiguousarray(state, dtype=np.float64)
            u = np.ascontiguousarray(control_input, dtype=np.float64)
            if x.shape[0] != 12:
                raise ValueError(f"state must be length 12, got {x.shape}")
            if u.shape[0] != 4:
                raise ValueError(f"control_input must be length 4, got {u.shape}")
            dx = np.empty(12, dtype=np.float64)
            # call: void quadcopter_model(const double x[12], const double u[4], double dxdt[12])
            self._dll_core(x, u, dx)  # type: ignore[attr-defined]
            return dx

        # Fallback to MATLAB engine call
        x_matlab = colvec(state.tolist())
        u_matlab = colvec(control_input.tolist())
        dxdt_matlab = self.eng.feval('quadcopter_model', x_matlab, u_matlab, nargout=1)
        return np.array(dxdt_matlab, dtype=np.float64).flatten()
    
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
        if not self._dll_available and self._has_rk4_step and self.eng is not None:
            x_matlab = colvec(self.state.tolist())
            u_matlab = colvec(voltage.tolist())
            res: Any = self.eng.feval('rk4_step', x_matlab, u_matlab, float(dt), nargout=2)
            x_next_mat, dxdt_avg_mat = res
            x_next_np = np.array(x_next_mat, dtype=np.float64).flatten()
            dxdt_np = np.array(dxdt_avg_mat, dtype=np.float64).flatten()
            self.state = x_next_np.astype(np.float64)
            return self.state, dxdt_np
        elif self._dll_available:
        
        # Fallback: RK4 via 4 MATLAB calls from Python
            k1 = self._compute_deriv(self.state, voltage)
            k2 = self._compute_deriv(self.state + 0.5 * dt * k1, voltage)
            k3 = self._compute_deriv(self.state + 0.5 * dt * k2, voltage)
            k4 = self._compute_deriv(self.state + dt * k3, voltage)
            dxdt_avg = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            # ensure dtype stability
            #self.state = np.asarray(self.state, dtype=np.float64)
            self.state += dt * dxdt_avg.astype(np.float64)
            return self.state, dxdt_avg.astype(np.float64)

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
                # Cast to float64 to avoid numpy.float32 leaking upstream
                X_np = np.array(X_mat, dtype=np.float64)
                DX_np = np.array(DX_mat, dtype=np.float64)
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
        X_np = np.vstack(traj).astype(np.float64)
        DX_np = np.vstack(acc).astype(np.float64)
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

            # # 1) Efficient rollout path
            # sim.reset()
            # _, trace = sim.rollout(u, dt=DT, return_trace=True)
            # dx_roll = trace["DX"]
            # a_roll = dx_roll[:,5]
     

            # 2) Classic per-step path using .step
            sim.reset()
            a_step = []
            for t in tqdm(range(len(u)), desc=f"sample {i} step sim", leave=True):
                _, dxdt = sim.step(np.full((4,), u[t]))  # shape (4,)
                a_step.append(dxdt[5])
            a_step = np.asarray(a_step)

            # Plot both simulated vs reference
            #axes[idx].plot(a_step, label="sim step")
            axes[idx].plot(a, label="reference")
            axes[idx].plot(a_step, label="sim step")
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
        