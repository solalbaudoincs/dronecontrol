from dronecontrol.simulink.engine import initialize_matlab_engine, colvec
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
from tqdm import tqdm
INPUT_FILTER_TAU = 0.2  # time constant for input low-pass filter (seconds)
DT = 0.05  # Default time step for simulation (in seconds)  

DEFAULT_P_CODE_PATH = "matlabfiles"


class DroneSimulator:

    def __init__(self, initial_state : np.ndarray = np.zeros(12), path_to_pcode: str = DEFAULT_P_CODE_PATH) -> None:
        self.eng = initialize_matlab_engine(path_to_pcode)
        self.state : np.ndarray = initial_state
        # filtered input state (initialized lazily on first step)
        self.filtered_input = None

    @property
    def pos(self):
        return self.state[0:3]
    
    @property
    def vel(self):
        return self.state[3:6]

    @property
    def angles(self):
        return self.state[6:9]

    def accel(self, control_input: list[float]) -> list[float]:
        """ Control input is a list of 4 floats: [thrust, tau_phi, tau_theta, tau_psi] wich are the tensions/torques applied by the motors."""
        _, dxdt = self.step(control_input)
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
        dxdt_matlab = self.eng.quadcopter_model(x_matlab, u_matlab)  # Assumes 12x1 output
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
        # RK4 intermediate steps (using the filtered input held constant during the step)
        k1 = self._compute_deriv(self.state, voltage)
        k2 = self._compute_deriv(self.state + 0.5 * dt * k1, voltage)
        k3 = self._compute_deriv(self.state + 0.5 * dt * k2, voltage)
        k4 = self._compute_deriv(self.state + dt * k3, voltage)
        
        # Weighted average update
        dxdt_avg = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        self.state += dt * dxdt_avg
        
        # For consistency, return new state and k1 as approximate dxdt
        return self.state, dxdt_avg
    


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

        dl, _ = prepare_scenario_data(cfg["scenarios"][0], cfg_path.parent)
        dl.setup("fit")
        data : AVDataset = dl.train_dataset
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sim = DroneSimulator(initial_state=np.zeros(12))
        print("Starting simulation comparison...")
        for idx, i in enumerate([0,1,2]):
            sim.reset()
            u : np.ndarray = data[i][0].squeeze().numpy()
            a = data[i][1].squeeze().numpy()
            a_simu = []

            for t in tqdm(range(len(u))):
                state, dxdt = sim.step(np.full((4,), u[t])) #type: ignore
                a_simu.append(dxdt[5])  # extract linear acceleration

            axes[idx].plot(np.array(a_simu))
            axes[idx].plot(a)
            axes[idx].set_title(f'dxdt for sample {i}')
            axes[idx].legend(["simulated","reference"])
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
    
