import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Callable

def colvec(seq: List[float]) -> matlab.single:
    """Convert Python sequence to MATLAB column vector (Nx1 double)."""
    return matlab.single([[float(v)] for v in seq])

def initialize_matlab_engine(pcode_path: str) -> matlab.engine.MatlabEngine:
    """
    Start or connect to a MATLAB Engine, add path, and verify model.
    Tries to connect to a shared engine first; otherwise starts one with
    faster startup options (no desktop/JVM). Then adds the p-code path.
    """

    # Try to connect to an existing shared engine (fast, avoids startup cost)
    try:
        names = matlab.engine.find_matlab()
    except Exception:
        names = []
    if names:
        eng = matlab.engine.connect_matlab(names[0])
    else:
        # Start MATLAB without desktop/JVM to reduce startup time
        # Note: adjust options if certain toolboxes require JVM
        eng = matlab.engine.start_matlab("-nojvm -nodisplay -nosplash")
    if type(eng) is not matlab.engine.MatlabEngine:
        raise ValueError("Failed to start MATLAB engine.")
    eng.addpath(pcode_path, nargout=0)
    
    which_output = eng.which('quadcopter_model', '-all')
    print("which output:", which_output)
    
    if which_output == 'quadcopter_model not found.' or 'built-in' in str(which_output).lower():
        eng.quit()
        raise ValueError("quadcopter_model not found.")
    
    # Test model
    x_test = colvec([0] * 12)
    u_test = colvec([1] * 4)
    dxdt_test = eng.feval('quadcopter_model', x_test, u_test, nargout=1)
    print("Model test successful. dxdt shape:", np.array(dxdt_test).shape)

    return eng

def run_discrete_feedback_simulation(eng: matlab.engine.MatlabEngine,
                                    x0: np.ndarray,
                                    controller: Callable[[np.ndarray, float], np.ndarray],
                                    t_end: float,
                                    dt: float = 0.05,
                                    t_start: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Discrete-time simulation with Python feedback: x[k+1] = x[k] + dx[k] * dt, u from controller.
    
    Args:
        eng (MatlabEngine): Initialized engine
        x0 (np.ndarray): Initial state (shape (12,))
        controller (Callable): Python function u = controller(x, t) -> np.ndarray (shape (4,))
        t_end (float): Simulation end time
        dt (float): Fixed time step (control & integration rate)
        t_start (float): Start time
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: T (steps+1,), X (steps+1, 12), U (steps, 4)
    """
    num_steps = int((t_end - t_start) / dt)
    T = np.linspace(t_start, t_end, num_steps + 1)
    X = np.zeros((num_steps + 1, 12))
    U = np.zeros((num_steps, 4))
    
    X[0] = x0
    
    for k in range(num_steps):
        t = T[k]
        x_k = X[k]
        
        # Python controller computes u from current x and t
        u_k = controller(x_k, t)
        U[k] = u_k
        
        # Get dx/dt from MATLAB model
        x_mat = colvec(x_k.tolist())
        u_mat = colvec(u_k.tolist())
        dx_mat = eng.quadcopter_model(x_mat, u_mat)
        dx_k = np.array(dx_mat).flatten()
        
        # Euler update: x[k+1] = x[k] + dx[k] * dt
        X[k + 1] = x_k + dx_k * dt
    
    print(f"Discrete simulation complete: {num_steps} steps of dt={dt}s")
    return T, X, U

def plot_discrete_results(T: np.ndarray, X: np.ndarray, U: np.ndarray,
                          states_to_plot: Optional[List[int]] = None,
                          labels: Optional[Dict[int, str]] = None,
                          setpoint: Optional[List[float]] = None) -> None:
    """
    Plot states X and controls U vs time.
    """
    if states_to_plot is None:
        states_to_plot = list(range(6))  # Position & velocity
    if labels is None:
        labels = {0: 'x', 1: 'y', 2: 'z', 3: 'vx', 4: 'vy', 5: 'vz'}
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Discrete-Time Quadcopter Simulation with Python Feedback')
    
    # Plot states
    for idx in states_to_plot:
        ax1.plot(T, X[:, idx], label=labels.get(idx, f'State {idx}'))
    ax1.set_ylabel('States')
    if setpoint:
        for i, sp in enumerate(setpoint[:3]):
            ax1.axhline(y=sp, color='r', linestyle='--', alpha=0.7, label=f'Setpoint {labels[i]}')
    ax1.legend()
    ax1.grid(True)
    
    # Plot controls u (note: U has one less entry than T/X)
    T_u = T[:-1]  # Align with U steps
    for j in range(4):
        ax2.plot(T_u, U[:, j], label=f'u{j+1}')
    ax2.set_ylabel('Controls u')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def find_hover_thrust(eng: matlab.engine.MatlabEngine, x0: np.ndarray, dt: float, tol: float = 0.001, max_iter: int = 100) -> float:
    """
    Binary search to find hover thrust that keeps z at 0.
    """
    low = 0.0
    high = 10.0  # Adjust upper bound as needed
    for _ in range(max_iter):
        mid = (low + high) / 2
        def constant_controller(x, t):
            return np.full(4, mid)
        
        T, X, _ = run_discrete_feedback_simulation(eng, x0, constant_controller, 5.0, dt)  # Short sim
        z_final = X[-1, 2]
        if abs(z_final) < tol:
            return mid
        elif z_final > 0:
            high = mid  # Too much thrust, reduce
        else:
            low = mid  # Too little, increase
    return (low + high) / 2

if __name__ == "__main__":

    pcode_path = r"matlabfiles"  # Adjust as needed
    x0 = np.zeros(12)
    t_end = 10.0
    dt = 0.005

    eng = initialize_matlab_engine(pcode_path)
    
    # Find hover thrust
    hover_thrust = find_hover_thrust(eng, x0, dt)
    print(f"Found hover thrust: {hover_thrust}")

    # Simple proportional controller for z only
    zref = 0.0
    k = 0.5  # Proportional gain
    
    def simple_z_controller(x: np.ndarray, t: float) -> np.ndarray:
        z = x[2]
        error = zref - z  # Error = setpoint - current
        u_total = hover_thrust# + k * error
        u = np.full(4, u_total)  # Equal thrust on all motors
        return u # Clip to reasonable bounds

    eng = initialize_matlab_engine("matlabfiles")

    T, X, U = run_discrete_feedback_simulation(eng, x0, simple_z_controller, t_end, dt)
    plot_discrete_results(T, X, U, states_to_plot=[0,1,2,3,4,5])

    eng.quit()

    # Or use function: replace pid_ctrl with lambda x,t: simple_pid_controller(x, t, setpoint=[1,0,3])
