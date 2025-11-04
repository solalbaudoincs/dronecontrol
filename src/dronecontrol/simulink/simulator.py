from dronecontrol.simulink.engine import initialize_matlab_engine, colvec
import matplotlib.pyplot as plt

DT = 0.05  # Default time step for simulation (in seconds)  

DEFAULT_P_CODE_PATH = "matlabfiles"


class DroneSimulator:
    
    def __init__(self, initial_state = [0]*12, path_to_pcode: str = DEFAULT_P_CODE_PATH) -> None:
        self.eng = initialize_matlab_engine(path_to_pcode)
        self.state = initial_state


    @property
    def pos(self):
        return self.state[0:3]
    
    @property
    def vel(self):
        return self.state[3:6]
    
    @property
    def angles(self):
        return self.state[6:9]
    
    @property
    def ang_vel(self):
        return self.state[9:12]
    
    def step(self, control_input: list[float], dt : float = DT) -> tuple[list[float],list[float]]:
        """Advance the simulation by one time step using the provided control input."""
        x_matlab = colvec(self.state)
        u_matlab = colvec(control_input)
        dxdt_matlab : list = self.eng.quadcopter_model(x_matlab, u_matlab) #type: ignore (if its not 12x1, will raise at runtime)
        dxdt = [float(val[0]) for val in dxdt_matlab]
        print("dxdt:", dxdt)
        # Simple Euler integration
        self.state = [s + dx * dt for s, dx in zip(self.state, dxdt)]
        return self.state, dxdt
    

    

