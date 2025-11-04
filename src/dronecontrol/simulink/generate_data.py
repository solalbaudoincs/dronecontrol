from typing import Optional

from dronecontrol.utils import ensure_dir
from .simulator import DroneSimulator, DT
import numpy as np
import logging


logger = logging.getLogger(__name__)



def generate_data(u_t : list[np.ndarray], x0 : np.ndarray, save_to : Optional[str]) -> list[np.ndarray]:
    """
    Generate simulation data using the DroneSimulator.
    
    erxample usage:
    u_t = [np.array([0.0, 0.0, 0.0, 9.81])] * 200  # hover input for 200 time steps
    x0 = np.zeros(12)  # initial state at rest on the ground    
    history, deriv_history = generate_data(u_t, x0, save_to="simulation_output_path")
    """
    assert u_t[0].shape == (4) and x0.shape  == (12)


    sim = DroneSimulator(
        initial_state=x0
    )

    logger.info("=== Starting data generation using DroneSimulator ===")
    total_time = len(u_t) * DT
    logger.info(f"Total simulation time: {total_time:.2f} seconds ({len(u_t)} steps at DT={DT}s)")

    history = [x0.copy()]
    deriv_history = []


    for t in range(len(u_t)):
        st, dxdt = sim.step(u_t[t])
        history.append(np.array(sim.state))
        deriv_history.append(np.array(dxdt))

    
    logger.info("=== Data generation completed ===")
    if save_to is not None:
        logger.info(f"Saving generated data to {save_to}")
        ensure_dir(save_to)
        np.savetxt(save_to + '/_states.csv', np.array(history), delimiter=',')
        np.savetxt(save_to + '/_derivatives.csv', np.array(deriv_history), delimiter=',')
        np.savetxt(save_to + '/_inputs.csv', np.array(u_t), delimiter=',')
        logger.info("Data saved successfully.")
    return history, deriv_history #type: ignore


def load_simulation_data(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load simulation data from CSV files."""
    states = np.loadtxt(path + '/_states.csv', delimiter=',')
    derivatives = np.loadtxt(path + '/_derivatives.csv', delimiter=',')
    inputs = np.loadtxt(path + '/_inputs.csv', delimiter=',')
    return states, derivatives, inputs