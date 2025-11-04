from .simulator import DroneSimulator, DT
import numpy as np
import logging


logger = logging.getLogger(__name__)



def generate_data(u_t : list[np.ndarray], x0 : np.ndarray) -> list[np.ndarray]:

    assert u_t[0].shape == (4) and x0.shape  == (12)


    sim = DroneSimulator(
        initial_state=x0
    )

    logger.info("=== Starting data generation using DroneSimulator ===")
    total_time = len(u_t) * DT
    logger.info(f"Total simulation time: {total_time:.2f} seconds ({len(u_t)} steps at DT={DT}s)")

    history = [x0.copy()]


    for t in range(len(u_t)):
        sim.step(u_t[t].tolist())
        history.append(np.array(sim.state))

    
    logger.info("=== Data generation completed ===")
    return history


